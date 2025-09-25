from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import os
import torch
from tqdm import tqdm
import wandb
from ml_framework.core.evaluator import Evaluator
from ml_framework.utils.misc import set_seed
from ml_framework.utils.checkpoint_util import TopKCheckpointManager
from ml_framework.utils.json_logger import JsonLogger
from ml_framework.utils.pytorch_util import dict_apply

def _compute_loss_with_components(policy, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Allow policies to optionally return (loss, components_dict). Backwards compat: plain tensor."""
    out = policy.compute_loss(batch)
    if isinstance(out, tuple) and len(out) == 2:
        loss, comp = out
        comp_f = {k: float(v) for k, v in comp.items()}
        return loss, comp_f
    return out, {}


def _setup_checkpoint_manager(cfg, output_dir: str) -> Optional[TopKCheckpointManager]:
    """Setup checkpoint manager if checkpointing is enabled."""
    if cfg.checkpoint_dir is None:
        return None
    
    checkpoint_dir = cfg.checkpoint_dir if os.path.isabs(cfg.checkpoint_dir) else os.path.join(output_dir, cfg.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return TopKCheckpointManager(
        save_dir=checkpoint_dir,
        monitor_key='train_loss',  # or could be 'val_loss' if available
        mode='min',
        k=cfg.keep_last if hasattr(cfg, 'keep_last') else 5,
        format_str='epoch={epoch:03d}-train_loss={train_loss:.6f}.ckpt'
    )


def _save_checkpoint_with_manager(checkpoint_manager: Optional[TopKCheckpointManager], 
                                epoch: int, policy, optimizer, metrics: Dict[str, float], cfg):
    """Save checkpoint using TopKCheckpointManager."""
    if checkpoint_manager is None:
        return
    
    # Prepare checkpoint data
    checkpoint_data = {
        'epoch': epoch,
        'train_loss': metrics.get('train_loss', 0.0),
        **metrics  # Include all metrics for filename formatting
    }
    
    ckpt_path = checkpoint_manager.get_ckpt_path(checkpoint_data)
    if ckpt_path is not None:
        state = {
            'epoch': epoch,
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': cfg.to_dict() if hasattr(cfg, 'to_dict') else cfg,
            'metrics': metrics
        }
        
        # Save normalizer if available
        if hasattr(policy, 'normalizer'):
            try:
                state['normalizer'] = policy.normalizer.state_dict()
            except Exception:
                state['normalizer'] = None
        
        torch.save(state, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


def _maybe_save_checkpoint(cfg, epoch: int, policy, optimizer, extra: Dict[str, Any]):
    """Legacy checkpoint function - deprecated, use _save_checkpoint_with_manager instead."""
    if cfg.checkpoint_dir is None:
        return
    if (epoch + 1) % cfg.checkpoint_every != 0:
        return
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch+1}.pth")
    state = {
        'epoch': epoch,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': cfg.to_dict()
    }
    if hasattr(policy, 'normalizer'):
        try:
            state['normalizer'] = policy.normalizer.state_dict()
        except Exception:
            state['normalizer'] = None
    state.update(extra)
    torch.save(state, ckpt_path)
    # Rolling retention
    if cfg.keep_last > 0:
        ckpts = sorted([f for f in os.listdir(cfg.checkpoint_dir) if f.startswith('epoch_')])
        if len(ckpts) > cfg.keep_last:
            for old in ckpts[:-cfg.keep_last]:
                try:
                    os.remove(os.path.join(cfg.checkpoint_dir, old))
                except OSError:
                    pass


def train_epochs(policy: torch.nn.Module,
                 train_loader,
                 val_loader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 cfg) -> None:
    policy.to(device)
    use_wandb = (wandb.run is not None) and cfg.wandb

    # Setup output directory for logging and checkpoints
    output_dir = getattr(cfg, 'output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup checkpoint manager
    checkpoint_manager = _setup_checkpoint_manager(cfg, output_dir)
    
    # Setup JSON logger
    json_logger = None
    if getattr(cfg, 'log_json', True):
        log_path = os.path.join(output_dir, 'training_log.json')
        json_logger = JsonLogger(log_path)
        json_logger.start()  # Initialize the logger
        print(f"[train] JSON logging to: {log_path}")

    # Setup evaluator if enabled
    evaluator = None
    if getattr(cfg, 'eval_enabled', True):
        # Derive env kwargs from existing SoftGym usage assumptions; expecting that training dataset was collected
        # with some known environment configuration. For now we require user to supply via config file's evaluation.env_kwargs
        # Not present in cfg dataclass; try to read attribute dynamically (can be injected by extended configs)
        env_kwargs = getattr(cfg, 'env_kwargs', None)
        if env_kwargs is None:
            # Provide a clear message; evaluation requires env parameters
            print('[train] Evaluation requested but cfg.env_kwargs missing; skipping evaluator initialization.')
        else:
            video_dir = None
            if cfg.eval_video:
                if cfg.checkpoint_dir is not None:
                    video_dir = os.path.join(cfg.checkpoint_dir, 'eval_videos')
                else:
                    # Create default video directory when checkpoint_dir is None
                    video_dir = os.path.join(output_dir, 'eval_videos')
            evaluator = Evaluator(
                env_kwargs=env_kwargs,
                max_episode_steps=cfg.eval_max_episode_steps or getattr(cfg, 'max_eval_steps', cfg.horizon),
                n_obs_steps=cfg.n_obs_steps,
                is_image_based=(cfg.model in ("transformer-image", "transformer-privileged")),
                model_type='privileged' if cfg.model == 'transformer-privileged' else 'image' if cfg.model == 'transformer-image' else 'lowdim',
                device=device,
                video_dir=video_dir,
                gif_size=cfg.eval_gif_size,
                verbose=True
            )
            print('[train] Evaluator initialized.')

    for epoch in range(cfg.max_epochs):
        policy.train()
        running_loss = 0.0
        comp_aggr: Dict[str, float] = {}
        for batch in tqdm(train_loader, desc=f"train {epoch}"):
            batch = dict_apply(batch, lambda x: x.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss, comps = _compute_loss_with_components(policy, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss)
            for k, v in comps.items():
                comp_aggr[k] = comp_aggr.get(k, 0.0) + v

        train_loss = running_loss / max(1, len(train_loader))
        comp_means = {f"train/{k}": v / max(1, len(train_loader)) for k, v in comp_aggr.items()}
        print(f"epoch {epoch} train_loss={train_loss:.4f}")

        # Validation
        policy.eval()
        val_loss_sum = 0.0
        val_comp_aggr: Dict[str, float] = {}
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val {epoch}", leave=False):
                batch = dict_apply(batch, lambda x: x.to(device))
                vloss, vcomps = _compute_loss_with_components(policy, batch)
                val_loss_sum += float(vloss)
                for k, v in vcomps.items():
                    val_comp_aggr[k] = val_comp_aggr.get(k, 0.0) + v
        val_loss = val_loss_sum / max(1, len(val_loader))
        val_comp_means = {f"val/{k}": v / max(1, len(val_loader)) for k, v in val_comp_aggr.items()}
        print(f"epoch {epoch} val_loss={val_loss:.4f}")

        log_payload = {
            'train/loss': train_loss,
            'val/loss': val_loss,
            'train/epoch': epoch
        }
        log_payload.update(comp_means)
        log_payload.update(val_comp_means)
        
        # Log to JSON if enabled
        if json_logger is not None:
            json_logger.log(log_payload)
        
        # Log to WandB if enabled    
        if use_wandb:
            wandb.log(log_payload, step=epoch)

        # Online environment evaluation
        if evaluator is not None and (epoch + 1) % cfg.eval_interval == 0:
            if cfg.eval_seed is not None:
                set_seed(cfg.eval_seed)
            print(f"[train] Running evaluation at epoch {epoch}...")
            metrics, gif_path = evaluator.run(
                policy,
                num_episodes=cfg.eval_num_episodes,
                save_video=cfg.eval_video,
                epoch=epoch,
                env_img_size=cfg.eval_env_img_size
            )
            # merge eval metrics into logging
            if use_wandb:
                wandb_payload = {k: v for k, v in metrics.items()}
                # Add video log if available (following old_diffusion_policy pattern)
                if gif_path is not None and os.path.exists(gif_path):
                    try:
                        wandb_payload['eval/eval_video'] = wandb.Video(gif_path, fps=10, format='gif')
                        print(f"[train] Logged eval video: {gif_path}")
                    except Exception as e:
                        print(f"[train] Failed to log video to WandB: {e}")
                wandb.log(wandb_payload, step=epoch)

        # Checkpoint using TopKCheckpointManager
        if checkpoint_manager is not None and (epoch + 1) % cfg.checkpoint_every == 0:
            checkpoint_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': epoch
            }
            _save_checkpoint_with_manager(checkpoint_manager, epoch, policy, optimizer, checkpoint_metrics, cfg)
        
        # Also support legacy checkpoint saving for backward compatibility
        # _maybe_save_checkpoint(cfg, epoch, policy, optimizer, extra={'train_loss': train_loss, 'val_loss': val_loss})