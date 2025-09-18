"""
Training module for diffusion policies.

Handles the complete training loop including setup, data loading, optimization, and logging.
"""

import os
import math
import tqdm
import wandb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms

from diffusion_policy.model.common.normalizer import LinearNormalizer
from .demonstrations import Demonstrations
from .evaluation import Evaluation


def setup_data_loading(args, env_kwargs):
    """
    Set up data loading components.

    Returns:
        dataset: The demonstrations dataset
        train_loader: Training data loader
        val_loader: Validation data loader
        img_transform: Image transformations (if any)
    """
    # Set up image transformations if needed
    img_transform = None
    if args.is_image_based and args.enable_img_transformations:
        img_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])

    # Load data
    assert args.saved_rollouts is not None, "Must provide path to demonstrations"
    print(f"Loading demonstrations from: {args.saved_rollouts}")
    dataset = Demonstrations(
        args.saved_rollouts,
        is_image_based=args.is_image_based,
        img_transform=img_transform,
        horizon=args.horizon,
        privileged=args.model_type in ['privileged', 'double_modality'],
        args=args
    )

    # Split into train and validation sets
    train_size = int(len(dataset) * args.train_data_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    if len(train_dataset) < args.batch_size:
        train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=args.batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Sample a batch to log shapes
    sample_batch = next(iter(train_loader))
    print("\nData shapes:")
    for k, v in sample_batch.items():
        print(f"  {k}: {v.shape}")

    return dataset, train_loader, val_loader, img_transform


def setup_optimizer(policy, ema_model, args):
    """
    Set up optimizer and learning rate scheduler.

    Returns:
        optimizer: Configured optimizer
        scheduler: Learning rate scheduler
    """
    # Collect parameters and setup optimizer
    if args.is_image_based and args.model_type == 'transformer':
        # Transformer‐hybrid uses its own optimizer grouping
        transformer_wd = args.transformer_weight_decay
        encoder_wd    = args.encoder_weight_decay
        optimizer = policy.get_optimizer(
            transformer_weight_decay=transformer_wd,
            obs_encoder_weight_decay=encoder_wd,
            learning_rate=args.lrate,
            betas=(args.beta1, args.beta2)
        )
        print("Using policy.get_optimizer for TransformerHybridImagePolicy.")
        # for reporting only
        param_list = list(policy.model.parameters()) + list(policy.obs_encoder.parameters())

    elif args.is_image_based and args.model_type == 'privileged':
        # Privileged transformer uses its own optimizer grouping
        optimizer = policy.get_optimizer(
            transformer_weight_decay=args.transformer_weight_decay,
            obs_encoder_weight_decay=args.encoder_weight_decay,
            state_encoder_weight_decay=args.encoder_weight_decay,
            shared_encoder_weight_decay=args.encoder_weight_decay,
            learning_rate=args.lrate,
            betas=(args.beta1, args.beta2)
        )
        print("Using policy.get_optimizer for PrivilegedTransformerPolicy.")
        # for reporting only
        param_list = list(policy.model.parameters())
        param_list += list(policy.obs_encoder.parameters())
        param_list += list(policy.state_encoder.parameters())
        if hasattr(policy, 'shared_encoder') and policy.shared_encoder is not None:
            param_list += list(policy.shared_encoder.parameters())

    elif args.is_image_based and args.model_type == 'double_modality':
        # Double modality transformer uses its own optimizer grouping
        optimizer = policy.get_optimizer(
            transformer_weight_decay=args.transformer_weight_decay,
            obs_encoder_weight_decay=args.encoder_weight_decay,
            state_encoder_weight_decay=args.encoder_weight_decay,
            shared_encoder_weight_decay=args.encoder_weight_decay,
            learning_rate=args.lrate,
            betas=(args.beta1, args.beta2)
        )
        print("Using policy.get_optimizer for DiffusionTransformerDoubleModalityPolicy.")
        # for reporting only
        param_list = list(policy.model.parameters())
        param_list += list(policy.obs_encoder.parameters())
        param_list += list(policy.state_encoder.parameters())
        if hasattr(policy, 'shared_encoder') and policy.shared_encoder is not None:
            param_list += list(policy.shared_encoder.parameters())

    else:
        # fallback for UNet‐image or lowdim policies
        if args.is_image_based and args.model_type == 'unet':
            param_list = list(policy.nets['action_model'].parameters())
            if 'vision_encoder' in policy.nets:
                param_list += list(policy.nets['vision_encoder'].parameters())
        elif not args.is_image_based:
            param_list = list(policy.model.parameters())
        else:
            raise NotImplementedError(f"Optimizer logic missing for model_type={args.model_type}")

        print("Using generic AdamW optimizer setup.")
        optimizer = optim.AdamW(
            param_list,
            lr=args.lrate,
            betas=(args.beta1, args.beta2),
            eps=1e-8,
            weight_decay=args.weight_decay
        )

    total_params = sum(p.numel() for grp in optimizer.param_groups for p in grp['params'])
    trainable_params = sum(p.numel() for grp in optimizer.param_groups for p in grp['params'] if p.requires_grad)
    print(f"Total Parameters (Optimizer Groups): {total_params:,}")
    print(f"Trainable Parameters (Optimizer Groups): {trainable_params:,}")

    # Create learning rate scheduler
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma
        )
    elif args.lr_scheduler == 'cosine':
        def lr_lambda(epoch):
            # linear warmup
            if epoch < args.lr_warmup_steps:
                return float(epoch) / float(max(1, args.lr_warmup_steps))
            # cosine decay after warmup
            progress = float(epoch - args.lr_warmup_steps) \
                       / float(max(1, args.max_train_epochs - args.lr_warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")

    return optimizer, scheduler


def fit_normalizer(policy, dataset, args):
    """
    Fit the normalizer on demonstration data.

    Args:
        policy: The diffusion policy
        dataset: Demonstrations dataset
        args: Arguments object
    """
    if not args.resume:
        print("Fitting normalizer on demonstration data...")
        fit_data = {}
        if args.model_type in ['privileged', 'double_modality']:
            # privileged/double_modality: normalize states and actions
            all_states = torch.cat([traj['state']  for traj in dataset.trajectories], dim=0)
            fit_data['state'] = all_states
        elif not args.is_image_based:
            # state-based (lowdim) policy: normalize obs and actions
            all_obs = torch.cat([traj['obs'] for traj in dataset.trajectories], dim=0)
            fit_data['obs'] = all_obs
        # everyone needs actions
        all_action = torch.cat([traj['action'] for traj in dataset.trajectories], dim=0)
        fit_data['action'] = all_action
        # finally, fit the normalizer
        policy.normalizer.fit(fit_data, last_n_dims=1, mode='limits')
    else:
        print("Skipping normalizer fit; loaded from checkpoint.")


def load_checkpoint(policy, ema_model, optimizer, scheduler, args):
    """
    Load checkpoint if resuming training.

    Returns:
        start_epoch: Starting epoch number
        resume_ckpt: Checkpoint data (or None)
        total_steps: Total gradient steps so far
    """
    start_epoch = 0
    resume_ckpt = None
    total_steps = 0

    if args.resume:
        # locate checkpoint folder
        resume_model_dir = os.path.join(args.model_save_dir, args.folder_name, 'checkpoints')
        if not os.path.isdir(resume_model_dir):
            raise ValueError(f"Checkpoint dir not found: {resume_model_dir}")
        ckpt_files = [f for f in os.listdir(resume_model_dir)
                      if f.startswith('epoch_') and f.endswith('.pth')]
        if not ckpt_files:
            raise ValueError(f"No checkpoint files in {resume_model_dir}")
        # pick the latest epoch
        def _epoch_num(f):
            try:
                return int(f.split('_')[1].split('.pth')[0])
            except:
                return -1
        latest_epoch, latest_file = max(
            [(_epoch_num(f), f) for f in ckpt_files],
            key=lambda x: x[0]
        )
        ckpt_path = os.path.join(resume_model_dir, latest_file)
        print(f"Resuming from checkpoint: {ckpt_path}")
        resume_ckpt = torch.load(ckpt_path, map_location='cpu')  # Will move to device later
        # restore model, optimizer, scheduler
        policy.load_state_dict(resume_ckpt['model_state_dict'])
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        # restore normalizer if present
        if hasattr(policy, 'normalizer') and resume_ckpt.get('normalizer'):
            policy.normalizer.load_state_dict(resume_ckpt['normalizer'])
        # restore EMA if used
        if args.use_ema and resume_ckpt.get('ema_state_dict'):
            ema_model.averaged_model.load_state_dict(resume_ckpt['ema_state_dict'])
        # pick up from next epoch
        start_epoch = resume_ckpt['epoch'] + 1
        total_steps = resume_ckpt.get('total_steps', 0)
        print(f"Resuming from total_steps = {total_steps}")

    return start_epoch, resume_ckpt, total_steps


def setup_wandb(policy, args, resume_ckpt, total_params, trainable_params):
    """
    Initialize Weights & Biases logging.

    Returns:
        Updated args with wandb run name
    """
    if args.wandb:
        project_id = "cloth-diff"

        if args.resume and resume_ckpt is not None and resume_ckpt.get('wandb_run_id'):
            # re-attach to the same run
            wandb.init(
                project=project_id,
                id=resume_ckpt['wandb_run_id'],
                resume="must"
            )
        else:
            # fresh run
            wandb.init(
                project=project_id,
                name=args.folder_name,
                config=vars(args)
            )
        args.folder_name = wandb.run.name
        wandb.watch(policy, log="all", log_freq=100)
        wandb.log({
            "description/total_parameters": total_params,
            "description/trainable_parameters": trainable_params
        }, step=0)

    return args


def train_epoch(policy, train_loader, optimizer, device, total_steps, args):
    """
    Run one training epoch.

    Returns:
        avg_train_loss: Average training loss for the epoch
        total_steps: Updated total gradient steps
    """
    policy.train()
    train_loss = 0.0

    for batch_idx, batch in enumerate(tqdm.tqdm(train_loader)):
        # Move batch to device
        for k, v in batch.items():
            batch[k] = v.to(device)

        # Forward / backward / step
        optimizer.zero_grad()
        loss = policy.compute_loss(batch)
        loss.backward()
        optimizer.step()
        # Count gradient step and optionally stop
        total_steps += 1
        if args.max_train_steps is not None and total_steps >= args.max_train_steps:
            print(f"Reached max_train_steps={args.max_train_steps}, stopping training early")
            break
        train_loss += loss.item()

    # Calculate average training loss
    avg_train_loss = train_loss / len(train_loader)
    return avg_train_loss, total_steps


def validate_epoch(policy, val_loader, device):
    """
    Run validation for one epoch.

    Returns:
        avg_val_loss: Average validation loss
    """
    policy.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move batch to device
            for k, v in batch.items():
                batch[k] = v.to(device)

            # Forward pass and loss computation
            loss = policy.compute_loss(batch)
            val_loss += loss.item()

    # Calculate average validation loss
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def run_training_loop(policy, ema_model, optimizer, scheduler, train_loader, val_loader,
                      evaluator, device, args, start_epoch, total_steps, model_dir):
    """
    Main training loop.

    Args:
        policy: The diffusion policy to train
        ema_model: EMA model (or None)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        train_loader: Training data loader
        val_loader: Validation data loader
        evaluator: Evaluation instance
        device: Training device
        args: Arguments object
        start_epoch: Starting epoch number
        total_steps: Initial total gradient steps
        model_dir: Directory to save checkpoints

    Returns:
        policy: Trained policy
        ema_model: Updated EMA model
    """
    for epoch in range(start_epoch, args.max_train_epochs):
        print(f"Epoch {epoch+1}/{args.max_train_epochs}")

        # Training phase
        avg_train_loss, total_steps = train_epoch(
            policy, train_loader, optimizer, device, total_steps, args
        )

        # Scheduler update
        scheduler.step()

        # Break out of epoch loop if step-limit reached
        if args.max_train_steps is not None and total_steps >= args.max_train_steps:
            break

        # Validation phase - only compute every 10th epoch
        avg_val_loss = 0.0
        if epoch % 10 == 0:
            avg_val_loss = validate_epoch(policy, val_loader, device)
            print(f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"Train Loss: {avg_train_loss:.6f}")

        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'gradient_steps': total_steps
        }

        if epoch % 10 == 0:
            metrics['val_loss'] = avg_val_loss

        # --- Logging to WandB ---
        if args.wandb:
            log_data_base = {
                "train/train_loss": avg_train_loss,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/gradient_steps": total_steps,
            }

            if epoch % 10 == 0:
                log_data_base["validation/val_loss"] = avg_val_loss

            # Evaluation and Checkpointing
            if (epoch + 1) % args.eval_interval == 0:
                 # --- Replace model with EMA model for evaluation if used ---
                original_state_dict = None
                if args.use_ema:
                    ema_params = ema_model.averaged_model.state_dict()
                    if args.model_type == 'privileged':
                        # privileged: only the diffusion core lives in policy.model
                        original_state_dict = policy.model.state_dict()
                        policy.model.load_state_dict(ema_params)
                    elif args.model_type == 'double_modality':
                        # double modality: diffusion core lives in policy.model
                        original_state_dict = policy.model.state_dict()
                        policy.model.load_state_dict(ema_params)
                    elif args.is_image_based and args.model_type == 'transformer':
                        original_state_dict = policy.model.state_dict()
                        policy.model.load_state_dict(ema_params)
                    elif args.is_image_based:
                        original_state_dict = policy.nets['action_model'].state_dict()
                        policy.nets['action_model'].load_state_dict(ema_params)
                    else:
                        original_state_dict = policy.model.state_dict()
                        policy.model.load_state_dict(ema_params)

                # Evaluate policy
                # Now returns: avg_normalized_performance_final, avg_reward, avg_ep_length, saved_gif_path
                avg_normalized_perf, std_normalized_perf, avg_reward, avg_ep_length, saved_gif_path = evaluator.evaluate(
                    policy,
                    num_episodes=args.num_eval_eps,
                    save_video=args.eval_videos,
                    epoch=epoch # Pass epoch for GIF naming
                )

                print(f"Validation Eval: Norm Perf Mean: {avg_normalized_perf:.4f}, Std: {std_normalized_perf:.4f}, "
                      f"Avg Reward: {avg_reward:.2f}, Avg Ep Length: {avg_ep_length:.2f}")

                # Log evaluation metrics to WandB
                log_data_eval = {
                    "validation/info_normalized_performance_mean": avg_normalized_perf,
                    "validation/info_normalized_performance_std": std_normalized_perf,
                    "validation/avg_rews": avg_reward,
                    "validation/avg_ep_length": avg_ep_length,
                    "validation/total_steps": total_steps
                }
                log_data_base.update(log_data_eval) # Add eval metrics to base log data

                # Add video log if available
                if args.eval_videos and saved_gif_path:
                     log_data_base["validation/eval_video"] = wandb.Video(saved_gif_path, fps=10, format="gif")

                # --- Restore original weights ---
                if args.use_ema and original_state_dict is not None:
                    if args.model_type == 'privileged':
                        policy.model.load_state_dict(original_state_dict)
                    elif args.is_image_based and args.model_type == 'transformer':
                        policy.model.load_state_dict(original_state_dict)
                    elif args.is_image_based:
                        policy.nets['action_model'].load_state_dict(original_state_dict)
                    else:
                        policy.model.load_state_dict(original_state_dict)

                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss if epoch % 10 == 0 else None,
                    'normalizer': policy.normalizer.state_dict() if hasattr(policy, 'normalizer') else None,
                    'wandb_run_id': wandb.run.id if args.wandb else None,
                    'wandb_run_name': args.folder_name,
                    'total_steps': total_steps,
                }

                if args.use_ema:
                    checkpoint['ema_state_dict'] = ema_model.averaged_model.state_dict() # Save EMA params

                torch.save(checkpoint, os.path.join(model_dir, f'epoch_{epoch+1}.pth'))
                print(f"Saved checkpoint at epoch {epoch+1}")

            # Log combined data to WandB at the end of the epoch
            wandb.log(log_data_base, step=epoch + 1)

    return policy, ema_model


def main_training(args, env_kwargs):
    """
    Main training function that orchestrates the entire training process.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print training configuration
    print("\n" + "="*60)
    print("DIFFUSION POLICY TRAINING CONFIGURATION")
    print("="*60)
    # collect only the args that actually drive training
    config = {
        "env_name": args.env_name,
        "observation_mode": args.env_kwargs["observation_mode"],
        "is_image_based": args.is_image_based,
        "model_type": args.model_type,
        "horizon": args.horizon,
        "n_obs_steps": args.n_obs_steps,
        "n_action_steps": args.n_action_steps,
        "num_inference_steps": args.num_inference_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lrate,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "lr_scheduler": args.lr_scheduler,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "lr_warmup_steps": args.lr_warmup_steps,
        "max_train_epochs": args.max_train_epochs,
        "max_train_steps": args.max_train_steps,
        "seed": args.seed,
        "use_ema": args.use_ema,
        "obs_as_global_cond": args.obs_as_global_cond,
        "obs_as_local_cond": args.obs_as_local_cond,
        "pred_action_steps_only": args.pred_action_steps_only,
        "cond_predict_scale": args.cond_predict_scale,
        "train_data_ratio": args.train_data_ratio,
        "eval_interval": args.eval_interval,
        "wandb": args.wandb,
    }

    # add only the policy‐type & modality‐specific params
    if args.is_image_based:
        img_cfg = {"env_img_size": args.env_img_size}
        if args.enable_img_transformations:
            img_cfg["enable_img_transformations"] = True
        if args.model_type == "unet":
            img_cfg.update({
                "cnn_channels": args.cnn_channels,
                "cnn_kernels": args.cnn_kernels,
                "cnn_strides": args.cnn_strides,
                "latent_dim": args.latent_dim,
            })
        elif args.model_type == "transformer":
            img_cfg.update({
                "transformer_n_emb": args.transformer_n_emb,
                "transformer_n_layer": args.transformer_n_layer,
                "transformer_n_head": args.transformer_n_head,
                "transformer_p_drop_emb": args.transformer_p_drop_emb,
                "transformer_p_drop_attn": args.transformer_p_drop_attn,
                "transformer_causal_attn": args.transformer_causal_attn,
                "transformer_time_as_cond": args.transformer_time_as_cond,
                "transformer_n_cond_layers": args.transformer_n_cond_layers,
                "visual_encoder": args.visual_encoder,
                "transformer_weight_decay": args.transformer_weight_decay,
                "encoder_weight_decay": args.encoder_weight_decay,
            })
            if args.crop_shape[0] > 0 and args.crop_shape[1] > 0:
                img_cfg["crop_shape"] = tuple(args.crop_shape)
        config.update(img_cfg)
    else:
        ld_cfg = {
            "observation_size": args.observation_size,
            "action_size": args.action_size,
            "diffusion_step_embed_dim": args.diffusion_step_embed_dim,
            "oa_step_convention": args.oa_step_convention,
            "weight_decay": args.weight_decay,
        }
        if args.model_type == "unet":
            ld_cfg.update({
                "unet_down_dims": args.unet_down_dims,
                "unet_kernel_size": args.unet_kernel_size,
                "unet_n_groups": args.unet_n_groups,
            })
        elif args.model_type == "transformer":
            ld_cfg.update({
                "transformer_n_emb": args.transformer_n_emb,
                "transformer_n_layer": args.transformer_n_layer,
                "transformer_n_head": args.transformer_n_head,
                "transformer_p_drop_emb": args.transformer_p_drop_emb,
                "transformer_p_drop_attn": args.transformer_p_drop_attn,
                "transformer_causal_attn": args.transformer_causal_attn,
                "transformer_time_as_cond": args.transformer_time_as_cond,
                "transformer_n_cond_layers": args.transformer_n_cond_layers,
            })
        config.update(ld_cfg)

    # prune any None values and pretty-print
    config = {k: v for k, v in config.items() if v is not None}
    max_key_len = max(len(k) for k in config) if config else 0
    for key in sorted(config):
        print(f"  {key.ljust(max_key_len)} : {config[key]}")
    print("="*60 + "\n")

    # Import here to avoid circular imports
    from .policy_factory import create_diffusion_policy

    # Create policy and EMA model
    policy, ema_model = create_diffusion_policy(args, env_kwargs)
    policy = policy.to(device)

    # Ensure we always have the real LinearNormalizer, and move it to device
    if not hasattr(policy, 'normalizer') or policy.normalizer is None:
        policy.normalizer = LinearNormalizer()
    policy.normalizer.to(device)

    # Make sure the network isn't accidentally frozen
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            param.requires_grad_(True)

    # Collect and inspect the actual sub-network parameters we want to train (for reporting only)
    if args.is_image_based:
        if args.model_type == 'transformer':
            # hybrid image-transformer
            param_list = list(policy.model.parameters())
        elif args.model_type == 'privileged':
            # privileged transformer: diffusion net + vision encoder + state encoder + shared encoder
            param_list = list(policy.model.parameters())
            param_list += list(policy.obs_encoder.parameters())
            param_list += list(policy.state_encoder.parameters())
            if hasattr(policy, 'shared_encoder') and policy.shared_encoder is not None:
                param_list += list(policy.shared_encoder.parameters())
        elif args.model_type == 'double_modality':
            # double modality transformer: diffusion net + vision encoder + state encoder + shared encoder
            param_list = list(policy.model.parameters())
            param_list += list(policy.obs_encoder.parameters())
            param_list += list(policy.state_encoder.parameters())
            if hasattr(policy, 'shared_encoder') and policy.shared_encoder is not None:
                param_list += list(policy.shared_encoder.parameters())
        else:
            # image-based UNet
            param_list = list(policy.nets['action_model'].parameters())
            if 'vision_encoder' in policy.nets:
                param_list += list(policy.nets['vision_encoder'].parameters())
    else:
        # low‐dimensional policy (UNet or transformer)
        param_list = list(policy.model.parameters())

    total_params     = sum(p.numel() for p in param_list)
    trainable_params = sum(p.numel() for p in param_list if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Set up data loading
    dataset, train_loader, val_loader, _ = setup_data_loading(args, env_kwargs)

    # Fit normalizer
    fit_normalizer(policy, dataset, args)

    # Set up optimizer and scheduler
    optimizer, scheduler = setup_optimizer(policy, ema_model, args)

    # Load checkpoint if resuming
    start_epoch, resume_ckpt, total_steps = load_checkpoint(
        policy, ema_model, optimizer, scheduler, args
    )

    # Initialize WandB if enabled
    args = setup_wandb(policy, args, resume_ckpt, total_params, trainable_params)

    # Create directories for saving models
    model_dir = os.path.join(args.model_save_dir, args.folder_name, 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model checkpoints will be saved to: {model_dir}")

    # Create evaluator
    evaluator = Evaluation(args, env_kwargs)

    # Run training loop
    policy, ema_model = run_training_loop(
        policy, ema_model, optimizer, scheduler, train_loader, val_loader,
        evaluator, device, args, start_epoch, total_steps, model_dir
    )

    # --- Final evaluation after training loop ---
    print("\nRunning final evaluation over 5 seeds (100 episodes total)...")
    original_state_dict = None
    if args.use_ema:
        ema_params = ema_model.averaged_model.state_dict()
        if args.model_type == 'privileged':
            original_state_dict = policy.model.state_dict()
            policy.model.load_state_dict(ema_params)
        elif args.model_type == 'double_modality':
            original_state_dict = policy.model.state_dict()
            policy.model.load_state_dict(ema_params)
        elif args.is_image_based and args.model_type == 'transformer':
            original_state_dict = policy.model.state_dict()
            policy.model.load_state_dict(ema_params)
        elif args.is_image_based:
            original_state_dict = policy.nets['action_model'].state_dict()
            policy.nets['action_model'].load_state_dict(ema_params)
        else:
            original_state_dict = policy.model.state_dict()
            policy.model.load_state_dict(ema_params)

    # Evaluate across 5 seeds (20 eps each)
    evaluator.evaluate_five_seeds(policy)

    if args.wandb:
        wandb.finish()

    return policy, ema_model
