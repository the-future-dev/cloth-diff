from __future__ import annotations
import sys
import torch

from ml_framework.utils.config import parse_config, pretty_config
import os
from ml_framework.utils.misc import set_seed, get_device
from ml_framework.utils.pytorch_util import optimizer_to, dict_apply
from ml_framework.data.data import make_dataloaders
from ml_framework.core.loop import train_epochs
from foundation_policy.policies.transformer_lowdim import TransformerLowDimPolicy
from foundation_policy.policies.transformer_image import TransformerImagePolicy
from foundation_policy.policies.transformer_privileged import TransformerPrivilegedPolicy
# Import diffusion policies
from diffusion_policy.policies.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from ml_framework.utils.shape_validation import create_shape_validator
from ml_framework.utils.misc import ensure_dir

from functools import reduce
import operator
import wandb


def _flatten_obs_dim(shape, is_image=False):
    """
    Compute observation dimension matching old diffusion policy behavior.
    
    Args:
        shape: Shape of observation tensor (e.g., [B, T, ...])
        is_image: Whether this is an image observation
        
    Returns:
        Computed observation dimension
        
    Note: This function now properly handles different observation types:
    - For image obs: flattens spatial+channel dims [B,T,H,W,C] -> H*W*C
    - For state obs: uses last dimension only [B,T,state_dim] -> state_dim
    """
    if len(shape) <= 2:
        return shape[-1]
    
    if is_image:
        # For image observations: flatten spatial and channel dimensions
        # [B, T, H, W, C] -> H * W * C (same as before)
        return reduce(operator.mul, shape[2:], 1)
    else:
        # For state observations: use last dimension only (old policy style)
        # [B, T, state_dim] -> state_dim
        return shape[-1]


def main(argv=None):
    cfg = parse_config(argv)
    print("Loaded config:\n" + pretty_config(cfg))

    set_seed(cfg.seed)
    device = get_device(cfg.device)

    if cfg.mode == "train":
        if cfg.dataset_path is None:
            print("--dataset_path is required for training", file=sys.stderr)
            sys.exit(2)
        is_img = cfg.model in ("transformer-image", "transformer-privileged")
        privileged = cfg.model == "transformer-privileged"
        train_loader, val_loader = make_dataloaders(
            cfg.dataset_path, cfg.batch_size, cfg.num_workers,
            is_image_based=is_img, privileged=privileged, horizon=cfg.horizon, cfg=cfg
        )

        # infer dims from first batch
        first = next(iter(train_loader))
        if privileged:
            Do_img = _flatten_obs_dim(list(first["image"].shape), is_image=True)
            Do_state = _flatten_obs_dim(list(first["state"].shape), is_image=False)
        else:
            obs_shape = list(first["obs"].shape)
            Do = _flatten_obs_dim(obs_shape, is_image=is_img)
        Da = first["action"].shape[-1]

        # ---- COMPREHENSIVE SHAPE VALIDATION AND LOGGING ----
        print("\n" + "="*80)
        print("COMPREHENSIVE SHAPE VALIDATION AND DEBUGGING")
        print("="*80)
        
        # Create shape validator
        shape_validator = create_shape_validator(verbose=True)
        
        # Log detailed batch shapes
        shape_validator.log_batch_shapes(first, "Training Batch (first)")
        
        # Log a validation batch for comparison
        val_first = next(iter(val_loader))
        shape_validator.log_batch_shapes(val_first, "Validation Batch (first)")
        
        # Validate consistency between train and val batches
        print("\nBATCH CONSISTENCY CHECK:")
        for key in first.keys():
            if key in val_first:
                train_shape = list(first[key].shape)
                val_shape = list(val_first[key].shape)
                # Check all dimensions except batch size (index 0)
                if train_shape[1:] != val_shape[1:]:
                    print(f"  ❌ {key}: train{train_shape} vs val{val_shape} - SHAPE MISMATCH!")
                    sys.exit(f"Shape mismatch in {key} between train and validation batches")
                else:
                    print(f"  ✅ {key}: shapes consistent (ignoring batch size)")
            else:
                print(f"  ⚠️  {key}: missing in validation batch")
        
        # Log computed dimensions with validation
        print(f"\nCOMPUTED DIMENSIONS:")
        if privileged:
            print(f"  Do_img (image obs dim): {Do_img}")
            print(f"  Do_state (state obs dim): {Do_state}")
            print(f"  Da (action dim): {Da}")
            
            # Validate dimension computation
            img_shape = list(first["image"].shape)
            state_shape = list(first["state"].shape)
            print(f"\nDIMENSION COMPUTATION VALIDATION:")
            print(f"  Image shape: {img_shape}")
            print(f"  → Do_img = _flatten_obs_dim({img_shape}, is_image=True) = {Do_img}")
            if len(img_shape) > 2:
                expected_img = reduce(operator.mul, img_shape[2:], 1)
                assert Do_img == expected_img, f"Image dim mismatch: {Do_img} != {expected_img}"
                print(f"  ✅ Image dimension correctly computed as {expected_img}")
            
            print(f"  State shape: {state_shape}")
            print(f"  → Do_state = _flatten_obs_dim({state_shape}, is_image=False) = {Do_state}")
            if len(state_shape) > 2:
                expected_state = state_shape[-1]
                assert Do_state == expected_state, f"State dim mismatch: {Do_state} != {expected_state}"
                print(f"  ✅ State dimension correctly computed as {expected_state}")
        else:
            print(f"  Do (observation dim): {Do}")
            print(f"  Da (action dim): {Da}")
            
            # Validate dimension computation
            print(f"\nDIMENSION COMPUTATION VALIDATION:")
            print(f"  Obs shape: {obs_shape}")
            print(f"  → Do = _flatten_obs_dim({obs_shape}, is_image={is_img}) = {Do}")
            if is_img and len(obs_shape) > 2:
                expected_obs = reduce(operator.mul, obs_shape[2:], 1)
                assert Do == expected_obs, f"Image obs dim mismatch: {Do} != {expected_obs}"
                print(f"  ✅ Image obs dimension correctly computed as {expected_obs}")
            elif not is_img:
                expected_obs = obs_shape[-1]
                assert Do == expected_obs, f"State obs dim mismatch: {Do} != {expected_obs}"
                print(f"  ✅ State obs dimension correctly computed as {expected_obs}")

        # choose policy by cfg.model
        # Determine model name (handles both string and dict cases)
        model_name = cfg.model.get('name') if isinstance(cfg.model, dict) else cfg.model
        
        if model_name in ("transformer-lowdim", "transformer_lowdim"):
            policy = TransformerLowDimPolicy(Do, Da, cfg.horizon, cfg.n_action_steps, cfg.n_obs_steps)
        elif model_name in ("transformer-image", "transformer_image"):
            policy = TransformerImagePolicy(Do, Da, cfg.horizon, cfg.n_action_steps, cfg.n_obs_steps)
        elif model_name in ("transformer-privileged", "transformer_privileged"):
            policy = TransformerPrivilegedPolicy(Do_img, Do_state, Da, cfg.horizon, cfg.n_action_steps, cfg.n_obs_steps, lowdim_weight=cfg.lowdim_weight)
        elif model_name in ("diffusion-transformer-lowdim", "diffusion_transformer_lowdim"):
            # Create noise scheduler
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.num_inference_steps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon"
            )
            
            # Get transformer config
            transformer_cfg = cfg.model['transformer'] if isinstance(cfg.model, dict) else getattr(cfg, 'transformer', {})
            obs_as_cond = cfg.model.get('obs_as_cond', False) if isinstance(cfg.model, dict) else getattr(cfg, 'obs_as_cond', False)
            
            # Compute dimensions
            input_dim = Da if obs_as_cond else (Do + Da)
            output_dim = input_dim
            cond_dim = Do if obs_as_cond else 0
            
            # Create transformer model
            transformer_model = TransformerForDiffusion(
                input_dim=input_dim,
                output_dim=output_dim,
                horizon=cfg.horizon,
                n_obs_steps=cfg.n_obs_steps,
                cond_dim=cond_dim,
                n_layer=transformer_cfg.get('n_layer', 8),
                n_head=transformer_cfg.get('n_head', 4),
                n_emb=transformer_cfg.get('n_emb', 256),
                p_drop_emb=transformer_cfg.get('p_drop_emb', 0.0),
                p_drop_attn=transformer_cfg.get('p_drop_attn', 0.3),
                causal_attn=transformer_cfg.get('causal_attn', True),
                obs_as_cond=obs_as_cond
            )
            
            policy = DiffusionTransformerLowdimPolicy(
                model=transformer_model,
                noise_scheduler=noise_scheduler,
                horizon=cfg.horizon,
                obs_dim=Do,
                action_dim=Da,
                n_action_steps=cfg.n_action_steps,
                n_obs_steps=cfg.n_obs_steps,
                num_inference_steps=cfg.num_inference_steps
            )
        else:
            print(f"Unsupported model: {model_name}", file=sys.stderr)
            sys.exit(2)

        # ---- COMPREHENSIVE POLICY VALIDATION ----
        policy.to(device)
        
        # Validate policy with shape validator
        print(f"\nPOLICY INITIALIZATION VALIDATION:")
        print(f"  Model type: {cfg.model}")
        print(f"  Policy class: {policy.__class__.__name__}")
        
        if privileged:
            print(f"  Policy dimensions: Do_img={Do_img}, Do_state={Do_state}, Da={Da}")
        else:
            print(f"  Policy dimensions: Do={Do}, Da={Da}")
            
        # Comprehensive policy validation
        validation_result = shape_validator.validate_policy_input_output(
            policy, first, cfg.model, device
        )
        
        # Assert that validation passed
        if not validation_result['validation_passed']:
            print("❌ POLICY VALIDATION FAILED!")
            for error in validation_result['errors']:
                print(f"   Error: {error}")
            sys.exit("Policy validation failed. Please check the errors above.")
        else:
            print("✅ POLICY VALIDATION PASSED!")
            
        # Validate computed dimensions match policy expectations
        if privileged and 'computed_dims' in validation_result:
            computed = validation_result['computed_dims']
            expected = {'Do_img': Do_img, 'Do_state': Do_state, 'Da': Da}
            try:
                shape_validator.assert_dimensions_match(computed, expected)
                print("✅ DIMENSION MATCHING PASSED!")
            except AssertionError as e:
                print(f"❌ DIMENSION MATCHING FAILED: {e}")
                sys.exit("Dimension validation failed")
        elif not privileged and 'computed_dims' in validation_result:
            computed = validation_result['computed_dims']
            expected = {'Do': Do, 'Da': Da}
            try:
                shape_validator.assert_dimensions_match(computed, expected)
                print("✅ DIMENSION MATCHING PASSED!")
            except AssertionError as e:
                print(f"❌ DIMENSION MATCHING FAILED: {e}")
                sys.exit("Dimension validation failed")

        # ---- TRAINING SETUP SUMMARY ----
        print("\n" + "="*80)
        print("TRAINING SETUP SUMMARY")
        print("="*80)
        print(f"Device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            except Exception:
                pass
        print(f"Model: {policy.__class__.__name__}")
        n_params = sum(p.numel() for p in policy.parameters())
        n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"Parameters: total={n_params:,}, trainable={n_trainable:,}")
        print(f"Config: horizon={cfg.horizon}, n_obs_steps={cfg.n_obs_steps}, n_action_steps={cfg.n_action_steps}, batch_size={cfg.batch_size}")
        print(f"WandB: {'enabled' if cfg.wandb else 'disabled'}; project={getattr(cfg, 'wandb_project', None)}; name={getattr(cfg, 'exp_name', None)}")
        
        # Final validation summary
        validation_summary = shape_validator.get_validation_summary()
        print(f"\nVALIDATION SUMMARY:")
        print(f"  Total validations: {validation_summary['total_validations']}")
        print(f"  Success rate: {validation_summary['success_rate']:.1%}")
        if validation_summary['latest_validation']:
            latest = validation_summary['latest_validation']
            print(f"  Latest result: {'PASSED' if latest['validation_passed'] else 'FAILED'}")
            if 'initial_loss' in latest:
                print(f"  Initial loss: {latest['initial_loss']:.6f}")
                
        print("="*80 + "\n")

        optim = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
        optimizer_to(optim, device)
        # Optional: wandb
        if cfg.wandb and not cfg.no_wandb:
            wandb.init(project=cfg.wandb_project, name=cfg.exp_name, config=cfg.to_dict())
            wandb.watch(policy, log="all", log_freq=100)
        # Training uses full-horizon supervision (loss over all horizon steps).
        # At inference, policies return a window of n_action_steps starting at index (n_obs_steps-1).
        # Ensure checkpoint dir default if not set
        if cfg.checkpoint_dir is None:
            cfg.checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
        
        # Include job name in checkpoint directory structure
        if cfg.exp_name:
            cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir, cfg.exp_name)
        ensure_dir(cfg.checkpoint_dir)

        train_epochs(policy, train_loader, val_loader, optim, device, cfg)
        if cfg.wandb and not cfg.no_wandb:
            wandb.finish()
    else:
        # simple evaluation: iterate dataset, compute loss, and output mean loss
        if cfg.dataset_path is None:
            print("--dataset_path is required for evaluation", file=sys.stderr)
            sys.exit(2)
        is_img = cfg.model in ("transformer-image", "transformer-privileged")
        privileged = cfg.model == "transformer-privileged"
        loader, _ = make_dataloaders(
            cfg.dataset_path, cfg.batch_size, cfg.num_workers,
            is_image_based=is_img, privileged=privileged, horizon=cfg.horizon, cfg=cfg
        )
        first = next(iter(loader))
        if privileged:
            Do_img = _flatten_obs_dim(list(first["image"].shape), is_image=True)
            Do_state = _flatten_obs_dim(list(first["state"].shape), is_image=False)
        else:
            obs_shape = list(first["obs"].shape)
            Do = _flatten_obs_dim(obs_shape, is_image=is_img)
        Da = first["action"].shape[-1]

        # choose policy by cfg.model
        if cfg.model in ("transformer-lowdim", "transformer_lowdim"):
            policy = TransformerLowDimPolicy(Do, Da, cfg.horizon, cfg.n_action_steps, cfg.n_obs_steps)
        elif cfg.model in ("transformer-image", "transformer_image"):
            policy = TransformerImagePolicy(Do, Da, cfg.horizon, cfg.n_action_steps, cfg.n_obs_steps)
        elif cfg.model in ("transformer-privileged", "transformer_privileged"):
            policy = TransformerPrivilegedPolicy(Do_img, Do_state, Da, cfg.horizon, cfg.n_action_steps, cfg.n_obs_steps, lowdim_weight=cfg.lowdim_weight)
        else:
            print(f"Unsupported model: {cfg.model}", file=sys.stderr)
            sys.exit(2)

        policy.to(device)
        policy.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            # Basic shape validation for eval
            print(f"Eval data shapes: obs {first['obs'].shape if 'obs' in first else 'N/A'}, action {first['action'].shape}")
            if privileged:
                print(f"Privileged: image {first['image'].shape}, state {first['state'].shape}")
            
            for batch in loader:
                batch = dict_apply(batch, lambda x: x.to(device))
                total += float(policy.compute_loss(batch))
                n += 1
        print(f"Eval loss: {total/max(1,n):.6f}")


if __name__ == "__main__":
    main()