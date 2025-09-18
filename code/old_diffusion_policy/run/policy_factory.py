"""
Policy factory module for creating diffusion policies.

Handles creation of different types of diffusion policies with proper configuration.
"""

import pickle
import json
import numpy as np
from core.env import SoftGymEnvSB3

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.noise_scheduler_factory import create_ddpm_scheduler
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.policy.diffusion_transformer_privileged_policy import DiffusionTransformerPrivilegedPolicy
from diffusion_policy.policy.diffusion_transformer_double_modality import DiffusionTransformerDoubleModalityPolicy


def create_noise_scheduler(args):
    """Create the noise scheduler for diffusion (centralized factory)."""
    return create_ddpm_scheduler(
        num_train_timesteps=args.scheduler_num_train_timesteps,
        beta_schedule=args.scheduler_beta_schedule,
        prediction_type=args.scheduler_prediction_type,
        clip_sample=args.scheduler_clip_sample,
        # Map older args if present
        **{
            k: v for k, v in {
                'variance_type': getattr(args, 'scheduler_variance_type', None),
                'beta_start': getattr(args, 'scheduler_beta_start', None),
                'beta_end': getattr(args, 'scheduler_beta_end', None)
            }.items() if v is not None
        }
    )


def create_privileged_policy(args, noise_scheduler, env_kwargs):
    """Create privileged transformer policy."""
    print(f"[DEBUG] Inferring privileged state‐dim from {args.saved_rollouts} ...")
    with open(args.saved_rollouts, 'rb') as _f:
        _demo = pickle.load(_f)
    if 'ob_trajs' in _demo:
        _st = _demo['ob_trajs']
    elif 'obs_trajs' in _demo:
        _st = _demo['obs_trajs']
    else:
        raise KeyError("Cannot infer state‐dim: no 'ob_trajs' or 'obs_trajs' in pickle")
    inferred_state_dim = _st[0].shape[-1]
    print(f"[DEBUG] → inferred_state_dim = {inferred_state_dim}")
    state_shape = [inferred_state_dim]

    env_spec = SoftGymEnvSB3(**env_kwargs)
    action_shape = list(env_spec.action_space.shape)
    image_c = getattr(env_spec, 'image_c', None) or 3
    image_shape = [args.env_img_size, args.env_img_size, image_c]
    shape_meta = {
        'obs': {
            'state': {'shape': state_shape, 'type': 'low_dim'},
            'image': {'shape': image_shape, 'type': 'rgb'}
        },
        'action': {'shape': action_shape}
    }
    del env_spec

    # interpret crop_shape
    if len(args.crop_shape) == 2 and args.crop_shape[0] > 0 and args.crop_shape[1] > 0:
        crop_shape_val = (args.crop_shape[0], args.crop_shape[1])
    else:
        crop_shape_val = None

    # parse shared_encoder_kwargs
    shared_encoder_kwargs = json.loads(args.shared_encoder_kwargs) if args.shared_encoder_kwargs else None

    # parse privileged_mask if provided
    privileged_mask_tensor = None
    if args.privileged_mask:
        mask_arr = np.load(args.privileged_mask)
        privileged_mask_tensor = mask_arr

    policy = DiffusionTransformerPrivilegedPolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        n_obs_steps=args.n_obs_steps,
        num_inference_steps=args.num_inference_steps,
        crop_shape=crop_shape_val,
        obs_encoder_group_norm=args.obs_encoder_group_norm,
        eval_fixed_crop=args.eval_fixed_crop,
        visual_encoder=args.visual_encoder,
        state_encoder_type=args.state_encoder_type,
        state_mlp_hidden_dims=args.state_mlp_hidden_dims,
        state_feat_dim=args.state_feat_dim,
        fuse_op=args.priv_fuse_op,
        shared_encoder_type=args.shared_encoder_type,
        shared_encoder_kwargs=shared_encoder_kwargs,
        disable_privileged_method=args.disable_privileged_method,
        privileged_mask=privileged_mask_tensor,
        disable_privileged_prob=args.disable_privileged_prob,
        pred_action_steps_only=args.pred_action_steps_only,
        n_layer=args.transformer_n_layer,
        n_cond_layers=args.transformer_n_cond_layers,
        n_head=args.transformer_n_head,
        n_emb=args.transformer_n_emb,
        p_drop_emb=args.transformer_p_drop_emb,
        p_drop_attn=args.transformer_p_drop_attn,
        causal_attn=args.transformer_causal_attn,
        time_as_cond=args.transformer_time_as_cond
    )

    return policy


def create_transformer_hybrid_policy(args, noise_scheduler, env_kwargs):
    """Create transformer hybrid image policy."""
    env_spec = SoftGymEnvSB3(**env_kwargs)
    obs_space = env_spec.observation_space

    # build initial shape_meta using dummy env's observation space
    shape_meta = {
        'obs': {},
        'action': {'shape': list(env_spec.action_space.shape)}
    }
    if hasattr(obs_space, 'spaces'):
        # Handle Dict observation space if necessary (less likely for cam_rgb)
        for key, space in obs_space.spaces.items():
            obs_type = 'rgb' if len(space.shape) == 3 else 'low_dim'
            if obs_type == 'rgb':
                shape = [args.env_img_size, args.env_img_size, space.shape[-1]]
            else:
                shape = list(space.shape)
            shape_meta['obs'][key] = {
                'shape': shape,
                'type': obs_type
            }
    else:
        # Handle Box observation space (most likely case for cam_rgb)
        obs_shape = obs_space.shape
        obs_type = 'rgb' if len(obs_shape) == 3 else 'low_dim'
        if obs_type == 'rgb':
            if obs_shape[-1] == 3 or obs_shape[-1] == 1:
                c = obs_shape[-1]
            elif obs_shape[0] == 3 or obs_shape[0] == 1:
                c = obs_shape[0]
            else:
                c = 3
            shape = [args.env_img_size, args.env_img_size, c]
        else:
            shape = list(obs_shape)
        shape_meta['obs']['obs'] = {
            'shape': shape,
            'type': obs_type
        }
    del env_spec

    # --- Log the final shape_meta being used ---
    print("\n============= Final shape_meta for Policy =============")
    for key in sorted(shape_meta['obs']):
        print(f"  {key} : {shape_meta['obs'][key]}")
    print("=======================================================\n")

    # interpret "0 0" or any non-positive dims as disabling the crop
    if len(args.crop_shape) == 2 and args.crop_shape[0] > 0 and args.crop_shape[1] > 0:
        crop_shape_val = (args.crop_shape[0], args.crop_shape[1])
    else:
        crop_shape_val = None

    policy = DiffusionTransformerHybridImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        n_obs_steps=args.n_obs_steps,
        num_inference_steps=args.num_inference_steps,
        crop_shape=crop_shape_val,
        visual_encoder=args.visual_encoder,
        n_layer=args.transformer_n_layer,
        n_cond_layers=args.transformer_n_cond_layers,
        n_head=args.transformer_n_head,
        n_emb=args.transformer_n_emb,
        p_drop_emb=args.transformer_p_drop_emb,
        p_drop_attn=args.transformer_p_drop_attn,
        causal_attn=args.transformer_causal_attn,
        time_as_cond=args.transformer_time_as_cond,
        obs_as_cond=args.obs_as_global_cond,
        pred_action_steps_only=args.pred_action_steps_only,
    )

    return policy


def create_image_unet_policy(args, noise_scheduler):
    """Create image-based UNet policy."""
    policy = DiffusionUnetImagePolicy(
        noise_scheduler=noise_scheduler,
        horizon=args.horizon,
        image_size=args.env_img_size,
        action_dim=args.action_size,
        n_action_steps=args.n_action_steps,
        n_obs_steps=args.n_obs_steps,
        num_inference_steps=args.num_inference_steps,
        channel_cond=args.channel_cond,
        cond_predict_scale=args.cond_predict_scale,
        cnn_channels=args.cnn_channels,
        cnn_kernels=args.cnn_kernels,
        cnn_strides=args.cnn_strides,
        obs_as_global_cond=args.obs_as_global_cond,
        latent_dim=args.latent_dim
    )
    return policy


def create_lowdim_transformer_policy(args, noise_scheduler):
    """Create low-dimensional transformer policy."""
    cond_dim = args.observation_size if args.obs_as_global_cond else 0
    input_dim = args.action_size if args.obs_as_global_cond \
                else (args.action_size + args.observation_size)

    model = TransformerForDiffusion(
        input_dim=input_dim,
        output_dim=args.action_size,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        cond_dim=cond_dim,
        obs_as_cond=args.obs_as_global_cond,
        n_emb=args.transformer_n_emb,
        n_layer=args.transformer_n_layer,
        n_head=args.transformer_n_head,
        p_drop_emb=args.transformer_p_drop_emb,
        p_drop_attn=args.transformer_p_drop_attn,
        causal_attn=args.transformer_causal_attn,
        time_as_cond=args.transformer_time_as_cond,
        n_cond_layers=args.transformer_n_cond_layers
    )

    policy = DiffusionTransformerLowdimPolicy(
        model=model,
        noise_scheduler=noise_scheduler,
        horizon=args.horizon,
        obs_dim=args.observation_size,
        action_dim=args.action_size,
        n_action_steps=args.n_action_steps,
        n_obs_steps=args.n_obs_steps,
        num_inference_steps=args.num_inference_steps,
        obs_as_cond=args.obs_as_global_cond,
        pred_action_steps_only=args.pred_action_steps_only
    )

    return policy, model


def create_lowdim_unet_policy(args, noise_scheduler):
    """Create low-dimensional UNet policy."""
    # Calculate input dimension based on conditioning
    if args.obs_as_local_cond or args.obs_as_global_cond:
        input_dim = args.action_size
    else:
        input_dim = args.observation_size + args.action_size

    # Calculate conditioning dimensions
    local_cond_dim = args.observation_size if args.obs_as_local_cond else None
    global_cond_dim = args.observation_size * args.n_obs_steps if args.obs_as_global_cond else None

    # Create the UNet model
    model = ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=local_cond_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=args.unet_down_dims,
        kernel_size=args.unet_kernel_size,
        n_groups=args.unet_n_groups,
        cond_predict_scale=args.cond_predict_scale
    )

    # Create the diffusion policy
    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=noise_scheduler,
        horizon=args.horizon,
        obs_dim=args.observation_size,
        action_dim=args.action_size,
        n_action_steps=args.n_action_steps,
        n_obs_steps=args.n_obs_steps,
        num_inference_steps=args.num_inference_steps,
        obs_as_local_cond=args.obs_as_local_cond,
        obs_as_global_cond=args.obs_as_global_cond,
        pred_action_steps_only=args.pred_action_steps_only,
        oa_step_convention=args.oa_step_convention
    )

    return policy, model


def create_double_modality_policy(args, noise_scheduler, env_kwargs):
    """Create double modality transformer policy (always uses both state and image)."""
    print(f"[DEBUG] Inferring state dimensions from {args.saved_rollouts} ...")
    with open(args.saved_rollouts, 'rb') as _f:
        _demo = pickle.load(_f)
    if 'ob_trajs' in _demo:
        _st = _demo['ob_trajs']
    elif 'obs_trajs' in _demo:
        _st = _demo['obs_trajs']
    else:
        raise KeyError("Cannot infer state dimensions: no 'ob_trajs' or 'obs_trajs' in pickle")
    inferred_state_dim = _st[0].shape[-1]
    print(f"[DEBUG] → inferred_state_dim = {inferred_state_dim}")
    state_shape = [inferred_state_dim]

    env_spec = SoftGymEnvSB3(**env_kwargs)
    action_shape = list(env_spec.action_space.shape)
    image_c = getattr(env_spec, 'image_c', None) or 3
    image_shape = [args.env_img_size, args.env_img_size, image_c]
    shape_meta = {
        'obs': {
            'state': {'shape': state_shape, 'type': 'low_dim'},
            'image': {'shape': image_shape, 'type': 'rgb'}
        },
        'action': {'shape': action_shape}
    }
    del env_spec

    # interpret crop_shape
    if len(args.crop_shape) == 2 and args.crop_shape[0] > 0 and args.crop_shape[1] > 0:
        crop_shape_val = (args.crop_shape[0], args.crop_shape[1])
    else:
        crop_shape_val = None

    # parse shared_encoder_kwargs
    shared_encoder_kwargs = json.loads(args.shared_encoder_kwargs) if args.shared_encoder_kwargs else None

    policy = DiffusionTransformerDoubleModalityPolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        n_obs_steps=args.n_obs_steps,
        num_inference_steps=args.num_inference_steps,
        crop_shape=crop_shape_val,
        obs_encoder_group_norm=args.obs_encoder_group_norm,
        eval_fixed_crop=args.eval_fixed_crop,
        visual_encoder=args.visual_encoder,
        state_encoder_type=args.state_encoder_type,
        state_mlp_hidden_dims=args.state_mlp_hidden_dims,
        state_feat_dim=args.state_feat_dim,
        fuse_op=args.priv_fuse_op,  # Using the same parameter name
        shared_encoder_type=args.shared_encoder_type,
        shared_encoder_kwargs=shared_encoder_kwargs,
        pred_action_steps_only=args.pred_action_steps_only,
        n_layer=args.transformer_n_layer,
        n_cond_layers=args.transformer_n_cond_layers,
        n_head=args.transformer_n_head,
        n_emb=args.transformer_n_emb,
        p_drop_emb=args.transformer_p_drop_emb,
        p_drop_attn=args.transformer_p_drop_attn,
        causal_attn=args.transformer_causal_attn,
        time_as_cond=args.transformer_time_as_cond
    )

    return policy


def create_diffusion_policy(args, env_kwargs):
    """
    Main factory function to create diffusion policy based on args.

    Returns:
        policy: The diffusion policy instance
        ema_model: EMA model instance (or None if not used)
    """
    # Create the noise scheduler
    noise_scheduler = create_noise_scheduler(args)

    # Double modality transformer policy branch (always uses both state and image)
    if args.model_type == 'double_modality':
        policy = create_double_modality_policy(args, noise_scheduler, env_kwargs)
        model = None
    # Privileged transformer policy branch
    elif args.model_type == 'privileged':
        policy = create_privileged_policy(args, noise_scheduler, env_kwargs)
        model = None
    elif args.is_image_based and args.model_type == 'transformer':
        policy = create_transformer_hybrid_policy(args, noise_scheduler, env_kwargs)
        model = None
    elif args.is_image_based:
        # Image-based policy (UNet)
        policy = create_image_unet_policy(args, noise_scheduler)
        model = None
    else:
        # state-based policy (lowdim)
        if args.model_type == 'transformer':
            policy, model = create_lowdim_transformer_policy(args, noise_scheduler)
        else:
            policy, model = create_lowdim_unet_policy(args, noise_scheduler)

    # Create EMA model if needed
    if args.use_ema:
        # pick the correct sub-module for EMA:
        if args.model_type == 'double_modality':
            # double modality transformer: use internal transformer model
            ema_target = policy.model
        elif args.model_type == 'privileged':
            # privileged transformer: use internal transformer model
            ema_target = policy.model
        elif args.is_image_based and args.model_type == 'transformer':
            # hybrid image-transformer: the diffusion network is policy.model
            ema_target = policy.model
        elif args.is_image_based:
            # image-unet policy: the UNet is stored in policy.nets['action_model']
            ema_target = policy.nets['action_model']
        else:
            # state-based branch: the model you passed into the policy
            ema_target = model

        ema_model = EMAModel(
            model=ema_target,
            update_after_step=0,
            inv_gamma=1.0,
            power=0.75,
            min_value=0.0,
            max_value=0.9999
        )
    else:
        ema_model = None

    return policy, ema_model
