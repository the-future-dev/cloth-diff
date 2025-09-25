import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.optim import AdamW

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

# New unified modules
from diffusion_policy.model.common.image_processor import ImageProcessor
from diffusion_policy.model.common.dimension_validator import DimensionValidator
from diffusion_policy.model.common.optimizer_factory import DiffusionPolicyOptimizer
from diffusion_policy.model.encoder_factory import EncoderFactory
from diffusion_policy.model.feature_fusion import FeatureFusion

# Legacy imports for compatibility
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionTransformerPrivilegedPolicy(BaseImagePolicy):
    """
    A diffusion‐based policy that at TRAIN time uses both `state` and `image`
    observations, but at INFERENCE can disable the `state` branch in four ways.
    """
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = None,

        # image‐encoder params
        crop_shape=(76, 76),
        obs_encoder_group_norm: bool = False,
        eval_fixed_crop: bool = False,
        visual_encoder: str = 'ResNet18Conv',

        # state‐encoder
        state_encoder_type: str = 'identity',   # 'identity' or 'mlp'
        state_mlp_hidden_dims: list = None,
        state_feat_dim: int = None,

        # feature fusion
        fuse_op: str = 'concat',                # 'concat' or 'sum'

        # shared multimodal encoder
        shared_encoder_type: str = None,        # 'mlp', 'transformer', 'perceiver', 'cross_attention'
        shared_encoder_kwargs: dict = None,

        # how to disable the state branch at test time
        disable_privileged_method: str = 'zero',   # 'zero','skip','gating','mask'
        privileged_mask: torch.Tensor = None,      # required if method=='mask'

        disable_privileged_prob: float = 0.0,
        # predict only next k actions
        pred_action_steps_only: bool = False,

        # diffusion‐transformer arch
        n_layer: int = 8,
        n_cond_layers: int = 0,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
        time_as_cond: bool = True,

        # extra kwargs for scheduler.step(...)
        **sample_kwargs
    ):
        super().__init__()
        
        # Validate shape_meta early
        DimensionValidator.validate_shape_meta(shape_meta, 'privileged')
        
        # store training‐time disable probability
        self.disable_privileged_prob = disable_privileged_prob
        self.use_contrastive_loss = True
        self.contrastive_weight = 0.3

        # parse dims
        action_dim = shape_meta['action']['shape'][0]
        obs_shape_meta = shape_meta['obs']
        state_dim = obs_shape_meta['state']['shape'][0]

        # normalizer
        self.normalizer = LinearNormalizer()

        # Create encoders using factory
        self.state_encoder = EncoderFactory.create_state_encoder(
            encoder_type=state_encoder_type,
            state_dim=state_dim,
            normalizer=self.normalizer,
            hidden_dims=state_mlp_hidden_dims,
            output_dim=state_feat_dim
        )
        
        # Create image encoder using factory
        self.obs_encoder = EncoderFactory.create_image_encoder(
            encoder_type=visual_encoder,
            obs_shape_meta=obs_shape_meta,
            crop_shape=crop_shape,
            obs_encoder_group_norm=obs_encoder_group_norm,
            eval_fixed_crop=eval_fixed_crop,
            action_dim=action_dim
        )

        # Get feature dimensions
        image_feat_dim, state_feat_dim = EncoderFactory.validate_encoder_compatibility(
            image_encoder=self.obs_encoder,
            state_encoder=self.state_encoder,
            fusion_type=fuse_op,
            n_obs_steps=n_obs_steps
        )

        # Create feature fusion module
        self.feature_fusion = FeatureFusion(
            image_dim=image_feat_dim,
            state_dim=state_feat_dim,
            fusion_type=fuse_op,
            shared_encoder_type=shared_encoder_type,
            shared_encoder_kwargs=shared_encoder_kwargs
        )
        
        cond_dim = self.feature_fusion.output_dim

        # disable‐privileged machinery
        self.disable_privileged_method = disable_privileged_method
        if disable_privileged_method == 'gating':
            self.priv_gating_alpha = nn.Parameter(torch.tensor(0.0))
        if disable_privileged_method == 'mask':
            assert privileged_mask is not None, (
                "Provide privileged_mask if method=='mask'"
            )
            DimensionValidator.validate_batch_shapes(
                {'mask': privileged_mask.unsqueeze(0)}, 
                {'obs': {'mask': {'shape': list(privileged_mask.shape)}}},
                horizon=privileged_mask.shape[0],
                n_obs_steps=n_obs_steps
            )
            self.register_buffer(
                'privileged_mask',
                privileged_mask.view(1, n_obs_steps, 1)
            )
        # build diffusion model
        self.model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            n_cond_layers=n_cond_layers
        )
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,  # obs_as_cond=True
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

        # store hyperparams
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.pred_action_steps_only = pred_action_steps_only
        self.num_inference_steps = (
            num_inference_steps or noise_scheduler.config.num_train_timesteps
        )
        self.sample_kwargs = sample_kwargs

        print(f"Created DiffusionTransformerPrivilegedPolicy with:")
        print(f"  State encoder: {type(self.state_encoder).__name__}")
        print(f"  Image encoder: {type(self.obs_encoder).__name__}")
        print(f"  Feature fusion: {type(self.feature_fusion).__name__}")
        print(f"  Conditioning dim: {cond_dim}")
        print(f"  Privileged disable method: {disable_privileged_method}")
        print(f"  Training disable prob: {disable_privileged_prob}")

    def _disable_state(self, state_feats: torch.Tensor) -> torch.Tensor:
        """Disable state features based on the configured method."""
        m = self.disable_privileged_method
        if m in ('zero','skip'):
            return torch.zeros_like(state_feats)
        elif m == 'gating':
            alpha = torch.sigmoid(self.priv_gating_alpha)
            return state_feats * alpha
        elif m == 'mask':
            return state_feats * self.privileged_mask
        else:
            raise ValueError(f"Unknown disable_privileged_method={m}")

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        cond: torch.Tensor = None,
        generator: torch.Generator = None
    ) -> torch.Tensor:
        model = self.model
        scheduler = self.noise_scheduler

        traj = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            traj[condition_mask] = condition_data[condition_mask]
            out = model(traj, t, cond)
            traj = scheduler.step(
                out, t, traj,
                generator=generator,
                **self.sample_kwargs
            ).prev_sample
        traj[condition_mask] = condition_data[condition_mask]
        return traj

    def predict_action(self, obs_dict: dict) -> dict:
        """Predict actions using both image and state observations."""
        # Validate inputs
        DimensionValidator.validate_model_inputs(obs_dict, 'privileged', self.n_obs_steps)
        
        B = obs_dict['state'].shape[0]
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        # Process state features
        state = obs_dict['state'][:,:self.n_obs_steps]
        state_feats = self.state_encoder(state)
        
        # DISABLE STATES DURING EVALUATION (uncomment to enable)
        # state_feats = self._disable_state(state_feats)

        # Process image features
        image = obs_dict['image'][:,:self.n_obs_steps]
        
        # Convert image format for encoder
        image_processed = ImageProcessor.to_encoder_format(
            image, 
            'DrQCNN' if hasattr(self.obs_encoder, 'feature_dim') else 'ResNet18Conv'
        )
        
        # Encode images
        if hasattr(self.obs_encoder, 'feature_dim'):  # DrQCNN encoder
            B_T = image_processed.shape[0]
            img_feats = self.obs_encoder(image_processed)
            image_feats = img_feats.view(B, self.n_obs_steps, -1)
        else:  # Robomimic encoder
            img_feats = self.obs_encoder({'image': image_processed})
            if isinstance(img_feats, dict):
                img_feats = next(iter(img_feats.values()))
            image_feats = img_feats.view(B, self.n_obs_steps, -1)

        # Validate encoder outputs
        DimensionValidator.validate_encoder_outputs(
            image_feats, state_feats, B, self.n_obs_steps, 'privileged'
        )

        # Fuse features
        cond = self.feature_fusion(image_feats, state_feats)
        
        # Prepare sampling
        cond_data = torch.zeros(
            (B, self.horizon, self.action_dim),
            device=device, dtype=dtype
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # Sample actions
        samples = self.conditional_sample(
            cond_data, cond_mask, cond=cond
        )
        naction = samples[..., :self.action_dim]
        action_pred = self.normalizer['action'].unnormalize(naction)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = self.n_obs_steps - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
        return {'action': action, 'action_pred': action_pred}

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """Compute training loss with optional modality dropout."""
        # Validate batch
        DimensionValidator.validate_batch_shapes(
            batch, 
            {'obs': {'state': {'shape': [self.state_encoder.state_dim]}, 
                    'image': {'shape': [32, 32, 3]}},  # Approximate image shape
             'action': {'shape': [self.action_dim]}},
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps
        )
        
        nactions = self.normalizer['action'].normalize(batch['action'])
        B, T, _ = nactions.shape
        To = self.n_obs_steps

        # Process state features
        state = batch['state'][:, :To]
        state_feats = self.state_encoder(state)

        # Process image features  
        image = batch['image'][:, :To]
        
        # Convert image format for encoder
        image_processed = ImageProcessor.to_encoder_format(
            image,
            'DrQCNN' if hasattr(self.obs_encoder, 'feature_dim') else 'ResNet18Conv'
        )
        
        # Encode images
        if hasattr(self.obs_encoder, 'feature_dim'):  # DrQCNN encoder
            B_T = image_processed.shape[0]
            img_feats = self.obs_encoder(image_processed)
            image_feats = img_feats.view(B, To, -1)
        else:  # Robomimic encoder
            img_feats = self.obs_encoder({'image': image_processed})
            if isinstance(img_feats, dict):
                img_feats = next(iter(img_feats.values()))
            image_feats = img_feats.view(B, To, -1)

        # Validate encoder outputs
        DimensionValidator.validate_encoder_outputs(
            image_feats, state_feats, B, To, 'privileged'
        )

        # Apply modality dropout during training
        if self.disable_privileged_prob > 0.0 and self.training:
            device = state_feats.device
            
            # Generate random values for each sample
            rand_vals = torch.rand(B, device=device)
            
            # Create masks for different dropout scenarios
            # 0-0.25: keep both modalities
            # 0.25-0.75: drop state, keep image  
            # 0.75-1.0: keep state, drop image
            disable_state_mask = (rand_vals >= 0.25) & (rand_vals < 0.75)
            disable_image_mask = rand_vals >= 0.75
            
            # Apply dropout
            if disable_state_mask.any():
                dropped_state = self._disable_state(state_feats)
                state_feats = torch.where(
                    disable_state_mask.view(-1, 1, 1), 
                    dropped_state, 
                    state_feats
                )
            
            if disable_image_mask.any():
                zero_image = torch.zeros_like(image_feats)
                image_feats = torch.where(
                    disable_image_mask.view(-1, 1, 1),
                    zero_image,
                    image_feats
                )
            
            # Log dropout statistics
            num_state_disabled = disable_state_mask.sum().item()
            num_image_disabled = disable_image_mask.sum().item()
            num_both_enabled = B - num_state_disabled - num_image_disabled
            if torch.rand(1).item() < 0.01:  # Log occasionally
                print(f"Modality dropout: {num_both_enabled}/{B} both enabled, "
                      f"{num_state_disabled}/{B} state disabled, "
                      f"{num_image_disabled}/{B} image disabled")

        # Fuse features
        cond = self.feature_fusion(image_feats, state_feats)

        # Prepare trajectory target
        if self.pred_action_steps_only:
            start = To - 1
            end = start + self.n_action_steps
            trajectory = nactions[:, start:end]
        else:
            trajectory = nactions

        # Create mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Forward diffusion
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=trajectory.device
        ).long()
        noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noisy[condition_mask] = trajectory[condition_mask]

        # Predict and compute loss
        pred = self.model(noisy, timesteps, cond)
        ptype = self.noise_scheduler.config.prediction_type
        target = noise if ptype=='epsilon' else trajectory
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * (~condition_mask).type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
        
        # Optional contrastive loss for cross-attention
        if (self.use_contrastive_loss and 
            hasattr(self.feature_fusion, 'shared_encoder_type') and
            self.feature_fusion.shared_encoder_type == 'cross_attention'):
            
            # Compute features with disabled modalities
            img_only_cond = self.feature_fusion.disable_modality(
                image_feats, torch.zeros_like(state_feats), disable_state=True
            )
            state_only_cond = self.feature_fusion.disable_modality(
                torch.zeros_like(image_feats), state_feats, disable_image=True
            )
            
            contrastive_loss = (
                F.mse_loss(cond, img_only_cond.detach()) +
                F.mse_loss(cond, state_only_cond.detach())
            )
            loss = loss + self.contrastive_weight * contrastive_loss
        
        return loss

    def get_optimizer(
        self,
        transformer_weight_decay: float,
        obs_encoder_weight_decay: float,
        state_encoder_weight_decay: float,
        shared_encoder_weight_decay: float,
        learning_rate: float,
        betas: tuple
    ) -> torch.optim.Optimizer:
        """Create optimizer with proper parameter grouping using optimizer factory."""
        enc_wd = {
            'obs': obs_encoder_weight_decay,
            'state': state_encoder_weight_decay,
            'shared': shared_encoder_weight_decay
        }
        return DiffusionPolicyOptimizer.create_transformer_optimizer(
            policy=self,
            transformer_weight_decay=transformer_weight_decay,
            encoder_weight_decay=enc_wd,
            learning_rate=learning_rate,
            betas=betas
        )

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Load pretrained normalizer stats for both state and action."""
        self.normalizer.load_state_dict(normalizer.state_dict()) 