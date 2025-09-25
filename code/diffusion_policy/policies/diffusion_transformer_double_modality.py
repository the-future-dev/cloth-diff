import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.optim import AdamW
from diffusion_policy.model.common.optimizer_factory import DiffusionPolicyOptimizer

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

# for image encoder (copied from HybridImagePolicy)
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from core.awac import awacDrQCNNEncoder

# our new helpers
from diffusion_policy.model.privileged.state_encoder import IdentityStateEncoder, MLPStateEncoder
from diffusion_policy.model.privileged.shared_encoder import get_shared_encoder

class DiffusionTransformerDoubleModalityPolicy(BaseImagePolicy):
    """
    A diffusion‐based policy that ALWAYS uses both `state` and `image`
    observations during BOTH training AND inference. No modality dropout.
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

        # parse dims
        action_dim = shape_meta['action']['shape'][0]
        obs_shape_meta = shape_meta['obs']
        state_dim = obs_shape_meta['state']['shape'][0]

        # normalizer
        self.normalizer = LinearNormalizer()

        # state‐encoder
        if state_encoder_type == 'identity':
            self.state_encoder = IdentityStateEncoder(state_dim, self.normalizer)
            state_feat_dim = state_dim
        elif state_encoder_type == 'mlp':
            assert state_mlp_hidden_dims and state_feat_dim, (
                "For 'mlp', pass state_mlp_hidden_dims + state_feat_dim"
            )
            assert all(dim > 0 for dim in state_mlp_hidden_dims), (
                "All dimensions in state_mlp_hidden_dims must be positive"
            )
            assert state_feat_dim > 0, (
                "state_feat_dim must be positive"
            )
            self.state_encoder = MLPStateEncoder(
                state_dim=state_dim,
                hidden_dims=state_mlp_hidden_dims,
                output_dim=state_feat_dim,
                normalizer=self.normalizer
            )
        else:
            raise ValueError(f"Unknown state_encoder_type={state_encoder_type}")

        # image‐encoder (copy from HybridImagePolicy)
        obs_config = {'low_dim':[], 'rgb':[], 'depth':[], 'scan':[]}
        obs_key_shapes = {}
        for key, attr in obs_shape_meta.items():
            if key == 'state': continue
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            t = attr.get('type','low_dim')
            if t == 'rgb':
                obs_config['rgb'].append(key)
            elif t == 'low_dim':
                obs_config['low_dim'].append(key)
            elif t == 'depth':
                obs_config['depth'].append(key)
            elif t == 'scan':
                obs_config['scan'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {t}")

        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph'
        )
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for m in config.observation.encoder.values():
                    if m.obs_randomizer_class == 'CropRandomizer':
                        m.obs_randomizer_class = None
            else:
                ch, cw = crop_shape
                for m in config.observation.encoder.values():
                    if m.obs_randomizer_class == 'CropRandomizer':
                        m.obs_randomizer_kwargs.crop_height = ch
                        m.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu'
        )

        if visual_encoder == 'DrQCNN':
            rgb_keys = obs_config['rgb']
            assert len(rgb_keys)==1, "DrQCNN only supports one RGB key"
            key = rgb_keys[0]
            H, W, C = obs_key_shapes[key]
            obs_encoder = awacDrQCNNEncoder(
                env_image_size=H, img_channel=C, feature_dim=50
            )
        else:
            obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
            if obs_encoder_group_norm:
                replace_submodules(
                    root_module=obs_encoder,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=x.num_features//16,
                        num_channels=x.num_features
                    )
                )
            if eval_fixed_crop:
                replace_submodules(
                    root_module=obs_encoder,
                    predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                    func=lambda x: dmvc.CropRandomizer(
                        input_shape=x.input_shape,
                        crop_height=x.crop_height,
                        crop_width=x.crop_width,
                        num_crops=x.num_crops,
                        pos_enc=x.pos_enc
                    )
                )

        self.obs_encoder = obs_encoder
        image_feat_dim = self.obs_encoder.output_shape()[0]

        # fusion
        assert fuse_op in ('concat','sum')
        self.fuse_op = fuse_op
        if fuse_op == 'sum':
            assert state_feat_dim == image_feat_dim, \
                "sum fusion requires matching dims"
            cond_dim = state_feat_dim
        else:
            cond_dim = state_feat_dim + image_feat_dim

        # shared encoder
        self.shared_encoder_type = shared_encoder_type
        if shared_encoder_type:
            self.shared_encoder = get_shared_encoder(
                shared_encoder_type,
                input_dim=cond_dim,
                **(shared_encoder_kwargs or {})
            )
            cond_dim = getattr(self.shared_encoder, 'output_dim', cond_dim)
        else:
            self.shared_encoder = None

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
        B = obs_dict['state'].shape[0]
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        # state branch - ALWAYS USED (no disabling)
        state = obs_dict['state'][:,:self.n_obs_steps]
        state_feats = self.state_encoder(state)

        # image branch - ALWAYS USED
        this_nobs = dict_apply(
            {'image': obs_dict['image']},
            lambda x: x[:,:self.n_obs_steps].reshape(-1, *x.shape[2:])
        )
        if isinstance(self.obs_encoder, awacDrQCNNEncoder):
            img_in = this_nobs['image'].permute(0,3,1,2).contiguous()
            img_feats = self.obs_encoder(img_in)
        else:
            img_feats = self.obs_encoder(this_nobs)
        image_feats = img_feats.view(B, self.n_obs_steps, -1)

        # fuse & shared - ALWAYS USE BOTH MODALITIES
        if self.fuse_op == 'sum':
            fused = state_feats + image_feats
        else:
            fused = torch.cat([state_feats, image_feats], dim=-1)
        if self.shared_encoder_type == 'cross_attention':
            cond = self.shared_encoder(image_feats, state_feats)
        else:
            cond = self.shared_encoder(fused) if self.shared_encoder else fused

        # cond_data / cond_mask
        cond_data = torch.zeros(
            (B, self.horizon, self.action_dim),
            device=device, dtype=dtype
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # sampling
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
        nactions = self.normalizer['action'].normalize(batch['action'])
        B, T, _ = nactions.shape
        To = self.n_obs_steps

        # state feats - ALWAYS USED during training
        state = batch['state'][:, :To]
        state_feats = self.state_encoder(state)

        # image feats - ALWAYS USED during training
        this_nobs = dict_apply(
            {'image': batch['image']},
            lambda x: x[:,:To].reshape(-1, *x.shape[2:])
        )
        if isinstance(self.obs_encoder, awacDrQCNNEncoder):
            img_in = this_nobs['image'].permute(0,3,1,2).contiguous()
            img_feats = self.obs_encoder(img_in)
        else:
            img_feats = self.obs_encoder(this_nobs)
        image_feats = img_feats.view(B, To, -1)

        # NO modality dropout during training - always use both modalities

        # fuse & shared - ALWAYS USE BOTH MODALITIES
        if self.fuse_op == 'sum':
            fused = state_feats + image_feats
        else:
            fused = torch.cat([state_feats, image_feats], dim=-1)

        # Shared Encoder
        if self.shared_encoder_type == 'cross_attention':
            cond = self.shared_encoder(image_feats, state_feats)
        else:
            cond = self.shared_encoder(fused) if self.shared_encoder else fused

        # trajectory target
        if self.pred_action_steps_only:
            start = To - 1
            end = start + self.n_action_steps
            trajectory = nactions[:, start:end]
        else:
            trajectory = nactions

        # mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # forward diffusion
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=trajectory.device
        ).long()
        noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noisy[condition_mask] = trajectory[condition_mask]

        # predict & loss
        pred = self.model(noisy, timesteps, cond)
        ptype = self.noise_scheduler.config.prediction_type
        target = noise if ptype=='epsilon' else trajectory
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * (~condition_mask).type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()

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
