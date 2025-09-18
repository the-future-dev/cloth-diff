"""
Configuration module for diffusion policy training/evaluation.

Handles argument parsing and environment setup.
"""

from collections import defaultdict
import argparse
from core.utils import str2bool, set_seed_everywhere, update_env_kwargs
from softgym.registered_env import env_arg_dict
from datetime import datetime

# Constants
reward_scales = defaultdict(lambda: 1.0)
clip_obs = defaultdict(lambda: None)


def setup_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser()

    ## general arguments
    parser.add_argument('--is_eval', type=str2bool, default=False, help="evaluation or training mode")
    parser.add_argument('--is_image_based', type=str2bool, default=False, help="state-based or image-based observations")
    parser.add_argument('--enable_img_transformations', type=str2bool, default=False, help="Whether to enable image transformations")
    parser.add_argument('--eval_videos', type=str2bool, default=False, help="whether or not to save evaluation video per episode")
    parser.add_argument('--eval_gif_size',  default=512, type=int, help="evaluation GIF width and height size")
    parser.add_argument('--model_save_dir', type=str, default='./data/diffusion', help="directory for saving trained model weights")
    parser.add_argument('--saved_rollouts', type=str, default=None, help="directory to load saved expert demonstrations from")
    parser.add_argument('--seed', type=int, default=1234, help="torch seed value")

    ## training arguments
    parser.add_argument('--train_data_ratio', type=float, default=0.95, help="ratio for training data for train-test split")
    parser.add_argument('--max_train_epochs', type=int, default=5000, help="ending epoch for training")
    parser.add_argument(
        '--max_train_steps',
        type=int,
        default=None,
        help="maximum number of gradient steps for fair dataset-size comparisons; if set, overrides max_train_epochs"
    )

    ## validation arguments
    parser.add_argument('--eval_interval', type=int, default=10, help="evaluation_interval")

    ## test arguments
    parser.add_argument('--test_checkpoint', type=str, default='./checkpoints/epoch_0.pth', help="checkpoint file for evaluation")
    parser.add_argument('--eval_over_five_seeds', default=False, type=str2bool, help="evaluation over 5 random seeds (100 episodes per seed)")

    ## arguments used in both validation and test
    parser.add_argument('--num_eval_eps', type=int, default=50, help="number of episodes to run during evaluation")

    ## logs
    parser.add_argument('--wandb', action='store_true', help="learning curves logged on weights and biases")
    parser.add_argument('--name', default=None, type=str, help='[optional] set experiment name. Useful to resume experiments.')

    ## diffusion model arguments
    parser.add_argument('--lrate', type=float, default=1e-4, help="initial learning rate for the policy network update")
    parser.add_argument('--beta1', type=float, default=0.95, help="betas for Adam Optimizer")
    parser.add_argument('--beta2', type=float, default=0.999, help="betas for Adam Optimizer")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size for model training")
    parser.add_argument('--scheduler_step_size', type=int, default=5, help="step size for optimizer scheduler")
    parser.add_argument('--scheduler_gamma', type=float, default=0.99, help="decay rate for optimizer scheduler")
    parser.add_argument('--discount_factor', type=float, default=0.99, help="discount factor for calculating discounted rewards")
    parser.add_argument('--observation_size', type=int, default=36, help="dimension of the observation space")
    parser.add_argument('--action_size', type=int, default=8, help="dimension of the action space")

    ## diffusion specific
    parser.add_argument('--horizon', type=int, default=16, help="diffusion model horizon")
    parser.add_argument('--n_obs_steps', type=int, default=2, help="number of observation steps")
    parser.add_argument('--n_action_steps', type=int, default=8, help="number of action steps to predict")
    parser.add_argument('--num_inference_steps', type=int, default=100, help="number of diffusion inference steps")
    parser.add_argument('--obs_as_global_cond', type=str2bool, default=True, help="use observations as global conditioning")
    parser.add_argument('--obs_as_local_cond', type=str2bool, default=False, help="use observations as local conditioning")
    parser.add_argument('--pred_action_steps_only', type=str2bool, default=False, help="predict only action steps")
    parser.add_argument('--use_ema', type=str2bool, default=True, help="use EMA model for evaluation")
    parser.add_argument('--model_type', choices=['unet', 'transformer', 'privileged', 'double_modality'], default='unet',
                        help="Model type for diffusion policy: unet, transformer, privileged (can disable state), or double_modality (always uses both state and image)")
    parser.add_argument('--disable_privileged_method', choices=['zero','skip','gating','mask'], default='zero',
                        help="Method to disable privileged state branch at test time (only for privileged model_type)")

    ## image-based model specific
    parser.add_argument('--env_img_size', type=int, default=128, help='Environment (observation) image size')
    parser.add_argument('--cnn_channels', nargs='+', type=int, default=[32, 64, 128, 256], help="CNN channels for image encoder")
    parser.add_argument('--cnn_kernels', nargs='+', type=int, default=[3, 3, 3, 3], help="CNN kernel sizes for image encoder")
    parser.add_argument('--cnn_strides', nargs='+', type=int, default=[2, 2, 2, 2], help="CNN strides for image encoder")
    parser.add_argument('--latent_dim', type=int, default=512, help="Latent dimension for image features")

    ## environment arguments
    parser.add_argument('--env_name', default='ClothFold')
    parser.add_argument('--env_kwargs_render', default=True, type=bool)
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)
    parser.add_argument('--env_kwargs_num_variations', default=1000, type=int)

    ## NEW: match run_bc interface
    parser.add_argument(
        '--load_ob_image_mode',
        default='direct',
        choices=['direct', 'separate_folder'],
        help='direct: load all images in memory; separate_folder: only load mini-batch images'
    )

    parser.add_argument('--lr_scheduler',
                        choices=['step', 'cosine'],
                        default='step',
                        help="learning rate scheduler type (step or cosine)")
    parser.add_argument('--lr_warmup_steps',
                        type=int,
                        default=500,
                        help="number of warm‐up epochs before cosine decay")

    # --- NEW: expose all scheduler hyperparameters as CLI flags ---
    parser.add_argument('--scheduler_num_train_timesteps',
                        type=int,
                        default=100,
                        help="diffusion scheduler: number of train timesteps")
    parser.add_argument('--scheduler_beta_start',
                        type=float,
                        default=0.0001,
                        help="diffusion scheduler: beta start")
    parser.add_argument('--scheduler_beta_end',
                        type=float,
                        default=0.02,
                        help="diffusion scheduler: beta end")
    parser.add_argument('--scheduler_beta_schedule',
                        type=str,
                        default="squaredcos_cap_v2",
                        help="diffusion scheduler: beta schedule")
    parser.add_argument('--scheduler_variance_type',
                        type=str,
                        default="fixed_small",
                        help="diffusion scheduler: variance type")
    parser.add_argument('--scheduler_clip_sample',
                        type=str2bool,
                        default=True,
                        help="diffusion scheduler: clip sample?")
    parser.add_argument('--scheduler_prediction_type',
                        type=str,
                        default="epsilon",
                        help="diffusion scheduler: prediction type")

    # --- NEW: expose policy architecture hyperparameters ---
    parser.add_argument('--channel_cond',
                        type=str2bool,
                        default=False,
                        help="image policy: use image channels as conditioning?")
    parser.add_argument('--cond_predict_scale',
                        type=str2bool,
                        default=True,
                        help="policy: condition predicts scale?")
    parser.add_argument('--diffusion_step_embed_dim',
                        type=int,
                        default=256,
                        help="UNet: diffusion step embedding dimension")
    parser.add_argument('--unet_down_dims',
                        nargs='+',
                        type=int,
                        default=[256, 512, 1024],
                        help="UNet: list of down‐sampling channel sizes")
    parser.add_argument('--unet_kernel_size',
                        type=int,
                        default=5,
                        help="UNet: convolutional kernel size")
    parser.add_argument('--unet_n_groups',
                        type=int,
                        default=8,
                        help="UNet: number of group‐norm groups")
    parser.add_argument('--oa_step_convention',
                        type=str2bool,
                        default=True,
                        help="lowdim UNet policy: use obs‐action step convention?")

    parser.add_argument('--resume',
                        action='store_true',
                        help="if set, resume training from the latest checkpoint")

    # --- Transformer-specific hyperparameters ---
    parser.add_argument('--transformer_n_emb',
                        type=int,
                        default=256,
                        help="Transformer embedding dimension (n_emb)")
    parser.add_argument('--transformer_n_layer',
                        type=int,
                        default=8,
                        help="Number of Transformer encoder layers (n_layer)")
    parser.add_argument('--transformer_n_head',
                        type=int,
                        default=4,
                        help="Number of attention heads (n_head)")
    parser.add_argument('--transformer_p_drop_emb',
                        type=float,
                        default=0.0,
                        help="Embedding dropout rate (p_drop_emb)")
    parser.add_argument('--transformer_p_drop_attn',
                        type=float,
                        default=0.3,
                        help="Attention dropout rate (p_drop_attn)")
    parser.add_argument('--transformer_causal_attn',
                        type=str2bool,
                        default=True,
                        help="Use causal attention? (causal_attn)")
    parser.add_argument('--transformer_time_as_cond',
                        type=str2bool,
                        default=True,
                        help="Condition on time token? (time_as_cond)")
    parser.add_argument('--transformer_n_cond_layers',
                        type=int,
                        default=0,
                        help="Number of Transformer layers for conditioning (n_cond_layers)")
    parser.add_argument('--crop_shape',
                        nargs=2,
                        type=int,
                        default=[76, 76],
                        help="Crop height and width for transformer image policy; use 0 0 to disable")

    # New arguments for optimizer setup
    parser.add_argument('--transformer_weight_decay',
                        type=float,
                        default=1e-6,
                        help="Weight decay for transformer parameters")
    parser.add_argument('--encoder_weight_decay',
                        type=float,
                        default=1e-4,
                        help="Weight decay for encoder parameters")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-6,
                        help="Generic weight decay for other policy types")

    # --- NEW: flag to choose image encoder ---
    parser.add_argument(
        '--visual_encoder',
        type=str,
        choices=['ResNet18Conv', 'DrQCNN'],
        default='ResNet18Conv',
        help="Which image encoder to use for transformer-hybrid policy"
    )

    # --- NEW: privileged policy specific arguments ---
    parser.add_argument('--disable_privileged_prob', type=float, default=0.0,
                        help="Probability of disabling the state branch during TRAINING for privileged model")
    parser.add_argument('--obs_encoder_group_norm', type=str2bool, default=False,
                        help="Use group normalization in privileged policy observation encoder")
    parser.add_argument('--eval_fixed_crop', type=str2bool, default=False,
                        help="Use fixed crop instead of random in privileged policy observation encoder at eval time")
    parser.add_argument('--state_encoder_type', choices=['identity','mlp'], default='identity',
                        help="State encoder type for privileged policy: 'identity' or 'mlp'")
    parser.add_argument('--state_mlp_hidden_dims', nargs='+', type=int, default=None,
                        help="Hidden dimensions for MLP state encoder in privileged policy")
    parser.add_argument('--state_feat_dim', type=int, default=None,
                        help="Output feature dimension for MLP state encoder in privileged policy")
    parser.add_argument('--priv_fuse_op', choices=['concat','sum'], default='concat',
                        help="Feature fusion operation for privileged policy: 'concat' or 'sum'")
    parser.add_argument('--shared_encoder_type', choices=['mlp','transformer','perceiver', 'cross_attention'], default=None,
                        help="Shared multimodal encoder type in privileged policy")
    parser.add_argument('--shared_encoder_kwargs', type=str, default=None,
                        help="JSON dict of kwargs for shared encoder in privileged policy")
    parser.add_argument('--privileged_mask', type=str, default=None,
                        help="Path to numpy .npy file with privileged mask for privileged policy")

    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of DataLoader workers')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Batches to prefetch per worker')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Use pin_memory=True in DataLoader')
    parser.add_argument('--persistent_workers', action='store_true',
                        help='Keep DataLoader workers alive between epochs')

    return parser


def setup_environment(args):
    """Set up environment-specific parameters."""
    env_name = args.env_name
    obs_mode = args.env_kwargs_observation_mode
    args.scale_reward = reward_scales[env_name]
    args.clip_obs = clip_obs[env_name] if obs_mode == 'key_point' else None
    args.env_kwargs = env_arg_dict[env_name]
    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs

    # Set is_image_based based on observation_mode if not explicitly set
    if args.env_kwargs['observation_mode'] == 'cam_rgb':
        args.is_image_based = True

    symbolic = args.env_kwargs['observation_mode'] != 'cam_rgb'
    args.encoder_type = 'identity' if symbolic else 'pixel'
    args.max_steps = 200

    env_kwargs = {
        'env': args.env_name,
        'symbolic': symbolic,
        'seed': args.seed,
        'max_episode_length': args.max_steps,
        'action_repeat': 1,
        'bit_depth': 8,
        'image_dim': None,
        'env_kwargs': args.env_kwargs,
        'normalize_observation': False,
        'scale_reward': args.scale_reward,
        'clip_obs': args.clip_obs,
        'obs_process': None,
    }

    now = datetime.now().strftime("%m.%d.%H.%M")
    args.folder_name = f'{args.env_name}_Diffusion_{now}' if not args.name else args.name

    # fix random seed
    set_seed_everywhere(args.seed)

    return env_kwargs
