"""
Unified encoder factory for diffusion policies.

Provides consistent creation of image and state encoders across all policy types.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List

# Import necessary components
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import replace_submodules
from core.awac import awacDrQCNNEncoder

from diffusion_policy.model.privileged.state_encoder import IdentityStateEncoder, MLPStateEncoder
from diffusion_policy.model.common.image_processor import ImageProcessor
from diffusion_policy.model.common.dimension_validator import DimensionValidator


class EncoderFactory:
    """
    Factory class for creating image and state encoders consistently.
    """
    
    @staticmethod
    def create_image_encoder(encoder_type: str,
                           obs_shape_meta: Dict[str, Any],
                           crop_shape: Optional[Tuple[int, int]] = None,
                           obs_encoder_group_norm: bool = False,
                           eval_fixed_crop: bool = False,
                           **encoder_kwargs) -> nn.Module:
        """
        Create image encoder based on type and configuration.
        
        Args:
            encoder_type: Type of encoder ('DrQCNN', 'ResNet18Conv', etc.)
            obs_shape_meta: Observation shape metadata
            crop_shape: Crop dimensions (height, width) or None to disable
            obs_encoder_group_norm: Use group normalization instead of batch norm
            eval_fixed_crop: Use fixed crop at evaluation time
            **encoder_kwargs: Additional encoder-specific arguments
            
        Returns:
            Configured image encoder module
        """
        # Validate inputs
        if not isinstance(obs_shape_meta, dict):
            raise TypeError("obs_shape_meta must be a dictionary")
        
        # Prepare observation configuration for robomimic
        obs_config = {'low_dim': [], 'rgb': [], 'depth': [], 'scan': []}
        obs_key_shapes = {}
        
        # Process observation metadata
        for key, attr in obs_shape_meta.items():
            if key == 'state':  # Skip state for image encoder
                continue
                
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            obs_type = attr.get('type', 'low_dim')
            
            if obs_type == 'rgb':
                obs_config['rgb'].append(key)
            elif obs_type == 'low_dim':
                obs_config['low_dim'].append(key)
            elif obs_type == 'depth':
                obs_config['depth'].append(key)
            elif obs_type == 'scan':
                obs_config['scan'].append(key)
            else:
                raise ValueError(f"Unsupported observation type: {obs_type}")
        
        if encoder_type == 'DrQCNN':
            return EncoderFactory._create_drq_encoder(obs_config, obs_key_shapes, **encoder_kwargs)
        else:
            return EncoderFactory._create_robomimic_encoder(
                encoder_type, obs_config, obs_key_shapes, crop_shape,
                obs_encoder_group_norm, eval_fixed_crop, **encoder_kwargs
            )
    
    @staticmethod
    def _create_drq_encoder(obs_config: Dict[str, List[str]],
                          obs_key_shapes: Dict[str, List[int]],
                          feature_dim: int = 50,
                          **kwargs) -> nn.Module:
        """Create DrQCNN encoder."""
        rgb_keys = obs_config['rgb']
        if len(rgb_keys) != 1:
            raise ValueError(f"DrQCNN only supports one RGB key, got {len(rgb_keys)}: {rgb_keys}")
        
        key = rgb_keys[0]
        if key not in obs_key_shapes:
            raise KeyError(f"RGB key '{key}' not found in obs_key_shapes")
        
        shape = obs_key_shapes[key]
        if len(shape) != 3:
            raise ValueError(f"Expected 3D shape for RGB observation, got {shape}")
        
        # Assume shape is [H, W, C]
        h, w, c = shape
        
        # Validate image dimensions
        if h <= 0 or w <= 0 or c <= 0:
            raise ValueError(f"Invalid image dimensions: H={h}, W={w}, C={c}")
        
        if c not in [1, 3, 4]:
            print(f"Warning: Unusual channel count for RGB image: {c}")
        
        encoder = awacDrQCNNEncoder(
            env_image_size=h,  # Assume square images or use height
            img_channel=c,
            feature_dim=feature_dim
        )
        
        print(f"Created DrQCNN encoder: input_size={h}x{w}x{c}, feature_dim={feature_dim}")
        return encoder
    
    @staticmethod
    def _create_robomimic_encoder(encoder_type: str,
                                obs_config: Dict[str, List[str]],
                                obs_key_shapes: Dict[str, List[int]],
                                crop_shape: Optional[Tuple[int, int]],
                                obs_encoder_group_norm: bool,
                                eval_fixed_crop: bool,
                                action_dim: int = 8,
                                **kwargs) -> nn.Module:
        """Create robomimic-based encoder."""
        # Get robomimic configuration
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph'
        )
        
        # Configure observations
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            
            # Configure crop randomizer
            if crop_shape is None:
                # Disable cropping
                for m in config.observation.encoder.values():
                    if m.obs_randomizer_class == 'CropRandomizer':
                        m.obs_randomizer_class = None
            else:
                # Set crop dimensions
                ch, cw = crop_shape
                for m in config.observation.encoder.values():
                    if m.obs_randomizer_class == 'CropRandomizer':
                        m.obs_randomizer_kwargs.crop_height = ch
                        m.obs_randomizer_kwargs.crop_width = cw
        
        # Initialize observation utilities
        ObsUtils.initialize_obs_utils_with_config(config)
        
        # Create policy to extract encoder
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu'
        )
        
        # Extract encoder
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        # Apply modifications
        if obs_encoder_group_norm:
            print("Replacing BatchNorm2d with GroupNorm in image encoder")
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features
                )
            )
        
        if eval_fixed_crop:
            print("Replacing CropRandomizer with fixed crop version")
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
        
        print(f"Created {encoder_type} robomimic encoder with crop_shape={crop_shape}")
        return obs_encoder
    
    @staticmethod
    def create_state_encoder(encoder_type: str,
                           state_dim: int,
                           normalizer=None,
                           hidden_dims: Optional[List[int]] = None,
                           output_dim: Optional[int] = None,
                           activation: str = 'relu',
                           **kwargs) -> nn.Module:
        """
        Create state encoder based on type.
        
        Args:
            encoder_type: Type of encoder ('identity', 'mlp')
            state_dim: Input state dimension
            normalizer: Normalizer for state features
            hidden_dims: Hidden layer dimensions for MLP encoder
            output_dim: Output dimension for MLP encoder
            activation: Activation function for MLP encoder
            **kwargs: Additional encoder arguments
            
        Returns:
            Configured state encoder module
        """
        # Validate inputs
        if state_dim <= 0:
            raise ValueError(f"State dimension must be positive, got {state_dim}")
        
        if encoder_type == 'identity':
            encoder = IdentityStateEncoder(state_dim, normalizer)
            print(f"Created identity state encoder: input_dim={state_dim}, output_dim={state_dim}")
            return encoder
        
        elif encoder_type == 'mlp':
            # Validate MLP-specific arguments
            if hidden_dims is None or len(hidden_dims) == 0:
                raise ValueError("MLP encoder requires non-empty hidden_dims")
            
            if output_dim is None or output_dim <= 0:
                raise ValueError("MLP encoder requires positive output_dim")
            
            if any(dim <= 0 for dim in hidden_dims):
                raise ValueError("All hidden dimensions must be positive")
            
            encoder = MLPStateEncoder(
                state_dim=state_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                normalizer=normalizer,
                activation=activation
            )
            
            print(f"Created MLP state encoder: input_dim={state_dim}, "
                  f"hidden_dims={hidden_dims}, output_dim={output_dim}")
            return encoder
        
        else:
            raise ValueError(f"Unknown state encoder type: {encoder_type}")
    
    @staticmethod
    def validate_encoder_compatibility(image_encoder: nn.Module,
                                     state_encoder: nn.Module,
                                     fusion_type: str,
                                     n_obs_steps: int) -> Tuple[int, int]:
        """
        Validate that encoders are compatible and return feature dimensions.
        
        Args:
            image_encoder: Image encoder module
            state_encoder: State encoder module
            fusion_type: Type of feature fusion ('concat', 'sum')
            n_obs_steps: Number of observation steps
            
        Returns:
            Tuple of (image_feat_dim, state_feat_dim)
        """
        # Get image feature dimension
        if hasattr(image_encoder, 'output_shape'):
            image_feat_dim = image_encoder.output_shape()[0]
        elif hasattr(image_encoder, 'output_dim'):
            image_feat_dim = image_encoder.output_dim
        else:
            # Try to infer from a forward pass
            print("Warning: Cannot determine image encoder output dimension, attempting inference...")
            try:
                # Create dummy input (this is a fallback and may not always work)
                with torch.no_grad():
                    dummy_input = torch.randn(1, 64, 64, 3)  # Dummy RGB image
                    output = image_encoder({'image': dummy_input})
                    if isinstance(output, dict):
                        image_feat_dim = next(iter(output.values())).shape[-1]
                    else:
                        image_feat_dim = output.shape[-1]
            except Exception as e:
                raise ValueError(f"Cannot determine image encoder output dimension: {e}")
        
        # Get state feature dimension
        if hasattr(state_encoder, 'output_dim'):
            state_feat_dim = state_encoder.output_dim
        elif hasattr(state_encoder, 'state_dim'):
            state_feat_dim = state_encoder.state_dim  # For identity encoder
        else:
            raise ValueError("Cannot determine state encoder output dimension")
        
        # Validate compatibility
        if fusion_type == 'sum':
            if image_feat_dim != state_feat_dim:
                raise ValueError(
                    f"Sum fusion requires matching feature dimensions: "
                    f"image={image_feat_dim}, state={state_feat_dim}"
                )
        
        print(f"Encoder compatibility validated: image_dim={image_feat_dim}, "
              f"state_dim={state_feat_dim}, fusion={fusion_type}")
        
        return image_feat_dim, state_feat_dim
    
    @staticmethod
    def test_encoder_forward_pass(image_encoder: nn.Module,
                                state_encoder: nn.Module,
                                obs_shape_meta: Dict[str, Any],
                                n_obs_steps: int,
                                batch_size: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Test encoder forward passes with dummy data.
        
        Args:
            image_encoder: Image encoder to test
            state_encoder: State encoder to test
            obs_shape_meta: Observation shape metadata
            n_obs_steps: Number of observation steps
            batch_size: Batch size for testing
            
        Returns:
            Tuple of (image_features, state_features)
        """
        device = next(image_encoder.parameters()).device
        
        # Create dummy image data
        image_key = None
        image_shape = None
        for key, info in obs_shape_meta.items():
            if info.get('type') == 'rgb':
                image_key = key
                image_shape = info['shape']  # [H, W, C]
                break
        
        if image_key is None:
            raise ValueError("No RGB observation found in obs_shape_meta")
        
        # Create dummy state data
        state_shape = obs_shape_meta.get('state', {}).get('shape', [18])  # Default state dim
        
        print(f"Testing encoders with batch_size={batch_size}, n_obs_steps={n_obs_steps}")
        print(f"Image shape: {image_shape}, State shape: {state_shape}")
        
        # Create dummy data
        h, w, c = image_shape
        dummy_images = torch.randn(batch_size, n_obs_steps, h, w, c, device=device)
        dummy_states = torch.randn(batch_size, n_obs_steps, state_shape[0], device=device)
        
        # Test image encoder
        try:
            # Convert to appropriate format for encoder
            if isinstance(image_encoder, awacDrQCNNEncoder):
                # DrQCNN expects [B*T, C, H, W]
                img_input = dummy_images.permute(0, 1, 4, 2, 3).contiguous()
                img_input = img_input.view(-1, c, h, w)
                img_features = image_encoder(img_input)
                img_features = img_features.view(batch_size, n_obs_steps, -1)
            else:
                # Robomimic encoder expects dict with proper format
                img_input = dummy_images.view(-1, h, w, c)
                img_features = image_encoder({image_key: img_input})
                if isinstance(img_features, dict):
                    img_features = next(iter(img_features.values()))
                img_features = img_features.view(batch_size, n_obs_steps, -1)
            
            print(f"Image encoder output shape: {img_features.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Image encoder forward pass failed: {e}")
        
        # Test state encoder
        try:
            state_features = state_encoder(dummy_states)
            print(f"State encoder output shape: {state_features.shape}")
            
        except Exception as e:
            raise RuntimeError(f"State encoder forward pass failed: {e}")
        
        # Validate output shapes
        expected_img_shape = (batch_size, n_obs_steps, img_features.shape[-1])
        expected_state_shape = (batch_size, n_obs_steps, state_features.shape[-1])
        
        if img_features.shape != expected_img_shape:
            raise ValueError(f"Image features shape mismatch: got {img_features.shape}, expected {expected_img_shape}")
        
        if state_features.shape != expected_state_shape:
            raise ValueError(f"State features shape mismatch: got {state_features.shape}, expected {expected_state_shape}")
        
        print("Encoder forward pass test completed successfully")
        return img_features, state_features
