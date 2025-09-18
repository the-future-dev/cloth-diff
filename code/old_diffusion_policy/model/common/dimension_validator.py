"""
Comprehensive dimension validation for diffusion policies.

Validates tensor shapes and dimensions across the entire pipeline to catch
inconsistencies early and provide clear error messages.
"""

import torch
from typing import Dict, Any, List, Tuple, Optional, Union


class DimensionValidator:
    """
    Validates tensor dimensions and shapes throughout the diffusion policy pipeline.
    """
    
    @staticmethod
    def validate_shape_meta(shape_meta: Dict[str, Any], model_type: str) -> None:
        """
        Validate shape_meta dictionary for consistency.
        
        Args:
            shape_meta: Dictionary containing shape information
            model_type: Type of model being used
        """
        # Check required keys
        required_keys = ['obs', 'action']
        for key in required_keys:
            if key not in shape_meta:
                raise KeyError(f"shape_meta missing required key: {key}")
        
        # Validate action shape
        action_shape = shape_meta['action']['shape']
        if not isinstance(action_shape, (list, tuple)) or len(action_shape) != 1:
            raise ValueError(f"Action shape must be 1D, got {action_shape}")
        
        if action_shape[0] <= 0:
            raise ValueError(f"Action dimension must be positive, got {action_shape[0]}")
        
        # Validate observations
        obs_meta = shape_meta['obs']
        if not isinstance(obs_meta, dict):
            raise TypeError("obs in shape_meta must be a dictionary")
        
        if len(obs_meta) == 0:
            raise ValueError("obs dictionary cannot be empty")
        
        # Validate each observation key
        for obs_key, obs_info in obs_meta.items():
            DimensionValidator._validate_obs_info(obs_key, obs_info, model_type)
        
        # Model-specific validations
        if model_type in ['privileged', 'double_modality']:
            if 'state' not in obs_meta:
                raise KeyError(f"Model type '{model_type}' requires 'state' in observations")
            
            # Check for image observations
            has_image = any(info.get('type') == 'rgb' for info in obs_meta.values())
            if not has_image:
                raise ValueError(f"Model type '{model_type}' requires at least one image observation")
    
    @staticmethod
    def _validate_obs_info(obs_key: str, obs_info: Dict[str, Any], model_type: str) -> None:
        """Validate individual observation info."""
        if not isinstance(obs_info, dict):
            raise TypeError(f"Observation info for '{obs_key}' must be a dictionary")
        
        if 'shape' not in obs_info:
            raise KeyError(f"Observation '{obs_key}' missing 'shape'")
        
        if 'type' not in obs_info:
            raise KeyError(f"Observation '{obs_key}' missing 'type'")
        
        shape = obs_info['shape']
        obs_type = obs_info['type']
        
        # Validate shape
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"Shape for '{obs_key}' must be list or tuple, got {type(shape)}")
        
        if len(shape) == 0:
            raise ValueError(f"Shape for '{obs_key}' cannot be empty")
        
        if any(dim <= 0 for dim in shape):
            raise ValueError(f"All shape dimensions for '{obs_key}' must be positive, got {shape}")
        
        # Validate type
        valid_types = ['rgb', 'low_dim', 'depth', 'scan']
        if obs_type not in valid_types:
            raise ValueError(f"Invalid observation type '{obs_type}' for '{obs_key}'. Valid types: {valid_types}")
        
        # Type-specific validations
        if obs_type == 'rgb':
            if len(shape) != 3:
                raise ValueError(f"RGB observation '{obs_key}' must have 3D shape, got {shape}")
            
            # Check if it's H,W,C format
            if shape[-1] not in [1, 3, 4]:
                print(f"Warning: RGB observation '{obs_key}' has unusual channel count: {shape[-1]}")
        
        elif obs_type == 'low_dim':
            if len(shape) != 1:
                raise ValueError(f"Low-dim observation '{obs_key}' must have 1D shape, got {shape}")
    
    @staticmethod
    def validate_batch_shapes(batch: Dict[str, torch.Tensor], 
                            shape_meta: Dict[str, Any],
                            horizon: int,
                            n_obs_steps: int) -> None:
        """
        Validate batch tensor shapes against expected shapes.
        
        Args:
            batch: Batch dictionary containing tensors
            shape_meta: Expected shape metadata
            horizon: Trajectory horizon
            n_obs_steps: Number of observation steps
        """
        batch_size = None
        
        # Validate action
        if 'action' in batch:
            action = batch['action']
            expected_action_shape = [horizon] + shape_meta['action']['shape']
            
            if len(action.shape) != 3:  # [B, T, action_dim]
                raise ValueError(f"Action tensor must be 3D [B, T, action_dim], got shape {action.shape}")
            
            if batch_size is None:
                batch_size = action.shape[0]
            elif action.shape[0] != batch_size:
                raise ValueError(f"Inconsistent batch size: action has {action.shape[0]}, expected {batch_size}")
            
            if action.shape[1] != horizon:
                raise ValueError(f"Action sequence length {action.shape[1]} != horizon {horizon}")
            
            if action.shape[2] != shape_meta['action']['shape'][0]:
                raise ValueError(f"Action dimension mismatch: got {action.shape[2]}, expected {shape_meta['action']['shape'][0]}")
        
        # Validate observations
        obs_meta = shape_meta['obs']
        for obs_key, obs_info in obs_meta.items():
            if obs_key in batch:
                obs_tensor = batch[obs_key]
                DimensionValidator._validate_obs_tensor(obs_key, obs_tensor, obs_info, batch_size, n_obs_steps)
        
        # Check for required keys in privileged/double_modality models
        if 'state' in obs_meta and 'state' not in batch:
            raise KeyError("Batch missing required 'state' tensor")
        
        if any(info.get('type') == 'rgb' for info in obs_meta.values()):
            image_keys = [k for k, v in obs_meta.items() if v.get('type') == 'rgb']
            for key in image_keys:
                if key in batch:
                    break
            else:
                raise KeyError(f"Batch missing required image tensor. Expected one of: {image_keys}")
    
    @staticmethod
    def _validate_obs_tensor(obs_key: str, 
                           obs_tensor: torch.Tensor,
                           obs_info: Dict[str, Any],
                           batch_size: int,
                           n_obs_steps: int) -> None:
        """Validate individual observation tensor."""
        expected_shape = obs_info['shape']
        obs_type = obs_info['type']
        
        # Check tensor properties
        if not isinstance(obs_tensor, torch.Tensor):
            raise TypeError(f"Observation '{obs_key}' must be a torch.Tensor, got {type(obs_tensor)}")
        
        if torch.isnan(obs_tensor).any():
            raise ValueError(f"Observation '{obs_key}' contains NaN values")
        
        if torch.isinf(obs_tensor).any():
            raise ValueError(f"Observation '{obs_key}' contains infinite values")
        
        # Check shape
        if obs_type == 'rgb':
            # Expect [B, T, H, W, C] or [B, T, C, H, W]
            if len(obs_tensor.shape) != 5:
                raise ValueError(f"Image observation '{obs_key}' must be 5D [B, T, H, W, C] or [B, T, C, H, W], got {obs_tensor.shape}")
            
            if obs_tensor.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch for '{obs_key}': got {obs_tensor.shape[0]}, expected {batch_size}")
            
            if obs_tensor.shape[1] < n_obs_steps:
                raise ValueError(f"Insufficient timesteps for '{obs_key}': got {obs_tensor.shape[1]}, need at least {n_obs_steps}")
            
            # Validate spatial dimensions (flexible format detection)
            if obs_tensor.shape[-1] in [1, 3, 4]:  # [B, T, H, W, C]
                h, w, c = obs_tensor.shape[2], obs_tensor.shape[3], obs_tensor.shape[4]
            elif obs_tensor.shape[2] in [1, 3, 4]:  # [B, T, C, H, W]
                c, h, w = obs_tensor.shape[2], obs_tensor.shape[3], obs_tensor.shape[4]
            else:
                raise ValueError(f"Cannot determine image format for '{obs_key}' with shape {obs_tensor.shape}")
            
            # Compare with expected shape (assuming H, W, C format in shape_meta)
            exp_h, exp_w, exp_c = expected_shape
            if (h, w, c) != (exp_h, exp_w, exp_c):
                print(f"Warning: Image '{obs_key}' shape mismatch. Expected (H,W,C)=({exp_h},{exp_w},{exp_c}), got ({h},{w},{c})")
        
        elif obs_type == 'low_dim':
            # Expect [B, T, feature_dim]
            if len(obs_tensor.shape) != 3:
                raise ValueError(f"Low-dim observation '{obs_key}' must be 3D [B, T, feature_dim], got {obs_tensor.shape}")
            
            if obs_tensor.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch for '{obs_key}': got {obs_tensor.shape[0]}, expected {batch_size}")
            
            if obs_tensor.shape[1] < n_obs_steps:
                raise ValueError(f"Insufficient timesteps for '{obs_key}': got {obs_tensor.shape[1]}, need at least {n_obs_steps}")
            
            if obs_tensor.shape[2] != expected_shape[0]:
                raise ValueError(f"Feature dimension mismatch for '{obs_key}': got {obs_tensor.shape[2]}, expected {expected_shape[0]}")
    
    @staticmethod
    def validate_model_inputs(inputs: Dict[str, torch.Tensor], 
                            model_type: str,
                            n_obs_steps: int) -> None:
        """
        Validate inputs to model forward pass.
        
        Args:
            inputs: Dictionary of input tensors
            model_type: Type of model
            n_obs_steps: Number of observation steps
        """
        if model_type in ['privileged', 'double_modality']:
            # Check required inputs
            required_keys = ['state', 'image']
            for key in required_keys:
                if key not in inputs:
                    raise KeyError(f"Model type '{model_type}' requires '{key}' input")
            
            state = inputs['state']
            image = inputs['image']
            
            # Validate state
            if len(state.shape) != 3:
                raise ValueError(f"State must be 3D [B, T, state_dim], got {state.shape}")
            
            # Validate image
            if len(image.shape) != 5:
                raise ValueError(f"Image must be 5D [B, T, ...], got {image.shape}")
            
            # Check batch consistency
            if state.shape[0] != image.shape[0]:
                raise ValueError(f"Batch size mismatch: state {state.shape[0]} vs image {image.shape[0]}")
            
            # Check timestep consistency
            if state.shape[1] < n_obs_steps or image.shape[1] < n_obs_steps:
                raise ValueError(f"Insufficient timesteps: state {state.shape[1]}, image {image.shape[1]}, need {n_obs_steps}")
    
    @staticmethod
    def validate_encoder_outputs(image_feats: torch.Tensor,
                               state_feats: Optional[torch.Tensor],
                               batch_size: int,
                               n_obs_steps: int,
                               model_type: str) -> None:
        """
        Validate encoder output dimensions.
        
        Args:
            image_feats: Image encoder output [B, T, img_dim]
            state_feats: State encoder output [B, T, state_dim] (optional)
            batch_size: Expected batch size
            n_obs_steps: Expected number of timesteps
            model_type: Type of model
        """
        # Validate image features
        if len(image_feats.shape) != 3:
            raise ValueError(f"Image features must be 3D [B, T, img_dim], got {image_feats.shape}")
        
        if image_feats.shape[0] != batch_size:
            raise ValueError(f"Image features batch size mismatch: got {image_feats.shape[0]}, expected {batch_size}")
        
        if image_feats.shape[1] != n_obs_steps:
            raise ValueError(f"Image features timestep mismatch: got {image_feats.shape[1]}, expected {n_obs_steps}")
        
        # Validate state features (if provided)
        if state_feats is not None:
            if len(state_feats.shape) != 3:
                raise ValueError(f"State features must be 3D [B, T, state_dim], got {state_feats.shape}")
            
            if state_feats.shape[0] != batch_size:
                raise ValueError(f"State features batch size mismatch: got {state_feats.shape[0]}, expected {batch_size}")
            
            if state_feats.shape[1] != n_obs_steps:
                raise ValueError(f"State features timestep mismatch: got {state_feats.shape[1]}, expected {n_obs_steps}")
            
            # Check for NaN/inf
            if torch.isnan(state_feats).any():
                raise ValueError("State features contain NaN values")
            
            if torch.isinf(state_feats).any():
                raise ValueError("State features contain infinite values")
        
        # Check for NaN/inf in image features
        if torch.isnan(image_feats).any():
            raise ValueError("Image features contain NaN values")
        
        if torch.isinf(image_feats).any():
            raise ValueError("Image features contain infinite values")
    
    @staticmethod
    def validate_fusion_inputs(state_feats: torch.Tensor,
                             image_feats: torch.Tensor,
                             fusion_type: str) -> None:
        """
        Validate inputs to feature fusion.
        
        Args:
            state_feats: State features [B, T, state_dim]
            image_feats: Image features [B, T, img_dim]
            fusion_type: Type of fusion ('concat', 'sum')
        """
        # Check shapes are compatible
        if state_feats.shape[:2] != image_feats.shape[:2]:
            raise ValueError(f"Feature shapes incompatible: state {state_feats.shape[:2]} vs image {image_feats.shape[:2]}")
        
        # Fusion-specific validations
        if fusion_type == 'sum':
            if state_feats.shape[2] != image_feats.shape[2]:
                raise ValueError(f"Sum fusion requires matching feature dims: state {state_feats.shape[2]} vs image {image_feats.shape[2]}")
        
        elif fusion_type == 'concat':
            # No additional constraints for concatenation
            pass
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
