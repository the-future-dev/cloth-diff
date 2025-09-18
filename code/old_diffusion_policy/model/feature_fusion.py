"""
Feature fusion module for multi-modal diffusion policies.

Provides consistent and extensible feature fusion strategies for combining
image and state features in diffusion policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

from diffusion_policy.model.privileged.shared_encoder import get_shared_encoder
from diffusion_policy.model.common.dimension_validator import DimensionValidator


class BaseFusion(ABC):
    """Abstract base class for feature fusion strategies."""
    
    @abstractmethod
    def forward(self, 
                image_feats: torch.Tensor, 
                state_feats: torch.Tensor) -> torch.Tensor:
        """
        Fuse image and state features.
        
        Args:
            image_feats: Image features [B, T, img_dim]
            state_feats: State features [B, T, state_dim]
            
        Returns:
            Fused features [B, T, fused_dim]
        """
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return output feature dimension."""
        pass


class ConcatFusion(BaseFusion):
    """Simple concatenation fusion."""
    
    def __init__(self, image_dim: int, state_dim: int):
        self.image_dim = image_dim
        self.state_dim = state_dim
        self._output_dim = image_dim + state_dim
    
    def forward(self, 
                image_feats: torch.Tensor, 
                state_feats: torch.Tensor) -> torch.Tensor:
        """Concatenate features along last dimension."""
        # Validate inputs
        DimensionValidator.validate_fusion_inputs(state_feats, image_feats, 'concat')
        
        return torch.cat([image_feats, state_feats], dim=-1)
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


class SumFusion(BaseFusion):
    """Element-wise sum fusion."""
    
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self._output_dim = feature_dim
    
    def forward(self, 
                image_feats: torch.Tensor, 
                state_feats: torch.Tensor) -> torch.Tensor:
        """Element-wise sum of features."""
        # Validate inputs
        DimensionValidator.validate_fusion_inputs(state_feats, image_feats, 'sum')
        
        return image_feats + state_feats
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


class ProjectedSumFusion(BaseFusion):
    """Sum fusion with learned projections."""
    
    def __init__(self, image_dim: int, state_dim: int, output_dim: int):
        self.image_proj = nn.Linear(image_dim, output_dim)
        self.state_proj = nn.Linear(state_dim, output_dim)
        self._output_dim = output_dim
    
    def forward(self, 
                image_feats: torch.Tensor, 
                state_feats: torch.Tensor) -> torch.Tensor:
        """Project features to same dimension then sum."""
        # Project to common dimension
        img_proj = self.image_proj(image_feats)
        state_proj = self.state_proj(state_feats)
        
        return img_proj + state_proj
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


class WeightedFusion(BaseFusion):
    """Learnable weighted fusion."""
    
    def __init__(self, image_dim: int, state_dim: int):
        self.image_dim = image_dim
        self.state_dim = state_dim
        self._output_dim = image_dim + state_dim
        
        # Learnable weights
        self.image_weight = nn.Parameter(torch.tensor(1.0))
        self.state_weight = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, 
                image_feats: torch.Tensor, 
                state_feats: torch.Tensor) -> torch.Tensor:
        """Weighted concatenation of features."""
        # Apply learnable weights
        weighted_image = self.image_weight * image_feats
        weighted_state = self.state_weight * state_feats
        
        return torch.cat([weighted_image, weighted_state], dim=-1)
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


class FeatureFusion(nn.Module):
    """
    Unified feature fusion module with optional shared encoder.
    
    Handles fusion of multi-modal features (image + state) with various
    fusion strategies and optional post-fusion processing.
    """
    
    def __init__(self,
                 image_dim: int,
                 state_dim: int,
                 fusion_type: str = 'concat',
                 shared_encoder_type: Optional[str] = None,
                 shared_encoder_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize feature fusion module.
        
        Args:
            image_dim: Image feature dimension
            state_dim: State feature dimension
            fusion_type: Type of fusion ('concat', 'sum', 'projected_sum', 'weighted')
            shared_encoder_type: Type of shared encoder (None, 'mlp', 'transformer', etc.)
            shared_encoder_kwargs: Arguments for shared encoder
        """
        super().__init__()
        
        # Validate inputs
        if image_dim <= 0 or state_dim <= 0:
            raise ValueError(f"Feature dimensions must be positive: image={image_dim}, state={state_dim}")
        
        self.image_dim = image_dim
        self.state_dim = state_dim
        self.fusion_type = fusion_type
        self.shared_encoder_type = shared_encoder_type
        
        # Create fusion strategy
        self.fusion = self._create_fusion_strategy(fusion_type, image_dim, state_dim)
        
        # Create shared encoder if specified
        if shared_encoder_type is not None:
            shared_encoder_kwargs = shared_encoder_kwargs or {}
            
            # Special handling for cross-attention encoder
            if shared_encoder_type == 'cross_attention':
                # Remove conflicting parameters that we provide explicitly
                filtered_kwargs = {k: v for k, v in shared_encoder_kwargs.items() 
                                 if k not in ['img_dim', 'state_dim']}
                
                self.shared_encoder = get_shared_encoder(
                    shared_encoder_type,
                    img_dim=image_dim,
                    state_dim=state_dim,
                    **filtered_kwargs
                )
                self._output_dim = getattr(self.shared_encoder, 'output_dim', self.fusion.output_dim)
            else:
                # For other encoders, use fusion output as input
                # Remove conflicting parameters
                filtered_kwargs = {k: v for k, v in shared_encoder_kwargs.items() 
                                 if k not in ['input_dim']}
                
                self.shared_encoder = get_shared_encoder(
                    shared_encoder_type,
                    input_dim=self.fusion.output_dim,
                    **filtered_kwargs
                )
                self._output_dim = getattr(self.shared_encoder, 'output_dim', self.fusion.output_dim)
        else:
            self.shared_encoder = None
            self._output_dim = self.fusion.output_dim
        
        print(f"Created FeatureFusion: image_dim={image_dim}, state_dim={state_dim}, "
              f"fusion_type={fusion_type}, shared_encoder={shared_encoder_type}, "
              f"output_dim={self._output_dim}")
    
    def _create_fusion_strategy(self, fusion_type: str, image_dim: int, state_dim: int) -> BaseFusion:
        """Create the appropriate fusion strategy."""
        if fusion_type == 'concat':
            return ConcatFusion(image_dim, state_dim)
        
        elif fusion_type == 'sum':
            if image_dim != state_dim:
                raise ValueError(f"Sum fusion requires matching dimensions: image={image_dim}, state={state_dim}")
            return SumFusion(image_dim)
        
        elif fusion_type == 'projected_sum':
            # Use the larger dimension as output
            output_dim = max(image_dim, state_dim)
            return ProjectedSumFusion(image_dim, state_dim, output_dim)
        
        elif fusion_type == 'weighted':
            return WeightedFusion(image_dim, state_dim)
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, 
                image_feats: torch.Tensor, 
                state_feats: torch.Tensor,
                return_separate: bool = False) -> torch.Tensor:
        """
        Forward pass through fusion module.
        
        Args:
            image_feats: Image features [B, T, img_dim]
            state_feats: State features [B, T, state_dim]
            return_separate: If True, return features before shared encoder
            
        Returns:
            Fused features [B, T, output_dim]
        """
        # Validate input shapes
        if len(image_feats.shape) != 3 or len(state_feats.shape) != 3:
            raise ValueError(f"Expected 3D tensors [B, T, D], got image: {image_feats.shape}, state: {state_feats.shape}")
        
        if image_feats.shape[:2] != state_feats.shape[:2]:
            raise ValueError(f"Batch and time dimensions must match: image {image_feats.shape[:2]} vs state {state_feats.shape[:2]}")
        
        if image_feats.shape[2] != self.image_dim:
            raise ValueError(f"Image feature dimension mismatch: got {image_feats.shape[2]}, expected {self.image_dim}")
        
        if state_feats.shape[2] != self.state_dim:
            raise ValueError(f"State feature dimension mismatch: got {state_feats.shape[2]}, expected {self.state_dim}")
        
        # Apply fusion strategy
        if self.shared_encoder_type == 'cross_attention':
            # Cross-attention encoder takes separate inputs
            fused = self.shared_encoder(image_feats, state_feats)
        else:
            # Standard fusion then optional shared encoder
            fused = self.fusion.forward(image_feats, state_feats)
            
            if return_separate:
                return fused
            
            if self.shared_encoder is not None:
                fused = self.shared_encoder(fused)
        
        return fused
    
    @property
    def output_dim(self) -> int:
        """Return output feature dimension."""
        return self._output_dim
    
    def get_fusion_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get learnable fusion weights if available."""
        if isinstance(self.fusion, WeightedFusion):
            return {
                'image_weight': self.fusion.image_weight,
                'state_weight': self.fusion.state_weight
            }
        return None
    
    def disable_modality(self, 
                        image_feats: torch.Tensor, 
                        state_feats: torch.Tensor,
                        disable_image: bool = False,
                        disable_state: bool = False) -> torch.Tensor:
        """
        Forward pass with selective modality disabling.
        
        Args:
            image_feats: Image features [B, T, img_dim]
            state_feats: State features [B, T, state_dim]
            disable_image: Zero out image features
            disable_state: Zero out state features
            
        Returns:
            Fused features with selected modalities disabled
        """
        # Create modified features
        mod_image_feats = torch.zeros_like(image_feats) if disable_image else image_feats
        mod_state_feats = torch.zeros_like(state_feats) if disable_state else state_feats
        
        return self.forward(mod_image_feats, mod_state_feats)
    
    def test_forward_pass(self, 
                         batch_size: int = 2, 
                         n_obs_steps: int = 2,
                         device: str = 'cpu') -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Test forward pass with dummy data.
        
        Args:
            batch_size: Batch size for testing
            n_obs_steps: Number of observation steps
            device: Device for testing
            
        Returns:
            Tuple of (output_features, info_dict)
        """
        device = torch.device(device)
        self.to(device)
        
        # Create dummy features
        dummy_image = torch.randn(batch_size, n_obs_steps, self.image_dim, device=device)
        dummy_state = torch.randn(batch_size, n_obs_steps, self.state_dim, device=device)
        
        print(f"Testing FeatureFusion forward pass:")
        print(f"  Input shapes: image={dummy_image.shape}, state={dummy_state.shape}")
        
        # Test forward pass
        with torch.no_grad():
            output = self.forward(dummy_image, dummy_state)
        
        expected_shape = (batch_size, n_obs_steps, self.output_dim)
        if output.shape != expected_shape:
            raise RuntimeError(f"Output shape mismatch: got {output.shape}, expected {expected_shape}")
        
        # Test modality disabling
        output_no_image = self.disable_modality(dummy_image, dummy_state, disable_image=True)
        output_no_state = self.disable_modality(dummy_image, dummy_state, disable_state=True)
        
        info = {
            'output_shape': output.shape,
            'output_min': output.min().item(),
            'output_max': output.max().item(),
            'output_mean': output.mean().item(),
            'fusion_weights': self.get_fusion_weights()
        }
        
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{info['output_min']:.3f}, {info['output_max']:.3f}]")
        print(f"  Fusion test completed successfully")
        
        return output, info


# Factory function for easy creation
def create_feature_fusion(image_dim: int,
                         state_dim: int,
                         fusion_type: str = 'concat',
                         shared_encoder_type: Optional[str] = None,
                         shared_encoder_kwargs: Optional[Dict[str, Any]] = None) -> FeatureFusion:
    """
    Factory function to create FeatureFusion module.
    
    Args:
        image_dim: Image feature dimension
        state_dim: State feature dimension  
        fusion_type: Type of fusion strategy
        shared_encoder_type: Type of shared encoder
        shared_encoder_kwargs: Shared encoder configuration
        
    Returns:
        Configured FeatureFusion module
    """
    return FeatureFusion(
        image_dim=image_dim,
        state_dim=state_dim,
        fusion_type=fusion_type,
        shared_encoder_type=shared_encoder_type,
        shared_encoder_kwargs=shared_encoder_kwargs
    )
