from __future__ import annotations
from typing import Dict, Optional
import torch

from diffusion_policy.policies.interfaces import Policy
from diffusion_policy.data.normalizer import IdentityNormalizer
from diffusion_policy.models.encoders import ObsEncoder, create_encoder

class BasePrivilegedPolicy(Policy, torch.nn.Module):
    """Base interface for privileged policies that train with state+image but infer image-only.

    Implementations should:
    - compute_loss on a batch dict that may include keys: 'state', 'image', 'obs', 'action'
    - predict_action using only the image observation
    - Support separate encoders for training (state+image) and inference (image-only)
    """
    def __init__(self, 
                 image_encoder: Optional[ObsEncoder] = None,
                 state_encoder: Optional[ObsEncoder] = None) -> None:
        super().__init__()
        self.normalizer = IdentityNormalizer()
        self.image_encoder = image_encoder  # Used for inference and training
        self.state_encoder = state_encoder  # Used only for training
    
    def set_image_encoder(self, encoder: ObsEncoder):
        """Set the image observation encoder (used for both training and inference)."""
        self.image_encoder = encoder
        
    def set_state_encoder(self, encoder: ObsEncoder):
        """Set the state observation encoder (used only for training)."""
        self.state_encoder = encoder

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device('cpu')

    @property
    def dtype(self):
        p = next(self.parameters(), None)
        return p.dtype if p is not None else torch.float32

    def encode_image_observations(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode image observations using the image encoder."""
        if "image" in obs_dict:
            obs = obs_dict["image"]
        else:
            obs = obs_dict["obs"]  # fallback to generic obs key
            
        if self.image_encoder is not None:
            return self.image_encoder(obs)
        else:
            # No encoder - flatten images or pass through
            if obs.dim() == 5:  # Image format [B, T, C, H, W]
                B, T, C, H, W = obs.shape
                return obs.reshape(B, T, C * H * W)
            else:
                return obs
    
    def encode_state_observations(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode state observations using the state encoder."""
        obs = obs_dict["state"]
        
        if self.state_encoder is not None:
            return self.state_encoder(obs)
        else:
            return obs

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def reset(self):
        pass

    def set_normalizer(self, normalizer: IdentityNormalizer):
        raise NotImplementedError()
