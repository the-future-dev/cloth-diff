from __future__ import annotations
from typing import Dict, Optional
import torch

from ml_framework.interfaces.interfaces import Policy
from ml_framework.data.normalizer import IdentityNormalizer
from ml_framework.models.encoders import ObsEncoder, create_encoder

class BaseLowdimPolicy(Policy, torch.nn.Module):
    """Base interface for low-dim policies with obs_encoder support and convenience device/dtype accessors."""
    def __init__(self, obs_encoder: Optional[ObsEncoder] = None) -> None:
        super().__init__()
        self.normalizer = IdentityNormalizer()
        self.obs_encoder = obs_encoder
    
    def set_obs_encoder(self, obs_encoder: ObsEncoder):
        """Set the observation encoder."""
        self.obs_encoder = obs_encoder

    # For compatibility with old ModuleAttrMixin
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device('cpu')

    @property
    def dtype(self):
        p = next(self.parameters(), None)
        return p.dtype if p is not None else torch.float32

    def encode_observations(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observations using the obs_encoder.
        
        Args:
            obs_dict: Dictionary containing observations
            
        Returns:
            encoded_obs: Encoded observations ready for backbone
        """
        obs = obs_dict["obs"]  # [B, T, obs_dim]
        
        if self.obs_encoder is not None:
            return self.obs_encoder(obs)
        else:
            # No encoder - pass through observations directly
            return obs

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def reset(self):
        pass

    def set_normalizer(self, normalizer: IdentityNormalizer):
        raise NotImplementedError()
