from __future__ import annotations
from typing import Dict, Optional
import torch

from ml_framework.interfaces.interfaces import Policy

class BasePrivilegedPolicy(Policy, torch.nn.Module):
    """Base interface for privileged"""
    def __init__(self) -> None:
        super().__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device('cpu')

    @property
    def dtype(self):
        p = next(self.parameters(), None)
        return p.dtype if p is not None else torch.float32

    def encode_image_observations(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs = obs_dict.get("image", obs_dict.get("obs"))
        if obs.dim() == 5:
            B, T, C, H, W = obs.shape
            return obs.reshape(B, T, C * H * W)
        return obs

    def encode_state_observations(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return obs_dict["state"]

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def reset(self):
        pass

    def set_normalizer(self, normalizer):
        """Set normalizer - each policy implementation should handle this appropriately."""
        self.normalizer = normalizer
