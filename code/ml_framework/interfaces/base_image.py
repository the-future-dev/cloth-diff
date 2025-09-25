from __future__ import annotations
from typing import Dict, Optional
import torch

from ml_framework.interfaces.interfaces import Policy

class BaseImagePolicy(Policy, torch.nn.Module):
    """Base interface for image policies (no pluggable encoder abstraction).

    Policies directly consume either image tensors [B,T,C,H,W] or pre-extracted
    feature tensors [B,T,D]. Concrete policies decide how to flatten/transform.
    """
    def __init__(self) -> None:
        super().__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device('cpu')

    @property
    def dtype(self):
        p = next(self.parameters(), None)
        return p.dtype if p is not None else torch.float32


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def reset(self):
        pass

    def set_normalizer(self, normalizer):
        """Set normalizer - each policy implementation should handle this appropriately."""
        self.normalizer = normalizer
