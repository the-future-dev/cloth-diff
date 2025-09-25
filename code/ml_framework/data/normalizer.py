from __future__ import annotations
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch.nn as nn

class Normalizer(ABC):
    @abstractmethod
    def fit(self, data: Dict[str, torch.Tensor]) -> None: ...

    @abstractmethod
    def normalize(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]: ...

    @abstractmethod
    def unnormalize(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]: ...

    @abstractmethod
    def state_dict(self, destination=None, prefix='', keep_vars=False): ...

    @abstractmethod
    def load_state_dict(self, state_dict, strict=True): ...



class IdentityNormalizer(Normalizer, nn.Module):
    """No-op normalizer used as a placeholder.

    Methods accept/return dicts of tensors for consistency with richer normalizers.
    """
    def __init__(self) -> None:
        super().__init__()

    def fit(self, data, **kwargs):  # ignore
        return None

    def normalize(self, data):
        return data

    def unnormalize(self, data):
        return data

    def state_dict(self, *args, **kwargs):
        """Override to be compatible with PyTorch's state_dict signature."""
        # If called with PyTorch's standard arguments, delegate to parent
        if args or any(k in kwargs for k in ['destination', 'prefix', 'keep_vars']):
            return super().state_dict(*args, **kwargs)
        # Otherwise, return our custom state (for backward compatibility)
        return {}

    def load_state_dict(self, state_dict, strict=True):
        """Override to be compatible with PyTorch's load_state_dict signature."""
        # IdentityNormalizer has no state to load, so just pass
        return None

class RunningMeanStd:
    """Tracks running mean and variance for normalization."""
    def __init__(self, shape, epsilon=1e-8):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon
        self.epsilon = epsilon

    def to(self, device):
        """Move normalizer buffers to specified device."""
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self

    def update(self, x: torch.Tensor):
        device = x.device
        batch_mean = torch.mean(x, dim=[0, 1]).to(device)
        batch_var = torch.var(x, dim=[0, 1], unbiased=False).to(device)
        batch_count = x.shape[0] * x.shape[1]
        
        delta = batch_mean - self.mean
        self.mean = (self.mean + delta * batch_count / (self.count + batch_count)).to(device)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        self.var = (M2 / (self.count + batch_count)).to(device)
        self.count += batch_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / torch.sqrt(self.var.to(x.device) + self.epsilon)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(self.var.to(x.device) + self.epsilon) + self.mean.to(x.device)