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
    def state_dict(self) -> Dict[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None: ...



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


class MeanStdNormalizer(Normalizer, nn.Module):
    """Per-feature mean/std normalizer for keys present in the provided fit() dict.

    Stores running statistics (simple aggregate over full dataset passed once).
    """
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer('_fitted', torch.zeros(1, dtype=torch.bool))
        self.stats = nn.ModuleDict()  # key -> buffers

    def fit(self, data: dict, last_n_dims: int | None = None):
        for k, v in data.items():
            if not torch.is_tensor(v):
                continue
            flat = v.float().reshape(-1, v.shape[-1]) if (last_n_dims == 1 and v.dim() > 1) else v.float().view(-1)
            mean = flat.mean(0, keepdim=True)
            std = flat.std(0, unbiased=False, keepdim=True)
            mod = nn.Module()
            mod.register_buffer('mean', mean)
            mod.register_buffer('std', std)
            self.stats[k] = mod
        self._fitted[:] = True

    def _apply_norm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        # Broadcast-safe normalization
        return (x - mean.view(*([1] * (x.dim() - mean.dim())), *mean.shape)) / (std.view(*([1] * (x.dim() - std.dim())), *std.shape) + self.eps)

    def normalize(self, data: dict):
        if not bool(self._fitted):
            return data
        out = {}
        for k, v in data.items():
            if k in self.stats and torch.is_tensor(v):
                st = self.stats[k]
                out[k] = self._apply_norm(v, st.mean, st.std)
            else:
                out[k] = v
        return out

    def unnormalize(self, data: dict):
        if not bool(self._fitted):
            return data
        out = {}
        for k, v in data.items():
            if k in self.stats and torch.is_tensor(v):
                st = self.stats[k]
                out[k] = v * (st.std + self.eps) + st.mean
            else:
                out[k] = v
        return out

    def state_dict(self, *args, **kwargs):
        """Override to be compatible with PyTorch's state_dict signature."""
        # If called with PyTorch's standard arguments, delegate to parent
        if args or any(k in kwargs for k in ['destination', 'prefix', 'keep_vars']):
            return super().state_dict(*args, **kwargs)
        # Otherwise, return our custom state
        base = {'_fitted': bool(self._fitted)}
        payload = {}
        for k, mod in self.stats.items():
            payload[k] = {'mean': mod.mean.clone(), 'std': mod.std.clone()}
        base['stats'] = payload
        return base

    def load_state_dict(self, state_dict, strict=True):
        """Override to be compatible with PyTorch's load_state_dict signature."""
        # Handle both our custom format and empty state
        if not state_dict or not isinstance(state_dict, dict):
            return
        
        # If it's PyTorch's standard state_dict format, delegate to parent
        if any(k.startswith('stats.') or k.startswith('_fitted') for k in state_dict.keys()):
            return super().load_state_dict(state_dict, strict=strict)
        
        # Handle our custom format
        self.stats = nn.ModuleDict()
        self._fitted[:] = state_dict.get('_fitted', False)
        for k, stat in state_dict.get('stats', {}).items():
            mod = nn.Module()
            mod.register_buffer('mean', stat['mean'])
            mod.register_buffer('std', stat['std'])
            self.stats[k] = mod
