import torch
import torch.nn as nn
from diffusion_policy.model.common.normalizer import LinearNormalizer

class IdentityStateEncoder(nn.Module):
    """
    Simply normalizes raw states via the shared LinearNormalizer.
    Input: state [B, T_obs, D_state]
    Output: [B, T_obs, D_state]
    """
    def __init__(self, state_dim: int, normalizer: LinearNormalizer):
        super().__init__()
        self.state_dim = state_dim
        self.normalizer = normalizer

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # apply per-feature normalization
        normalized_state = self.normalizer['state'].normalize(state)
        # ensure the output matches the expected state_dim
        assert normalized_state.shape[-1] == self.state_dim, f"Expected state dimension {self.state_dim}, but got {normalized_state.shape[-1]}"
        return normalized_state


class MLPStateEncoder(nn.Module):
    """
    MLP over normalized states. Reshapes [B,T,D]→[B*T,D]→MLP→[B*T, D_out]→[B,T,D_out]
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dims: list,
        output_dim: int,
        normalizer: LinearNormalizer,
        activation: str = 'relu',
    ):
        super().__init__()
        self.state_dim = state_dim
        self.normalizer = normalizer

        dims = [state_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 1:
                if activation.lower() == 'relu':
                    layers.append(nn.ReLU())
                # you can insert other activations here
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # normalize first
        x = self.normalizer['state'].normalize(state)
        B, T, _ = x.shape
        x = x.view(B * T, -1)
        x = self.net(x)
        return x.view(B, T, -1) 