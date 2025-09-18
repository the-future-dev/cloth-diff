"""
Shared sampling utilities for diffusion models.

Provides a single, tested pathway to run conditional DDPM sampling across
policies, reducing duplication and improving correctness.
"""
from typing import Optional, Dict, Any
import torch


@torch.no_grad()
def ddpm_conditional_sample(
    model: torch.nn.Module,
    scheduler,
    condition_data: torch.Tensor,
    condition_mask: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    num_inference_steps: Optional[int] = None,
    sample_kwargs: Optional[Dict[str, Any]] = None,
    generator: Optional[torch.Generator] = None,
):
    """Run DDPM sampling with masking-based conditioning.

    Args:
        model: Diffusion model with signature f(x_t, t, cond) -> eps/sample
        scheduler: Diffusers scheduler instance (e.g., DDPMScheduler)
        condition_data: Tensor with ground-truth values to enforce via mask
        condition_mask: Boolean mask indicating conditioned entries
        cond: Optional conditioning embedding fed to the model
        num_inference_steps: Number of inference timesteps; if None, use scheduler default
        sample_kwargs: Extra kwargs forwarded to scheduler.step
        generator: Optional RNG generator for deterministic sampling

    Returns:
        Sampled trajectory tensor with the same shape as condition_data.
    """
    device = condition_data.device
    dtype = condition_data.dtype
    sample_kwargs = sample_kwargs or {}

    traj = torch.randn(
        size=condition_data.shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )

    if num_inference_steps is not None:
        scheduler.set_timesteps(num_inference_steps)
    else:
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)

    for t in scheduler.timesteps:
        # 1) enforce conditioning before each step
        traj[condition_mask] = condition_data[condition_mask]

        # 2) predict model output
        model_output = model(traj, t, cond)

        # 3) compute previous sample x_t -> x_{t-1}
        out = scheduler.step(
            model_output,
            t,
            traj,
            generator=generator,
            **sample_kwargs,
        )
        traj = out.prev_sample

    # final enforcement
    traj[condition_mask] = condition_data[condition_mask]
    return traj
