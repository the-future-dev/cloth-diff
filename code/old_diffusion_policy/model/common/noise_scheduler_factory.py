"""
Factory utilities for creating noise schedulers for diffusion models.
Centralizes scheduler configuration to ensure consistency across policies.
"""

from typing import Optional, Dict, Any
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def create_ddpm_scheduler(
    num_train_timesteps: int = 1000,
    beta_schedule: str = "squaredcos_cap_v2",
    prediction_type: str = "epsilon",
    clip_sample: bool = False,
    timestep_spacing: str = "leading",
    rescale_betas_zero_snr: bool = False,
    **kwargs: Any
) -> DDPMScheduler:
    """Create a DDPMScheduler with shared defaults.

    Args:
        num_train_timesteps: Number of diffusion steps during training.
        beta_schedule: Beta schedule type.
        prediction_type: Model prediction target ('epsilon' or 'sample').
        clip_sample: Whether to clip predicted samples.
        timestep_spacing: Spacing schedule for inference timesteps.
        rescale_betas_zero_snr: Enable zero SNR beta rescaling.
        **kwargs: Forward-compatible extra args.

    Returns:
        Configured DDPMScheduler instance.
    """
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        prediction_type=prediction_type,
        clip_sample=clip_sample,
        timestep_spacing=timestep_spacing,
        rescale_betas_zero_snr=rescale_betas_zero_snr,
        **kwargs
    )
    return scheduler
