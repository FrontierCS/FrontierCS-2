"""Public interfaces for latent-video sampling."""

import numpy as np
import torch
from torch import Tensor
from typing import Optional, Tuple, Literal


def generate_full_sequence_schedule(sampling_timesteps: int, horizon: int) -> np.ndarray:
    """Return a schedule for ``horizon`` generated frames."""
    raise NotImplementedError("Implement the sampling schedule")


def generate_pyramid_schedule(sampling_timesteps: int, horizon: int) -> np.ndarray:
    """Return a schedule for ``horizon`` generated frames."""
    raise NotImplementedError("Implement the sampling schedule")


def generate_sequential_schedule(sampling_timesteps: int, horizon: int) -> np.ndarray:
    """Return a schedule for ``horizon`` generated frames."""
    raise NotImplementedError("Implement the sampling schedule")


def ddim_idx_to_timestep(
    indices: torch.Tensor,
    num_timesteps: int,
    sampling_timesteps: int,
) -> torch.Tensor:
    """Map schedule indices to diffusion timesteps."""
    raise NotImplementedError("Implement schedule conversion")


def generate_scheduling_matrix(
    num_frames: int,
    num_timesteps: int,
    sampling_timesteps: int,
    mode: Literal["full_sequence", "pyramid", "sequential"] = "pyramid",
    n_context_frames: int = 0,
) -> torch.Tensor:
    """Return a timestep matrix for the requested latent-video shape."""
    raise NotImplementedError("Implement scheduling-matrix construction")


def dfot_sample(
    diffusion,
    model,
    shape: Tuple[int, ...],
    context: Optional[Tensor] = None,
    n_context_frames: int = 0,
    scheduling_mode: Literal["full_sequence", "pyramid", "sequential"] = "pyramid",
    num_sampling_steps: int = 50,
    model_kwargs: Optional[dict] = None,
    device: Optional[torch.device] = None,
    progress: bool = True,
    eta: float = 0.0,
    clip_denoised: bool = False,
    n_generate_frames: Optional[int] = None,
    history_stabilization_level: float = 0.0,
) -> Tensor:
    """Return a sampled latent video with the requested ``shape``."""
    raise NotImplementedError("Implement the future-video sampler")
