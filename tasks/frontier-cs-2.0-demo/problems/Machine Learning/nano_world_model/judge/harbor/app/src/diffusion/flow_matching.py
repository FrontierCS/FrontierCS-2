"""
Flow Matching for NanoWM — drop-in replacement for GaussianDiffusion.

Implements conditional flow matching (Lipman et al. 2022 / Liu et al. 2022)
with a linear interpolation path:

    Forward:  x_t = τ * x_0 + (1 - τ) * ε,   τ = 1 - (t + 1) / T ∈ [0, 1)
    Target:   u = x_0 - ε                       (constant vector field)
    Loss:     MSE(model(x_t, t), u)

Inference uses simple Euler integration of the learned ODE:

    x_{t-1} = x_t + (τ_{t-1} - τ_t) * u_θ(x_t, t)

Key differences from GaussianDiffusion:
  - No β schedule; noise level is purely linear in τ
  - Training target is always u = x_0 - ε (no SNR weighting needed)
  - Inference needs far fewer steps (~10-20 vs 50-250 for DDIM)
  - Same dfot_sample_loop / dfot_ddim_sample API so df_sample.py is unchanged

API is intentionally identical to GaussianDiffusion so the training loop,
sampling utilities, and dfot_sample() all work without modification.
"""

import enum
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class PredName(enum.Enum):
    """Routing sentinel used by create_diffusion() to select FlowMatching."""
    FLOW = "flow"


def _mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def _t_to_tau(t: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """
    Map integer timestep t ∈ [0, T-1] to continuous flow time τ ∈ [0, 1).

    Matches GaussianDiffusion's convention where t=T-1 is the noisiest and
    t=0 is the least noisy (t=-1 = fully clean, handled upstream).

    Flow matching convention:
      τ = 0  →  x_t = ε  (pure noise)
      τ = 1  →  x_t = x_0 (pure data)

    Mapping (so denoising from high t to low t = integrating τ from 0→1):
      t = T-1  → τ = 0       (noisiest diffusion t → pure noise in FM)
      t = 0    → τ = 1-1/T ≈ 1 (least noisy diffusion t → nearly clean in FM)
    """
    return 1.0 - (t.float() + 1.0) / num_timesteps


def _broadcast_tau(tau: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expand τ to broadcast against a [B, ...] or [B, F, ...] target."""
    while len(tau.shape) < len(target.shape):
        tau = tau[..., None]
    return tau


class FlowMatching:
    """
    Conditional flow matching world model — same interface as GaussianDiffusion.

    Parameters
    ----------
    num_timesteps : int
        Number of discrete timestep levels (default 1000, same as diffusion).
        Kept large so the existing scheduling matrices / logit-normal sampler
        work without modification.
    snr_gamma : float
        Ignored (kept for API parity). Flow matching uses uniform loss weighting.
    """

    def __init__(self, num_timesteps: int = 1000, snr_gamma: float = 0.0):
        self.num_timesteps = num_timesteps
        self.snr_gamma = snr_gamma  # unused, kept for API compat

        # Dummy betas array so any code that reads diffusion.betas doesn't crash.
        # All entries = 1/T so the cumulative product decays linearly, mirroring
        # the linear interpolation path.
        self.betas = np.full(num_timesteps, 1.0 / num_timesteps, dtype=np.float64)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample x_t from the forward process: x_t = τ * x_0 + (1 - τ) * ε.

        Signature matches GaussianDiffusion.q_sample.  Returns x_t only
        (not the noise) to stay compatible with dfot_sample() which calls
        this as `context = diffusion.q_sample(context, t_stab)`.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        tau = _t_to_tau(t, self.num_timesteps)
        tau = _broadcast_tau(tau, x_start)
        return tau * x_start + (1.0 - tau) * noise

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Optional[dict] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute flow matching training loss for one batch.

        Returns a dict with keys "mse" and "loss" (identical for FM since
        we don't apply SNR weighting).  Shape matches GaussianDiffusion so
        the training loop can call `.mean()` on loss["loss"] unchanged.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise)
        target = x_start - noise  # u = x_0 - ε

        model_output = model(x_t, t, **model_kwargs)
        assert model_output.shape == target.shape == x_start.shape

        mse = _mean_flat((target - model_output) ** 2)
        return {"mse": mse, "loss": mse}

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _predict_xstart(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recover x_0 from x_t and the predicted vector field u_θ.

        Derivation:
            x_t = τ * x_0 + (1 - τ) * ε
            u   = x_0 - ε
          ⟹ x_0 = x_t + (1 - τ) * u
        """
        tau = _t_to_tau(t, self.num_timesteps)
        tau = _broadcast_tau(tau, x_t)
        return x_t + (1.0 - tau) * u

    # ------------------------------------------------------------------
    # DFoT sampling (same API as GaussianDiffusion)
    # ------------------------------------------------------------------

    def dfot_ddim_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        curr_t: torch.Tensor,
        next_t: torch.Tensor,
        clip_denoised: bool = True,
        denoised_fn=None,
        model_kwargs: Optional[dict] = None,
        eta: float = 0.0,          # unused for FM (deterministic ODE)
    ) -> dict:
        """Return ``sample`` and ``pred_xstart`` tensors matching ``x``."""
        raise NotImplementedError("Sampling transitions are outside this retained backend")

    def dfot_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        scheduling_matrix: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        n_context_frames: int = 0,
        clip_denoised: bool = True,
        denoised_fn=None,
        model_kwargs: Optional[dict] = None,
        device=None,
        progress: bool = False,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Return a latent-video tensor with ``shape``."""
        raise NotImplementedError("Sampling loops are outside this retained backend")
