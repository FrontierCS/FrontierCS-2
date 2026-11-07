"""VAE precision policy — train/inference must use the same autocast mode.

Mirrors CogVideoX's disable_first_stage_autocast=True: VAE stays fp32 even when
the trainer is bf16-mixed, avoiding colored-speckle in decoded frames.
"""

import contextlib

import torch


_VALID_PRECISIONS = ("fp32", "bf16", "match_trainer")


def vae_autocast_context(vae, precision: str):
    if precision not in _VALID_PRECISIONS:
        raise ValueError(f"vae_precision={precision!r}; expected one of {_VALID_PRECISIONS}")
    device_type = vae.device.type if vae.device.type in ("cuda", "cpu") else "cuda"
    if precision == "fp32":
        return torch.autocast(device_type=device_type, enabled=False)
    if precision == "bf16":
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True)
    return contextlib.nullcontext()


def encode_first_stage(vae, x, precision: str = "fp32", sample: bool = True):
    """Pixels -> scaled latents (pre-multiplied by vae.config.scaling_factor)."""
    with vae_autocast_context(vae, precision):
        posterior = vae.encode(x).latent_dist
        z = posterior.sample() if sample else posterior.mode()
        return z.mul_(vae.config.scaling_factor)


def decode_first_stage(vae, z, precision: str = "fp32"):
    """Scaled latents -> pixels (divides by vae.config.scaling_factor internally)."""
    with vae_autocast_context(vae, precision):
        return vae.decode(z / vae.config.scaling_factor).sample
