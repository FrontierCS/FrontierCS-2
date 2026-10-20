import torch

from utils.vae_ops import decode_first_stage, encode_first_stage

from .base import LatentShape


class SDVAELatentCodec:
    """Typed adapter around a diffusers AutoencoderKL.

    This class intentionally does not subclass nn.Module. The training module
    keeps the wrapped VAE registered as `self.vae` so old Lightning checkpoints
    that contain `vae.*` keys continue to load with the same state_dict layout.
    """

    kind = "sd_vae"
    has_decoder = True

    def __init__(self, vae, precision: str, latent_shape: LatentShape):
        self.vae = vae
        self.precision = precision
        self.latent_shape = latent_shape

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return encode_first_stage(self.vae, frames, precision=self.precision)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return decode_first_stage(self.vae, latents, precision=self.precision)

    def requires_grad_(self, requires_grad: bool) -> "SDVAELatentCodec":
        self.vae.requires_grad_(requires_grad)
        return self

    def eval(self) -> "SDVAELatentCodec":
        self.vae.eval()
        return self

    def to(self, device) -> "SDVAELatentCodec":
        self.vae.to(device)
        return self
