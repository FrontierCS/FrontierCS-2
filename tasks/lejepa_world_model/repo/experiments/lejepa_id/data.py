"""Data generation: latent sources and OU augmentation."""

import math
import torch


def _gennorm_unit_var_scale(alpha):
    """Scale β so gennorm(α, β) has unit variance: β = sqrt(Γ(1/α) / Γ(3/α))."""
    return math.exp(0.5 * (math.lgamma(1.0 / alpha) - math.lgamma(3.0 / alpha)))


def sample_latents(D, N, dist="gaussian", device="cuda", alpha=None):
    """Sample D points in R^N (unit variance)."""
    if dist == "gaussian":
        return torch.randn(D, N, device=device)
    elif dist == "laplace":
        return torch.distributions.Laplace(0, 1 / (2 ** 0.5)).sample((D, N)).to(device)
    elif dist == "gennorm":
        if alpha is None:
            raise ValueError("gennorm requires alpha")
        scale = _gennorm_unit_var_scale(alpha)
        u = torch.distributions.Gamma(1.0 / alpha, 1.0).sample((D, N)).to(device)
        sign = torch.randint(0, 2, (D, N), device=device).float() * 2 - 1
        return scale * sign * u.pow(1.0 / alpha)
    else:
        raise ValueError(f"Unknown distribution: {dist}")


def ou_augment(z, rho, n_views=2, dist="gaussian", alpha=None):
    """OU channel: z' = ρz + √(1-ρ²)η, η drawn from same dist as source.
    Returns (V, B, N)."""
    fac = (1 - rho ** 2) ** 0.5
    D, N = z.shape
    eta = sample_latents(n_views * D, N, dist=dist, device=z.device, alpha=alpha)
    eta = eta.reshape(n_views, D, N)
    return rho * z.unsqueeze(0) + fac * eta