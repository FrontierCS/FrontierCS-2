"""Nonlinear mixing functions."""

import torch


# ── 2D mixing functions ──────────────────────────────────────────────────────

def mix_spiral(z):
    """g(z) = R(π‖z‖) z — measure-preserving spiral diffeomorphism."""
    norms = z.norm(dim=-1) * torch.pi
    c, s = norms.cos(), norms.sin()
    R = torch.stack([torch.stack([c, -s], dim=-1),
                     torch.stack([s,  c], dim=-1)], dim=-2)
    return (R @ z.unsqueeze(-1)).squeeze(-1)


def mix_banana(z):
    """Banana: x0 = z0, x1 = z1 + z0²."""
    return torch.stack([z[..., 0], z[..., 1] + z[..., 0] ** 2], dim=-1)


def mix_sinusoid(z):
    """Sinusoidal shear: x0 = z0 + sin(1.5 z1), x1 = z1."""
    return torch.stack([z[..., 0] + torch.sin(1.5 * z[..., 1]), z[..., 1]], dim=-1)


MIXINGS_2D = {
    "spiral": mix_spiral,
    "banana": mix_banana,
    "sinusoid": mix_sinusoid,
    # "nvp" handled via make_coupling_mixing(N=2, n_layers=...)
}


# ── Coupling-layer mixing (any dimension) ────────────────────────────────────

def make_coupling_mixing(N, n_layers=4, seed=1337, device="cuda"):
    """RealNVP-style coupling layers. Works for any even N (including N=2)."""
    half = N // 2
    torch.manual_seed(seed)
    Ws = []
    for _ in range(n_layers):
        W, _ = torch.linalg.qr(torch.randn(half, half, device=device))
        Ws.append(W * 2.0)

    def mix(z):
        for i, W in enumerate(Ws):
            z1, z2 = z[..., :half], z[..., half:]
            if i % 2 == 0:
                z2 = z2 + torch.tanh(z1 @ W)
            else:
                z1 = z1 + torch.tanh(z2 @ W)
            z = torch.cat([z1, z2], dim=-1)
        return z

    return mix
