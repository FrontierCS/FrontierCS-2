"""Evaluation metrics — standardized across all experiments."""

import torch

def bidirectional_r2(a, b):
    """R²(a->b) and R²(b->a) via torch lstsq on GPU. a, b are tensors."""
    def _r2(x, y):
        x1 = torch.cat([x, torch.ones(len(x), 1, device=x.device)], dim=1)
        W = torch.linalg.lstsq(x1, y).solution
        ss_res = ((y - x1 @ W) ** 2).sum()
        ss_tot = ((y - y.mean(0)) ** 2).sum()
        return (1 - ss_res / ss_tot).item()
    return _r2(a, b), _r2(b, a)


def compute_all_metrics(z, x, h, h_prime, rho, N):
    """All metrics on GPU. z, x, h, h_prime are torch tensors."""
    r2_zx, r2_xz = bidirectional_r2(z, x)
    r2_zh, r2_hz = bidirectional_r2(z, h)

    # Orthogonality
    z1 = torch.cat([z, torch.ones(len(z), 1, device=z.device)], dim=1)
    W = torch.linalg.lstsq(z1, h).solution
    A = W[:N].T
    orth_err = torch.linalg.norm(A.T @ A - torch.eye(N, device=A.device), 'fro').item()
    orth_err_normalized = orth_err / (N ** 0.5)

    # Bound quantities
    cov_h = torch.cov(h.T)
    epsilon = torch.linalg.norm(cov_h - torch.eye(N, device=h.device), 'fro').item()
    trace_cov = torch.trace(cov_h).item()
    L_h = ((h_prime - h) ** 2).sum(dim=1).mean().item()
    delta = max(L_h - 2 * (1 - rho) * trace_cov, 0.0)
    spectral_gap = 2 * rho * (1 - rho)
    D_bound = delta / spectral_gap if spectral_gap > 0 else float("inf")
    approx_bound = D_bound + (epsilon + D_bound) ** 2

    # Procrustes
    M = (h.T @ z) / len(z)
    U, S, Vt = torch.linalg.svd(M)
    Q = U @ Vt
    procrustes_mse = ((h - z @ Q.T) ** 2).sum(dim=1).mean().item()

    return {
        "r2_zx": r2_zx, "r2_xz": r2_xz,
        "r2_zh": r2_zh, "r2_hz": r2_hz,
        "orth_err": orth_err, "orth_err_normalized": orth_err_normalized,
        "epsilon": epsilon, "delta": delta, "D_bound": D_bound,
        "approx_bound": approx_bound, "procrustes_mse": procrustes_mse,
        "L_h": L_h, "trace_cov": trace_cov,
    }


def compute_recovery_metrics(z, h, N, suffix=""):
    """R² in both directions + orthogonality. Suffix appended to keys."""
    r2_zh, r2_hz = bidirectional_r2(z, h)
    z1 = torch.cat([z, torch.ones(len(z), 1, device=z.device)], dim=1)
    W = torch.linalg.lstsq(z1, h).solution
    A = W[:N].T
    orth_err = torch.linalg.norm(A.T @ A - torch.eye(N, device=A.device), 'fro').item()
    return {
        f"r2_zh{suffix}": r2_zh, f"r2_hz{suffix}": r2_hz,
        f"orth_err{suffix}": orth_err,
        f"orth_err_normalized{suffix}": orth_err / (N ** 0.5),
    }
