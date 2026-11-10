"""Starter: broadcast the difference of every pair, square, sum. Exact and slow: it builds an
N x M x D temporary."""
import torch


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """out[i, j] = sum_k (x[i, k] - y[j, k]) ** 2 for x (N, D) and y (M, D), float32 on CUDA;
    returns an (N, M) float32 CUDA tensor."""
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
