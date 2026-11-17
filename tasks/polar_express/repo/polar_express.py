from itertools import repeat
from math import inf, sqrt

import numpy as np
from numpy.polynomial import Polynomial
import torch


def optimal_cubic(l, u):
    """Find an approximation to the constant function x -> 1
    of the form p(x) = a*x + b*x^3 that minimizes the maximum approximation error over the interval [l, u].

    Returns:
        a, b: coefficients of the optimal odd cubic approximant.
    """
    alpha = sqrt(3/(u**2 + l*u + l**2))
    beta = 4/(2 + l*u*(l + u)*(alpha**3))
    return (3/2)*alpha*beta, (-1/2)*(alpha**3)*beta


def optimal_quintic(l, u):
    """Return three finite odd-quintic coefficients for the interval ``[l, u]``."""
    raise NotImplementedError("optimal_quintic has been removed")

degree_to_coefficients = {3: optimal_cubic, 5: optimal_quintic}

def optimal_composition(l, num_iters, degree=5, safety_factor_eps=0, cushion=0):
    """Return ``num_iters`` odd-polynomial coefficient sequences."""
    raise NotImplementedError("optimal_composition has been removed")


coeffs_list = optimal_composition(l=1e-3, num_iters=10, degree=5, safety_factor_eps=1e-2, cushion=0.02)
# print("Polar Express Coefficient Series:", *coeffs_list, sep="\n")


@torch.compile
def PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Return a finite tensor with the same shape as ``G`` without mutating it."""
    raise NotImplementedError("PolarExpress has been removed")
