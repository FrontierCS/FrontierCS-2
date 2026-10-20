"""Loss interfaces used by the experiment engine."""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Differentiable representation regularizer interface."""

    def __init__(self, knots=17, n_slices=256, t_max=3.0):
        super().__init__()
        raise NotImplementedError("Implement SIGReg")

    def forward(self, h):
        """Return a differentiable scalar for a (views, batch, features) tensor."""
        raise NotImplementedError("Implement SIGReg.forward")


def whitening_loss(h):
    """Return a differentiable scalar for a (views, batch, features) tensor."""
    raise NotImplementedError("Implement whitening_loss")


def alignment_loss(h):
    """Return a differentiable scalar for a (views, batch, features) tensor."""
    raise NotImplementedError("Implement alignment_loss")


def infonce_loss(h, sigma):
    """Return a differentiable scalar for a (views, batch, features) tensor."""
    raise NotImplementedError("Implement infonce_loss")
