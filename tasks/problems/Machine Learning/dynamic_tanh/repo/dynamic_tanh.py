import torch.nn as nn


class DynamicTanh(nn.Module):
    """Element-wise replacement module required by the model conversion hook."""

    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        raise NotImplementedError("implement the replacement module")

    def forward(self, x):
        raise NotImplementedError("implement the replacement mapping")


def convert_ln_to_dyt(module):
    """Return a model whose LayerNorm sites have been replaced."""
    raise NotImplementedError("implement the model conversion")
