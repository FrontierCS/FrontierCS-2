"""Evaluator-side adaptations for the byte-identical upstream entrypoint."""

import os as _os

from pruned_imports import install as _install_pruned

_install_pruned()

import numpy as _np
import torch as _torch
import torch.nn as _nn


class _NoopInception(_nn.Module):
    BLOCK_INDEX_BY_DIM = {2048: 0}

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return [_torch.zeros((x.shape[0], 2, 1, 1), device=x.device)]


try:
    import pytorch_fid.inception as _inception

    _inception.InceptionV3 = _NoopInception
except Exception:
    pass


if _os.environ.get("NANOWM_ENTRYPOINT"):
    try:
        from execution_audit import install as _install_audit
        _isolated = bool(_os.environ.get("NANOWM_CANDIDATE_REPO"))
        if _isolated:
            from sampler_bridge import install as _install_bridge
            _install_bridge()
        _install_audit(_os.environ["NANOWM_ENTRYPOINT"],
                       _os.environ["NANOWM_EXECUTION_AUDIT"], isolated=_isolated)
    except BaseException:
        # Python normally swallows sitecustomize errors; never fall back to the
        # trusted reference implementation when candidate isolation is missing.
        import traceback as _traceback
        _traceback.print_exc()
        _os._exit(70)
