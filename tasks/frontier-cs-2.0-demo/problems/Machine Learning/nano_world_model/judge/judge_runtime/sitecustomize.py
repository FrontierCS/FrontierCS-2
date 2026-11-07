"""Evaluator-side adaptations for the byte-identical upstream entrypoint (judge/native_run.py).

Always: the pruned imports and an unscored FID network. With NANOWM_LATENTS_RECORD (the
candidate's machine) each dfot_sample result is saved there as <call>.npy, calls numbered from
0000; with NANOWM_LATENTS_REPLAY (the trusted runner) dfot_sample returns those latents in call
order instead of sampling. Decode, writer and ground truth read neither the context nor the RNG.
"""

import os as _os

try:
    import itertools as _itertools
    from pathlib import Path as _Path

    from pruned_imports import install as _install_pruned

    _install_pruned()

    import numpy as _np
    import pytorch_fid.inception as _inception
    import torch as _torch
    import torch.nn as _nn

    class _NoopInception(_nn.Module):
        BLOCK_INDEX_BY_DIM = {2048: 0}

        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return [_torch.zeros((x.shape[0], 2, 1, 1), device=x.device)]

    _inception.InceptionV3 = _NoopInception

    _calls = _itertools.count()
    if _os.environ.get("NANOWM_LATENTS_RECORD"):
        import diffusion.df_sample as _sampling

        _record = _Path(_os.environ["NANOWM_LATENTS_RECORD"])
        _native = _sampling.dfot_sample

        def _recorded(*args, **kwargs):
            sample = _native(*args, **kwargs)
            _np.save(_record / f"{next(_calls):04d}.npy", sample.detach().cpu().numpy())
            return sample

        _sampling.dfot_sample = _recorded
    elif _os.environ.get("NANOWM_LATENTS_REPLAY"):
        import diffusion.df_sample as _sampling

        _replay = _Path(_os.environ["NANOWM_LATENTS_REPLAY"])

        def _replayed(*args, device, **kwargs):
            path = _replay / f"{next(_calls):04d}.npy"
            return _torch.from_numpy(_np.load(path, allow_pickle=False)).to(device)

        _sampling.dfot_sample = _replayed
except BaseException:
    # Python reports a failing sitecustomize and carries on; never run without the adapters.
    import traceback as _traceback
    _traceback.print_exc()
    _os._exit(70)
