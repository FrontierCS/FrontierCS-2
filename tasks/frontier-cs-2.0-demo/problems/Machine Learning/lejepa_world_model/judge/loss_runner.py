"""Runs inside an isolate instance (the shared launcher): the submitted losses' preflight,
purity check and training run. The judge stages this file as an input; the instance gives the
isolation (no judge files, no network, no new processes, private HOME/TMPDIR)."""

from __future__ import annotations

import importlib
import os
import runpy
import sys
from pathlib import Path

import yaml

# Exit status of the purity mode when a loss carries state between calls. No
# other path of this runner uses it, and submitted source cannot name an exit.
STATEFUL_LOSS_EXIT = 77
# The engine calls every scored loss once per optimizer update: 1,001 per run.
PURITY_CALLS = 1001
# Rows per view of the updates between comparisons. The comparisons themselves
# use the scored 256; replaying all 1,001 updates at 256 costs a whole training
# run of the paper loss (about 55 s here) inside the 400-second evaluator limit.
PURITY_FILLER_BATCH = 32
# Number of engine updates the used loss has already served at each comparison.
PURITY_PROBES = (1, 2, 10, 100, 300, 500, 750, 1000, 1001)
# One thread on CPU repeats these operations bit for bit in one process, and
# every accepted loss measured here differed by exactly zero. The comparison is
# still relative rather than exact because the fresh side is by construction a
# first call, and first calls into the CPU linear-algebra backend are the one
# place this pipeline is known to move in the sixth decimal (see _scores in
# evaluator.py). 1e-4 is about 800 float32 ulps and has no absolute term, so
# rescaling a loss cannot hide a change under it; a step-dependent weight that
# stays inside it at every probe cannot act as a schedule.
PURITY_RTOL = 1e-4
PURITY_SEED = 20260513



def _preflight() -> None:
    import torch

    module = importlib.import_module("lejepa_id.losses")
    checks = [
        lambda x: module.SIGReg(knots=9, n_slices=13, t_max=2.0)(x),
        module.whitening_loss,
        module.alignment_loss,
    ]
    for function in checks:
        value_in = torch.randn(2, 32, 2, dtype=torch.float32, requires_grad=True)
        value_out = function(value_in)
        if (value_out.ndim != 0 or value_out.dtype != value_in.dtype
                or value_out.device != value_in.device or not torch.isfinite(value_out)):
            raise RuntimeError("loss must be a finite scalar preserving dtype and device")
        value_out.backward()
        if (value_in.grad is None or not torch.isfinite(value_in.grad).all()
                or value_in.grad.abs().sum() == 0):
            raise RuntimeError("loss must provide finite, nonzero gradients")


def _fresh_losses():
    """Import the submitted losses anew and build SIGReg as the engine does."""
    import torch

    for name in [name for name in sys.modules
                 if name == "lejepa_id" or name.startswith("lejepa_id.")]:
        del sys.modules[name]
    # Import-time and constructor draws are part of a fresh loss, so both sides
    # of every comparison take them from the same generator state.
    torch.manual_seed(PURITY_SEED)
    module = importlib.import_module("lejepa_id.losses")
    return module, module.SIGReg().to("cpu")


def _engine_losses(module, sigreg, h):
    """One engine update's loss calls, in the engine's order."""
    align = module.alignment_loss(h)
    sig = sigreg(h)
    whiten = module.whitening_loss(h)
    return align, sig, whiten


def _loss_outputs(module, sigreg, h, seed):
    """Value and input gradient of each scored loss for one engine update."""
    import torch

    torch.manual_seed(seed)
    outputs = []
    for value in _engine_losses(module, sigreg, h):
        if value.ndim != 0 or not torch.isfinite(value):
            raise RuntimeError("loss must be a finite scalar preserving dtype and device")
        (grad,) = torch.autograd.grad(value, h, retain_graph=True, allow_unused=True)
        grad = torch.zeros_like(h) if grad is None else grad
        if not torch.isfinite(grad).all():
            raise RuntimeError("loss must provide finite, nonzero gradients")
        outputs += [value.detach().clone(), grad.detach().clone()]
    return outputs


def _same_outputs(fresh, used) -> bool:
    import torch

    for a, b in zip(fresh, used, strict=True):
        if a.shape != b.shape:
            return False
        if not torch.equal(a, b):
            scale = torch.maximum(a.abs().max(), b.abs().max())
            if not (a - b).abs().max() <= PURITY_RTOL * scale:
                return False
    return True


def _purity() -> None:
    """Reject a loss whose output depends on the calls made before it.

    The engine passes each loss the representation tensor and nothing else: no
    step, epoch, learning rate or optimizer. A loss that is a function of that
    argument, and of the draws it takes from the global generator during the
    call, therefore returns the same value and input gradient whether it was
    just imported or has already served any number of optimizer updates. Each
    probe makes that comparison on an input the used loss has not seen, so a
    cache keyed on the input cannot stand in for the missing state, and module
    globals, instance attributes and buffers, closures, function attributes
    and generators all start empty on the fresh side.
    """
    import torch
    from lejepa_id.data import ou_augment, sample_latents
    from lejepa_id.mixing import mix_spiral
    from lejepa_id.models import make_mlp_encoder

    # The loss sees what the engine gives it: a two-view encoder output with a
    # graph behind it, from the scored generator, mixing and encoder family.
    torch.manual_seed(PURITY_SEED)
    encoder = make_mlp_encoder(2, hidden=16, device="cpu")

    def observe(batch):
        with torch.no_grad():
            z = sample_latents(batch, 2, device="cpu")
            return mix_spiral(ou_augment(z, 0.95))

    def encode(x):
        return encoder(x.flatten(0, 1)).reshape(x.shape)

    inputs = [observe(256) for _ in PURITY_PROBES]
    seeds = [PURITY_SEED + 1 + index for index in range(len(PURITY_PROBES))]
    # Discarded: backend initialization belongs to neither side.
    _loss_outputs(*_fresh_losses(), encode(inputs[0]), seeds[0])
    fresh = [_loss_outputs(*_fresh_losses(), encode(x), seed)
             for x, seed in zip(inputs, seeds)]

    module, sigreg = _fresh_losses()
    for calls in range(1, PURITY_CALLS + 1):
        align, sig, _whiten = _engine_losses(module, sigreg, encode(observe(PURITY_FILLER_BATCH)))
        encoder.zero_grad()
        (0.001 * sig + 0.999 * align).backward()
        if calls in PURITY_PROBES:
            index = PURITY_PROBES.index(calls)
            used = _loss_outputs(module, sigreg, encode(inputs[index]), seeds[index])
            if not _same_outputs(fresh[index], used):
                print("loss output depends on earlier calls", file=sys.stderr)
                raise SystemExit(STATEFUL_LOSS_EXIT)



def _initialize(experiments: Path) -> None:
    os.chdir(experiments)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(experiments))


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in {"preflight", "purity", "train"}:
        raise SystemExit("usage: loss_runner.py preflight|purity|train EXPERIMENTS [args...]")
    mode = sys.argv[1]
    experiments = Path(sys.argv[2]).resolve()
    if mode == "train":
        # train EXPERIMENTS CONFIG OUT run.py-args...: the judge's workload config with its
        # results directory pointed at the instance's out/ (read back by the judge).
        config = yaml.safe_load(Path(sys.argv[3]).read_text(encoding="utf-8"))
        config["out"] = sys.argv[4]
        workload = Path(os.environ["TMPDIR"]) / "workload.yaml"
        workload.write_text(yaml.safe_dump(config), encoding="utf-8")
        run_args = ["--config", str(workload), *sys.argv[5:]]
    _initialize(experiments)
    if mode == "preflight":
        _preflight()
    elif mode == "purity":
        _purity()
    else:
        sys.argv = ["run.py", *run_args]
        runpy.run_path(str(experiments / "run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
