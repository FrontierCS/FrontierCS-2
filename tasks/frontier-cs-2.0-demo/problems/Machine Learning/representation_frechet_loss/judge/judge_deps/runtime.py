"""Private offline inputs and bounded child-process lifecycle."""

import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def admit_reference(repo, *, pristine=None):
    """Admit only a trusted restored reference, never a solver submission.

    This guard remains mandatory for the parent scientific spine. Submitted
    queues are admitted separately, as data for the isolated actor only.
    """
    expected = json.loads((HERE / "source.json").read_text())["reference_files"]
    pristine = pristine or HERE.parent / "harbor/app"
    # The exporter adds the rendered public statement to its pristine substrate.
    if "README.md" not in expected and (pristine / "README.md").is_file():
        expected["README.md"] = sha256(pristine / "README.md")
    if any(p.is_symlink() for p in repo.rglob("*")):
        raise ValueError("reference source may not contain symlinks")
    actual = {str(p.relative_to(repo)): p for p in repo.rglob("*")
              if p.is_file() and ".git" not in p.relative_to(repo).parts
              and "__pycache__" not in p.relative_to(repo).parts}
    if set(actual) != set(expected) or any(
            p.is_symlink() or sha256(p) != expected[name] for name, p in actual.items()):
        raise ValueError("only the exact upstream reference is admitted as the trusted scientific spine")


def check_queue_runtime():
    """Bind the private source/tools before loading the observer integration."""
    root = HERE / "queue_boundary"
    expected = json.loads((HERE / "queue_runtime.json").read_text())["files"]
    for name, digest in expected.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError("private queue runtime changed: " + name)


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {("numpy", "ndarray"), ("numpy", "dtype")}
        for prefix in ("numpy.core", "numpy._core"):
            allowed.update((prefix + ".multiarray", name)
                           for name in ("_reconstruct", "scalar"))
            allowed.add((prefix + ".numeric", "_frombuffer"))
        if (module, name) not in allowed:
            raise pickle.UnpicklingError("unexpected statistics object")
        return super().find_class(module, name)


def prepare_assets(model_root, scratch):
    assets = json.loads((HERE / "assets.json").read_text())
    for item in assets:
        source = model_root / item["installed_path"]
        if not source.is_file() or source.stat().st_size != item["size_bytes"]:
            raise ValueError("missing or wrong-size offline asset: " + item["installed_path"])
        if sha256(source) != item["sha256"]:
            raise ValueError("offline asset hash mismatch: " + item["installed_path"])
    import numpy as np

    cache = scratch / "cache"
    for item in assets:
        if not item.get("repo_id", "").startswith("timm/"):
            continue
        hub = cache / "hub" / ("models--" + item["repo_id"].replace("/", "--"))
        target = hub / "snapshots" / item["revision"] / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(model_root / item["installed_path"])
        (hub / "refs").mkdir(exist_ok=True)
        (hub / "refs/main").write_text(item["revision"])
    stats = scratch / "stats"
    stats.mkdir()
    bundle_path = model_root / "fd-loss/data/fid_stats/paper_ref_stats.pkl"
    with bundle_path.open("rb") as stream:
        bundle = ArrayUnpickler(stream).load()
    if not isinstance(bundle, dict):
        raise ValueError("invalid released statistics bundle")
    for name, arrays in bundle.items():
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz"):
            raise ValueError("invalid statistics filename")
        if not isinstance(arrays, dict) or not {"mu", "sigma"} <= arrays.keys():
            raise ValueError("missing moments in statistics bundle")
        for value in arrays.values():
            if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError("non-finite or nonnumeric released statistics")
        dim = arrays["mu"].size
        if arrays["mu"].shape != (dim,) or arrays["sigma"].shape != (dim, dim):
            raise ValueError("invalid released moment shapes")
        if not np.allclose(arrays["sigma"], arrays["sigma"].T, atol=1e-10):
            raise ValueError("asymmetric released covariance")
        np.savez(stats / name, **arrays)
    return stats, {
        "HF_HOME": str(cache), "HF_HUB_CACHE": str(cache / "hub"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TORCH_HOME": str(model_root / "torch"),
    }


class SetupDeadlineExpired(TimeoutError):
    """The trusted runner deadline expired before launching any child."""


class SuiteDeadlineExpired(TimeoutError):
    """The shared scientific clock expired after scientific execution began."""


class ChildRunner:
    """One total deadline, including a cleanup reserve, for the complete suite."""

    def __init__(self, seconds, *, envelope_deadline=None, on_scientific_start=None):
        if seconds <= 20:
            raise ValueError("insufficient workload deadline")
        self.deadline = time.monotonic() + seconds - 10
        if envelope_deadline is not None:
            if not math.isfinite(envelope_deadline):
                raise ValueError("invalid workload envelope deadline")
            # Preserve the existing 90 + 10 second scientific cleanup reserve.
            self.deadline = min(self.deadline, envelope_deadline - 100)
            if self.deadline <= time.monotonic():
                raise SetupDeadlineExpired("workload setup deadline exhausted")
        self.active = None
        self.started = False
        self.on_scientific_start = on_scientific_start

    def stop(self):
        if self.active is not None:
            # Kill the whole dedicated group, including torchrun descendants.
            try:
                os.killpg(self.active.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.active.wait(timeout=5)
            self.active = None

    def run(self, argv, *, cwd, env, log, scientific=False):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            if not self.started:
                raise SetupDeadlineExpired("workload setup deadline exhausted")
            raise SuiteDeadlineExpired("workload deadline exhausted")
        with log.open("wb") as stream:
            self.active = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            try:
                if scientific and not self.started:
                    if self.on_scientific_start is not None:
                        self.on_scientific_start()
                    self.started = True
                code = self.active.wait(timeout=remaining)
                if code:
                    # Observer children also enforce the remaining shared clock.
                    if scientific and time.monotonic() >= self.deadline - 2:
                        raise SuiteDeadlineExpired("workload deadline exhausted")
                    # Retain the actual cause in private Verifier process logs;
                    # evaluate() returns only the fixed public failure message.
                    import sys
                    with log.open("rb") as failure:
                        failure.seek(max(0, log.stat().st_size - 8192))
                        sys.stderr.write(failure.read().decode("utf-8", errors="replace"))
                    raise RuntimeError("authentic workload process failed")
            except subprocess.TimeoutExpired as error:
                if not self.started:
                    raise SetupDeadlineExpired("workload setup deadline exhausted") from error
                raise
            finally:
                self.stop()
