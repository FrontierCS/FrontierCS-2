"""Run a fresh child with bounded pipes and an irreversible seal receipt."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from wire import MAX_MESSAGE_BYTES, MAX_SOURCE_BYTES, InvalidResponse


class WorkerInfrastructureError(RuntimeError):
    pass


def run_worker(script, request, *, timeout=1650, extra_env=None):
    body = json.dumps(request, allow_nan=False, separators=(",", ":")).encode()
    if len(body) > MAX_SOURCE_BYTES * 2:
        raise ValueError("worker request exceeds bound")
    env = {"PATH": os.path.dirname(sys.executable) + ":/usr/bin:/bin",
           "LANG": "C.UTF-8", "PYTHONHASHSEED": "3407",
           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "CUDA_CACHE_DISABLE": "1"}
    for name, value in (extra_env or {}).items():
        if name not in {"CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"}:
            raise ValueError("unexpected worker environment input")
        env[name] = value
    with tempfile.TemporaryDirectory(prefix="nanowm-child-") as directory:
        env.update(HOME=directory, TMPDIR=directory)
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as diagnostics:
            seal_read, seal_write = os.pipe()
            process = None
            try:
                process = subprocess.Popen(
                    [sys.executable, "-I", str(Path(script).resolve()),
                     str(seal_write), str(os.getpid())],
                    stdin=subprocess.PIPE, stdout=output, stderr=diagnostics,
                    close_fds=True, pass_fds=(seal_write,), cwd=directory,
                    env=env, start_new_session=True,
                )
                os.close(seal_write)
                seal_write = None
                try:
                    process.communicate(body, timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    if os.read(seal_read, 4096) != b"sealed-v1\n":
                        raise WorkerInfrastructureError("worker timed out before kernel sealing") from exc
                    raise InvalidResponse("candidate process exceeded its time limit") from exc
                seal = os.read(seal_read, 4096)
                if seal != b"sealed-v1\n":
                    diagnostics.seek(0)
                    private = diagnostics.read(8192).decode("utf-8", errors="replace")
                    raise WorkerInfrastructureError("worker failed before kernel sealing: " + private)
                if process.returncode != 0:
                    raise InvalidResponse("candidate worker failed after sealing")
                output.seek(0)
                raw = output.read(MAX_MESSAGE_BYTES + 1)
                if len(raw) > MAX_MESSAGE_BYTES:
                    raise InvalidResponse("candidate output exceeds bound")
                try:
                    result = json.loads(raw)
                except (ValueError, UnicodeError, RecursionError) as exc:
                    raise InvalidResponse("candidate response is not valid JSON") from exc
                if not isinstance(result, dict) or set(result) != {"tensor", "rng"}:
                    raise InvalidResponse("candidate cannot provide targets or scores")
                return result
            finally:
                os.close(seal_read)
                if seal_write is not None:
                    os.close(seal_write)
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
