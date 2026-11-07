"""Trusted lifetime/transport for a queue worker under the shared isolate launcher.

Each queue runs queue_actor.py, which imports the submitted FeatureQueue, as a long-lived
worker started with the shared launcher (/opt/frontier/isolate.py, `start`; on the GPU
runner its userns backend): fresh user, mount, network, PID and IPC namespaces, no judge files,
an environment built from nothing, no new processes (syscalls="no_fork"), private directories
deleted when it ends. Its only inputs are the queue-only source snapshot and the actor files;
tensors cross its stdin/stdout by value (queue_wire). A worker that does not start is an
infrastructure fault, never a candidate failure: a sandbox that cannot be built raises
IsolateError, and a worker that does not reach READY raises QueueWorkerStartError. Until READY the
worker runs only trusted code (the actor, torch and the CUDA context), so its startup has its own
bound, STARTUP_SECONDS, not the submission's per-call budget.
"""

from pathlib import Path
import math
import select
import time

import queue_wire as wire

import isolate  # /opt/frontier, on the import path by frontier.pth

HERE = Path(__file__).resolve().parent
ACTOR_FILES = ("queue_actor.py", "queue_wire.py")
STDERR_PREFIX_BYTES = 8192
# Bounds of one queue worker. Its stderr is the author's diagnostic only; the cap keeps a
# chatty queue from ending the workload (the instance stops at output_bytes).
WORKER_OUTPUT_BYTES = 256 << 20
WORKER_PATH = "/usr/bin:/bin"
# Seconds the instance outlives the queue deadline, so the actor reports its own deadline first.
LIFETIME_MARGIN_SECONDS = 30
# Bound of a worker's startup (isolate launch, interpreter, import torch, CUDA context) until
# READY; it runs no submitted code. Round 5's 15 s (the per-call budget) expired on a loaded runner.
STARTUP_SECONDS = 180


class QueueWorkerStartError(RuntimeError):
    """A queue worker did not reach READY: a judge/infrastructure fault, never a candidate failure."""


def worker_options(allow_cuda):
    """isolate.start keywords of one queue worker (tests swap them for the local backend)."""
    env = {"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "0" if allow_cuda else "",
           "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
           "CUDA_CACHE_PATH": "{tmp}"}
    return {"devices": ["nvidia"] if allow_cuda else [], "env": env, "path": WORKER_PATH,
            "syscalls": "no_fork", "output_bytes": WORKER_OUTPUT_BYTES}


class QueueProcess:
    def __init__(self, source, *, python, deadline, seconds=15, allow_cuda=False):
        if not math.isfinite(deadline) or deadline <= time.monotonic():
            raise ValueError("Queue lifetime must have a finite future deadline")
        self.deadline, self.seconds = deadline, seconds
        self.seq, self.closed = 0, False
        self.result, self.stderr_prefix = None, b""
        self.instance = None
        inputs = {"solution": Path(source).resolve(strict=True)}
        inputs.update((name, HERE / name) for name in ACTOR_FILES)
        argv = [str(python), "-I", "-B", "{in}/queue_actor.py", "{in}/solution", "{work}",
                "cuda" if allow_cuda else "cpu", str(deadline)]
        try:
            self.instance = isolate.start(
                argv, timeout_seconds=deadline - time.monotonic() + LIFETIME_MARGIN_SECONDS,
                inputs=inputs, **worker_options(allow_cuda))
            ready = wire.receive(self.instance.stdout.fileno(),
                                 min(deadline, time.monotonic() + STARTUP_SECONDS))
            if ready != (0, wire.READY, []):
                raise ValueError("Queue worker did not establish readiness")
        except isolate.IsolateError:
            self.close()
            raise
        except Exception as error:
            self.close()
            fault = QueueWorkerStartError(f"queue worker did not start: {type(error).__name__}: {error}")
            fault.worker_stderr = self.stderr_prefix.decode(errors="replace")
            raise fault from error
        except BaseException:
            self.close()
            raise

    def _deadline(self):
        return min(self.deadline, time.monotonic() + self.seconds)

    @property
    def returncode(self):
        return None if self.result is None else self.result.returncode

    def call(self, op, tensors=()):
        if self.closed or self.seq >= 16384:
            self.close()
            raise RuntimeError("Queue worker closed or request budget exhausted")
        self.seq += 1
        deadline = self._deadline()
        try:
            wire.send(self.instance.stdin.fileno(), self.seq, op, tensors, deadline)
            seq, result_op, result = wire.receive(self.instance.stdout.fileno(), deadline)
            if (seq, result_op) != (self.seq, op):
                raise ValueError("Unexpected operation or stale queue reply")
            if select.select([self.instance.stdout], [], [], 0)[0]:
                raise ValueError("Unsolicited bytes or exit after queue reply")
            return seq, result
        except BaseException as error:
            self.close()
            error.worker_stderr = self.stderr_prefix.decode(errors="replace")
            raise

    def close(self):
        """End the worker (every process it started) and keep a bounded stderr prefix. These
        untrusted bytes are never used as tensor replies, validity or scores."""
        if self.closed:
            return
        self.closed = True
        if self.instance is not None:
            self.result = self.instance.close()
            self.stderr_prefix = self.result.stderr[:STDERR_PREFIX_BYTES]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
