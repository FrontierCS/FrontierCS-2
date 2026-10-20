"""Trusted lifetime/transport for a confined queue worker."""

from pathlib import Path
import math
import os
import select
import signal
import subprocess
import tempfile
import threading
import time

import queue_wire as wire


class QueueProcess:
    def __init__(self, source, scratch_root, *, python, deadline, seconds=15, allow_cuda=False, namespace=False):
        if not math.isfinite(deadline) or deadline <= time.monotonic():
            raise ValueError("Queue lifetime must have a finite future deadline")
        self.deadline, self.seconds = deadline, seconds
        self.seq, self.closed = 0, False
        self.scratch = tempfile.TemporaryDirectory(prefix="fd-queue-", dir=scratch_root)
        if os.geteuid() == 0:
            os.chown(self.scratch.name, 65534, 65534)
        actor = Path(__file__).with_name("queue_actor.py")
        self.process = None
        self.stderr_prefix = bytearray()
        self.stderr_reader = None
        try:
            if allow_cuda or namespace:
                from queue_namespace import command
                argv = command(Path(source).resolve(), actor.parent, python, cuda=allow_cuda)
            else:
                argv = [str(python), "-I", "-B", str(actor), str(Path(source).resolve()), self.scratch.name, "cpu"]
            argv.append(str(deadline))
            self.process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                close_fds=True, start_new_session=True,
                env={"PATH": "/usr/bin:/bin", "HOME": self.scratch.name, "TMPDIR": self.scratch.name,
                     "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                     "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "0" if allow_cuda else "",
                     "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
                     "CUDA_CACHE_PATH": self.scratch.name})
            self.stderr_reader = threading.Thread(target=self._drain_stderr, daemon=True)
            self.stderr_reader.start()
            if wire.receive(self.process.stdout.fileno(), self._deadline()) != (0, wire.READY, []):
                raise ValueError("Queue worker did not establish readiness")
        except BaseException as error:
            self.close()
            error.worker_stderr = self.stderr_prefix.decode(errors="replace")
            raise

    def _drain_stderr(self):
        # Drain continuously, retaining only a bounded author diagnostic. These
        # untrusted bytes are never used as tensor replies, validity or scores.
        while chunk := self.process.stderr.read(4096):
            self.stderr_prefix.extend(chunk[:max(0, 8192 - len(self.stderr_prefix))])

    def _deadline(self):
        return min(self.deadline, time.monotonic() + self.seconds)

    def call(self, op, tensors=()):
        if self.closed or self.seq >= 16384:
            self.close()
            raise RuntimeError("Queue worker closed or request budget exhausted")
        self.seq += 1
        deadline = self._deadline()
        try:
            wire.send(self.process.stdin.fileno(), self.seq, op, tensors, deadline)
            seq, result_op, result = wire.receive(self.process.stdout.fileno(), deadline)
            if (seq, result_op) != (self.seq, op):
                raise ValueError("Unexpected operation or stale queue reply")
            if select.select([self.process.stdout], [], [], 0)[0]:
                raise ValueError("Unsolicited bytes or exit after queue reply")
            return seq, result
        except BaseException as error:
            self.close()
            error.worker_stderr = self.stderr_prefix.decode(errors="replace")
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=5)
            self.process.stdin.close()
            self.process.stdout.close()
            if self.stderr_reader is not None:
                self.stderr_reader.join(timeout=1)
            self.process.stderr.close()
        self.scratch.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
