"""Bounded sampler subprocess and value RPC within the standard Verifier sandbox."""

from __future__ import annotations

import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

from bridge_protocol import (
    MAX_FRAME_BYTES, ProtocolError, decode, encode, exact_fields,
    model_arguments, normal_arguments, public_configuration, sample_arguments, tensor, validate_result,
)


def _process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[19], fields[0]
    except (OSError, IndexError):
        return None


def _owned_descendants(pid):
    result = {}
    pending = [pid]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        try:
            children = [int(p) for p in Path(f"/proc/{current}/task/{current}/children").read_text().split()]
        except OSError:
            continue
        for child in children:
            identity = _process_identity(child)
            if identity is not None:
                result[child] = identity[0]
                pending.append(child)
    return result


def worker_command(app: Path, worker: Path, protocol: Path, *, runtime=None):
    """Launch in the standard Verifier sandbox, without a nested OS sandbox."""
    for path in (app, worker, protocol):
        if path.is_symlink() or not path.exists():
            raise ProtocolError("Worker inputs must be ordinary existing paths.")
    if not app.is_dir() or not worker.is_file() or not protocol.is_file():
        raise ProtocolError("Invalid worker input kind.")
    import sys
    python = sys.executable
    if runtime is not None:
        exact_fields(runtime, {"prefix", "base", "version"})
        if runtime["version"] != "3.10":
            raise ProtocolError("The selected task requires Python 3.10.")
        prefix, base = Path(runtime["prefix"]).resolve(), Path(runtime["base"]).resolve()
        if (not (prefix / "lib/python3.10/site-packages").is_dir()
                or not (prefix / "bin/python").is_file()
                or not (base / "lib/python3.10").is_dir()):
            raise ProtocolError("The selected Python runtime is incomplete.")
        python = str(prefix / "bin/python")
    return [python, "-s", "-B", "-u", str(worker.resolve())]


def worker_environment(app: Path, protocol: Path):
    # Pass only the public import paths and numeric runtime settings. Platform
    # authentication and author/provider variables are never copied to the worker.
    return {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "TMPDIR": "/tmp",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join((str(app.resolve()), str(protocol.resolve().parent))),
            "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


class SamplerBridge:
    """One protocol for both candidate and exactly restored reference source.

    The trusted caller owns model/normal functions. The worker receives only
    serialized values, never these functions or their captured state.
    """

    def __init__(self, app, worker, *, lifetime_seconds, max_model_calls,
                 max_random_calls, model_forward, random_normal, runtime=None):
        if lifetime_seconds <= 0 or max_model_calls <= 0 or max_random_calls <= 0:
            raise ValueError("Explicit positive protocol limits are required.")
        self.command = worker_command(Path(app), Path(worker), Path(__file__).with_name("bridge_protocol.py"), runtime=runtime)
        self.environment = worker_environment(Path(app), Path(__file__).with_name("bridge_protocol.py"))
        self.cwd = str(Path(app).resolve())
        self.deadline = time.monotonic() + lifetime_seconds
        self.max_model_calls = max_model_calls
        self.max_random_calls = max_random_calls
        self.model_forward = model_forward
        self.random_normal = random_normal
        self.model_calls = self.random_calls = 0
        self.request_id = 0
        self.sample_id = 0
        self.buffer = b""
        self.process = None
        self.started_at = self.finished_at = None

    def __enter__(self):
        self.started_at = time.monotonic()
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, start_new_session=True,
                                        close_fds=True, env=self.environment, cwd=self.cwd)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        return self

    def _wait(self, fd, event):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Sampler exceeded its lifetime.")
        with selectors.DefaultSelector() as selector:
            selector.register(fd, event)
            if not selector.select(remaining):
                raise TimeoutError("Sampler exceeded its lifetime.")

    def send(self, value):
        data = memoryview(encode(value))
        fd = self.process.stdin.fileno()
        while data:
            self._wait(fd, selectors.EVENT_WRITE)
            try:
                written = os.write(fd, data)
            except BlockingIOError:
                continue
            except BrokenPipeError as exc:
                raise ProtocolError("Sampler worker closed its input.") from exc
            data = data[written:]

    def receive(self):
        fd = self.process.stdout.fileno()
        while b"\n" not in self.buffer:
            self._wait(fd, selectors.EVENT_READ)
            try:
                chunk = os.read(fd, min(65536, MAX_FRAME_BYTES + 1 - len(self.buffer)))
            except BlockingIOError:
                continue
            if not chunk:
                raise ProtocolError("Sampler worker exited without a complete frame.")
            self.buffer += chunk
            if len(self.buffer) > MAX_FRAME_BYTES:
                raise ProtocolError("Sampler worker exceeded its frame bound.")
        raw, self.buffer = self.buffer.split(b"\n", 1)
        return decode(raw + b"\n")

    def configure(self, public_config):
        public_config = public_configuration(public_config)
        self.send({"type": "configure", "version": 1, "config": public_config})
        response = self.receive()
        if response != {"type": "ready", "version": 1} or type(response["version"]) is not int:
            raise ProtocolError("Invalid worker handshake.")

    def sample(self, arguments):
        arguments = sample_arguments(arguments)
        self.sample_id += 1
        before = self.model_calls
        self.send({"type": "sample", "id": self.sample_id, "arguments": arguments})
        while True:
            message = self.receive()
            if type(message) is not dict:
                raise ProtocolError("Expected a protocol object.")
            if message.get("type") == "sample_result":
                exact_fields(message, {"type", "id", "tensor"})
                if type(message["id"]) is not int or message["id"] != self.sample_id:
                    raise ProtocolError("Sample identity mismatch.")
                if self.model_calls == before:
                    raise ProtocolError("Sampler did not invoke the trusted frozen model.")
                return validate_result(message["tensor"], arguments)
            exact_fields(message, {"type", "id", "op", "arguments"})
            self.request_id += 1
            if (message["type"] != "request" or type(message["id"]) is not int
                    or message["id"] != self.request_id):
                raise ProtocolError("RPC identity mismatch.")
            if message["op"] == "model_forward":
                if self.model_calls >= self.max_model_calls:
                    raise ProtocolError("Model call bound exceeded.")
                args = model_arguments(message["arguments"])
                result = tensor(self.model_forward(args))
                # The counter advances only after the trusted callback returns.
                self.model_calls += 1
            elif message["op"] == "random_normal":
                if self.random_calls >= self.max_random_calls:
                    raise ProtocolError("Random call bound exceeded.")
                args = normal_arguments(message["arguments"])
                result = tensor(self.random_normal(args))
                if result["shape"] != args["shape"] or result["dtype"] != args["dtype"]:
                    raise ProtocolError("Trusted normal service returned invalid metadata.")
                self.random_calls += 1
            else:
                raise ProtocolError("Operation is not part of the immutable tensor interface.")
            self.send({"type": "response", "id": self.request_id, "tensor": result})

    def close(self):
        if self.process is not None:
            # Reap this process group and observed descendants, including children
            # with a separate session. The platform owns the outer job lifecycle.
            descendants = _owned_descendants(self.process.pid)
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            for pid, birth in descendants.items():
                try:
                    fd = os.pidfd_open(pid)
                except ProcessLookupError:
                    continue
                try:
                    identity = _process_identity(pid)
                    if identity is not None and identity[0] == birth:
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                finally:
                    os.close(fd)
            self.process.wait(timeout=5)
            self.process.stdin.close()
            self.process.stdout.close()
            cleanup_deadline = time.monotonic() + 5
            while descendants:
                descendants = {
                    pid: birth for pid, birth in descendants.items()
                    if (identity := _process_identity(pid)) is not None
                    and identity[0] == birth and identity[1] != "Z"
                }
                if descendants and time.monotonic() >= cleanup_deadline:
                    raise RuntimeError("Owned sampler descendants have not terminated.")
                if descendants:
                    time.sleep(0.01)
            self.finished_at = time.monotonic()

    def __exit__(self, *_):
        self.close()
