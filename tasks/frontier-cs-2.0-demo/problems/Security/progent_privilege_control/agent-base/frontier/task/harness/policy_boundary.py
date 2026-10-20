"""Trusted side of the sandboxed policy worker (one worker per episode)."""

import ctypes
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from policy_protocol import MAX_RESPONSE, encode, parse_worker_message


class BoundaryFailure(RuntimeError):
    """The worker broke the protocol, crashed or timed out: a candidate-side failure."""


def protect_parent():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0 or libc.prctl(3, 0, 0, 0, 0) != 0:
        raise BoundaryFailure("cannot protect trusted parent memory")


class PolicyWorker:
    """Launch ``sealed_bootstrap.py`` on a candidate tree and exchange framed JSON.

    ``call(op, payload, model)`` sends one operation and services the worker's model
    requests through ``model(system, user, temperature) -> str`` until the worker
    reports completion. ``model`` raises ``ModelRefused`` to return an error string to
    the worker instead of content (cap reached); any other exception propagates.
    """

    def __init__(self, tree, *, op_timeout=900.0, idle_timeout=120.0):
        protect_parent()
        # PROGENT_UNSANDBOXED=1 is a local-development switch of the Agent-side copy;
        # the verifier image never sets it.
        bootstrap = "plain_bootstrap.py" if os.environ.get("PROGENT_UNSANDBOXED") == "1" else "sealed_bootstrap.py"
        self.op_timeout = op_timeout
        self.idle_timeout = idle_timeout
        self.buffer = bytearray()
        self.closed = False
        self.process = None
        bridge = Path(__file__).resolve().parent
        command = [
            str(Path(sys.executable).absolute()), "-I", "-B",
            str(bridge / bootstrap), str(Path(tree).resolve()),
            str(bridge / "policy_worker.py"), str(bridge / "policy_protocol.py"), str(os.getpid()),
        ]
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "PYTHONHASHSEED": "0"},
                close_fds=True, start_new_session=True, bufsize=0,
            )
            os.set_blocking(self.process.stdout.fileno(), False)
            os.set_blocking(self.process.stdin.fileno(), False)
        except (OSError, ValueError) as error:
            self.abort()
            raise RuntimeError("mandatory policy isolation unavailable") from error

    def _write(self, data, deadline):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdin, selectors.EVENT_WRITE)
            while data:
                remaining = min(deadline - time.monotonic(), self.idle_timeout)
                if remaining <= 0 or not selector.select(remaining):
                    raise BoundaryFailure("policy worker stopped reading")
                try:
                    count = os.write(self.process.stdin.fileno(), data)
                except BlockingIOError:
                    continue
                except OSError as error:
                    raise BoundaryFailure("policy worker closed input") from error
                data = data[count:]

    def _line(self, deadline):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self.buffer:
                remaining = min(deadline - time.monotonic(), self.idle_timeout)
                if remaining <= 0 or not selector.select(remaining):
                    raise BoundaryFailure("policy worker timed out")
                try:
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise BoundaryFailure("policy worker exited")
                self.buffer.extend(chunk)
                if len(self.buffer) > MAX_RESPONSE + 1:
                    raise BoundaryFailure("policy worker message exceeds bound")
        frame, rest = bytes(self.buffer).split(b"\n", 1)
        self.buffer[:] = rest
        return frame + b"\n"

    def call(self, op, payload, model):
        if self.closed:
            raise BoundaryFailure("policy worker is closed")
        deadline = time.monotonic() + self.op_timeout
        try:
            self._write(encode({"op": op, "payload": payload}), deadline)
            while True:
                # A model round-trip is the parent's time, not the worker's: the
                # idle timer restarts after each reply.
                raw = self._line(deadline)
                try:
                    message = parse_worker_message(raw, op)
                except (ValueError, UnicodeError) as error:
                    raise BoundaryFailure("invalid policy worker message") from error
                if message["kind"] == "done":
                    if self.buffer:
                        raise BoundaryFailure("unsolicited policy worker output")
                    return message
                try:
                    reply = {"op": "llm_result", "content": model(message["system"], message["user"],
                                                                  message["temperature"])}
                except ModelRefused as refused:
                    reply = {"op": "llm_result", "error": str(refused)}
                self._write(encode(reply), deadline)
        except BaseException:
            self.abort()
            raise

    def abort(self):
        if self.closed:
            return
        self.closed = True
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            for stream in (self.process.stdin, self.process.stdout):
                if stream and not stream.closed:
                    stream.close()


class ModelRefused(RuntimeError):
    """Raised by the model callback to deny one policy-model request (cap reached)."""
