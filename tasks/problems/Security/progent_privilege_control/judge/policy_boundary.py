"""Trusted side of the policy worker: one worker per episode, under the shared isolate launcher."""

from pathlib import Path
import os
import selectors
import sys
import time

import isolate  # the shared launcher, beside the verifier in the Verifier base

from policy_protocol import MAX_RESPONSE, encode, parse_worker_message

BRIDGE = Path(__file__).resolve().parent
WORKER_CPU_SECONDS = 20    # the statement's 20 CPU seconds; model waits do not count
WORKER_OPEN_FILES = 64
WORKER_LIFETIME_SECONDS = 24 * 3600  # a backstop: the episode's own deadlines end it far earlier


class BoundaryFailure(RuntimeError):
    """The worker broke the protocol, crashed or timed out: a candidate-side failure."""


class PolicyWorker:
    """Run ``policy_worker.py`` on a candidate tree in an isolate instance and exchange
    framed JSON over the instance's stdin and stdout.

    The instance (tasks/shared/lib/isolate.py) reads only the system directories, the
    judge's interpreter, its own copy of the tree and the two bridge modules; it gets an
    environment built from nothing, no network, one process and thread (syscalls="strict"), no
    access to the judge process, and private directories deleted with it, so nothing carries
    over between episodes. ``call(op, payload, model)`` sends one operation and services the
    worker's model requests through ``model(system, user, temperature) -> str`` until the
    worker reports completion: the model key and gateway stay in this process. ``model`` raises
    ``ModelRefused`` to return an error string to the worker instead of content (cap reached);
    any other exception propagates. A launcher that cannot build the boundary raises
    ``isolate.IsolateError`` (an evaluator error, never a candidate failure).
    """

    def __init__(self, tree, *, op_timeout=900.0, idle_timeout=120.0):
        self.op_timeout = op_timeout
        self.idle_timeout = idle_timeout
        self.buffer = bytearray()
        self.closed = False
        self.instance = isolate.start(
            [sys.executable, "-I", "-B", "{in}/policy_worker.py"],
            tree=Path(tree).resolve(),
            inputs={name: BRIDGE / name for name in ("policy_worker.py", "policy_protocol.py")},
            timeout_seconds=WORKER_LIFETIME_SECONDS, cpu_seconds=WORKER_CPU_SECONDS,
            open_files=WORKER_OPEN_FILES, syscalls="strict")
        self.stdin, self.stdout = self.instance.stdin, self.instance.stdout
        os.set_blocking(self.stdout.fileno(), False)
        os.set_blocking(self.stdin.fileno(), False)

    def _write(self, data, deadline):
        with selectors.DefaultSelector() as selector:
            selector.register(self.stdin, selectors.EVENT_WRITE)
            while data:
                remaining = min(deadline - time.monotonic(), self.idle_timeout)
                if remaining <= 0 or not selector.select(remaining):
                    raise BoundaryFailure("policy worker stopped reading")
                try:
                    count = os.write(self.stdin.fileno(), data)
                except BlockingIOError:
                    continue
                except OSError as error:
                    raise BoundaryFailure("policy worker closed input") from error
                data = data[count:]

    def _line(self, deadline):
        with selectors.DefaultSelector() as selector:
            selector.register(self.stdout, selectors.EVENT_READ)
            while b"\n" not in self.buffer:
                remaining = min(deadline - time.monotonic(), self.idle_timeout)
                if remaining <= 0 or not selector.select(remaining):
                    raise BoundaryFailure("policy worker timed out")
                try:
                    chunk = os.read(self.stdout.fileno(), 65536)
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
        self.instance.close()


class ModelRefused(RuntimeError):
    """Raised by the model callback to deny one policy-model request (cap reached)."""
