"""Trusted source binding and bounded child-process lifecycle.

Trust model. Submitted Python runs in exactly one kind of process: a training
child (``train_entry.py`` after its first stage marker). Everything that
produces a number that is scored -- environment construction, command
sampling, physics, reward terms, RNG and the rollout loop -- runs in a separate
``evaluate_policy.py`` process that imports the environment only from the
pristine tree and receives the candidate policy as a serialised ``jax.export``
(StableHLO) artifact, which is data, not Python.

Every training child (candidate and reference alike) runs in the shared test
sandbox (``isolate.start``; the userns backend on the Verifier GPU runner):
its own copy of the tree as cwd, ``train_entry.py`` as a read-only input, the
judge's interpreter and the GPU, a fresh environment, and nothing of the judge
(the bundle, the private workloads, the pristine copy, the journal, other
policies). Every process it starts ends with it and its directories are
deleted; the policy artifact is its only declared output. The pristine tree is
admitted once against the manifest at suite start; trusted evaluation children
run as plain subprocesses.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
import isolate  # /opt/frontier, on the import path by frontier.pth

# Where a training child's markers go: stdout is the judge's pipe; train_entry.py moves the
# child's own stdout to stderr. After the stages, the child writes EXPORTED once the artifact
# is in place and then exits.
EXPORTED = b"E"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_manifest():
    document = json.loads((HERE / "source.json").read_text())
    if (set(document) != {"schema_version", "upstream", "tree_files", "trusted_children"}
            or document["schema_version"] != 1):
        raise ValueError("invalid trusted source manifest")
    return document


def tree_files(root):
    """Every regular file below root; anything else is rejected."""
    root = Path(root)
    found = {}
    for directory, names, files in os.walk(root, followlinks=False):
        for name in names + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("source tree may not contain symlinks")
            if name in files:
                if not path.is_file():
                    raise ValueError("source tree may contain only regular files")
                found[path.relative_to(root).as_posix()] = path
    return found


def admit_pristine(tree, manifest):
    """Admit only the exact pristine tree: same file set, same bytes."""
    expected = manifest["tree_files"]
    actual = tree_files(tree)
    if set(actual) != set(expected):
        raise ValueError("pristine tree file set changed")
    for name, path in actual.items():
        if sha256(path) != expected[name]:
            raise ValueError("pristine tree file changed: " + name)


def admit_trusted_children(manifest):
    for name, digest in manifest["trusted_children"].items():
        path = HERE / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError("trusted child script changed: " + name)


class SetupDeadlineExpired(TimeoutError):
    """The trusted runner deadline expired before any scientific child ran."""


class SuiteDeadlineExpired(TimeoutError):
    """The shared scientific clock expired after scientific execution began."""


class PhaseTimeout(Exception):
    """One child overran the named stage's limit (not the shared suite clock)."""


class PhaseFailed(Exception):
    """One child failed during the named stage."""


class ChildRunner:
    """One total deadline, including a cleanup reserve, for the complete suite."""

    def __init__(self, seconds, *, envelope_deadline=None, on_scientific_start=None):
        if seconds <= 20:
            raise ValueError("insufficient workload deadline")
        self.deadline = time.monotonic() + seconds - 10
        if envelope_deadline is not None:
            if not math.isfinite(envelope_deadline):
                raise ValueError("invalid workload envelope deadline")
            # Preserve a 90 + 10 second scientific cleanup reserve.
            self.deadline = min(self.deadline, envelope_deadline - 100)
            if self.deadline <= time.monotonic():
                raise SetupDeadlineExpired("workload setup deadline exhausted")
        self.active = None      # a trusted subprocess.Popen
        self.instance = None    # a sandboxed training child (isolate.Instance)
        self.started = False
        self.on_scientific_start = on_scientific_start

    def stop(self):
        """End the active child: a trusted child's process group, or the sandbox instance."""
        if self.instance is not None:
            instance, self.instance = self.instance, None
            instance.close()
        if self.active is not None:
            try:
                os.killpg(self.active.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.active.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            self.active = None

    def _require(self, seconds):
        """A phase starts only if the shared clock can hold its full limit."""
        if self.deadline - time.monotonic() < seconds:
            if not self.started:
                raise SetupDeadlineExpired("workload setup deadline exhausted")
            raise SuiteDeadlineExpired("workload deadline exhausted")

    def _scientific_start(self):
        if not self.started:
            if self.on_scientific_start is not None:
                self.on_scientific_start()
            self.started = True

    def run(self, argv, *, cwd, env, log, limit):
        """Run one trusted child (no submitted code) to exit within ``limit`` seconds.
        Returns ``{"exit": seconds}``; an overrun raises ``PhaseTimeout("exit")``, a non-zero
        exit ``PhaseFailed("exit")``."""
        self._require(limit)
        with log.open("wb") as stream:
            self.active = subprocess.Popen(
                argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            try:
                self._scientific_start()
                began = time.monotonic()
                try:
                    code = self.active.wait(timeout=limit)
                except subprocess.TimeoutExpired as error:
                    raise PhaseTimeout("exit") from error
                if code:
                    raise PhaseFailed("exit")
                return {"exit": time.monotonic() - began}
            finally:
                self.stop()

    def run_sandboxed(self, argv, *, limit, stages, **sandbox):
        """Run one training child in a fresh isolate instance under a sequence of
        separately limited stages; returns ``(spent, result)``.

        ``stages`` is a sequence of ``(marker, seconds, label)``. The child must write each
        one-byte ``marker`` in order to its stdout (the judge's pipe), each within ``seconds``
        of the previous marker (the first: of the instance's start). After the last marker the
        child has ``limit`` seconds to write ``EXPORTED`` and exit; that last stage is labelled
        ``"exit"``. ``spent`` is ``{label: seconds spent in that stage}``; ``result`` is the
        instance's isolate.Result (stderr, declared outputs). A stage that overruns raises
        ``PhaseTimeout(label)``; a wrong marker, an early end of the pipe, a non-zero exit or a
        sandbox limit raises ``PhaseFailed(label)``. Either carries the Result as ``.result``.
        ``sandbox`` are isolate.start's keyword arguments (tree, inputs, outputs, env, ...).
        """
        whole = limit + sum(seconds for _, seconds, _ in stages)
        self._require(whole)
        self.instance = isolate.start(argv, timeout_seconds=whole + 60, **sandbox)
        pipe = self.instance.stdout.fileno()
        spent, label = {}, "exit"
        try:
            self._scientific_start()
            began = time.monotonic()
            for marker, seconds, label in (*stages, (EXPORTED, limit, "exit")):
                ready, _, _ = select.select([pipe], [], [], seconds)
                if not ready:
                    raise PhaseTimeout(label)
                if os.read(pipe, 1) != marker:
                    raise PhaseFailed(label)
                if label != "exit":
                    spent[label], began = time.monotonic() - began, time.monotonic()
            # The pipe ends when the child exits (nothing else holds its write end).
            ready, _, _ = select.select([pipe], [], [], max(0.0, limit - (time.monotonic() - began)))
            if not ready:
                raise PhaseTimeout("exit")
            if os.read(pipe, 1) != b"":
                raise PhaseFailed("exit")
            spent["exit"] = time.monotonic() - began
        except (PhaseTimeout, PhaseFailed) as error:
            instance, self.instance = self.instance, None
            error.result = instance.close()
            raise
        except BaseException:
            self.stop()
            raise
        instance, self.instance = self.instance, None
        result = instance.close()
        # close() may find the child between closing its pipe and exiting: SIGKILL then.
        if result.limit is not None or result.returncode not in (0, -signal.SIGKILL):
            error = PhaseFailed("exit")
            error.result = result
            raise error
        return spent, result
