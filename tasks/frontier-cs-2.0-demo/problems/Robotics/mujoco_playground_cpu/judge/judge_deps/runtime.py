"""Trusted source binding and bounded child-process lifecycle.

Trust model. Submitted Python runs in exactly one kind of process: a candidate
training child (``train_entry.py`` after its first stage marker). Everything that
produces a number that is scored -- environment construction, command
sampling, physics, reward terms, RNG and the rollout loop -- runs in a separate
``evaluate_policy.py`` process that imports the environment only from the
pristine tree and receives the candidate policy as a serialised ``jax.export``
(StableHLO) artifact, which is data, not Python.

A candidate training child may share the judge's UID, so this module does not
trust anything on disk after such a child has run:

* every descendant of a child is killed before its result is used (the runner
  is a child subreaper, so a daemonised descendant cannot escape the reap);
* the pristine tree and the trusted child scripts are re-hashed against the
  manifest loaded into this process's memory before every trusted launch, and
  the tree must contain no file outside the manifest (no planted ``.pyc``);
* accepted policy artifacts are pinned by SHA-256 in memory and re-checked
  before each evaluation;
* every child gets a fresh HOME/TMPDIR/cwd and runs ``python -I -B``;
* the runner refuses to start if the Python runtime itself is writable by the
  UID it runs as (the verifier image installs it root-owned and evaluates as an
  unprivileged account), unless ``guards.require_immutable_runtime`` is false.

Not defended: a kernel-level escape, or a same-UID process reading private
files such as ``workloads.json`` during its own training run (it has no channel
to exfiltrate them other than the policy it returns).
"""

import ctypes
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
PR_SET_DUMPABLE = 4
PR_SET_CHILD_SUBREAPER = 36


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def harden_process():
    """Become the reaper of all descendants and refuse same-UID ptrace."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        subreaper = libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0
        libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    except (OSError, AttributeError):
        subreaper = False
    return subreaper


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


def descendants(pid):
    """All live descendants of pid, found through /proc parent links."""
    parents = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as stream:
                fields = stream.read().rsplit(b") ", 1)[1].split()
            parents[int(entry)] = int(fields[1])
        except (OSError, IndexError, ValueError):
            continue
    found, frontier = [], [pid]
    while frontier:
        parent = frontier.pop()
        for child, owner in parents.items():
            if owner == parent and child not in found:
                found.append(child)
                frontier.append(child)
    return found


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
        self.active = None
        self.started = False
        self.on_scientific_start = on_scientific_start

    def stop(self):
        """Kill the active child, its process group and every stray descendant."""
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
        for _ in range(50):
            stray = descendants(os.getpid())
            if not stray:
                return
            for pid in stray:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            try:
                while os.waitpid(-1, os.WNOHANG)[0]:
                    pass
            except ChildProcessError:
                pass
            time.sleep(0.1)
        raise RuntimeError("could not reap every descendant of a workload child")

    def _require(self, seconds):
        """A phase starts only if the shared clock can hold its full limit."""
        if self.deadline - time.monotonic() < seconds:
            if not self.started:
                raise SetupDeadlineExpired("workload setup deadline exhausted")
            raise SuiteDeadlineExpired("workload deadline exhausted")

    def run(self, argv, *, cwd, env, log, limit, stages=()):
        """Run one scientific child under a sequence of separately limited stages.

        ``stages`` is a sequence of ``(marker, seconds, label)``. The child must
        write each one-byte ``marker`` in order to the inherited descriptor
        named by ``{ready_fd}`` in argv, each within ``seconds`` of the previous
        marker (the first: of process start). After the last marker the child
        has ``limit`` seconds to exit; that last stage is labelled ``"exit"``.
        Returns ``{label: seconds spent in that stage}``. A stage that overruns
        raises ``PhaseTimeout(label)``; a wrong marker, a closed descriptor or a
        non-zero exit raises ``PhaseFailed(label)``.
        """
        self._require(limit + sum(seconds for _, seconds, _ in stages))
        read_end = write_end = None
        options = {}
        if stages:
            read_end, write_end = os.pipe()
            argv = [part.format(ready_fd=write_end) if "{ready_fd}" in part else part
                    for part in argv]
            options["pass_fds"] = (write_end,)
        spent = {}
        with log.open("wb") as stream:
            try:
                self.active = subprocess.Popen(
                    argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True, **options)
            finally:
                if write_end is not None:
                    os.close(write_end)
            try:
                if not self.started:
                    if self.on_scientific_start is not None:
                        self.on_scientific_start()
                    self.started = True
                began = time.monotonic()
                for marker, seconds, label in stages:
                    ready, _, _ = select.select([read_end], [], [], seconds)
                    if not ready:
                        raise PhaseTimeout(label)
                    if os.read(read_end, 1) != marker:
                        raise PhaseFailed(label)
                    spent[label], began = time.monotonic() - began, time.monotonic()
                try:
                    code = self.active.wait(timeout=limit)
                except subprocess.TimeoutExpired as error:
                    raise PhaseTimeout("exit") from error
                spent["exit"] = time.monotonic() - began
                if code:
                    raise PhaseFailed("exit")
                return spent
            finally:
                if read_end is not None:
                    os.close(read_end)
                self.stop()
