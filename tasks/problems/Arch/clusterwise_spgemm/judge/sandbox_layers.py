"""The isolate layers this judge adds to the shared launcher."""

import dataclasses
import errno
import os
from pathlib import Path
import resource
import stat

import isolate

SCHED_SETAFFINITY = 203   # x86_64


class Pinned(isolate.Layer):
    """The command runs on these CPUs (a subset of the judge's affinity) and cannot leave them."""

    def __init__(self, cpus):
        if (isinstance(cpus, (str, bytes)) or not cpus or not all(type(c) is int for c in cpus)
                or not set(cpus) <= os.sched_getaffinity(0)):
            raise isolate.IsolateError("cpus must be a non-empty subset of the judge's "
                                                "CPU affinity")
        self.cpus = sorted(set(cpus))

    def prepare(self, rules):
        rules.deny_syscall(SCHED_SETAFFINITY)

    def preexec(self):
        os.sched_setaffinity(0, self.cpus)


class AddressSpace(isolate.Layer):
    """RLIMIT_AS for every process of the command (per process, unlike memory_bytes)."""

    def __init__(self, limit: int):
        if type(limit) is not int or limit <= 0:
            raise isolate.IsolateError("address space limit must be a positive integer")
        self.limit = limit

    def preexec(self):
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = self.limit if hard == resource.RLIM_INFINITY else min(self.limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


class OutputsDir(isolate.Layer):
    """Outputs too large to hold in memory: each named file the command leaves in out/ is moved
    into `directory` (together at most `max_bytes`); Result.outputs maps its name to that Path,
    or None when it is absent or not a single-link regular file."""

    def __init__(self, directory, names, max_bytes: int):
        directory = os.fspath(directory)
        if not os.path.isabs(directory) or not os.path.isdir(directory):
            raise isolate.IsolateError("outputs dir must be an existing absolute "
                                                "directory")
        for name in names:
            if not isinstance(name, str) or not name or "/" in name or name in (".", ".."):
                raise isolate.IsolateError(f"invalid output name {name!r}")
        self.directory, self.names, self.max_bytes = directory, list(names), max_bytes

    def result(self, result, run):
        moved, budget = {}, self.max_bytes
        for name in self.names:
            target = _move_bounded(os.path.join(run.paths["out"], name),
                                   os.path.join(self.directory, name), budget)
            moved[name] = target
            budget -= os.stat(target).st_size if target else 0
        return dataclasses.replace(result, outputs={**result.outputs, **moved})


def _move_bounded(path: str, target: str, cap: int):
    """Move the instance's regular file `path` to the judge's `target` (replacing it), keeping
    at most `cap` bytes. Every process of the instance has ended, so it cannot change."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            return None
        try:
            os.replace(path, target)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            # Another filesystem: a bounded copy from the checked descriptor.
            sink = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            try:
                left = max(cap, 0)
                while left > 0:
                    sent = os.sendfile(sink, descriptor, None, min(left, 1 << 30))
                    if not sent:
                        break
                    left -= sent
            finally:
                os.close(sink)
        if os.stat(target).st_size > cap:
            os.truncate(target, max(cap, 0))
        return Path(target)
    finally:
        os.close(descriptor)
