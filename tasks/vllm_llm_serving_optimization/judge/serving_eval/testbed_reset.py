"""Verified reset of one prepared SWE testbed between correctness runs.

A check worker is one container per original instance, reused for the three
serving variants in the fixed order vanilla, reference, candidate. Between
runs it must look like the image again: tracked content is restored by Git and
re-hashed by ``inspect_testbed`` before the next patch; ignored files are
compared with the inventory taken before the first run, and any new or changed
one (rebuilt extensions, caches, bytecode) is deleted so the next run
regenerates it. The receipt of every reset travels inside the signed envelope.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import time

MAX_INVENTORY_FILES = 500_000
MAX_INVENTORY_BYTES = 8 * 1024 ** 3
GIT_TIMEOUT_SECONDS = 1200


class ResetError(RuntimeError):
    pass


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ResetError('testbed reset exhausted the original request deadline')
    return remaining


def _digest(path, deadline=None):
    with path.open('rb') as handle:
        if deadline is None:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
        digest = hashlib.sha256()
        while True:
            _remaining(deadline)
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


class TestbedReset:
    def __init__(self, testbed, *, run=None, output_limit=None):
        self.testbed = Path(testbed)
        self.run = subprocess.run if run is None else run
        if output_limit is not None and (type(output_limit) is not int or output_limit <= 0):
            raise ValueError('positive publisher reset output bound required')
        self.output_limit = output_limit
        self.inventory = None
        self.head = None

    def _git(self, *args):
        deadline = getattr(self, '_deadline', None)
        timeout = GIT_TIMEOUT_SECONDS if deadline is None else min(GIT_TIMEOUT_SECONDS, _remaining(deadline))
        argv = ['git', '-c', 'core.fileMode=false', '-c', 'safe.directory=' + str(self.testbed), *args]
        if self.run is subprocess.run and self.output_limit is not None:
            from .swe_test_worker import _run
            result = _run(argv, cwd=self.testbed, timeout=timeout, output_limit=self.output_limit)
            if result.exit_code or result.stop_reason:
                raise ResetError('bounded git execution failed during testbed reset')
            output = result.output
        else:
            result = self.run(argv, cwd=self.testbed, capture_output=True, timeout=timeout)
            if result.returncode != 0:
                raise ResetError('git ' + args[0] + ' failed during testbed reset')
            output = result.stdout
        if deadline is not None:
            _remaining(deadline)
        return output

    def _ignored(self):
        """Every ignored regular file (symlinks are not hashed, only named)."""
        raw = self._git('ls-files', '-z', '--others', '--ignored', '--exclude-standard')
        files = {}
        total = 0
        for entry in raw.split(b'\0'):
            deadline = getattr(self, '_deadline', None)
            if deadline is not None:
                _remaining(deadline)
            if not entry:
                continue
            name = os.fsdecode(entry)
            path = self.testbed / name
            if path.is_symlink():
                files[name] = ('symlink', os.readlink(path))
                continue
            if not path.is_file():
                continue
            size = path.stat().st_size
            total += size
            if len(files) >= MAX_INVENTORY_FILES or total > MAX_INVENTORY_BYTES:
                raise ResetError('ignored-file inventory exceeds the reset bound')
            files[name] = (size, _digest(path, deadline))
        return files

    def _require_clean(self):
        if self._git('status', '--porcelain', '--untracked-files=all'):
            raise ResetError('testbed worktree is not clean')

    def snapshot(self):
        """Record the image state before the first run."""
        if self.inventory is not None:
            raise ResetError('testbed snapshot already taken')
        self._require_clean()
        self.head = self._git('rev-parse', 'HEAD').decode().strip()
        self.inventory = self._ignored()
        return {'head': self.head, 'ignored_files': len(self.inventory)}

    def reset(self, *, deadline=None):
        """Restore the snapshot state; fail closed when Git cannot."""
        self._deadline = deadline
        try:
            return self._reset()
        finally:
            self._deadline = None

    def _reset(self):
        if self.inventory is None:
            raise ResetError('testbed reset requires a prior snapshot')
        started = time.monotonic()
        lock = self.testbed / '.git/index.lock'
        if lock.exists():
            lock.unlink()
        self._git('reset', '-q', '--hard', 'HEAD')
        self._git('clean', '-q', '-ffd')
        if self._git('rev-parse', 'HEAD').decode().strip() != self.head:
            raise ResetError('testbed HEAD changed between runs')
        self._require_clean()
        current = self._ignored()
        removed = []
        for name, value in current.items():
            if self._deadline is not None:
                _remaining(self._deadline)
            if self.inventory.get(name) != value:
                path = self.testbed / name
                if path.is_symlink() or path.is_file():
                    path.unlink()
                removed.append(name)
        missing = sorted(set(self.inventory) - set(current))
        # Deleted inventory files cannot be restored; the run itself removed
        # them, so a later run would also not need them. Record, do not hide.
        receipt = {'head': self.head, 'removed_changed_or_new_ignored': len(removed),
                   'removed_examples': sorted(removed)[:32], 'missing_inventory_files': len(missing),
                   'missing_examples': missing[:32], 'verified_ignored_files': len(current) - len(removed),
                   'elapsed_seconds': time.monotonic() - started}
        if (self.testbed / '.git/index.lock').exists():
            raise ResetError('testbed index lock reappeared after reset')
        return receipt


def kill_other_processes(keep=()):
    """SIGKILL every process in the container except this one and ``keep``."""
    me = os.getpid()
    protected = {me, os.getppid(), *keep}
    killed = 0
    for round_index in range(3):
        remaining = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid in protected or pid == 1 and me != 1:
                # PID 1 is the provider's init when the service is not it.
                continue
            try:
                os.kill(pid, 9)
                remaining.append(pid)
                killed += 1
            except ProcessLookupError:
                continue
            except PermissionError:
                raise ResetError('cannot terminate a stray process in the testbed container')
        if not remaining:
            break
        time.sleep(0.2 * (round_index + 1))
    return killed
