"""Before/after container-state guard around candidate executions.

Shared verbatim (byte-identical) by the judges of batch_local_counting and
parallel_dbscan. A timed candidate execution must start from the same
container state as every other one: anything it leaves behind (a file, a
process, an IPC object) could carry an earlier result into a later
round or execution. The evaluator takes `snapshot()` before an execution and
calls `require_unchanged()` after it, both outside the timed region.

Compared state:
- every filesystem path except /proc, /sys and /dev, plus /dev/shm and
  /dev/mqueue: type, mode, inode, size, mtime and ctime of each entry;
- processes owned by the evaluator's uid (a surviving process is killed);
- the System V IPC tables in /proc/sysvipc.

The candidate runs as the evaluator's uid. Only a directory that uid owns or
can write can gain, lose or rename entries, and only a file it owns or can
write can change, so after one full scan the guard rescans exactly those
directories (recursively) and files. The `volatile` directories (the candidate
tree, in which the evaluator itself creates and removes result directories)
have their own timestamps excluded; their extended attributes are compared
instead. Everything inside them is still compared.
"""
from __future__ import annotations

import os
import signal
import stat
from pathlib import Path

_SKIP = frozenset({"/proc", "/sys", "/dev"})
_EXTRA = ("/dev/shm", "/dev/mqueue")
_IPC_TABLES = ("/proc/sysvipc/shm", "/proc/sysvipc/msg", "/proc/sysvipc/sem")


class StateGuardViolation(RuntimeError):
    """A candidate execution changed container state outside its result directories."""


def _xattrs(path: str) -> tuple:
    try:
        return tuple(sorted((name, os.getxattr(path, name, follow_symlinks=False))
                            for name in os.listxattr(path, follow_symlinks=False)))
    except OSError:
        return ()


def _mutable(path: str, info: os.stat_result, uid: int) -> bool:
    if uid == 0 or info.st_uid == uid:
        return True
    if stat.S_ISLNK(info.st_mode):
        return False
    try:
        return os.access(path, os.W_OK)
    except OSError:
        return False


def _processes() -> dict[tuple[int, str], int]:
    found: dict[tuple[int, str], int] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            owner = os.stat(f"/proc/{name}").st_uid
            fields = Path(f"/proc/{name}/stat").read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if fields and fields[0] != "Z":  # a zombie holds no state
            found[(int(name), fields[19])] = owner  # field 22: start time
    return found


def _ipc_objects() -> frozenset[str]:
    found: set[str] = set()
    for table in _IPC_TABLES:
        try:
            lines = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) >= 2:
                found.add(f"{table}:{fields[0]}:{fields[1]}")
    return frozenset(found)


class StateGuard:
    def __init__(self, volatile=()):
        self.uid = os.getuid()
        self.volatile = frozenset(os.path.realpath(str(path)) for path in volatile)
        roots: list[str] = []
        files: list[str] = []

        def plan(directory: str) -> None:
            try:
                entries = os.scandir(directory)
            except OSError:
                return
            with entries:
                for entry in entries:
                    if entry.path in _SKIP:
                        continue
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        if _mutable(entry.path, info, self.uid):
                            roots.append(entry.path)  # rescanned whole
                        else:
                            plan(entry.path)
                    elif _mutable(entry.path, info, self.uid):
                        files.append(entry.path)

        for top in ("/",) + _EXTRA:
            try:
                info = os.lstat(top)
            except OSError:
                continue
            if top != "/" and _mutable(top, info, self.uid):
                roots.append(top)
            else:
                plan(top)
        self.roots = tuple(roots)
        self.files = tuple(files)

    def _entry(self, path: str, info: os.stat_result) -> tuple:
        if stat.S_ISDIR(info.st_mode):
            stamp = (_xattrs(path) if path in self.volatile
                     else (info.st_mtime_ns, info.st_ctime_ns))
            return ("dir", info.st_mode, info.st_ino, stamp)
        return ("file", info.st_mode, info.st_ino, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns)

    def _files(self) -> dict[str, tuple]:
        state: dict[str, tuple] = {}

        def record(path: str) -> os.stat_result | None:
            try:
                info = os.lstat(path)
            except OSError:
                return None
            state[path] = self._entry(path, info)
            return info

        def walk(directory: str) -> None:
            try:
                entries = os.scandir(directory)
            except OSError:
                return
            with entries:
                for entry in entries:
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    state[entry.path] = self._entry(entry.path, info)
                    if stat.S_ISDIR(info.st_mode):
                        walk(entry.path)

        for root in self.roots:
            info = record(root)
            if info is not None and stat.S_ISDIR(info.st_mode):
                walk(root)
        for path in self.files:
            record(path)
        return state

    def snapshot(self) -> tuple:
        return self._files(), _processes(), _ipc_objects()

    def require_unchanged(self, before: tuple) -> None:
        files_before, processes_before, ipc_before = before
        survivors = [pid for (pid, start), owner in _processes().items()
                     if (pid, start) not in processes_before and owner == self.uid
                     and pid != os.getpid()]
        for pid in survivors:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if survivors:
            raise StateGuardViolation(
                "a candidate execution left a process running after it exited")
        if _ipc_objects() - ipc_before:
            raise StateGuardViolation(
                "a candidate execution left a System V IPC object behind")
        files_after = self._files()
        changed = sorted(path for path in files_before.keys() | files_after.keys()
                         if files_before.get(path) != files_after.get(path))
        if changed:
            raise StateGuardViolation(
                "a candidate execution changed files outside its result directories: "
                + changed[0][-80:])
