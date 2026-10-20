"""Experimental draft-child boundary; install before importing submitted code.

This is author research, not yet an accepted task runtime. The local CPU path
uses Landlock; the CUDA path requires its namespace before installing seccomp.
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
from pathlib import Path


class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class ArgCompare(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint), ("op", ctypes.c_uint),
        ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64),
    ]


def _check(value: int, operation: str) -> int:
    if value < 0:
        raise OSError(ctypes.get_errno(), operation)
    return value


def restrict_filesystem(read_paths: list[Path], write_paths: list[Path]) -> int:
    if platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("unsupported sandbox architecture")
    libc = ctypes.CDLL(None, use_errno=True)
    abi = _check(libc.syscall(444, 0, 0, 1), "Landlock ABI query failed")
    if abi < 3:
        raise RuntimeError("Landlock ABI 3 or newer is required")
    # All filesystem rights through ABI 3, including cross-directory refer and
    # truncation. IOCTL_DEV is intentionally not handled: CUDA requires ioctls.
    all_rights = (1 << 15) - 1
    read_rights = (1 << 2) | (1 << 3)
    file_rights = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
    attr = Ruleset(all_rights)
    ruleset = _check(libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0), "Landlock create failed")
    try:
        for path, rights in [(p, read_rights) for p in read_paths] + [(p, all_rights) for p in write_paths]:
            path = path.resolve(strict=True)
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(rights if path.is_dir() else rights & file_rights, fd)
                _check(libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0), "Landlock rule failed")
            finally:
                os.close(fd)
        _check(libc.prctl(38, 1, 0, 0, 0), "no_new_privs failed")
        _check(libc.syscall(446, ruleset, 0), "Landlock restrict failed")
    finally:
        os.close(ruleset)
    return abi


def restrict_syscalls(library="libseccomp.so.2", *, synchronize_threads=False) -> None:
    # Resolve the ABI soname directly: the private namespace does not expose
    # /sbin/ldconfig, which ctypes.util.find_library invokes on Linux.
    seccomp = ctypes.CDLL(library, use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(ArgCompare)]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint32]
    ctx = seccomp.seccomp_init(0x7FFF0000)
    if not ctx:
        raise RuntimeError("seccomp_init failed")

    def deny(name: str, error: int = errno.EPERM, compare: ArgCompare | None = None) -> None:
        syscall = seccomp.seccomp_syscall_resolve_name(name.encode("ascii"))
        if syscall == -1 and name == "fchmodat2" and platform.machine() in ("x86_64", "aarch64"):
            # Ubuntu 22.04 libseccomp predates the name, but these Linux ABIs
            # assign syscall 452. Keep the denial on older exact task images.
            syscall = 452
        if syscall == -1:
            raise RuntimeError(f"libseccomp lacks syscall {name}")
        result = seccomp.seccomp_rule_add_array(
            ctx, 0x00050000 | error, syscall, int(compare is not None),
            ctypes.byref(compare) if compare is not None else None,
        )
        if result != 0:
            raise RuntimeError(f"seccomp rule {name} failed: {result}")

    try:
        if synchronize_threads:
            # CUDA creates trusted helper threads before candidate import.
            # TSYNC must confine every existing thread; failure is terminal.
            result = seccomp.seccomp_attr_set(ctx, 4, 1)  # SCMP_FLTATR_CTL_TSYNC
            if result != 0:
                raise RuntimeError(f"seccomp thread synchronization unavailable: {result}")
        for name in (
            "ptrace", "process_vm_readv", "process_vm_writev", "kcmp",
            "pidfd_open", "pidfd_getfd", "pidfd_send_signal",
            "kill", "tkill", "tgkill", "rt_sigqueueinfo", "rt_tgsigqueueinfo",
            "setsid", "setpgid", "setpriority", "ioprio_set",
            "sched_setaffinity", "sched_setparam", "sched_setscheduler", "sched_setattr",
            "migrate_pages", "move_pages", "process_madvise",
            "socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
            "io_uring_setup", "io_uring_enter", "io_uring_register",
            "fork", "vfork", "execve", "execveat", "unshare", "setns",
            "mount", "umount2", "pivot_root", "chroot", "open_by_handle_at",
            "bpf", "perf_event_open", "userfaultfd", "keyctl", "add_key", "request_key",
            "shmget", "shmat", "shmctl", "semget", "semctl", "msgget", "msgctl",
            # Landlock's file-content rules do not cover these metadata writes.
            "chmod", "fchmod", "fchmodat", "fchmodat2",
            "chown", "fchown", "lchown", "fchownat",
            "utime", "utimes", "futimesat", "utimensat",
            "setxattr", "lsetxattr", "fsetxattr", "removexattr", "lremovexattr", "fremovexattr",
        ):
            deny(name)
        # glibc getrlimit uses prlimit64(0, resource, NULL, old_limit).
        # Limit changes and all explicit process IDs remain forbidden.
        deny("prlimit64", compare=ArgCompare(0, 1, 0, 0))  # SCMP_CMP_NE
        deny("prlimit64", compare=ArgCompare(2, 1, 0, 0))
        # libc falls back to clone for thread creation when clone3 is unavailable.
        deny("clone3", errno.ENOSYS)
        deny("clone", compare=ArgCompare(0, 7, 0x10000, 0))
        result = seccomp.seccomp_load(ctx)
        if result != 0:
            raise RuntimeError(f"seccomp_load failed: {result}")
    finally:
        seccomp.seccomp_release(ctx)


def install(read_paths: list[Path], write_paths: list[Path]) -> int:
    # Resolve/load libseccomp first; install restrictions while single-threaded.
    # No untrusted module or CUDA helper thread may run before this boundary.
    if len(list(Path("/proc/self/task").iterdir())) != 1:
        raise RuntimeError("sandbox must be installed before creating threads")
    restrict_syscalls()
    return restrict_filesystem(read_paths, write_paths)
