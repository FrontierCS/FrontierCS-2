"""Task-local Linux isolation for submitted C++ compilation and execution."""
from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import resource

LIBC = ctypes.CDLL(None, use_errno=True)
SECCOMP = ctypes.CDLL("libseccomp.so.2", use_errno=True)
SECCOMP.seccomp_init.argtypes = [ctypes.c_uint32]
SECCOMP.seccomp_init.restype = ctypes.c_void_p
SECCOMP.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
SECCOMP.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
SECCOMP.seccomp_load.argtypes = [ctypes.c_void_p]
SECCOMP.seccomp_release.argtypes = [ctypes.c_void_p]


class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class Comparison(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("a", ctypes.c_uint64), ("b", ctypes.c_uint64)]


def _checked(result):
    if result < 0:
        raise OSError(ctypes.get_errno(), "candidate isolation failed")
    return result


def restrict(read_paths, write_paths=(), *, runtime=False, affinity=None,
             file_bytes=7516192768, address_bytes=48 * 1024**3, cpu_seconds=10800):
    """Fail closed if the host cannot enforce the task's declared sandbox."""
    if affinity:
        os.sched_setaffinity(0, affinity)
    abi = _checked(LIBC.syscall(444, 0, 0, 1))
    if abi < 3:
        raise RuntimeError("Linux Landlock ABI >= 3 is required")
    handled = (1 << 15) - 1
    ruleset = Ruleset(handled)
    fd = _checked(LIBC.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0))
    try:
        read_paths = [*read_paths, f"/proc/{os.getpid()}/fd"]
        for paths, access in ((read_paths, 1 | 4 | 8), (write_paths, handled)):
            for name in paths:
                path = Path(name).resolve()
                if not path.exists():
                    continue
                pathfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    allowed = access if path.is_dir() else access & (1 | 2 | 4 | (1 << 14))
                    rule = PathRule(allowed, pathfd)
                    _checked(LIBC.syscall(445, fd, 1, ctypes.byref(rule), 0))
                finally:
                    os.close(pathfd)
        _checked(LIBC.prctl(38, 1, 0, 0, 0))
        _checked(LIBC.syscall(446, fd, 0))
    finally:
        os.close(fd)

    context = SECCOMP.seccomp_init(0x7FFF0000)
    if not context:
        raise RuntimeError("seccomp initialization failed")
    try:
        names = ["socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
                 "ptrace", "process_vm_readv", "process_vm_writev", "pidfd_open",
                 "pidfd_getfd", "pidfd_send_signal", "kill", "tkill", "tgkill",
                 "mount", "umount2", "pivot_root", "setns", "unshare", "bpf",
                 "open_by_handle_at", "io_uring_setup", "perf_event_open", "userfaultfd",
                 "shmget", "shmat", "shmctl", "msgget", "semget", "setpriority",
                 "sched_setscheduler", "sched_setparam", "setsid", "setpgid"]
        if runtime:
            names += ["fork", "vfork", "sched_setaffinity"]
        for name in names:
            number = SECCOMP.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and SECCOMP.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0):
                raise RuntimeError("seccomp rule failed")
        if runtime:
            # glibc falls back to clone; permit OpenMP threads, not new processes.
            number = SECCOMP.seccomp_syscall_resolve_name(b"clone3")
            if number >= 0:
                _checked(SECCOMP.seccomp_rule_add(context, 0x00050000 | errno.ENOSYS, number, 0))
            rule = Comparison(0, 7, 0x10000, 0)
            _checked(SECCOMP.seccomp_rule_add(context, 0x00050000 | errno.EPERM,
                                             SECCOMP.seccomp_syscall_resolve_name(b"clone"),
                                             1, rule))
        _checked(SECCOMP.seccomp_load(context))
    finally:
        SECCOMP.seccomp_release(context)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
    resource.setrlimit(resource.RLIMIT_AS, (address_bytes, address_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))


def system_paths():
    return ["/usr", "/lib", "/lib64", "/bin", "/etc/ld.so.cache",
            "/sys/devices/system/cpu", "/dev/null", "/dev/urandom"]
