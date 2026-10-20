"""Private prototype: irreversibly seal a prepared numerical worker process.

Call only in a disposable interpreter after trusted dependency initialization
and before importing any candidate module. This is not a GPU acceptance claim.
"""

from __future__ import annotations

import ctypes
import errno
import os
import resource


class IsolationUnavailable(RuntimeError):
    pass


def bind_to_parent(expected_parent):
    libc = ctypes.CDLL(None, use_errno=True)
    if os.getppid() != expected_parent or libc.prctl(1, 9, 0, 0, 0) != 0:
        raise IsolationUnavailable("cannot bind sampler lifetime to its trusted parent")
    if os.getppid() != expected_parent:
        raise IsolationUnavailable("trusted parent exited during sampler startup")


class _Comparison(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_uint),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


def seal_process(*, allowed_ioctl_fds=()) -> dict[str, object]:
    """Deny all but in-process numerical work, atomically across all threads."""
    try:
        library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as exc:
        raise IsolationUnavailable("required kernel isolation library unavailable") from exc

    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint32]
    library.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
        ctypes.c_uint, ctypes.POINTER(_Comparison),
    ]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                           ctypes.c_ulong, ctypes.c_ulong]

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise IsolationUnavailable("cannot disable process dumps")
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise IsolationUnavailable("cannot set no_new_privs")

    # An allowlist covers indirect libc/torch access as well as Python APIs.
    # No new files, sockets, executable processes, ptrace or io_uring are allowed.
    allowed = (
        "read", "write", "readv", "writev", "pread64", "pwrite64", "close",
        "fstat", "lseek",
        "mmap", "mmap2", "munmap", "mremap", "mprotect", "brk", "madvise",
        "mlock", "munlock", "mlock2", "mincore",
        "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "sigaltstack",
        "futex", "futex_time64", "futex_waitv", "set_robust_list", "rseq",
        "set_tid_address", "sched_yield", "sched_getaffinity", "sched_getparam",
        "sched_getscheduler", "sched_get_priority_min", "sched_get_priority_max",
        "clock_gettime", "clock_gettime64", "clock_getres", "gettimeofday",
        "nanosleep", "clock_nanosleep", "clock_nanosleep_time64",
        "getpid", "getppid", "gettid", "getuid", "geteuid", "getgid", "getegid",
        "getrandom", "getrusage", "times", "uname", "sysinfo",
        "poll", "ppoll", "select", "pselect6", "epoll_wait", "epoll_pwait",
        "restart_syscall", "exit", "exit_group",
    )
    context = library.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise IsolationUnavailable("cannot initialize syscall filter")

    def rule(name: str, comparisons=(), action=0x7FFF0000) -> None:
        number = library.seccomp_syscall_resolve_name(name.encode("ascii"))
        if number < 0:
            return
        values = (_Comparison * len(comparisons))(*comparisons)
        if library.seccomp_rule_add_array(context, action, number,
                                          len(values), values) != 0:
            raise IsolationUnavailable("cannot add syscall filter rule: " + name)

    pid = os.getpid()
    try:
        # TSYNC is required: dependency-initialization threads must also be sealed.
        if library.seccomp_attr_set(context, 4, 1) != 0:
            raise IsolationUnavailable("thread-synchronized filter unavailable")
        for name in allowed:
            rule(name)
        # F_SETOWN/F_SETLEASE can signal other processes; descriptor duplication
        # could replace a GPU descriptor with a pipe and defeat ioctl scoping.
        for command in (1, 2, 3):  # F_GETFD, F_SETFD, F_GETFL
            rule("fcntl", [_Comparison(1, 4, command, 0)])
        rule("fcntl", [_Comparison(1, 4, 4, 0), _Comparison(2, 7, 0x2000, 0)])
        for descriptor in allowed_ioctl_fds:
            if type(descriptor) is not int or descriptor < 3:
                raise IsolationUnavailable("invalid numerical device descriptor")
            rule("ioctl", [_Comparison(0, 4, descriptor, 0)])
        # clone3 hides its flags behind a pointer; ENOSYS makes glibc use clone.
        rule("clone3", action=0x00050000 | errno.ENOSYS)
        # Only CLONE_THREAD with no exit signal; no fork-like child can survive.
        rule("clone", [_Comparison(0, 7, 0x000100FF, 0x00010000)])
        rule("tgkill", [_Comparison(0, 4, pid, 0)])
        result = library.seccomp_load(context)
        if result != 0:
            raise IsolationUnavailable("kernel rejected synchronized syscall filter")
    finally:
        library.seccomp_release(context)
    return {"sealed": True, "pid": pid, "thread_sync": True,
            "default_action": "EPERM", "new_files_allowed": False,
            "ioctl_descriptors": list(allowed_ioctl_fds)}
