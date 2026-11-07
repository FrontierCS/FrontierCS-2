"""Experimental kernel-only boundary: all pathname operations denied."""
import ctypes, errno, os, sys

class Ruleset(ctypes.Structure):
    _fields_ = [('handled_access_fs', ctypes.c_uint64),
                ('handled_access_net', ctypes.c_uint64), ('scoped', ctypes.c_uint64)]


class Beneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int)]


class Comparison(ctypes.Structure):
    _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_int),
                ('datum_a', ctypes.c_uint64), ('datum_b', ctypes.c_uint64)]


def confine(readable):
    if sys.platform != 'linux' or os.uname().machine != 'x86_64':
        raise RuntimeError('required Linux amd64 isolation unavailable')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    lib = ctypes.CDLL('libseccomp.so.2', use_errno=True)

    def checked(rc, name):
        if rc < 0:
            raise OSError(ctypes.get_errno(), name)
        return rc

    abi = checked(libc.syscall(444, 0, 0, 1), 'Landlock ABI query')
    if abi < 6:
        raise RuntimeError('Landlock ABI6 required, no weakened fallback')
    checked(libc.prctl(38, 1, 0, 0, 0), 'no_new_privs')
    # ABI6 rights: all16 filesystem bits, TCP, signals and abstract Unix sockets.
    rules = Ruleset((1 << 16) - 1, 3, 3)
    fd = checked(libc.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0), 'Landlock create')
    try:
        for path, directory in readable:
            access = (1 << 2) | ((1 << 3) if directory else 0)
            source = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = Beneath(access, source)
                checked(libc.syscall(445, fd, 1, ctypes.byref(rule), 0), 'Landlock add rule')
            finally:
                os.close(source)
        checked(libc.syscall(446, fd, 0), 'Landlock restrict')
    finally:
        os.close(fd)

    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                         ctypes.c_uint, ctypes.POINTER(Comparison)]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise RuntimeError('seccomp allocation failed')
    try:
        allowed = (
            'read write close fstat lseek mmap mprotect munmap brk '
            'rt_sigaction rt_sigprocmask rt_sigreturn poll ppoll select pselect6 '
            'clock_gettime getpid gettid getppid getrandom exit exit_group '
            'getdents64 getcwd uname futex sched_yield madvise'
        )
        for name in allowed.split():
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number < 0 or lib.seccomp_rule_add(context, 0x7FFF0000, number, 0):
                raise RuntimeError('required syscall rule unavailable: ' + name)
        if lib.seccomp_load(context):
            raise RuntimeError('seccomp enforcement unavailable')
    finally:
        lib.seccomp_release(context)

