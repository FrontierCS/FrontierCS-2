import ctypes
import errno
import json
import os
import sys


def restrict(policy):
    library = ctypes.CDLL(None, use_errno=True)
    library.syscall.restype = ctypes.c_long
    abi = library.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError('Landlock is required; refusing unsandboxed execution')
    handled = (1 << 13) - 1
    if abi >= 2:
        handled |= 1 << 13
    if abi >= 3:
        handled |= 1 << 14
    attributes = ctypes.c_uint64(handled)
    ruleset = library.syscall(444, ctypes.byref(attributes), ctypes.sizeof(attributes), 0)
    if ruleset < 0:
        raise RuntimeError('cannot create filesystem sandbox')

    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]

    read_access = (1 << 0) | (1 << 2) | (1 << 3)
    for mode, paths in [('read', policy['read']), ('write', policy['write'])]:
        for path in set(paths):
            if not os.path.exists(path):
                continue
            access = handled if mode == 'write' else read_access
            if not os.path.isdir(path):
                access &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
            descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
            rule = Beneath(access, descriptor)
            status = library.syscall(445, ruleset, 1, ctypes.byref(rule), 0)
            os.close(descriptor)
            if status:
                raise RuntimeError('cannot add filesystem sandbox rule')
    if library.prctl(38, 1, 0, 0, 0) or library.syscall(446, ruleset, 0):
        raise RuntimeError('cannot enforce filesystem sandbox')
    os.close(ruleset)
    seccomp = ctypes.CDLL('libseccomp.so.2')
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x7fff0000)
    blocked = ['socket', 'connect', 'bind', 'listen', 'accept', 'accept4', 'ptrace', 'process_vm_readv', 'process_vm_writev', 'mount', 'umount2', 'pivot_root', 'chroot', 'unshare', 'setns', 'bpf', 'perf_event_open', 'userfaultfd', 'io_uring_setup', 'keyctl', 'add_key', 'request_key', 'reboot', 'kexec_load', 'init_module', 'finit_module', 'delete_module']
    if policy['runtime']:
        blocked += ['socketpair', 'clone', 'clone3', 'fork', 'vfork', 'kill', 'tkill', 'tgkill']
    for name in blocked:
        number = seccomp.seccomp_syscall_resolve_name(name.encode())
        if number >= 0 and seccomp.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0):
            raise RuntimeError('cannot install syscall policy')
    if seccomp.seccomp_load(context):
        raise RuntimeError('cannot enforce syscall sandbox')
    seccomp.seccomp_release(context)


if __name__ == '__main__':
    policy = json.loads(sys.argv[1])
    os.chdir(sys.argv[2])
    restrict(policy)
    os.execve(sys.argv[3], sys.argv[3:], dict(os.environ))
