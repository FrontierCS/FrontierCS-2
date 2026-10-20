"""Canonical CMake build/execute contract.

The compiler and every child keep the original single-core, 4 GiB, finite
wall/CPU/output limits. A native handle is resolved again immediately before
execution. No target-name or source-filename-only fallback is permitted.

Native candidate/reference executables additionally run under a guard that
leaves them no way to create or modify a file or to reach another process:
a seccomp allow-list installed before exec (stage 1), a second filter installed
by a preloaded judge library before main() that also removes stat and exec
(stage 2), RLIMIT_FSIZE 0, a fixed environment and piped output.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import ctypes
import fcntl
import resource
import signal
import struct
import subprocess
import threading
import time
from cmake_entrypoint import request_codemodel, resolve_cmake_entrypoint


class NativeError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeExecutable:
    source: Path
    build: Path
    entrypoint: str
    target: str
    artifact: Path

    @property
    def parent(self):
        return self.artifact.parent

    def resolve_for_execution(self):
        actual = resolve_cmake_entrypoint(self.source, self.build,
                                         self.entrypoint, self.target)
        if actual != self.artifact:
            raise NativeError('CMake artifact changed since build')
        return actual

    def __str__(self):
        return str(self.artifact)


class _SockFilter(ctypes.Structure):
    _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte),
                ('jf', ctypes.c_ubyte), ('k', ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [('len', ctypes.c_ushort),
                ('filter', ctypes.POINTER(_SockFilter))]


# Syscalls a pinned, single-process, compute-only native run may make. Everything
# else returns ENOSYS. The list is what glibc 2.39 start-up, malloc, stdio on
# already-open descriptors and clock reads need; it contains nothing that
# creates, renames, removes or re-labels a file, and no IPC, socket, signal or
# process-inspection call.
NATIVE_ALLOWED_SYSCALLS = (
    0,    # read
    1,    # write
    3,    # close
    5,    # fstat            (dynamic loader; denied again by the stage-2 guard)
    8,    # lseek
    9,    # mmap
    10,   # mprotect
    11,   # munmap
    12,   # brk
    13,   # rt_sigaction
    14,   # rt_sigprocmask
    15,   # rt_sigreturn
    17,   # pread64
    19,   # readv
    20,   # writev
    21,   # access
    24,   # sched_yield
    25,   # mremap
    28,   # madvise
    35,   # nanosleep
    39,   # getpid
    59,   # execve           (the one exec of the measured binary; denied by stage 2)
    60,   # exit
    63,   # uname
    96,   # gettimeofday
    98,   # getrusage
    102, 104, 107, 108,  # getuid getgid geteuid getegid
    131,  # sigaltstack
    158,  # arch_prctl
    186,  # gettid
    201,  # time
    202,  # futex
    204,  # sched_getaffinity
    218,  # set_tid_address
    219,  # restart_syscall
    228,  # clock_gettime
    229,  # clock_getres
    230,  # clock_nanosleep
    231,  # exit_group
    262,  # newfstatat       (dynamic loader; denied again by the stage-2 guard)
    273,  # set_robust_list
    309,  # getcpu
    318,  # getrandom
    334,  # rseq
    436,  # close_range      (Python child closing inherited descriptors)
)
# open(2)/openat(2) are allowed only read-only: any of O_WRONLY, O_RDWR,
# O_CREAT or O_TRUNC returns EROFS. O_TMPFILE needs a write access mode.
_OPEN_WRITE_MASK = 0o3 | 0o100 | 0o1000
NATIVE_OUTPUT_LIMIT = 32 * 1024**2
NATIVE_PIPE_BYTES = 1024**2
# Fixed, judge-owned environment for every native run. Nothing from the
# verifier's own environment (role, paths, TMPDIR, HOME) reaches the process.
NATIVE_ENV = {'LC_ALL': 'C', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
              'MKL_NUM_THREADS': '1'}
NATIVE_SYSTEM_INTERPRETERS = frozenset({
    '/lib64/ld-linux-x86-64.so.2',
    '/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2',
})


def validate_native_elf(artifact):
    """Require the system dynamic loader that installs the preloaded guard.

    Static binaries never load LD_PRELOAD and therefore cannot inherit the
    stage-2 restrictions. Check the actual executable before every dispatch,
    including probes, instrumented binaries and the guard self-test.
    """
    def invalid():
        raise NativeError('native executable must be a dynamic x86-64 ELF with the system interpreter')

    with Path(artifact).open('rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        header = stream.read(64)
        if len(header) != 64 or header[:7] != b'\x7fELF\x02\x01\x01':
            invalid()
        fields = struct.unpack('<16sHHIQQQIHHHHHH', header)
        _, kind, machine, version, _, phoff, _, _, ehsize, phsize, phnum, *_ = fields
        if (kind not in (2, 3) or machine != 62 or version != 1 or ehsize != 64
                or phsize != 56 or not 0 < phnum <= 1024 or phoff < 64
                or phoff > size or phnum * phsize > size - phoff):
            invalid()
        interpreter = None
        dynamic = False
        for index in range(phnum):
            stream.seek(phoff + index * phsize)
            segment = stream.read(phsize)
            if len(segment) != phsize:
                invalid()
            segtype, _, offset, _, _, length, _, _ = struct.unpack('<IIQQQQQQ', segment)
            if segtype not in (2, 3):  # PT_DYNAMIC / PT_INTERP
                continue
            if offset > size or length > size - offset or length == 0:
                invalid()
            if segtype == 2:
                dynamic = True
                continue
            if interpreter is not None or length > 256:
                invalid()
            stream.seek(offset)
            raw = stream.read(length)
            if not raw.endswith(b'\0') or b'\0' in raw[:-1]:
                invalid()
            try:
                interpreter = raw[:-1].decode('ascii')
            except UnicodeDecodeError:
                invalid()
        if not dynamic or interpreter not in NATIVE_SYSTEM_INTERPRETERS:
            invalid()
        # Both approved spellings must resolve to the immutable system loader,
        # never to anything supplied in the candidate's disposable source tree.
        loader = Path('/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2')
        if not loader.is_file() or Path(interpreter).resolve() != loader.resolve():
            invalid()


def native_filter_instructions():
    """Stage 1 of the native guard, installed before exec and inherited by it.

    Linux x86_64 only: reject other audit architectures and the x32 syscall
    namespace. sched_setaffinity and every task-creation syscall kill the
    process (unchanged parallel guard). Read-only open/openat, prlimit64 on
    self, prctl(PR_SET_SECCOMP) (so the stage-2 guard can only tighten) and the
    allow-list above succeed; every other syscall returns ENOSYS.
    """
    kill, allow = 0x80000000, 0x7fff0000  # KILL_PROCESS / ALLOW
    errno = lambda code: 0x00050000 | code
    instructions = [
        (0x20, 0, 0, 4),             # load seccomp_data.arch
        (0x15, 1, 0, 0xc000003e),    # AUDIT_ARCH_X86_64 only
        (0x06, 0, 0, kill),
        (0x20, 0, 0, 0),             # load seccomp_data.nr
        (0x45, 0, 1, 0x40000000),    # reject __X32_SYSCALL_BIT
        (0x06, 0, 0, kill),
    ]
    # x86_64 clone, fork, vfork, sched_setaffinity, clone3.
    for number in (56, 57, 58, 203, 435):
        instructions += [(0x15, 0, 1, number), (0x06, 0, 0, kill)]
    # open(path, flags, ...) and openat(dirfd, path, flags, ...): low word of flags.
    for number, offset in ((2, 24), (257, 32)):
        instructions += [(0x15, 0, 4, number), (0x20, 0, 0, offset),
                         (0x45, 1, 0, _OPEN_WRITE_MASK),
                         (0x06, 0, 0, allow), (0x06, 0, 0, errno(30))]  # EROFS
    # prlimit64(pid == 0, ...) and prctl(PR_SET_SECCOMP == 22, ...) only.
    for number, value in ((302, 0), (157, 22)):
        instructions += [(0x15, 0, 4, number), (0x20, 0, 0, 16),
                         (0x15, 0, 1, value),
                         (0x06, 0, 0, allow), (0x06, 0, 0, errno(1))]   # EPERM
    for number in NATIVE_ALLOWED_SYSCALLS:
        instructions += [(0x15, 0, 1, number), (0x06, 0, 0, allow)]
    instructions.append((0x06, 0, 0, errno(38)))                        # ENOSYS
    return instructions


def install_native_guard():
    """Pinning has already succeeded. The seccomp filter survives exec.

    There is no fallback: if the filter cannot be installed the process does
    not start.
    """
    instructions = native_filter_instructions()
    filters = (_SockFilter * len(instructions))(
        *(_SockFilter(*row) for row in instructions))
    program = _SockFprog(len(instructions), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                     ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    # PR_SET_NO_NEW_PRIVS makes unprivileged installation possible and immutable.
    if prctl(38, 1, 0, 0, 0) != 0:
        raise NativeError('native guard no_new_privs failed')
    # PR_SET_SECCOMP / SECCOMP_MODE_FILTER: no fallback on unsupported runtime.
    if prctl(22, 2, ctypes.addressof(program), 0, 0) != 0:
        raise NativeError('native guard seccomp installation failed')


def protect_verifier_process():
    """Make this process's /proc/<pid>/mem and friends unreadable to children.

    The verifier holds the reference's results in memory while the candidate
    runs under the same UID.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:   # PR_SET_DUMPABLE = 0
        raise NativeError('verifier process protection failed')


def run_bounded(args, cwd, deadline, *, before_process=None, receipts=None,
                preload=None):
    """Bounded Popen/wait4 contract, with checked native dispatch.

    A native command (first element a NativeExecutable or Path) runs with the
    fixed NATIVE_ENV, RLIMIT_FSIZE 0, /dev/null on stdin and the native guard.
    Output of every command is captured through a pipe with a bounded reader;
    nothing is written to the working directory by this function.

    Returns (output, peak RSS bytes, elapsed seconds). The elapsed time is
    measured by this verifier and cannot be written by the measured program:
    CLOCK_MONOTONIC is read in the forked child by trusted code after pinning,
    limits and the guard are in place, immediately before exec, and again by a
    waiting thread the moment the kernel reports the process reaped. It covers
    exec, dynamic loading, the whole program and its exit.
    """
    if before_process is not None:
        before_process()
    started = time.monotonic()
    remaining = min(900, deadline - time.monotonic())
    if remaining <= 0:
        raise NativeError('verification timeout')
    if not args:
        raise NativeError('empty command')
    command = list(args)
    guard_native = isinstance(command[0], (NativeExecutable, Path))
    if guard_native and os.uname().machine != "x86_64":
        raise NativeError("native guard requires Linux x86_64")
    if isinstance(command[0], NativeExecutable):
        command[0] = command[0].resolve_for_execution()
    if guard_native:
        validate_native_elf(command[0])
    command = [str(x) for x in command]
    if guard_native:
        env = dict(NATIVE_ENV)
        if preload is not None:
            env['LD_PRELOAD'] = str(preload)
    else:
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '1', 'LC_ALL': 'C'}
    file_limit = 0 if guard_native else 32*1024**2

    def limits():
        os.setsid()
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (900, 900))
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
        if guard_native:
            install_native_guard()
        # Last trusted action before exec. The descriptor is close-on-exec.
        os.write(clock_writer, struct.pack('=q', time.clock_gettime_ns(time.CLOCK_MONOTONIC)))

    reader, writer = os.pipe()
    clock_reader, clock_writer = os.pipe()
    chunks = []
    captured = 0

    def drain():
        """Non-blocking; returns True at end of file."""
        nonlocal captured
        while True:
            try:
                block = os.read(reader, 1 << 16)
            except BlockingIOError:
                return False
            if not block:
                return True
            captured += len(block)
            if captured > NATIVE_OUTPUT_LIMIT:
                raise NativeError('output limit')
            chunks.append(block)

    try:
        # The whole expected output fits in the pipe, so the measured process
        # never waits for this reader inside its own timed interval.
        try:
            fcntl.fcntl(writer, 1031, NATIVE_PIPE_BYTES)   # F_SETPIPE_SZ
        except OSError:
            pass
        capacity = fcntl.fcntl(writer, 1032)               # F_GETPIPE_SZ
        os.set_blocking(reader, False)
        try:
            p = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=writer, stderr=subprocess.STDOUT,
                                 preexec_fn=limits)
        finally:
            os.close(writer)
            os.close(clock_writer)
        end = time.monotonic() + remaining
        reaped = {}

        def reap():
            # Blocks in the kernel; the timestamp is taken as soon as it returns.
            pid, status, usage = os.wait4(p.pid, 0)
            reaped.update(finished=time.clock_gettime_ns(time.CLOCK_MONOTONIC),
                          status=status, usage=usage)

        waiter = threading.Thread(target=reap, daemon=True)
        waiter.start()
        try:
            while True:
                waiter.join(.01)
                if not waiter.is_alive():
                    break
                if time.monotonic() > end:
                    raise NativeError('process timeout')
                drain()
        finally:
            if waiter.is_alive():
                os.killpg(p.pid, signal.SIGKILL)
                waiter.join()
        status, usage = reaped['status'], reaped['usage']
        p.returncode = os.waitstatus_to_exitcode(status)
        stamp = os.read(clock_reader, 8)
        if len(stamp) != 8:
            raise NativeError('process start time missing')
        elapsed = (reaped['finished'] - struct.unpack('=q', stamp)[0]) / 1e9
        # The process is gone; what it wrote is already in the pipe.
        while not drain():
            if time.monotonic() > end:
                raise NativeError('output pipe still open after exit')
            time.sleep(.001)
    finally:
        os.close(reader)
        os.close(clock_reader)
    if receipts is not None:
        receipts.append({'argv': command, 'exit_code': p.returncode,
                         'wall_seconds': time.monotonic()-started,
                         'elapsed_seconds': elapsed,
                         'cpu_seconds': usage.ru_utime+usage.ru_stime,
                         'peak_rss_bytes': usage.ru_maxrss*1024,
                         'output_bytes': captured, 'pipe_capacity_bytes': capacity})
    if p.returncode:
        raise NativeError('build or executable failed')
    if usage.ru_maxrss*1024 > 4*1024**3:
        raise NativeError('memory limit')
    if not elapsed > 0:
        raise NativeError('invalid elapsed time')
    return b''.join(chunks).decode('utf8', errors='replace'), usage.ru_maxrss*1024, elapsed


def build_native_guard(directory, deadline, *, before_process=None, receipts=None):
    """Compile the stage-2 guard and the self-test that proves both stages."""
    directory = Path(directory)
    source = Path(__file__).resolve().parent/'judge'
    guard = directory/'libedlib-native-guard.so'
    selftest = directory/'native-guard-selftest'
    options = dict(before_process=before_process, receipts=receipts)
    run_bounded(['cc', '-O2', '-fPIC', '-shared', source/'native_guard.c', '-o', guard],
                directory, deadline, **options)
    run_bounded(['cc', '-O2', source/'native_guard_selftest.c', '-o', selftest],
                directory, deadline, **options)
    return guard, selftest


def build_cmake_entrypoint(source, build, entrypoint, target, deadline, *,
                           sanitize=False, before_process=None, receipts=None):
    """Build serially, then bind the original compiled source to its artifact."""
    source, build = Path(source).resolve(), Path(build).resolve()
    request_codemodel(build)
    options = dict(before_process=before_process, receipts=receipts)
    command = ['cmake', '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release']
    if sanitize:
        command += ['-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer',
                    '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined']
    run_bounded(command, source, deadline, **options)
    run_bounded(['cmake', '--build', build, '-j1'], source, deadline, **options)
    artifact = resolve_cmake_entrypoint(source, build, entrypoint, target)
    return NativeExecutable(source, build, entrypoint, target, artifact)
