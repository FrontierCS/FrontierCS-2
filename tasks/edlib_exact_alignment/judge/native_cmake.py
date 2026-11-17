"""Canonical CMake build/execute contract.

Builds run serially on one pinned CPU with bounded wall time and output. A native handle is
resolved again immediately before execution. No target-name or source-filename-only fallback
is permitted. Every candidate/reference executable runs in a fresh instance of the shared
isolate launcher (tasks/shared/lib/isolate.py; run_native).
"""
from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import time
import isolate  # the shared launcher, beside edit_scope in the Verifier base
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


# Fixed, judge-owned environment for every native run (the launcher adds PATH, HOME, TMPDIR
# and LANG of the instance). Nothing from the verifier's own environment reaches the process.
NATIVE_ENV = {'LC_ALL': 'C', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
              'MKL_NUM_THREADS': '1'}
NATIVE_OUTPUT_LIMIT = 32 * 1024**2
NATIVE_MEMORY_BYTES = 4 * 1024**3
NATIVE_CPU_SECONDS = 900


def run_native(executable, args, stage, deadline):
    """One candidate/reference executable run in a fresh shared isolate instance.

    `stage` is a directory holding the executable and its inputs; the instance gets a private
    writable copy of it as its working directory ({work}). `executable` and `args` name files
    relative to it. The instance cannot read the judge, write anything that outlives the run,
    use the network, signal, create threads or processes (syscalls="strict"), and is pinned to
    one CPU (the judge pins itself around the call; the launcher and the command inherit it).
    Returns (stderr followed by stdout, elapsed seconds): the elapsed time is the launcher's
    fork-to-exit stamp of the command alone (Result.command_seconds).
    """
    remaining = min(900, deadline - time.monotonic())
    if remaining <= 0:
        raise NativeError('verification timeout')
    previous = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(previous)})
    try:
        result = isolate.run(
            ['{work}/' + str(executable), *[str(a) for a in args]], tree=stage,
            timeout_seconds=remaining, env=NATIVE_ENV, syscalls='strict',
            memory_bytes=NATIVE_MEMORY_BYTES, cpu_seconds=NATIVE_CPU_SECONDS,
            output_bytes=NATIVE_OUTPUT_LIMIT)
    finally:
        os.sched_setaffinity(0, previous)
    if result.timed_out:
        raise NativeError('process timeout')
    if result.limit:
        raise NativeError(result.limit + ' limit')
    if result.returncode:
        raise NativeError('build or executable failed')
    # stderr first: the only valid stderr is the instrumented build's collector line, written
    # unbuffered while the program's stdout is still in its stdio buffer (the order the former
    # single pipe gave). Any other byte on stderr makes the output invalid either way.
    output = result.stderr + result.stdout
    if len(output) > NATIVE_OUTPUT_LIMIT:
        raise NativeError('output limit')
    if not result.command_seconds > 0:
        raise NativeError('invalid elapsed time')
    return output.decode('utf8', errors='replace'), result.command_seconds


def run_bounded(args, cwd, deadline, *, before_process=None, receipts=None):
    """A trusted build tool (cmake, the compiler) on one pinned CPU, bounded in time and output.

    Compiling reads the submitted source but runs none of it; every executable built from it
    runs through run_native. Returns the tool's combined output.
    """
    if before_process is not None:
        before_process()
    started = time.monotonic()
    remaining = min(900, deadline - time.monotonic())
    if remaining <= 0:
        raise NativeError('verification timeout')
    command = [str(x) for x in args]
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'LC_ALL': 'C'}
    cpu = min(os.sched_getaffinity(0))
    try:
        p = subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=remaining, start_new_session=True,
                           preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
    except subprocess.TimeoutExpired as error:
        raise NativeError('process timeout') from error
    if receipts is not None:
        receipts.append({'argv': command, 'exit_code': p.returncode,
                         'wall_seconds': time.monotonic()-started,
                         'output_bytes': len(p.stdout)})
    if len(p.stdout) > NATIVE_OUTPUT_LIMIT:
        raise NativeError('output limit')
    if p.returncode:
        raise NativeError('build or executable failed')
    return p.stdout.decode('utf8', errors='replace')


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
