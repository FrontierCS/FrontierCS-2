"""Private build and process boundaries for the authentic application."""
import os
import hashlib
from pathlib import Path
import resource
import shutil
import signal
import subprocess
from submission_patch import _make_owner_writable

TASK = Path(__file__).resolve().parent
ENTRY = Path('apps/hyper/BipartiteButterflies.C')
FLAGS = '-std=c++14 -fopenmp -march=native -O3 -DOPENMP -DBYTERLE -include cstdint'.split()


def clean_environment():
    # exec also removes inherited verifier memory before any submitted code runs.
    return {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'OMP_NUM_THREADS': '1',
            'OMP_DYNAMIC': 'FALSE', 'OPENBLAS_NUM_THREADS': '1'}


def bounded(command, *, cwd, timeout, env=None):
    def limits():
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
        resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    # Candidate descendants cannot setsid/setpgid; always reap the entire group.
    with subprocess.Popen(list(map(str, command)), cwd=cwd, env=env or clean_environment(),
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          start_new_session=True, close_fds=True, preexec_fn=limits) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def build(repo):
    repo = repo.resolve()
    pristine = TASK / 'harbor/app'
    fixed = {p.relative_to(pristine): p for p in pristine.rglob('*') if p.is_file()}
    submitted = {p.relative_to(repo): p for p in repo.rglob('*') if not p.is_dir()}
    def editable(path):
        return str(path) in ('apps/hyper/butterfly_count_total.h', 'apps/hyper/butterfly_count.h', 'apps/hyper/butterfly_ecount.h', 'apps/hyper/butterfly_peel.h', 'apps/hyper/butterfly_epeel.h', 'apps/hyper/butterfly_putils.h', 'apps/hyper/butterfly_utils.h') or str(path).startswith('apps/hyper/solver_extensions/')
    if any(p.is_symlink() for p in repo.rglob('*')):
        raise RuntimeError('submission does not build')
    for path in fixed.keys() | submitted.keys():
        if not editable(path) and (path not in fixed or path not in submitted or
                                  hashlib.sha256(fixed[path].read_bytes()).digest() !=
                                  hashlib.sha256(submitted[path].read_bytes()).digest()):
            raise RuntimeError('submission does not build')
    if not (repo / ENTRY).is_file():
        raise RuntimeError('authentic entrypoint is missing')
    private = repo.parent / 'trusted-build'
    private.mkdir()
    compiler_tmp = private / 'tmp'
    compiler_tmp.mkdir()
    compiler_env = dict(clean_environment(), TMPDIR=str(compiler_tmp))
    # Only pristine source enters this parent binary. Candidate source is compiled
    # separately under Landlock; even preprocessor/assembler file reads are confined.
    parent = private / 'parent'
    shutil.copytree(TASK / 'harbor/app', parent)
    _make_owner_writable(parent)
    for name in ('trusted_count_total.h', 'trusted_isolation.h', 'sandbox_build.cpp'):
        shutil.copyfile(TASK / name, private / name)
    source = parent / ENTRY
    text = source.read_text()
    original = '#include "butterfly_count_total.h"'
    if text.count(original) != 1:
        raise RuntimeError('authentic entrypoint is missing')
    source.write_text(text.replace(original, '#include "' + str(private / 'trusted_count_total.h') + '"'))
    sandbox = private / 'sandbox-build'
    p = bounded(['/usr/bin/g++', '-std=c++14', '-fopenmp', private / 'sandbox_build.cpp',
                 '-o', sandbox], cwd=private, timeout=30, env=compiler_env)
    if p.returncode:
        raise RuntimeError('submission does not build')
    binary = parent / 'apps/hyper/BipartiteButterflies'
    p = bounded(['/usr/bin/make', '-j4', 'PCC=g++', 'PCFLAGS=' + ' '.join(FLAGS) + ' -ldl'],
                cwd=binary.parent, timeout=180, env=compiler_env)
    if p.returncode or not binary.is_file():
        raise RuntimeError('authentic entrypoint did not build')
    wrapper = repo / 'apps/hyper/fs2_worker.cpp'
    wrapper.write_text('''#include "BipartiteButterflies.C"
extern "C" long fs2_count_total(bipartiteCSR* g, bool v, long w, long mw, long ma, int t, int r) {
  return CountTotal(*g, v, w, mw, ma, static_cast<CountType>(t), static_cast<RankType>(r));
}
''')
    library = repo / 'apps/hyper/fs2_worker.so'
    # Makefile COMMON normally creates these Ligra include symlinks in apps/hyper.
    p = bounded([sandbox, repo, '/usr/bin/g++', *FLAGS, '-I', repo / 'ligra',
                 '-fPIC', '-shared', wrapper, '-o', library],
                cwd=repo, timeout=180)
    if p.returncode or not library.is_file() or library.is_symlink():
        raise RuntimeError('submission does not build')
    # Keep the loader target outside the compiler's writable tree after cleanup.
    final_library = private / 'worker.so'
    shutil.copyfile(library, final_library)
    final_library.chmod(0o400)
    return binary, final_library
