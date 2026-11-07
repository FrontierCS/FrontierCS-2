"""Builds of the authentic application: the pristine counting CLI and the candidate worker library."""
import os
from pathlib import Path
import shutil
import stat
import subprocess

import isolate  # the shared launcher, beside edit_scope in the Verifier base

TASK = Path(__file__).resolve().parent
ENTRY = Path('apps/hyper/BipartiteButterflies.C')
PRISTINE = TASK / 'pristine'
FLAGS = '-std=c++14 -fopenmp -march=native -O3 -DOPENMP -DBYTERLE -include cstdint'.split()
COMPILE_LIMITS = {'timeout_seconds': 180, 'cpu_seconds': 180, 'memory_bytes': 8 << 30,
                  'file_bytes': 64 << 20, 'output_bytes': 64 << 20}


def make_owner_writable(root):
    """The pristine tree is published 0555/0444; only a disposable copy becomes writable."""
    for directory, names, files in os.walk(root, followlinks=False):
        for path in [Path(directory), *(Path(directory) / name for name in names + files)]:
            if path.is_symlink():
                continue
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode) or stat.S_ISREG(mode):
                os.chmod(path, stat.S_IMODE(mode) | stat.S_IWUSR, follow_symlinks=False)


def clean_environment():
    return {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'OMP_NUM_THREADS': '1',
            'OMP_DYNAMIC': 'FALSE', 'OPENBLAS_NUM_THREADS': '1'}


def build(repo):
    """Build the pristine CLI (trusted: pristine sources and the judge's worker header only)
    and compile the tree's CountTotal into a worker library inside the shared isolate launcher.
    Returns (CLI binary, worker library), both in a judge-private directory."""
    repo = repo.resolve()
    if not (repo / ENTRY).is_file():
        raise RuntimeError('authentic entrypoint is missing')
    private = repo.parent / 'trusted-build'
    private.mkdir()
    compiler_tmp = private / 'tmp'
    compiler_tmp.mkdir()
    parent = private / 'parent'
    shutil.copytree(PRISTINE, parent)
    make_owner_writable(parent)
    shutil.copyfile(TASK / 'trusted_count_total.h', private / 'trusted_count_total.h')
    source = parent / ENTRY
    text = source.read_text()
    original = '#include "butterfly_count_total.h"'
    if text.count(original) != 1:
        raise RuntimeError('authentic entrypoint is missing')
    source.write_text(text.replace(original, '#include "' + str(private / 'trusted_count_total.h') + '"'))
    binary = parent / 'apps/hyper/BipartiteButterflies'
    try:
        p = subprocess.run(['/usr/bin/make', '-j4', 'PCC=g++', 'PCFLAGS=' + ' '.join(FLAGS) + ' -ldl'],
                           cwd=binary.parent, env=dict(clean_environment(), TMPDIR=str(compiler_tmp)),
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('authentic entrypoint did not build') from error
    if p.returncode or not binary.is_file():
        raise RuntimeError('authentic entrypoint did not build')
    (repo / 'apps/hyper/fs2_worker.cpp').write_text('''#include "BipartiteButterflies.C"
extern "C" long fs2_count_total(bipartiteCSR* g, bool v, long w, long mw, long ma, int t, int r) {
  return CountTotal(*g, v, w, mw, ma, static_cast<CountType>(t), static_cast<RankType>(r));
}
''')
    # Submitted source is compiled in a fresh confined instance (its own copy of the tree,
    # no judge files, no network); only the library comes back.
    # Makefile COMMON normally creates these Ligra include symlinks in apps/hyper.
    result = isolate.run(['/usr/bin/g++', *FLAGS, '-I', '{work}/ligra', '-fPIC', '-shared',
                               '{work}/apps/hyper/fs2_worker.cpp', '-o', '{out}/worker.so'],
                              tree=str(repo), outputs=['worker.so'], env={'LANG': 'C'},
                              path='/usr/bin:/bin', **COMPILE_LIMITS)
    library = result.outputs.get('worker.so')
    if not result.ok or not library:
        raise RuntimeError('submission does not build')
    # The two files a counting run gets as its instance tree.
    run = private / 'run'
    run.mkdir()
    shutil.copy2(binary, run / binary.name)
    (run / 'worker.so').write_bytes(library)
    (run / 'worker.so').chmod(0o400)
    return run / binary.name, run / 'worker.so'
