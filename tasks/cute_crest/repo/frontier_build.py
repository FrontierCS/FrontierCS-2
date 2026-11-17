#!/usr/bin/env python3
"""Offline build of the concolic tester: toolchain, CIL, harness, strategy.

The same file is used in the solver workspace and by the judge.  Upstream
CREST sources are compiled as they are; the search strategy is the only part
taken from src/frontier_solution/.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

import solution_policy

CXX_HARNESS = ['g++', '-std=gnu++98', '-I.', '-Wall', '-O2']
CXX_STRATEGY = ['g++', '-std=c++17', '-O2', '-Wall', '-fno-plt', '-fno-pic', '-no-pie']
BASE = ['basic_types', 'symbolic_execution', 'symbolic_interpreter', 'symbolic_path',
        'symbolic_predicate', 'symbolic_expression']
# Flags that let the 2008-era CIL front end read current glibc headers.
CIL_COMPAT = ['-std=gnu89', '-w', '-U_FORTIFY_SOURCE', '-D_FORTIFY_SOURCE=0', '-D_Float32=float',
              '-D_Float64=double', '-D_Float128=double', '-D_Float32x=double',
              '-D_Float64x=double', '-D__extension__=']


class BuildError(RuntimeError):
    pass


def run(args, cwd, env, timeout=900):
    p = subprocess.run([str(a) for a in args], cwd=cwd, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout, text=True, errors='replace')
    if p.returncode:
        raise BuildError('build command failed: ' + str(args[0]) + '\n' + p.stdout[-4000:])
    return p.stdout


def environment(root, out, tmp=None):
    """Unpacks the offline toolchain below `out` (once) and returns the
    environment to build with.  A toolchain that is already unpacked is used
    as it is, so `out` may be read-only then; `tmp` names the scratch
    directory for compiler temporaries (default: out/tmp)."""
    root, out = Path(root).resolve(), Path(out).resolve()
    tc = out / 'toolchain'
    assets = root / 'frontier_assets'
    if not (tc / 'ready').exists():
        tc.mkdir(parents=True, exist_ok=True)
        for pin in json.loads((assets / 'pins.json').read_text()):
            path = assets / pin['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != pin['sha256']:
                raise BuildError('offline asset integrity failure')
            if path.suffix == '.deb':
                run(['dpkg-deb', '-x', path, tc], root, os.environ)
            else:
                with tarfile.open(path) as archive:
                    archive.extractall(tc, filter='data')
        wrappers = tc / 'wrappers'
        wrappers.mkdir(exist_ok=True)
        for p in (tc / 'usr/bin').glob('ocaml*'):
            if p.is_file() and p.read_bytes().startswith(b'#!/usr/bin/ocamlrun'):
                w = wrappers / p.name
                w.write_text('#!/bin/sh\nexec ' + shlex.quote(str(tc / 'usr/bin/ocamlrun')) + ' '
                             + shlex.quote(str(p)) + ' "$@"\n')
                w.chmod(0o755)
        (tc / 'ocamlfind.conf').write_text(
            'destdir="' + str(tc / 'usr/lib/ocaml') + '"\npath="' + str(tc / 'usr/lib/ocaml') + '"\n')
        alias = tc / 'z3/lib/libz3.so.4.15'
        if not alias.exists():
            alias.symlink_to('libz3.so')
        (tc / 'ready').touch()
    wrappers = tc / 'wrappers'
    conf = tc / 'ocamlfind.conf'
    temp = Path(tmp).resolve() if tmp else out / 'tmp'
    temp.mkdir(parents=True, exist_ok=True)
    env = {'PATH': ':'.join([str(wrappers), str(tc / 'usr/bin'), str(root / 'bin'),
                             str(root / 'cil/bin'), '/usr/bin', '/bin']),
           'OCAMLLIB': str(tc / 'usr/lib/ocaml'),
           'CAML_LD_LIBRARY_PATH': str(tc / 'usr/lib/ocaml/stublibs'),
           'LD_LIBRARY_PATH': str(tc / 'z3/lib') + ':' + str(tc / 'lib/x86_64-linux-gnu'),
           'OCAMLFIND_CONF': str(conf), 'TMPDIR': str(temp), 'LC_ALL': 'C', 'TZ': 'UTC'}
    return env, tc


EXECUTABLES = ['bin/crestc', 'cil/bin/cilly', 'cil/configure', 'cil/config.guess', 'cil/config.sub',
               'cil/install-sh', 'cil/lib/patcher']


def make_writable(directory):
    """Image layers and sealed build contexts drop write and execute bits."""
    directory = Path(directory)
    for path in [directory] + sorted(directory.rglob('*')):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200 | (0o100 if path.is_dir() else 0))


def build(root, out):
    """Builds CIL, libcrest, process_cfg, the search harness and the write jail."""
    root, out = Path(root).resolve(), Path(out).resolve()
    for name in EXECUTABLES:
        path = root / name
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o755)
    env, tc = environment(root, out)
    for d in ['include', 'lib']:
        (root / d).mkdir(exist_ok=True)
    if not (root / 'cil/bin/cilly.byte').exists():
        run(['./configure'], root / 'cil', env)
        run(['make', '-j1'], root / 'cil', env)
    adapter = out / 'adapter.o'
    run(['g++', '-std=c++11', '-O2', '-I' + str(root / 'src'), '-I' + str(tc / 'z3/include'), '-c',
         root / 'qualification/z3_adapter.cc', '-o', adapter], root, env)
    base = ' '.join('base/' + n + '.o' for n in BASE) + ' ' + str(adapter)
    run(['make', '-j1', 'CXXFLAGS=-std=gnu++98 -I. -Wall -O2', 'CFLAGS=-std=gnu++98 -I. -Wall -O2',
         'LDFLAGS=-L' + str(tc / 'z3/lib') + ' -Wl,-rpath,' + str(tc / 'z3/lib'),
         'LOADLIBES=-lz3', 'BASE_LIBS=' + base], root / 'src', env)
    run(CXX_HARNESS + ['run_crest/search_harness.cc'] + ['base/' + n + '.o' for n in BASE]
        + [adapter, '-L' + str(tc / 'z3/lib'), '-Wl,-rpath,' + str(tc / 'z3/lib'), '-lz3',
           '-o', root / 'bin/search_harness'], root / 'src', env)
    run(['gcc', '-O2', '-shared', '-fPIC', '-o', root / 'lib/jail.so', root / 'benchmarks/jail.c',
         '-ldl'], root, env)
    return env, tc


def build_strategy(root, solution_dir, out_dir, env=None):
    """Checks and compiles a solution directory into out_dir/strategy_host.

    Raises solution_policy.PolicyViolation when the source or the compiled
    object breaks the solution rules, BuildError when it does not compile.
    """
    root, solution_dir, out_dir = Path(root).resolve(), Path(solution_dir).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(env or os.environ)
    sources = solution_policy.check_sources(solution_dir)
    objects = []
    for source in sources:
        obj = out_dir / (source.stem + '.solution.o')
        run(CXX_STRATEGY + ['-I' + str(root / 'src'), '-I' + str(solution_dir), '-c', source,
                            '-o', obj], solution_dir, env, timeout=300)
        objects.append(obj)
    solution_policy.check_objects(objects)
    host_obj = out_dir / 'strategy_host.o'
    run(CXX_STRATEGY + ['-I' + str(root / 'src'), '-c', root / 'src/strategy/strategy_host.cc',
                        '-o', host_obj], root, env, timeout=300)
    binary = out_dir / 'strategy_host'
    run(CXX_STRATEGY + ['-static', host_obj] + objects + ['-o', binary], root, env, timeout=300)
    return binary


GREP_SOURCE = Path('/frontier/task/benchmark-src/grep-2.2')


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    environment_, _ = build(here, here / '.frontier-build')
    build_strategy(here, here / 'src/frontier_solution', here / '.frontier-build/strategy', environment_)
    print('built bin/search_harness and .frontier-build/strategy/strategy_host')
    source = Path(os.environ.get('CREST_GREP_SOURCE', str(GREP_SOURCE)))
    if source.is_dir() and not (here / 'benchmarks/grep/grep').is_file():
        sys.path.insert(0, str(here / 'benchmarks'))
        import build_grep
        build_grep.build(here, environment_, source, here / 'benchmarks/grep')
        print('built benchmarks/grep (instrumented grep 2.2)')
