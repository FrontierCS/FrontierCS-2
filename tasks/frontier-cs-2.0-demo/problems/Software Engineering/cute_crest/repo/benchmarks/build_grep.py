#!/usr/bin/env python3
"""Instruments the development program, grep 2.2, with CREST.

usage: build_grep.py SOURCE_DIR OUTPUT_DIR
SOURCE_DIR is benchmarks/grep-2.2 of the CREST distribution (GNU grep 2.2
with CREST's symbolic-input driver).  OUTPUT_DIR receives the instrumented
binary `grep` and the `branches` and `cfg_branches` files the harness reads.
"""
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import frontier_build  # noqa: E402

ARGV = ['grep', 'aaaaaaaaaaaaaaaaaaaa', 'in.txt']


def build(root, env, source, output):
    root, source, output = Path(root).resolve(), Path(source).resolve(), Path(output).resolve()
    work = output / 'src'
    if work.exists():
        shutil.rmtree(work)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, work)
    frontier_build.make_writable(work)
    compat = ' '.join(frontier_build.CIL_COMPAT)
    # The distribution ships configured Makefiles; keep make from regenerating them.
    for path in sorted(work.rglob('*')):
        if path.is_file():
            path.touch()
    frontier_build.run(['make', '-C', 'intl', 'CC=gcc', 'CFLAGS=-O2 -std=gnu89 -w', 'libintl.a'], work, env)
    frontier_build.run(
        ['make', '-f', 'Makefile.CREST', 'grep', 'CC=cilly',
         'CPPFLAGS=-I%s/include -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64' % root,
         'CFLAGS=-g -O2 %s --save-temps --doCrestInstrument' % compat,
         'LDFLAGS=-L%s/lib' % root, 'LIBS=-lcrest -lstdc++'], work / 'src', env)
    frontier_build.run([root / 'bin/process_cfg'], work / 'src', env)
    for name in ('grep', 'branches', 'cfg_branches'):
        shutil.copyfile(work / 'src' / name, output / name)
    (output / 'grep').chmod(0o755)
    return output


if __name__ == '__main__':
    root_ = HERE.parent
    env_, _ = frontier_build.environment(root_, root_ / '.frontier-build')
    build(root_, env_, sys.argv[1], sys.argv[2])
    print('built', sys.argv[2])
