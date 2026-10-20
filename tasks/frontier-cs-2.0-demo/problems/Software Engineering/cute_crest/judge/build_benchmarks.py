#!/usr/bin/env python3
"""Builds everything the evaluator needs, once, at verifier image build.

usage: build_benchmarks.py OUT [--sources DIR]

OUT/tree    the harness tree (a copy of harbor/app), built
OUT/build   its toolchain
OUT/bench   the instrumented programs named by programs.json
OUT/src     the program sources as built, with their licence files, so that
            every binary in the image is accompanied by its source

Sources are fetched by pinned URL and SHA-256 from benchmark_sources.json;
with --sources DIR the archives are read from DIR instead (same hashes).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request

TASK = Path(__file__).resolve().parent
SOURCES = json.loads((TASK / 'benchmark_sources.json').read_text())
PROGRAMS = json.loads((TASK / 'programs.json').read_text())
CREST_PREFIX = 'crest-3e5ececd3793c88f18af47853236af34ffde4569/'


def fetch(source, directory, offline):
    path = Path(directory) / source['file']
    if not path.is_file():
        if offline:
            raise SystemExit('missing source archive: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(source['url'], timeout=300) as response:
            path.write_bytes(response.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
        raise SystemExit('source archive hash mismatch: ' + source['file'])
    return path


def extract(archive, prefix, destination):
    """Extracts the members below `prefix` into `destination`."""
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        members = []
        for member in tar.getmembers():
            if member.name.startswith(prefix) and len(member.name) > len(prefix):
                member.name = member.name[len(prefix):]
                members.append(member)
        tar.extractall(destination, members=members, filter='data')


def touch_all(directory):
    for path in sorted(Path(directory).rglob('*')):
        if path.is_file() and not path.is_symlink():
            path.touch()


def count_branches(path):
    tokens = Path(path).read_text().split()
    i = count = 0
    while i < len(tokens):
        pairs = int(tokens[i + 1])
        count += 2 * pairs
        i += 2 + 2 * pairs
    return count


def build_vim(fb, tree, env, source, output):
    compat = ' '.join(fb.CIL_COMPAT) + ' -include ' + str(tree / 'benchmarks/cilcompat.h')
    src = source / 'src'
    configure_env = dict(env, CC='gcc', CFLAGS='-O2 -std=gnu89 -w')
    fb.run(['./configure', '--disable-gui', '--disable-gpm', '--without-x'], src, configure_env)
    # The `crest` target ends with a process_cfg path relative to the CREST
    # checkout; build the instrumented binary through it and run process_cfg here.
    fb.run(['make', 'crest', 'CREST_CC=cilly --save-temps --doCrestInstrument',
            'CREST_CFLAGS=-DCREST -I%s/include %s' % (tree, compat),
            'CREST_LDFLAGS=-L%s/lib' % tree, 'CREST_LIBS=-lcrest -lstdc++', '-k'],
           src, env, timeout=3600, check=False)
    if not (src / 'vim').is_file():
        raise SystemExit('the instrumented editor did not link')
    fb.run([tree / 'bin/process_cfg'], src, env)
    return src, 'vim'


SED_SOURCES = ['sed.c', 'compile.c', 'execute.c', 'utils.c', 'regex.c', 'getopt.c', 'getopt1.c']


def build_sed(fb, tree, env, source, output):
    """Instruments the stream editor together with its bundled regex and
    getopt code: the pattern matcher is most of what the search explores.
    The seven units are compiled in one directory, in this order (branch ids
    are assigned in compilation order)."""
    compat = fb.CIL_COMPAT + ['-include', str(tree / 'benchmarks/cilcompat.h')]
    touch_all(source)
    fb.run(['./configure'], source, dict(env, CC='gcc', CFLAGS='-O2 -std=gnu89 -w'))
    fb.run(['patch', '-p1', '-i', TASK / 'benchmarks/sed-3.02-crest-driver.patch'], source / 'sed', env)
    combined = source / 'crest-build'
    combined.mkdir()
    for directory in ('sed', 'lib'):
        for path in sorted((source / directory).iterdir()):
            if path.suffix == '.h' or path.name in SED_SOURCES:
                shutil.copyfile(path, combined / path.name)
    shutil.copyfile(source / 'config.h', combined / 'config.h')
    flags = ['-O2', '-I%s/include' % tree] + compat
    for name in SED_SOURCES:
        fb.run(['cilly', '--save-temps', '--doCrestInstrument', '-DHAVE_CONFIG_H', '-I.'] + flags
               + ['-c', name], combined, env)
    fb.run(['cilly', '--save-temps', '--doCrestInstrument'] + flags + ['-o', 'sed']
           + [name[:-2] + '.o' for name in SED_SOURCES] + ['-L%s/lib' % tree, '-lcrest', '-lstdc++'],
           combined, env)
    fb.run([tree / 'bin/process_cfg'], combined, env)
    return combined, 'sed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--sources')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tree = out / 'tree'
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(TASK / 'harbor/app', tree)
    sys.path.insert(0, str(tree))
    import frontier_build as fb
    fb.make_writable(tree)
    sys.path.insert(0, str(tree / 'benchmarks'))
    import build_grep

    # fb.run(check=False) is only needed for the editor's make target.
    original_run = fb.run

    def run(arguments, cwd, env, timeout=900, check=True):
        try:
            return original_run(arguments, cwd, env, timeout)
        except fb.BuildError:
            if check:
                raise
            return ''
    fb.run = run

    env, _ = fb.build(tree, out / 'build')
    downloads = Path(args.sources) if args.sources else out / 'downloads'
    archives = {s['id']: fetch(s, downloads, bool(args.sources)) for s in SOURCES['sources']}
    src = out / 'src'
    extract(archives['crest'], CREST_PREFIX + 'benchmarks/grep-2.2/', src / 'grep-2.2')
    extract(archives['crest'], CREST_PREFIX + 'benchmarks/vim-5.7/', src / 'vim-5.7')
    extract(archives['sed'], 'sed-3.02/', src / 'sed-3.02')

    bench = out / 'bench'
    built = {}
    build_grep.build(tree, env, src / 'grep-2.2', bench / '_grep')
    built['grep'] = (bench / '_grep', 'grep')
    built['vim'] = build_vim(fb, tree, env, src / 'vim-5.7', bench)
    built['sed'] = build_sed(fb, tree, env, src / 'sed-3.02', bench)
    for program in PROGRAMS['development'] + PROGRAMS['final']:
        name = program['argv'][0]
        directory, binary = built[name]
        target = bench / program['directory']
        target.mkdir(parents=True, exist_ok=True)
        for item in (binary, 'branches', 'cfg_branches'):
            shutil.copyfile(directory / item, target / item)
        (target / binary).chmod(0o755)
        if count_branches(target / 'branches') != program['branch_count']:
            raise SystemExit('unexpected branch count for ' + program['id'])
    # The grep build keeps its own copy of the source it compiled.
    shutil.rmtree(src / 'grep-2.2')
    shutil.move(str(bench / '_grep' / 'src'), str(src / 'grep-2.2'))
    shutil.rmtree(bench / '_grep')
    if not args.sources:
        shutil.rmtree(downloads, ignore_errors=True)
    for path in [out] + sorted(out.rglob('*')):
        if not path.is_symlink():
            path.chmod((path.stat().st_mode | 0o444 | (0o111 if path.stat().st_mode & 0o100 else 0)) & ~0o022)
    print('built', out)


if __name__ == '__main__':
    main()
