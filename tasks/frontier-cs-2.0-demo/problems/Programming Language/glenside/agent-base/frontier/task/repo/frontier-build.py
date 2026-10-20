#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shutil
import subprocess


def build(output, harness=None, target_dir=None, source_root=None):
    app = Path(__file__).resolve().parent
    output = Path(output).resolve()
    source = (Path(source_root).resolve() if source_root else app) / 'code'
    tree = output / 'code'
    if tree == source or tree in source.parents or source in tree.parents:
        raise ValueError('build output must not overlap the source tree')
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(source, tree)
    repair = (app / 'cpu-feature-gates.patch').read_text()
    sections = repair.split('--- a/')
    repair = ''.join('--- a/' + section for section in sections[1:] if not section.startswith('src/language/rewrites.rs\n'))
    subprocess.run(['patch', '--batch', '-p1'], input=repair, text=True, cwd=tree, check=True, stdout=subprocess.DEVNULL)
    rewrites = tree / 'src/language/rewrites.rs'
    text = rewrites.read_text()
    text = text.replace('pub fn lstm_to_flexasr()', '#[cfg(feature = "tvm")]\npub fn lstm_to_flexasr()')
    text = text.replace('$(#[$meta])*\n            #[test]', '$(#[$meta])*\n            #[cfg(feature = "tvm")]\n            #[test]')
    rewrites.write_text(text)
    command = ['cargo', 'build', '--locked', '--offline', '--no-default-features']
    if harness:
        (tree / 'examples').mkdir(exist_ok=True)
        shutil.copyfile(harness, tree / 'examples/frontier_probe.rs')
        command += ['--example', 'frontier_probe']
    else:
        command += ['--lib']
    environment = dict(os.environ)
    environment['CARGO_TARGET_DIR'] = str(Path(target_dir or output / 'target').resolve())
    environment['CARGO_BUILD_JOBS'] = '2'
    environment['CARGO_PROFILE_DEV_OPT_LEVEL'] = '1'
    environment['CARGO_PROFILE_DEV_DEBUG'] = '0'
    subprocess.run(command, cwd=tree, env=environment, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--harness')
    parser.add_argument('--target-dir')
    parser.add_argument('--source-root')
    arguments = parser.parse_args()
    build(arguments.output, arguments.harness, arguments.target_dir, arguments.source_root)
