"""Prepare and verify the fixed instance's original install command at image build."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tomllib

from .accuracy import _case, _harness, build_test_request, parse_test_receipt


REQUESTS_ORIGINAL_TEST_PINS = (
    'pytest-httpbin==0.2.0', 'pytest-mock==0.11.0', 'httpbin==0.4.1',
    'Flask==0.10.1', 'Werkzeug==0.11.4', 'Jinja2==2.8', 'MarkupSafe==0.23',
    'itsdangerous==0.24', 'decorator==4.0.9', 'six==1.10.0',
)


def prepare_original_test_dependencies(instance_id, python, testbed, wheelhouse):
    """Restore missing declared plugins without downgrading the target harness."""
    if instance_id != 'psf__requests-3362':
        return
    declared = set((testbed / 'requirements.txt').read_text().splitlines())
    if not set(REQUESTS_ORIGINAL_TEST_PINS) <= declared:
        raise RuntimeError('original Requests test dependency declarations differ')
    # Legacy MarkupSafe needs old wheel-build tools, isolated from the target
    # interpreter's existing pytest/setuptools and discarded before image exit.
    with tempfile.TemporaryDirectory(prefix='swe-original-plugin-build-') as temporary:
        builder = Path(temporary) / 'builder'
        subprocess.run([python, '-m', 'venv', str(builder)], check=True)
        build_python = str(builder / 'bin/python')
        subprocess.run([build_python, '-m', 'pip', 'install',
                        'setuptools==44.0.0', 'wheel==0.38.4'], check=True)
        subprocess.run([build_python, '-m', 'pip', 'wheel', '--no-build-isolation', '--no-deps',
                        '--wheel-dir', str(wheelhouse), *REQUESTS_ORIGINAL_TEST_PINS], check=True)
    subprocess.run([python, '-m', 'pip', 'install', '--no-index', '--find-links',
                    str(wheelhouse), *REQUESTS_ORIGINAL_TEST_PINS], check=True)


def setup_commands(row):
    _, specs, make_script = _harness()
    commands = make_script(row, specs[row['repo']][row['version']], 'testbed', '/testbed',
                           row['base_commit'], row['test_patch'])
    prefix = commands[:-6]
    installs = [command for command in prefix if command.startswith('python -m pip install ')]
    if len(installs) != 1:
        raise RuntimeError('fixed instance must have exactly one original pip setup command')
    return prefix


def build_requirements(testbed):
    path = testbed / 'pyproject.toml'
    if path.exists():
        requirements = tomllib.loads(path.read_text()).get('build-system', {}).get('requires')
        if requirements is not None:
            if not isinstance(requirements, list) or not all(isinstance(value, str) for value in requirements):
                raise RuntimeError('invalid fixed build requirements')
            return requirements
    return ['setuptools>=40.8.0', 'wheel']


def prepare(instance_id, testbed=Path('/testbed'), wheelhouse=Path('/opt/frontier/swe-wheelhouse'), phase='all'):
    row = _case(instance_id)
    commands = setup_commands(row)
    python = '/opt/miniconda3/envs/testbed/bin/python'
    requirements = build_requirements(testbed)
    script = '\n'.join(['#!/bin/bash', 'set -euxo pipefail', *commands]) + '\n'
    if phase in ('prepare', 'all'):
        wheelhouse.mkdir(parents=True, exist_ok=True)
        prepare_original_test_dependencies(instance_id, python, testbed, wheelhouse)
        # Wheels match the instance Python/ABI, not the separate worker's.
        if requirements:
            subprocess.run([python, '-m', 'pip', 'wheel', '--wheel-dir', str(wheelhouse),
                            *requirements], cwd=testbed, check=True)
        subprocess.run(['/bin/bash', '-c', script], cwd=testbed, check=True)
    if phase == 'prepare':
        return
    offline = dict(os.environ, PIP_NO_INDEX='1', PIP_FIND_LINKS=str(wheelhouse), PIP_DISABLE_PIP_VERSION_CHECK='1')
    subprocess.run(['/bin/bash', '-c', script], cwd=testbed, env=offline, check=True)
    subprocess.run(['git', '-c', 'core.fileMode=false', 'diff', '--exit-code'], cwd=testbed, check=True)
    # The unfixed base can fail tests, but the original harness must parse every target.
    from .swe_test_worker import PreparedSWEWorker
    request = build_test_request(instance_id, '')
    # The Docker verification stage is discarded; only its success marker is
    # copied into the pristine prepared runtime stage.
    saved_environment = dict(os.environ)
    try:
        os.environ.update(offline)
        receipt = PreparedSWEWorker(instance_id=instance_id, testbed=testbed,
                                    max_output_bytes=32 * 1024 * 1024).run_correctness(request)
        if receipt.get('execution') != 'complete':
            raise RuntimeError('offline test preflight failed: ' + json.dumps({
                key: receipt.get(key) for key in ('execution', 'stopped_stage', 'syntax_probe_log')})[:4096])
        parse_test_receipt(request, receipt)
    finally:
        os.environ.clear()
        os.environ.update(saved_environment)
    (wheelhouse / 'preflight.json').write_text(json.dumps({
        'instance_id': instance_id, 'base_commit': row['base_commit'],
        'build_requirements': requirements, 'canonical_setup_offline_passed': True,
        'canonical_tests_offline_parsed': True,
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance-id', required=True)
    parser.add_argument('--phase', choices=('prepare', 'verify'), required=True)
    args = parser.parse_args()
    prepare(args.instance_id, phase=args.phase)
