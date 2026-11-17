"""Pytest collection of the real upstream shell entrypoint as one workload.

The collector executes Bash on the original file; it neither interprets the
shell file as Python nor supplies model/training computation. It is judge-owned
and used on every real Verifier workload, not just structural verification.
The run executes in a fresh instance of the shared isolate launcher
(/opt/frontier/isolate.py; on the runner its userns backend): a copy of
the patched tree, the staged inputs, the GPU and the offline models only; no
judge file, no platform environment, nothing left behind. Its reports come back
as files (REPORTS, files.json).
"""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
from pathlib import Path
import shlex
import sys
import time

import pytest

import isolate  # /opt/frontier, on the import path by frontier.pth

# The run reports copied out of the instance (instance_entry.py) and its file listing.
REPORTS = ('progress.json', 'effective_config.json', 'runtime_adaptations.json')
FILE_LIST = 'files.json'


@contextmanager
def collection_context(tree, output):
    """Prevent candidate test configuration/plugins and keep all logs private."""
    previous_cwd = Path.cwd()
    keys = ('PYTEST_DISABLE_PLUGIN_AUTOLOAD', 'PYTEST_ADDOPTS', 'PYTEST_PLUGINS')
    previous_env = {key: os.environ.get(key) for key in keys}
    ini = Path(__file__).with_name('shell_pytest.ini')
    try:
        os.chdir(tree)
        os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
        os.environ['PYTEST_ADDOPTS'] = shlex.join(['--noconftest', '-c', str(ini), '--rootdir', str(tree)])
        os.environ.pop('PYTEST_PLUGINS', None)
        with (output/'private-collection.log').open('w') as log:
            with redirect_stdout(log), redirect_stderr(log):
                yield
    finally:
        os.chdir(previous_cwd)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class ShellWorkloadPlugin:
    """Runs the entrypoint once, in an isolate instance: `inputs` are staged read-only
    under {in}, `sandbox` holds the other isolate.run options. A boundary that cannot be
    built is kept in `infrastructure_error` (pytest would otherwise report it as a failed run)."""

    def __init__(self, tree, entrypoint, inputs, sandbox, output, deadline):
        self.tree = Path(tree).resolve()
        self.entrypoint = entrypoint
        self.script = self.tree/entrypoint
        if self.script.is_symlink() or not self.script.is_file():
            raise ValueError('Missing regular upstream entrypoint.')
        if not self.script.resolve().is_relative_to(self.tree):
            raise ValueError('Entrypoint escapes the patched tree.')
        self.inputs = dict(inputs)
        self.sandbox = dict(sandbox)
        self.output = Path(output)
        self.deadline = deadline
        self.executions = 0
        self.returncode = None
        self.infrastructure_error = None

    def pytest_collect_file(self, file_path, parent):
        if file_path.resolve() == self.script.resolve():
            collector = ShellWorkloadFile.from_parent(parent, path=file_path)
            collector.workload = self
            return collector
        return None

    def execute(self):
        if self.executions:
            raise RuntimeError('The workload entrypoint may execute only once per run.')
        remaining = self.deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Total evaluation deadline exhausted.')
        self.executions += 1
        try:
            result = isolate.run(
                [sys.executable, '{in}/instance_entry.py', '{in}', '{out}', '{home}/run'],
                timeout_seconds=remaining, tree=self.tree, inputs=self.inputs,
                outputs=(*REPORTS, FILE_LIST), **self.sandbox)
        except isolate.IsolateError as error:
            self.infrastructure_error = error
            raise
        (self.output/'private-runtime.log').write_bytes(result.stdout + result.stderr)
        for name, data in result.outputs.items():
            if data is not None:
                (self.output/name).write_bytes(data)
        self.returncode = result.returncode
        if result.timed_out:
            raise TimeoutError('Original upstream workload exceeded the deadline.')
        if not result.ok:
            raise RuntimeError('Original upstream workload failed.')


class ShellWorkloadFile(pytest.File):
    def collect(self):
        yield ShellWorkloadItem.from_parent(self, name='upstream_workload')


class ShellWorkloadItem(pytest.Item):
    def runtest(self):
        self.parent.workload.execute()

    def repr_failure(self, excinfo, style=None):
        # Do not forward private paths, submitted logs or exception details.
        return 'Original upstream workload did not complete.'

    def reportinfo(self):
        return self.path, 0, 'original upstream workload'
