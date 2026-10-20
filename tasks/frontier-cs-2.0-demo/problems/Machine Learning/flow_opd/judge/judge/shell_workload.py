"""Pytest collection of the real upstream shell entrypoint as one workload.

The collector executes Bash on the original file; it neither interprets the
shell file as Python nor supplies model/training computation. It is judge-owned
and used on every real Verifier workload, not just structural verification.
"""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

import pytest


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
    def __init__(self, tree, entrypoint, env, output, deadline):
        self.tree = Path(tree).resolve()
        self.entrypoint = entrypoint
        self.script = self.tree/entrypoint
        if self.script.is_symlink() or not self.script.is_file():
            raise ValueError('Missing regular upstream entrypoint.')
        if not self.script.resolve().is_relative_to(self.tree):
            raise ValueError('Entrypoint escapes the patched tree.')
        self.env = dict(env)
        self.output = Path(output)
        self.deadline = deadline
        self.executions = 0
        self.returncode = None

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
        with (self.output/'private-runtime.log').open('wb') as log:
            process = subprocess.Popen(['bash', self.entrypoint], cwd=self.tree,
                                       env=self.env, stdout=log, stderr=log,
                                       start_new_session=True)
            try:
                self.returncode = process.wait(timeout=remaining)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
        if self.returncode != 0:
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
