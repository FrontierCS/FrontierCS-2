import json
import math
import os
import signal
import subprocess
import tempfile
from pathlib import Path


class IsolatedProcessError(RuntimeError):
    """Bounded completed-process receipt; launch and timeout faults stay separate."""

    def __init__(self, returncode, stdout, stderr):
        super().__init__("isolated process failed: " + stderr)
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def run_isolated(command, *, mounts, cwd, timeout, environment, payload=None):
    def translate(value):
        for source, destination, _ in sorted(mounts, key=lambda mount: -len(mount[1])):
            if value == destination or value.startswith(destination + '/'):
                return source + value[len(destination):]
        return value

    with tempfile.TemporaryDirectory(prefix='isolated-') as temporary:
        environment = {name: translate(value) for name, value in environment.items()}
        environment.update({'TMPDIR': temporary, 'HOME': temporary})
        if 'PATH' in environment:
            environment['PATH'] = ':'.join(translate(path) for path in environment['PATH'].split(':'))
        policy = {
            'read': ['/usr', '/bin', '/lib', '/lib64', '/etc/alternatives', '/etc/ld.so.cache', '/dev/null', '/dev/urandom', '/dev/random'],
            'write': [temporary, '/dev/null'],
            'runtime': len(mounts) == 1,
        }
        for source, _, writable in mounts:
            policy['read'].append(source)
            if writable:
                policy['write'].append(source)
        launcher = Path(__file__).with_name('isolate_exec.py')
        launch = ['/usr/bin/python3', str(launcher), json.dumps(policy), translate(cwd), *map(translate, command)]
        launch = ['prlimit', '--as=3221225472', '--fsize=67108864', f'--cpu={math.ceil(timeout)}', '--', *launch]
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(launch, stdin=subprocess.PIPE, stdout=output, stderr=errors, env=environment, start_new_session=True)
            try:
                process.communicate((payload or '').encode(), timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise RuntimeError('sandbox deadline exceeded') from None
            if process.returncode:
                errors.seek(0)
                output.seek(0)
                raise IsolatedProcessError(process.returncode, output.read(2_097_152).decode(errors='replace'), errors.read(16384).decode(errors='replace'))
            output.seek(0)
            result = output.read(2_097_153)
            if len(result) > 2_097_152:
                raise RuntimeError('oversized process result')
            return result.decode()
