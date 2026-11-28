"""One leased parent of a workload group and its pod, over the judge's Pumpkin session.

A lease of the judge's session is one GPU parent with its companions: the trusted
scheduler and, for an SWE group, each instance's interactive and check containers.
Pumpkin starts the containers; this starts the catalog's service in each companion,
then runs one command at a time in the parent or a companion, with its stdin and its
output as files. The lease ends with the session, and its sandbox with it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import secrets
import shutil
import subprocess
import time

from runtime_config import canonical_digest, execution_variants
from serving_eval import sidecar_env
from sidecar_catalog import SCHEDULER, SCHEDULER_PORT, SCHEDULER_PYTHON, SCRATCH, expected_sidecars

SANDBOXCTL = '/.pumpkin/bin/sandboxctl'
# Where a container keeps each command's stdin and output, and its service's log.
WORK = SCRATCH + '/frontier-verifier'
SERVICE_LOG = SCRATCH + '/frontier-service.log'
# Each service outlives the command that starts it, in a session of its own.
START = 'cd "$0" && setsid "$@" > ' + SERVICE_LOG + ' 2>&1 < /dev/null &'
# The scheduler answers before any job is sent: one that never starts is the platform's failure,
# where a job that waits for it in vain would be charged to the candidate. The command's timeout
# bounds the wait.
SCHEDULER_READY = """import sys, time, urllib.request
while True:
    try:
        urllib.request.urlopen(sys.argv[1], timeout=5).close()
        break
    except OSError:
        time.sleep(0.1)
"""
SCHEDULER_READY_SECONDS = 120
COMMAND = 'mkdir "$0.out" && exec "$@" < "$0/stdin" > "$0.out/stdout" 2> "$0.out/stderr"'
ENDED = frozenset({'closed', 'expired', 'failed'})
CHUNK_BYTES = 65536


class SidecarExecutionError(RuntimeError):
    """Unscored execution/transport evidence, without candidate attribution."""


class RemoteExecutionError(SidecarExecutionError):
    """CPU observed a bounded remote operation fail after local setup succeeded.

    This conveys no infrastructure exemption; the enclosing C1/C2 owner decides
    the affected scope. Local filesystem, command and allocation errors use the
    original trusted error channel instead.
    """

    def __init__(self, message, *, cause):
        if cause not in {'deadline', 'output_limit', 'child_exit', 'protocol_error'}:
            raise ValueError('unknown bounded remote operation failure')
        super().__init__(message)
        self.cause = cause


class PlatformError(SidecarExecutionError):
    """Pumpkin refused an op, lost the lease, started no command or failed a transfer."""

    def __init__(self, message, signal):
        super().__init__(message)
        # The lease's sandbox ends with the judge's call, whatever became of the lease.
        self.signal = dict(signal, cleanup_required=False)


class SchedulerPreinputInfrastructureError(SidecarExecutionError):
    """The scheduler's own readiness failure before any candidate input, never stderr."""

    def __init__(self, signal):
        super().__init__('Scheduler pre-input readiness failed')
        self.signal = dict(signal)


def _ctl(*args):
    """The reply of `sandboxctl <args>`: a refusal is the platform's failure."""
    done = subprocess.run([SANDBOXCTL, *map(str, args)], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, check=False)
    try:
        reply = json.loads(done.stdout)
    except ValueError:
        reply = None
    if not isinstance(reply, dict) or reply.get('kind') == 'refused':
        raise PlatformError('sandboxctl ' + ' '.join(map(str, args[:2])) + ' failed',
                            {'cause': 'refused_op', 'op': str(args[1])})
    return reply


class PodSession:
    def __init__(self, runtime_config, mode, workload, *, identity, deadline, cleanup_seconds,
                 capture_seconds, evidence):
        self.config, self.mode = runtime_config, mode
        # The judge's session of the workload's name leases its parents.
        self.grant = workload
        self.variants = execution_variants(runtime_config)
        names = expected_sidecars(mode, variants=self.variants) if workload == 'swe' else (SCHEDULER,)
        self.sidecars = {'names': list(names)}
        # The run and attempt that the pod's services and the CPU controllers both name.
        self.identity = dict(identity)
        self.verification_id = identity['verification_id']
        self.deadline = deadline
        self.reserve_seconds = cleanup_seconds + capture_seconds
        self.evidence = Path(evidence)
        self.lease = None
        self.runner_id = None
        self.operations = 0
        self.ran = 0

    def _call(self, op, *args):
        return _ctl('call', self.grant + '.' + op, self.lease, *args)

    def _wait(self, done):
        """The first status of the lease that `done` accepts."""
        pause = 0.2
        while True:
            reply = self._call('status')
            if done(reply):
                return reply
            time.sleep(pause)
            pause = min(5.0, pause * 1.5)

    def _op(self, op, *args):
        """Start a command or a transfer, and return how it ended."""
        self._call(op, *args)
        number, self.ran = self.ran, self.ran + 1

        def done(reply):
            last = reply.get('last') or {}
            ours = last.get('command', last.get('transfer')) == number
            return reply['state'] != 'running' and (
                reply['state'] != 'ready' or (ours and last.get('end') is not None))
        reply = self._wait(done)
        if reply['state'] != 'ready':
            raise PlatformError('the parent ended during a ' + op,
                                {'cause': 'lease_' + reply['state'], 'op': op})
        return reply['last']['end']

    def _transfer(self, op, source, to, container):
        end = self._op(op, source, to, *([] if container is None else ['--container', container]))
        if 'moved' not in end:
            raise PlatformError('a ' + op + ' of the parent failed', {'cause': 'failed_transfer', 'op': op})

    def _command(self, argv, timeout, container):
        end = self._op('run', '--timeout', timeout,
                       *([] if container is None else ['--container', container]), '--', *argv)
        if end == 'unstarted':
            raise PlatformError('a command of the parent did not start', {'cause': 'unstarted_command', 'op': 'run'})
        return end

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise SidecarExecutionError('parent work deadline reached')
        return remaining

    def __enter__(self):
        self.evidence.mkdir(parents=True, exist_ok=False)
        self.lease = _ctl('call', self.grant + '.open')['lease']
        try:
            ready = self._wait(lambda reply: reply['state'] != 'starting')
            if ready['state'] != 'ready':
                raise PlatformError('the parent did not start', {'cause': 'lease_' + ready['state'], 'op': 'open'})
            # Work ends early enough to stop the server and collect what it left within the lease.
            self.deadline = min(self.deadline, time.monotonic() + ready['remaining']['instance_seconds']
                                - self.reserve_seconds)
            self.runner_id = '{}-{}'.format(self.grant, self.lease)
            # Every container of the pod reaches every other's ports, each caller from the same
            # address: the scheduler holds the lease's key, and each SWE service its own token,
            # both given before the candidate's server starts.
            key = secrets.token_hex(32)
            for name in self.sidecars['names']:
                profile = self.config['sidecars'][name]
                identity = {'run_id': self.identity['run_id'], 'attempt_id': self.identity['attempt_id'],
                            'operation_id': self.runner_id, 'phase': self.mode, 'name': name,
                            'profile_digest': canonical_digest(profile)}
                secret = (sidecar_env.KEY_ENV + '=' + key if name == SCHEDULER
                          else sidecar_env.TOKEN_ENV + '=' + sidecar_env.service_token(key, name))
                started = self._command(['/bin/sh', '-c', START, profile['workdir'], '/usr/bin/env',
                    'FRONTIER_SIDECAR_IDENTITY_JSON=' + json.dumps(identity, separators=(',', ':')),
                    secret, *profile['args']], math.ceil(self.remaining()), name)
                if started != {'exited': {'code': 0}}:
                    raise PlatformError('a service of the pod did not start',
                                        {'cause': 'unstarted_service', 'op': 'run', 'sidecar_id': name})
            ready = self._command([SCHEDULER_PYTHON, '-c', SCHEDULER_READY,
                                   'http://127.0.0.1:{}/ready'.format(SCHEDULER_PORT)],
                                  SCHEDULER_READY_SECONDS, SCHEDULER)
            if ready != {'exited': {'code': 0}}:
                raise PlatformError('the scheduler of the pod did not start',
                                    {'cause': 'unstarted_service', 'op': 'run', 'sidecar_id': SCHEDULER})
            (self.evidence / 'parent.json').write_text(json.dumps({'runner_id': self.runner_id,
                'phase': self.mode, 'sidecars': self.sidecars['names']}, indent=2) + '\n')
            return self
        except BaseException:
            self.close()
            raise

    def run(self, argv, *, stdin=b'', output, max_stdout_bytes, max_stderr_bytes,
            operation_deadline=None, on_stdout=None, container=None):
        """One command in the parent, or in `container`; never retry a possibly executed batch."""
        work_deadline = self.deadline if operation_deadline is None else min(self.deadline, operation_deadline)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        self.operations += 1
        remote = '{}/{:04d}'.format(WORK, self.operations)
        started = time.monotonic()
        failure = None
        try:
            (output / 'in').mkdir()
            (output / 'in/stdin').write_bytes(stdin)
            self._transfer('put', output / 'in', remote, container)
            shutil.rmtree(output / 'in')
            seconds = work_deadline - time.monotonic()
            if seconds <= 0:
                raise RemoteExecutionError('operation reached its absolute owner deadline', cause='deadline')
            end = self._command(['/bin/sh', '-c', COMMAND, remote, *argv], math.ceil(seconds), container)
            if end == 'timed_out':
                raise RemoteExecutionError('operation reached its absolute owner deadline', cause='deadline')
            self._transfer('get', remote + '.out', output / 'out', container)
            for name, limit in (('stdout', max_stdout_bytes), ('stderr', max_stderr_bytes)):
                (output / 'out' / name).replace(output / name)
                if (output / name).stat().st_size > limit:
                    raise RemoteExecutionError('output exceeded its publisher bound', cause='output_limit')
            if on_stdout is not None:
                with (output / 'stdout').open('rb') as stream:
                    for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
                        on_stdout(chunk)
            if end != {'exited': {'code': 0}}:
                raise RemoteExecutionError('batch execution failed', cause='child_exit')
        except BaseException as error:
            failure = type(error).__name__
            raise
        finally:
            (output / 'transport.json').write_text(json.dumps({'runner_id': self.runner_id,
                'phase': self.mode, 'container': container or 'main', 'operation': self.operations,
                'elapsed_seconds': time.monotonic() - started, 'error_type': failure,
                'owned_work_deadline': work_deadline, 'score_ready': False}, indent=2) + '\n')
        return output / 'stdout'

    def close(self):
        if self.lease is None:
            return
        self._call('close')
        self._wait(lambda reply: reply['state'] in ENDED)
        self.lease = None

    def __exit__(self, *exc):
        self.close()
