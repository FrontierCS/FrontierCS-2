"""One scoped Driver GPU parent; bounded, host-key-pinned whole-batch SSH.

CONNECT and the non-root NSS adaptation follow Preview's native SVG adapter.
Only the existing verification bearer reaches the CPU proxy process. Nothing
here provisions through a provider or puts a credential in a remote command.
The Driver starts the TaskConfig's declared sidecars with the parent and lists
them on the runner view; the view's scheduler public key is the trust anchor
for every piece of evidence that later crosses the untrusted parent.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import selectors
import shlex
import signal
import socket
import ssl
import subprocess
import sys
import time
import threading
from urllib.parse import urlparse

from driver_client import (AuthenticatedProvisioningError, DriverClient, DriverWorkIntervalError,
                           RemoteExecutionError, SidecarExecutionError, _component)
from runtime_config import phase_limits, execution_variants
from sidecar_catalog import SCHEDULER, expected_sidecars, VARIANTS

ENDED_STATES = frozenset({'terminated', 'failed', 'expired', 'releasing'})


class DriverGPU(DriverClient):
    """The Verifier's GPU-runner routes; sidecars have no route of their own."""


def validate_sidecars(runner, mode, *, variants=VARIANTS):
    """Bind the Driver-listed declared sidecars to this phase's fixed catalog."""
    listed = runner.get('sidecars')
    if not isinstance(listed, list):
        raise SidecarExecutionError('Driver runner view lists no declared sidecars')
    rows = {}
    for row in listed:
        if not isinstance(row, dict) or not isinstance(row.get('name'), str) or row['name'] in rows:
            raise SidecarExecutionError('Driver sidecar view is malformed or repeats a name')
        rows[row['name']] = row
    if set(rows) != set(expected_sidecars(mode, variants=variants)):
        raise SidecarExecutionError('Driver did not start exactly the declared sidecars of this phase')
    key = rows[SCHEDULER].get('attestation_public_key')
    if not isinstance(key, str) or not key.startswith('ssh-ed25519 ') or '\n' in key.strip():
        raise SidecarExecutionError('scheduler sidecar lacks its Driver attestation public key')
    for name, row in rows.items():
        if name != SCHEDULER and row.get('attestation_public_key') is not None:
            raise SidecarExecutionError('a non-scheduler sidecar received an attestation key')
        if row.get('state') in ENDED_STATES:
            raise SidecarExecutionError('a declared sidecar ended before the phase started')
    return {'public_key': key.strip(), 'names': sorted(rows),
            'states': {name: row.get('state') for name, row in sorted(rows.items())}}


def validate_runner(runner, client):
    ident = _component(runner.get('runner_id'))
    endpoint = client.url + '/' + ident + '/connect'
    gateway = runner.get('ssh_gateway')
    if (runner.get('state') != 'ready' or runner.get('gpu_count') != 1 or runner.get('gpu_type') != 'H100'
            or runner.get('proxy_endpoint') != endpoint or not isinstance(gateway, dict)
            or gateway.get('schema_version') != 'frontier.ssh-gateway/v1'
            or gateway.get('username') != 'frontier'
            or gateway.get('authentication') != 'connect_bearer'
            or gateway.get('max_channels') != 1):
        raise SidecarExecutionError('Driver parent or SSH gateway differs from its owned one-H100 contract')
    key = gateway.get('server_host_public_key')
    try:
        parts = key.split()
        if len(parts) != 2 or '\n' in key or '\r' in key:
            raise ValueError()
        raw = base64.b64decode(parts[1], validate=True)
        length = int.from_bytes(raw[:4], 'big')
        if not length or raw[4:4 + length].decode('ascii') != parts[0]:
            raise ValueError()
        fingerprint = 'SHA256:' + base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
        if gateway.get('server_host_key_sha256') != fingerprint:
            raise ValueError()
    except (AttributeError, TypeError, ValueError, UnicodeError):
        raise SidecarExecutionError('Driver SSH public key and fingerprint do not agree') from None
    return ident


def ssh_environment(scratch):
    """Support the platform-assigned non-root UID without changing the image."""
    environment = dict(os.environ)
    for key in ('LD_PRELOAD', 'NSS_WRAPPER_PASSWD', 'NSS_WRAPPER_GROUP'):
        environment.pop(key, None)
    environment['HOME'] = str(scratch)
    try:
        pwd.getpwuid(os.getuid())
    except KeyError:
        libraries = tuple(Path('/usr/lib').glob('*/libnss_wrapper.so'))
        if len(libraries) != 1:
            raise SidecarExecutionError('non-root SSH NSS adapter is unavailable')
        for name, extra in (
            ('passwd', f'frontier-runtime:x:{os.getuid()}:{os.getgid()}:Verifier:{scratch}:/usr/sbin/nologin'),
            ('group', f'frontier-runtime:x:{os.getgid()}:')):
            target = scratch / name
            target.write_text(Path('/etc', name).read_text().rstrip('\n') + '\n' + extra + '\n')
            target.chmod(0o600)
            environment['NSS_WRAPPER_' + name.upper()] = str(target)
        environment['LD_PRELOAD'] = str(libraries[0])
    return environment


class GPUSession:
    def __init__(self, client, runtime_config, mode, *, gpu_seconds, request_key,
                 scratch, evidence, cleanup_deadline, calibration_clock=None,
                 provisioning_owner=None):
        self.limits = phase_limits(runtime_config, mode, requested_gpu_seconds=gpu_seconds)
        self.variants = execution_variants(runtime_config)
        if not math.isfinite(cleanup_deadline) or cleanup_deadline < client.deadline:
            raise SidecarExecutionError('cleanup must have an explicit remaining owner deadline')
        self.client, self.mode = client, mode
        self.key = _component(request_key)
        self.scratch, self.evidence = Path(scratch), Path(evidence)
        self.cleanup_deadline = cleanup_deadline
        self.calibration_clock = calibration_clock
        self.capture_deadline = None
        self.runner = None
        self.runner_id = None
        self.sidecars = None
        self.operations = 0
        self.create_started = False
        self.provisioning_owner = provisioning_owner
        self.create_failure = None

    def _save(self, name, value):
        (self.evidence / name).write_text(json.dumps(value, indent=2) + '\n')

    def __enter__(self):
        self.scratch.mkdir(parents=True, exist_ok=False)
        self.scratch.chmod(0o700)
        self.evidence.mkdir(parents=True, exist_ok=False)
        self.client.enable_control_diagnostics(self.evidence / 'driver-control.json')
        try:
            if self.provisioning_owner is not None:
                (self.evidence / 'provisioning-plan.json').write_bytes(self.provisioning_owner.plan_bytes)
                (self.evidence / 'provisioning-slots.json').write_bytes(self.provisioning_owner.slot_bytes)
                self._save('provisioning-claim.json', self.provisioning_owner.claim)
            self.create_started = True
            ownership = ({'provisioning_owner': self.provisioning_owner}
                         if self.provisioning_owner is not None else {})
            try:
                self.runner = self.client.request('POST', '', key=self.key,
                    body={'gpu_type': 'H100', 'gpu_count': 1,
                          'max_duration_seconds': self.limits['gpu_seconds']}, **ownership)
            except AuthenticatedProvisioningError as error:
                self.create_failure = error.provenance
                (self.evidence / 'driver-platform-failure.json').write_bytes(self.create_failure.encoded)
                raise
            # Retain an accepted identity before validating the rest, so a bad
            # gateway response still terminates exactly that owned resource.
            self.runner_id = _component(self.runner.get('runner_id'))
            grant = self.runner.get('granted_seconds')
            if type(grant) is not int or not 0 < grant <= self.limits['gpu_seconds']:
                raise SidecarExecutionError('Driver returned an invalid or oversized phase lifetime')
            if self.calibration_clock is not None:
                # Bound cleanup immediately: either evidence writes or marker
                # fsync can fail after a valid shorter grant has been observed.
                cutoffs = self.calibration_clock.preview_grant(self.runner_id, self.runner.get('granted_seconds'))
                self.client.deadline = min(self.client.deadline, cutoffs['work_deadline'])
                self.cleanup_deadline = min(self.cleanup_deadline, cutoffs['delete_deadline'])
                self.capture_deadline = cutoffs['capture_deadline']
            self._save('parent.json', self.runner)
            if self.calibration_clock is not None:
                cutoffs = self.calibration_clock.accept_grant(self.runner_id, self.runner.get('granted_seconds'))
                self.client.deadline = min(self.client.deadline, cutoffs['work_deadline'])
                self.cleanup_deadline = min(self.cleanup_deadline, cutoffs['delete_deadline'])
                self.capture_deadline = min(self.capture_deadline, cutoffs['capture_deadline'])
                self._save('calibration-cutoffs.json', cutoffs)
                if time.monotonic() >= self.client.deadline:
                    raise DriverWorkIntervalError(phase=self.mode, granted_seconds=grant)
            validate_runner(self.runner, self.client)
            if grant < self.limits['gpu_seconds'] and self.calibration_clock is None:
                raise SidecarExecutionError('a shorter Driver grant requires its bound calibration clock')
            self.sidecars = validate_sidecars(self.runner, self.mode,
                                             variants=self.variants)
            self._save('sidecars.json', self.sidecars)
            self.client.remaining()
            known = self.scratch / 'known_hosts'
            known.write_text('gpu-runner ' + self.runner['ssh_gateway']['server_host_public_key'] + '\n')
            known.chmod(0o600)
            self.environment = ssh_environment(self.scratch)
            self.environment.update({'FRONTIER_DRIVER_INTERNAL_URL': self.client.driver_url,
                'FRONTIER_VERIFICATION_ID': self.client.verification_id,
                'FRONTIER_VERIFIER_TOKEN': self.client._token,
                'VLLM_CONNECT_DEADLINE': str(self.client.deadline)})
            proxy = shlex.join([sys.executable, str(Path(__file__).resolve()), '_connect', self.runner_id])
            self.ssh = ['/usr/bin/ssh', '-F', '/dev/null', '-T',
                '-o', 'StrictHostKeyChecking=yes', '-o', 'GlobalKnownHostsFile=/dev/null',
                '-o', 'UserKnownHostsFile=' + str(known), '-o', 'HostKeyAlias=gpu-runner',
                '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'IdentityFile=none',
                '-o', 'ClearAllForwardings=yes', '-o', 'PermitLocalCommand=no',
                '-o', 'ProxyCommand=' + proxy, 'frontier@gpu-runner']
            return self
        except BaseException:
            self.close()
            raise

    def run(self, argv, *, stdin=b'', output, max_stdout_bytes, max_stderr_bytes,
            operation_deadline=None, on_stdout=None):
        """One actual SSH operation; never retry a possibly executed batch."""
        if (not isinstance(argv, list) or not argv
                or any(not isinstance(arg, str) or '\0' in arg for arg in argv)
                or not isinstance(stdin, bytes)):
            raise SidecarExecutionError('invalid fixed whole-batch command or stdin')
        if (any(type(n) is not int or n <= 0 for n in (max_stdout_bytes, max_stderr_bytes))
                or max_stdout_bytes + max_stderr_bytes > self.limits['ssh_bytes_per_direction']
                or len(stdin) > self.limits['ssh_bytes_per_direction']):
            raise SidecarExecutionError('SSH operation exceeds its publisher direction bound')
        if self.operations >= self.limits['ssh_sessions']:
            raise SidecarExecutionError('Verifier phase SSH pool exhausted')
        self.client.remaining()
        work_deadline = self.client.deadline
        if operation_deadline is not None:
            if (isinstance(operation_deadline, bool) or not isinstance(operation_deadline, (int, float))
                    or not math.isfinite(operation_deadline)):
                raise SidecarExecutionError('finite owned operation deadline required')
            work_deadline = min(work_deadline, operation_deadline)
        if on_stdout is not None and not callable(on_stdout):
            raise SidecarExecutionError('trusted incremental stdout observer required')
        def remaining():
            seconds = work_deadline - time.monotonic()
            if seconds <= 0:
                raise RemoteExecutionError('SSH operation reached its absolute owner deadline', cause='deadline')
            return min(self.client.remaining(), seconds)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        self.operations += 1
        counts = {'stdout': 0, 'stderr': 0, 'stdin': 0}
        process = None
        failure = None
        cleanup_errors = []
        observed_exit = None
        started = time.monotonic()
        try:
            process = subprocess.Popen(self.ssh + [shlex.join(argv)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.environment, start_new_session=True)
            with selectors.DefaultSelector() as selector, (output/'stdout').open('wb') as out, (output/'stderr').open('wb') as err:
                streams = {'stdout': (out, max_stdout_bytes), 'stderr': (err, max_stderr_bytes)}
                for stream, name in ((process.stdout, 'stdout'), (process.stderr, 'stderr')):
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, name)
                if stdin:
                    os.set_blocking(process.stdin.fileno(), False)
                    selector.register(process.stdin, selectors.EVENT_WRITE, 'stdin')
                else:
                    process.stdin.close()
                while selector.get_map() or observed_exit is None:
                    for key, _ in selector.select(min(0.2, remaining())):
                        if key.data == 'stdin':
                            try:
                                size = os.write(key.fd, stdin[counts['stdin']:counts['stdin'] + 65536])
                            except BrokenPipeError:
                                selector.unregister(key.fileobj)
                                key.fileobj.close()
                                continue
                            counts['stdin'] += size
                            if counts['stdin'] == len(stdin):
                                selector.unregister(key.fileobj)
                                key.fileobj.close()
                        else:
                            chunk = os.read(key.fd, 65536)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            stream, limit = streams[key.data]
                            available = max(0, limit - counts[key.data])
                            stream.write(chunk[:available])
                            counts[key.data] += len(chunk)
                            if counts[key.data] > limit:
                                raise RemoteExecutionError('SSH output exceeded its publisher bound', cause='output_limit')
                            if key.data == 'stdout' and on_stdout is not None:
                                on_stdout(chunk)
                    observation = os.waitid(os.P_PID, process.pid,
                                            os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    if observation is not None:
                        observed_exit = (observation.si_status if observation.si_code == os.CLD_EXITED
                                         else -observation.si_status)
                if observed_exit != 0 or counts['stdin'] != len(stdin):
                    raise RemoteExecutionError('SSH batch execution failed or did not consume complete input',
                                               cause='child_exit')
        except BaseException as error:
            failure = type(error).__name__
            raise
        finally:
            if process is not None:
                # Keep the session leader unreaped until every group signal is
                # sent. A successful SSH leader can still leave pipe holders.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=max(0, self.cleanup_deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    cleanup_errors.append('process_reap_deadline')
                for stream in (process.stdin, process.stdout, process.stderr):
                    try:
                        stream.close()
                    except Exception:
                        cleanup_errors.append('pipe_close')
            (output/'transport.json').write_text(json.dumps({'runner_id': self.runner_id,
                'phase': self.mode, 'ssh_operation': self.operations, 'bytes': counts,
                'elapsed_seconds': time.monotonic() - started,
                'exit_code': None if process is None else process.returncode,
                'error_type': failure, 'cleanup_errors': cleanup_errors,
                'owned_work_deadline': work_deadline, 'score_ready': False}, indent=2) + '\n')
        if cleanup_errors:
            raise SidecarExecutionError('SSH cleanup exceeded its independently owned reserve')
        return output / 'stdout'

    def close(self):
        if self.runner_id is None:
            if self.create_started and self.runner is None and self.create_failure is None:
                self._save('cleanup-error.json', {'cleanup_uncertain': True,
                    'owner': 'Driver lifecycle', 'reason': 'create response unavailable',
                    'request_key': self.key})
            return
        old_deadline = self.client.deadline
        try:
            self.client.deadline = self.cleanup_deadline
            cleanup = self.client.request('DELETE', '/' + self.runner_id)
            self._save('cleanup.json', cleanup)
            if cleanup.get('runner_id') != self.runner_id or cleanup.get('state') != 'terminated':
                raise SidecarExecutionError('owned GPU parent termination is unconfirmed')
            self.runner_id = None
        except BaseException:
            self._save('cleanup-error.json', {'cleanup_uncertain': True, 'owner': 'Driver lifecycle'})
            raise
        finally:
            self.client.deadline = old_deadline

    def __exit__(self, kind, value, traceback):
        self.close()


def connect_socket(client, runner_id):
    """CONNECT never follows a response-supplied provider address."""
    parsed = urlparse(client.driver_url)
    host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80)
    connection = socket.create_connection((host, port), timeout=client.remaining())
    try:
        if parsed.scheme == 'https':
            connection = ssl.create_default_context().wrap_socket(connection, server_hostname=host)
        authority = '[' + host + ']' if ':' in host else host
        path = urlparse(client.url).path + '/' + _component(runner_id) + '/connect'
        connection.sendall((f'CONNECT {path} HTTP/1.1\r\nHost: {authority}:{port}\r\n'
                            f'Authorization: Bearer {client._token}\r\n\r\n').encode('ascii'))
        header = bytearray()
        while not header.endswith(b'\r\n\r\n'):
            if len(header) >= 8192:
                raise SidecarExecutionError('Driver CONNECT headers exceeded protocol bound')
            connection.settimeout(client.remaining())
            byte = connection.recv(1)
            if not byte:
                raise SidecarExecutionError('Driver closed CONNECT before upgrade')
            header.extend(byte)
        if not bytes(header).split(b'\r\n', 1)[0].startswith(b'HTTP/1.1 200 '):
            raise SidecarExecutionError('Driver refused owned GPU CONNECT')
        return connection
    except BaseException:
        connection.close()
        raise


def pump(connection, client):
    # Independent directions preserve SSH flow control under simultaneous large
    # stdin/stdout. Blocking SSL recv handles its decrypted pending buffer too.
    done = threading.Event()
    errors = []

    def stop():
        done.set()
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def receive():
        try:
            while not done.is_set():
                chunk = connection.recv(65536)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
        except Exception as error:
            errors.append(type(error).__name__)
        finally:
            stop()

    connection.settimeout(client.remaining())
    timer = threading.Timer(client.remaining(), stop)
    timer.daemon = True
    thread = threading.Thread(target=receive, daemon=True)
    timer.start()
    thread.start()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(sys.stdin.buffer, selectors.EVENT_READ)
            while not done.is_set():
                for key, _ in selector.select(min(0.2, client.remaining())):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(sys.stdin.buffer)
                        connection.shutdown(socket.SHUT_WR)
                    else:
                        connection.sendall(chunk)
        if errors:
            raise SidecarExecutionError('Driver SSH stream failed')
        client.remaining()
    finally:
        timer.cancel()
        stop()


if __name__ == '__main__':
    try:
        if len(sys.argv) != 3 or sys.argv[1] != '_connect':
            raise SidecarExecutionError('private SSH proxy invocation required')
        deadline = float(os.environ['VLLM_CONNECT_DEADLINE'])
        if not math.isfinite(deadline):
            raise ValueError()
        client = DriverGPU(os.environ['FRONTIER_DRIVER_INTERNAL_URL'],
            os.environ['FRONTIER_VERIFICATION_ID'], os.environ['FRONTIER_VERIFIER_TOKEN'], deadline=deadline)
        with connect_socket(client, sys.argv[2]) as connection:
            pump(connection, client)
    except Exception:
        # Do not print environment, request headers, credentials or raw errors.
        print('vLLM GPU proxy transport failed', file=sys.stderr)
        raise SystemExit(1)
