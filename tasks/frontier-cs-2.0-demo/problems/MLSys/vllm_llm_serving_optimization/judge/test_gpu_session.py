"""Real localhost CONNECT and subprocess tests, not Driver/Modal SSH acceptance."""
import base64
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parents[1] / 'verifier'))
import calibration_clock
from gpu_session import DriverGPU, GPUSession, connect_socket, validate_runner
from driver_client import SidecarExecutionError
from sidecar_catalog import SCHEDULER, expected_sidecars
from runtime_config import project_task_config
from test_runtime_config import synthetic_task


def gateway():
    raw = b'\0\0\0\x0bssh-ed25519\0\0\0\x20' + b'x' * 32
    return {'schema_version': 'frontier.ssh-gateway/v1', 'username': 'frontier',
        'authentication': 'connect_bearer', 'max_channels': 1,
        'server_host_public_key': 'ssh-ed25519 ' + base64.b64encode(raw).decode(),
        'server_host_key_sha256': 'SHA256:' + base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')}


class GPUWireTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []
        self.bad_gateway = False
        self.bad_cleanup = False
        self.retry_post = 0
        self.retry_delete = 0
        owner = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args):
                pass
            def send(self, code, value):
                raw = json.dumps(value).encode()
                self.send_response(code)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.calls.append(('create', self.path, self.headers.get('Idempotency-Key'), body))
                self.assert_auth()
                if owner.retry_post:
                    owner.retry_post -= 1
                    if owner.retry_post == 1:
                        self.connection.shutdown(socket.SHUT_RDWR)
                        self.connection.close()
                        return
                    self.send(425, {'code': 'idempotency_in_progress'})
                    return
                value = owner.runner()
                if owner.bad_gateway:
                    value['ssh_gateway']['server_host_key_sha256'] = 'SHA256:wrong'
                self.send(202, value)
            def assert_auth(self):
                if self.headers.get('Authorization') != 'Bearer synthetic-test-bearer':
                    raise AssertionError('test-only verification bearer absent')
            def do_DELETE(self):
                self.assert_auth()
                owner.calls.append(('delete', self.path))
                if owner.retry_delete:
                    owner.retry_delete -= 1
                    self.send(425, {'code': 'idempotency_in_progress'})
                    return
                self.send(200, {'runner_id': 'runner_other' if owner.bad_cleanup else 'runner_one', 'state': 'terminated'})
            def do_CONNECT(self):
                self.assert_auth()
                owner.calls.append(('connect', self.path))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'SSH-2.0-fixture\r\n')
                self.wfile.flush()
                self.close_connection = True
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.client = DriverGPU(f'http://127.0.0.1:{self.server.server_port}', 'verification_one',
            'synthetic-test-bearer', deadline=time.monotonic() + 10)
        task = synthetic_task()
        task['network']['audit']['max_ssh_bytes_per_direction'] = 100000
        self.config = project_task_config(task)

    def runner(self):
        sidecars = [{'name': name, 'state': 'ready'} for name in expected_sidecars('train')]
        sidecars[0]['attestation_public_key'] = 'ssh-ed25519 ' + 'A' * 68
        assert sidecars[0]['name'] == SCHEDULER
        return {'runner_id': 'runner_one', 'state': 'ready', 'gpu_count': 1, 'gpu_type':'H100','granted_seconds':10,
            'proxy_endpoint': self.client.url + '/runner_one/connect', 'ssh_gateway': gateway(),
            'sidecars': sidecars}

    def session(self):
        return GPUSession(self.client, self.config, 'train', gpu_seconds=10, request_key='batch_one',
            scratch=self.root/'scratch', evidence=self.root/'evidence', cleanup_deadline=time.monotonic()+12)

    def test_scoped_create_connect_fixed_hostkey_argv_and_exact_cleanup(self):
        with self.session() as session:
            with connect_socket(self.client, session.runner_id) as connection:
                self.assertEqual(connection.recv(200), b'SSH-2.0-fixture\r\n')
            self.assertIn('StrictHostKeyChecking=yes', session.ssh)
            self.assertEqual(session.ssh[-1], 'frontier@gpu-runner')
            self.assertNotIn('synthetic-test-bearer', ' '.join(session.ssh))
            self.assertEqual((session.scratch/'known_hosts').read_text(), 'gpu-runner '+gateway()['server_host_public_key']+'\n')
        self.assertEqual([row[0] for row in self.calls], ['create', 'connect', 'delete'])
        self.assertEqual(self.calls[0][3], {'gpu_type': 'H100', 'gpu_count': 1, 'max_duration_seconds': 10})
        self.assertEqual(json.loads((self.root/'evidence/cleanup.json').read_text())['state'], 'terminated')

    def test_foreign_gateway_and_fingerprint_rejected_and_accepted_parent_cleaned(self):
        for field, value in [('proxy_endpoint', 'http://foreign/connect'), ('gpu_count', 2)]:
            runner = self.runner()
            runner[field] = value
            with self.assertRaises(SidecarExecutionError):
                validate_runner(runner, self.client)
        self.bad_gateway = True
        with self.assertRaises(SidecarExecutionError):
            with self.session():
                self.fail('invalid host key accepted')
        self.assertEqual([row[0] for row in self.calls], ['create', 'delete'])

    def test_real_process_io_byte_bounds_and_no_execution_retry(self):
        with self.session() as session:
            # This replaces SSH solely to exercise real bounded local pipes.
            session.ssh = [sys.executable, '-c', 'import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data); sys.stderr.write("stderr")']
            path = session.run(['fixture-only'], stdin=b'complete batch input', output=self.root/'io',
                               max_stdout_bytes=100, max_stderr_bytes=100)
            self.assertEqual(path.read_bytes(), b'complete batch input')
            self.assertEqual((self.root/'io/stderr').read_bytes(), b'stderr')
            session.ssh = [sys.executable, '-c', 'import sys; sys.stdout.write("x"*10000)']
            with self.assertRaises(SidecarExecutionError):
                session.run(['fixture-only'], output=self.root/'overflow', max_stdout_bytes=100, max_stderr_bytes=100)
            self.assertEqual((self.root/'overflow/stdout').stat().st_size, 100)
            self.assertEqual(session.operations, 2)
            self.assertFalse(json.loads((self.root/'overflow/transport.json').read_text())['score_ready'])

    def test_deadline_kills_process_and_cleanup_retains_own_deadline(self):
        with self.session() as session:
            session.ssh = [sys.executable, '-c', 'import time; time.sleep(20)']
            self.client.deadline = time.monotonic() + 0.1
            before = time.monotonic()
            with self.assertRaises(SidecarExecutionError):
                session.run(['fixture-only'], output=self.root/'timeout', max_stdout_bytes=100, max_stderr_bytes=100)
            self.assertLess(time.monotonic()-before, 2)
        self.assertEqual(self.calls[-1][0], 'delete')

    def test_incremental_observer_receives_output_before_remote_exit(self):
        with self.session() as session:
            session.ssh = [sys.executable, '-c',
                'import os,time;os.write(1,b"first");time.sleep(.1);os.write(1,b"second")']
            observed = []
            path = session.run(['fixture-only'], output=self.root/'incremental',
                max_stdout_bytes=100, max_stderr_bytes=100,
                on_stdout=lambda chunk: observed.append((time.monotonic(), chunk)))
            self.assertEqual(b''.join(chunk for _, chunk in observed), b'firstsecond')
            self.assertEqual(path.read_bytes(), b'firstsecond')
            self.assertGreaterEqual(len(observed), 2)
            self.assertGreater(observed[-1][0] - observed[0][0], .05)

    def test_dribbling_output_cannot_extend_operation_deadline(self):
        with self.session() as session:
            session.ssh = [sys.executable, '-c',
                'import os,time;\nwhile True:os.write(1,b"x");time.sleep(.01)']
            started = time.monotonic()
            with self.assertRaises(SidecarExecutionError):
                session.run(['fixture-only'], output=self.root/'dribble',
                    max_stdout_bytes=1000, max_stderr_bytes=100,
                    operation_deadline=started + .12)
            self.assertLess(time.monotonic() - started, 1)
            receipt = json.loads((self.root/'dribble/transport.json').read_text())
            self.assertEqual(receipt['cleanup_errors'], [])
            self.assertGreater(receipt['bytes']['stdout'], 0)

    def test_observer_persistence_defect_propagates_without_success(self):
        with self.session() as session:
            session.ssh = [sys.executable, '-c', 'import os;os.write(1,b"record")']
            def failed_observer(chunk):
                raise OSError('injected trusted ledger fsync defect')
            with self.assertRaises(OSError):
                session.run(['fixture-only'], output=self.root/'observer-failure',
                    max_stdout_bytes=100, max_stderr_bytes=100, on_stdout=failed_observer)
            self.assertEqual(json.loads((self.root/'observer-failure/transport.json').read_text())
                             ['error_type'], 'OSError')

    def test_lost_create_then_in_progress_replays_same_key_and_retries_cleanup(self):
        self.retry_post, self.retry_delete = 2, 1
        with self.session():
            pass
        creates = [row for row in self.calls if row[0] == 'create']
        self.assertEqual(len(creates), 3)
        self.assertEqual(len({row[2] for row in creates}), 1)
        self.assertEqual(len([row for row in self.calls if row[0] == 'delete']), 2)
        self.assertEqual(json.loads((self.root/'evidence/cleanup.json').read_text())['state'], 'terminated')

    def calibration_session(self):
        self.client.deadline = time.monotonic() + 30
        clock = calibration_clock.begin({
            'driver_origin': self.client.driver_url, 'verification_id': self.client.verification_id,
            'request_key': 'batch_one',
            'request': {'gpu_type': 'H100', 'gpu_count': 1, 'max_duration_seconds': 20},
            'config_sha256': 'sha256:' + 'a' * 64, 'plan_sha256': 'sha256:' + 'b' * 64},
            self.root/'clock.json', cap_seconds=30, cleanup_seconds=2, capture_seconds=2)
        session = GPUSession(self.client, self.config, 'train', gpu_seconds=20, request_key='batch_one',
            scratch=self.root/'scratch', evidence=self.root/'evidence',
            cleanup_deadline=time.monotonic()+35, calibration_clock=clock)
        return session, clock

    def assert_grant_bound_on_failure(self, stage):
        session, clock = self.calibration_session()
        # Real localhost Driver returns10s against the20s request. The shorter
        # cutoff must be installed before either injected local write failure.
        expected = clock.preview_grant('runner_one', 10)
        deletes = []
        request = self.client.request
        def observed_request(method, suffix, **kwargs):
            if method == 'DELETE':
                deletes.append(self.client.deadline)
            return request(method, suffix, **kwargs)
        save = session._save
        def failed_save(name, value):
            if name == 'parent.json':
                raise OSError('synthetic parent evidence write failure')
            return save(name, value)
        failure = (patch.object(session, '_save', side_effect=failed_save) if stage == 'parent'
                   else patch.object(calibration_clock.os, 'fsync', side_effect=OSError('synthetic marker fsync failure')))
        with patch.object(self.client, 'request', side_effect=observed_request), failure:
            with self.assertRaises(OSError):
                with session:
                    self.fail('unpersisted grant allowed batch execution')
        self.assertEqual(deletes, [expected['delete_deadline']])
        self.assertEqual(session.client.deadline, expected['work_deadline'])
        self.assertEqual(session.capture_deadline, expected['capture_deadline'])
        self.assertEqual([row[0] for row in self.calls], ['create', 'delete'])
        self.assertIsNone(clock.granted_seconds)
        self.assertEqual(json.loads((self.root/'evidence/cleanup.json').read_bytes())['state'], 'terminated')

    def test_parent_evidence_failure_uses_observed_short_grant_cleanup_deadline(self):
        self.assert_grant_bound_on_failure('parent')

    def test_marker_fsync_failure_uses_observed_short_grant_cleanup_deadline(self):
        self.assert_grant_bound_on_failure('marker')

    def test_short_grant_acceptance_persists_identity_and_applies_all_cutoffs(self):
        session, clock = self.calibration_session()
        expected = clock.preview_grant('runner_one', 10)
        with session:
            self.assertEqual(session.client.deadline, expected['work_deadline'])
        self.assertEqual(clock.granted_seconds, 10)
        self.assertEqual(clock.runner_id, 'runner_one')
        self.assertEqual(session.cleanup_deadline, expected['delete_deadline'])
        self.assertEqual(session.capture_deadline, expected['capture_deadline'])
        self.assertEqual([row[0] for row in self.calls], ['create', 'delete'])

    def test_uncertain_cleanup_cannot_be_reported_as_success(self):
        self.bad_cleanup = True
        with self.assertRaises(SidecarExecutionError):
            with self.session():
                pass
        self.assertTrue(json.loads((self.root/'evidence/cleanup-error.json').read_text())['cleanup_uncertain'])


if __name__ == '__main__':
    unittest.main()
