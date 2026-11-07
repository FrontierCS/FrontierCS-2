"""Real localhost relay against a fake scheduler; no Driver, SSH or Modal."""
import io
import json
import threading
from types import SimpleNamespace
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import scheduler_relay as relay


class FakeScheduler:
    def __init__(self):
        self.jobs = {}
        self.submits = []
        self.polls = 0
        self.ready_after = 0
        self.ready_calls = 0
        self.receipts = {('candidate', 3): b'{"receipt": "three"}'}
        self.complete_after_polls = 2
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body):
                raw = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == '/ready':
                    owner.ready_calls += 1
                    self.reply(200, {'ready': owner.ready_calls > owner.ready_after})
                elif self.path.startswith('/jobs/'):
                    owner.polls += 1
                    job = owner.jobs.get(int(self.path.rsplit('/', 1)[1]))
                    if job is None:
                        self.reply(404, {'error': 'unknown job'})
                        return
                    if owner.polls >= owner.complete_after_polls:
                        job['state'] = 'complete'
                        job['signed'] = {'fake': True}
                    self.reply(200, job)
                elif self.path.startswith('/receipts/'):
                    _, _, variant, index = self.path.split('/')
                    raw = owner.receipts.get((variant, int(index)))
                    self.reply(200 if raw else 404, raw or {'error': 'unknown receipt'})
                else:
                    self.reply(404, {'error': 'unknown'})

            def do_POST(self):
                raw = self.rfile.read(int(self.headers['Content-Length']))
                owner.submits.append(raw)
                document = json.loads(raw)
                digest = 'sha256:' + __import__('hashlib').sha256(raw).hexdigest()
                job = owner.jobs.setdefault(document['sequence'], {'sequence': document['sequence'],
                                                                   'request_sha256': digest, 'state': 'running'})
                self.reply(202, job)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = FakeScheduler()
        self.addCleanup(self.scheduler.close)
        self.url_patch = patch.object(relay, 'SCHEDULER', self.scheduler.url)
        self.url_patch.start()
        self.addCleanup(self.url_patch.stop)
        self.poll_patch = patch.object(relay, 'POLL_SECONDS', 0.01)
        self.poll_patch.start()
        self.addCleanup(self.poll_patch.stop)

    def run_relay(self, action, raw, **overrides):
        argv = [action, '--max-input-bytes', str(overrides.get('input', len(raw))),
                '--max-output-bytes', str(overrides.get('output', 100000)),
                '--timeout-seconds', str(overrides.get('timeout', 5))]
        output = io.BytesIO()
        # The relay touches only the binary buffers of its standard streams.
        with patch.object(relay.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(raw))), \
                patch.object(relay.sys, 'stdout', SimpleNamespace(buffer=output)):
            relay.main(argv)
        return output.getvalue()

    def test_job_waits_for_readiness_submits_once_and_returns_the_complete_view(self):
        self.scheduler.ready_after = 2
        raw = json.dumps({'sequence': 4, 'kind': 'batch'}).encode()
        data = self.run_relay('job', raw)
        view = json.loads(data)
        self.assertEqual((view['state'], view['sequence']), ('complete', 4))
        self.assertEqual(view['signed'], {'fake': True})
        self.assertEqual(self.scheduler.submits, [raw])
        self.assertGreaterEqual(self.scheduler.ready_calls, 3)

    def test_resubmitting_the_same_job_is_idempotent_and_other_requests_are_refused(self):
        raw = json.dumps({'sequence': 1}).encode()
        self.run_relay('job', raw)
        self.scheduler.complete_after_polls = 0
        self.run_relay('job', raw)
        self.assertEqual(len(self.scheduler.submits), 2)
        self.assertEqual(len(self.scheduler.jobs), 1)
        other = json.dumps({'sequence': 1, 'changed': True}).encode()
        with self.assertRaises(relay.RelayError):
            self.run_relay('job', other)

    def test_bounds_and_deadline_fail_closed(self):
        raw = json.dumps({'sequence': 2}).encode()
        with self.assertRaises(relay.RelayError):
            self.run_relay('job', raw, input=len(raw) - 1)
        self.scheduler.complete_after_polls = 10 ** 6
        with self.assertRaises(relay.RelayError):
            self.run_relay('job', raw, timeout=1)
        self.scheduler.complete_after_polls = 0
        with self.assertRaises(relay.RelayError):
            self.run_relay('job', raw, output=10)
        with self.assertRaises(relay.RelayError):
            self.run_relay('job', json.dumps({'sequence': -1}).encode())

    def test_receipt_fetch_returns_raw_bytes_or_fails(self):
        data = self.run_relay('receipt', json.dumps({'variant': 'candidate', 'index': 3}).encode())
        self.assertEqual(data, b'{"receipt": "three"}')
        with self.assertRaises(relay.RelayError):
            self.run_relay('receipt', json.dumps({'variant': 'candidate', 'index': 4}).encode())
        with self.assertRaises(relay.RelayError):
            self.run_relay('receipt', json.dumps({'variant': 'other', 'index': 3}).encode())


if __name__ == '__main__':
    unittest.main()


class BoundedDiagnosticTests(unittest.TestCase):
    def test_readiness_stage_is_diagnostic_only_and_body_fields_are_dropped(self):
        failure = {'stage': 'preinput_sidecar_ready', 'sidecar_id': 'swe-test-000-vanilla',
                   'candidate_input_forwarded': False}
        value = json.loads(relay.error_diagnostic(relay.RelayError('job_failed', failure=failure)))
        self.assertEqual(value['failure'], failure)
        value = json.loads(relay.error_diagnostic(relay.RelayError('job_failed',
                            failure={**failure, 'body': 'SECRET'})))
        self.assertNotIn('failure', value)
        self.assertNotIn('SECRET', str(value))

    def test_stream_http_rejection_retains_status_without_body(self):
        for status in (400, 409, 502):
            with patch.object(relay, 'wait_ready'), patch.object(relay, '_request', return_value=(status, b'SECRET')):
                with self.assertRaises(relay.RelayError) as caught:
                    relay.job_stream(b'{"sequence":0}', 999999999, 10000, emit=lambda data: None)
            diagnostic = json.loads(relay.error_diagnostic(caught.exception))
            self.assertEqual(diagnostic['reason'], 'scheduler_rejected_streamed_job')
            self.assertEqual(diagnostic['http_status'], status)
            self.assertNotIn('SECRET', str(diagnostic))

    def test_stream_failed_state_and_owner_deadline_are_distinct(self):
        raw = b'{"sequence":0}'
        digest = 'sha256:' + __import__('hashlib').sha256(raw).hexdigest()
        for state, reason in [('failed', 'scheduler_streamed_job_failed'),
                              ('running', 'scheduler_stream_reached_owner_deadline')]:
            view = json.dumps(dict(sequence=0, request_sha256=digest, event_count=0, event=None, state=state)).encode()
            with patch.object(relay, 'wait_ready'), patch.object(relay, '_request', side_effect=[(202,b''),(200,view)]):
                with self.assertRaises(relay.RelayError) as caught:
                    relay.job_stream(raw, 0, 10000, emit=lambda data: None)
            self.assertEqual(json.loads(relay.error_diagnostic(caught.exception))['reason'], reason)

    def test_unknown_exception_and_untrusted_fields_are_bounded(self):
        secret = 'credential-' * 10000
        for error in [RuntimeError(secret), ValueError(secret), OSError(secret),
                      relay.RelayError(secret, http_status=secret, state=secret)]:
            diagnostic = relay.error_diagnostic(error)
            self.assertLess(len(diagnostic.encode()), 256)
            self.assertNotIn('credential', diagnostic)
