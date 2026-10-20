"""Real local HTTP hangs and byte dribbling cannot hold the trusted broker."""
import deployment_test_guard  # noqa: F401
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import unittest

JUDGE = Path(__file__).resolve().parents[1] / 'tasks/frontier-cs-2.0-demo/problems/MLSys/vllm_llm_serving_optimization/judge'
sys.path.insert(0, str(JUDGE))
from bounded_http import RequestFailure, request_signed
from terminal_contract import Slot


class BoundedHTTPTests(unittest.TestCase):
    def setUp(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                try:
                    if self.path == '/dribble':
                        for _ in range(100):
                            self.wfile.write(b' ')
                            self.wfile.flush()
                            time.sleep(.02)
                    elif self.path == '/oversize':
                        self.wfile.write(b'x' * 10000)
                    elif self.path == '/bad':
                        self.wfile.write(b'not-json')
                    else:
                        self.wfile.write(json.dumps({'identity': 'fixed', 'ok': True}).encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.slot = Slot('run', 'attempt', 'verification', 'operation', 'eval', 'swe/public',
            'candidate', 'case', 'sha256:' + 'a' * 64, 'sha256:' + 'b' * 64)

    def request(self, path, seconds=1):
        return request_signed(f'http://127.0.0.1:{self.server.server_port}' + path,
            'GET', {'Authorization': 'test-only-single-request-capability'}, None,
            slot=self.slot, deadline=time.monotonic() + seconds, cleanup_seconds=.2,
            maximum_frame_bytes=2048, maximum_stderr_bytes=512)

    def test_actual_helper_roundtrip(self):
        self.assertEqual(self.request('/ok'), {'identity': 'fixed', 'ok': True})

    def test_dribbling_connection_is_case_local_deadline_and_sibling_survives(self):
        started = time.monotonic()
        with self.assertRaises(RequestFailure) as error:
            self.request('/dribble', .15)
        self.assertLess(time.monotonic() - started, .7)
        record = error.exception.terminal
        self.assertEqual(record['status'], 'timeout')
        self.assertEqual(record['failure_context']['decision']['disposition'], 'case_fail')
        self.assertFalse(record['failure_context']['decision']['retry_eligible'])
        self.assertTrue(self.request('/healthy-sibling')['ok'])

    def test_oversize_and_malformed_service_results_are_candidate_case_failure(self):
        for path in ['/oversize', '/bad']:
            with self.subTest(path=path), self.assertRaises(RequestFailure) as error:
                self.request(path)
            self.assertIsNone(error.exception.terminal['result'])
            self.assertFalse(error.exception.terminal['failure_context']['decision']['retry_eligible'])
