"""Actual clean episode exec and local model response, no external provider."""
from vllm_task import REPO_ROOT  # noqa: E402
import deployment_test_guard  # noqa: F401
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

JUDGE = REPO_ROOT / 'tasks/problems/MLSys/vllm_llm_serving_optimization/judge'
sys.path.insert(0, str(JUDGE))
from batch_coordinator import BackendError
import dispatch_broker
from dispatch_broker import run_episode
from serving_eval.settings import EpisodeSettings
from terminal_contract import Slot
from terminal_ledger import Ledger


class Tool:
    tainted = False
    def __init__(self, broken=False):
        self.broken = broken
    def __getstate__(self):
        raise AssertionError('auth-owning parent must never be pickled into worker')
    def read_patch(self):
        if self.broken:
            raise BackendError('observed candidate-used sidecar death')
        return 'fixed patch\n'


class DispatchBrokerTests(unittest.TestCase):
    def setUp(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                raw = json.dumps({'id': 'local-fixture', 'object': 'chat.completion', 'created': 0,
                    'model': EpisodeSettings().model, 'choices': [{'index': 0,
                    'message': {'role': 'assistant', 'content': '```bash\necho VLLM_SERVING_OPT_SUBMIT\n```'},
                    'finish_reason': 'stop'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ledger = Ledger(Path(self.temp.name) / 'ledger.sqlite')
        self.slot = Slot('run', 'attempt', 'verification', 'operation', 'eval', 'swe/public',
            'candidate', 'case', 'sha256:' + 'a' * 64, 'sha256:' + 'b' * 64)
        self.ledger.register(self.slot, 'case')

    def execute(self, slot, tool):
        return run_episode(self.ledger, slot,
            {'instance_id': slot.instance_id, 'problem_statement': 'local frozen fixture'},
            workload='swe', endpoint=f'http://127.0.0.1:{self.server.server_port}/v1',
            settings=EpisodeSettings(), testbed=tool, deadline=time.monotonic() + 5,
            cleanup_seconds=.3, maximum_frame_bytes=65536, maximum_stderr_bytes=4096)

    def test_actual_fresh_exec_model_call_and_patch_rpc(self):
        result = self.execute(self.slot, Tool())
        self.assertEqual(result['status'], 'complete', result)
        self.assertEqual(result['result']['episode']['patch'], 'fixed patch\n')
        self.assertEqual(result['result']['episode']['n_calls'], 1)
        self.assertTrue(self.ledger.cleanup(self.slot)['confirmed'])

    def test_isolated_worker_does_not_write_bytecode_into_source(self):
        worker_tree = Path(self.temp.name) / 'clean-worker'
        shutil.copytree(JUDGE, worker_tree, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        with patch.object(dispatch_broker, '__file__', str(worker_tree / 'dispatch_broker.py')):
            result = self.execute(self.slot, Tool())
        self.assertEqual(result['status'], 'complete', result)
        self.assertEqual(result['result']['episode']['patch'], 'fixed patch\n')
        self.assertEqual(result['result']['episode']['n_calls'], 1)
        self.assertTrue(self.ledger.cleanup(self.slot)['confirmed'])
        self.assertEqual(list(worker_tree.rglob('__pycache__')), [])
        self.assertEqual(list(worker_tree.rglob('*.pyc')), [])

    def test_sidecar_failure_kills_only_case_and_healthy_sibling_runs(self):
        result = self.execute(self.slot, Tool(broken=True))
        self.assertEqual(result['status'], 'runtime_error', result)
        self.assertEqual(result['failure_context']['decision']['disposition'], 'case_fail')
        self.assertFalse(result['failure_context']['decision']['retry_eligible'])
        sibling = replace(self.slot, instance_id='healthy')
        self.ledger.register(sibling, 'healthy')
        self.assertEqual(self.execute(sibling, Tool())['status'], 'complete')
        self.assertEqual(self.ledger.unresolved(), [])
