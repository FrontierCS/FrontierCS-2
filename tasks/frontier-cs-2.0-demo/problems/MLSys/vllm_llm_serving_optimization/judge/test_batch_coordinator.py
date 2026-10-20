"""Local fake HTTP adapter tests; no real model, shell backend, or cloud trial."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import random
import threading
import time
import unittest
from unittest.mock import patch
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from batch_coordinator import FINAL_HIDDEN_JPS, BackendError, PreparedGroup, PreparedTestbed, ModelClient, TestbedClient, _compact_record_trace, poisson_schedule, run_group
from instance_manifest import expected_groups
from serving_eval import bfcl, agent_runner
from serving_eval.accuracy import load_swe_cases
from serving_eval.settings import EpisodeSettings
from serving_eval.swe_inputs import load_swe_inputs, verify_projection
from serving_eval.tool_auth import _key_id
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat


class Fixture:
    def __init__(self):
        self.ready = []
        self.requests = []
        self.calls = {}
        self.identities = {}
        self.answers = {}
        self.bad_ready = False
        self.fail_model = False
        self.fail_patch = False
        self.tool_delay = 0.0
        self.tool_exit = 0
        private = Ed25519PrivateKey.generate()
        private_text = private.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()).decode()
        public_text = private.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH).decode()
        context = {'schema_version': 'frontier.tool-auth-context/v1', 'run_id': 'fixture-run',
                   'attempt_id': 'fixture-attempt', 'operation_id': 'fixture-operation',
                   'sidecar_id': 'fixture-scheduler'}
        self.auth_environment = patch.dict(os.environ, {
            'FRONTIER_TOOL_AUTH_PRIVATE_KEY_OPENSSH': private_text,
            'FRONTIER_TOOL_AUTH_PUBLIC_KEY_OPENSSH': public_text,
            'FRONTIER_TOOL_AUTH_KEY_ID': _key_id(public_text),
            'FRONTIER_TOOL_AUTH_CONTEXT_JSON': json.dumps(context, separators=(',', ':')),
        })
        self.auth_environment.start()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, value):
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == '/v1/models':
                    self.reply({'object': 'list', 'data': [{'id': EpisodeSettings().model, 'object': 'model', 'created': 0, 'owned_by': 'local-fixture'}]})
                    return
                ident = self.path.split('/')[1]
                fixture.ready.append((ident, time.perf_counter()))
                self.reply({'identity': fixture.identities[ident], 'ready': not fixture.bad_ready})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                fixture.requests.append((self.path, body, time.perf_counter()))
                if self.path == '/v1/chat/completions':
                    if fixture.fail_model:
                        self.send_response(400)
                        self.send_header('Content-Length', '0')
                        self.end_headers()
                        return
                    ident = body['vllm_xargs']['job_id']
                    count = fixture.calls.get(ident, 0)
                    fixture.calls[ident] = count + 1
                    if ident in fixture.answers:
                        content = '[core_memory_retrieve_all()]' if count == 0 else fixture.answers[ident]
                    else:
                        content = '```bash\ncat README.md\n```' if count == 0 else '```bash\necho VLLM_SERVING_OPT_SUBMIT\n```'
                    self.reply({'id': 'fixture', 'object': 'chat.completion', 'created': 0, 'model': EpisodeSettings().model, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}]})
                elif self.path.endswith('/execute'):
                    time.sleep(fixture.tool_delay)
                    self.reply({'identity': body['identity'], 'exit_code': fixture.tool_exit, 'output': 'fixture tool output'})
                elif self.path.endswith('/patch'):
                    time.sleep(0.015)
                    self.reply({'identity': body['identity'], 'exit_code': 1 if fixture.fail_patch else 0, 'patch': 'fixture patch\n'})
                else:
                    raise AssertionError(self.path)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = 'http://127.0.0.1:' + str(self.server.server_port)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.auth_environment.stop()

    def group(self, workload):
        group = PreparedGroup('fixture-operation', 'fixture-group', 'fixture-placement', 'train', workload, 'public', self.endpoint + '/v1', 'fixture-source')
        if workload == 'swe':
            ids = expected_groups('train')['swe/public']
            beds = tuple(PreparedTestbed(ident, self.endpoint + '/' + ident, 'child-' + ident, 'train') for ident in ids)
            group = replace(group, testbeds=beds)
            self.identities = {bed.instance_id: {**group.identity(), 'instance_id': bed.instance_id, 'child_id': bed.child_id, 'phase': bed.phase} for bed in beds}
        return group


class BatchCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture()
        self.addCleanup(self.fixture.close)

    def test_original_poisson_seeds_and_no_new_public_rate(self):
        for workload, rate, seed in [('swe', 0.5, 20260604), ('bfcl', 1, 20260605)]:
            rng = random.Random(seed)
            elapsed, expected = 0, []
            for _ in range(8):
                elapsed += rng.expovariate(rate)
                expected.append(elapsed)
            self.assertEqual(poisson_schedule(workload, 8, rate), expected)
        with self.assertRaises(BackendError):
            run_group(self.fixture.group('swe'), jps=100)
        group = replace(self.fixture.group('swe'), mode='eval', rate='hidden')
        with self.assertRaises(BackendError):
            run_group(group, jps=0.5)

    def test_final_hidden_groups_accept_only_the_configured_rates(self):
        # Final runs hidden groups at the configured
        # rates without the calibration label; any other rate is refused.
        self.assertEqual(FINAL_HIDDEN_JPS, {'swe': 1.0, 'bfcl': 1.25})

        class Admitted(Exception):
            pass
        clock = SimpleNamespace(perf_counter=lambda: (_ for _ in ()).throw(Admitted()))
        for workload in ('swe', 'bfcl'):
            group = replace(self.fixture.group(workload), mode='eval', rate='hidden')
            for calibration in (False, True):
                with self.subTest(workload=workload, calibration=calibration), \
                        patch('batch_coordinator.time', clock), self.assertRaises(Admitted):
                    run_group(group, jps=FINAL_HIDDEN_JPS[workload], calibration=calibration)
            for jps in (None, 1.5, {'swe': 0.5, 'bfcl': 1.0}[workload]):
                with self.subTest(workload=workload, jps=jps), self.assertRaises(BackendError):
                    run_group(group, jps=jps)

    def test_scheduler_swe_inputs_are_exact_and_answer_free(self):
        source = Path(__file__).parent / 'serving_eval/swe_data/instances.json'
        self.assertTrue(verify_projection(source))
        runtime = load_swe_inputs('eval')
        full = load_swe_cases('eval')
        self.assertEqual(runtime, [{'instance_id': row['instance_id'],
                                    'problem_statement': row['problem_statement']}
                                   for row in full])
        self.assertFalse({'test_patch', 'FAIL_TO_PASS', 'PASS_TO_PASS'} &
                         set().union(*(set(row) for row in runtime)))

    def test_full_public_swe_http_tools_and_patch_are_timed_after_readiness(self):
        group = self.fixture.group('swe')
        instances = [{'instance_id': row['instance_id'], 'problem_statement': row['problem_statement']} for row in load_swe_cases('train')]
        result = run_group(group, swe_instances=instances)
        self.assertFalse(result['score_ready'])
        self.assertNotIn('score', result)
        self.assertEqual(result['expected_instances'], 5)
        self.assertEqual(len(self.fixture.ready), 5)
        self.assertLess(max(t for _, t in self.fixture.ready), result['schedule_origin_monotonic'])
        self.assertEqual(len(self.fixture.requests), 20)
        for row in result['episodes']:
            self.assertFalse(row['requires_error_attribution'])
            episode = row['episode']
            self.assertEqual(episode['exit_status'], 'submitted')
            self.assertEqual(episode['n_calls'], 2)
            self.assertEqual(episode['patch'], 'fixture patch\n')
            self.assertGreaterEqual(episode['latency_seconds'], 0.015)
            self.assertAlmostEqual(episode['ended_monotonic'] - episode['started_monotonic'], episode['latency_seconds'])
            self.assertGreaterEqual(row['dispatched_monotonic'], result['schedule_origin_monotonic'] + row['scheduled_offset_seconds'])
            paths = [event.get('path') for event in row['trace'] if event['kind'] == 'testbed']
            self.assertEqual(paths, ['/ready', '/execute', '/patch'])
            events = [event for event in row['trace'] if event['kind'] == 'model']
            self.assertEqual(len(events), 2)
            self.assertTrue(all(event['request']['sha256'].startswith('sha256:') for event in events))
            requests = [body for path, body, _ in self.fixture.requests
                        if path == '/v1/chat/completions'
                        and body['vllm_xargs']['job_id'] == row['instance_id']]
            self.assertNotIn('seed', requests[0])
            self.assertIn('fixture tool output', requests[1]['messages'][-1]['content'])

    def test_full_public_bfcl_original_inprocess_tools_and_correctness(self):
        group = self.fixture.group('bfcl')
        ids = expected_groups('train')['bfcl/public']
        instances = bfcl.load_memory_instances(ids)
        answers = bfcl.load_memory_answers(ids)
        self.assertTrue(all('ground_truth' not in row for row in instances))
        self.fixture.answers = {ident: answers[ident][0] for ident in ids}
        result = run_group(group)
        self.assertEqual(len(result['episodes']), 8)
        self.assertFalse(result['score_ready'])
        self.assertEqual(self.fixture.ready, [])
        for row in result['episodes']:
            self.assertEqual(row['episode']['exit_status'], 'ok')
            self.assertNotIn('correct', row['episode'])
            self.assertEqual(row['episode']['output'], self.fixture.answers[row['instance_id']])
            self.assertEqual(row['episode']['n_steps'], 1)
            self.assertEqual([event['kind'] for event in row['trace']], ['model', 'bfcl_tools', 'model'])
            self.assertTrue(row['trace'][0]['request']['sha256'].startswith('sha256:'))
            self.assertTrue(row['trace'][1]['results']['content'])
            requests = [body for path, body, _ in self.fixture.requests
                        if path == '/v1/chat/completions'
                        and body['vllm_xargs']['job_id'] == row['instance_id']]
            self.assertEqual(requests[0]['seed'], 0)
            self.assertIn("'role': 'tool'", requests[1]['messages'][-1]['content'])

    def test_large_trace_responses_are_bounded_receipts_without_scoring_loss(self):
        large = 'x' * (16 * 1024 * 1024)
        response = {'output': large, 'exit_code': 0}
        trace = [
            {
                'kind': 'testbed',
                'path': '/execute',
                'started_monotonic': float(index),
                'ended_monotonic': float(index + 1),
                'response': response,
                **({'error_origin': 'testbed_transport_or_protocol',
                    'error_type': 'FixtureError'} if index == 3 else {}),
            }
            for index in range(4)
        ]
        scoring_episode = {
            'patch': 'complete final patch\n',
            'output': 'complete BFCL final answer',
            'latency_seconds': 12.5,
            'exit_status': 'submitted',
        }
        record = {
            'episode': scoring_episode.copy(),
            'patch_sha256': 'sha256:fixed-scoring-field',
            'requires_error_attribution': True,
            'trace': trace,
        }

        expected_raw = json.dumps(
            response, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
            allow_nan=False,
        ).encode()
        _compact_record_trace(record)

        self.assertEqual(record['episode'], scoring_episode)
        self.assertEqual(record['patch_sha256'], 'sha256:fixed-scoring-field')
        self.assertTrue(record['requires_error_attribution'])
        self.assertEqual(len(record['trace']), 4)
        self.assertLess(len(json.dumps(record['trace']).encode()), 32 * 1024)
        for event in record['trace']:
            receipt = event['response']
            self.assertEqual(receipt['bytes'], len(expected_raw))
            self.assertEqual(receipt['sha256'], 'sha256:' + hashlib.sha256(expected_raw).hexdigest())
            self.assertTrue(receipt['truncated'])
            self.assertEqual(receipt['encoding'], 'base64')
        self.assertEqual(record['trace'][-1]['error_origin'], 'testbed_transport_or_protocol')
        self.assertEqual(record['trace'][-1]['error_type'], 'FixtureError')

    def test_trace_compaction_failure_is_explicit_and_unscored(self):
        record = {
            'episode': {'output': 'must not be scored'},
            'requires_error_attribution': False,
            'trace': [{}],
        }
        _compact_record_trace(record)
        self.assertIsNone(record['episode'])
        self.assertEqual(record['execution_error']['origin'], 'trusted_trace_compaction')
        self.assertEqual(record['execution_error']['type'], 'BackendError')
        self.assertTrue(record['requires_error_attribution'])
        self.assertEqual(record['trace'], [])

    def test_model_error_origin_distinguishes_http_response_from_transport(self):
        class StatusError(Exception):
            pass

        class Completions:
            error = None

            def create(worker_self, **kwargs):
                raise worker_self.error

        completions = Completions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        module = ModuleType('openai')
        module.APIStatusError = StatusError
        module.DefaultHttpxClient = lambda **kwargs: object()
        module.OpenAI = lambda **kwargs: client
        for error, expected in (
            (StatusError('response'), 'model_api_response'),
            (ConnectionError('transport'), 'model_transport_or_client_unclassified'),
        ):
            trace = []
            completions.error = error
            with self.subTest(expected=expected), patch.dict(sys.modules, {'openai': module}):
                model = ModelClient('http://model:8000/v1', 10, trace)
                with self.assertRaises(type(error)):
                    model.create(model='fixture')
            self.assertEqual(trace[0]['error_origin'], expected)

    def test_model_http_error_keeps_partial_patch_and_response_provenance(self):
        group = self.fixture.group('swe')
        bed = group.testbeds[0]
        trace = []
        model = ModelClient(group.model_endpoint, 900, trace)
        self.addCleanup(model.close)
        client = TestbedClient(group, bed, trace)
        client.ready()
        self.fixture.fail_model = True
        result = agent_runner.run_instance({'instance_id': bed.instance_id, 'problem_statement': 'fixture'}, client=model, prepared_testbed=client, settings=EpisodeSettings())
        self.assertEqual(result.exit_status, 'limit_with_patch')
        self.assertEqual(result.patch, 'fixture patch\n')
        self.assertEqual(result.n_calls, 0)
        self.assertEqual([event['error_origin'] for event in trace if 'error_origin' in event], ['model_api_response'])
        self.assertEqual(trace[-1]['path'], '/patch')

    def test_tool_timeout_response_has_separate_transport_grace(self):
        group = self.fixture.group('swe')
        client = TestbedClient(group, group.testbeds[0], [])
        self.fixture.tool_delay = 0.08
        self.fixture.tool_exit = 124
        start = time.perf_counter()
        with patch('batch_coordinator.TOOL_TRANSPORT_GRACE_SECONDS', 0.2):
            code, output = client.run('fixture command', timeout=0.05)
        self.assertGreaterEqual(time.perf_counter() - start, 0.08)
        self.assertEqual(code, 124)
        self.assertEqual(output, 'fixture tool output')
        self.assertEqual(self.fixture.requests[-1][1]['timeout_seconds'], 0.05)

    def test_failed_patch_extraction_is_backend_error_not_empty_success(self):
        group = self.fixture.group('swe')
        client = TestbedClient(group, group.testbeds[0], [])
        self.fixture.fail_patch = True
        with self.assertRaises(BackendError):
            client.read_patch()

    def test_absent_or_not_ready_testbed_blocks_before_model_requests(self):
        group = self.fixture.group('swe')
        instances = [{'instance_id': row['instance_id'], 'problem_statement': row['problem_statement']} for row in load_swe_cases('train')]
        with self.assertRaises(BackendError):
            run_group(replace(group, testbeds=()), swe_instances=instances)
        self.fixture.bad_ready = True
        with self.assertRaises(BackendError):
            run_group(group, swe_instances=instances)
        self.assertEqual(self.fixture.requests, [])


if __name__ == '__main__':
    unittest.main()
