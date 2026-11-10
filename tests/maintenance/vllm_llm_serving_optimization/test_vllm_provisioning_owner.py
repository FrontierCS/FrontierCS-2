"""Authenticated request-path tests: child payloads and outer errors have no authority."""
from vllm_task import TASK_ROOT  # noqa: E402
import deployment_test_guard  # noqa: F401
import copy
from contextlib import contextmanager
from email.message import Message
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.error

TASK = TASK_ROOT
sys.path[:0] = [str(TASK / 'judge'), str(TASK / 'verifier')]
import driver_client
from driver_client import AuthenticatedProvisioningError, DriverClient
from provisioning_owner import ProvisioningOwner, for_phase
from controller_contract_build import emit
from gpu_session import GPUSession
from containment_policy import Boundary, LocalPlatformSignal, classify_failure
from types import SimpleNamespace

FIXTURE = Path(__file__).parent / 'fixtures/vllm/verifier-platform-failure.json'


class ProvisioningOwnerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_bytes())
        self.environment = dict(zip(('FRONTIER_RUN_ID', 'FRONTIER_ATTEMPT_ID', 'FRONTIER_VERIFICATION_ID',
            'FRONTIER_TASK_CONFIG_DIGEST', 'FRONTIER_VERIFIER_IMAGE_DIGEST'),
            (self.fixture[k] for k in ('run_id', 'attempt_id', 'verification_id',
                                      'task_config_digest', 'verifier_image_digest'))))
        self.request = {'gpu_type': 'H100', 'gpu_count': 1, 'max_duration_seconds': 120}
        self.key = 'verifier-create-fixture'
        self.client = DriverClient('http://driver.invalid', self.fixture['verification_id'],
                                   'synthetic-test-token', deadline=time.monotonic() + 10)

    def owner(self):
        return ProvisioningOwner(self.environment, {'original': ['group', 'candidate']},
                                  [{'instance_id': 'original'}], self.request, self.key)

    def failure(self, owner, cause='gpu_create_failed'):
        value = copy.deepcopy(self.fixture)
        value['controller_binding'] = owner.claim
        value['cause'] = cause
        value['proof'] = {'producer': {'gpu_create_failed': 'fixed_gpu_create_capacity',
            'sandbox_create_failed': 'fixed_sandbox_create_capacity',
            'sidecar_unavailable': 'declared_sidecar_create_capacity'}[cause]}
        if cause == 'sidecar_unavailable':
            value['proof']['profile_digest'] = 'sha256:' + 'b' * 64
        return value

    def problem(self, value, *, status=502, code='gpu_backend_unavailable', raw=None):
        headers = Message()
        headers['Content-Type'] = 'application/problem+json'
        body = raw if raw is not None else json.dumps({'status': status, 'code': code,
                       'title': 'synthetic capacity', 'platform_failure': value}).encode()
        return urllib.error.HTTPError(self.client.url, status, 'synthetic', headers, io.BytesIO(body))

    def call(self, owner):
        return self.client.request('POST', '', body=self.request, key=self.key, provisioning_owner=owner)

    def test_core_claim_byte_vectors_match_the_rust_fixture(self):
        owner = self.owner()
        self.assertEqual(owner.request_bytes, b'["H100",1,120,"verifier_gpu"]')
        for name in ('create_request_digest', 'create_idempotency_digest'):
            self.assertEqual(owner.claim[name], self.fixture['controller_binding'][name])

    def test_all_three_producers_only_through_the_authenticated_http_error_branch(self):
        for cause in ('gpu_create_failed', 'sandbox_create_failed', 'sidecar_unavailable'):
            owner = self.owner()
            with self.subTest(cause=cause), patch.object(driver_client, '_open_before_deadline',
                    side_effect=self.problem(self.failure(owner, cause))) as opened:
                with self.assertRaises(AuthenticatedProvisioningError) as raised:
                    self.call(owner)
                self.assertEqual(raised.exception.provenance.cause, cause)
                self.assertEqual(owner.operation_id, self.fixture['operation_id'])
                self.assertEqual(opened.call_count, 1)
                request = opened.call_args.args[0]
                self.assertEqual(request.get_header('Authorization'), 'Bearer synthetic-test-token')
                self.assertEqual(json.loads(request.data)['controller_binding'], owner.claim)

    def test_outer_internal_cleanup_cancel_and_fence_never_accept_an_embedded_proof(self):
        for status, code in ((500, 'internal'), (500, 'driver_cleanup_uncertain'),
                             (410, 'driver_cancelled'), (500, 'verifier_invalid_result'),
                             (503, 'gpu_backend_unavailable'), (502, 'quota_exhausted')):
            owner = self.owner()
            with self.subTest(status=status, code=code), patch.object(driver_client, '_open_before_deadline',
                    side_effect=self.problem(self.failure(owner), status=status, code=code)):
                with self.assertRaises(ValueError):
                    self.call(owner)
                self.assertIsNone(owner.operation_id)

    def test_foreign_identity_claim_and_unknown_proofs_cannot_acquire_authority(self):
        changes = [(name, 'foreign') for name in ('run_id', 'attempt_id', 'verification_id',
                   'task_config_digest', 'verifier_image_digest', 'operation_id')]
        changes += [('candidate_independent', 1), ('input_forwarded_bytes', False),
                    ('input_forwarded_bytes', 1), ('cause', 'network_control_failed'),
                    ('controller_binding', {}), ('untrusted_extra', True),
                    ('proof', {'producer': 'fixed_sandbox_create_capacity'})]
        for name, value in changes:
            owner = self.owner()
            failure = self.failure(owner)
            failure[name] = value
            with self.subTest(field=name, value=value), patch.object(driver_client, '_open_before_deadline',
                    side_effect=self.problem(failure)):
                with self.assertRaises(ValueError):
                    self.call(owner)
                self.assertIsNone(owner.operation_id)

    def test_duplicate_problem_fields_rejected_before_any_binding(self):
        owner = self.owner()
        raw = b'{"status":502,"status":500,"platform_failure":{}}'
        with patch.object(driver_client, '_open_before_deadline', side_effect=self.problem(None, raw=raw)):
            with self.assertRaises(ValueError):
                self.call(owner)
        self.assertIsNone(owner.operation_id)

    def test_oversized_problem_never_enters_the_proof_consumer(self):
        owner = self.owner()
        with patch.object(driver_client, '_open_before_deadline',
                side_effect=self.problem(None, raw=b' ' * (driver_client._CONTROL_BYTES + 1))), \
                patch.object(owner, 'consume_failure') as consume:
            with self.assertRaises(ValueError):
                self.call(owner)
        consume.assert_not_called()

    def test_precreate_plan_uses_verified_cpu_content_and_original_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'judge').mkdir()
            (root / 'judge/reference.patch').write_bytes(b'reference')
            (root / 'judge/source.py').write_bytes(b'immutable CPU source')
            emit(root, ['judge'], ['judge/reference.patch'], root / 'controller-contract.json',
                 root / 'controller-contract-limits.json')
            legacy_config = {'sidecars': {'scheduler': {'args': []}}}
            owner = for_phase(self.environment, legacy_config, {}, 'train', ('bfcl/public',), b'candidate',
                              root, self.request, self.key)
            slots = json.loads(owner.slot_bytes)
            self.assertEqual(len(slots), 24)
            self.assertEqual({slot['variant'] for slot in slots}, {'vanilla', 'reference', 'candidate'})
            self.assertTrue(all('operation_id' not in slot for slot in slots))
            self.assertIsNone(owner.operation_id)
            (root / 'judge/reference.patch').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'immutable image contract'):
                for_phase(self.environment, legacy_config, {}, 'train', ('bfcl/public',), b'candidate',
                          root, self.request, self.key)

    def test_child_json_and_fixture_signals_are_not_authenticated_driver_responses(self):
        owner = self.owner()
        slot = SimpleNamespace(run_id=self.fixture['run_id'], attempt_id=self.fixture['attempt_id'],
                               operation_id=self.fixture['operation_id'])
        for forged in (self.failure(owner), {'platform_failure': self.failure(owner)},
                       SimpleNamespace(code='gpu_create_failed', candidate_independent=True)):
            with self.subTest(forged=type(forged).__name__):
                decision = classify_failure(slot, Boundary('platform', 'provisioning', 'candidate_service'),
                    candidate_input_dispatched=True, platform_signal=forged)
                self.assertEqual(decision['disposition'], 'runtime_error')
                self.assertFalse(decision['retry_eligible'])
                self.assertIsNone(owner.operation_id)
        # The old fixture type has no production conversion into this owner.
        signal = LocalPlatformSignal(slot.run_id, slot.attempt_id, slot.operation_id,
            'gpu_create_failed', 'driver_gpu_provisioning', True, True)
        with self.assertRaises(ValueError):
            owner.consume_failure(signal, status=502, code='gpu_backend_unavailable')

    def test_success_echo_is_bound_and_replays_cannot_change_operation_or_claim(self):
        owner = self.owner()
        value = {'runner_id': self.fixture['operation_id'], 'controller_binding': owner.claim}
        @contextmanager
        def wire(request, deadline):
            body = io.BytesIO(json.dumps(value).encode())
            response = SimpleNamespace(status=202, geturl=lambda: self.client.url,
                isclosed=lambda: False, read1=body.read)
            yield response, SimpleNamespace(settimeout=lambda _: None)
        with patch.object(driver_client, '_open_before_deadline', wire):
            self.call(owner)
            self.call(owner)
            value['runner_id'] = 'gpurunner_01ARZ3NDEKTSV4RRFFQ69G5FAW'
            with self.assertRaises(ValueError):
                self.call(owner)
        owner = self.owner()
        value['controller_binding'] = {**owner.claim, 'frozen_group_variant_plan_digest': 'sha256:' + 'f' * 64}
        with patch.object(driver_client, '_open_before_deadline', wire), self.assertRaises(ValueError):
            self.call(owner)

    def test_status_media_and_cleanup_diagnostics_cannot_be_overridden_by_proof(self):
        for change in ('wrong_inner_status', 'wrong_media', 'cleanup_title'):
            owner = self.owner()
            problem = self.problem(self.failure(owner))
            if change == 'wrong_inner_status':
                value = {'status': 503, 'code': 'gpu_backend_unavailable', 'title': 'capacity',
                         'platform_failure': self.failure(owner)}
                problem = self.problem(None, raw=json.dumps(value).encode())
            elif change == 'wrong_media':
                problem.headers.replace_header('Content-Type', 'application/json')
            else:
                value = {'status': 502, 'code': 'gpu_backend_unavailable',
                         'title': 'pending GPU cleanup failed', 'platform_failure': self.failure(owner)}
                problem = self.problem(None, raw=json.dumps(value).encode())
            with self.subTest(change=change), patch.object(driver_client, '_open_before_deadline', side_effect=problem):
                with self.assertRaises((ValueError, driver_client.SidecarExecutionError)) as raised:
                    self.call(owner)
                self.assertNotIsInstance(raised.exception, AuthenticatedProvisioningError)
                self.assertIsNone(owner.operation_id)

    def test_cpu_memory_and_filesystem_errors_are_not_platform_or_scientific_terminals(self):
        for error in (MemoryError('CPU decode'), OSError('CPU persist')):
            owner = self.owner()
            with self.subTest(error=type(error).__name__), patch.object(driver_client, '_open_before_deadline',
                    side_effect=self.problem(self.failure(owner))), patch.object(driver_client, '_unique_object', side_effect=error):
                with self.assertRaises(type(error)):
                    self.call(owner)
                self.assertIsNone(owner.operation_id)

    def test_reset_http_problem_replays_same_request_without_capacity_authority(self):
        for status in (425, 502):
            owner = self.owner()
            problem = self.problem(self.failure(owner), status=status)
            requests = []

            @contextmanager
            def wire(request, deadline):
                requests.append((request.get_header('Idempotency-key'), request.data))
                if len(requests) == 1:
                    raise problem
                value = {'runner_id': self.fixture['operation_id'], 'controller_binding': owner.claim}
                body = io.BytesIO(json.dumps(value).encode())
                response = SimpleNamespace(status=202, geturl=lambda: self.client.url,
                    isclosed=lambda: False, read1=body.read)
                yield response, SimpleNamespace(settimeout=lambda _: None)

            with self.subTest(status=status), patch.object(problem, 'read',
                    side_effect=ConnectionResetError('truncated authenticated response')), \
                    patch.object(driver_client, '_open_before_deadline', wire), \
                    patch.object(owner, 'consume_failure', wraps=owner.consume_failure) as consume:
                self.call(owner)
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0], requests[1])
            consume.assert_not_called()
            self.assertEqual(owner.operation_id, self.fixture['operation_id'])

    def test_first_operation_and_terminal_cannot_be_rebound(self):
        owner = self.owner()
        value = self.failure(owner)
        owner.consume_failure(value, status=502, code='gpu_backend_unavailable')
        owner.consume_failure(value, status=502, code='gpu_backend_unavailable')
        value['operation_id'] = 'gpurunner_01ARZ3NDEKTSV4RRFFQ69G5FAW'
        with self.assertRaises(ValueError):
            owner.consume_failure(value, status=502, code='gpu_backend_unavailable')

    def test_lost_response_reuses_key_and_body_without_another_allocation(self):
        owner = self.owner()
        requests, allocations = [], set()
        @contextmanager
        def wire(request, deadline):
            requests.append((request.get_header('Idempotency-key'), request.data))
            allocations.add(requests[-1])
            if len(requests) == 1:
                raise urllib.error.URLError('lost response after allocation')
            raise self.problem(self.failure(owner))
            yield
        with patch.object(driver_client, '_open_before_deadline', wire), patch.object(driver_client.time, 'sleep'):
            with self.assertRaises(AuthenticatedProvisioningError):
                self.call(owner)
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(allocations), 1)
        self.assertEqual(requests[0], requests[1])

    def test_failed_create_never_guesses_delete_or_constructs_a_successful_session(self):
        owner = self.owner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((TASK / 'verifier/runtime-config.json').read_bytes())
            session = GPUSession(self.client, config, 'train', gpu_seconds=120, request_key=self.key,
                scratch=root / 'scratch', evidence=root / 'evidence',
                cleanup_deadline=time.monotonic() + 20, provisioning_owner=owner)
            with patch.object(driver_client, '_open_before_deadline',
                    side_effect=self.problem(self.failure(owner))) as opened:
                with self.assertRaises(AuthenticatedProvisioningError):
                    with session:
                        self.fail('failed create cannot enter a session')
            self.assertIsNone(session.runner_id)
            self.assertIsNone(session.runner)
            self.assertEqual(opened.call_count, 1)
            self.assertEqual(opened.call_args.args[0].method, 'POST')
            self.assertTrue((root / 'evidence/driver-platform-failure.json').exists())
            self.assertFalse((root / 'evidence/cleanup-error.json').exists())
