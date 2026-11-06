"""Signed terminal readiness failure cannot become six independent relay failures."""
from vllm_task import TASK_ROOT  # noqa: E402
import deployment_test_guard  # noqa: F401
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

TASK = TASK_ROOT
sys.path[:0] = [str(TASK / 'judge'), str(TASK / 'verifier')]
import attestation
from driver_client import SchedulerPreinputInfrastructureError, SidecarExecutionError
from relay_client import SchedulerRelay, canonical_bytes
from scheduler_failure_contract import SCHEMA as FAILURE_SCHEMA
from scheduler_stream import SCHEMA, frame
from sidecar_catalog import SCHEDULER, testbed_name
from test_attestation import keypair


class Factory:
    environment = {'FRONTIER_RUN_ID': 'run', 'FRONTIER_ATTEMPT_ID': 'attempt'}
    def __call__(self, *args, **kwargs):
        return None


class SchedulerFailureTests(unittest.TestCase):
    def test_readiness_authority_excludes_check_scheduler_and_other_role_selectors(self):
        from scheduler_failure_contract import validate
        for selector, cause in [('scheduler', 'connection'), ('swe-000-check', 'timeout'),
                                (testbed_name(0, 'candidate'), 'connection'),
                                (testbed_name(0, 'vanilla'), 'unclassified')]:
            with self.subTest(selector=selector, cause=cause), self.assertRaises(ValueError):
                validate({'schema': FAILURE_SCHEMA, 'variant': 'vanilla', 'failure': {
                    'stage': 'preinput_sidecar_ready', 'cause': cause, 'sidecar_id': selector,
                    'candidate_input_forwarded': False}})

    def test_swe_model_output_failure_and_serving_api_failure_have_distinct_scopes(self):
        from relay_client import remote_terminal_classification
        service = {'instance_id': 'case', 'episode': {'exit_status': 'api_error', 'error': 'BadRequestError'},
            'requires_error_attribution': True, 'trace': [{'kind': 'model',
                'error_origin': 'model_api_response', 'error_type': 'BadRequestError'}]}
        boundary, failed = remote_terminal_classification('batch', 'swe/public', service)
        self.assertTrue(failed)
        self.assertEqual(boundary.kind, 'candidate_service')
        for row in ({'episode': None, 'execution_error': {'type': 'UnknownFailure'}},
                    {'episode': None, 'execution_error': {'type': 'RequestFailure'},
                     'trace': [{'kind': 'testbed', 'error_origin': 'testbed_transport_or_protocol'}]}):
            boundary, failed = remote_terminal_classification('batch', 'swe/public', row)
            self.assertTrue(failed)
            self.assertEqual(boundary.kind, 'swe_case')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        private, self.public = keypair()
        self.identity = {'run_id': 'run', 'attempt_id': 'attempt', 'operation_id': 'parent',
                         'phase': 'eval', 'name': SCHEDULER, 'profile_digest': 'sha256:' + 'a' * 64}
        self.signer = attestation.Signer(private, self.public, self.identity)
        self.calls = []
        self.session = SimpleNamespace(runner_id='parent', mode='eval', operations=0,
            client=SimpleNamespace(remaining=lambda: 10, verification_id='verification'),
            limits={'ssh_sessions': 10}, run=self.run_relay)
        self.relay = SchedulerRelay(self.session, public_key=self.public,
            relay_limits={'job_bytes': 10000, 'response_bytes': 100000},
            max_stderr_bytes=512, evidence=Path(self.tmp.name) / 'relay',
            stream=True, cleanup_seconds=1, controller_factory=Factory())

    def view(self, *, sequence=0, variant='vanilla', digest='sha256:' + 'b' * 64):
        payload = {'schema': FAILURE_SCHEMA, 'variant': variant, 'failure': {
            'stage': 'preinput_sidecar_ready', 'cause': 'connection',
            'sidecar_id': testbed_name(0, variant), 'candidate_input_forwarded': False}}
        signed = self.signer.sign(kind='batch', sequence=sequence, request_sha256=digest, payload=payload)
        return {'state': 'failed:SidecarReadinessError', 'sequence': sequence,
                'request_sha256': digest, 'signed': signed}

    def run_relay(self, argv, *, stdin, on_stdout, **kwargs):
        document = json.loads(stdin)
        self.calls.append(document)
        view = self.view(sequence=document['sequence'], variant=document['variant'],
                         digest='sha256:' + hashlib.sha256(stdin).hexdigest())
        on_stdout(frame({'schema': SCHEMA, 'event_kind': 'complete', 'event_index': 0,
                         'value': view}, 100000))

    def test_terminal_failure_stops_without_retry_or_later_variant_sequence_cascade(self):
        for variant in ('vanilla', 'reference', 'candidate'):
            with self.assertRaises(SchedulerPreinputInfrastructureError) as raised:
                self.relay.job('batch', variant, {}, timeout=10, name=variant)
            self.assertEqual(raised.exception.signal['variant'], 'vanilla')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.relay.sequence, 0)

    def test_signed_generic_scheduler_terminal_is_cached_without_infra_exemption(self):
        from cpu_reference_verdict import TrustedMeasurementError
        def generic_relay(argv, *, stdin, on_stdout, **kwargs):
            document = json.loads(stdin)
            self.calls.append(document)
            payload = {'schema': FAILURE_SCHEMA, 'variant': document['variant'], 'failure': {
                'stage': 'scheduler_control', 'cause': 'unclassified',
                'sidecar_id': None, 'candidate_input_forwarded': None}}
            digest = 'sha256:' + hashlib.sha256(stdin).hexdigest()
            signed = self.signer.sign(kind='batch', sequence=document['sequence'],
                                      request_sha256=digest, payload=payload)
            on_stdout(frame({'schema': SCHEMA, 'event_kind': 'complete', 'event_index': 0,
                'value': {'state': 'failed:SidecarReadinessError', 'sequence': document['sequence'],
                          'request_sha256': digest, 'signed': signed}}, 100000))
        self.session.run = generic_relay
        for variant in ('vanilla', 'reference', 'candidate'):
            with self.assertRaises(TrustedMeasurementError):
                self.relay.job('batch', variant, {}, timeout=10, name=variant)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.relay.sequence, 0)

    def test_wrong_sequence_role_request_signature_and_run_are_never_infra(self):
        original = self.view()
        cases = [(original, {'sequence': 1}), (original, {'variant': 'reference'}),
                 (original, {'request_sha256': 'sha256:' + 'c' * 64}),
                 ({**original, 'signed': {}}, {})]
        for view, overrides in cases:
            args = {'kind': 'batch', 'variant': 'vanilla', 'sequence': 0,
                    'request_sha256': 'sha256:' + 'b' * 64, 'progress': {}, **overrides}
            with self.assertRaises(SidecarExecutionError) as raised:
                self.relay._validated_view(canonical_bytes(view), **args)
            self.assertNotIsInstance(raised.exception, SchedulerPreinputInfrastructureError)
        self.relay.controller_factory.environment = {'FRONTIER_RUN_ID': 'other', 'FRONTIER_ATTEMPT_ID': 'attempt'}
        with self.assertRaises(SidecarExecutionError) as raised:
            self.relay._validated_view(canonical_bytes(original), kind='batch', variant='vanilla',
                sequence=0, request_sha256='sha256:' + 'b' * 64, progress={})
        self.assertNotIsInstance(raised.exception, SchedulerPreinputInfrastructureError)
