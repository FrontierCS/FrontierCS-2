"""CPU-side relay client against a synthetic session; signatures are real."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'verifier'))
import attestation
from driver_client import RemoteExecutionError, SidecarExecutionError
from relay_client import (JOB_SCHEMA, RELAY_ATTEMPTS, SchedulerRelay, canonical_bytes,
                          remote_terminal_classification)
from containment_policy import Boundary, LocalPlatformSignal, classify_failure
from sidecar_catalog import SCHEDULER
from test_attestation import keypair

IDENTITY = {'run_id': 'run', 'attempt_id': 'attempt', 'operation_id': 'runner_one', 'phase': 'eval',
            'name': SCHEDULER, 'profile_digest': 'sha256:' + '1' * 64}


class SyntheticSession:
    """Plays the GPU parent: records relay argv and answers like the scheduler."""

    def __init__(self, signer, root):
        self.signer = signer
        self.root = root
        self.runner_id = 'runner_one'
        self.mode = 'eval'
        self.operations = 0
        self.limits = {'ssh_sessions': 10}
        self.client = SimpleNamespace(remaining=lambda: 10)
        self.calls = []
        self.fail_first = 0
        self.answer = None
        self.receipts = {}

    def run(self, argv, *, stdin, output, max_stdout_bytes, max_stderr_bytes):
        self.operations += 1
        self.calls.append((argv, stdin, max_stdout_bytes, max_stderr_bytes))
        if self.operations >= self.limits['ssh_sessions']:
            raise SidecarExecutionError('Verifier phase SSH pool exhausted')
        if self.fail_first:
            self.fail_first -= 1
            raise RemoteExecutionError('synthetic transport failure', cause='child_exit')
        Path(output).mkdir(parents=True)
        action = argv[3]
        if action == 'job':
            document = json.loads(stdin)
            digest = 'sha256:' + hashlib.sha256(stdin).hexdigest()
            payload = {'variant': document['variant'], 'echo': document['sequence']}
            signed = self.signer.sign(kind=document['kind'], sequence=document['sequence'],
                                      request_sha256=digest, payload=payload)
            view = {'sequence': document['sequence'], 'request_sha256': digest, 'state': 'complete', 'signed': signed}
            if self.answer is not None:
                view = self.answer(view)
            data = json.dumps(view).encode()
        else:
            selector = json.loads(stdin)
            data = self.receipts[(selector['variant'], selector['index'])]
        if len(data) > max_stdout_bytes:
            raise RemoteExecutionError('SSH output exceeded its publisher bound', cause='output_limit')
        path = Path(output) / 'stdout'
        path.write_bytes(data)
        return path


class RelayClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        private, self.public = keypair()
        self.signer = attestation.Signer(private, self.public, IDENTITY)
        self.session = SyntheticSession(self.signer, self.root)
        self.limits = {'job_bytes': 10000, 'response_bytes': 100000, 'receipt_bytes': 200000,
                       'batch_timeout_seconds': {'train': 10, 'eval': 20}, 'correctness_timeout_seconds': 1200,
                       'receipt_timeout_seconds': 30}

    def relay(self, public=None):
        self.relays = getattr(self, 'relays', 0) + 1
        return SchedulerRelay(self.session, public_key=public or self.public, relay_limits=self.limits,
                              max_stderr_bytes=512, evidence=self.root / ('relay' if self.relays == 1 else f'relay-{self.relays}'))

    def test_swe_partial_patch_and_unknown_errors_are_case_failures_for_all_variants(self):
        for variant in ('vanilla', 'reference', 'candidate'):
            for error in ('BadRequestError', 'PreviouslyUnseenFailure'):
                with self.subTest(variant=variant, error=error):
                    remote = {'variant': variant, 'episode': {'patch': 'partial patch',
                        'exit_status': 'limit_with_patch', 'error': error},
                        'requires_error_attribution': True,
                        'trace': [{'kind': 'model', 'error_origin': 'model_api_response',
                                   'error_type': error}]}
                    boundary, failed = remote_terminal_classification('batch', 'swe/hidden', remote)
                    self.assertTrue(failed)
                    self.assertEqual(boundary, Boundary('swe_case', 'execution'))
                    self.assertEqual(classify_failure(SimpleNamespace(), boundary,
                        candidate_input_dispatched=True)['disposition'], 'case_fail')

    def test_clean_swe_and_bfcl_step_limit_still_complete(self):
        for group, episode in (('swe/public', {'patch': '', 'error': None}),
                               ('bfcl/public', {'instance_id': 'original', 'exit_status': 'force_quit', 'error': 'max_steps',
                                                'output': '', 'started_monotonic': 1.,
                                                'ended_monotonic': 2., 'latency_seconds': 1.})):
            with self.subTest(group=group):
                _, failed = remote_terminal_classification('batch', group,
                    {'instance_id': 'original', 'episode': episode, 'requires_error_attribution': False, 'trace': []})
                self.assertFalse(failed)

    def test_remote_infrastructure_claim_cannot_forge_trusted_platform_exception(self):
        remote = {'episode': {'patch': '', 'error': 'gpu_create_failed'},
                  'trace': [], 'requires_error_attribution': True,
                  'failure_context': {'boundary': {'kind': 'platform', 'stage': 'provisioning'}}}
        boundary, failed = remote_terminal_classification('batch', 'swe/hidden', remote)
        self.assertTrue(failed)
        slot = SimpleNamespace(run_id='run', attempt_id='attempt', operation_id='parent')
        signal = LocalPlatformSignal('run', 'attempt', 'parent', 'gpu_create_failed',
                                     'driver_gpu_provisioning', True)
        self.assertEqual(classify_failure(slot, boundary, candidate_input_dispatched=True,
                         platform_signal=signal)['disposition'], 'case_fail')
        trusted = classify_failure(slot, Boundary('platform', 'provisioning', 'swe_case'),
                                   candidate_input_dispatched=False, platform_signal=signal)
        self.assertEqual((trusted['disposition'], trusted['scientific_effect']),
                         ('infrastructure', 'unscored'))

    def test_job_runs_one_bounded_relay_op_and_binds_the_signed_answer(self):
        relay = self.relay()
        payload = relay.job('batch', 'vanilla', {'prepared': 1}, timeout=20, name='vanilla-batch')
        self.assertEqual(payload, {'variant': 'vanilla', 'echo': 0})
        argv, stdin, stdout_bytes, stderr_bytes = self.session.calls[0]
        self.assertEqual(argv[:3], ['/opt/frontier/task-runtime', '/opt/frontier/runtime-python',
                                    '/opt/frontier/judge/scheduler_relay.py'])
        self.assertEqual(argv[3:], ['job', '--max-input-bytes', str(len(stdin)), '--max-output-bytes', '100000',
                                    '--timeout-seconds', '20'])
        self.assertEqual((stdout_bytes, stderr_bytes), (100000, 512))
        document = json.loads(stdin)
        self.assertEqual((document['schema'], document['kind'], document['sequence'], document['prepared']),
                         (JOB_SCHEMA, 'batch', 0, 1))
        self.assertEqual(stdin, canonical_bytes(document))
        summary = json.loads((self.root / 'relay/vanilla-batch/envelope-summary.json').read_text())
        self.assertEqual((summary['kind'], summary['sequence'], summary['identity']['name']), ('batch', 0, SCHEDULER))
        self.assertFalse((self.root / 'relay/vanilla-batch/envelope.json').exists())
        self.assertTrue((self.root / 'relay/vanilla-batch/relay-0/stdout').is_file(), 'the signed answer is kept once, raw')
        self.assertEqual(relay.job('batch', 'reference', {}, timeout=20, name='reference-batch')['echo'], 1)

    def test_lost_answers_are_re_requested_within_the_pool_but_never_beyond(self):
        self.session.fail_first = 2
        relay = self.relay()
        self.assertEqual(relay.job('batch', 'vanilla', {}, timeout=20, name='vanilla-batch')['echo'], 0)
        self.assertEqual(len(self.session.calls), 3)
        self.assertEqual(len({call[1] for call in self.session.calls}), 1, 'the same bytes are resent')
        self.session.fail_first = RELAY_ATTEMPTS
        with self.assertRaises(SidecarExecutionError):
            relay.job('batch', 'reference', {}, timeout=20, name='reference-batch')
        self.assertTrue((self.root / 'relay/reference-batch/error.json').is_file())
        self.session.fail_first = 1
        self.session.operations = self.session.limits['ssh_sessions'] - 1
        with self.assertRaises(SidecarExecutionError):
            relay.job('batch', 'candidate', {}, timeout=20, name='candidate-batch')

    def test_foreign_key_wrong_binding_or_tampered_answer_is_refused(self):
        _, other = keypair()
        with self.assertRaisesRegex(SidecarExecutionError, 'attestation'):
            self.relay(other).job('batch', 'vanilla', {}, timeout=20, name='a')
        for change in ('sequence', 'request', 'variant', 'state', 'phase'):
            def tamper(view, change=change):
                if change == 'sequence':
                    view['sequence'] = 5
                elif change == 'request':
                    view['request_sha256'] = 'sha256:' + '0' * 64
                elif change == 'variant':
                    view['signed'] = self.signer.sign(kind='batch', sequence=0, request_sha256=view['request_sha256'],
                                                      payload={'variant': 'reference'})
                elif change == 'state':
                    view['state'] = 'failed:JobError'
                    view.pop('signed')
                return view
            self.session.answer = tamper
            with self.subTest(change=change):
                if change == 'phase':
                    self.session.mode = 'train'
                with self.assertRaises(SidecarExecutionError):
                    self.relay().job('batch', 'vanilla', {}, timeout=20, name=change)
                self.session.mode = 'eval'
        self.session.answer = None
        with self.assertRaises(SidecarExecutionError):
            self.relay().job('batch', 'vanilla', {'big': 'x' * 20000}, timeout=20, name='big')

    def test_receipts_are_bound_by_size_and_digest(self):
        relay = self.relay()
        raw = json.dumps({'instance_id': 'x', 'execution': 'complete'}).encode()
        self.session.receipts[('candidate', 3)] = raw
        digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
        receipt = relay.receipt('candidate', 3, expected_sha256=digest, expected_bytes=len(raw), name='r3')
        self.assertEqual(receipt['execution'], 'complete')
        self.assertFalse((self.root / 'relay/r3/stdout').exists(), 'full receipts are not retained as evidence')
        excerpt = json.loads((self.root / 'relay/r3/receipt-excerpt.json').read_text())
        self.assertEqual((excerpt['sha256'], excerpt['bytes']), (digest, len(raw)))
        self.assertIn('complete', excerpt['content'])
        big = b'{"log": "' + b'x' * 100000 + b'"}'
        self.session.receipts[('candidate', 4)] = big
        relay.receipt('candidate', 4, expected_sha256='sha256:' + hashlib.sha256(big).hexdigest(),
                      expected_bytes=len(big), name='r4')
        excerpt = json.loads((self.root / 'relay/r4/receipt-excerpt.json').read_text())
        self.assertTrue(excerpt['truncated'])
        self.assertLess(len(excerpt['head']) + len(excerpt['tail']), 70000)
        argv = self.session.calls[-1][0]
        self.assertEqual(argv[3:], ['receipt', '--max-input-bytes', str(len(self.session.calls[-1][1])),
                                    '--max-output-bytes', '200000', '--timeout-seconds', '30'])
        with self.assertRaises(SidecarExecutionError):
            relay.receipt('candidate', 3, expected_sha256=digest, expected_bytes=len(raw) + 1, name='r3b')
        with self.assertRaises(SidecarExecutionError):
            relay.receipt('candidate', 3, expected_sha256='sha256:' + '0' * 64, expected_bytes=len(raw), name='r3c')
        with self.assertRaises(SidecarExecutionError):
            relay.receipt('candidate', 3, expected_sha256=digest, expected_bytes=300000, name='r3d')


if __name__ == '__main__':
    unittest.main()
