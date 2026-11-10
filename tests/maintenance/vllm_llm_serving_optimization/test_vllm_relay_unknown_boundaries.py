"""Unknown remote hooks close owned slots while CPU evidence failures propagate."""
import deployment_test_guard  # noqa: F401
import random
from pathlib import Path
import unittest
from unittest.mock import patch

import test_vllm_phase_controller as controller_fixture
import test_vllm_scheduler_stream as stream_fixture
import attestation
from batch_coordinator import PreparedGroup
from controller_contract_build import emit
from driver_client import RemoteExecutionError
from instance_manifest import expected_groups
from relay_client import SchedulerRelay


class RelayUnknownBoundaryTests(unittest.TestCase):
    def owned(self, variant, workload='bfcl'):
        fixture = controller_fixture.PhaseControllerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        source = fixture.root / 'source'
        (source / 'verifier').mkdir(parents=True)
        (source / 'verifier/runtime-plan.json').write_bytes(
            (stream_fixture.TASK / 'verifier/runtime-plan.json').read_bytes())
        emit(source, ['verifier'], [], source / 'controller-contract.json',
             source / 'controller-contract-limits.json')
        group = PreparedGroup('parent', 'parent', 'parent', 'train', workload, 'public',
                              'http://main:8000/v1', 'sha256:' + 'a' * 64)
        private, public = stream_fixture.keypair()
        identity = {'run_id': 'run', 'attempt_id': 'attempt', 'operation_id': 'parent',
                    'phase': 'train', 'name': stream_fixture.SCHEDULER,
                    'profile_digest': 'sha256:' + 'a' * 64}
        session = stream_fixture.Session(attestation.Signer(private, public, identity))
        session.runner_id, session.mode = 'parent', 'train'
        session.ids = tuple(expected_groups('train')[workload + '/public'])
        factory = fixture.factory(session=session, groups={variant: group}, source_root=source)
        relay = SchedulerRelay(session, public_key=public,
            relay_limits={'job_bytes': 10000, 'response_bytes': 1000000},
            max_stderr_bytes=512, evidence=fixture.root / 'relay', stream=True,
            cleanup_seconds=1, controller_factory=factory)
        return relay, factory

    @staticmethod
    def injected():
        return RuntimeError('opaque remote fault ' + str(random.Random(734).getrandbits(96)))

    def assert_closed(self, controller, expected_kind):
        self.assertEqual(controller.ledger.unresolved(), [])
        for slot in controller.slots:
            record = controller.ledger.read(slot)
            self.assertEqual(record['status'], 'runtime_error')
            self.assertEqual(record['failure_context']['boundary']['kind'], expected_kind)
            self.assertNotIn('infrastructure', str(record))

    def test_unknown_terminal_classifier_closes_c1_or_c2_all_variants(self):
        for variant in ('vanilla', 'reference', 'candidate'):
            for workload in ('bfcl', 'swe'):
                with self.subTest(variant=variant, workload=workload):
                    relay, factory = self.owned(variant, workload)
                    with patch('relay_client.remote_terminal_classification', side_effect=self.injected()):
                        relay.job('batch', variant, {}, timeout=10, name='batch')
                    self.assert_closed(factory.controllers[0],
                                       'swe_case' if workload == 'swe' else 'candidate_service')

    def test_unknown_final_view_and_payload_validator_close_group_c1(self):
        for variant in ('vanilla', 'reference', 'candidate'):
            for hook in ('_validated_view', 'payload_validator'):
                with self.subTest(variant=variant, hook=hook):
                    relay, factory = self.owned(variant)
                    if hook == 'payload_validator':
                        relay.payload_validator = lambda *args: (_ for _ in ()).throw(self.injected())
                    with patch.object(relay, '_validated_view', side_effect=self.injected()) if hook == '_validated_view' else patch.object(relay, 'payload_validator', relay.payload_validator):
                        with self.assertRaises(RemoteExecutionError):
                            relay.job('batch', variant, {}, timeout=10, name='batch')
                    self.assert_closed(factory.controllers[1], 'candidate_service')
                    self.assertTrue(all(factory.controllers[0].ledger.read(s)['status'] == 'complete'
                                        for s in factory.controllers[0].slots))

    def test_cpu_ledger_commit_and_evidence_io_propagate(self):
        for operation in ('commit_once', 'evidence'):
            with self.subTest(operation=operation):
                relay, factory = self.owned('candidate')
                if operation == 'commit_once':
                    import terminal_ledger
                    target = patch.object(terminal_ledger.Ledger, 'commit_once',
                                          side_effect=OSError('CPU ledger unavailable'))
                else:
                    import remote_terminal
                    target = patch.object(remote_terminal, '_retain_prefix',
                                          side_effect=OSError('CPU evidence unavailable'))
                with target, self.assertRaises(OSError):
                    relay.job('batch', 'candidate', {}, timeout=10, name='batch')
