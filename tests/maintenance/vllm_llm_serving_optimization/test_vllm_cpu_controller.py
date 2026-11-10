"""CPU authority survives remote loss, retaining frozen first-terminal results."""
from vllm_task import REPO_ROOT  # noqa: E402
import deployment_test_guard  # noqa: F401
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

JUDGE = REPO_ROOT / 'tasks/problems/MLSys/vllm_llm_serving_optimization/judge'
sys.path.insert(0, str(JUDGE))
from containment_policy import Boundary
from controller_envelope import verify_terminal_bytes
from controller_identity import content_contract
from cpu_controller import CPUController
from terminal_contract import Slot, canonical, terminal


class CPUControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'trusted.py').write_bytes(b'immutable source')
        contract = content_contract(self.root, ['trusted.py'],
            maximum_contract_bytes=8192, maximum_source_bytes=8192)
        self.contract = self.root / 'controller-contract.json'
        self.contract.write_bytes(canonical(contract))
        self.slot = Slot('run', 'attempt', 'verification', 'operation', 'eval', 'swe/public',
                         'candidate', 'case', 'sha256:' + 'a' * 64, 'sha256:' + 'b' * 64)
        self.sibling = replace(self.slot, instance_id='sibling')
        self.arguments = dict(environment={'FRONTIER_RUN_ID': 'run', 'FRONTIER_ATTEMPT_ID': 'attempt',
            'FRONTIER_VERIFICATION_ID': 'verification',
            'FRONTIER_TASK_CONFIG_DIGEST': 'sha256:' + 'c' * 64,
            'FRONTIER_VERIFIER_IMAGE_DIGEST': 'sha256:' + 'd' * 64},
            content_contract=self.contract, source_root=self.root, evidence=self.root / 'controller',
            maximum_contract_bytes=8192, maximum_source_bytes=8192,
            maximum_terminal_bytes=16384, maximum_manifest_bytes=8192,
            started=time.monotonic(), deadline=time.monotonic() + 2)
        self.controller = CPUController([self.slot, self.sibling], **self.arguments)

    def test_remote_loss_closes_only_unresolved_and_envelopes_verify(self):
        self.controller.mark_dispatched(self.slot)
        healthy = terminal(self.slot, status='complete', cause='normal_completion',
            elapsed=.1, cap=2, result={'real_measurement': 3},
            result_limit_bytes=4096, result_limit_depth=16)
        observed = self.controller.observe(self.slot, canonical(healthy), verify=json.loads,
            boundary=Boundary('swe_case', 'execution'), prefix_bytes=512)
        closed = self.controller.fail_unresolved(boundary=Boundary('candidate_service', 'execution'))
        self.assertEqual(closed[0], observed)
        self.assertEqual(closed[1]['status'], 'runtime_error')
        self.assertEqual(self.controller.ledger.unresolved(), [])
        for path in (self.root / 'controller').glob('*.terminal.json'):
            record = verify_terminal_bytes(path.read_bytes(), context=self.controller.signer.context,
                slots=[self.slot, self.sibling], public_key=self.controller.signer.public_key,
                maximum_bytes=16384)
            self.assertIn(record, closed)
        self.assertEqual(self.controller.fail_unresolved(
            boundary=Boundary('candidate_service', 'execution')), closed)

    def test_key_and_journal_loss_are_trusted_defects(self):
        with patch.object(self.controller.signer, 'sign', side_effect=OSError('CPU key unavailable')):
            with self.assertRaises(OSError):
                self.controller.fail_unresolved(boundary=Boundary('candidate_service', 'execution'))
        with patch.object(self.controller.ledger, 'read', side_effect=OSError('CPU ledger missing')):
            with self.assertRaises(OSError):
                self.controller.fail_unresolved(boundary=Boundary('candidate_service', 'execution'))

    def test_remote_cannot_select_foreign_slot_or_overwrite_envelope(self):
        with self.assertRaises(ValueError):
            self.controller.mark_dispatched(replace(self.slot, instance_id='foreign'))
        self.controller.fail_unresolved(boundary=Boundary('candidate_service', 'execution'))
        path = next((self.root / 'controller').glob('*.terminal.json'))
        path.write_bytes(b'corrupt own evidence')
        with self.assertRaises(ValueError):
            self.controller.fail_unresolved(boundary=Boundary('candidate_service', 'execution'))
