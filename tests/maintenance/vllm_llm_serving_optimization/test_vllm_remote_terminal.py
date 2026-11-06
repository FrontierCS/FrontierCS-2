"""Remote corruption is scientific failure; CPU integrity defects stay fatal."""
from vllm_task import REPO_ROOT  # noqa: E402
import deployment_test_guard  # noqa: F401
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

JUDGE = REPO_ROOT / 'tasks/frontier-cs-2.0-demo/problems/MLSys/vllm_llm_serving_optimization/judge'
sys.path.insert(0, str(JUDGE))
from containment_policy import Boundary
from controller_envelope import Context, ControllerSigner, slots_digest, verify_terminal_bytes
from remote_terminal import observe_remote
from terminal_contract import Slot, canonical, terminal
from terminal_ledger import Ledger


class RemoteTerminalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.slot = Slot('run', 'attempt', 'verification', 'operation', 'eval', 'swe/public',
            'candidate', 'case', 'sha256:' + 'a' * 64, 'sha256:' + 'b' * 64)
        self.ledger = Ledger(self.root / 'ledger.sqlite')
        self.ledger.register(self.slot, 'operation/case')
        self.ledger.mark_dispatched(self.slot)

    def observe(self, raw, verify, **changes):
        args = dict(boundary=Boundary('candidate_service', 'response'),
            elapsed=.1, cap=1, maximum_bytes=16384, prefix_bytes=512,
            evidence=self.root / 'raw-prefix')
        args.update(changes)
        return observe_remote(self.ledger, self.slot, raw, verify=verify, **args)

    def test_malformed_unsigned_and_bad_signature_never_become_measurements(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        context = Context('run', 'attempt', 'verification',
            *['sha256:' + c * 64 for c in 'cdef'], slots_digest([self.slot]))
        signer = ControllerSigner(context, [self.slot], Ed25519PrivateKey.generate(),
            maximum_terminal_bytes=16384, maximum_manifest_bytes=8192)
        complete = terminal(self.slot, status='complete', cause='normal_completion',
            elapsed=.1, cap=1, result={'latency_seconds': 1},
            result_limit_bytes=1024, result_limit_depth=16)
        unsigned = signer.sign(complete)
        del unsigned['signature']
        bad = signer.sign(complete)
        bad['record']['result']['latency_seconds'] = 0
        for index, raw in enumerate([b'not-json', canonical(unsigned), canonical(bad)]):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                ledger = Ledger(root / 'ledger.sqlite')
                ledger.register(self.slot, 'operation/case')
                ledger.mark_dispatched(self.slot)
                value = observe_remote(ledger, self.slot, raw,
                    verify=lambda frame: verify_terminal_bytes(frame, context=context,
                        slots=[self.slot], public_key=signer.public_key, maximum_bytes=16384),
                    boundary=Boundary('candidate_service', 'response'), elapsed=.1, cap=1,
                    maximum_bytes=16384, prefix_bytes=512, evidence=root / 'raw-prefix')
                self.assertEqual(value['status'], 'runtime_error')
                self.assertIsNone(value['result'])
                self.assertFalse(value['failure_context']['decision']['retry_eligible'])
                self.assertEqual((root / 'raw-prefix').read_bytes(), raw[:512])

    def test_invalid_case_record_is_local_failure_and_sibling_is_retained(self):
        sibling = replace(self.slot, instance_id='healthy')
        self.ledger.register(sibling, 'operation/healthy')
        healthy = terminal(sibling, status='complete', cause='normal_completion', elapsed=.1,
            cap=1, result={'ok': True}, result_limit_bytes=1024, result_limit_depth=16)
        self.ledger.commit_once(sibling, healthy)
        value = self.observe(b'unsigned', lambda raw: {'status': 'infra'},
            boundary=Boundary('swe_case', 'tool'))
        self.assertEqual(value['failure_context']['decision']['disposition'], 'case_fail')
        self.assertEqual(self.ledger.read(sibling)['result'], {'ok': True})
        self.assertEqual(self.ledger.unresolved(), [])

    def test_remote_cross_slot_and_oversize_are_normalized(self):
        foreign = terminal(replace(self.slot, instance_id='foreign'), status='runtime_error',
            cause='child_exit', elapsed=.1, cap=1)
        value = self.observe(b'foreign', lambda raw: foreign)
        self.assertEqual(value['cause'], 'protocol_error')
        self.assertEqual(value['slot']['instance_id'], 'case')

    def test_trusted_ledger_and_evidence_defects_remain_release_blockers(self):
        with patch.object(self.ledger, 'commit_once', side_effect=OSError('CPU journal fsync')):
            with self.assertRaises(OSError):
                self.observe(b'bad', lambda raw: {})
        with patch('remote_terminal._retain_prefix', side_effect=OSError('CPU evidence full')):
            with self.assertRaises(OSError):
                self.observe(b'bad', lambda raw: {})
        self.assertIsNone(self.ledger.read(self.slot))
