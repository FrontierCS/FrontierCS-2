"""Diagnostic result/evidence failure boundaries without Driver or GPU calls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'verifier'))
import four_group_diagnostic as diagnostic
from instance_manifest import expected_groups
from test_runtime_plan import funded_fixture


class DiagnosticCaptureTests(unittest.TestCase):
    def test_incomplete_group_and_join_failure_retain_archive_but_never_pass(self):
        plan, config = funded_fixture()
        for receipts_present in (False, True):
            with self.subTest(receipts_present=receipts_present), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                environment = {'FRONTIER_RESULT_PATH': str(root/'result.json'),
                    'FRONTIER_EVIDENCE_PATH': str(root/'evidence.tar.zst'),
                    'FRONTIER_VERIFICATION_MODE': 'eval',
                    'FRONTIER_DATASET_PATH': str(root/'dataset.json'),
                    'FRONTIER_PAYLOAD_PATH': str(root/'payload.tar')}
                captured = []
                def fail_group(*_args, **_kwargs):
                    work = root/'four-group-diagnostic'
                    marker = work/'parent-clocks'
                    marker.mkdir()
                    (marker/'eval-bfcl-hidden.json').write_text('short-grant-marker')
                    if receipts_present:
                        for group in expected_groups('eval'):
                            path = work/'operator-evidence'/group.replace('/', '-')
                            path.mkdir(parents=True)
                            (path/'parent-clock.json').write_text('{}')
                            (path/'parent-cutoffs.json').write_text('{}')
                    raise RuntimeError('parent failure' if not receipts_present else 'join failure')
                def archive(source, destination, _maximum):
                    captured.append((source/'failure.json').exists())
                    destination.write_text('bounded synthetic archive')
                with patch.object(diagnostic, 'validate_diagnostic', return_value={}), \
                     patch.object(diagnostic, 'dataset_selector'), \
                     patch.object(diagnostic, 'extract_patch', return_value=b'patch'), \
                     patch.object(diagnostic, 'run_parallel_phase', side_effect=fail_group), \
                     patch.object(diagnostic, 'write_evidence', side_effect=archive):
                    diagnostic.platform_wrapper(config, plan, environment)
                self.assertEqual(captured, [True])
                result = json.loads((root/'result.json').read_text())
                self.assertEqual(result['status'], 'failed')
                capture = json.loads((root/'four-group-diagnostic/capture-status.json').read_text())
                self.assertFalse(capture['score_ready'])
                self.assertEqual(len(capture['missing_groups']), 0 if receipts_present else 4)
                self.assertTrue((root/'evidence.tar.zst').exists())

    def test_complete_four_groups_capture_zero_scientific_score(self):
        plan, config = funded_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment = {'FRONTIER_RESULT_PATH': str(root/'result.json'),
                'FRONTIER_EVIDENCE_PATH': str(root/'evidence.tar.zst'),
                'FRONTIER_VERIFICATION_MODE': 'eval',
                'FRONTIER_DATASET_PATH': str(root/'dataset.json'),
                'FRONTIER_PAYLOAD_PATH': str(root/'payload.tar')}
            def groups(*_args, **_kwargs):
                self.assertEqual(_args[2], 'eval')
                work = root/'four-group-diagnostic'
                for group in expected_groups('eval'):
                    path = work/'operator-evidence'/group.replace('/', '-')
                    path.mkdir(parents=True)
                    (path/'parent-clock.json').write_text('{}')
                    (path/'parent-cutoffs.json').write_text('{}')
                return {'score': 99, 'pass': True}
            with patch.object(diagnostic, 'validate_diagnostic', return_value={}), \
                 patch.object(diagnostic, 'dataset_selector'), \
                 patch.object(diagnostic, 'extract_patch', return_value=b'patch'), \
                 patch.object(diagnostic, 'run_parallel_phase', side_effect=groups), \
                 patch.object(diagnostic, 'write_evidence', side_effect=lambda _src, target, _max: target.write_text('archive')):
                diagnostic.platform_wrapper(config, plan, environment)
            result = json.loads((root/'result.json').read_text())
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['raw_score'], 0.0)
            self.assertEqual(result['metrics']['diagnostic_complete'], 1)
            self.assertEqual(result['metrics']['workload_groups'], 4)


    def test_train_routes_two_public_groups_and_retains_failure_evidence(self):
        plan, config = funded_fixture()
        for fails in (False, True):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                environment = {'FRONTIER_RESULT_PATH': str(root/'result.json'),
                    'FRONTIER_EVIDENCE_PATH': str(root/'evidence.tar.zst'),
                    'FRONTIER_VERIFICATION_MODE': 'train',
                    'FRONTIER_DATASET_PATH': str(root/'dataset.json'),
                    'FRONTIER_PAYLOAD_PATH': str(root/'payload.tar')}
                routed = []
                def groups(_config, _plan, mode, **_kwargs):
                    routed.append(mode)
                    work = root/'four-group-diagnostic'
                    for group in expected_groups(mode):
                        receipt = work/'operator-evidence'/group.replace('/', '-')
                        receipt.mkdir(parents=True)
                        (receipt/'parent-clock.json').write_text('{}')
                        (receipt/'parent-cutoffs.json').write_text('{}')
                    if fails:
                        raise RuntimeError('synthetic train join failure')
                    return {'score': 88, 'pass': True}
                with patch.object(diagnostic, 'dataset_selector') as selector, \
                     patch.object(diagnostic, 'extract_patch', return_value=b'patch'), \
                     patch.object(diagnostic, 'run_parallel_phase', side_effect=groups), \
                     patch.object(diagnostic, 'write_evidence', side_effect=lambda _src, target, _max: target.write_text('archive')):
                    diagnostic.platform_wrapper(config, plan, environment)
                self.assertEqual(routed, ['train'])
                selector.assert_called_once_with(root/'dataset.json', 'train')
                result = json.loads((root/'result.json').read_text())
                capture = json.loads((root/'four-group-diagnostic/capture-status.json').read_text())
                self.assertEqual(capture['mode'], 'train')
                self.assertEqual(capture['captured_groups'], list(expected_groups('train')))
                self.assertEqual(capture['missing_groups'], [])
                if fails:
                    self.assertEqual(result['status'], 'failed')
                    self.assertTrue((root/'four-group-diagnostic/failure.json').is_file())
                else:
                    self.assertEqual(result['status'], 'passed')
                    self.assertEqual(result['raw_score'], 0.0)
                    self.assertEqual(result['metrics']['workload_groups'], 2)




    def test_public_train_reaches_real_phase_without_hidden_rate_selection(self):
        # Exercise the real phase guard and three-variant execution seam. The
        # prior wrapper-only mock accepted hidden_jps even for public train.
        from test_phase_run import PhaseJoinTests
        fixture = PhaseJoinTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        plan, config = funded_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment = {'FRONTIER_RESULT_PATH': str(root/'result.json'),
                'FRONTIER_EVIDENCE_PATH': str(root/'evidence.tar.zst'),
                'FRONTIER_VERIFICATION_MODE': 'train',
                'FRONTIER_DATASET_PATH': str(root/'dataset.json'),
                'FRONTIER_PAYLOAD_PATH': str(root/'payload.tar')}
            def phase(_config, _plan, mode, **kwargs):
                self.assertEqual(mode, 'train')
                self.assertTrue(kwargs['calibration'])
                # The synthetic observations are ordinary public measurements.
                # The actual mode/rate guard runs unchanged in either case.
                result = fixture.invoke(hidden_jps=kwargs['hidden_jps'])
                for group in expected_groups(mode):
                    receipt = kwargs['evidence']/group.replace('/', '-')
                    receipt.mkdir(parents=True)
                    (receipt/'parent-clock.json').write_text('{}')
                    (receipt/'parent-cutoffs.json').write_text('{}')
                return result
            with patch.object(diagnostic, 'dataset_selector'), \
                 patch.object(diagnostic, 'extract_patch', return_value=b''), \
                 patch.object(diagnostic, 'run_parallel_phase', side_effect=phase), \
                 patch.object(diagnostic, 'write_evidence', side_effect=lambda _s, p, _m: p.write_text('archive')):
                diagnostic.platform_wrapper(config, plan, environment)
            self.assertEqual(json.loads((root/'result.json').read_text())['status'], 'passed',
                (root/'four-group-diagnostic/failure.json').read_text() if (root/'four-group-diagnostic/failure.json').exists() else 'no failure receipt')
            self.assertEqual(len([e for e in fixture.events if e.startswith('scheduler-')]), 6)


class DiagnosticRecoveryTests(unittest.TestCase):
    def test_recovery_attempts_remaining_parents_after_missing_clock_or_delete_failure(self):
        from types import SimpleNamespace
        import hashlib
        plan, config = funded_fixture()
        env = {'FRONTIER_DRIVER_INTERNAL_URL': 'http://driver',
               'FRONTIER_VERIFICATION_ID': 'verification_one',
               'FRONTIER_VERIFIER_TOKEN': 'synthetic-only'}
        for mode in ('train', 'eval'):
            for failure in ('missing_clock', 'delete_failure', 'unbound', 'none'):
                with self.subTest(mode=mode, failure=failure):
                    groups = list(expected_groups(mode))
                    seen, deleted = [], []
                    def clock(identity, path, **kwargs):
                        group = groups[len(seen)]
                        seen.append(group)
                        self.assertEqual(path, Path('/clocks')/(mode+'-'+group.replace('/', '-')+'.json'))
                        expected = hashlib.sha256(('verification_one\0'+mode+'\0'+group+'\0'+
                            hashlib.sha256(b'patch').hexdigest()).encode()).hexdigest()
                        self.assertEqual(identity['request_key'], expected)
                        self.assertTrue(kwargs['resume'])
                        if len(seen) == 1 and failure == 'missing_clock':
                            raise FileNotFoundError('synthetic')
                        runner = None if len(seen) == 1 and failure == 'unbound' else 'runner-'+str(len(seen))
                        return SimpleNamespace(runner_id=runner, deadlines={'delete_deadline': 50})
                    class Client:
                        def __init__(self, *_args, **kwargs):
                            self.deadline = kwargs['deadline']
                        def request(self, method, route):
                            self_outer.assertEqual(method, 'DELETE')
                            self_outer.assertEqual(self.deadline, 40)
                            deleted.append(route)
                            if route == '/runner-1' and failure == 'delete_failure':
                                raise TimeoutError('synthetic')
                            return {'runner_id': route[1:], 'state': 'terminated'}
                    self_outer = self
                    with patch.object(diagnostic, 'begin', side_effect=clock), \
                         patch.object(diagnostic, 'DriverGPU', Client):
                        result = diagnostic.cleanup_only(config, plan, environment=env,
                            candidate_patch=b'patch', clocks=Path('/clocks'), deadline=40, mode=mode)
                    self.assertEqual(seen, groups)
                    self.assertIn('/runner-'+str(len(groups)), deleted)
                    self.assertEqual(result['cleanup_complete'], failure == 'none')
                    self.assertFalse(result['scientific_execution_replayed'])
                    self.assertTrue(all(result['parents'][g]['state'] == 'terminated' for g in groups[1:]))

if __name__ == '__main__': unittest.main()
