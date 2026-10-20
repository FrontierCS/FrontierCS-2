"""Synthetic group isolation and complete-join checks; no GPU or inference."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'verifier'))
import parallel_phase
from instance_manifest import expected_groups
from test_runtime_plan import funded_fixture


class GroupJoinTests(unittest.TestCase):
    def test_complete_groups_join_once_and_missing_variant_fails_closed(self):
        groups = expected_groups('eval')
        complete = {group: {variant: {group: [group, variant]}
                            for variant in parallel_phase.VARIANTS} for group in groups}
        with patch.object(parallel_phase, 'evaluate_final', return_value={'score': 1, 'pass': False}) as scoring:
            self.assertEqual(parallel_phase.join_groups('eval', complete)['score'], 1)
            self.assertEqual(scoring.call_count, 1)
            self.assertEqual(set(scoring.call_args.args[0]), set(groups))
            incomplete = dict(complete)
            incomplete.pop('bfcl/hidden')
            with self.assertRaises(parallel_phase.TrustedMeasurementError):
                parallel_phase.join_groups('eval', incomplete)
            broken = dict(complete)
            broken['swe/public'] = dict(broken['swe/public'])
            broken['swe/public'].pop('reference')
            with self.assertRaises(parallel_phase.TrustedMeasurementError):
                parallel_phase.join_groups('eval', broken)
            self.assertEqual(scoring.call_count, 1)
            for malformed in (None, {'swe/public': None}, {key: None for key in groups}):
                with self.assertRaises(parallel_phase.TrustedMeasurementError):
                    parallel_phase.join_groups('eval', malformed)

    def test_two_train_groups_have_separate_parent_grants_and_failure_stops_score(self):
        plan, config = funded_fixture()
        calls = []
        owner = object()
        group_results = {}
        for group in expected_groups('train'):
            group_results[group] = {variant: {group: [group, variant]}
                                    for variant in parallel_phase.VARIANTS}
        class Session:
            def __init__(self, client, runtime_config, mode, *, gpu_seconds, request_key,
                         scratch, evidence, cleanup_deadline, calibration_clock, provisioning_owner):
                if provisioning_owner is not owner:
                    raise AssertionError('caller did not pass its precreate owner')
                calls.append((gpu_seconds, request_key, scratch, evidence))
                self.mode = mode
                self.runner_id = 'runner_' + evidence.parent.name.replace('-', '_')
                self.clock = calibration_clock
                self.clock.accept_grant(self.runner_id, 8)
            def __enter__(self): return self
            def __exit__(self, *_): return False
        class Client:
            verification_id = 'test_verification'
            def __init__(self, *_args, **_kwargs): pass
        with tempfile.TemporaryDirectory() as scratch, \
             patch.object(parallel_phase, 'DriverGPU', Client), \
             patch.object(parallel_phase, 'GPUSession', Session), \
             patch.object(parallel_phase, 'PhaseAuditPlan') as audit, \
             patch.object(parallel_phase, 'for_phase', return_value=owner) as owned, \
             patch.object(parallel_phase, 'run_phase', side_effect=lambda *args, **kwargs: group_results[kwargs['selected_group']]) as runner, \
             patch.object(parallel_phase, 'evaluate_train', return_value={'score': 1, 'pass': False}) as scoring:
            root = Path(scratch)
            kwargs = dict(environment={'FRONTIER_DRIVER_INTERNAL_URL': 'http://driver',
                'FRONTIER_VERIFICATION_ID': 'test_verification',
                'FRONTIER_VERIFIER_TOKEN': 'fixture'}, candidate_patch=b'', root=root,
                evidence=root/'evidence', deadline=1000)
            parallel_phase.run_parallel_phase(config, plan, 'train', **kwargs)
            self.assertEqual({row[0] for row in calls}, {10})
            self.assertEqual(len({row[1] for row in calls}), 2)
            self.assertEqual(len({row[2] for row in calls}), 2)
            for group in expected_groups('train'):
                receipt = root/'evidence'/group.replace('/', '-')/'parent-clock.json'
                marker = __import__('json').loads(receipt.read_text())
                self.assertEqual(marker['granted_seconds'], 8)
                cutoffs = __import__('json').loads((receipt.parent/'parent-cutoffs.json').read_text())
                self.assertAlmostEqual(cutoffs['capture_deadline'], marker['t0'] + 7, delta=0.001)
            self.assertEqual({call.kwargs['selected_group'] for call in runner.call_args_list}, set(expected_groups('train')))
            for call in runner.call_args_list:
                self.assertIs(call.args[2], plan)
                self.assertEqual(set(call.kwargs) & {'interactive_profiles', 'correctness_profiles',
                                                     'scheduler_profile', 'run_id', 'attempt_id'}, set())
            self.assertEqual(scoring.call_count, 1)
            self.assertEqual(owned.call_count, 2)
            self.assertEqual(audit.return_value.group.call_count, 2)
            runner.side_effect = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('parent failed'))
            with self.assertRaisesRegex(RuntimeError, 'independent workload parent failed'):
                parallel_phase.run_parallel_phase(config, plan, 'train', **dict(kwargs, evidence=root/'failed', root=root/'retry'))
            self.assertEqual(scoring.call_count, 1)
            self.assertTrue((root/'failed/incomplete-groups.json').is_file())
            trusted_error = parallel_phase.TrustedMeasurementError('vanilla', 'swe/public',
                                                                   'missing_or_invalid_timed_episode')
            runner.side_effect = trusted_error
            with self.assertRaises(parallel_phase.TrustedMeasurementError) as caught:
                parallel_phase.run_parallel_phase(config, plan, 'train',
                    **dict(kwargs, evidence=root/'trusted', root=root/'trusted-retry'))
            self.assertIs(caught.exception, trusted_error)
            self.assertEqual(scoring.call_count, 1)
            marker = __import__('json').loads((root/'trusted/incomplete-groups.json').read_text())
            self.assertFalse(marker['score_ready'])
            self.assertTrue(all(detail['attribution'] == 'infrastructure'
                for detail in marker['failure_details'].values()))


if __name__ == '__main__': unittest.main()
