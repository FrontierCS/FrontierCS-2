"""Public TRAIN feedback arithmetic, omission and non-disclosure contracts."""
from vllm_task import TASK_ROOT  # noqa: E402
import deployment_test_guard  # noqa: F401
import copy
import json
import tomllib
import math
from pathlib import Path
import sys
import unittest

TASK = TASK_ROOT
sys.path[:0] = [str(TASK / 'verifier'), str(TASK / 'judge')]
from cpu_reference_verdict import RuntimeGroupFailure
from fixed_targets import evaluate_fixed
from instance_manifest import expected_groups
from reference_verdict import InstanceObservation
from test_fixed_targets import observations, target
from verify import result_projection


FIELDS = {'geometric_mean_latency_seconds', 'speedup_over_vanilla',
          'p90_latency_seconds', 'p95_latency_seconds', 'p99_latency_seconds', 'correct'}
PROGRESS = {'paired_timing', 'p90', 'p95', 'p99', 'correctness'}


class TrainFeedbackTests(unittest.TestCase):
    def evaluate(self, candidate=None, mode='train'):
        return evaluate_fixed(candidate if candidate is not None else observations(mode), target(), mode=mode)

    def test_complete_fixed_groups_emit_actual_candidate_timings_and_vanilla_ratios(self):
        candidate = {key: [InstanceObservation(ident, float(index + 1), index != 0)
                          for index, ident in enumerate(ids)]
                     for key, ids in expected_groups('train').items()}
        result = self.evaluate(candidate)
        snapshot = copy.deepcopy(result)
        response = result_projection(result, 'train')
        self.assertEqual(result, snapshot)
        self.assertEqual(response['raw_score'], result['score'])
        self.assertNotIn('private_metrics', response)
        metrics = response['metrics']
        self.assertEqual(set(metrics), {'beat_spec_progress', 'score_unbounded'} | {
            workload + '_public_' + field for workload in ('swe', 'bfcl') for field in FIELDS})
        for workload in ('swe', 'bfcl'):
            n = len(candidate[workload + '/public'])
            gm = math.exp(math.fsum(math.log(index + 1) for index in range(n)) / n)
            prefix = workload + '_public_'
            self.assertAlmostEqual(metrics[prefix + 'geometric_mean_latency_seconds'], gm)
            self.assertAlmostEqual(metrics[prefix + 'speedup_over_vanilla'], 20 / gm)
            self.assertEqual(metrics[prefix + 'correct'], n - 1)
            for percentile in (90, 95, 99):
                self.assertEqual(metrics[prefix + f'p{percentile}_latency_seconds'],
                                 (percentile * n + 99) // 100)
        self.assertEqual(set(metrics['beat_spec_progress']), PROGRESS)
        for metric, counts in metrics['beat_spec_progress'].items():
            passed = sum(group['comparisons'][metric]
                         for group in result['reference_verdict']['groups'].values())
            self.assertEqual(counts['passed'], passed)
            self.assertEqual(counts['failed'], 2 - passed)
            self.assertEqual(counts['total'], 2)
            self.assertEqual(set(counts), {'passed', 'failed', 'total', 'pass_percent', 'fail_percent'})

    def test_runtime_error_and_missing_case_omit_unavailable_timing_without_fake_zero(self):
        for runtime_error in (True, False):
            candidate = observations('train')
            candidate['swe/public'] = (RuntimeGroupFailure('candidate', 'swe/public',
                expected_groups('train')['swe/public']) if runtime_error else candidate['swe/public'][:-1])
            result = self.evaluate(candidate)
            metrics = result_projection(result, 'train')['metrics']
            self.assertEqual(metrics['swe_public_correct'], 0 if runtime_error else 4)
            self.assertEqual({key for key in metrics if key.startswith('swe_')}, {'swe_public_correct'})
            self.assertIn('bfcl_public_geometric_mean_latency_seconds', metrics)
            self.assertEqual(metrics['beat_spec_progress']['p99']['failed'], 1)
            self.assertFalse(result['pass'])
            json.dumps(metrics, allow_nan=False)

    def test_reference_values_case_identities_hidden_details_never_copied(self):
        result = self.evaluate()
        result['reference_verdict']['groups']['swe/public'].update(
            reference_tails={'p90': 'PRIVATE_REFERENCE'}, reference_correct='PRIVATE_COUNT',
            required_correct_count='PRIVATE_THRESHOLD', unusable_instances=['PRIVATE_CASE_ID'],
            paired_speedup_over_reference='PRIVATE_RATIO', private_unknown='PRIVATE_UNKNOWN')
        result['continuous']['swe']['groups']['hidden'] = {'speedup': 'PRIVATE_HIDDEN'}
        result['unknown'] = 'PRIVATE_UNKNOWN'
        response = result_projection(result, 'train')
        self.assertNotIn('PRIVATE_', json.dumps(response))
        feedback = tomllib.loads((TASK / 'task.toml').read_text())['feedback']
        self.assertEqual(set(response['metrics']), set(feedback)
                         | {'beat_spec_progress', 'score_unbounded'})
        final = self.evaluate(mode='eval')
        self.assertEqual(result_projection(final, 'eval')['metrics'],
                         {'score_unbounded': final['score']})

    def test_incomplete_comparison_evidence_is_not_counted_as_failed(self):
        result = self.evaluate()
        del result['reference_verdict']['groups']['bfcl/public']
        result['pass'] = result['reference_verdict']['pass'] = False
        metrics = result_projection(result, 'train')['metrics']
        self.assertNotIn('beat_spec_progress', metrics)
        self.assertIn('swe_public_geometric_mean_latency_seconds', metrics)
        self.assertFalse(any(key.startswith('bfcl_') and key.endswith('_correct') for key in metrics))


if __name__ == '__main__':
    unittest.main()
