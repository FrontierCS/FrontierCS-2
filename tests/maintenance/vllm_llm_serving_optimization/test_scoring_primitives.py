import math
import unittest

from latency import tail_latencies
from quality_penalty import accuracy_multiplier


class PrimitiveTests(unittest.TestCase):
    def test_approved_small_sample_nearest_ranks(self):
        self.assertEqual(tail_latencies([5, 1, 4, 2, 3]), {'p90': 5, 'p95': 5, 'p99': 5})
        self.assertEqual(tail_latencies(range(1, 51)), {'p90': 45, 'p95': 48, 'p99': 50})
        self.assertEqual(tail_latencies(range(1, 101)), {'p90': 90, 'p95': 95, 'p99': 99})
        self.assertEqual(tail_latencies([2, 2, 2]), {'p90': 2, 'p95': 2, 'p99': 2})

    def test_invalid_timings_are_not_silently_dropped(self):
        for values in ([], [1, 0], [1, -1], [math.nan], [math.inf], [True], ['2']):
            with self.subTest(values=values), self.assertRaises(ValueError):
                tail_latencies(values)

    def test_original_vanilla_accuracy_examples(self):
        self.assertEqual(accuracy_multiplier(10 / 50, 8 / 50), 1)
        self.assertAlmostEqual(accuracy_multiplier(10 / 50, 7 / 50), 1 / 6)
        self.assertEqual(accuracy_multiplier(0, 0), 1)
        self.assertEqual(accuracy_multiplier(0, 1), 1)
        self.assertEqual(accuracy_multiplier(0.5, 0.75), 1)
        self.assertAlmostEqual(accuracy_multiplier(1, 0), 0.05)

    def test_accuracy_domain_is_explicit(self):
        for value in (-0.01, 1.01, math.nan, math.inf, True, '0.5'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                accuracy_multiplier(value, 0.5)


class PairedTimingTests(unittest.TestCase):
    def test_fixed_denominator_and_execution_failure_floor(self):
        from paired_timing import paired_speedups
        paired = paired_speedups(['ok', 'missing', 'api', 'zero'], {'ok': 10, 'missing': 10, 'api': 10, 'zero': 10}, {'ok': 1, 'api': 0.1, 'zero': 0}, candidate_api_error_ids=['api'])
        self.assertEqual(paired.ratios, {'ok': 10, 'missing': 1 / 8, 'api': 1 / 8, 'zero': 1 / 8})
        self.assertFalse(paired.complete)
        self.assertEqual(paired.unusable, {'missing': 'missing', 'api': 'api_error', 'zero': 'unusable_latency'})

    def test_valid_results_keep_timing_without_invented_status_penalty(self):
        from paired_timing import paired_speedups
        paired = paired_speedups(['force_quit', 'wrong_answer'], {'force_quit': 100, 'wrong_answer': 1}, {'force_quit': 1, 'wrong_answer': 100})
        self.assertEqual(paired.ratios, {'force_quit': 100, 'wrong_answer': 0.01})
        self.assertTrue(paired.complete)
        self.assertEqual(paired.unusable, {})

    def test_bad_baseline_and_wrong_denominator_are_errors(self):
        from paired_timing import paired_speedups
        for expected, baseline, candidate in [(['a'], {}, {}), (['a'], {'a': 0}, {'a': 1}), (['a', 'a'], {'a': 1}, {'a': 1}), (['a'], {'a': 1}, {'b': 1})]:
            with self.subTest(expected=expected, baseline=baseline), self.assertRaises(ValueError):
                paired_speedups(expected, baseline, candidate)

    def test_equal_rate_geomean_and_real_slowdown(self):
        from paired_timing import combine_rate_speedups, paired_speedups
        self.assertAlmostEqual(combine_rate_speedups(10, 100), math.sqrt(1000))
        paired = paired_speedups(['slow'], {'slow': 1}, {'slow': 20})
        self.assertEqual(paired.ratios['slow'], 1 / 20)
        self.assertTrue(paired.complete)

    def test_speedup_improvements_above_old_cap_are_comparable(self):
        from paired_timing import score_from_speedup
        self.assertEqual(score_from_speedup(0.125), 0)
        self.assertEqual(score_from_speedup(1), 0)
        self.assertAlmostEqual(score_from_speedup(10), 50)
        self.assertLess(score_from_speedup(100), score_from_speedup(1000))
        self.assertLess(score_from_speedup(1e20), 100)


class ContinuousTests(unittest.TestCase):
    def test_pool_accuracy_then_apply_one_penalty(self):
        from continuous import RateMeasurement, workload_continuous
        ids = tuple(str(i) for i in range(50))
        vanilla = dict.fromkeys(ids, 10)
        candidate = dict.fromkeys(ids, 1)
        groups = {'public': RateMeasurement(ids, vanilla, candidate, 10, 8), 'hidden': RateMeasurement(ids, vanilla, candidate, 10, 6)}
        result = workload_continuous(groups, mode='eval')
        self.assertAlmostEqual(result['speedup'], 10)
        self.assertAlmostEqual(result['vanilla_accuracy'], 0.2)
        self.assertAlmostEqual(result['candidate_accuracy'], 0.14)
        self.assertAlmostEqual(result['accuracy_multiplier'], 1 / 6)
        self.assertAlmostEqual(result['score'], 50 / 6)
        self.assertEqual(result['instance_evaluations'], 100)

    def test_train_uses_public_and_keeps_both_workload_weights(self):
        from continuous import RateMeasurement, continuous_score
        group = RateMeasurement(('a',), {'a': 10}, {'a': 1}, 1, 1)
        slower = RateMeasurement(('a',), {'a': 1}, {'a': 1}, 1, 1)
        result = continuous_score({'swe': {'public': group}, 'bfcl': {'public': slower}}, mode='train')
        self.assertAlmostEqual(result['score'], 40)
        with self.assertRaises(ValueError):
            continuous_score({'swe': {'public': group}}, mode='train')
        with self.assertRaises(ValueError):
            continuous_score({'swe': {'public': group}, 'bfcl': {'public': group}}, mode='eval')


if __name__ == '__main__':
    unittest.main()
