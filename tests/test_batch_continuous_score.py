"""Batch timing scores preserve improvements above the former 16x cap."""

import deployment_test_guard  # noqa: F401  # isort: skip

import importlib.util
import math
import os
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = Path(os.environ.get("FRONTIER_BATCH_SCORE_EXPORT", str(
    ROOT / "tasks/frontier-cs-2.0-demo/problems/Database/batch_local_counting"
)))
spec = importlib.util.spec_from_file_location("batch_score_evaluator", TASK / "judge/evaluator.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class BatchContinuousScore(unittest.TestCase):
    def test_floor_and_four_times_speedup_anchor(self):
        self.assertEqual(evaluator._score(1.0, 2.0), 0.0)
        self.assertEqual(evaluator._score(1.0, 1.0), 0.0)
        self.assertEqual(evaluator._score(4.0, 1.0), 50.0)
        self.assertAlmostEqual(evaluator._score(2.0, 1.0), 100.0 / 3.0)
        self.assertEqual(evaluator._score(0.0, 1.0), 0.0)
        self.assertEqual(evaluator._score(1.0, 0.0), 0.0)

    def test_formerly_clipped_speedups_keep_strict_order(self):
        speedups = [16.0, 32.0, 64.0, 256.0, 65536.0, 2.0 ** 1023]
        scores = [evaluator._score(speedup, 1.0) for speedup in speedups]
        self.assertTrue(all(0.0 < score < 100.0 for score in scores))
        self.assertTrue(all(left < right for left, right in zip(scores, scores[1:])))
        self.assertAlmostEqual(scores[0], 200.0 / 3.0)

    def test_paired_round_median_and_raw_speedup_remain_unchanged(self):
        logs = [math.log2(value) for value in (2.0, 8.0, 32.0, 64.0, 128.0)]
        score, speedup = evaluator._score_log_ratios(logs)
        objective = statistics.median(logs)
        self.assertEqual(speedup, 2.0 ** objective)
        self.assertEqual(score, 100.0 * (objective / (2.0 + objective)))
        self.assertNotEqual(score, round(score, 2))
        floor_score, raw_speedup = evaluator._score_log_ratios([-4.0, -3.0, -2.0, -1.0, 10.0])
        self.assertEqual(floor_score, 0.0)
        self.assertEqual(raw_speedup, 0.25)

    def test_reference_verdict_does_not_follow_aggregate_score(self):
        rows = [{"workload": name, "metrics": {
            "exact_counts": True, "batch_latency_seconds": True,
        }} for name in evaluator.CASE_NAMES]
        rows[0]["metrics"]["batch_latency_seconds"] = False
        metrics = evaluator._metrics(evaluator._score(1024.0, 1.0), 0.0, rows)
        self.assertEqual(metrics["beats_reference"], 0)
        self.assertEqual(metrics["beat_spec_workload_passes"], rows)
        self.assertEqual(metrics["reference_score"], 0.0)
        rows[0]["metrics"]["batch_latency_seconds"] = True
        self.assertEqual(evaluator._metrics(0.0, 50.0, rows)["beats_reference"], 1)


if __name__ == "__main__":
    unittest.main()
