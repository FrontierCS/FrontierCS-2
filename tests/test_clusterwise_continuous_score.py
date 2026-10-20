"""Clusterwise continuous score is asymptotic: parity is 50 and 100 is unreachable."""

import deployment_test_guard  # noqa: F401 # isort: skip

import ast
import math
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tasks/frontier-cs-2.0-demo/problems/Arch/clusterwise_spgemm/judge/evaluator.py"

# The full evaluator imports modules absent from the repository test runner;
# extract the standard-library-only score function.
tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
namespace = {"math": math, "statistics": statistics}
exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                              and node.name == "_continuous_score"], type_ignores=[]),
             str(EVALUATOR), "exec"), namespace)
score = namespace["_continuous_score"]


class ContinuousScoreTests(unittest.TestCase):
    def test_parity_is_fifty_and_score_is_monotone_below_one_hundred(self):
        self.assertAlmostEqual(score([10.0] * 8, [10.0] * 8), 50.0)
        self.assertAlmostEqual(score([5.0] * 8, [10.0] * 8), 100 * 2 / 3)
        self.assertAlmostEqual(score([20.0] * 8, [10.0] * 8), 100 / 3)
        previous = 0.0
        for speedup in (0.01, 0.5, 1.0, 1.5, 2.0, 4.0, 100.0, 1e6):
            value = score([1.0], [speedup])
            self.assertGreater(value, previous)
            self.assertLess(value, 100.0)
            previous = value

    def test_geometric_mean_over_workloads(self):
        self.assertAlmostEqual(score([1.0, 4.0], [2.0, 2.0]), 50.0)
        self.assertAlmostEqual(score([1.0, 1.0], [2.0, 8.0]), 80.0)

    def test_finite_extremes_stay_bounded_without_ratio_overflow(self):
        smallest = math.nextafter(0.0, 1.0)
        largest = float.fromhex('0x1.fffffffffffffp+1023')
        for candidate, reference in ((1.0, 1e17), (1.0, 1e307),
                                     (1e-308, 1e308), (1e308, 1e-308),
                                     (smallest, largest), (largest, smallest)):
            with self.subTest(candidate=candidate, reference=reference):
                value = score([candidate], [reference])
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0.0)
                self.assertLess(value, 100.0)
        self.assertEqual(score([1.0], [1e17]), math.nextafter(100.0, 0.0))

    def test_balanced_extreme_ratios_retain_parity(self):
        smallest = math.nextafter(0.0, 1.0)
        largest = float.fromhex('0x1.fffffffffffffp+1023')
        self.assertEqual(score([smallest, largest], [largest, smallest]), 50.0)

    def test_rejects_incomplete_or_invalid_measurements(self):
        for candidate, reference in (([], []), ([1.0], [1.0, 2.0]), ([0.0], [1.0]),
                                     ([1.0], [math.inf]), ([math.nan], [1.0])):
            with self.assertRaises(ValueError):
                score(candidate, reference)


if __name__ == "__main__":
    unittest.main()
