"""flow_opd: the per-workload gate stays strict below the scorer's maximum and
accepts equality only on ceiling rows, so PASS is attainable (issue #314)."""

import deployment_test_guard  # noqa: F401  # isort: skip

import importlib.util
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = Path(os.environ.get("FRONTIER_FLOW_OPD_EXPORT", str(
    ROOT / "tasks/frontier-cs-2.0-demo/problems/Machine Learning/flow_opd"
)))
spec = importlib.util.spec_from_file_location("flow_opd_scoring", TASK / "judge/judge/scoring.py")
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)

# The frozen final reference from the issue: two workloads at the 1.0 ceiling.
REF = [1.0, 1.0, 0.2308, 0.2308, 0.8235, 0.8235, 0.8333, 0.0]
IDS = [f"final-{i}" for i in range(len(REF))]


def run(candidate_values):
    reference = dict(zip(IDS, REF))
    candidate = dict(zip(IDS, candidate_values))
    _, _, _, metrics = scoring.aggregate(
        IDS, candidate, reference,
        candidate_valid=True, reference_valid=True, final=True,
    )
    return metrics


class FlowOpdCeilingGate(unittest.TestCase):
    def test_beats_workload_rule(self):
        # Below the maximum: strictly greater only.
        self.assertTrue(scoring.beats_workload(0.5, 0.4))
        self.assertFalse(scoring.beats_workload(0.4, 0.4))
        self.assertFalse(scoring.beats_workload(0.3, 0.4))
        # At the maximum: equality passes; anything below fails; cannot exceed.
        self.assertTrue(scoring.beats_workload(1.0, 1.0))
        self.assertFalse(scoring.beats_workload(0.9999, 1.0))

    def test_pass_is_attainable_reaching_ceiling_and_beating_the_rest(self):
        candidate = [1.0, 1.0, 0.5, 0.5, 0.9, 0.9, 0.9, 0.5]
        metrics = run(candidate)
        self.assertEqual(metrics["beats_reference"], 1)
        self.assertTrue(all(r["metrics"]["ocr_reward"] for r in metrics["beat_spec_workload_passes"]))

    def test_reference_copy_still_fails_on_non_ceiling_rows(self):
        metrics = run(list(REF))
        self.assertEqual(metrics["beats_reference"], 0)
        passes = {r["workload"]: r["metrics"]["ocr_reward"] for r in metrics["beat_spec_workload_passes"]}
        # The two ceiling rows tie at 1.0 and pass; every below-ceiling row fails.
        self.assertTrue(passes["final-0"] and passes["final-1"])
        self.assertFalse(any(passes[i] for i in IDS[2:]))

    def test_missing_the_ceiling_defeats_the_conjunction(self):
        candidate = [0.9999, 1.0, 0.5, 0.5, 0.9, 0.9, 0.9, 0.5]
        metrics = run(candidate)
        self.assertEqual(metrics["beats_reference"], 0)
        passes = {r["workload"]: r["metrics"]["ocr_reward"] for r in metrics["beat_spec_workload_passes"]}
        self.assertFalse(passes["final-0"])


if __name__ == "__main__":
    unittest.main()
