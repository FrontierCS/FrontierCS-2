"""fast_kv_compaction and progent_privilege_control: absolute continuous scores.

Recorded eval-mode verifier metrics (fixtures/absolute-score/recorded.json:
final and development verifications, with the values of the
offline absolute rescoring) must come out of the evaluators unchanged.
"""

import deployment_test_guard  # noqa: F401  # isort: skip

import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "tasks/frontier-cs-2.0-demo/problems"
KV = PROBLEMS / "Machine Learning/fast_kv_compaction"
PROGENT = PROBLEMS / "Security/progent_privilege_control"
RECORDED = json.loads((Path(__file__).parent / "fixtures/absolute-score/recorded.json").read_text())
SUITES = ("banking", "slack", "travel", "workspace")


def load_evaluator(task: Path, name: str):
    """Import judge/evaluator.py without leaving its sibling modules in sys.modules."""
    judge = str(task / "judge")
    before = set(sys.modules)
    sys.path.insert(0, judge)
    try:
        spec = importlib.util.spec_from_file_location(name, task / "judge/evaluator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(judge)
        for added in set(sys.modules) - before:
            if str(getattr(sys.modules[added], "__file__", "") or "").startswith(judge + "/"):
                del sys.modules[added]


KV_EV = load_evaluator(KV, "fast_kv_compaction_evaluator")
PROGENT_EV = load_evaluator(PROGENT, "progent_privilege_control_evaluator")


def kv_inputs(record):
    pairs = record["paired_workload_metrics"]
    workloads = tuple(row["workload"] for row in pairs["workloads"])
    rows = [dict(row["candidate"]) for row in pairs["workloads"]]
    targets = {row["workload"]: dict(row["target"]) for row in pairs["workloads"]}
    return pairs["mode"], workloads, rows, targets


class FastKvAbsoluteScore(unittest.TestCase):
    def test_recorded_metrics_reproduce_offline_rescoring(self):
        records = RECORDED["fast_kv_compaction"]
        self.assertGreaterEqual(sum(r["mode"] == "eval" for r in records), 5)
        for record in records:
            with self.subTest(record=record["record"]):
                mode, workloads, rows, targets = kv_inputs(record)
                score = KV_EV.absolute_score(mode, workloads, rows, targets)
                self.assertEqual(round(score, 4), record["score"])
                metrics = KV_EV._complete_metrics(mode, workloads, rows, targets, score)
                self.assertEqual(round(metrics["reference_score"], 4), record["reference_score"])
                self.assertAlmostEqual(metrics["margin"], score - metrics["reference_score"], places=12)
                if "published_margin" in record:
                    self.assertLessEqual(abs(metrics["margin"] - record["published_margin"]), 1e-4)
                # The same inputs give the earlier platform score (kept for operators).
                relative = [KV_EV.gate.article_score(mode, row, targets[w]) for w, row in zip(workloads, rows)]
                self.assertAlmostEqual(sum(relative) / len(relative), record["raw_score"], places=9)

    def test_reference_scores_from_frozen_targets(self):
        manifest = json.loads((KV / "judge/targets/manifest.json").read_text())
        for workloads, expected in ((KV_EV.PUBLIC_ARTICLES + KV_EV.HIDDEN_ARTICLES, 60.6278),
                                    (KV_EV.PUBLIC_ARTICLES, 60.6548)):
            targets = KV_EV.gate.validate_manifest(manifest, workloads)
            self.assertEqual(round(KV_EV.reference_absolute_score(workloads, targets), 4), expected)

    def test_formula(self):
        target = {"questions": 20, "correct": 10.0, "seconds": 10.0, "bytes": 1000}
        def article(mode, correct=10, seconds=10.0, footprint=1000):
            row = {"questions": 20, "correct": correct, "seconds": seconds, "bytes": footprint}
            return KV_EV.absolute_score(mode, ("w",), [row], {"w": target})
        self.assertAlmostEqual(article("time"), 100 * 0.5 * 0.875)
        self.assertAlmostEqual(article("time", seconds=0.01), 100 * 0.5)          # 1000x: e = 1
        self.assertAlmostEqual(article("time", seconds=10_000.0), 100 * 0.5 * 0.75)  # 1/1000x: e = 0
        self.assertAlmostEqual(article("time", seconds=1.0), 100 * 0.5 * (0.75 + 0.25 * (0.5 + 1 / 6)))
        # The other axis only discounts.
        self.assertAlmostEqual(article("time", seconds=1.0, footprint=10), article("time", seconds=1.0))
        self.assertAlmostEqual(article("time", seconds=1.0, footprint=10_000), article("time"))
        self.assertAlmostEqual(article("bytes", footprint=100), article("time", seconds=1.0))
        self.assertAlmostEqual(article("bytes", footprint=100, seconds=100.0), article("bytes"))
        # Accuracy scales the article without a gate.
        self.assertAlmostEqual(article("time", correct=0, seconds=1.0), 0.0)
        self.assertAlmostEqual(article("time", correct=20), 2 * article("time"))
        with self.assertRaises(ValueError):
            article("other")


def progent_inputs(record):
    """Per-suite summaries as decide() receives them, rebuilt from the recorded counts."""
    metrics = record["metrics"]
    reference_train = json.loads((PROGENT / "judge/reference_train.json").read_text())
    per_suite = {}
    for suite in SUITES:
        attacked = metrics[f"{suite}_attacked_episodes"]
        successes = round(metrics[f"{suite}_utility_under_attack"] * attacked)
        candidate = {"attacked_episodes": attacked, "breached": metrics[f"{suite}_breached_episodes"],
                     "utility_under_attack": successes / attacked if attacked else 0.0,
                     "benign_utility": metrics[f"{suite}_benign_utility"], "worker_failures": 0, "capped": 0,
                     "tokens": 0}
        reference = (metrics["all_suite_reference"][suite] if record["mode"] == "eval"
                     else reference_train[suite])
        per_suite[suite] = (candidate, reference)
    return per_suite


def run_decide(record):
    role = "final" if record["mode"] == "eval" else "train"
    prefix = "final/" if role == "final" else "train/"
    rows = [dict(workload=prefix + s, metrics={}) for s in SUITES]
    metrics = {"beats_reference": 0, "beat_spec_workload_passes": rows, "reference_score": 0.0, "margin": 0.0}
    passed, score = PROGENT_EV.decide(progent_inputs(record), rows, metrics, role)
    return passed, score, metrics


class ProgentAbsoluteScore(unittest.TestCase):
    def test_recorded_metrics_reproduce_offline_rescoring(self):
        records = RECORDED["progent_privilege_control"]
        self.assertGreaterEqual(sum(r["mode"] == "eval" for r in records), 15)
        for record in records:
            with self.subTest(record=record["record"]):
                passed, score, metrics = run_decide(record)
                self.assertEqual(round(score, 4), record["score"])
                for suite in SUITES:
                    for name in ("utility_under_attack", "breached_episodes", "attacked_episodes"):
                        key = f"{suite}_{name}"
                        self.assertEqual(metrics[key], record["metrics"][key])
                # The earlier platform score is unchanged (kept for operators).
                self.assertAlmostEqual(metrics["reference_relative_score"], record["raw_score"], places=9)
                self.assertEqual(metrics["beats_reference"], int(record["raw_score"] == 100.0))
                if record["mode"] == "eval":
                    self.assertEqual(round(metrics["reference_score"], 4), record["reference_score"])
                    self.assertAlmostEqual(metrics["margin"], score - metrics["reference_score"], places=12)
                    self.assertLessEqual(abs(metrics["margin"] - record["published_margin"]), 1e-4)
                else:
                    self.assertEqual((metrics["reference_score"], metrics["margin"]), (0.0, 0.0))

    def test_suite_score(self):
        self.assertEqual(PROGENT_EV.suite_score(0.5, 0, 80), 50.0)
        self.assertAlmostEqual(PROGENT_EV.suite_score(0.5, 20, 80), 37.5)
        self.assertEqual(PROGENT_EV.suite_score(0.0, 0, 0), 0.0)
        self.assertTrue(math.isfinite(PROGENT_EV.suite_score(1.0, 80, 80)))


if __name__ == "__main__":
    unittest.main()
