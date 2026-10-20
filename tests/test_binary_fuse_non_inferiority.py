"""Guard the Binary Fuse certified non-inferiority accuracy contract (issue #321).

The three performance metrics remain strict per-workload wins. The two accuracy
metrics are certified non-inferiority guards against preregistered reference-only
anchors: a workload fails only when the candidate's certified lower bound exceeds
its anchor. These checks exercise the published judge modules with synthetic
counts; they are protocol checks, not measurements.
"""

from __future__ import annotations

import deployment_test_guard  # noqa: F401 # isort: skip

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = Path(
    os.environ.get(
        "BINARY_FUSE_EXPORT",
        str(ROOT / "tasks/frontier-cs-2.0-demo/problems/Theory/binary_fuse_filter"),
    )
)
QUERIES = 8 * 1048576


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


class BinaryFuseNonInferiorityTests(unittest.TestCase):
    def setUp(self):
        old_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        self.addCleanup(lambda: setattr(sys, "dont_write_bytecode", old_bytecode))
        old_modules, old_path = dict(sys.modules), list(sys.path)

        def restore():
            sys.path[:] = old_path
            for key in set(sys.modules) - set(old_modules):
                sys.modules.pop(key, None)
            sys.modules.update(old_modules)

        self.addCleanup(restore)
        sys.path.insert(0, str(TASK / "judge"))
        for name in ("accuracy_metrics", "submission_patch", "evaluator"):
            sys.modules.pop(name, None)
        self.metrics = module(TASK / "judge/accuracy_metrics.py", "accuracy_metrics")
        self.evaluator = module(TASK / "judge/evaluator.py", "evaluator")
        self.profile = json.loads(
            (TASK / "judge/reference/statistical-calibration.json").read_text()
        )

    # ----------------------------------------------------------- helpers

    def sample(self, false_positives, resident_bytes=2261032, n=1000000):
        rows = [
            {
                "effective_false_positives": false_positives,
                "effective_absent_queries": QUERIES,
                "resident_bytes": resident_bytes,
            }
        ] * 5
        return self.metrics.summarize_accuracy(rows, n)

    def anchors(self):
        return self.evaluator.load_statistical_calibration(require_runtime=False)

    # ------------------------------------------------- preregistration

    def test_anchors_are_preregistered_reference_only_values(self):
        anchors = self.anchors()
        rounds = self.profile["reference_repeat_summaries"]
        self.assertEqual(self.profile["anchors"], anchors)
        self.assertEqual(self.profile["rule"], self.metrics.RULE)
        self.assertIs(self.profile["successor_solver_results_seen"], False)
        self.assertEqual(sorted(anchors), sorted(self.evaluator.workload_ids(True)))
        for workload, row in anchors.items():
            points = [r[workload] for r in rounds]
            nominal = self.metrics.nominal_false_positive_rate(workload)
            self.assertEqual(
                row["effective_fpr"],
                max(nominal, max(p["effective_fpr"] for p in points)),
            )
            self.assertEqual(
                row["space_overhead_pct"], max(p["space_overhead_pct"] for p in points)
            )
            self.assertGreaterEqual(row["effective_fpr"], nominal)

    def test_calibration_binds_the_exact_measurement_sources(self):
        manifest = json.loads((TASK / "judge/statistical_calibration.json").read_text())
        binding = self.evaluator.measurement_binding()
        self.assertEqual(manifest["measurement_sha256"], binding)
        self.assertEqual(self.profile["measurement_sha256"], binding)
        digest = "sha256:" + hashlib.sha256(
            (TASK / "judge/reference/statistical-calibration.json").read_bytes()
        ).hexdigest()
        self.assertEqual(manifest["statistical_profile_sha256"], digest)
        with tempfile.TemporaryDirectory() as temp:
            judge = Path(temp) / "judge"
            shutil.copytree(TASK / "judge", judge)
            spec = judge / "beat_spec.json"
            spec.write_text(spec.read_text().replace("<= anchor", "<= 2 * anchor"))
            mutated = module(judge / "evaluator.py", "mutated_evaluator")
            with self.assertRaises(self.metrics.CalibrationRequired):
                mutated.load_statistical_calibration(require_runtime=False)

    def test_production_gate_fails_closed_without_a_runtime_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            judge = Path(temp) / "judge"
            shutil.copytree(TASK / "judge", judge)
            (judge / "reference/runtime-qualification.json").unlink(missing_ok=True)
            gated = module(judge / "evaluator.py", "gated_evaluator")
            with self.assertRaises(self.metrics.RuntimeQualificationRequired):
                gated.load_statistical_calibration()
            self.assertTrue(gated.load_statistical_calibration(require_runtime=False))

    # -------------------------------------------------- the guard itself

    def test_reference_rounds_never_fail_their_own_anchors(self):
        anchors = self.anchors()
        m = QUERIES
        for workload, row in anchors.items():
            for repeat in self.profile["reference_repeat_summaries"]:
                k = round(repeat[workload]["effective_fpr"] * m)
                self.assertEqual(k / m, repeat[workload]["effective_fpr"])
                lower, upper = self.metrics.fpr_interval(k, m)
                self.assertTrue(
                    self.metrics.non_inferior(
                        {"point": k / m, "lower": lower, "upper": upper},
                        row["effective_fpr"],
                    ),
                    workload,
                )

    def test_nominal_filters_pass_and_certified_excess_fails(self):
        anchors = self.anchors()
        cases = {
            # workload, nominal count, counts that must pass, counts that must fail
            "b16-n1000000-f0-s20261007-mix0": (128, (0, 100, 128, 150, 190), (260, 512)),
            "b8-n1000000-f0-s20261007-mix0": (32768, (30000, 32768, 33500), (35500, 65536)),
        }
        for workload, (nominal, passing, failing) in cases.items():
            anchor = anchors[workload]["effective_fpr"]
            self.assertAlmostEqual(nominal / QUERIES, self.metrics.nominal_false_positive_rate(workload))
            for k in passing:
                with self.subTest(workload=workload, k=k):
                    self.assertTrue(self.metrics.non_inferior(self.sample(k)["effective_fpr"], anchor))
            for k in failing:
                with self.subTest(workload=workload, k=k):
                    self.assertFalse(self.metrics.non_inferior(self.sample(k)["effective_fpr"], anchor))

    def test_space_guard_rejects_certified_overhead_above_the_anchor(self):
        anchors = self.anchors()
        workload = "b16-n1000000-f0-s20261007-mix0"
        anchor = anchors[workload]["space_overhead_pct"]
        reference = self.sample(114, resident_bytes=2261032)
        self.assertTrue(self.metrics.non_inferior(reference["space_overhead_pct"], anchor))
        smaller = self.sample(128, resident_bytes=2100000)
        self.assertTrue(self.metrics.non_inferior(smaller["space_overhead_pct"], anchor))
        bloated = self.sample(128, resident_bytes=3000000)
        self.assertFalse(self.metrics.non_inferior(bloated["space_overhead_pct"], anchor))

    def test_unresolved_domain_and_invalid_inputs_cannot_pass(self):
        anchor = self.anchors()["b16-n1000000-f0-s20261007-mix0"]["space_overhead_pct"]
        zero = self.sample(0)
        self.assertIsNone(zero["space_overhead_pct"]["point"])
        self.assertFalse(self.metrics.non_inferior(zero["space_overhead_pct"], anchor))
        everything = self.sample(QUERIES)
        self.assertFalse(self.metrics.non_inferior(everything["space_overhead_pct"], anchor))
        with self.assertRaises(ValueError):
            self.metrics.non_inferior({"point": 0.5, "lower": 0.6, "upper": 0.7}, anchor)
        with self.assertRaises(ValueError):
            self.metrics.non_inferior({"point": float("nan"), "lower": 0.0, "upper": 1.0}, anchor)
        for bad in (None, float("inf"), "0.1"):
            with self.subTest(anchor=bad), self.assertRaises(self.metrics.CalibrationRequired):
                self.metrics.non_inferior(self.sample(128)["effective_fpr"], bad)

    def test_anchor_derivation_rejects_incomplete_or_unresolved_rounds(self):
        rounds = copy.deepcopy(self.profile["reference_repeat_summaries"])
        ids = self.evaluator.workload_ids(True)
        with self.assertRaises(self.metrics.CalibrationRequired):
            self.metrics.reference_anchors(rounds[:4], ids)
        broken = copy.deepcopy(rounds)
        broken[0][ids[0]]["effective_fpr"] = 0.0
        with self.assertRaises(self.metrics.CalibrationRequired):
            self.metrics.reference_anchors(broken, ids)
        broken = copy.deepcopy(rounds)
        del broken[1][ids[1]]
        with self.assertRaises(self.metrics.CalibrationRequired):
            self.metrics.reference_anchors(broken, ids)

    # ------------------------------------------------- scoring conjunction

    def test_score_rows_keeps_strict_performance_wins_and_the_conjunction(self):
        anchors = self.anchors()
        workload = "b16-n1000000-f0-s20261007-mix0"
        accuracy = self.sample(128, resident_bytes=2200000)
        reference = self.sample(114, resident_bytes=2261032)

        def row(resident, build, lookup, acc):
            return {
                "workload": workload,
                "candidate": {"resident_bytes": resident, "build_ns": build, "lookup_ns": lookup, **acc},
                "reference": {"resident_bytes": 2261032, "build_ns": 100.0, "lookup_ns": 20.0, **reference},
            }

        score, decisions, beats = self.evaluator.score_rows(
            [row(2200000, 90.0, 19.0, accuracy)], anchors
        )
        self.assertEqual(beats, 1)
        self.assertTrue(all(decisions[0]["metrics"].values()))
        self.assertGreater(score, 50.0)
        # A tie on any performance metric, or a certified accuracy excess, fails.
        _, decisions, beats = self.evaluator.score_rows([row(2261032, 90.0, 19.0, accuracy)], anchors)
        self.assertEqual((beats, decisions[0]["metrics"]["resident_bytes"]), (0, False))
        _, decisions, beats = self.evaluator.score_rows(
            [row(2200000, 90.0, 19.0, self.sample(512, resident_bytes=2200000))], anchors
        )
        self.assertEqual((beats, decisions[0]["metrics"]["effective_fpr"]), (0, False))
        with self.assertRaises(self.metrics.CalibrationRequired):
            self.evaluator.score_rows([row(2200000, 90.0, 19.0, accuracy)], {workload: {"effective_fpr": 1e-5}})

    # ---------------------------------------------- existing hard gates

    def test_validity_gates_reject_always_true_filters_and_false_negatives(self):
        e = self.evaluator
        q = e.QUERIES
        good = {
            "construction": True, "false_negatives": 0, "mixed_false_negatives": 0, "leaked_bytes": 0,
            "absent_queries": q, "false_positives": 16, "effective_false_positives": 128,
            "effective_absent_queries": 8 * q, "mixed_false_positives": [16, 8, 0, 12],
            "mixed_absent_queries": [q, q // 2, 0, 3 * q // 4], "shuffled_hit_queries": q // 4,
            "resident_bytes": 2261032, "build_ns": 1000.0, "lookup_ns": [10.0, 10.0, 10.0, 10.0],
        }
        e.validate_sample(good, 16, q)
        for name, change in (
            ("always_true", {"false_positives": q, "effective_false_positives": 8 * q, "mixed_false_positives": [q, q // 2, 0, 3 * q // 4]}),
            ("false_negative", {"false_negatives": 1}),
            ("leaked_allocation", {"leaked_bytes": 64}),
            ("short_effective_stream", {"effective_absent_queries": 4 * q}),
            ("stale_output", {"absent_queries": q - 1}),
        ):
            with self.subTest(case=name), self.assertRaises(e.InvalidCandidate):
                e.validate_sample({**good, **change}, 16, q)


if __name__ == "__main__":
    unittest.main()
