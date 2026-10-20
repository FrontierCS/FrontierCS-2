from __future__ import annotations

# Enforce offline tests before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip
import copy
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import join_tree_contract as native
import validate_task_layout as layout

TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/Database/join_tree_rewriting"
EVALUATION = {"solver_visible_metrics": list(native.TRAIN_FEEDBACK)}


class NativeJoinTreeContractTests(unittest.TestCase):
    def test_reviewed_native_source_keeps_complete_private_final_and_aggregate_train(
        self,
    ):
        native.check_native_contract(TASK, EVALUATION)
        # Both the fallback and ordinary beat-spec path remain callable.
        layout.check_beat_spec(TASK, EVALUATION)

    def test_task_name_and_policy_label_never_waive_source_review(self):
        with tempfile.TemporaryDirectory() as text:
            root = Path(text) / "join_tree_rewriting"
            root.mkdir()
            with self.assertRaisesRegex(
                native.NativeContractFailure, "regular reviewed"
            ):
                native.check_native_contract(root, EVALUATION)
            for unreviewed in ([], ["candidate_median_us"],
                               [*native.TRAIN_FEEDBACK, "candidate_median_us"]):
                with self.assertRaisesRegex(
                    native.NativeContractFailure, "reviewed aggregate allowlist"
                ):
                    native.check_native_contract(
                        TASK, {"solver_visible_metrics": unreviewed}
                    )
            with self.assertRaisesRegex(
                native.NativeContractFailure, "unreviewed task"
            ):
                native.check_native_contract(
                    root.parent, EVALUATION
                )

    def test_every_reviewed_protocol_source_drift_is_rejected_before_execution(self):
        for member in native.REVIEWED_FILES:
            with self.subTest(member=member), tempfile.TemporaryDirectory() as text:
                root = Path(text) / "join_tree_rewriting"
                for name in native.REVIEWED_FILES:
                    destination = root / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(TASK / name, destination)
                (root / member).write_bytes((root / member).read_bytes() + b"\n")
                with mock.patch.object(
                    native,
                    "load_reviewed_functions",
                    side_effect=AssertionError("executed drift"),
                ):
                    with self.assertRaisesRegex(
                        native.NativeContractFailure, "unreviewed native"
                    ):
                        native.check_native_contract(
                            root, EVALUATION
                        )

    def test_symlink_or_substitute_spec_cannot_select_native_compatibility(self):
        with tempfile.TemporaryDirectory() as text:
            root = Path(text) / "join_tree_rewriting"
            for name in native.REVIEWED_FILES:
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(TASK / name, destination)
            source = root / "judge/evaluator.py"
            source.unlink()
            source.symlink_to(TASK / "judge/evaluator.py")
            with self.assertRaisesRegex(
                native.NativeContractFailure, "regular reviewed"
            ):
                native.check_native_contract(root, EVALUATION)
            source.unlink()
            shutil.copyfile(TASK / "judge/evaluator.py", source)
            # User-approved 2026-10-07: judge/beat_spec.json is now reviewed
            # source hash-pinned in REVIEWED_FILES, so a substitute/forged spec
            # is rejected by the hash pin (stricter than forbidding the file).
            (root / "judge/beat_spec.json").write_text("{}")
            with self.assertRaisesRegex(
                native.NativeContractFailure,
                "unreviewed native protocol source: judge/beat_spec.json",
            ):
                native.check_native_contract(root, EVALUATION)

    def test_real_comparison_never_drops_missing_candidate_or_normalizes_by_reference(
        self,
    ):
        expected = native.check_source(TASK, EVALUATION)
        comparison, adapter = native.load_reviewed_functions(TASK)
        queries = {q: native.query_probe(comparison, q) for q in expected}
        missing = comparison["compare_query"](
            identity=expected[-1],
            correctness=None,
            timing=None,
            identical=False,
            repetitions=5,
            failure="candidate_not_measured",
        )
        queries[expected[-1]] = missing
        score, metrics = comparison["summarize_queries"](expected, queries)
        self.assertEqual(score, 38 * 50.0 / 39)
        self.assertEqual(metrics["expected_case_count"], 39)
        self.assertEqual(metrics["case_count"], 38)
        self.assertEqual(adapter["private_verdict"](metrics)["beats_reference"], 0)
        self.assertEqual(
            len(adapter["private_verdict"](metrics)["beat_spec_progress"]), 78
        )
        self.assertNotIn("reference_score", metrics)
        incomplete = copy.deepcopy(metrics)
        incomplete["queries"].pop(expected[-1])
        with self.assertRaises(adapter["VerifierFailure"]):
            adapter["private_verdict"](incomplete)

    def test_actual_reference_mismatch_is_an_error_not_a_candidate_zero(self):
        comparison, _ = native.load_reviewed_functions(TASK)
        q = native.query_probe(comparison, "dev/1")
        correctness = copy.deepcopy(q["correctness_measurements"])
        correctness["matches_original"]["Reference"] = [False]
        with self.assertRaises(comparison["MeasurementError"]):
            comparison["compare_query"](
                identity="dev/1",
                correctness=correctness,
                timing=q["timing_measurements"],
                identical=False,
                repetitions=5,
            )

    def test_projection_probe_rejects_public_leak_partial_final_and_forged_verdict(
        self,
    ):
        expected = native.check_source(TASK, EVALUATION)
        for fault in ("train_leak", "eval_leak", "old_19_final", "forged_verdict"):
            with self.subTest(fault=fault):
                comparison, adapter = native.load_reviewed_functions(TASK)
                original = adapter["verify"]

                def changed_verify(
                    *, adapter=adapter, fault=fault, original=original, **kwargs
                ):
                    result = original(**kwargs)
                    if fault == "train_leak" and kwargs["mode"] == "train":
                        result["metrics"]["reference_median_us"] = 20.0
                    elif fault == "eval_leak" and kwargs["mode"] == "eval":
                        result["metrics"]["beats_reference"] = 1
                    elif fault == "old_19_final" and kwargs["mode"] == "eval":
                        result["private_metrics"]["beat_spec_progress"] = dict(
                            list(
                                result["private_metrics"]["beat_spec_progress"].items()
                            )[:38]
                        )
                    elif fault == "forged_verdict" and kwargs["mode"] == "eval":
                        result["private_metrics"]["beats_reference"] = 1
                    os.environ["FRONTIER_SUBMISSION_ROLE"] = "probe-contamination"
                    adapter["write_json"](kwargs["result_path"], result)
                    return result

                adapter["verify"] = changed_verify
                with mock.patch.dict(
                    os.environ, {"FRONTIER_SUBMISSION_ROLE": "prior-role"}
                ):
                    with self.assertRaises(native.NativeContractFailure):
                        native.check_projection(TASK, comparison, adapter, expected)
                    self.assertEqual(
                        os.environ["FRONTIER_SUBMISSION_ROLE"], "prior-role"
                    )

    def test_other_tasks_still_require_a_real_beat_spec(self):
        with tempfile.TemporaryDirectory() as text:
            root = Path(text) / "other_task"
            (root / "judge").mkdir(parents=True)
            with self.assertRaisesRegex(layout.LayoutFailure, "missing regular file"):
                layout.check_beat_spec(root, {"solver_visible_metrics": []})


if __name__ == "__main__":
    unittest.main()
