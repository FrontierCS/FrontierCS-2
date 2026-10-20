from __future__ import annotations

# Enforce offline tests before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import native_beat_spec  # noqa: E402
import validate_task_layout as layout  # noqa: E402
import vllm_native_contract as native  # noqa: E402

TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/MLSys/vllm_llm_serving_optimization"
EVALUATION = {"solver_visible_metrics": native.SOLVER_FEEDBACK, "authored_max_dev_submissions": 25}


def reviewed_copy(root: Path) -> None:
    for name in native.REVIEWED_FILES:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TASK / name, destination)


class NativeVllmContractTests(unittest.TestCase):
    def test_train_projection_exact_whitelist_rejects_leaked_or_missing_feedback(self):
        functions = native.load_reviewed_functions(TASK)
        native.check_projection(functions)
        original = functions['result_projection']
        for mutation in ('leak', 'missing'):
            def altered(result, mode):
                value = original(result, mode)
                if mode == 'train':
                    if mutation == 'leak':
                        value['metrics']['reference_threshold'] = 999
                    else:
                        value['metrics'].pop('swe_public_geometric_mean_latency_seconds')
                return value
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    native.NativeContractFailure, 'reviewed public aggregates'):
                native.check_projection(dict(functions, result_projection=altered))

    def test_reviewed_native_source_projects_every_group_decision_and_public_train(self):
        native.check_native_contract(TASK, EVALUATION)
        self.assertIn("vllm_llm_serving_optimization", native_beat_spec.NATIVE_SPECS)
        layout.check_beat_spec(TASK, EVALUATION)

    def test_task_name_feedback_label_and_aggregation_never_waive_source_review(self):
        with tempfile.TemporaryDirectory() as text:
            root = Path(text) / "vllm_llm_serving_optimization"
            root.mkdir()
            with self.assertRaisesRegex(native.NativeContractFailure, "regular reviewed"):
                native.check_native_contract(root, EVALUATION)
            with self.assertRaisesRegex(native.NativeContractFailure, "public whitelist"):
                native.check_native_contract(TASK, dict(EVALUATION, solver_visible_metrics=["p99"]))
            with self.assertRaisesRegex(native.NativeContractFailure, "unreviewed task"):
                native.check_native_contract(root.parent, EVALUATION)
            with self.assertRaisesRegex(layout.LayoutFailure, "differ from the native"):
                with mock.patch.dict(
                    native_beat_spec.NATIVE_SPECS["vllm_llm_serving_optimization"],
                    {"metrics": ["paired_timing"]},
                ):
                    layout.check_beat_spec(TASK, EVALUATION)

    def test_every_reviewed_protocol_source_drift_is_rejected_before_execution(self):
        for member in native.REVIEWED_FILES:
            with self.subTest(member=member), tempfile.TemporaryDirectory() as text:
                root = Path(text) / "vllm_llm_serving_optimization"
                reviewed_copy(root)
                (root / member).write_bytes((root / member).read_bytes() + b"\n")
                with mock.patch.object(
                    native, "load_reviewed_functions", side_effect=AssertionError("executed drift")
                ):
                    with self.assertRaisesRegex(native.NativeContractFailure, "unreviewed native"):
                        native.check_native_contract(root, EVALUATION)

    def test_symlink_or_substitute_spec_cannot_select_native_compatibility(self):
        with tempfile.TemporaryDirectory() as text:
            root = Path(text) / "vllm_llm_serving_optimization"
            reviewed_copy(root)
            source = root / "verifier/verify.py"
            source.unlink()
            source.symlink_to(TASK / "verifier/verify.py")
            with self.assertRaisesRegex(native.NativeContractFailure, "regular reviewed"):
                native.check_native_contract(root, EVALUATION)
            source.unlink()
            shutil.copyfile(TASK / "verifier/verify.py", source)
            (root / "judge/beat_spec.json").write_text("{}")
            with self.assertRaisesRegex(
                native.NativeContractFailure, "unreviewed native protocol source: judge/beat_spec.json"
            ):
                native.check_native_contract(root, EVALUATION)
            native.check_native_contract(TASK, EVALUATION)

    def test_projection_probe_rejects_a_verifier_that_hides_or_invents_decisions(self):
        functions = native.load_reviewed_functions(TASK)
        native.check_projection(functions)
        hidden = dict(functions)
        original = functions["result_projection"]

        def without_progress(result, mode):
            value = original(result, mode)
            value.get("private_metrics", {}).pop("beat_spec_progress", None)
            return value

        hidden["result_projection"] = without_progress
        with self.assertRaisesRegex(native.NativeContractFailure, "count every group decision"):
            native.check_projection(hidden)
        generous = dict(functions)

        def always_passing(result, mode):
            value = original(result, mode)
            if "private_metrics" in value:
                value["private_metrics"]["beats_reference"] = 1
            return value

        generous["result_projection"] = always_passing
        with self.assertRaisesRegex(native.NativeContractFailure, "failed group decision"):
            native.check_projection(generous)
        chatty = dict(functions)
        chatty["result_projection"] = lambda result, mode: {**original(result, mode), "metrics": {"p99": 1.0}}
        with self.assertRaisesRegex(native.NativeContractFailure, "reviewed public aggregates"):
            native.check_projection(chatty)

    def test_synthetic_operator_results_follow_the_reviewed_verdict_shape(self):
        complete = native.operator_result("eval")
        self.assertTrue(complete["pass"])
        self.assertEqual(set(complete["reference_verdict"]["groups"]), set(native.GROUPS["eval"]))
        self.assertEqual(complete["validity"]["groups"], {})
        self.assertIs(complete["bfcl_flip_gate_enabled"], False)
        failing = native.operator_result("eval", failing={"swe/public": ("correctness",)})
        self.assertFalse(failing["pass"])
        self.assertFalse(failing["reference_verdict"]["groups"]["swe/public"]["comparisons"]["correctness"])
        self.assertTrue(failing["valid"])
        train = native.operator_result("train")
        self.assertEqual(set(train["reference_verdict"]["groups"]), {"swe/public", "bfcl/public"})


if __name__ == "__main__":
    unittest.main()
