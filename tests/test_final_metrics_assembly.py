"""Exercise train/final result assembly without executing scientific workloads."""

# Install isolation before runtime imports.
import deployment_test_guard  # noqa: F401  # isort: skip
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jsonschema

ROOT = Path(__file__).resolve().parents[1] / "tasks/frontier-cs-2.0-demo/problems"
sys.path.insert(0, str(ROOT.parents[2] / "scripts"))
import task_manifest  # noqa: E402
TASKS = (
    "batch_local_counting",
    "bounded_evm_path_prediction",
    "evm_state_checkpoints",
    "graph_mining_method_selection",
    "bbq_packet_scheduling",
    "sparse_video_attention_policy",
    "polar_express",
)


class _AcceptingScope:
    """The shared verifier's edit scope, accepting every patch: the judged tree is `tree`."""

    CATEGORIES = {}

    def __init__(self, tree):
        self.tree = tree

    def check(self, *_args, **_kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(ok=True, tree=self.tree, changes=(), metrics={},
                               dispose=lambda: None)


class FinalMetricsAssembly(unittest.TestCase):
    @patch.dict(sys.modules)
    @patch.object(sys, "dont_write_bytecode", True)
    def test_train_final_pass_fail_and_zero(self):
        for name in TASKS:
            task = next(ROOT.glob("*/" + name))
            export_root = os.environ.get("FRONTIER_PRIVATE_METRICS_EXPORT_ROOT")
            if export_root and (Path(export_root) / name).is_dir():
                task = Path(export_root) / name
            native = task_manifest.is_native(task)
            # A task.toml task runs the shared verifier.
            path = (task_manifest.SHARED / "verifier/fcs2_verifier.py" if native
                    else task / "verifier/fcs2_verifier.py")
            for helper in ("beat_spec_feedback", "candidate_diagnostic"):
                sys.modules.pop(helper, None)
            sys.path.insert(0, str(path.parent))
            spec = importlib.util.spec_from_file_location("assembly_" + name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.path.pop(0)
            # A task.toml task has no judge/beat_spec.json: its metric names are in the
            # archived spec, and the shared verifier reads [feedback] from config.yaml.
            spec_path = (ROOT.parents[2] / "docs/task-authoring" / name / "provenance/beat_spec.json"
                         if native else task / "judge/beat_spec.json")
            beat_spec = json.loads(spec_path.read_text())
            for mode in ("train", "eval"):
                for wins, score in ((True, 17.0), (False, 17.0), (False, 0.0)):
                    with (
                        self.subTest(task=name, mode=mode, wins=wins, score=score),
                        tempfile.TemporaryDirectory() as td,
                    ):
                        scratch = Path(td)
                        (scratch / "beat_spec.json").write_text(json.dumps(beat_spec))
                        (scratch / "solution").write_bytes(b"")
                        if native:
                            (scratch / "config.yaml").write_bytes(task_manifest.generated_files(task)[
                                f"{task_manifest.GENERATED}/judge/config.yaml"])
                        metrics = {
                            "beats_reference": int(wins),
                            "reference_score": 15.0,
                            "margin": score - 15.0,
                            "beat_spec_workload_passes": [
                                {
                                    "workload": "private-workload",
                                    "metrics": {
                                        m["name"]: wins for m in beat_spec["metrics"]
                                    },
                                }
                            ],
                        }
                        if name == "sparse_video_attention_policy":
                            metrics["beat_spec_workload_passes"] = [
                                {
                                    "workload": f"private-workload-{index}",
                                    "metrics": {
                                        m["name"]: wins for m in beat_spec["metrics"]
                                    },
                                }
                                # Graded suite: 8 Wan + 4 HunyuanVideo workloads.
                                for index in range(12)
                            ]
                        result_path = scratch / "result.json"
                        # Keep real JSON result/evidence writing and real feedback projection.
                        with (
                            patch.dict(os.environ),
                            patch.object(module, "extract_payload"),
                            patch.object(
                                module,
                                "entrypoint_path",
                                return_value=scratch / "solution",
                            ),
                            patch.object(module, "stage_task", return_value=scratch),
                            patch.object(module, "load_evaluator",
                                         return_value=(None, None) if native else None),
                            patch.object(module, "load_edit_scope", create=True,
                                         return_value=(_AcceptingScope(scratch), None)),
                            patch.object(
                                module,
                                "run_evaluator",
                                return_value=(score, score, "ok", metrics),
                            ),
                            patch.object(module, "sha256_file", return_value="0" * 64),
                        ):
                            result = module.verify(
                                bundle=scratch,
                                entrypoint="solution.patch",
                                payload=scratch / "payload",
                                dataset=scratch / "dataset",
                                mode=mode,
                                result_path=result_path,
                                evidence_path=scratch / "evidence.tar",
                            )
                        self.assertEqual(json.loads(result_path.read_text()), result)
                        self.assertEqual(result["raw_score"], score)
                        self.assertTrue((scratch / "evidence.tar").is_file())
                        if mode == "eval":
                            if name in TASKS[:4]:
                                contract = json.loads(
                                    (
                                        ROOT.parents[2]
                                        / "tests/fixtures/platform-contracts/run-metrics-v1.schema.json"
                                    ).read_text()
                                )
                                jsonschema.validate(
                                    result["private_metrics"],
                                    {
                                        "$ref": "#/$defs/PrivateMetrics",
                                        "$defs": contract["$defs"],
                                    },
                                )
                            self.assertEqual(
                                result["private_metrics"]["beats_reference"], int(wins)
                            )
                            self.assertEqual(
                                result["private_metrics"]["reference_score"], 15.0
                            )
                            self.assertEqual(
                                result["metrics"], {"score_unbounded": score}
                            )
                        else:
                            self.assertNotIn("private_metrics", result)
                            self.assertIn("beat_spec_progress", result["metrics"])
                            self.assertNotIn("private-workload", json.dumps(result))
                            self.assertNotIn("reference_score", json.dumps(result))
                            self.assertNotIn("beats_reference", result["metrics"])


if __name__ == "__main__":
    unittest.main()
