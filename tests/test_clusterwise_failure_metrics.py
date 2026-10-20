"""Clusterwise gate failures retain a valid private beat-spec denominator."""

import deployment_test_guard  # noqa: F401 # isort: skip

import ast
import importlib.util
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/Arch/clusterwise_spgemm"
JUDGE = TASK / "judge"

# The full evaluator imports numpy and compiler helpers that are intentionally
# absent from the repository test runner. These failure-contract helpers are
# standard-library-only, so extract and test those exact definitions here.
source = (JUDGE / "evaluator.py").read_text(encoding="utf-8")
tree = ast.parse(source)
selected = [
    node
    for node in tree.body
    if (
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "BEAT_METRICS"
            for target in node.targets
        )
    )
    or (
        isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_failure_workloads",
            "_failure_metrics",
            "_is_resource_failure",
            "evaluate",
        }
    )
]
namespace = {
    "ROOT": JUDGE,
    "json": json,
    "traceback": traceback,
    "time": time,
    "tempfile": tempfile,
    "os": os,
    "Path": Path,
    "sys": sys,
    "shutil": shutil,
}
exec(
    compile(
        ast.Module(body=selected, type_ignores=[]), str(JUDGE / "evaluator.py"), "exec"
    ),
    namespace,
)


class ClusterwiseFailureMetricsTests(unittest.TestCase):
    def test_gate_failure_has_every_final_workload_and_metric(self):
        metrics = namespace["_failure_metrics"](
            True,
            reason="native process failed",
            completed_workloads=0,
            runtime={"memory_peak_bytes": 123},
        )

        self.assertEqual(metrics["beats_reference"], 0)
        self.assertEqual(metrics["reference_score"], 50.0)
        self.assertEqual(metrics["margin"], -50.0)
        rows = metrics["beat_spec_workload_passes"]
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [row["workload"] for row in rows],
            [f"final-{i:02d}" for i in range(8)],
        )
        for row in rows:
            self.assertEqual(
                row["metrics"],
                {
                    "execution_time_ms": False,
                    "throughput_mflops": False,
                },
            )
        self.assertEqual(metrics["private_gate_diagnostic"]["completed_workloads"], 0)

    def test_memory_failure_is_not_a_scored_zero(self):
        self.assertTrue(namespace["_is_resource_failure"](MemoryError("out of memory")))
        killed = ValueError("native process failed")
        killed.process_returncode = -9
        self.assertTrue(namespace["_is_resource_failure"](killed))
        self.assertFalse(
            namespace["_is_resource_failure"](ValueError("native process failed"))
        )

    def test_actual_evaluator_exception_path_projects_fail_and_retains_diagnostic(self):
        context = SimpleNamespace(receipt={"memory_peak_bytes": 123})
        with (
            patch.dict(
                namespace,
                {
                    "Context": lambda **kwargs: context,
                    "LIMITS": SimpleNamespace(evaluator_seconds=60),
                    "CONFIG": {},
                    "install": lambda *args: None,
                    "_workloads": lambda final: [],
                    "sha": lambda path: str(path),
                    "remaining": lambda *args, **kwargs: None,
                    "require_disk_capture": lambda *args: None,
                    "apply_submission_patch": lambda *args, **kwargs: SimpleNamespace(
                        ok=False
                    ),
                },
            ),
            patch.dict(os.environ, {"FRONTIER_SUBMISSION_ROLE": "final"}),
        ):
            result = namespace["evaluate"]("bad.patch")
        self.assertEqual(result[:2], (0.0, 0.0))
        self.assertEqual(result[3]["beats_reference"], 0)
        self.assertEqual(
            result[3]["private_gate_diagnostic"]["runtime"], context.receipt
        )

        spec = importlib.util.spec_from_file_location(
            "clusterwise_feedback", TASK / "verifier/beat_spec_feedback.py"
        )
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        private = helper.private_metrics(
            result[3], beat_spec_path=JUDGE / "beat_spec.json"
        )
        self.assertEqual(private["beats_reference"], 0)
        self.assertEqual(private["margin"], -50.0)
        self.assertNotIn("private_gate_diagnostic", private)
        public = helper.public_metrics(
            result[3], beat_spec_path=JUDGE / "beat_spec.json"
        )
        self.assertEqual(set(public), {"beat_spec_progress"})
        for row in private["beat_spec_progress"].values():
            self.assertEqual(
                row,
                {
                    "passed": 0,
                    "failed": 8,
                    "total": 8,
                    "pass_percent": 0.0,
                    "fail_percent": 100.0,
                },
            )

    def test_setup_and_memory_failures_escape_scoring(self):
        for error in (
            MemoryError("allocation failed"),
            RuntimeError("wrong verifier capacity"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch.dict(
                    namespace,
                    {
                        "Context": lambda error=error, **kwargs: (_ for _ in ()).throw(
                            error
                        ),
                        "LIMITS": SimpleNamespace(evaluator_seconds=60),
                    },
                ),
            ):
                with self.assertRaises(type(error)):
                    namespace["evaluate"]("bad.patch")

    def test_reference_gate_and_adapter_type_error_do_not_become_candidate_fail(self):
        common = {
            "Context": lambda **kwargs: SimpleNamespace(receipt={}),
            "LIMITS": SimpleNamespace(evaluator_seconds=60),
            "CONFIG": {},
            "install": lambda *args: None,
            "_workloads": lambda final: [],
            "sha": lambda path: str(path),
            "remaining": lambda *args, **kwargs: None,
            "require_disk_capture": lambda *args: None,
            "verify_source_tree": lambda *args: None,
        }

        def incompatible(*args, **kwargs):
            raise TypeError("apply_submission_patch got an unexpected keyword argument")

        with patch.dict(namespace, {**common, "apply_submission_patch": incompatible}):
            with self.assertRaises(TypeError):
                namespace["evaluate"]("bad.patch")

        def invalid_reference(path, root, config, **kwargs):
            return SimpleNamespace(
                ok=path == "candidate.patch", workdir=root / "scratch/tree"
            )

        with patch.dict(
            namespace, {**common, "apply_submission_patch": invalid_reference}
        ):
            with self.assertRaises(ValueError):
                namespace["evaluate"]("candidate.patch")

    def test_final_adapter_returns_score_zero_fail_and_private_evidence(self):
        feedback_spec = importlib.util.spec_from_file_location(
            "beat_spec_feedback", TASK / "verifier/beat_spec_feedback.py"
        )
        feedback = importlib.util.module_from_spec(feedback_spec)
        feedback_spec.loader.exec_module(feedback)
        adapter_spec = importlib.util.spec_from_file_location(
            "clusterwise_failure_adapter", TASK / "verifier/fcs2_verifier.py"
        )
        adapter = importlib.util.module_from_spec(adapter_spec)
        with patch.dict(sys.modules, {"beat_spec_feedback": feedback}):
            adapter_spec.loader.exec_module(adapter)
        metrics = namespace["_failure_metrics"](
            True,
            reason="incorrect numeric capture",
            completed_workloads=0,
            runtime={"memory_peak_bytes": 123},
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            root = Path(directory)
            payload = root / "payload.tar"
            body = b"invalid patch"
            with tarfile.open(payload, "w") as archive:
                member = tarfile.TarInfo("payload/solution.patch")
                member.size = len(body)
                archive.addfile(member, io.BytesIO(body))
            dataset = root / "dataset.tar"
            dataset.write_bytes(b"dataset")
            with patch.object(
                adapter,
                "load_evaluator",
                return_value=lambda path: (
                    0.0,
                    0.0,
                    "correctness gate failed",
                    metrics,
                ),
            ):
                result = adapter.verify(
                    bundle=JUDGE,
                    entrypoint="payload/solution.patch",
                    payload=payload,
                    dataset=dataset,
                    mode="eval",
                    result_path=root / "result.json",
                    evidence_path=root / "evidence.tar.zst",
                )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["raw_score"], 0.0)
            self.assertEqual(result["private_metrics"]["beats_reference"], 0)
            self.assertEqual(result["metrics"], {"score_unbounded": 0.0})
            self.assertNotIn("private_gate_diagnostic", result["private_metrics"])
            evidence = adapter._zstandard.ZstdDecompressor().decompress(
                (root / "evidence.tar.zst").read_bytes()
            )
            with tarfile.open(fileobj=io.BytesIO(evidence)) as archive:
                stored = json.load(archive.extractfile("metrics.json"))
            self.assertEqual(
                stored["metrics"]["private_gate_diagnostic"]["reason"],
                "incorrect numeric capture",
            )


if __name__ == "__main__":
    unittest.main()
