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
TASK = ROOT / "tasks/problems/Arch/clusterwise_spgemm"
JUDGE = TASK / "judge"
sys.path.insert(0, str(ROOT / "scripts"))
import task_manifest  # noqa: E402

# A task.toml task runs the shared verifier and its beat-spec helper.
SHARED_VERIFIER = task_manifest.SHARED / "verifier"

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
            "evaluate",
        }
    )
]
class ResourceCeiling(RuntimeError):
    pass


class TrustedAggregateDeadline(RuntimeError):
    pass


namespace = {
    "ResourceCeiling": ResourceCeiling,
    "TrustedAggregateDeadline": TrustedAggregateDeadline,
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

    def test_candidate_resource_failures_are_scored_zeros(self):
        """Owner rule: a killed native process (CPU-time or OOM kill), compilation past its
        budget and a measured memory ceiling in the candidate phase score zero."""
        killed = ValueError("native process failed")
        killed.process_returncode = -9
        ceiling = namespace["ResourceCeiling"]("total measured memory exceeds 56-GiB acceptance target")
        for error in (killed, ceiling, TimeoutError("native process deadline exceeded")):
            with self.subTest(error=type(error).__name__), patch.dict(namespace, {
                    "Context": lambda **kwargs: SimpleNamespace(receipt={}),
                    "LIMITS": SimpleNamespace(evaluator_seconds=60, build_seconds=600),
                    "CONFIG": {}, "install": lambda *args: None, "_workloads": lambda final: [],
                    "remaining": lambda *args, **kwargs: None,
                    "require_disk_capture": lambda *args: None,
                    "_reference_tree": lambda scratch, deadline: Path(scratch) / "reference",
                    "_editable_files": lambda tree: {str(tree): b""},
                    "PRISTINE": JUDGE / "pristine",
                    "build": lambda tree, *args, error=error: (_ for _ in ()).throw(error)
                    if Path(tree).name == "candidate" else None}):
                result = namespace["evaluate"]("candidate")
            self.assertEqual(result[:2], (0.0, 0.0))
            self.assertEqual(result[3]["beats_reference"], 0)

    def test_actual_evaluator_exception_path_projects_fail_and_retains_diagnostic(self):
        context = SimpleNamespace(receipt={"memory_peak_bytes": 123})
        with (
            patch.dict(
                namespace,
                {
                    "Context": lambda **kwargs: context,
                    "LIMITS": SimpleNamespace(evaluator_seconds=60, build_seconds=600),
                    "CONFIG": {},
                    "install": lambda *args: None,
                    "_workloads": lambda final: [],
                    "remaining": lambda *args, **kwargs: None,
                    "require_disk_capture": lambda *args: None,
                    "_reference_tree": lambda scratch, deadline: Path(scratch) / "reference",
                    "_editable_files": lambda tree: {str(tree): b""},
                    "PRISTINE": JUDGE / "pristine",
                    # The candidate's build fails a gate: a candidate rejection.
                    "build": lambda tree, *args: (_ for _ in ()).throw(
                        ValueError("forbidden symbol in the candidate object")
                    ) if Path(tree).name == "candidate" else None,
                },
            ),
            patch.dict(os.environ, {"FRONTIER_SUBMISSION_ROLE": "final"}),
        ):
            result = namespace["evaluate"]("candidate")
        self.assertEqual(result[:2], (0.0, 0.0))
        self.assertEqual(result[3]["beats_reference"], 0)
        self.assertEqual(
            result[3]["private_gate_diagnostic"]["runtime"], context.receipt
        )

        spec = importlib.util.spec_from_file_location(
            "clusterwise_feedback", SHARED_VERIFIER / "beat_spec_feedback.py"
        )
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        private = helper.private_metrics(result[3])
        self.assertEqual(private["beats_reference"], 0)
        self.assertEqual(private["margin"], -50.0)
        self.assertNotIn("private_gate_diagnostic", private)
        public = helper.public_metrics(
            result[3], visible=list(task_manifest.read_task(TASK)["feedback"])
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
            "LIMITS": SimpleNamespace(evaluator_seconds=60, build_seconds=600),
            "CONFIG": {},
            "install": lambda *args: None,
            "_workloads": lambda final: [],
            "remaining": lambda *args, **kwargs: None,
            "require_disk_capture": lambda *args: None,
            "_editable_files": lambda tree: {str(tree): b""},
            "PRISTINE": JUDGE / "pristine",
        }

        def incompatible(*args, **kwargs):
            raise TypeError("_reference_tree got an unexpected keyword argument")

        with patch.dict(namespace, {**common, "_reference_tree": incompatible}):
            with self.assertRaises(TypeError):
                namespace["evaluate"]("candidate")

        def invalid_reference(tree, *args):
            if Path(tree).name == "reference":
                raise ValueError("forbidden symbol in the reference object")

        with patch.dict(
            namespace, {**common, "build": invalid_reference,
                        "_reference_tree": lambda scratch, deadline: Path(scratch) / "reference"}
        ):
            with self.assertRaises(ValueError):
                namespace["evaluate"]("candidate")

    def test_final_adapter_returns_score_zero_fail_and_private_evidence(self):
        feedback_spec = importlib.util.spec_from_file_location(
            "beat_spec_feedback", SHARED_VERIFIER / "beat_spec_feedback.py"
        )
        feedback = importlib.util.module_from_spec(feedback_spec)
        feedback_spec.loader.exec_module(feedback)
        adapter_spec = importlib.util.spec_from_file_location(
            "clusterwise_failure_adapter", SHARED_VERIFIER / "fcs2_verifier.py"
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
            # A patch the shared edit scope accepts, so the evaluator runs.
            body = (JUDGE / "reference.patch").read_bytes()
            with tarfile.open(payload, "w") as archive:
                member = tarfile.TarInfo("payload/solution.patch")
                member.size = len(body)
                archive.addfile(member, io.BytesIO(body))
            dataset = root / "dataset.tar"
            dataset.write_bytes(b"dataset")
            with patch.object(
                adapter,
                "load_evaluator",
                return_value=(lambda tree: (
                    0.0,
                    0.0,
                    "correctness gate failed",
                    metrics,
                ), None),
            ):
                result = adapter.verify(
                    bundle=task_manifest.judge_bundle(TASK),
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
