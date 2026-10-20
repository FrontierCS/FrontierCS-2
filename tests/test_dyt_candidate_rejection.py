"""Dynamic Tanh candidate validation failures are scientific zero/FAIL results."""

import deployment_test_guard  # noqa: F401 # isort: skip

import copy
import importlib.util
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import zstandard

ROOT = Path(__file__).resolve().parents[1]
TASK = Path(os.environ.get(
    "DYT_VERDICT_EXPORT",
    str(ROOT / "tasks/frontier-cs-2.0-demo/problems/Machine Learning/dynamic_tanh"),
))


class DynamicTanhCandidateRejectionTests(unittest.TestCase):
    def setUp(self):
        modules = dict(sys.modules)
        old_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), old_path))

        def restore():
            for key in set(sys.modules) - set(modules):
                del sys.modules[key]
            sys.modules.update(modules)

        self.addCleanup(restore)
        sys.path.insert(0, str(TASK / "verifier"))
        sys.modules.pop("beat_spec_feedback", None)
        spec = importlib.util.spec_from_file_location(
            "dyt_rejection_verifier", TASK / "verifier/fcs2_verifier.py")
        self.verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.verifier)
        self.metrics = {"all_final_metrics": {
            "candidate_failure": {"schema": "dyt.private-candidate-failure/v1",
                                  "stage": "semantic_validation",
                                  "cause_kind": "runtime_error"},
            "runtime": {"gpu_name": "NVIDIA L4", "visible_gpu_count": 1},
        }}

    def verify(self, metrics, mode="eval", score=0, failure=None):
        v = self.verifier
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            root = Path(tmp)
            payload, dataset = root / "payload", root / "dataset"
            payload.write_bytes(b"payload")
            dataset.write_bytes(b"dataset")
            with (
                patch.object(v, "load_candidate_diagnostic", return_value=None),
                patch.object(v, "extract_payload"),
                patch.object(v, "entrypoint_path", return_value=payload),
                patch.object(v, "stage_task", return_value=TASK / "judge"),
                patch.object(v, "load_evaluator"),
                patch.object(v, "run_evaluator", return_value=(
                    score, score, "candidate rejected", metrics), side_effect=failure),
            ):
                result = v.verify(
                    bundle=TASK / "judge", entrypoint="solution.patch",
                    payload=payload, dataset=dataset, mode=mode,
                    result_path=root / "result.json",
                    evidence_path=root / "evidence.tar.zst",
                )
            self.assertEqual(json.loads((root / "result.json").read_text()), result)
            with zstandard.ZstdDecompressor().stream_reader(
                (root / "evidence.tar.zst").open("rb")
            ) as reader, tarfile.open(fileobj=reader, mode="r|") as archive:
                evidence = json.load(archive.extractfile(archive.next()))
            return result, evidence

    def test_final_validation_rejection_has_fail_and_original_private_evidence(self):
        for stage in ("semantic_validation", "model_workload"):
            metrics = copy.deepcopy(self.metrics)
            metrics["all_final_metrics"]["candidate_failure"]["stage"] = stage
            result, evidence = self.verify(metrics)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["raw_score"], 0)
            self.assertEqual(result["metrics"], {"score_unbounded": 0})
            self.assertEqual(result["private_metrics"], {"beats_reference": 0})
            self.assertEqual(evidence["metrics"], metrics)

    def test_train_keeps_private_failure_private(self):
        result, evidence = self.verify(self.metrics, "train")
        self.assertEqual(result["metrics"], {"score_unbounded": 0})
        self.assertNotIn("private_metrics", result)
        self.assertEqual(evidence["metrics"], self.metrics)

    def test_valid_workloads_keep_original_reference_comparisons(self):
        names = [x["name"] for x in json.loads(
            (TASK / "judge/beat_spec.json").read_text())["metrics"]]
        for beats in (0, 1):
            metrics = {
                "beats_reference": beats, "reference_score": 25,
                "margin": 5,
                "beat_spec_workload_passes": [
                    {"workload": "private-workload", "metrics": {
                        name: bool(beats) for name in names}},
                ],
            }
            result, _ = self.verify(metrics, score=30)
            self.assertEqual(result["private_metrics"]["beats_reference"], beats)
            self.assertEqual(result["private_metrics"]["reference_score"], 25)
            self.assertIn("beat_spec_progress", result["private_metrics"])

    def test_zero_alone_is_not_candidate_rejection(self):
        result, _ = self.verify({})
        self.assertNotIn("private_metrics", result)

    def test_malformed_candidate_failure_cannot_fabricate_a_verdict(self):
        changes = [
            {"schema": "untrusted"}, {"stage": "reference_execution"},
            {"cause_kind": "infrastructure_failure"}, {"stage": []},
            {"cause_kind": {}}, {"extra": "raw process output"},
        ]
        for changeset in changes:
            metrics = copy.deepcopy(self.metrics)
            metrics["all_final_metrics"]["candidate_failure"].update(changeset)
            with self.subTest(changes=changeset), self.assertRaises(
                self.verifier.VerifierFailure):
                self.verify(metrics)
        for score in (1, -1):
            with self.subTest(score=score), self.assertRaises(
                self.verifier.VerifierFailure):
                self.verify(self.metrics, score=score)
        metrics = copy.deepcopy(self.metrics)
        metrics["beats_reference"] = 1
        with self.assertRaises(self.verifier.VerifierFailure):
            self.verify(metrics)

    def test_reference_diagnostic_is_not_candidate_fail(self):
        result, evidence = self.verify({"reference_diagnostics": {"runtime": "failed"}})
        self.assertNotIn("private_metrics", result)
        self.assertEqual(evidence["metrics"], {"reference_diagnostics": {"runtime": "failed"}})

    def test_infrastructure_failure_stays_failed(self):
        v = self.verifier
        failure = v.VerifierFailure("evaluator_failed", "trusted runtime failed")
        with self.assertRaises(v.VerifierFailure):
            self.verify(self.metrics, failure=failure)


if __name__ == "__main__":
    unittest.main()
