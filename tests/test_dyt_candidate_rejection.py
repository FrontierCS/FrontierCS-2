"""Dynamic Tanh candidate validation failures are scored zeros through the shared verifier.

A failed validity check returns (0, 0, fixed message, metrics) from evaluate(tree): train
metrics are empty (the shared verifier admits only beat-spec metrics in train), and final
keeps the private failure class in evidence only.
"""

import deployment_test_guard  # noqa: F401 # isort: skip

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
TASK = ROOT / "tasks/problems/Machine Learning/dynamic_tanh"
SHARED = ROOT / "tasks/shared"
sys.path.insert(0, str(ROOT / "scripts"))
import task_manifest  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DynamicTanhCandidateRejectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old_path = list(sys.path)
        cls.addClassCleanup(lambda: sys.path.__setitem__(slice(None), old_path))
        sys.path.insert(0, str(SHARED / "verifier"))
        sys.path.insert(0, str(SHARED / "lib"))
        cls.verifier = _load("dyt_shared_verifier", SHARED / "verifier/fcs2_verifier.py")
        cls.bundle = task_manifest.judge_bundle(TASK)
        cls.evaluator = _load("dyt_rejection_evaluator", cls.bundle / "evaluator.py")

    def candidate_failure(self, mode):
        role = "final" if mode == "eval" else "development"
        error = self.evaluator.CandidateError(self.evaluator.CANDIDATE_FAILURE_MESSAGE)
        with patch.dict(os.environ, {"FRONTIER_SUBMISSION_ROLE": role}), \
                patch.object(self.evaluator, "_load_manifest", side_effect=error):
            return self.evaluator.evaluate(str(TASK / "repo"))

    def verify(self, result, mode):
        v = self.verifier
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, dataset = root / "payload.tar", root / "dataset"
            with tarfile.open(payload, "w") as archive:
                body = b"diff --git a/dynamic_tanh.py b/dynamic_tanh.py\n"
                info = tarfile.TarInfo("payload/solution.patch")
                info.size = len(body)
                import io
                archive.addfile(info, io.BytesIO(body))
            dataset.write_bytes(b"dataset")
            scope = type("Scope", (), {"ok": True, "tree": TASK / "repo", "changes": (),
                                       "metrics": {}, "dispose": lambda self: None})()
            with patch.object(v, "load_evaluator", return_value=(lambda tree: result, None)), \
                    patch("edit_scope.check", return_value=scope):
                out = v.verify(bundle=self.bundle, entrypoint="payload/solution.patch",
                               payload=payload, dataset=dataset, mode=mode,
                               result_path=root / "result.json",
                               evidence_path=root / "evidence.tar.zst")
            with zstandard.ZstdDecompressor().stream_reader(
                (root / "evidence.tar.zst").open("rb")
            ) as reader, tarfile.open(fileobj=reader, mode="r|") as archive:
                evidence = json.load(archive.extractfile(archive.next()))
            return out, evidence

    def test_train_failure_is_a_scored_zero_with_empty_metrics(self):
        result = self.candidate_failure("train")
        self.assertEqual(result, (0.0, 0.0, self.evaluator.CANDIDATE_FAILURE_MESSAGE, {}))
        out, _ = self.verify(result, "train")
        self.assertEqual((out["status"], out["raw_score"]), ("passed", 0.0))
        self.assertEqual(out["metrics"], {"score_unbounded": 0.0})

    def test_final_failure_keeps_the_private_class_in_evidence_only(self):
        result = self.candidate_failure("eval")
        failure = result[3]["all_final_metrics"]["candidate_failure"]
        self.assertEqual(failure["schema"], "dyt.private-candidate-failure/v1")
        out, evidence = self.verify(result, "eval")
        self.assertEqual(out["metrics"], {"score_unbounded": 0.0})
        self.assertEqual(out["private_metrics"], {"beats_reference": 0})
        self.assertEqual(evidence["metrics"], result[3])

    def test_timed_path_message_names_block_and_quantity(self):
        error = self.evaluator.TimedPathRejection("nonfinite", "x")
        error.protocol = "training"
        self.assertEqual(self.evaluator._candidate_message(error),
                         "submission failed validation or authentic model execution: "
                         "timed-path validation, training block, nonfinite values that do not match")


if __name__ == "__main__":
    unittest.main()
