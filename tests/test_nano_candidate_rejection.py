from __future__ import annotations

import deployment_test_guard  # noqa: F401 # isort: skip

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK = Path(
    os.environ.get(
        "NANO_VERDICT_EXPORT",
        str(
            ROOT
            / "tasks/frontier-cs-2.0-demo/problems/Machine Learning/nano_world_model"
        ),
    )
)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


class NanoCandidateRejectionTests(unittest.TestCase):
    def setUp(self):
        old_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        self.addCleanup(lambda: setattr(sys, "dont_write_bytecode", old_bytecode))
        old_modules, old_path = dict(sys.modules), list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), old_path))

        def restore():
            for key in set(sys.modules) - set(old_modules):
                del sys.modules[key]
            sys.modules.update(old_modules)

        self.addCleanup(restore)
        sys.path.insert(0, str(TASK / "verifier"))
        sys.modules.pop("beat_spec_feedback", None)
        self.verifier = module(TASK / "verifier/fcs2_verifier.py", "nano_test_verifier")
        sys.modules.pop("submission_patch", None)
        # These control-flow tests mock patch application; no YAML config is used.
        with patch.dict(sys.modules, {"yaml": SimpleNamespace(safe_load=lambda _: {})}):
            self.evaluator = module(TASK / "judge/evaluator.py", "nano_test_evaluator")

    def assert_rejected(self, result):
        self.assertEqual(result[:2], (0, 0))
        self.assertEqual(result[3], {"candidate_rejection": True, "beats_reference": 0})

    def test_invalid_paths_and_patch_have_explicit_rejection(self):
        for value in (None, "bad\0path"):
            self.assert_rejected(self.evaluator.evaluate(value))
        with patch.object(
            self.evaluator,
            "_apply",
            return_value=SimpleNamespace(
                ok=False,
                message="invalid patch",
                workdir=None,
                candidate_rejection=True,
            ),
        ):
            self.assert_rejected(self.evaluator.evaluate("solution.patch"))

    def test_unclassified_patch_failure_stays_unscored(self):
        with patch.object(
            self.evaluator,
            "_apply",
            return_value=SimpleNamespace(
                ok=False,
                message="pristine missing",
                workdir=None,
                candidate_rejection=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "submission preparation failed"):
                self.evaluator.evaluate("solution.patch")

    def test_patch_helper_marks_only_closed_validation(self):
        helper = module(TASK / "judge/submission_patch.py", "nano_patch_helper")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "solution.patch"
            payload.write_text("")
            rejected = helper.apply_submission_patch(payload, root, {})
            self.assertTrue(rejected.candidate_rejection)
            payload.write_text("irrelevant")
            self.assertFalse(
                helper.apply_submission_patch(
                    root / "absent", root, {}
                ).candidate_rejection
            )
            with patch.object(helper, "validate_patch", return_value=(True, "", {})):
                self.assertFalse(
                    helper.apply_submission_patch(payload, root, {}).candidate_rejection
                )
                pristine = root / "harbor/app"
                pristine.mkdir(parents=True)
                with patch.object(
                    helper.subprocess,
                    "run",
                    return_value=SimpleNamespace(
                        returncode=1, stderr="git failure", stdout=""
                    ),
                ):
                    self.assertFalse(
                        helper.apply_submission_patch(
                            payload, root, {}
                        ).candidate_rejection
                    )
                with patch.object(
                    helper.subprocess,
                    "run",
                    side_effect=FileNotFoundError("git missing"),
                ):
                    with self.assertRaises(FileNotFoundError):
                        helper.apply_submission_patch(payload, root, {})

    def test_policy_io_failure_stays_unscored(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            with (
                patch.object(
                    self.evaluator,
                    "_apply",
                    return_value=SimpleNamespace(ok=True, workdir=repo),
                ),
                patch.object(
                    self.evaluator, "_policy_check", side_effect=OSError("read failed")
                ),
            ):
                with self.assertRaises(OSError):
                    self.evaluator.evaluate("solution.patch")

    def test_authored_getattr_guard_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            folder = repo / "src/diffusion"
            folder.mkdir(parents=True)
            for name in ("df_sample.py", "gaussian_diffusion.py", "flow_matching.py"):
                (folder / name).write_text("pass\n")
            (folder / "df_sample.py").write_text(
                'num_timesteps = getattr(diffusion, "num_timesteps", 1000)\n'
            )
            with (
                patch.object(
                    self.evaluator,
                    "_apply",
                    return_value=SimpleNamespace(ok=True, workdir=str(repo)),
                ),
                patch.object(self.evaluator, "_reference_rows") as reference,
            ):
                result = self.evaluator.evaluate("solution.patch")
                self.assert_rejected(result)
                self.assertEqual(result[2], "forbidden process-state name: getattr")
                reference.assert_not_called()

    def test_reference_runtime_failure_stays_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            with (
                patch.object(
                    self.evaluator,
                    "_apply",
                    return_value=SimpleNamespace(ok=True, workdir=str(repo)),
                ),
                patch.object(self.evaluator, "_policy_check"),
                patch.object(
                    self.evaluator,
                    "_reference_rows",
                    side_effect=RuntimeError("trusted runtime"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "trusted runtime"):
                    self.evaluator.evaluate("solution.patch")

    def test_candidate_execution_failure_has_explicit_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            with (
                patch.object(
                    self.evaluator,
                    "_apply",
                    return_value=SimpleNamespace(ok=True, workdir=str(repo)),
                ),
                patch.object(self.evaluator, "_policy_check"),
                patch.object(self.evaluator, "_reference_rows", return_value=[{}]),
                patch.object(
                    self.evaluator,
                    "_run_entrypoint",
                    side_effect=self.evaluator.CandidateExecutionError(),
                ),
            ):
                self.assert_rejected(self.evaluator.evaluate("solution.patch"))

    def test_valid_scoring_keeps_strict_metric_comparisons(self):
        # Beat spec v3: the clip count comes from the development manifest.
        clips = self.evaluator._expected_workloads("dev")
        self.assertEqual(clips, 4)
        reference = [{"psnr": 20.0, "ssim": 0.7, "lpips": 0.2}] * clips
        for candidate, expected in (
            (reference, 0),
            ([{"psnr": 21.0, "ssim": 0.8, "lpips": 0.1}] * clips, 1),
            # One v3 clip that fails PSNR fails the conjunction.
            ([{"psnr": 21.0, "ssim": 0.8, "lpips": 0.1}] * (clips - 1)
             + [{"psnr": 20.0, "ssim": 0.8, "lpips": 0.1}], 0),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp) / "repo"
                repo.mkdir()
                with (
                    patch.object(
                        self.evaluator,
                        "_apply",
                        return_value=SimpleNamespace(ok=True, workdir=str(repo)),
                    ),
                    patch.object(self.evaluator, "_policy_check"),
                    patch.object(
                        self.evaluator, "_reference_rows", return_value=reference
                    ),
                    patch.object(
                        self.evaluator, "_run_entrypoint", return_value=candidate
                    ),
                ):
                    value = self.evaluator.evaluate("solution.patch")
                    self.assertNotIn("candidate_rejection", value[3])
                    result = self.verify(value, "eval")
                    self.assertEqual(
                        result["private_metrics"]["beats_reference"], expected
                    )
                    self.assertIn("reference_score", result["private_metrics"])
                    self.assertIn("beat_spec_progress", result["private_metrics"])

    def verify(self, value, mode):
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
                patch.object(v, "load_evaluator", return_value=lambda _: value),
            ):
                return v.verify(
                    bundle=TASK / "judge",
                    entrypoint="solution.patch",
                    payload=payload,
                    dataset=dataset,
                    mode=mode,
                    result_path=root / "result.json",
                    evidence_path=root / "evidence.tar.zst",
                )

    def test_final_fail_has_no_fabricated_reference_or_workload_metrics(self):
        result = self.verify(self.evaluator._rejected("invalid candidate"), "eval")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["private_metrics"], {"beats_reference": 0})
        self.assertEqual(result["metrics"], {"score_unbounded": 0})

    def test_train_preserves_score_only_feedback(self):
        result = self.verify(self.evaluator._rejected("invalid candidate"), "train")
        self.assertEqual(result["metrics"], {"score_unbounded": 0})
        self.assertNotIn("private_metrics", result)

    def test_malformed_rejections_fail_closed(self):
        for metrics in (
            {"candidate_rejection": False, "beats_reference": 0},
            {"candidate_rejection": True, "beats_reference": False},
            {"candidate_rejection": True, "beats_reference": 1},
            {"candidate_rejection": True, "beats_reference": 0, "margin": 0},
        ):
            for mode in ("train", "eval"):
                with (
                    self.subTest(metrics=metrics, mode=mode),
                    self.assertRaises(self.verifier.VerifierFailure),
                ):
                    self.verify((0, 0, "invalid", metrics), mode)
        with self.assertRaises(self.verifier.VerifierFailure):
            self.verify(
                (1, 1, "invalid", {"candidate_rejection": True, "beats_reference": 0}),
                "eval",
            )


if __name__ == "__main__":
    unittest.main()
