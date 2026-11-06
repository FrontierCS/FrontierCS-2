"""detok_latent_denoising: one untimed warm-up process runs before any timed
process, so the first timed measurement is not cold (issue #315)."""

import deployment_test_guard  # noqa: F401  # isort: skip

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TASK = Path(os.environ.get("FRONTIER_DETOK_EXPORT", str(
    ROOT / "tasks/frontier-cs-2.0-demo/problems/Machine Learning/detok_latent_denoising"
)))
JUDGE = TASK / "judge"
sys.path.insert(0, str(ROOT / "tasks/shared/lib"))  # isolate, as gpu-tools ships it


def sandbox_result(stdout: bytes = b"", outputs=None, returncode: int = 0, limit=None):
    import isolate
    return isolate.Result(returncode=returncode, stdout=stdout, stderr=b"",
                               outputs=dict(outputs or {}), limit=limit,
                               timed_out=limit == "time", command_seconds=1.0, backend="userns")


def result_tar(files: dict) -> bytes:
    """A method result tar: {relative name: bytes}."""
    import io
    import tarfile
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def load_evaluator():
    sys.path.insert(0, str(JUDGE))
    try:
        spec = importlib.util.spec_from_file_location("detok_evaluator", JUDGE / "evaluator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path and sys.path[0] == str(JUDGE):
            sys.path.pop(0)


class DetokWarmup(unittest.TestCase):
    def test_untimed_warmup_runs_first_and_is_discarded(self):
        evaluator = load_evaluator()
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        workdir = Path(scratch.name) / "repo"
        workdir.mkdir()
        suite = Path(scratch.name) / "suite"
        suite.mkdir()
        for i in range(3):
            (suite / f"workload-{i}").mkdir()
        workloads = sorted(p for p in suite.iterdir() if p.is_dir())

        calls = []

        def fake_run(tree, workload, output, candidate=False):
            calls.append({"workload": Path(workload).name, "output": Path(output).name})
            return {k: 10.0 for k in evaluator.SPEC_METRICS}

        with mock.patch.object(evaluator, "_suite_root", return_value=suite), \
                mock.patch.object(evaluator.shutil, "copytree"), \
                mock.patch.object(evaluator.subprocess, "run"), \
                mock.patch.object(evaluator, "_run", side_effect=fake_run):
            evaluator.evaluate(str(workdir))

        # 1 untimed warm-up + 3 candidate + 3 reference timed processes.
        self.assertEqual(len(calls), 7)
        # The warm-up is first, targets the first workload, and its output dir is
        # the discarded "warmup-output" (never one of the timed/scored outputs).
        self.assertEqual(calls[0]["output"], "warmup-output")
        self.assertEqual(calls[0]["workload"], workloads[0].name)
        self.assertEqual(sum(c["output"] == "warmup-output" for c in calls), 1)
        self.assertNotIn("warmup-output", [c["output"] for c in calls[1:]])


    def test_a_failing_submission_process_is_a_scored_zero_naming_it(self):
        evaluator = load_evaluator()
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        workdir = Path(scratch.name) / "repo"
        workdir.mkdir()
        suite = Path(scratch.name) / "suite"
        for i in range(3):
            (suite / f"workload-{i}").mkdir(parents=True)

        def fake_run(tree, workload, output, candidate=False):
            if candidate:
                raise evaluator.CandidateFailure("method process failed")
            return {k: 10.0 for k in evaluator.SPEC_METRICS}

        with mock.patch.object(evaluator, "_suite_root", return_value=suite), \
                mock.patch.object(evaluator.shutil, "copytree"), \
                mock.patch.object(evaluator.subprocess, "run"), \
                mock.patch.object(evaluator, "_run", side_effect=fake_run):
            result = evaluator.evaluate(str(workdir))
        self.assertEqual(result, (0.0, 0.0, "the submission's method process failed", {}))


    @unittest.skipUnless(importlib.util.find_spec("PIL") and importlib.util.find_spec("numpy"),
                         "needs Pillow and NumPy")
    def test_a_missing_reconstruction_png_is_a_candidate_failure(self):
        import numpy as np
        evaluator = load_evaluator()
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name)
        tree, workload, output, models = root / "repo", root / "w", root / "out", root / "models"
        tree.mkdir()
        (tree / evaluator.AUTHENTIC_ENTRYPOINT).write_text("")
        for name in ("train", "val"):
            (workload / name).mkdir(parents=True)
        for name in ("train.txt", "val.txt", "fid_stats.npz"):
            (workload / name).write_text("")
        (workload / "seed.txt").write_text("1")
        for name in ("detok/detok-BB-gamm3.0-m0.7.pth", "lpips/vgg.pth",
                     "torch/hub/checkpoints/vgg16-397923af.pth",
                     "torch/hub/checkpoints/convnext_small-0c510722.pth",
                     "torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"):
            (models / name).parent.mkdir(parents=True, exist_ok=True)
            (models / name).write_text("")
        packed = result_tar({  # both PNG directories, but no PNG in them
            "log/eval_summary.txt": b"use_ema=False psnr=1.0\nuse_ema=True psnr=1.0\n",
            "eval/epoch_001_use_ema=False/notes.txt": b"",
            "eval/epoch_001_use_ema=True/notes.txt": b""})
        done = sandbox_result(b"Average PSNR (all ranks): 1\n" * 2, {"result.tar": packed})
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "FRONTIER_SUBMISSION_ROLE": "development"}), \
                mock.patch.object(evaluator, "OFFLINE_MODEL_ROOT", models), \
                mock.patch.object(evaluator.isolate, "run", return_value=done), \
                mock.patch.object(evaluator, "_targets", return_value=[np.zeros((4, 4, 3), np.uint8)]):
            with self.assertRaises(evaluator.CandidateFailure) as caught:
                evaluator._run(tree, workload, output, candidate=True)
            self.assertIn("reconstruction PNG", str(caught.exception))
            with self.assertRaises(RuntimeError):
                evaluator._run(tree, workload, root / "out2")


if __name__ == "__main__":
    unittest.main()
