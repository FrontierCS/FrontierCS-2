"""nano_world_model on the shared verifier (#757): the sampler policy is check_tree; the
candidate's machine receives only its tree, the runner side of the judge and public data; a
candidate whose sampling fails on its own machine is a scored zero failing every clip once the
reference ran, and a fault of the device check or the trusted runner stays unscored."""
from __future__ import annotations

import deployment_test_guard  # noqa: F401 # isort: skip

import importlib.util
import json
import os
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/Machine Learning/nano_world_model"
SHARED = ROOT / "tasks/shared"
sys.path.insert(0, str(ROOT / "scripts"))
import task_manifest  # noqa: E402


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def payload(patch_text: str, directory: Path) -> Path:
    path = directory / "payload.tar"
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        data = patch_text.encode()
        info = tarfile.TarInfo("payload/solution.patch")
        info.size = len(data)
        import io
        archive.addfile(info, io.BytesIO(data))
    return path


def diff(path: str, old: str, new: str) -> str:
    """A patch replacing one starter line `old` with the line(s) `new` (with context)."""
    import difflib
    lines = (TASK / "repo" / path).read_text().splitlines(keepends=True)
    index = lines.index(old)
    changed = lines[:index] + new.splitlines(keepends=True) + lines[index + 1:]
    return "".join(difflib.unified_diff(lines, changed, f"a/{path}", f"b/{path}"))


class NanoSharedVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SHARED / "verifier"))
        sys.path.insert(0, str(SHARED / "lib"))
        sys.path.insert(0, str(SHARED / "verifier-gpu"))
        cls.verifier = module(SHARED / "verifier/fcs2_verifier.py", "nano_shared_verifier")
        cls.evaluator = module(task_manifest.judge_bundle(TASK) / "evaluator.py", "nano_test_evaluator")
        cls.scratch = Path(tempfile.mkdtemp(prefix="nano-verifier-test-"))
        cls.bundle = cls.scratch / "task"
        shutil.copytree(task_manifest.judge_bundle(TASK), cls.bundle)
        cls.line = next(line for line in (TASK / "repo/src/diffusion/df_sample.py")
                        .read_text().splitlines(keepends=True) if line.strip() == "import torch")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.scratch, ignore_errors=True)

    def run_verifier(self, patch_text: str, mode: str, evaluate=None) -> dict:
        work = Path(tempfile.mkdtemp(dir=self.scratch))
        result = work / "result.json"
        real = self.verifier.load_evaluator

        def load(task_dir):
            loaded, check_tree = real(task_dir)
            return (evaluate or loaded), check_tree

        with patch.object(self.verifier, "load_evaluator", side_effect=load):
            self.verifier.verify(bundle=self.bundle, entrypoint="payload/solution.patch",
                                 payload=payload(patch_text, work),
                                 dataset=TASK / "datasets" / f"{'eval' if mode == 'eval' else 'train'}.json",
                                 mode=mode, result_path=result,
                                 evidence_path=work / "evidence.tar.zst")
        return json.loads(result.read_text())

    def test_policy_violation_is_content_policy(self):
        patch_text = diff("src/diffusion/df_sample.py", self.line, "import os\n")
        result = self.run_verifier(patch_text, "train",
                                   evaluate=lambda _: self.fail("evaluate must not run"))
        self.assertEqual(result["metrics"], {"score_unbounded": 0.0, "rejected": "content_policy"})
        self.assertEqual(result["raw_score"], 0.0)

    def test_authored_getattr_guard_is_preserved(self):
        patch_text = diff("src/diffusion/df_sample.py", self.line,
                          'import torch\nx = getattr(torch, "float32")\n')
        result = self.run_verifier(patch_text, "eval",
                                   evaluate=lambda _: self.fail("evaluate must not run"))
        self.assertEqual(result["metrics"]["rejected"], "content_policy")
        self.assertEqual(result["private_metrics"], {"beats_reference": 0, "rejected": "content_policy"})

    def test_non_python_helper_is_out_of_scope(self):
        patch_text = ("--- /dev/null\n+++ b/src/diffusion/frontier_solution/notes.txt\n"
                      "@@ -0,0 +1 @@\n+notes\n")
        result = self.run_verifier(patch_text, "train",
                                   evaluate=lambda _: self.fail("evaluate must not run"))
        self.assertEqual(result["metrics"]["rejected"], "out_of_scope")

    def test_unimportable_helper_name_is_out_of_scope(self):
        patch_text = ("--- /dev/null\n+++ b/src/diffusion/frontier_solution/my-helper.py\n"
                      "@@ -0,0 +1 @@\n+x = 1\n")
        result = self.run_verifier(patch_text, "train",
                                   evaluate=lambda _: self.fail("evaluate must not run"))
        self.assertEqual(result["metrics"]["rejected"], "out_of_scope")

    def test_allowed_helper_reaches_evaluate_with_the_applied_tree(self):
        seen = {}

        def evaluate(tree):
            seen["helper"] = (Path(tree) / "src/diffusion/frontier_solution/helper.py").read_text()
            return self.evaluator._failed("dev", "candidate evaluation failed")

        patch_text = ("--- /dev/null\n+++ b/src/diffusion/frontier_solution/helper.py\n"
                      "@@ -0,0 +1 @@\n+import math\n")
        result = self.run_verifier(patch_text, "train", evaluate=evaluate)
        self.assertEqual(seen["helper"], "import math\n")
        self.assertEqual(result["raw_score"], 0.0)
        progress = result["metrics"]["beat_spec_progress"]
        self.assertEqual({name: row["passed"] for name, row in progress.items()},
                         {"psnr": 0, "ssim": 0, "lpips": 0})
        self.assertEqual({row["total"] for row in progress.values()}, {4})

    # evaluate(tree) against a fake run_sandboxes: the candidate run returns `latents` (name ->
    # .npy bytes) with its step results, the trusted run returns the rows it is given.
    def evaluate(self, candidate_steps, latents, rows, role="development", exceeded=False):
        gpu = sys.modules["fcs2_verifier_gpu"]
        seen = {}

        def run_sandboxes(runs, *, into):
            (run,) = runs
            seen[run.sandbox] = run
            out = Path(into) / "0"
            if run.sandbox == "candidate":
                seen["candidate_files"] = sorted(str(p.relative_to(run.files))
                                                 for p in Path(run.files).iterdir())
                seen["runtime"] = sorted(p.name for p in (Path(run.files) / "runtime").iterdir())
                (out / "out" / "latents").mkdir(parents=True)
                for name, data in latents.items():
                    (out / "out" / "latents" / name).write_bytes(data)
                return [gpu.SandboxResult(candidate_steps, None if exceeded else out, exceeded,
                                          [b""] * len(candidate_steps))]
            seen["trusted_latents"] = sorted(p.name for p in (Path(run.files) / "latents").glob("*")) \
                if (Path(run.files) / "latents").is_dir() else None
            out.mkdir(parents=True)
            (out / "scores.json").write_text(json.dumps(rows))
            return [gpu.SandboxResult([gpu.StepResult(0, False, 1.0)], out, False, [b""])]

        public = Path(tempfile.mkdtemp(dir=self.scratch))
        for suite in ("dev", "final"):
            (public / suite / "point_maze").mkdir(parents=True)
        with (
            patch.object(self.evaluator, "run_sandboxes", side_effect=run_sandboxes),
            patch.object(self.evaluator, "sandbox_workdir",
                         side_effect=lambda name: f"/workspace/{name}"),
            patch.object(self.evaluator, "PUBLIC_DATA", public),
            patch.dict(os.environ, {"FRONTIER_SUBMISSION_ROLE": role}),
        ):
            return self.evaluator.evaluate(str(TASK / "repo")), seen

    @staticmethod
    def latent(value=0.5, shape=(1, 4, 4, 32, 32), dtype="<f4"):
        import io
        import numpy
        stream = io.BytesIO()
        numpy.save(stream, numpy.full(shape, value, dtype=dtype))
        return stream.getvalue()

    def ok(self):
        return self.gpu.StepResult(0, False, 1.0)

    @property
    def gpu(self):
        return sys.modules["fcs2_verifier_gpu"]

    def rows(self, n, value=20.0):
        return [{"psnr": value, "ssim": 0.7, "lpips": 0.2}] * n

    def test_the_candidate_machine_gets_only_the_tree_runtime_and_public_data(self):
        latents = {f"{i:04d}.npy": self.latent() for i in range(4)}
        value, seen = self.evaluate([self.ok(), self.ok()], latents,
                                    {"reference": self.rows(4), "candidate": self.rows(4)})
        self.assertEqual(seen["candidate_files"], ["data", "out", "runtime", "tree"])
        run = seen["candidate"]
        self.assertEqual(seen["runtime"], ["native_run.py", "sitecustomize.py"])
        self.assertEqual([step.seconds for step in run.steps], [300, 4950])
        self.assertEqual(run.steps[1].env["NANOWM_LATENTS_RECORD"], "/workspace/candidate/out/latents")
        # One MKL code path on any runner CPU: the candidate's machine and the trusted runner
        # compute the reference's latents bit for bit alike.
        self.assertEqual(run.steps[1].env["MKL_CBWR"], "COMPATIBLE")
        self.assertEqual(run.outputs, ["out/latents"])
        self.assertEqual(seen["trusted_latents"], sorted(latents))
        self.assertEqual(seen["trusted"].steps[0].argv[-1], "/workspace/trusted/latents")
        self.assertEqual(value[0], 50.0)

    def assert_failed(self, result, message, clips=4):
        (score, unbounded, text, metrics), seen = result
        self.assertEqual((score, unbounded, text), (0.0, 0.0, message))
        self.assertEqual(metrics["beats_reference"], 0)
        self.assertEqual(len(metrics["beat_spec_workload_passes"]), clips)
        self.assertFalse(any(v for row in metrics["beat_spec_workload_passes"]
                             for v in row["metrics"].values()))
        # The reference still ran; the trusted runner decoded nothing.
        self.assertIsNone(seen["trusted_latents"])
        self.assertEqual(len(seen["trusted"].steps[0].argv), 5)

    def test_candidate_failures_are_a_scored_zero_failing_every_clip(self):
        reference = {"reference": self.rows(4), "candidate": None}
        good = {f"{i:04d}.npy": self.latent() for i in range(4)}
        cases = (
            ([self.ok(), self.gpu.StepResult(1, False, 3.0)], good, False,
             "candidate evaluation failed"),
            ([self.ok(), self.gpu.StepResult(None, True, 5015.0)], good, False,
             "candidate evaluation exceeded its 4950-second deadline"),
            ([self.ok(), self.gpu.StepResult(None, False, 9.0, lost_session=True)], {}, False,
             "candidate evaluation lost its session"),
            ([self.ok(), self.ok()], good, True,
             "candidate evaluation returned more than the output cap"),
            ([self.ok(), self.ok()], dict(list(good.items())[:3]), False,
             "candidate did not return a valid latent video for every clip"),
            ([self.ok(), self.ok()], {**good, "0003.npy": self.latent(float("nan"))}, False,
             "candidate did not return a valid latent video for every clip"),
            ([self.ok(), self.ok()], {**good, "0003.npy": self.latent(dtype="<f8")}, False,
             "candidate did not return a valid latent video for every clip"),
            ([self.ok(), self.ok()], {**good, "0003.npy": self.latent(shape=(1, 4, 4, 32, 31))},
             False, "candidate did not return a valid latent video for every clip"),
        )
        for steps, latents, exceeded, message in cases:
            with self.subTest(message=message, latents=sorted(latents)):
                self.assert_failed(self.evaluate(steps, latents, reference, exceeded=exceeded),
                                   message)

    def test_final_failure_fails_all_six_clips(self):
        self.assert_failed(self.evaluate([self.ok(), self.gpu.StepResult(1, False, 3.0)], {},
                                         {"reference": self.rows(6), "candidate": None},
                                         role="final"),
                           "candidate evaluation failed", clips=6)

    def test_device_check_failure_and_reference_failure_stay_unscored(self):
        lost = self.gpu.StepResult(None, False, 9.0, lost_session=True)
        for steps in ([self.gpu.StepResult(1, False, 3.0), None], [lost, None]):
            with self.assertRaisesRegex(RuntimeError, "device check"):
                self.evaluate(steps, {}, {"reference": self.rows(4), "candidate": None})
        broken = [{"psnr": float("nan"), "ssim": 0.7, "lpips": 0.2}] * 4
        with self.assertRaisesRegex(RuntimeError, "trusted reference evaluation failed"):
            self.evaluate([self.ok(), self.gpu.StepResult(1, False, 3.0)], {},
                          {"reference": broken, "candidate": None})

    def test_valid_scoring_keeps_strict_metric_comparisons(self):
        latents = {f"{i:04d}.npy": self.latent() for i in range(4)}
        reference = self.rows(4)
        for candidate, expected in (
            (reference, 0),
            ([{"psnr": 21.0, "ssim": 0.8, "lpips": 0.1}] * 4, 1),
            ([{"psnr": 21.0, "ssim": 0.8, "lpips": 0.1}] * 3
             + [{"psnr": 20.0, "ssim": 0.8, "lpips": 0.1}], 0),
        ):
            (value, _) = self.evaluate([self.ok(), self.ok()], latents,
                                       {"reference": reference, "candidate": candidate})
            self.assertEqual(value[3]["beats_reference"], expected)

    def test_reference_tree_applies_the_judge_reference(self):
        native_run = module(self.bundle / "native_run.py", "nano_test_native_run")
        scratch = Path(tempfile.mkdtemp(dir=self.scratch))
        tree = native_run._reference_tree(scratch)
        self.assertNotEqual((tree / "src/diffusion/df_sample.py").read_bytes(),
                            (TASK / "repo/src/diffusion/df_sample.py").read_bytes())

    def test_public_inputs_hold_only_context_frames(self):
        # The manifests' clips: no context frame is a scored frame of another clip of its suite.
        for suite in ("dev", "final"):
            slices = json.loads((TASK / "judge/judge_data" / f"{suite}_subset.json").read_text())["slices"]
            context = {(s["traj_idx"], s["start_frame"]) for s in slices}
            scored = {(s["traj_idx"], s["start_frame"] + 5 * k) for s in slices for k in (1, 2, 3)}
            self.assertFalse(context & scored)


if __name__ == "__main__":
    unittest.main()
