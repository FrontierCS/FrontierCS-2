"""The native wrapper must preserve the scientific source seal during staging."""
import deployment_test_guard  # noqa: F401 # isort: skip

import base64
import hashlib
import importlib.util
import io
import json
from unittest import mock
import shutil
import stat
import sys
import tempfile
import tarfile
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / (
    "tasks/frontier-cs-2.0-demo/problems/Robotics/diffusion_policy_cpu")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inventory(root):
    return {str(path.relative_to(root)): (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        stat.S_IMODE(path.stat().st_mode))
        for path in root.rglob("*") if path.is_file()}


class SealedStagingTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(TASK / "verifier"))
        self.addCleanup(sys.path.remove, str(TASK / "verifier"))
        self.wrapper = load(TASK / "verifier/fcs2_verifier.py", "diffusion_sealed_wrapper")
        self.patch = load(TASK / "judge/submission_patch.py", "diffusion_sealed_patch")

    def test_current_datasets_bind_the_complete_published_judge_inventory(self):
        # These modes match the immutable COPY/chmod contract, including documents.
        source = {}
        for path in sorted((TASK / "judge").rglob("*")):
            self.assertFalse(path.is_symlink())
            if path.is_file():
                relative = path.relative_to(TASK / "judge").as_posix()
                mode = 0o555 if path.stat().st_mode & 0o111 else 0o444
                if relative.startswith("harbor/app/") and relative.endswith(".sh"):
                    mode = 0o555
                source[relative] = {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mode": mode}
        expected = hashlib.sha256(json.dumps(
            source, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        for native in ("train", "eval"):
            with self.subTest(mode=native):
                selector = json.loads((TASK / "datasets" / (native + ".json")).read_bytes())
                body = base64.b64decode(selector["archive_base64"], validate=True)
                self.assertEqual(selector["archive_sha256"], "sha256:" + hashlib.sha256(body).hexdigest())
                with tarfile.open(fileobj=io.BytesIO(body), mode="r:") as composite:
                    manifest = json.load(composite.extractfile("manifest.json"))
                    self.assertEqual(manifest["source_sha256"], expected)
                    for key, entry in manifest["datasets"].items():
                        archive_body = composite.extractfile(key + ".tar").read()
                        self.assertEqual(hashlib.sha256(archive_body).hexdigest(), entry["sha256"])
                        with tarfile.open(fileobj=io.BytesIO(archive_body), mode="r:") as archive:
                            seal_raw = archive.extractfile("seal.json").read()
                        self.assertEqual(hashlib.sha256(seal_raw).hexdigest(), entry["seal_sha256"])
                        self.assertEqual(json.loads(seal_raw)["source_sha256"], expected)

    def test_sealed_file_inventory_and_executable_modes_survive_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "source"
            (bundle / "nested").mkdir(parents=True)
            (bundle / "evaluator.py").write_text("def evaluate(path): return 0,0,'',{}\n")
            (bundle / "nested/run.sh").write_text("#!/bin/sh\nexit 0\n")
            (bundle / "evaluator.py").chmod(0o444)
            (bundle / "nested/run.sh").chmod(0o555)
            (bundle / "nested").chmod(0o555)
            bundle.chmod(0o555)
            scratch = root / "scratch"
            scratch.mkdir()
            before = inventory(bundle)
            staged = self.wrapper.stage_task(bundle, scratch)
            self.assertEqual(inventory(staged), before)
            self.assertEqual(inventory(bundle), before)
            self.assertTrue(staged.stat().st_mode & stat.S_IWUSR)
            self.assertTrue((staged / "nested").stat().st_mode & stat.S_IWUSR)
            shutil.rmtree(staged)
            (bundle / "nested").chmod(0o755)
            bundle.chmod(0o755)

    def test_candidate_edits_only_the_separate_writable_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = root / "task/harbor/app"
            app.mkdir(parents=True)
            (app / "sampler.py").write_text("steps = 20\n")
            (app / "sampler.py").chmod(0o444)
            app.chmod(0o555)
            before = inventory(app)
            submission = root / "solution.patch"
            submission.write_text("diff --git a/sampler.py b/sampler.py\n"
                                  "--- a/sampler.py\n+++ b/sampler.py\n"
                                  "@@ -1 +1 @@\n-steps = 20\n+steps = 10\n")
            result = self.patch.apply_submission_patch(submission, root / "task", {
                "submission": {"repo_root": "/app", "allow_paths": ["sampler.py"]}})
            self.assertTrue(result.ok, result.message)
            try:
                self.assertNotEqual(result.workdir, app)
                self.assertEqual((result.workdir / "sampler.py").read_text(), "steps = 10\n")
                self.assertTrue((result.workdir / "sampler.py").stat().st_mode & stat.S_IWUSR)
                self.assertEqual(inventory(app), before)
            finally:
                shutil.rmtree(result.cleanup_root)
                app.chmod(0o755)

    def test_directory_links_cannot_change_external_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "source"
            bundle.mkdir()
            (bundle / "evaluator.py").write_text("pass\n")
            outside = root / "outside"
            outside.mkdir(mode=0o555)
            (bundle / "link").symlink_to(outside, target_is_directory=True)
            scratch = root / "scratch"
            scratch.mkdir()
            self.wrapper.stage_task(bundle, scratch)
            self.assertEqual(stat.S_IMODE(outside.stat().st_mode), 0o555)
            outside.chmod(0o755)

    def test_acquisition_directory_is_stable_exclusive_and_cleaned(self):
        prepare = load(TASK / "judge/prepare_runtime.py", "diffusion_runtime_build")
        self.assertEqual(str(prepare.BUILD_DIRECTORY), "/tmp/diffusion-runtime-build")
        with tempfile.TemporaryDirectory() as directory:
            fixed = Path(directory) / "build"
            with mock.patch.object(prepare, "BUILD_DIRECTORY", fixed):
                for _ in range(2):
                    with prepare.acquisition_directory() as root:
                        self.assertEqual(root, fixed)
                        self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
                        (root / "wheel.whl").write_bytes(b"wheel")
                        with self.assertRaises(FileExistsError):
                            with prepare.acquisition_directory():
                                pass
                    self.assertFalse(fixed.exists())
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    with prepare.acquisition_directory():
                        raise RuntimeError("interrupted")
                self.assertFalse(fixed.exists())


if __name__ == "__main__":
    unittest.main()
