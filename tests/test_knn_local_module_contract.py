"""Four GPU kernel tasks allow local modules while protecting trusted paths."""
import deployment_test_guard  # noqa: F401 # isort: skip

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "tasks/frontier-cs-2.0-demo/problems/MLSys"


class KernelModuleContract:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = TASK_ROOT / self.problem_id
        judge_path = str(self.task / "judge")
        sys.path.insert(0, judge_path)
        self.addCleanup(sys.path.remove, judge_path)
        if self.problem_id == "dbscan_gpu_kernel_optimization":
            previous = sys.modules.pop("checker", None)
            def restore_checker():
                sys.modules.pop("checker", None)
                if previous is not None:
                    sys.modules["checker"] = previous
            self.addCleanup(restore_checker)
        spec = importlib.util.spec_from_file_location("kernel_contract_evaluator", self.task / "judge/evaluator.py")
        self.evaluator = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.evaluator
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(self.evaluator)

    def helper(self, files):
        workspace = self.root / "workspace"
        repo = workspace / "repo"
        (repo / self.library).mkdir(parents=True)
        (repo / self.library / "kernel.py").write_text("def kernel():\n    return None\n")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "-c", "user.name=Contract test", "-c", "user.email=test@example.invalid",
                        "commit", "-qm", "starter"], cwd=repo, check=True)
        for name, body in files.items():
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        script = (self.task / "agent-base/frontier/task/bin/make_submission.sh").read_text()
        path = self.root / "make_submission.sh"
        # Relocate only the absolute workspace mount for a real helper execution.
        path.write_text(script.replace("/workspace", str(workspace)))
        return subprocess.run(["sh", str(path)], capture_output=True, text=True), workspace

    def test_official_helper_packages_new_regression_module_and_judge_accepts_it(self):
        name = self.library + "/test_regress.py"
        result, workspace = self.helper({name: "def test_local():\n    assert 1 + 1 == 2\n"})
        self.assertEqual(result.returncode, 0, result.stderr)
        patch = workspace / "payload/solution.patch"
        self.assertIn("diff --git a/" + name, patch.read_text())
        self.assertIn("new file mode", patch.read_text())
        valid, message, metrics = self.evaluator.validate_patch(patch)
        self.assertTrue(valid, message)
        self.assertEqual(metrics["changed_files"], 1)

    def test_trusted_paths_and_reserved_filenames_remain_rejected(self):
        for name in ("judge/evaluator.py", "judge/" + self.reference_name, "reference.patch",
                     self.reference_name, self.library + "/" + self.reference_name,
                     self.library + "/nested/" + self.reference_name, self.library + "/conftest.py",
                     self.library + "/nested/conftest.py", self.library + "/kernel.so",
                     "../" + self.library + "/test_regress.py"):
            with self.subTest(path=name):
                self.assertFalse(self.evaluator._validate_path(name)[0])

    def test_official_helper_rejects_reserved_filenames_without_silently_stripping_them(self):
        name = self.library + "/nested/conftest.py"
        result, workspace = self.helper({name: "pass\n"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reserved or trusted file may not change: " + name, result.stderr)
        self.assertFalse((workspace / "payload/solution.patch").exists())

    def test_test_named_module_does_not_bypass_process_network_or_binary_policy(self):
        for content in ("import subprocess\n", "import requests\n", "import faiss\n"):
            with self.subTest(content=content):
                patch = self.root / "bad.patch"
                name = self.library + "/test_regress.py"
                patch.write_text(f"diff --git a/{name} b/{name}\n"
                                 f"new file mode 100644\n--- /dev/null\n+++ b/{name}\n"
                                 "@@ -0,0 +1 @@\n+" + content)
                self.assertFalse(self.evaluator.validate_patch(patch)[0])


class KnnLocalModuleContractTests(KernelModuleContract, unittest.TestCase):
    problem_id = "knn_gpu_kernel_optimization"
    library = "knnlib"
    reference_name = "refknn.py"


class KmeansLocalModuleContractTests(KernelModuleContract, unittest.TestCase):
    problem_id = "kmeans_gpu_kernel_optimization"
    library = "kmeanslib"
    reference_name = "refkmeans.py"


class IvfpqLocalModuleContractTests(KernelModuleContract, unittest.TestCase):
    problem_id = "ivf_pq_gpu_kernel_optimization"
    library = "ivfpqlib"
    reference_name = "refivfpq.py"


class DbscanLocalModuleContractTests(KernelModuleContract, unittest.TestCase):
    problem_id = "dbscan_gpu_kernel_optimization"
    library = "dbscanlib"
    reference_name = "refdbscan.py"


if __name__ == "__main__":
    unittest.main()
