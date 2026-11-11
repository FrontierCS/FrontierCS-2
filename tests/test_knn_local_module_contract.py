"""Four GPU kernel tasks allow local modules while protecting trusted paths.

A task.toml task states its editable surface in task.toml [submission]; the shared
edit_scope module enforces it on both sides (the Agent's make_submission.sh and the shared
verifier) and the judge's check_tree adds the content rules. A legacy task keeps its own
make_submission.sh and evaluator.validate_patch. The same contract holds for both.
"""

import deployment_test_guard  # noqa: F401 # isort: skip

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "tasks/problems/MLSys"
sys.path.insert(0, str(ROOT / "scripts"))

import task_manifest  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class KernelModuleContract:
    # Added-line content the judge rejects as content_policy. A task whose workers run in the
    # shared isolate launcher keeps only the library ban (process/network words are free there).
    banned_contents = ("import subprocess\n", "import requests\n", "import faiss\n")
    sandboxed_contents = ()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = TASK_ROOT / self.problem_id
        self.native = task_manifest.is_native(self.task)
        judge = (
            task_manifest.judge_bundle(self.task)
            if self.native
            else self.task / "judge"
        )
        judge_path = str(judge)
        sys.path.insert(0, judge_path)
        self.addCleanup(sys.path.remove, judge_path)
        shared = str(
            ROOT / "tasks/shared/lib"
        )  # isolate, as the Verifier images ship it
        sys.path.append(shared)
        self.addCleanup(sys.path.remove, shared)
        if self.problem_id == "dbscan_gpu_kernel_optimization":
            previous = sys.modules.pop("checker", None)

            def restore_checker():
                sys.modules.pop("checker", None)
                if previous is not None:
                    sys.modules["checker"] = previous

            self.addCleanup(restore_checker)
        self.evaluator = _load("kernel_contract_evaluator", judge / "evaluator.py")
        self.addCleanup(sys.modules.pop, "kernel_contract_evaluator", None)
        if self.native:
            self.edit_scope = task_manifest._edit_scope()
            definition = task_manifest.read_task(self.task)
            self.rules = self.edit_scope.Rules.from_mapping(
                task_manifest.edit_scope_rules(definition)
            )
            self.policy = task_manifest.submission_policy(definition)
            self.starter = self.task / "repo"

    # ---------------------------------------------------------------- the two formats
    def helper(self, files):
        """Run the task's Agent submission tool on a workspace holding `files`; return
        (returncode, stderr, workspace)."""
        workspace = self.root / "workspace"
        repo = workspace / "repo"
        if self.native:
            shutil.copytree(self.starter, repo)
            for path in [repo, *repo.rglob("*")]:
                path.chmod(path.stat().st_mode | 0o200)
        else:
            (repo / self.library).mkdir(parents=True)
            (repo / self.library / "kernel.py").write_text(
                "def kernel():\n    return None\n"
            )
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Contract test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "starter",
            ],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "update-ref", "refs/frontier/baseline", "HEAD"],
            cwd=repo,
            check=True,
        )
        for name, body in files.items():
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        if self.native:
            bin_dir = ROOT / "tasks/shared/agent/bin"
            sys.path.insert(0, str(ROOT / "tasks/shared/lib"))
            self.addCleanup(sys.path.remove, str(ROOT / "tasks/shared/lib"))
            tool = _load("kernel_contract_submission", bin_dir / "fcs2_submission.py")
            try:
                tool.make_submission(
                    repo, self.policy, starter=self.starter, workspace=workspace
                )
            except tool.Refused as error:
                return 2, f"submission refused: {error}", workspace
            return 0, "", workspace
        script = (
            self.task / "agent-base/frontier/task/bin/make_submission.sh"
        ).read_text()
        path = self.root / "make_submission.sh"
        # Relocate only the absolute workspace mount for a real helper execution.
        path.write_text(script.replace("/workspace", str(workspace)))
        result = subprocess.run(["sh", str(path)], capture_output=True, text=True)
        return result.returncode, result.stderr, workspace

    def judge_accepts(self, patch: Path) -> tuple[bool, str]:
        """Whether the judge accepts the patch before measuring it."""
        if not self.native:
            valid, message, _ = self.evaluator.validate_patch(patch)
            return valid, message
        result = self.edit_scope.check(patch.read_bytes(), self.starter, self.rules)
        try:
            if not result.ok:
                return False, result.category
            verdict = self.evaluator.check_tree(
                str(result.tree), str(self.starter), list(result.changes)
            )
            return verdict is None, verdict[0] if verdict else ""
        finally:
            result.dispose()

    def path_allowed(self, name: str) -> bool:
        if not self.native:
            return self.evaluator._validate_path(name)[0]
        return self.edit_scope.matches(
            name, self.rules.allowed_paths
        ) and not self.edit_scope.matches(name, self.rules.deny_paths)

    # ---------------------------------------------------------------- the contract
    def test_official_helper_packages_new_regression_module_and_judge_accepts_it(self):
        name = self.library + "/test_regress.py"
        code, stderr, workspace = self.helper(
            {name: "def test_local():\n    assert 1 + 1 == 2\n"}
        )
        self.assertEqual(code, 0, stderr)
        patch = workspace / "payload/solution.patch"
        self.assertIn("diff --git a/" + name, patch.read_text())
        self.assertIn("new file mode", patch.read_text())
        valid, message = self.judge_accepts(patch)
        self.assertTrue(valid, message)

    def test_trusted_paths_and_reserved_filenames_remain_rejected(self):
        for name in (
            "judge/evaluator.py",
            "judge/" + self.reference_name,
            "reference.patch",
            self.reference_name,
            self.library + "/" + self.reference_name,
            self.library + "/nested/" + self.reference_name,
            self.library + "/conftest.py",
            self.library + "/nested/conftest.py",
            self.library + "/kernel.so",
            "../" + self.library + "/test_regress.py",
        ):
            with self.subTest(path=name):
                self.assertFalse(self.path_allowed(name))

    def test_official_helper_rejects_reserved_filenames_without_silently_stripping_them(
        self,
    ):
        name = self.library + "/nested/conftest.py"
        code, stderr, workspace = self.helper({name: "pass\n"})
        self.assertNotEqual(code, 0)
        if self.native:
            self.assertIn("outside the task's editable surface", stderr)
            self.assertIn(name, stderr)
        else:
            self.assertIn("reserved or trusted file may not change: " + name, stderr)
        self.assertFalse((workspace / "payload/solution.patch").exists())

    def test_test_named_module_does_not_bypass_the_content_policy(self):
        for content in self.banned_contents + self.sandboxed_contents:
            with self.subTest(content=content):
                patch = self.root / "bad.patch"
                name = self.library + "/test_regress.py"
                patch.write_text(
                    f"diff --git a/{name} b/{name}\n"
                    f"new file mode 100644\n--- /dev/null\n+++ b/{name}\n"
                    "@@ -0,0 +1 @@\n+" + content
                )
                valid, message = self.judge_accepts(patch)
                self.assertEqual(valid, content in self.sandboxed_contents)
                if self.native and not valid:
                    self.assertEqual(message, "content_policy")


class KnnLocalModuleContractTests(KernelModuleContract, unittest.TestCase):
    problem_id = "knn_gpu_kernel_optimization"
    library = "knnlib"
    reference_name = "refknn.py"
    banned_contents = ("import faiss\n",)
    sandboxed_contents = ("import subprocess\n", "import requests\n")


class KmeansLocalModuleContractTests(KernelModuleContract, unittest.TestCase):
    problem_id = "kmeans_gpu_kernel_optimization"
    library = "kmeanslib"
    reference_name = "refkmeans.py"
    banned_contents = ("import faiss\n",)
    sandboxed_contents = ("import subprocess\n", "import requests\n")


class IvfpqLocalModuleContractTests(KernelModuleContract, unittest.TestCase):
    problem_id = "ivf_pq_gpu_kernel_optimization"
    library = "ivfpqlib"
    reference_name = "refivfpq.py"
    banned_contents = ("import faiss\n",)
    sandboxed_contents = ("import subprocess\n", "import requests\n")


class DbscanLocalModuleContractTests(KernelModuleContract, unittest.TestCase):
    problem_id = "dbscan_gpu_kernel_optimization"
    library = "dbscanlib"
    reference_name = "refdbscan.py"
    banned_contents = ("import faiss\n",)
    sandboxed_contents = ("import subprocess\n", "import requests\n")


if __name__ == "__main__":
    unittest.main()
