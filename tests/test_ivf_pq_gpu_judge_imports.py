"""ivf_pq_gpu_kernel_optimization's judge resolves its measurement worker under the shared verifier.

The shared verifier stages /opt/frontier/task, imports evaluator.py with only the staged task
directory on sys.path, and restores sys.path afterwards. A judge that put pristine/benchmark on
sys.path at import time and imported `ivfpq_worker` inside evaluate() failed every payload with
ModuleNotFoundError (registrar, source-6b1d7b6e). The worker must come from the judge's own
pristine copy, never from the candidate tree.
"""
import deployment_test_guard  # noqa: F401 # isort: skip

import ast
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/MLSys/ivf_pq_gpu_kernel_optimization"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tasks/shared/verifier"))
# The shared launcher: the Verifier image puts it on the default import path (frontier.pth).
sys.path.insert(0, str(ROOT / "tasks/shared/lib"))

import fcs2_verifier  # noqa: E402
import task_manifest  # noqa: E402


class IvfPqJudgeImportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.scratch = Path(temp.name)
        self.task_dir = fcs2_verifier.stage_task(task_manifest.judge_bundle(TASK), self.scratch)
        self.path_before = sys.path[:]
        self.evaluate, self.check_tree = fcs2_verifier.load_evaluator(self.task_dir)

    def test_every_evaluator_import_resolves_without_extra_sys_path(self):
        source = (TASK / "judge/evaluator.py").read_text()
        local = {path.stem for path in (TASK / "judge").glob("*.py")}
        # Shared modules on the image's default import path (/opt/frontier).
        shared = {path.stem for path in (ROOT / "tasks/shared/lib").glob("*.py")}
        for node in ast.walk(ast.parse(source)):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0
                     else [])
            for name in names:
                top = name.split(".")[0]
                with self.subTest(module=name):
                    self.assertTrue(top in sys.stdlib_module_names or top in local or top in shared
                                    or top == "__future__",
                                    f"{name} is neither stdlib, a judge/ module nor a shared one")
        path_edits = [node.lineno for node in ast.walk(ast.parse(source))
                      if isinstance(node, ast.Attribute) and node.attr == "path"
                      and isinstance(node.value, ast.Name) and node.value.id == "sys"]
        self.assertEqual(path_edits, [], "the judge must not rely on sys.path edits")

    def test_worker_is_the_staged_pristine_copy_and_sys_path_is_restored(self):
        self.assertEqual(sys.path, self.path_before)
        evaluator = self.evaluate.__wrapped__.__globals__
        worker = evaluator["ivfpq_worker"]
        self.assertEqual(Path(worker.__file__).resolve(),
                         (self.task_dir / "pristine/benchmark/ivfpq_worker.py").resolve())
        for name in ("verified_data_path", "validate_distance_config", "summarize_samples"):
            self.assertTrue(callable(getattr(worker, name)))

    def test_evaluate_reaches_the_data_check_not_an_import_error(self):
        tree = self.scratch / "tree"
        shutil.copytree(self.task_dir / "pristine", tree)
        with mock.patch.dict(os.environ, {"FRONTIER_SUBMISSION_ROLE": "development"}):
            with self.assertRaisesRegex(RuntimeError, "SIFT"):
                self.evaluate(str(tree))


if __name__ == "__main__":
    unittest.main()
