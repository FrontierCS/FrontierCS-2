"""Dynamic Tanh measurement guard (Preview issue #647).

The accuracy run (``main.py``) and the evaluator's timed calls ran in
distinguishable processes, and a submission switched implementations on
``sys.argv[0]``. The evaluator now rejects process/harness introspection in every
candidate-owned file and runs candidate calls under the training process's argv.
"""
import deployment_test_guard  # noqa: F401 # isort: skip

import difflib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "tasks/frontier-cs-2.0-demo/problems"
DYT = PROBLEMS / "Machine Learning/dynamic_tanh"
class DynamicTanhIntrospectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = dict(sys.modules)
        path = list(sys.path)

        def restore():
            sys.path[:] = path
            for key in set(sys.modules) - set(modules):
                del sys.modules[key]
            sys.modules.update(modules)

        cls.addClassCleanup(restore)
        sys.path.insert(0, str(DYT / "judge"))
        for name in ("candidate_failure", "submission_patch"):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location("dyt_guard_evaluator", DYT / "judge/evaluator.py")
        cls.evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.evaluator)

    def candidate_tree(self, files):
        tree = Path(tempfile.mkdtemp(prefix="dyt-guard-"))
        self.addCleanup(shutil.rmtree, tree, True)
        for name, text in files.items():
            (tree / name).parent.mkdir(parents=True, exist_ok=True)
            (tree / name).write_text(text)
        return tree

    def test_argv_switch_and_other_probes_are_rejected(self):
        # The switch used by the #647 final, and other ways to tell the processes apart.
        probes = (
            'import os, sys\ndef _is_main_py():\n    return os.path.basename(sys.argv[0]) == "main.py"\n',
            "import __main__\nFAST = not hasattr(__main__, 'main')\n",
            "import inspect\nFAST = any('engine' in f.filename for f in inspect.stack())\n",
            "import sys\nFAST = sys._getframe(1).f_code.co_name != 'train_one_epoch'\n",
            "import os\nFAST = 'FRONTIER_REPO_ROOT' not in os.environ\n",
            "FAST = 'evaluator' in open('/proc/self/cmdline').read()\n",
            "import sys\nFAST = 'evaluator' in sys.modules\n",
        )
        for text in probes:
            with self.subTest(text=text.splitlines()[-1]):
                self.assertTrue(self.evaluator._introspection_violation(
                    self.candidate_tree({"dynamic_tanh.py": "def convert_ln_to_dyt(m):\n    return m\n",
                                         "solver_extensions/fast.py": text})))
                self.assertTrue(self.evaluator._introspection_violation(
                    self.candidate_tree({"dynamic_tanh.py": text})))

    def test_reference_and_shipped_sources_pass(self):
        tree = self.candidate_tree({})
        shutil.copytree(DYT / "agent-base/frontier/task/repo", tree, dirs_exist_ok=True)
        self.assertFalse(self.evaluator._introspection_violation(tree))
        subprocess.run(["git", "apply", "--whitespace=nowarn", str(DYT / "judge/reference.patch")],
                       cwd=tree, check=True, capture_output=True,
                       env={**os.environ, "GIT_CEILING_DIRECTORIES": str(tree.parent)})
        self.assertFalse(self.evaluator._introspection_violation(tree))

    def test_candidate_calls_see_the_training_argv(self):
        saved = list(sys.argv)
        command = ["/repo/frontier-bench.sh", "--model", "vit", "--seed", "3"]
        with self.evaluator._training_argv(command):
            self.assertEqual(sys.argv, ["main.py", "--model", "vit", "--seed", "3"])
        self.assertEqual(sys.argv, saved)

    def test_evaluate_rejects_the_switch_before_any_gpu_work(self):
        base = (DYT / "agent-base/frontier/task/repo/dynamic_tanh.py").read_text()
        changed = base + '\nimport sys\n_MAIN = sys.argv[0].endswith("main.py")\n'
        patch = "".join(difflib.unified_diff(base.splitlines(True), changed.splitlines(True),
                                             "a/dynamic_tanh.py", "b/dynamic_tanh.py"))
        patch = "diff --git a/dynamic_tanh.py b/dynamic_tanh.py\n" + patch
        with tempfile.TemporaryDirectory() as tmp:
            # The image layout: the judge with the pristine repository at harbor/app.
            judge = Path(tmp) / "task"
            shutil.copytree(DYT / "judge", judge)
            shutil.copytree(DYT / "agent-base/frontier/task/repo", judge / "harbor/app")
            spec = importlib.util.spec_from_file_location("dyt_guard_evaluator_image", judge / "evaluator.py")
            evaluator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluator)
            path = Path(tmp) / "solution.patch"
            path.write_text(patch)
            with mock.patch.dict(os.environ, {"FRONTIER_SUBMISSION_ROLE": "train"}), \
                    mock.patch.dict(sys.modules, {"torch": None}):
                score, unbounded, message, metrics = evaluator.evaluate(str(path))
        self.assertEqual((score, unbounded), (0.0, 0.0))
        self.assertEqual(message, evaluator.INTROSPECTION_MESSAGE)
        self.assertIn("candidate_diagnostic", metrics)


if __name__ == "__main__":
    unittest.main()
