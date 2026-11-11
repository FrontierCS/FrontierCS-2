"""Dynamic Tanh measurement guard.

The accuracy run (``main.py``) and the evaluator's timed calls ran in
distinguishable processes, and a submission switched implementations on
``sys.argv[0]``. The judge rejects process/harness introspection in every
candidate-owned file (task.toml ``forbidden_content``, applied by the shared edit
scope) and runs candidate calls under the training process's argv.
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

ROOT = Path(__file__).resolve().parents[1]
DYT = ROOT / "tasks/problems/Machine Learning/dynamic_tanh"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tasks/shared/lib"))
import edit_scope  # noqa: E402
import task_manifest  # noqa: E402

PROBES = (
    'import os, sys\ndef _is_main_py():\n    return os.path.basename(sys.argv[0]) == "main.py"\n',
    "import __main__\nFAST = not hasattr(__main__, 'main')\n",
    "import inspect\nFAST = any('engine' in f.filename for f in inspect.stack())\n",
    "import sys\nFAST = sys._getframe(1).f_code.co_name != 'train_one_epoch'\n",
    "import os\nFAST = 'FRONTIER_REPO_ROOT' not in os.environ\n",
    "FAST = 'evaluator' in open('/proc/self/cmdline').read()\n",
    "import sys\nFAST = 'evaluator' in sys.modules\n",
    "# comments count too: ARGV\n",
)


def _diff(path: str, before: str, after: str) -> bytes:
    new = not before
    lines = difflib.unified_diff(
        before.splitlines(True),
        after.splitlines(True),
        "/dev/null" if new else f"a/{path}",
        f"b/{path}",
    )
    head = f"diff --git a/{path} b/{path}\n" + ("new file mode 100644\n" if new else "")
    return (head + "".join(lines)).encode()


class DynamicTanhIntrospectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        task = task_manifest.read_task(DYT)
        cls.rules = edit_scope.Rules.from_mapping(task_manifest.edit_scope_rules(task))
        cls.base = (DYT / "repo/dynamic_tanh.py").read_text()

    def check(self, patch: bytes):
        result = edit_scope.check(patch, DYT / "repo", self.rules, build_tree=False)
        result.dispose()
        return result

    def test_argv_switch_and_other_probes_are_rejected(self):
        for text in PROBES:
            with self.subTest(text=text.splitlines()[-1]):
                in_module = self.check(
                    _diff("dynamic_tanh.py", self.base, self.base + "\n" + text)
                )
                self.assertEqual(
                    (in_module.ok, in_module.category), (False, "content_policy")
                )
                helper = self.check(_diff("solver_extensions/fast.py", "", text))
                self.assertEqual(
                    (helper.ok, helper.category), (False, "content_policy")
                )

    def test_the_pattern_is_judge_only(self):
        task = task_manifest.read_task(DYT)
        self.assertNotIn(
            "forbidden_content", task_manifest.edit_scope_rules(task, agent=True)
        )

    def test_reference_and_shipped_sources_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "repo"
            shutil.copytree(DYT / "repo", tree)
            subprocess.run(
                [
                    "git",
                    "apply",
                    "--whitespace=nowarn",
                    str(DYT / "judge/reference.patch"),
                ],
                cwd=tree,
                check=True,
                capture_output=True,
                env={**os.environ, "GIT_CEILING_DIRECTORIES": str(tree.parent)},
            )
            reference = (tree / "dynamic_tanh.py").read_text()
        result = self.check(_diff("dynamic_tanh.py", self.base, reference))
        self.assertTrue(result.ok, result.message)

    def test_candidate_calls_see_the_training_argv(self):
        spec = importlib.util.spec_from_file_location(
            "dyt_guard_evaluator", task_manifest.judge_bundle(DYT) / "evaluator.py"
        )
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        saved = list(sys.argv)
        command = ["/repo/frontier-bench.sh", "--model", "vit", "--seed", "3"]
        with evaluator._training_argv(command):
            self.assertEqual(sys.argv, ["main.py", "--model", "vit", "--seed", "3"])
        self.assertEqual(sys.argv, saved)


if __name__ == "__main__":
    unittest.main()
