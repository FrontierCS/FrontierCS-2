import deployment_test_guard  # noqa: F401  # isort: skip

import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tasks/frontier-cs-2.0-demo/problems/Arch/clusterwise_spgemm/judge/submission_patch.py"


def _load_helper():
    name = "clusterwise_submission_patch"
    spec = importlib.util.spec_from_file_location(name, HELPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ClusterwiseSubmissionHelperTest(unittest.TestCase):
    def test_evaluator_scratch_and_deadline_arguments_are_supported(self):
        module = _load_helper()
        signature = inspect.signature(module.apply_submission_patch)
        self.assertIn("scratch_parent", signature.parameters)
        self.assertIn("deadline", signature.parameters)

        # The real evaluator supplies both keyword arguments before it can
        # inspect a candidate. An allowed empty patch exercises the complete
        # temporary-workdir path without compiling the task.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pristine = root / "pristine"
            pristine.mkdir()
            (pristine / "source.txt").write_text("source\n")
            patch = root / "change.patch"
            patch.write_text(
                "diff --git a/source.txt b/source.txt\n"
                "--- a/source.txt\n"
                "+++ b/source.txt\n"
                "@@ -1 +1 @@\n"
                "-source\n"
                "+changed\n"
            )
            result = module.apply_submission_patch(
                patch,
                root,
                {"submission": {"allow_paths": ["source.txt"]}},
                pristine=pristine,
                scratch_parent=root,
                deadline=1.0 + module.time.monotonic(),
            )
            self.assertTrue(result.ok)
            self.assertTrue(result.workdir.is_relative_to(root))
