"""Exercise the real author patch with the evaluator's bounded scratch API."""
import deployment_test_guard  # noqa: F401 # isort: skip

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

TASK = Path(__file__).resolve().parents[1] / (
    "tasks/frontier-cs-2.0-demo/problems/Arch/clusterwise_spgemm")


class ClusterwisePatchContractTests(unittest.TestCase):
    def test_author_patch_uses_owned_scratch_and_preserves_pristine_source(self):
        spec = importlib.util.spec_from_file_location(
            "clusterwise_patch_contract", TASK / "judge/submission_patch.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(module)
        pristine = TASK / "judge/harbor/app/BIN.h"
        before = hashlib.sha256(pristine.read_bytes()).hexdigest()
        config = json.loads((TASK / "judge/runtime-config.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            result = module.apply_submission_patch(
                TASK / "judge/reference.patch", TASK / "judge", config,
                scratch_parent=scratch, deadline=time.monotonic() + 60)
            self.assertTrue(result.ok, result.message)
            self.assertEqual(result.cleanup_root.parent, scratch)
            self.assertEqual(result.workdir.parent, result.cleanup_root)
            revised = result.workdir / pristine.relative_to(TASK / "judge/harbor/app")
            self.assertNotEqual(hashlib.sha256(revised.read_bytes()).hexdigest(), before)
            self.assertEqual(hashlib.sha256(pristine.read_bytes()).hexdigest(), before)
            shutil.rmtree(result.cleanup_root)
            self.assertEqual(list(scratch.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
