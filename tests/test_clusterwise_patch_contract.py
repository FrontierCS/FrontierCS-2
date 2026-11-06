"""The author patch through the shared edit scope and the judge's own reference tree (#757).

The shared verifier applies a submission with tasks/shared/lib/edit_scope.py before the
evaluator runs; the judge applies its reference patch itself, inside the evaluator's scratch
directory and bounded by the evaluator's deadline.
"""
import deployment_test_guard  # noqa: F401 # isort: skip

import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/Arch/clusterwise_spgemm"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tasks/shared/lib"))  # isolate, as in the Verifier base
import task_manifest  # noqa: E402

JUDGE = task_manifest.judge_bundle(TASK)


def _evaluator():
    names = ("runtime_context", "runtime_contract", "trusted_runner", "product_oracle",
             "dataset_loader", "link_guard")
    saved_path, saved = list(sys.path), {n: sys.modules.pop(n, None) for n in names}
    sys.path.insert(0, str(JUDGE))
    try:
        spec = importlib.util.spec_from_file_location("clusterwise_contract_evaluator",
                                                      JUDGE / "evaluator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        for name, module in saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module


class ClusterwisePatchContractTests(unittest.TestCase):
    def test_author_patch_passes_the_shared_edit_scope_and_preserves_pristine_source(self):
        scope = task_manifest._edit_scope()
        rules = scope.Rules.from_mapping(task_manifest.edit_scope_rules(task_manifest.read_task(TASK)))
        pristine = JUDGE / "pristine" / "BIN.h"
        before = hashlib.sha256(pristine.read_bytes()).hexdigest()
        result = scope.check((JUDGE / "reference.patch").read_bytes(), JUDGE / "pristine", rules)
        try:
            self.assertTrue(result.ok, result.message)
            self.assertNotEqual(hashlib.sha256((result.tree / "BIN.h").read_bytes()).hexdigest(), before)
            self.assertEqual(hashlib.sha256(pristine.read_bytes()).hexdigest(), before)
        finally:
            result.dispose()

    def test_reference_tree_is_built_in_the_evaluator_scratch_within_its_deadline(self):
        evaluator = _evaluator()
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            tree = evaluator._reference_tree(scratch, time.monotonic() + 60)
            self.assertEqual(tree, scratch / "reference" / "tree")
            self.assertEqual(evaluator._editable_files(tree).keys() & {"BIN.h"}, {"BIN.h"})
            self.assertNotEqual((tree / "BIN.h").read_bytes(), (JUDGE / "pristine/BIN.h").read_bytes())
            shutil.rmtree(scratch / "reference")
            self.assertEqual(list(scratch.iterdir()), [])
            # An exhausted evaluator deadline stops the judge before it applies anything.
            deadline_error = evaluator.remaining.__globals__["TrustedAggregateDeadline"]
            with self.assertRaises(deadline_error):
                evaluator._reference_tree(scratch, time.monotonic() - 1)


if __name__ == "__main__":
    unittest.main()
