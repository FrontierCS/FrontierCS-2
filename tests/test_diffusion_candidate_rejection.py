"""Typed candidate refusals are scored; trusted prerequisites remain errors."""
import deployment_test_guard  # noqa: F401 # isort: skip

from contextlib import ExitStack, nullcontext
import difflib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

TASK = Path(__file__).resolve().parents[1] / (
    "tasks/frontier-cs-2.0-demo/problems/Robotics/diffusion_policy_cpu")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CandidateRejectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        old_modules, old_path, old_env = dict(sys.modules), list(sys.path), dict(os.environ)

        def restore():
            for name in set(sys.modules) - set(old_modules):
                del sys.modules[name]
            sys.modules.update(old_modules)
            sys.path[:] = old_path
            os.environ.clear()
            os.environ.update(old_env)

        self.addCleanup(restore)
        sys.path[:0] = [str(TASK / "judge"), str(TASK / "verifier")]
        for name in ("evaluator", "task_guards", "submission_patch", "private_context",
                     "measurement_runtime", "native_data", "option_b", "runtime_layout",
                     "candidate_failure", "beat_spec_feedback"):
            sys.modules.pop(name, None)
        self.evaluator = load("evaluator", TASK / "judge/evaluator.py")
        self.wrapper = load("diffusion_rejection_wrapper", TASK / "verifier/fcs2_verifier.py")
        import task_guards
        import yaml
        self.guards = task_guards
        self.config = yaml.safe_load((TASK / "judge/config.yaml").read_text())
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        for name in ("beat_spec.json", "candidate-diagnostic.json", "candidate_failure.py"):
            shutil.copy2(TASK / "judge" / name, self.bundle / name)
        (self.bundle / "evaluator.py").write_text("# evaluator injected by test\n")
        os.environ["FRONTIER_DATASET_PATH"] = str(TASK / "datasets/train.tar")
        os.environ["FRONTIER_VERIFICATION_MODE"] = "train"
        os.environ["FRONTIER_VERIFIER_SCRATCH"] = str(self.root)

    def trusted_prerequisites(self):
        """Only prerequisite IO is synthetic; candidate source audit is real."""
        import native_data
        import private_context
        import runtime_layout
        context = {"seal": {"runtime": {"prefix": "unused", "checkpoint": "unused"}},
                   "suites": {"dev": {}, "final": {}}}
        stack = ExitStack()
        stack.enter_context(mock.patch.object(native_data, "materialize_dataset",
                                             return_value=nullcontext(self.root)))
        stack.enter_context(mock.patch.object(private_context, "load_context", return_value=context))
        stack.enter_context(mock.patch.object(private_context, "authorize"))
        stack.enter_context(mock.patch.object(private_context, "accepted_calibrations",
                                             return_value={"dev": {}, "final": {}}))
        stack.enter_context(mock.patch.object(runtime_layout, "validate_layout"))
        stack.enter_context(mock.patch.object(os, "sched_setaffinity"))
        return stack

    def signature_patch(self):
        source = (TASK / "judge/harbor/app" / self.guards.POLICY).read_text()
        changed = source.replace("def conditional_sample(", "def conditional_sample (", 1)
        self.assertNotEqual(source, changed)
        patch = self.root / "signature.patch"
        patch.write_text("".join(difflib.unified_diff(
            source.splitlines(keepends=True), changed.splitlines(keepends=True),
            fromfile="a/" + self.guards.POLICY, tofile="b/" + self.guards.POLICY)))
        return patch

    def audit_measurement(self, _config, path, *_args, **_kwargs):
        with self.evaluator.prepared_submission(path, self.config):
            pass
        self.fail("signature modification unexpectedly passed source audit")

    def transport(self, mode, evaluate, patch):
        os.environ["FRONTIER_VERIFICATION_MODE"] = mode
        payload = self.root / "payload.tar"
        with tarfile.open(payload, "w", format=tarfile.USTAR_FORMAT) as archive:
            data = patch.read_bytes()
            member = tarfile.TarInfo("payload/solution.patch")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        with mock.patch.object(self.wrapper, "load_evaluator", return_value=evaluate):
            return self.wrapper.verify(
                bundle=self.bundle, entrypoint="payload/solution.patch", payload=payload,
                dataset=TASK / f"datasets/{mode}.json", mode=mode,
                result_path=self.root / "result.json", evidence_path=self.root / "evidence.tar.zst")

    def test_actual_signature_whitespace_stays_invalid_and_reaches_train_and_final(self):
        import measurement_runtime
        patch = self.signature_patch()
        with self.assertRaises(self.guards.InvalidSubmission):
            self.audit_measurement(None, patch)
        with self.trusted_prerequisites(), mock.patch.object(
                measurement_runtime, "measure_suite", side_effect=self.audit_measurement):
            for mode in ("train", "eval"):
                result = self.transport(mode, self.evaluator.evaluate, patch)
                self.assertEqual((result["status"], result["raw_score"]), ("passed", 0.0))
                if mode == "train":
                    self.assertEqual(result["metrics"]["candidate_diagnostic"]["category"], "source_policy")
                    self.assertNotIn("private_metrics", result)
                else:
                    self.assertEqual(result["metrics"], {"score_unbounded": 0.0})
                    self.assertEqual(result["private_metrics"], {"beats_reference": 0})

    def test_infrastructure_and_unexpected_errors_stay_evaluator_errors(self):
        import measurement_runtime
        patch = self.signature_patch()
        for error in (OSError("disk unavailable"), ValueError("malformed trusted data"),
                      self.guards.TaskIntegrityError("seal mismatch"), RuntimeError("unexpected")):
            with self.subTest(error=type(error).__name__), self.trusted_prerequisites(), mock.patch.object(
                    measurement_runtime, "measure_suite", side_effect=error):
                with self.assertRaises(self.wrapper.VerifierFailure) as caught:
                    self.transport("eval", self.evaluator.evaluate, patch)
                self.assertEqual(caught.exception.code, "evaluator_error")

    def test_prerequisite_invalid_submission_is_not_candidate_zero(self):
        import private_context
        with self.trusted_prerequisites(), mock.patch.object(
                private_context, "load_context", side_effect=self.guards.InvalidSubmission("unexpected prerequisite")):
            with self.assertRaises(self.guards.InvalidSubmission):
                self.evaluator.evaluate(str(self.signature_patch()))

    def test_pristine_and_reference_patch_still_pass_source_audit(self):
        pristine = (TASK / "judge/harbor/app" / self.guards.POLICY).read_bytes()
        self.guards.verify_method_only(pristine, pristine)
        with self.evaluator.prepared_submission(TASK / "judge/reference.patch", self.config) as (_, evidence):
            self.assertEqual(evidence.helper_count, 0)

    def test_unclassified_patch_preparation_failure_is_not_candidate_rejection(self):
        failed = SimpleNamespace(ok=False, workdir=None, candidate_rejection=False)
        with mock.patch.object(self.evaluator, "apply_submission_patch", return_value=failed):
            with self.assertRaises(self.guards.TaskIntegrityError):
                with self.evaluator.prepared_submission(self.signature_patch(), self.config):
                    pass

    def test_patch_helper_distinguishes_policy_refusal_from_missing_infrastructure(self):
        import submission_patch
        patch = self.signature_patch()
        missing = submission_patch.apply_submission_patch(patch, self.root, self.config)
        self.assertFalse(missing.ok)
        self.assertFalse(missing.candidate_rejection)
        invalid = self.root / "invalid.patch"
        invalid.write_text("not a patch\n")
        rejected = submission_patch.apply_submission_patch(invalid, TASK / "judge", self.config)
        self.assertFalse(rejected.ok)
        self.assertTrue(rejected.candidate_rejection)
        with mock.patch.object(submission_patch.subprocess, "run", side_effect=FileNotFoundError("git missing")):
            with self.assertRaises(FileNotFoundError):
                submission_patch.apply_submission_patch(patch, TASK / "judge", self.config)

    def test_diagnostic_free_text_or_changed_pinned_helper_is_refused(self):
        import candidate_failure
        value = candidate_failure.diagnostic("source_policy")
        value["message"] += " secret detail"
        with self.assertRaises(ValueError):
            candidate_failure.public_diagnostic(value)
        (self.bundle / "candidate_failure.py").write_text("raise RuntimeError('changed')\n")
        with self.assertRaises(self.wrapper.VerifierFailure) as caught:
            self.wrapper.load_candidate_diagnostic(self.bundle)
        self.assertEqual(caught.exception.code, "verifier_configuration_error")


if __name__ == "__main__":
    unittest.main()
