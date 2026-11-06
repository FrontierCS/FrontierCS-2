"""Candidate refusals are scored rejections; trusted prerequisites remain errors (#757)."""
import deployment_test_guard  # noqa: F401 # isort: skip

import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/Robotics/diffusion_policy_cpu"
sys.path.insert(0, str(ROOT / "scripts"))
import task_manifest  # noqa: E402


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
        self.judge = task_manifest.judge_bundle(TASK)
        shared = task_manifest.SHARED / "verifier"
        # The Verifier base puts tasks/shared/lib (isolate) beside the verifier modules.
        sys.path[:0] = [str(self.judge), str(shared), str(task_manifest.SHARED / "lib")]
        for name in ("evaluator", "task_guards", "private_context", "measurement_runtime",
                     "native_data", "option_b", "runtime_layout", "hardware_dataset",
                     "beat_spec_feedback"):
            sys.modules.pop(name, None)
        self.evaluator = load("evaluator", self.judge / "evaluator.py")
        self.wrapper = load("diffusion_rejection_wrapper", shared / "fcs2_verifier.py")
        import task_guards
        self.guards = task_guards
        os.environ["FRONTIER_VERIFIER_SCRATCH"] = str(self.root)

    def trusted_prerequisites(self):
        """Only prerequisite IO is synthetic; the candidate source rules are real."""
        import hardware_dataset
        import native_data
        import private_context
        import runtime_layout
        context = {"seal": {"runtime": {"prefix": "unused", "checkpoint": "unused"}},
                   "suites": {"dev": {}, "final": {}}}
        stack = ExitStack()
        stack.enter_context(mock.patch.object(hardware_dataset, "actual_identity", return_value={}))
        stack.enter_context(mock.patch.object(hardware_dataset, "select_dataset",
                                              return_value=self.root / "selected.tar"))
        stack.enter_context(mock.patch.object(native_data, "materialize_dataset",
                                              return_value=nullcontext(self.root)))
        stack.enter_context(mock.patch.object(private_context, "load_context", return_value=context))
        stack.enter_context(mock.patch.object(private_context, "accepted_calibrations",
                                              return_value={"dev": {}, "final": {}}))
        stack.enter_context(mock.patch.object(runtime_layout, "validate_layout"))
        stack.enter_context(mock.patch.object(os, "umask"))
        return stack

    def patch_of(self, old, new):
        repo = self.root / "repo"
        if not repo.exists():
            shutil.copytree(TASK / "repo", repo)
            env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CEILING_DIRECTORIES": str(self.root)}
            for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t",
                                                         "commit", "-qm", "base"]):
                subprocess.run(["git", "-C", str(repo), *args], check=True, env=env)
        policy = repo / self.guards.POLICY
        source = policy.read_text()
        policy.write_text(source.replace(old, new, 1))
        patch = subprocess.run(["git", "-C", str(repo), "diff"], check=True, capture_output=True).stdout
        policy.write_text(source)
        return patch

    def transport(self, mode, patch, evaluate=None):
        os.environ["FRONTIER_VERIFICATION_MODE"] = mode
        os.environ["FRONTIER_DATASET_PATH"] = str(TASK / f"datasets/{mode}.json")
        payload = self.root / "payload.tar"
        with tarfile.open(payload, "w", format=tarfile.USTAR_FORMAT) as archive:
            member = tarfile.TarInfo("payload/solution.patch")
            member.size = len(patch)
            archive.addfile(member, io.BytesIO(patch))
        loaded = (evaluate or self.evaluator.evaluate, self.evaluator.check_tree)
        with mock.patch.object(self.wrapper, "load_evaluator", return_value=loaded):
            return self.wrapper.verify(
                bundle=self.judge, entrypoint="payload/solution.patch", payload=payload,
                dataset=TASK / f"datasets/{mode}.json", mode=mode,
                result_path=self.root / "result.json", evidence_path=self.root / "evidence.tar.zst")

    def test_signature_whitespace_is_an_out_of_scope_rejection_in_train_and_final(self):
        patch = self.patch_of("def conditional_sample(", "def conditional_sample (")
        for mode in ("train", "eval"):
            result = self.transport(mode, patch)
            self.assertEqual((result["status"], result["raw_score"]), ("passed", 0.0))
            self.assertEqual(result["metrics"], {"score_unbounded": 0.0, "rejected": "out_of_scope"})

    def test_protocol_violation_during_measurement_is_a_content_policy_rejection(self):
        import measurement_runtime
        body = self.patch_of("        raise NotImplementedError(\"Return a normalized", "        # candidate\n        raise NotImplementedError(\"Return a normalized")
        with self.trusted_prerequisites(), mock.patch.object(
                measurement_runtime, "measure_suite",
                side_effect=self.guards.InvalidSubmission("Candidate measurement did not complete")):
            for mode in ("train", "eval"):
                result = self.transport(mode, body)
                self.assertEqual(result["metrics"], {"score_unbounded": 0.0, "rejected": "content_policy"})
                if mode == "eval":
                    self.assertEqual(result["private_metrics"]["beats_reference"], 0)

    def test_candidate_timeouts_are_scored_zeros(self):
        import measurement_runtime
        body = self.patch_of("        raise NotImplementedError(\"Return a normalized",
                             "        # candidate\n        raise NotImplementedError(\"Return a normalized")
        with self.trusted_prerequisites(), mock.patch.object(
                measurement_runtime, "measure_suite",
                side_effect=self.guards.CandidateTimeout("The suite measurement budget ran out")):
            for mode in ("train", "eval"):
                result = self.transport(mode, body)
                self.assertEqual((result["status"], result["raw_score"]), ("passed", 0.0))
                self.assertEqual(result["metrics"], {"score_unbounded": 0.0})
                if mode == "eval":
                    self.assertEqual(result["private_metrics"], {"beats_reference": 0})

    def test_episode_timeout_is_the_candidates_only_on_the_candidate_side(self):
        import subprocess as sp
        import measurement_runtime
        context = {"root": str(self.root), "mode": "dev",
                   "seal": {"runtime": {"checkpoint": "c", "prefix": str(self.root)}}}
        with mock.patch.object(measurement_runtime, "bounded_process",
                               side_effect=sp.TimeoutExpired("episode", 150)):
            for side, error in (("candidate", self.guards.CandidateTimeout),
                                ("reference", self.guards.TaskIntegrityError)):
                case = {"side": side, "run": {"index": 0 if side == "candidate" else 1}}
                with self.assertRaises(error):
                    measurement_runtime.launch_episode(context, "dev", case, self.root,
                                                       self.judge / "pristine")
        failure = self.root / "episode-02"
        failure.mkdir()
        (failure / "failure.json").write_text('{"category": "candidate_timeout"}')
        with mock.patch.object(measurement_runtime, "bounded_process", return_value=(1, 3.0)):
            with self.assertRaises(self.guards.CandidateTimeout):
                measurement_runtime.launch_episode(context, "dev",
                                                   {"side": "candidate", "run": {"index": 2}},
                                                   self.root, self.judge / "pristine")

    def test_infrastructure_and_unexpected_errors_stay_evaluator_errors(self):
        import measurement_runtime
        body = self.patch_of("        raise NotImplementedError(\"Return a normalized", "        # candidate\n        raise NotImplementedError(\"Return a normalized")
        for error in (OSError("disk unavailable"), ValueError("malformed trusted data"),
                      self.guards.TaskIntegrityError("integrity"), RuntimeError("unexpected")):
            with self.subTest(error=type(error).__name__), self.trusted_prerequisites(), mock.patch.object(
                    measurement_runtime, "measure_suite", side_effect=error):
                result = self.transport("eval", body)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["failure"]["code"], "evaluator_error")

    def test_prerequisite_invalid_submission_is_not_candidate_zero(self):
        import private_context
        os.environ["FRONTIER_VERIFICATION_MODE"] = "train"
        os.environ["FRONTIER_DATASET_PATH"] = str(TASK / "datasets/train.json")
        with self.trusted_prerequisites(), mock.patch.object(
                private_context, "load_context",
                side_effect=self.guards.InvalidSubmission("unexpected prerequisite")):
            with self.assertRaises(self.guards.InvalidSubmission):
                self.evaluator.evaluate(str(self.judge / "pristine"))

    def test_pristine_and_reference_pass_the_source_rules(self):
        self.assertIsNone(self.evaluator.check_tree(str(self.judge / "pristine"),
                                                    str(self.judge / "pristine"), []))
        with self.evaluator.reference_source() as tree:
            self.assertIsNone(self.evaluator.check_tree(str(tree), str(self.judge / "pristine"), []))

    def test_check_tree_keeps_the_rules_task_toml_cannot_express(self):
        pristine = self.judge / "pristine"
        cases = {
            "out_of_scope": lambda tree: (tree / self.guards.POLICY).write_text(
                (pristine / self.guards.POLICY).read_text() + "# outside the body\n"),
            "too_large": lambda tree: [(tree / self.guards.HELPERS / f"h{i}.py").write_text("x = 1\n")
                                       for i in range(self.guards.MAX_HELPER_FILES + 1)],
            "content_policy": lambda tree: (tree / self.guards.HELPERS / "h.py").write_text("def (:\n"),
        }
        for category, change in cases.items():
            with self.subTest(category=category):
                scratch = Path(tempfile.mkdtemp(dir=self.root))
                tree = self.evaluator._writable_copy(pristine, scratch)
                (tree / self.guards.HELPERS).mkdir(parents=True, exist_ok=True)
                change(tree)
                self.assertEqual(self.evaluator.check_tree(str(tree), str(pristine), [])[0], category)


if __name__ == "__main__":
    unittest.main()
