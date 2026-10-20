"""Guard the issue131-provenance-r003 RAGEN judge contract as published in Preview.

The earlier structure-rejection judge was replaced by trusted action, prompt and
assistant provenance checks installed around the real model call. A trusted
provenance rejection is an isolated scored zero carrying a closed diagnostic;
patch and source preflight failures are plain scored zeros that never dispatch
GPU work; infrastructure, execution-evidence and reference failures stay
unscored; strict per-group comparison against the exact reference is retained.
Fake tensors and processes here are protocol checks, not GPU acceptance.
"""

from __future__ import annotations

import deployment_test_guard  # noqa: F401 # isort: skip

import ast
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK = Path(
    os.environ.get(
        "RAGEN_PROVENANCE_EXPORT",
        str(
            ROOT
            / "tasks/frontier-cs-2.0-demo/problems/Machine Learning/ragen_multi_turn_rl"
        ),
    )
)
CONTEXT = "ragen/llm_agent/ctx_manager.py"
PROVENANCE = frozenset(
    {"action_provenance", "prompt_provenance", "assistant_provenance"}
)
REJECTED = "candidate structure rejected"
PREFLIGHT = "candidate source preflight rejected before execution"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def observations(success=0.5, reward=1.0, state="same-state"):
    return {
        "groups": {
            str(i): {"success_rate": success, "average_episodic_reward": reward}
            for i in range(6)
        },
        "initial_state_hashes": {str(i): state for i in range(6)},
    }


class RagenProvenanceRejectionTests(unittest.TestCase):
    def setUp(self):
        old_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        self.addCleanup(lambda: setattr(sys, "dont_write_bytecode", old_bytecode))
        old_modules, old_path = dict(sys.modules), list(sys.path)

        def restore():
            sys.path[:] = old_path
            for key in set(sys.modules) - set(old_modules):
                sys.modules.pop(key, None)
            sys.modules.update(old_modules)

        self.addCleanup(restore)
        sys.path[:0] = [str(TASK / "judge"), str(TASK / "verifier")]
        for name in (
            "candidate_failure",
            "private_execution",
            "action_provenance",
            "submission_patch",
            "beat_spec_feedback",
        ):
            sys.modules.pop(name, None)
        self.failure = module(TASK / "judge/candidate_failure.py", "candidate_failure")
        self.private = module(TASK / "judge/private_execution.py", "private_execution")
        self.provenance = module(
            TASK / "judge/action_provenance.py", "action_provenance"
        )
        self.evaluator = module(TASK / "judge/evaluator.py", "test_ragen_evaluator")
        self.verifier = module(
            TASK / "verifier/fcs2_verifier.py", "test_ragen_verifier"
        )

    # ------------------------------------------------------------- helpers

    def rejection(self, reason="action_provenance"):
        return self.failure.CandidateSourceError(
            self.evaluator.provenance_diagnostic(reason, CONTEXT, 1)
        )

    def evaluate(self, candidate, reference=None):
        """Run evaluate() with patched patch application, CUDA and GPU dispatch.

        `candidate`/`reference` are either payload dicts the fake GPU run returns
        or exception instances it raises. Returns the result and dispatched roles.
        """
        e = self.evaluator
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name / "repo" for name in ("candidate", "reference")]
            for path in paths:
                path.mkdir(parents=True)
            applied = [
                SimpleNamespace(ok=True, message="ok", workdir=str(p)) for p in paths
            ]
            roles = []

            def gpu(workdir, seed, evidence, role):
                roles.append(role)
                outcome = candidate if role == "candidate" else reference
                if isinstance(outcome, BaseException):
                    if hasattr(outcome, "private_execution_evidence"):
                        evidence.update(outcome.private_execution_evidence)
                    raise outcome
                evidence[role] = {"valid": True, "process_returncode": 0}
                return copy.deepcopy(outcome)

            with (
                patch.object(e, "apply_submission_patch", side_effect=applied),
                patch.object(e, "_source_valid", return_value=(True, "ok")),
                patch.object(
                    e, "_require_verifier_cuda", return_value={"cuda_available": True}
                ),
                patch.object(e, "_run_gpu", side_effect=gpu),
            ):
                result = e.evaluate("solution.patch")
            return result, roles

    def verify(self, value, mode, evaluate=None):
        v = self.verifier
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            root = Path(tmp)
            payload = root / "payload"
            dataset = root / "dataset"
            payload.write_bytes(b"p")
            dataset.write_bytes(b"d")
            with (
                patch.object(v, "extract_payload"),
                patch.object(v, "entrypoint_path", return_value=payload),
                patch.object(v, "stage_task", return_value=TASK / "judge"),
                patch.object(
                    v, "load_evaluator", return_value=evaluate or (lambda _: value)
                ),
            ):
                return v.verify(
                    bundle=TASK / "judge",
                    entrypoint="solution.patch",
                    payload=payload,
                    dataset=dataset,
                    mode=mode,
                    result_path=root / "result.json",
                    evidence_path=root / "evidence.tar.zst",
                )

    # --------------------------------------------- zeros before execution

    def test_patch_failure_is_a_plain_scored_zero_without_gpu_dispatch(self):
        e = self.evaluator
        failed = SimpleNamespace(ok=False, message="patch failed", workdir=None)

        def forbidden(*args, **kwargs):
            raise AssertionError("a rejected patch reached GPU dispatch")

        with (
            patch.object(e, "apply_submission_patch", return_value=failed),
            patch.object(e, "_run_gpu", side_effect=forbidden),
            patch.object(e, "_require_verifier_cuda", side_effect=forbidden),
        ):
            self.assertEqual(e.evaluate("patch"), (0.0, 0.0, "patch failed", {}))

    def test_unreadable_or_invalid_source_is_a_scored_zero_before_execution(self):
        e = self.evaluator
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self.assertEqual(
                e._source_valid(workdir), (False, "ctx_manager.py is unreadable")
            )
            context = workdir / CONTEXT
            context.parent.mkdir(parents=True)
            context.write_text("def broken(:\n", encoding="utf-8")
            self.assertEqual(
                e._source_valid(workdir), (False, "ctx_manager.py does not compile")
            )

            def forbidden(*args, **kwargs):
                raise AssertionError("invalid source reached execution")

            with (
                patch.object(
                    e,
                    "apply_submission_patch",
                    return_value=SimpleNamespace(
                        ok=True, message="ok", workdir=str(workdir)
                    ),
                ),
                patch.object(e, "_run_gpu", side_effect=forbidden),
                patch.object(e, "_require_verifier_cuda", side_effect=forbidden),
            ):
                self.assertEqual(
                    e.evaluate("patch"),
                    (0.0, 0.0, "ctx_manager.py does not compile", {}),
                )

    def test_plain_zero_results_carry_no_private_metrics(self):
        for mode in ("eval", "train"):
            result = self.verify(self.evaluator._zero("invalid patch"), mode)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["metrics"], {"score_unbounded": 0.0})
            self.assertNotIn("private_metrics", result)

    # ----------------------------------------- trusted provenance rejection

    def test_trusted_provenance_rejection_is_an_isolated_scored_zero(self):
        for reason in sorted(PROVENANCE):
            with self.subTest(reason=reason):
                value, roles = self.evaluate(self.rejection(reason))
                # The exact reference is never dispatched for a rejected candidate.
                self.assertEqual(roles, ["candidate"])
                diagnostic = self.failure.diagnostic(reason, CONTEXT, 1)
                self.assertEqual(
                    value, (0.0, 0.0, REJECTED, {"candidate_diagnostic": diagnostic})
                )
                final = self.verify(value, "eval")
                self.assertEqual(final["metrics"], {"score_unbounded": 0.0})
                self.assertNotIn("private_metrics", final)
                train = self.verify(value, "train")
                self.assertEqual(
                    train["metrics"],
                    {"score_unbounded": 0.0, "candidate_diagnostic": diagnostic},
                )

    def test_source_preflight_rejection_is_reported_distinctly(self):
        diagnostic = self.failure.diagnostic("syntax_error", CONTEXT, 7)
        value, roles = self.evaluate(self.failure.CandidateSourceError(diagnostic))
        self.assertEqual(roles, ["candidate"])
        self.assertEqual(
            value, (0.0, 0.0, PREFLIGHT, {"candidate_diagnostic": diagnostic})
        )
        self.assertEqual(
            self.verify(value, "train")["metrics"]["candidate_diagnostic"], diagnostic
        )

    def test_rejection_results_must_stay_isolated_for_the_verifier(self):
        value, _ = self.evaluate(self.rejection())
        diagnostic = value[3]["candidate_diagnostic"]
        tampered = {
            "extra_metrics": (
                *value[:3],
                {**value[3], "private_execution": {"candidate": {"valid": True}}},
            ),
            "nonzero_score": (1.0, 1.0, value[2], value[3]),
            "untrusted_message": (
                *value[:3],
                {"candidate_diagnostic": {**diagnostic, "message": "private secret"}},
            ),
            "unapproved_category": (
                *value[:3],
                {"candidate_diagnostic": {**diagnostic, "category": "row_count"}},
            ),
            "extra_field": (
                *value[:3],
                {"candidate_diagnostic": {**diagnostic, "extra": 1}},
            ),
            "legacy_marker": (
                0.0,
                0.0,
                REJECTED,
                {"candidate_rejection": True, "beats_reference": 0},
            ),
        }
        for change, result in tampered.items():
            for mode in ("eval", "train"):
                with (
                    self.subTest(change=change, mode=mode),
                    self.assertRaises(self.verifier.VerifierFailure),
                ):
                    self.verify(result, mode)

    def test_closed_diagnostics_do_not_accept_untrusted_text(self):
        diagnostic = self.failure.diagnostic("action_provenance", CONTEXT, 1)
        self.assertEqual(self.failure.public_diagnostic(diagnostic), diagnostic)
        for change in (
            {"message": "private secret"},
            {"category": "unapproved"},
            {"source": "judge/evaluator.py"},
            {"line": 0},
            {"extra": 1},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.failure.public_diagnostic({**diagnostic, **change})
        with self.assertRaises(ValueError):
            self.provenance.ProvenanceError("unapproved")

    def test_closed_reason_set_is_shared_by_proxy_evaluator_and_diagnostics(self):
        self.assertEqual(set(self.provenance.REASONS), PROVENANCE)
        self.assertLessEqual(PROVENANCE, set(self.failure.GUIDANCE))
        for reason in PROVENANCE:
            self.assertTrue(
                self.failure.GUIDANCE[reason].startswith("Candidate structure rejected")
            )
            self.assertEqual(
                self.evaluator.provenance_diagnostic(reason, CONTEXT, 1)["category"],
                reason,
            )
        # Every literal reason the trusted proxy raises is declared; a reason
        # passed through a variable is still validated by ProvenanceError itself.
        tree = ast.parse((TASK / "judge/action_provenance.py").read_text("utf-8"))
        raised = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Raise)
                and isinstance(node.exc, ast.Call)
                and isinstance(node.exc.func, ast.Name)
                and node.exc.func.id == "ProvenanceError"
            ):
                self.assertEqual(len(node.exc.args), 1)
                if isinstance(node.exc.args[0], ast.Constant):
                    raised.add(node.exc.args[0].value)
        self.assertTrue(raised)
        self.assertLessEqual(raised, PROVENANCE)

    def test_real_dispatch_classifies_only_declared_provenance_files(self):
        e = self.evaluator
        for reason, declared in (
            ("action_provenance", True),
            ("prompt_provenance", True),
            ("assistant_provenance", True),
            ("row_count", False),
            ("tensor_shapes", False),
        ):
            with (
                self.subTest(reason=reason),
                tempfile.TemporaryDirectory() as tmp,
                patch.dict(os.environ),
            ):
                root = Path(tmp)
                (root / "models/qwen2.5-0.5b-instruct").mkdir(parents=True)
                os.environ["FRONTIER_MODEL_ROOT"] = str(root / "models")
                work = root / "repo"
                work.mkdir()
                calls = []

                def run(command, *, calls=calls, work=work, reason=reason, **kwargs):
                    calls.append(list(command))
                    completed = SimpleNamespace(args=list(command), stdout="", stderr="")
                    if len(calls) == 4:
                        evidence = work / ".frontier-private-execution"
                        evidence.mkdir()
                        (evidence / "provenance-rejection.json").write_text(
                            json.dumps(
                                {"schema": "ragen.action-provenance/v1", "reason": reason}
                            )
                        )
                        completed.returncode = 1
                        return completed
                    completed.returncode, completed.stdout = 0, "null"
                    return completed

                with (
                    patch.object(e, "_install_instrumentation"),
                    patch.object(e.subprocess, "run", side_effect=run),
                ):
                    if declared:
                        with self.assertRaises(self.failure.CandidateSourceError) as caught:
                            e._run_gpu(work, 271828, {}, "candidate")
                        self.assertEqual(
                            caught.exception.diagnostic["category"], reason
                        )
                    else:
                        # An undeclared reason is not a trusted rejection: the failed
                        # process stays an unscored execution-evidence failure.
                        with self.assertRaises(self.private.ExecutionEvidenceError):
                            e._run_gpu(work, 271828, {}, "candidate")
                self.assertEqual(len(calls), 4)

    # ------------------------------------------------ unscored failures

    def test_evidence_and_unknown_runtime_failures_stay_unscored(self):
        evidence_error = self.private.ExecutionEvidenceError(
            {
                "candidate": {
                    "valid": False,
                    "process_returncode": 1,
                    "issues": ["training_process_failed"],
                }
            }
        )
        for error in (RuntimeError("CUDA failure"), evidence_error):
            with self.subTest(error=type(error)), self.assertRaises(type(error)):
                self.evaluate(error)

        def raise_evidence(_):
            raise evidence_error

        failed = self.verify(None, "eval", evaluate=raise_evidence)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failure"]["code"], "invalid_execution_evidence")
        self.assertEqual(failed["metrics"], {})

        def crash(_):
            raise RuntimeError("CUDA failure")

        with self.assertRaises(self.verifier.VerifierFailure) as caught:
            self.verify(None, "eval", evaluate=crash)
        self.assertEqual(caught.exception.code, "evaluator_error")

    def test_reference_failure_is_infrastructure_not_a_scientific_fail(self):
        with self.assertRaises(self.evaluator.InfrastructureError) as caught:
            self.evaluate(observations(), RuntimeError("reference failed"))
        evidence = caught.exception.private_execution_evidence
        self.assertEqual(evidence["candidate"], {"valid": True, "process_returncode": 0})
        self.assertIsNone(evidence["reference"])

    def test_generic_structure_runtime_error_has_no_candidate_marker(self):
        def crash():
            raise RuntimeError("unclassified")

        with self.assertRaises(RuntimeError):
            self.private.observe_validation(crash, "generic")()
        record = self.private._DIAGNOSTICS["generic"].record
        self.assertFalse(
            any(event["event"] == "candidate_rejection" for event in record["events"])
        )

    def typed_failure(self, reason="terminal_reward"):
        record = self.private.DiagnosticRecord()
        record.add(event="structure", status="entered")
        record.add(event="structure", status="failed")
        record.add(event="candidate_rejection", reason=reason)
        record.failure("structure", self.private.CandidateStructureError(reason))
        return self.private.ExecutionEvidenceError({"candidate": {
            "valid": False, "process_returncode": 1, "diagnostics": record.record,
        }})

    def test_typed_candidate_rejection_scores_zero_in_train_and_final(self):
        value, roles = self.evaluate(self.typed_failure(), observations())
        self.assertEqual(roles, ["candidate", "reference"])
        self.assertEqual(value[:2], (0.0, 0.0))
        self.assertEqual(value[3]["beats_reference"], 0)
        self.assertEqual(value[3]["candidate_diagnostic"]["reason"], "terminal_reward")
        for mode in ("train", "eval"):
            result = self.verify(value, mode)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["raw_score"], 0.0)
            if mode == "eval":
                self.assertEqual(result["private_metrics"]["beats_reference"], 0)
            else:
                self.assertEqual(result["metrics"]["candidate_diagnostic"]["reason"], "terminal_reward")

    def test_typed_zero_retains_reference_control_in_evidence(self):
        value, _ = self.evaluate(self.typed_failure(), observations())
        with patch.object(self.verifier, "write_evidence") as write:
            self.verify(value, "eval")
        record = json.loads(write.call_args.args[1]["metrics.json"])
        execution = record["metrics"]["private_execution"]
        self.assertTrue(execution["reference"]["valid"])
        self.assertEqual(execution["reference"]["process_returncode"], 0)
        self.assertFalse(execution["candidate"]["valid"])
        self.assertEqual(record["metrics"]["candidate_diagnostic"]["reason"], "terminal_reward")

    def test_verifier_rejects_unbound_typed_zero(self):
        value, _ = self.evaluate(self.typed_failure(), observations())
        for fault in ("missing_reference", "failed_reference", "wrong_reason", "duplicate_reason"):
            altered = copy.deepcopy(value)
            execution = altered[3]["private_execution"]
            if fault == "missing_reference":
                execution.pop("reference")
            elif fault == "failed_reference":
                execution["reference"]["valid"] = False
            elif fault == "wrong_reason":
                altered[3]["candidate_diagnostic"] = self.failure.structure_diagnostic("binary_masks")
            else:
                execution["candidate"]["diagnostics"]["events"].append(
                    {"event": "candidate_rejection", "reason": "terminal_reward"})
            with self.subTest(fault=fault), self.assertRaises(self.verifier.VerifierFailure):
                self.verify(altered, "eval")

    def test_typed_rejection_requires_successful_exact_reference(self):
        with self.assertRaises(self.evaluator.InfrastructureError):
            self.evaluate(self.typed_failure(), RuntimeError("trusted reference failed"))
        failure = self.typed_failure()
        with self.assertRaises(self.private.ExecutionEvidenceError):
            self.evaluate(failure, failure)

    def test_typed_rejection_requires_unique_complete_diagnostic(self):
        for fault in ("drop", "duplicate", "bad_returncode", "unknown_reason"):
            failure = self.typed_failure()
            candidate = failure.private_execution_evidence["candidate"]
            if fault == "drop":
                candidate["diagnostics"]["dropped_events"] = 1
            elif fault == "duplicate":
                candidate["diagnostics"]["events"].append({"event":"candidate_rejection", "reason":"terminal_reward"})
            elif fault == "bad_returncode":
                candidate["process_returncode"] = 137
            else:
                candidate["diagnostics"]["events"][2]["reason"] = "unknown"
            with self.subTest(fault=fault), self.assertRaises((self.private.ExecutionEvidenceError, ValueError)):
                self.evaluate(failure, observations())

    # ------------------------------------------------- measured scoring

    def test_measured_scoring_retains_strict_per_group_comparison(self):
        for candidate_success, reward, expected in ((0.5, 1.0, 0), (0.75, 2.0, 1)):
            with self.subTest(expected=expected):
                value, roles = self.evaluate(
                    observations(candidate_success, reward), observations(0.5, 1.0)
                )
                self.assertEqual(roles, ["candidate", "reference"])
                self.assertNotIn("candidate_diagnostic", value[3])
                self.assertEqual(value[3]["beats_reference"], expected)
                final = self.verify(value, "eval")
                self.assertEqual(final["private_metrics"]["beats_reference"], expected)
                self.assertIn("reference_score", final["private_metrics"])

    def test_differing_initial_states_are_a_scored_zero(self):
        value, _ = self.evaluate(observations(state="other"), observations())
        self.assertEqual(value, (0.0, 0.0, "candidate and reference initial states differ", {}))

    # --------------------------------------------- emitted tensor guard

    def test_emitted_tensor_validator_raises_typed_contract_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "ragen/llm_agent"
            folder.mkdir(parents=True)
            for filename in ("ctx_manager.py", "es_manager.py"):
                (folder / filename).write_text("# pristine fixture\n")
            with patch.object(self.evaluator, "install_execution"):
                self.evaluator._install_instrumentation(root, root / "out.json", 1)
            source = (folder / "ctx_manager.py").read_text()
            tree = ast.parse(source)
            compile(tree, "ctx_manager.py", "exec")
            validator = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_frontier_validate_result"
            )
            raises = [node for node in ast.walk(validator) if isinstance(node, ast.Raise)]
            self.assertTrue(raises)
            for node in raises:
                self.assertIsInstance(node.exc, ast.Call)
                self.assertIsInstance(node.exc.func, ast.Attribute)
                self.assertEqual(node.exc.func.attr, "CandidateStructureError")
                self.assertIn(node.exc.args[0].value, self.private.STRUCTURE_REASONS)
                self.assertIsInstance(node.exc.args[0], ast.Constant)
            self.assertNotIn("candidate_rejection", source)
            self.assertNotIn("provenance", source)

    def test_trajectory_layout_rejects_misaligned_responses(self):
        rows = {
            "input_ids": [[0, 11, 12, 13], [21, 22, 23, 24]],
            "attention_mask": [[0, 1, 1, 1], [1, 1, 1, 1]],
            "position_ids": [[0, 1, 2, 3], [1, 2, 3, 4]],
            "responses": [[11, 12, 13], [22, 23, 24]],
            "loss_mask": [[0, 1, 1], [1, 1, 1]],
            "rm_scores": [[0, 0, 1], [0, 0, 0]],
            "original_rm_scores": [[0, 0, 1], [0, 0, 0]],
        }
        self.private.validate_trajectory_layout(rows, [1, 2], [1, 2], 0)
        bad = copy.deepcopy(rows)
        for name in ("responses", "loss_mask", "rm_scores", "original_rm_scores"):
            bad[name] = [row[-2:] for row in bad[name]]
        with self.assertRaises(RuntimeError):
            self.private.validate_trajectory_layout(bad, [1, 2], [1, 2], 0)


if __name__ == "__main__":
    unittest.main()
