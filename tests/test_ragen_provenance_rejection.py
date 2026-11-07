"""Guard the RAGEN judge contract on the shared verifier.

Trusted action, prompt and assistant provenance checks are installed around the real model
call. A trusted provenance rejection, a source-preflight failure and a typed DataProto
rejection are scored zeros with a fixed message: train metrics stay empty, final keeps the
closed diagnostic in private evidence only. A missing public signature is an out_of_scope
rejection by the check_tree hook; source that does not compile or is still stubbed is a plain
scored zero that never dispatches GPU work; infrastructure, execution-evidence and reference
failures stay unscored; strict per-group comparison against the exact reference is retained.
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
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK = Path(
    os.environ.get(
        "RAGEN_PROVENANCE_EXPORT",
        str(
            ROOT
            / "tasks/problems/Machine Learning/ragen_multi_turn_rl"
        ),
    )
)
SHARED = ROOT / "tasks/shared"
sys.path.insert(0, str(ROOT / "scripts"))
import task_manifest  # noqa: E402

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
        sys.path[:0] = [str(TASK / "judge"), str(SHARED / "verifier"), str(SHARED / "lib"),
                        str(SHARED / "verifier-gpu")]
        for name in (
            "source_preflight",
            "private_execution",
            "action_provenance",
            "edit_scope",
            "beat_spec_feedback",
        ):
            sys.modules.pop(name, None)
        self.failure = module(TASK / "judge/source_preflight.py", "source_preflight")
        self.private = module(TASK / "judge/private_execution.py", "private_execution")
        self.provenance = module(
            TASK / "judge/action_provenance.py", "action_provenance"
        )
        self.evaluator = module(TASK / "judge/evaluator.py", "test_ragen_evaluator")
        self.verifier = module(
            SHARED / "verifier/fcs2_verifier.py", "test_ragen_verifier"
        )
        self.bundle = task_manifest.judge_bundle(TASK)

    # ------------------------------------------------------------- helpers

    def rejection(self, reason="action_provenance"):
        return self.failure.CandidateSourceError(
            self.evaluator.provenance_diagnostic(reason, CONTEXT, 1)
        )

    def evaluate(self, candidate, reference=None, role="development"):
        """Run evaluate(tree) with patched reference preparation and sandbox runs.

        `candidate`/`reference` are either payload dicts the fake PPO evidence yields
        or exception instances judging it raises. Returns the result and the roles
        whose PPO evidence was judged.
        """
        e = self.evaluator
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name / "repo" for name in ("candidate", "reference")]
            for path in paths:
                path.mkdir(parents=True)
            roles = []

            def rollouts(result, evidence, role, seed):
                roles.append(role)
                outcome = candidate if role == "candidate" else reference
                if isinstance(outcome, BaseException):
                    if hasattr(outcome, "private_execution_evidence"):
                        evidence.update(outcome.private_execution_evidence)
                    raise outcome
                evidence[role] = {"valid": True, "process_returncode": 0}
                return copy.deepcopy(outcome)

            with (
                patch.dict(os.environ, {"FRONTIER_SUBMISSION_ROLE": role}),
                patch.object(e, "_reference_tree", return_value=paths[1]),
                patch.object(e, "_source_valid", return_value=(True, "ok")),
                patch.object(e, "sandbox_workdir", return_value="/workspace/sandbox"),
                patch.object(e, "_inspect", return_value=None),
                patch.object(e, "_sandbox_run", return_value=None),
                patch.object(e, "run_sandboxes", side_effect=lambda runs, into: list(runs)),
                patch.object(e, "_machine", return_value={"cuda_available": True}),
                patch.object(e, "_rollouts", side_effect=rollouts),
            ):
                result = e.evaluate(str(paths[0]))
            return result, roles

    def verify(self, value, mode, evaluate=None):
        """Run the shared verifier with an accepted edit scope and a stub evaluator."""
        v = self.verifier
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            root = Path(tmp)
            payload = root / "payload.tar"
            dataset = root / "dataset"
            payload.write_bytes(b"p")
            dataset.write_bytes(b"d")
            scope = SimpleNamespace(ok=True, tree=TASK / "repo", changes=(), metrics={},
                                    dispose=lambda: None)
            with (
                patch.object(v, "extract_payload"),
                patch.object(v, "entrypoint_path", return_value=payload),
                patch("edit_scope.check", return_value=scope),
                patch.object(
                    v, "load_evaluator",
                    return_value=(evaluate or (lambda _: value), None),
                ),
            ):
                return v.verify(
                    bundle=self.bundle,
                    entrypoint="payload/solution.patch",
                    payload=payload,
                    dataset=dataset,
                    mode=mode,
                    result_path=root / "result.json",
                    evidence_path=root / "evidence.tar.zst",
                )

    # --------------------------------------------- zeros before execution

    def test_missing_public_signature_is_out_of_scope(self):
        e = self.evaluator
        self.assertIsNone(e.check_tree(str(TASK / "repo"), str(TASK / "repo"), []))
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp)
            context = tree / CONTEXT
            context.parent.mkdir(parents=True)
            source = (TASK / "repo" / CONTEXT).read_text(encoding="utf-8")
            context.write_text(source.replace("def formulate_rollouts", "def renamed"),
                               encoding="utf-8")
            self.assertEqual(e.check_tree(str(tree), str(TASK / "repo"), []),
                             ("out_of_scope", [CONTEXT]))
            context.write_text("def broken(:\n", encoding="utf-8")
            self.assertIsNone(e.check_tree(str(tree), str(TASK / "repo"), []))

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
                patch.object(e, "_reference_tree", side_effect=forbidden),
                patch.object(e, "run_sandboxes", side_effect=forbidden),
            ):
                self.assertEqual(
                    e.evaluate(str(workdir)),
                    (0.0, 0.0, "ctx_manager.py does not compile", {}),
                )

    def test_plain_zero_results_carry_only_beats_reference_zero(self):
        for mode in ("eval", "train"):
            result = self.verify(self.evaluator._zero("invalid patch"), mode)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["metrics"], {"score_unbounded": 0.0})
            if mode == "eval":  # a final scored zero keeps beats_reference 0
                self.assertEqual(result["private_metrics"], {"beats_reference": 0})
            else:
                self.assertNotIn("private_metrics", result)

    # ----------------------------------------- trusted provenance rejection

    def test_trusted_provenance_rejection_is_an_isolated_scored_zero(self):
        for reason in sorted(PROVENANCE):
            with self.subTest(reason=reason):
                value, roles = self.evaluate(self.rejection(reason))
                # The exact reference's evidence is never judged for a rejected candidate.
                self.assertEqual(roles, ["candidate"])
                self.assertEqual(value, (0.0, 0.0, REJECTED, {}))
                train = self.verify(value, "train")
                self.assertEqual(train["metrics"], {"score_unbounded": 0.0})
                diagnostic = self.failure.diagnostic(reason, CONTEXT, 1)
                value, _ = self.evaluate(self.rejection(reason), role="final")
                self.assertEqual(
                    value, (0.0, 0.0, REJECTED, {"source_diagnostic": diagnostic})
                )
                final = self.verify(value, "eval")
                self.assertEqual(final["metrics"], {"score_unbounded": 0.0})
                self.assertEqual(final["private_metrics"], {"beats_reference": 0})

    def test_source_preflight_rejection_is_reported_distinctly(self):
        diagnostic = self.failure.diagnostic("syntax_error", CONTEXT, 7)
        value, roles = self.evaluate(self.failure.CandidateSourceError(diagnostic))
        self.assertEqual(roles, ["candidate"])
        self.assertEqual(value, (0.0, 0.0, PREFLIGHT, {}))
        self.assertEqual(self.verify(value, "train")["metrics"], {"score_unbounded": 0.0})
        value, _ = self.evaluate(self.failure.CandidateSourceError(diagnostic), role="final")
        self.assertEqual(value[3], {"source_diagnostic": diagnostic})

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
        sandbox = sys.modules["fcs2_verifier_gpu"]
        for reason, declared in (
            ("action_provenance", True),
            ("prompt_provenance", True),
            ("assistant_provenance", True),
            ("row_count", False),
            ("tensor_shapes", False),
        ):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                outputs = Path(tmp)
                evidence = outputs / "work/.frontier-private-execution"
                evidence.mkdir(parents=True, mode=0o700)
                (evidence / "provenance-rejection.json").write_text(
                    json.dumps({"schema": "ragen.action-provenance/v1", "reason": reason}))
                done = sandbox.StepResult(0, False, 1.0)
                result = sandbox.SandboxResult(
                    [done, done, done, sandbox.StepResult(1, False, 9.0)], outputs, False,
                    [None] * 4)
                if declared:
                    with self.assertRaises(self.failure.CandidateSourceError) as caught:
                        e._rollouts(result, {}, "candidate", 271828)
                    self.assertEqual(caught.exception.diagnostic["category"], reason)
                else:
                    # An undeclared reason is not a trusted rejection: the failed
                    # process stays an unscored execution-evidence failure.
                    with self.assertRaises(self.private.ExecutionEvidenceError):
                        e._rollouts(result, {}, "candidate", 271828)

    def test_a_lost_ppo_session_is_the_candidates_scored_zero(self):
        sandbox = sys.modules["fcs2_verifier_gpu"]
        done = sandbox.StepResult(0, False, 1.0)
        lost = sandbox.SandboxResult(
            [done, done, done, sandbox.StepResult(None, False, 30.0, lost_session=True)], None,
            False, [None] * 4)
        evidence = {}
        with self.assertRaises(self.private.ExecutionEvidenceError):
            self.evaluator._rollouts(lost, evidence, "candidate", 271828)
        self.assertEqual(self.evaluator._candidate_run_failure(evidence),
                         "the submission's PPO run lost its session")
        self.assertIsNone(self.evaluator._machine(lost, "candidate"))
        with self.assertRaises(self.evaluator.InfrastructureError):
            self.evaluator._machine(lost, "reference")
        # From the build step on submitted code has run: the submission's; before, infrastructure.
        build = sandbox.SandboxResult(
            [done, sandbox.StepResult(None, False, 3.0, lost_session=True), None, None], None,
            False, [None] * 4)
        self.assertIsNone(self.evaluator._machine(build, "candidate"))
        evidence = {}
        with self.assertRaises(self.private.ExecutionEvidenceError):
            self.evaluator._rollouts(build, evidence, "candidate", 271828)
        self.assertIn("training_session_lost", evidence["candidate"]["issues"])
        device = sandbox.SandboxResult(
            [sandbox.StepResult(None, False, 3.0, lost_session=True), None, None, None], None,
            False, [None] * 4)
        with self.assertRaises(self.evaluator.InfrastructureError):
            self.evaluator._machine(device, "candidate")

    def test_scoring_uses_the_replayed_groups_and_a_replay_mismatch_is_invalid_evidence(self):
        sandbox = sys.modules["fcs2_verifier_gpu"]
        e = self.evaluator
        reported = observations(success=1.0, reward=9.0)
        replayed = observations(success=0.25, reward=-0.5)["groups"]
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            (outputs / "work/.frontier-private-execution").mkdir(parents=True, mode=0o700)
            (outputs / "work/.frontier_rollouts.json").write_text(json.dumps(
                {**reported, "structural_ok": True, "episodes": [{"env_id": 0}]}))
            done = sandbox.StepResult(0, False, 1.0)
            result = sandbox.SandboxResult([done] * 4, outputs, False, [None] * 4)
            valid = {"valid": True, "issues": []}
            with (patch.object(e, "collect_execution", side_effect=lambda _: dict(valid, issues=[])),
                  patch.object(e, "_replay", return_value={"groups": replayed, "mismatch": None})):
                payload = e._rollouts(result, {}, "candidate", 271828)
            self.assertEqual(payload["groups"], replayed)
            self.assertEqual(payload["reported_groups"], reported["groups"])
            evidence = {}
            with (patch.object(e, "collect_execution", side_effect=lambda _: dict(valid, issues=[])),
                  patch.object(e, "_replay", return_value={"groups": replayed,
                                                           "mismatch": "episode 3: initial state"}),
                  self.assertRaises(self.private.ExecutionEvidenceError)):
                e._rollouts(result, evidence, "candidate", 271828)
            self.assertIn("rollout_replay_mismatch", evidence["candidate"]["issues"])

    def test_a_ppo_timeout_is_the_candidate_timeout_issue(self):
        sandbox = sys.modules["fcs2_verifier_gpu"]
        with tempfile.TemporaryDirectory() as tmp:
            done = sandbox.StepResult(0, False, 1.0)
            result = sandbox.SandboxResult(
                [done, done, done, sandbox.StepResult(None, True, 755.0)], Path(tmp), False,
                [None] * 4)
            evidence = {}
            with self.assertRaises(self.private.ExecutionEvidenceError):
                self.evaluator._rollouts(result, evidence, "candidate", 271828)
        self.assertIn("training_process_timeout", evidence["candidate"]["issues"])
        self.assertEqual(self.evaluator._candidate_run_failure(evidence),
                         "the submission's PPO run exceeded its 750-second deadline")

    # ------------------------------------------------ unscored failures

    def test_evidence_and_unknown_runtime_failures_stay_unscored(self):
        evidence_error = self.private.ExecutionEvidenceError(
            {
                "candidate": {
                    "valid": False,
                    "process_returncode": 1,
                    "issues": ["rollout_evidence_missing"],
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

        crashed = self.verify(None, "eval", evaluate=crash)
        self.assertEqual(crashed["status"], "failed")
        self.assertEqual(crashed["failure"]["code"], "evaluator_error")

    def test_candidate_crash_or_timeout_is_a_scored_zero_after_the_reference(self):
        for issue, message in (
                ("training_process_failed", "the submission's PPO run failed"),
                ("training_process_timeout",
                 "the submission's PPO run exceeded its 750-second deadline")):
            with self.subTest(issue=issue):
                error = self.private.ExecutionEvidenceError({"candidate": {
                    "valid": False, "process_returncode": 1, "issues": [issue]}})
                value, roles = self.evaluate(error, observations())
                self.assertEqual(roles, ["candidate", "reference"])
                self.assertEqual(value, (0.0, 0.0, message, {}))
                with self.assertRaises(self.evaluator.InfrastructureError):
                    self.evaluate(error, RuntimeError("reference failed"))

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
        for mode, role in (("train", "development"), ("eval", "final")):
            value, roles = self.evaluate(self.typed_failure(), observations(), role=role)
            self.assertEqual(roles, ["candidate", "reference"])
            self.assertEqual(value[:3], (0.0, 0.0, "candidate training DataProto violates the task contract"))
            if mode == "train":
                self.assertEqual(value[3], {})
            else:
                self.assertEqual(value[3]["candidate_rejection"], "terminal_reward")
            result = self.verify(value, mode)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["raw_score"], 0.0)
            self.assertEqual(result["metrics"], {"score_unbounded": 0.0})
            if mode == "eval":  # a final scored zero keeps beats_reference 0
                self.assertEqual(result["private_metrics"], {"beats_reference": 0})
            else:
                self.assertNotIn("private_metrics", result)

    def test_typed_zero_retains_reference_control_in_evidence(self):
        value, _ = self.evaluate(self.typed_failure(), observations(), role="final")
        with patch.object(self.verifier, "write_evidence") as write:
            self.verify(value, "eval")
        record = json.loads(write.call_args.args[1]["metrics.json"])
        execution = record["metrics"]["private_execution"]
        self.assertTrue(execution["reference"]["valid"])
        self.assertEqual(execution["reference"]["process_returncode"], 0)
        self.assertFalse(execution["candidate"]["valid"])
        self.assertEqual(record["metrics"]["candidate_rejection"], "terminal_reward")

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
                self.evaluator._install_instrumentation(
                    root, PurePosixPath("/workspace/sandbox/work"),
                    PurePosixPath("/workspace/sandbox/observer"), 1)
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
