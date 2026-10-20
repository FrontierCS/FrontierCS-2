"""Offline checks for request-anchored GPU leases and Population judge handoff."""

import deployment_test_guard  # noqa: F401  # isort: skip

import importlib.util
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "tasks/frontier-cs-2.0-demo/problems"
POPULATION = PROBLEMS / "Machine Learning/representation_frechet_loss"
ADAPTERS = [POPULATION / "verifier/fcs2_verifier_gpu.py"]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeadlineEnvelopeTests(unittest.TestCase):
    def test_all_published_adapters_inventory(self):
        self.assertEqual(len(list(PROBLEMS.rglob("verifier/fcs2_verifier_gpu.py"))), 15)
        for index, path in enumerate(ADAPTERS):
            adapter = load(path, f"deadline_adapter_{index}")
            with self.subTest(task=path.parent.parent.name):
                self.assertEqual(
                    adapter._runner_deadline(
                        {"granted_seconds": 600}, 7200, anchor=100
                    ),
                    699,
                )
                for invalid in (True, False, None, 0, -1, 7201, 1.5, "600"):
                    with self.assertRaises(adapter.VerifierGpuError):
                        adapter._runner_deadline(
                            {"granted_seconds": invalid}, 7200, anchor=100
                        )

    def test_provision_and_upload_spend_same_envelope(self):
        for index, path in enumerate(ADAPTERS):
            if path.parent.parent.name == "nano_world_model":
                continue  # Its existing phase-control suite covers this variant.
            adapter = load(path, f"spend_adapter_{index}")
            with (
                self.subTest(task=path.parent.parent.name),
                tempfile.TemporaryDirectory() as temp,
            ):
                inputs = Path(temp) / "inputs.tar"
                inputs.write_bytes(b"x")
                with mock.patch.object(adapter.time, "monotonic", return_value=300):
                    deadline = adapter._runner_deadline(
                        {"granted_seconds": 600}, 7200, anchor=100
                    )
                    self.assertEqual(adapter._remaining(deadline), 399)
                    with mock.patch.object(
                        adapter.subprocess,
                        "run",
                        return_value=SimpleNamespace(returncode=0, stderr=b""),
                    ) as child:
                        adapter._upload_inputs(
                            ["ssh"], inputs, 600, {}, runner_deadline=deadline
                        )
                        self.assertLessEqual(child.call_args.kwargs["timeout"], 120)
                with (
                    mock.patch.object(adapter.time, "monotonic", return_value=680),
                    mock.patch.object(
                        adapter.subprocess,
                        "run",
                        return_value=SimpleNamespace(returncode=0, stderr=b""),
                    ) as child,
                ):
                    adapter._upload_inputs(
                        ["ssh"], inputs, 600, {}, runner_deadline=deadline
                    )
                    self.assertLessEqual(child.call_args.kwargs["timeout"], 19)
                with (
                    mock.patch.object(adapter.time, "monotonic", return_value=700),
                    mock.patch.object(adapter.subprocess, "Popen") as child,
                ):
                    with self.assertRaises(adapter.VerifierGpuError):
                        adapter._run_remote_verifier(
                            ["ssh"],
                            "work",
                            Path(temp) / "output",
                            600,
                            {},
                            runner_deadline=deadline,
                        )
                    child.assert_not_called()

    def test_invalid_or_consumed_grant_still_cleans_owned_runner(self):
        for index, path in enumerate(ADAPTERS):
            if path.parent.parent.name == "nano_world_model":
                continue
            adapter = load(path, f"cleanup_adapter_{index}")
            for grant, elapsed in ((True, 0), (100, 200)):
                runner = {"runner_id": "owned", "granted_seconds": grant}
                with (
                    self.subTest(task=path.parent.parent.name, grant=grant),
                    tempfile.TemporaryDirectory() as temp,
                    mock.patch.dict(
                        os.environ, {"FRONTIER_MAX_EVIDENCE_BYTES": "8192"}
                    ),
                    mock.patch.object(
                        adapter,
                        "_config",
                        return_value=("url", "verification", "token", "gpu", 7200, 1),
                    ),
                    mock.patch.object(adapter, "_input_archive"),
                    mock.patch.object(adapter, "_create_runner", return_value=runner),
                    mock.patch.object(
                        adapter.time, "monotonic", side_effect=[100, 100 + elapsed]
                    ),
                    mock.patch.object(
                        adapter, "_terminate_runner", return_value=True
                    ) as delete,
                    mock.patch.object(adapter, "_upload_inputs") as upload,
                ):
                    with self.assertRaises(adapter.VerifierGpuError) as caught:
                        adapter.run(
                            payload=Path(temp) / "payload",
                            dataset=Path(temp) / "dataset",
                            mode="eval",
                            result_path=Path(temp) / "result",
                            evidence_path=Path(temp) / "evidence",
                        )
                    self.assertIn(
                        "invalid GPU granted duration"
                        if grant is True
                        else "deadline is exhausted",
                        str(caught.exception),
                    )
                    delete.assert_called_once_with(
                        "url", "verification", "token", "owned"
                    )
                    upload.assert_not_called()

    def test_population_remote_duration_handoff(self):
        adapter = load(POPULATION / "verifier/fcs2_verifier_gpu.py", "handoff_adapter")
        command = adapter._remote_verifier_command("eval", 1, remaining_seconds=123.456)
        self.assertIn("FRONTIER_VERIFIER_REMAINING_SECONDS=123.456000", command)
        self.assertIn("/opt/frontier/verify.py", command)
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(adapter.VerifierGpuError):
                adapter._remote_verifier_command("eval", remaining_seconds=invalid)

    def test_population_gpu_entrypoint_starts_one_local_clock(self):
        def verify(*args):
            self.assertEqual(os.environ["FRONTIER_VERIFIER_LOCAL_DEADLINE"], "250.0")
            return 0

        with (
            mock.patch.dict(os.environ, {"FRONTIER_VERIFIER_REMAINING_SECONDS": "150"}),
            mock.patch.dict(
                sys.modules, {"fcs2_verifier": SimpleNamespace(main=verify)}
            ),
            mock.patch("time.monotonic", return_value=100),
            mock.patch.object(sys, "path", list(sys.path)),
            self.assertRaises(SystemExit) as exit_result,
        ):
            runpy.run_path(str(POPULATION / "verifier/verify.py"), run_name="__main__")
        self.assertEqual(exit_result.exception.code, 0)

    def test_population_child_preserves_existing_reserve(self):
        runtime = load(POPULATION / "judge/judge_deps/runtime.py", "population_runtime")
        with mock.patch.object(runtime.time, "monotonic", return_value=100):
            self.assertEqual(runtime.ChildRunner(7110).deadline, 7200)
            self.assertEqual(
                runtime.ChildRunner(7110, envelope_deadline=1000).deadline, 900
            )
            self.assertEqual(
                runtime.ChildRunner(7110, envelope_deadline=10000).deadline, 7200
            )
            with self.assertRaises(TimeoutError):
                runtime.ChildRunner(7110, envelope_deadline=200)
            with self.assertRaises(ValueError):
                runtime.ChildRunner(7110, envelope_deadline=float("nan"))
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(runtime.time, "monotonic", return_value=100):
                runner = runtime.ChildRunner(7110, envelope_deadline=1000)
            with (
                mock.patch.object(runtime.time, "monotonic", return_value=901),
                mock.patch.object(runtime.subprocess, "Popen") as process,
            ):
                with self.assertRaises(runtime.SetupDeadlineExpired):
                    runner.run(
                        ["worker"], cwd=Path(temp), env={}, log=Path(temp) / "log"
                    )
                process.assert_not_called()
                runner.started = True
                with self.assertRaises(TimeoutError) as expired:
                    runner.run(
                        ["worker"], cwd=Path(temp), env={}, log=Path(temp) / "log"
                    )
                self.assertNotIsInstance(
                    expired.exception, runtime.SetupDeadlineExpired
                )

    def test_population_evaluator_debits_setup_and_forwards_deadline(self):
        judge = POPULATION / "judge"
        with (
            mock.patch.object(sys, "path", [str(judge), *sys.path]),
            mock.patch.dict(
                sys.modules,
                {
                    "yaml": SimpleNamespace(
                        safe_load=lambda _: {"runtime": {"timeout_seconds": 7200}}
                    )
                },
            ),
        ):
            evaluator = load(judge / "evaluator.py", "population_evaluator")
        for end, expected in ((1000, 840), (10000, 7140), (150, None), (180, None)):
            with (
                self.subTest(end=end),
                tempfile.TemporaryDirectory() as temp,
                mock.patch.dict(
                    os.environ, {"FRONTIER_VERIFIER_LOCAL_DEADLINE": str(end)}
                ),
                mock.patch.object(evaluator.time, "monotonic", return_value=100),
                mock.patch.object(evaluator, "OFFLINE_TARGETS", ()),
                mock.patch.object(evaluator, "_sha", return_value="same"),
                mock.patch.object(
                    evaluator,
                    "read_private_evidence",
                    return_value={"schema_version": 1, "events": [{"event": "scientific_started"}]},
                ),
                mock.patch.object(
                    evaluator,
                    "apply_submission_patch",
                    return_value=SimpleNamespace(ok=True, workdir=Path(temp) / "repo"),
                ),
                mock.patch.object(
                    evaluator.subprocess,
                    "run",
                    side_effect=subprocess.TimeoutExpired("workload", expected or 0),
                ) as process,
            ):
                if expected is None:
                    with self.assertRaises(evaluator.VerifierSetupDeadlineExpired):
                        evaluator._evaluate("solution", Path(temp) / "evidence")
                    process.assert_not_called()
                else:
                    with self.assertRaises(evaluator.VerifierScientificFailure) as failure:
                        evaluator._evaluate("solution", Path(temp) / "evidence")
                    self.assertEqual(failure.exception.verifier_failure_code, "verifier_suite_timeout")
                    self.assertEqual(process.call_args.kwargs["timeout"], expected)
                    self.assertEqual(
                        process.call_args.kwargs["env"][
                            "FRONTIER_VERIFIER_LOCAL_DEADLINE"
                        ],
                        str(end),
                    )

    def test_trusted_reference_failures_are_unscored_and_candidate_invalidity_is_zero(self):
        judge = POPULATION / "judge"
        with mock.patch.object(sys, "path", [str(judge), *sys.path]):
            evaluator = load(judge / "evaluator.py", "reference_failure_evaluator")
        for side, error, expected in (
            ("reference", "TimeoutExpired", "verifier_reference_failed"),
            ("reference", "RuntimeError", "verifier_reference_failed"),
            ("reference", "ValueError", "verifier_reference_failed"),
            ("candidate", "TimeoutExpired", "verifier_suite_timeout"),
            ("candidate", "TimeoutError", "verifier_suite_timeout"),
            ("candidate", "SuiteDeadlineExpired", "verifier_suite_timeout"),
            ("candidate", "RuntimeError", None),
            ("candidate", "ValueError", None),
        ):
            evidence = {"schema_version": 1, "events": [{
                "event": "phase_failed", "side": side, "error_type": error}]}
            with self.subTest(side=side, error=error):
                if expected:
                    with self.assertRaises(evaluator.VerifierScientificFailure) as failure:
                        evaluator._raise_runtime_failure(evidence)
                    self.assertEqual(failure.exception.verifier_failure_code, expected)
                    self.assertEqual(failure.exception.private_execution_evidence, evidence)
                else:
                    self.assertIsNone(evaluator._raise_runtime_failure(evidence))
        with self.assertRaises(evaluator.VerifierScientificFailure):
            evaluator._raise_runtime_failure(None, parent_timeout=True)

    def test_failed_child_replay_respects_trusted_side_and_error(self):
        judge = POPULATION / "judge"
        with mock.patch.object(sys, "path", [str(judge), *sys.path]):
            evaluator = load(judge / "evaluator.py", "child_failure_replay_evaluator")
        for side, kind, expected in (
            ("candidate", "ValueError", None),
            ("reference", "ValueError", "verifier_reference_failed"),
            ("candidate", "SuiteDeadlineExpired", "verifier_suite_timeout"),
        ):
            with (
                self.subTest(side=side, kind=kind),
                tempfile.TemporaryDirectory() as temp,
                mock.patch.object(evaluator, "OFFLINE_TARGETS", ()),
                mock.patch.object(evaluator, "_sha", return_value="same"),
                mock.patch.object(evaluator, "apply_submission_patch",
                    return_value=SimpleNamespace(ok=True, workdir=Path(temp) / "repo")),
                mock.patch.object(evaluator, "read_private_evidence", return_value={
                    "schema_version": 1, "events": [{"event": "phase_failed",
                        "side": side, "error_type": kind}]}),
                mock.patch.object(evaluator.subprocess, "run", return_value=SimpleNamespace(
                    returncode=1, stdout="", stderr="candidate-controlled text")),
            ):
                if expected:
                    with self.assertRaises(evaluator.VerifierScientificFailure) as failure:
                        evaluator._evaluate("solution", Path(temp) / "evidence")
                    self.assertEqual(failure.exception.verifier_failure_code, expected)
                else:
                    result = evaluator._evaluate("solution", Path(temp) / "evidence")
                    self.assertEqual(result[0], 0)
                    self.assertEqual(result[2], "GPU workload failed")

    def test_missing_trusted_journal_never_manufactures_zero(self):
        judge = POPULATION / "judge"
        with mock.patch.object(sys, "path", [str(judge), *sys.path]):
            evaluator = load(judge / "evaluator.py", "missing_journal_evaluator")
        for evidence in (None, {"schema_version": 1, "events": []}):
            with self.assertRaises(evaluator.VerifierScientificFailure) as failure:
                evaluator._raise_runtime_failure(evidence)
            self.assertEqual(failure.exception.verifier_failure_code,
                             "verifier_execution_evidence_invalid")
            with self.assertRaises(evaluator.VerifierScientificFailure) as failure:
                evaluator._raise_runtime_failure(evidence, parent_timeout=True)
            self.assertEqual(failure.exception.verifier_failure_code, "verifier_suite_timeout")
        with mock.patch.object(evaluator, "read_private_evidence", side_effect=ValueError("bad journal")):
            with self.assertRaises(evaluator.VerifierScientificFailure) as failure:
                evaluator._runtime_evidence("journal")
            self.assertEqual(failure.exception.verifier_failure_code,
                             "verifier_execution_evidence_invalid")

    def test_failure_code_boundary_is_closed(self):
        with mock.patch.object(sys, "path", [str(POPULATION / "verifier"), *sys.path]):
            adapter = load(POPULATION / "verifier/fcs2_verifier.py", "failure_boundary_adapter")
        class Failure(RuntimeError):
            private_execution_evidence = None
        for code in ("verifier_reference_failed", "verifier_suite_timeout",
                     "verifier_execution_evidence_invalid", "solver_selected_code"):
            failure = Failure("trusted failure")
            failure.verifier_failure_code = code
            with self.subTest(code=code):
                with self.assertRaises(adapter.VerifierFailure) as raised:
                    adapter.run_evaluator(mock.Mock(side_effect=failure), Path("submission"), "eval")
                self.assertEqual(raised.exception.code,
                                 code if code != "solver_selected_code" else "evaluator_error")

    def test_final_budget_changes_only_final_and_total_verifier_capacity(self):
        import yaml
        config = yaml.safe_load((POPULATION / "judge/config.yaml").read_text())
        policy = json.loads((POPULATION / "policy.json").read_text())
        self.assertEqual(config["runtime"]["timeout_seconds"], 7200)
        self.assertEqual(config["runtime"]["final_timeout_seconds"], 10800)
        self.assertEqual(policy["quota"]["eval_timeout_seconds"], 10800)
        self.assertEqual(policy["quota"]["wall_time_seconds"], 18000)
        self.assertEqual(policy["quota"]["max_train_calls"], 3)
        self.assertEqual(policy["quota"]["max_train_wall_seconds"], 21600)
        self.assertEqual(policy["gpu"]["agent"]["max_gpu_seconds"], 7200)
        self.assertEqual(policy["gpu"]["verifier"]["max_gpu_seconds"], 32400)

    def test_parent_timeout_before_scientific_launch_is_unscored(self):
        judge = POPULATION / "judge"
        with (
            mock.patch.object(sys, "path", [str(judge), *sys.path]),
            mock.patch.dict(
                sys.modules,
                {
                    "yaml": SimpleNamespace(
                        safe_load=lambda _: {"runtime": {"timeout_seconds": 7200}}
                    )
                },
            ),
        ):
            evaluator = load(judge / "evaluator.py", "parent_timeout_evaluator")
        for events in (
            None,
            {"events": [{"event": "suite_started"}, {"event": "phase_started"}]},
        ):
            with (
                tempfile.TemporaryDirectory() as temp,
                mock.patch.dict(
                    os.environ, {"FRONTIER_VERIFIER_LOCAL_DEADLINE": "1000"}
                ),
                mock.patch.object(evaluator.time, "monotonic", return_value=100),
                mock.patch.object(evaluator, "OFFLINE_TARGETS", ()),
                mock.patch.object(evaluator, "_sha", return_value="same"),
                mock.patch.object(
                    evaluator,
                    "apply_submission_patch",
                    return_value=SimpleNamespace(ok=True, workdir=Path(temp) / "repo"),
                ),
                mock.patch.object(
                    evaluator, "read_private_evidence", return_value=events
                ),
                mock.patch.object(
                    evaluator.subprocess,
                    "run",
                    side_effect=subprocess.TimeoutExpired("prepare_assets", 840),
                ),
            ):
                with self.assertRaises(evaluator.VerifierSetupDeadlineExpired):
                    evaluator._evaluate("solution", Path(temp) / "evidence")

    def test_setup_child_timeout_does_not_mark_scientific_start(self):
        runtime = load(POPULATION / "judge/judge_deps/runtime.py", "setup_runtime")
        for scientific in (False, True):
            marker = mock.Mock()
            with (
                tempfile.TemporaryDirectory() as temp,
                mock.patch.object(runtime.time, "monotonic", return_value=100),
                mock.patch.object(runtime.subprocess, "Popen") as process,
            ):
                runner = runtime.ChildRunner(
                    7110, envelope_deadline=1000, on_scientific_start=marker
                )
                process.return_value.wait.side_effect = subprocess.TimeoutExpired(
                    "child", 800
                )
                with mock.patch.object(runner, "stop"):
                    with self.assertRaises(
                        (TimeoutError, subprocess.TimeoutExpired)
                    ) as error:
                        runner.run(
                            ["child"],
                            cwd=Path(temp),
                            env={},
                            log=Path(temp) / "log",
                            scientific=scientific,
                        )
                self.assertEqual(
                    isinstance(error.exception, runtime.SetupDeadlineExpired),
                    not scientific,
                )
                self.assertEqual(marker.call_count, int(scientific))

    def test_scientific_marker_failure_stops_spawned_child(self):
        runtime = load(POPULATION / "judge/judge_deps/runtime.py", "marker_runtime")
        for fail_at in ("spawn", "marker"):
            with (
                tempfile.TemporaryDirectory() as temp,
                mock.patch.object(runtime.time, "monotonic", return_value=100),
                mock.patch.object(runtime.subprocess, "Popen") as process,
            ):
                marker = mock.Mock(side_effect=ValueError("journal unavailable"))
                runner = runtime.ChildRunner(
                    7110, envelope_deadline=1000, on_scientific_start=marker
                )
                if fail_at == "spawn":
                    process.side_effect = OSError("launch failed")
                with mock.patch.object(runner, "stop") as stop:
                    with self.assertRaises((OSError, ValueError)):
                        runner.run(
                            ["science"],
                            cwd=Path(temp),
                            env={},
                            log=Path(temp) / "log",
                            scientific=True,
                        )
                    self.assertFalse(runner.started)
                    self.assertEqual(marker.call_count, int(fail_at == "marker"))
                    self.assertEqual(stop.call_count, int(fail_at == "marker"))

    def test_added_scientific_marker_fits_trusted_journal(self):
        recipe = json.loads(
            (POPULATION / "judge/judge_deps/workloads.json").read_text()
        )
        for workloads in recipe["suites"].values():
            # Suite start/end, two sides x two phases x start/end, one new marker.
            self.assertLessEqual(2 + len(workloads) * 8 + 1, 32)

    def test_setup_timeout_verifier_result_has_no_score(self):
        with mock.patch.object(sys, "path", [str(POPULATION / "verifier"), *sys.path]):
            verifier = load(
                POPULATION / "verifier/fcs2_verifier.py", "population_verifier"
            )

        class SetupExpired(RuntimeError):
            verifier_failure_code = "verifier_setup_timeout"

        def evaluate(_):
            raise SetupExpired("setup spent the envelope")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload, dataset = root / "payload", root / "dataset"
            payload.write_bytes(b"payload")
            dataset.write_bytes(b"dataset")
            with (
                mock.patch.object(
                    verifier, "load_candidate_diagnostic", return_value=None
                ),
                mock.patch.object(verifier, "extract_payload"),
                mock.patch.object(verifier, "entrypoint_path", return_value=payload),
                mock.patch.object(verifier, "stage_task", return_value=root),
                mock.patch.object(verifier, "load_evaluator", return_value=evaluate),
                mock.patch.object(verifier, "write_evidence"),
                mock.patch.dict(os.environ, dict(os.environ)),
            ):
                result = verifier.verify(
                    bundle=root,
                    entrypoint="payload/solution.patch",
                    payload=payload,
                    dataset=dataset,
                    mode="eval",
                    result_path=root / "result.json",
                    evidence_path=root / "evidence.tar.zst",
                )
            written = json.loads((root / "result.json").read_text())
            self.assertEqual(result, written)
            self.assertEqual(written["status"], "failed")
            self.assertEqual(written["failure"]["code"], "verifier_setup_timeout")
            self.assertNotIn("score", written)
            self.assertNotIn("score_unbounded", written)

    def test_worker_startup_exhaustion_propagates_as_infrastructure(self):
        judge = POPULATION / "judge"
        with (
            mock.patch.object(sys, "path", [str(judge), *sys.path]),
            mock.patch.dict(
                sys.modules,
                {
                    "yaml": SimpleNamespace(
                        safe_load=lambda _: {"runtime": {"timeout_seconds": 7200}}
                    )
                },
            ),
        ):
            evaluator = load(judge / "evaluator.py", "startup_evaluator")
        with mock.patch.object(sys, "path", [str(POPULATION / "verifier"), *sys.path]):
            verifier = load(
                POPULATION / "verifier/fcs2_verifier.py", "startup_verifier"
            )
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.dict(os.environ, {"FRONTIER_VERIFIER_LOCAL_DEADLINE": "250"}),
            mock.patch.object(evaluator.time, "monotonic", return_value=100),
            mock.patch.object(evaluator, "OFFLINE_TARGETS", ()),
            mock.patch.object(evaluator, "_sha", return_value="same"),
            mock.patch.object(
                evaluator,
                "apply_submission_patch",
                return_value=SimpleNamespace(ok=True, workdir=Path(temp) / "repo"),
            ),
            mock.patch.object(
                evaluator.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=75,
                    stdout='{"verifier_failure": "verifier_setup_timeout"}',
                    stderr="",
                ),
            ),
        ):
            with self.assertRaises(verifier.VerifierFailure) as failure:
                verifier.run_evaluator(
                    evaluator.evaluate, Path(temp) / "solution", "eval"
                )
            self.assertEqual(failure.exception.code, "verifier_setup_timeout")
            for returncode, stdout, stderr in (
                (1, '{"verifier_failure": "verifier_setup_timeout"}', ""),
                (75, "", '{"verifier_failure": "verifier_setup_timeout"}'),
                (75, "candidate log: verifier_setup_timeout", ""),
            ):
                with (
                    mock.patch.object(evaluator, "read_private_evidence", return_value={
                        "schema_version": 1, "events": [{"event": "phase_failed",
                            "side": "candidate", "error_type": "ValueError"}]}),
                    mock.patch.object(
                        evaluator.subprocess, "run", return_value=SimpleNamespace(
                            returncode=returncode, stdout=stdout, stderr=stderr)),
                ):
                    scored = evaluator._evaluate("solution", Path(temp) / "evidence")
                    self.assertEqual(scored[0], 0)


if __name__ == "__main__":
    unittest.main()
