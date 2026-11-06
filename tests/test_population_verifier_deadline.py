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
# Population now runs on the shared GPU adapter (gpu-tools v4), which has its own tests; the
# legacy-adapter inventory below still covers the tasks that keep their own copy.
ADAPTERS = []
sys.path.insert(0, str(ROOT / "scripts"))
import task_manifest  # noqa: E402


def population_evaluator(name):
    """The judge as the Verifier stages it (generated config.yaml, pristine/ beside it)."""
    judge = task_manifest.judge_bundle(POPULATION)
    with mock.patch.object(sys, "path", [str(judge), *sys.path]):
        return load(judge / "evaluator.py", name)


def failed(side, error, phase="train"):
    return {"schema_version": 1, "events": [
        {"event": "scientific_started"},
        {"event": "phase_started", "side": side, "phase": phase},
        {"event": "phase_failed", "side": side, "phase": phase, "error_type": error}]}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeadlineEnvelopeTests(unittest.TestCase):
    def test_all_published_adapters_inventory(self):
        # Every forked GPU adapter still in the tree belongs to a policy.json task; a task.toml
        # task uses the shared adapter (gpu-tools) and must not carry one.
        forked = sorted(PROBLEMS.rglob("verifier/fcs2_verifier_gpu.py"))
        self.assertTrue(forked)
        for path in forked:
            self.assertFalse((path.parent.parent / "task.toml").is_file(), path)
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


    def test_population_evaluator_debits_setup_and_forwards_deadline(self):
        evaluator = population_evaluator("population_evaluator")
        for end, expected in ((1000, 840), (10000, 7140), (150, None), (180, None)):
            with (
                self.subTest(end=end),
                tempfile.TemporaryDirectory() as temp,
                mock.patch.dict(os.environ, {"FRONTIER_VERIFIER_LOCAL_DEADLINE": str(end)}),
                mock.patch.object(evaluator.time, "monotonic", return_value=100),
                mock.patch.object(evaluator, "OFFLINE_TARGETS", ()),
                mock.patch.object(evaluator, "read_private_evidence",
                                  return_value=failed("candidate", "TimeoutExpired")),
                mock.patch.object(evaluator.subprocess, "run",
                                  side_effect=subprocess.TimeoutExpired("workload", expected or 0)) as process,
            ):
                if expected is None:
                    with self.assertRaises(evaluator.VerifierSetupDeadlineExpired):
                        evaluator._evaluate(temp, Path(temp) / "evidence")
                    process.assert_not_called()
                else:
                    result = evaluator._evaluate(temp, Path(temp) / "evidence")
                    self.assertEqual(result[:3], (0.0, 0.0,
                                     "the submission's train exceeded the suite time limit"))
                    self.assertEqual(process.call_args.kwargs["timeout"], expected)
                    self.assertEqual(process.call_args.kwargs["env"]["FRONTIER_VERIFIER_LOCAL_DEADLINE"], str(end))

    def test_reference_failures_raise_and_candidate_failures_are_zero(self):
        evaluator = population_evaluator("reference_failure_evaluator")
        for side, error, expected in (
            ("reference", "TimeoutExpired", "verifier_reference_failed"),
            ("reference", "RuntimeError", "verifier_reference_failed"),
            ("candidate", "TimeoutExpired", "the submission's train exceeded the suite time limit"),
            ("candidate", "SuiteDeadlineExpired", "the submission's train exceeded the suite time limit"),
            ("candidate", "RuntimeError", "the submission's train failed"),
            ("candidate", "ValueError", "the submission's train failed"),
        ):
            evidence = failed(side, error)
            with self.subTest(side=side, error=error):
                if side == "reference":
                    with self.assertRaises(evaluator.VerifierScientificFailure) as failure:
                        evaluator._runtime_failure(evidence)
                    self.assertEqual(failure.exception.verifier_failure_code, expected)
                else:
                    self.assertEqual(evaluator._runtime_failure(evidence)[:3], (0.0, 0.0, expected))
        # A parent timeout while the candidate's phase is still running is the candidate's.
        running = {"schema_version": 1, "events": [{"event": "phase_started", "side": "candidate", "phase": "eval"}]}
        self.assertEqual(evaluator._runtime_failure(running, parent_timeout=True)[2],
                         "the submission's eval exceeded the suite time limit")

    def test_a_queue_worker_the_judge_could_not_start_is_unscored(self):
        # Round 5 (verification_01M6Q3TA5WT65GBGZS9EAF18KK): a restored queue's worker missed
        # READY after the candidate's training; the observer report names the trusted fault.
        evaluator = population_evaluator("queue_start_fault_evaluator")
        for side in ("candidate", "reference"):
            evidence = failed(side, "RuntimeError")
            evidence["events"][-1]["report"] = {
                "gpu": True, "queue_worker_returncodes": [-9, -9, -9],
                "trusted_fault": "QueueWorkerStartError: queue worker did not start: TimeoutError"}
            with self.subTest(side=side), self.assertRaises(evaluator.VerifierScientificFailure) as failure:
                evaluator._runtime_failure(evidence)
            self.assertEqual(failure.exception.verifier_failure_code, "verifier_queue_worker_failed")
        # Without a trusted fault in the report the candidate's failed phase stays a scored zero.
        evidence = failed("candidate", "RuntimeError")
        evidence["events"][-1]["report"] = {"gpu": True, "queue_worker_returncodes": [-9]}
        self.assertEqual(evaluator._runtime_failure(evidence)[:3], (0.0, 0.0, "the submission's train failed"))

    def test_failed_child_replay_respects_trusted_side_and_error(self):
        evaluator = population_evaluator("child_failure_replay_evaluator")
        for side, kind, expected in (
            ("candidate", "ValueError", "the submission's train failed"),
            ("reference", "ValueError", "verifier_reference_failed"),
            ("candidate", "SuiteDeadlineExpired", "the submission's train exceeded the suite time limit"),
        ):
            with (
                self.subTest(side=side, kind=kind),
                tempfile.TemporaryDirectory() as temp,
                mock.patch.dict(os.environ, {"FRONTIER_SUBMISSION_ROLE": "development"}),
                mock.patch.object(evaluator, "OFFLINE_TARGETS", ()),
                mock.patch.object(evaluator, "read_private_evidence", return_value=failed(side, kind)),
                mock.patch.object(evaluator.subprocess, "run", return_value=SimpleNamespace(
                    returncode=1, stdout="", stderr="candidate-controlled text")),
            ):
                if side == "reference":
                    with self.assertRaises(evaluator.VerifierScientificFailure) as failure:
                        evaluator._evaluate(temp, Path(temp) / "evidence")
                    self.assertEqual(failure.exception.verifier_failure_code, expected)
                else:
                    # Train metrics stay empty: the shared verifier admits only [feedback] there.
                    self.assertEqual(evaluator._evaluate(temp, Path(temp) / "evidence"),
                                     (0.0, 0.0, expected, {}))

    def test_missing_trusted_journal_never_manufactures_zero(self):
        evaluator = population_evaluator("missing_journal_evaluator")
        for evidence in (None, {"schema_version": 1, "events": []}):
            for parent_timeout in (False, True):
                with self.assertRaises(evaluator.VerifierScientificFailure) as failure:
                    evaluator._runtime_failure(evidence, parent_timeout=parent_timeout)
                self.assertEqual(failure.exception.verifier_failure_code,
                                 "verifier_execution_evidence_invalid")
        with mock.patch.object(evaluator, "read_private_evidence", side_effect=ValueError("bad journal")):
            with self.assertRaises(evaluator.VerifierScientificFailure) as failure:
                evaluator._runtime_evidence("journal")
            self.assertEqual(failure.exception.verifier_failure_code, "verifier_execution_evidence_invalid")

    def test_final_budget_changes_only_final_and_total_verifier_capacity(self):
        task = task_manifest.read_task(POPULATION)
        config = json.loads((task_manifest.judge_bundle(POPULATION) / "config.yaml").read_text())
        self.assertEqual(config["runtime"]["timeout_seconds"], 7200)
        self.assertEqual(config["runtime"]["final_timeout_seconds"], 10800)
        self.assertEqual(task["quota"]["eval_timeout_seconds"], 10800)
        self.assertEqual(task["quota"]["wall_time_seconds"], 18000)
        self.assertEqual(task["quota"]["max_train_calls"], 3)
        self.assertEqual(task["quota"]["max_train_wall_seconds"], 21600)
        self.assertEqual(task["gpu"]["agent"]["max_gpu_seconds"], 7200)
        self.assertEqual(task["gpu"]["verifier"]["max_gpu_seconds"], 32400)

    def test_parent_timeout_before_scientific_launch_is_unscored(self):
        evaluator = population_evaluator("parent_timeout_evaluator")
        for events in (None, {"events": [{"event": "suite_started"}, {"event": "phase_started"}]}):
            with (
                tempfile.TemporaryDirectory() as temp,
                mock.patch.dict(os.environ, {"FRONTIER_VERIFIER_LOCAL_DEADLINE": "1000"}),
                mock.patch.object(evaluator.time, "monotonic", return_value=100),
                mock.patch.object(evaluator, "OFFLINE_TARGETS", ()),
                mock.patch.object(evaluator, "read_private_evidence", return_value=events),
                mock.patch.object(evaluator.subprocess, "run",
                                  side_effect=subprocess.TimeoutExpired("prepare_assets", 840)),
            ):
                with self.assertRaises(evaluator.VerifierSetupDeadlineExpired):
                    evaluator._evaluate(temp, Path(temp) / "evidence")


if __name__ == "__main__":
    unittest.main()
