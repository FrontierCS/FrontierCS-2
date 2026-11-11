"""Trusted measurement failures block scoring; candidate runtime faults deny credit."""

import deployment_test_guard  # noqa: F401

import copy
import sys
import unittest

from vllm_task import TASK_ROOT  # noqa: E402

TASK = TASK_ROOT
sys.path[:0] = [str(TASK / "verifier"), str(TASK / "judge")]
import cpu_reference_verdict as cpu
import reference_verdict as original
import test_batch_measurements as projection_fixtures
from cpu_batch_measurements import project_complete_batch
from instance_manifest import expected_groups
from relay_client import remote_terminal_classification


class RuntimeGroupTests(unittest.TestCase):
    def observations(self, mode="train"):
        return {
            key: [original.InstanceObservation(ident, 10.0, True) for ident in ids]
            for key, ids in expected_groups(mode).items()
        }

    def test_normal_arithmetic_is_identical_for_both_phases(self):
        for mode in ("train", "eval"):
            vanilla = self.observations(mode)
            reference = self.observations(mode)
            candidate = {
                key: [
                    original.InstanceObservation(row.instance_id, 9.0, True)
                    for row in rows
                ]
                for key, rows in vanilla.items()
            }
            actual = cpu._evaluate_phase(vanilla, candidate, reference, mode=mode)
            actual.pop("runtime_errors")
            actual.pop("execution")
            self.assertEqual(
                actual,
                original._evaluate_phase(vanilla, candidate, reference, mode=mode),
            )

    def test_each_variant_service_fault_is_runtime_error_no_fake_timing(self):
        for mode in ("train", "eval"):
            for key, ids in expected_groups(mode).items():
                for variant in ("vanilla", "reference", "candidate"):
                    with self.subTest(mode=mode, group=key, variant=variant):
                        values = {
                            name: self.observations(mode)
                            for name in ("vanilla", "reference", "candidate")
                        }
                        values[variant][key] = cpu.RuntimeGroupFailure(
                            variant, key, tuple(ids)
                        )
                        if variant != "candidate":
                            with self.assertRaises(original.MeasurementError):
                                cpu._evaluate_phase(
                                    values["vanilla"],
                                    values["candidate"],
                                    values["reference"],
                                    mode=mode,
                                )
                            continue
                        result = cpu._evaluate_phase(
                            values["vanilla"],
                            values["candidate"],
                            values["reference"],
                            mode=mode,
                        )
                        from verify import result_projection

                        projection = result_projection(result, mode)
                        self.assertEqual(projection["raw_score"], result["score"])
                        if mode == "eval":
                            self.assertEqual(
                                projection["private_metrics"]["beats_reference"], 0
                            )
                        self.assertFalse(result["pass"])
                        self.assertEqual(result["execution"], "runtime_error")
                        workload = key.split("/")[0]
                        self.assertEqual(result["continuous"][workload]["score"], 0.0)
                        self.assertIsNone(
                            result["reference_verdict"]["groups"][key][
                                "reference_tails"
                            ]
                        )
                        other = "swe" if workload == "bfcl" else "bfcl"
                        weight = 0.8 if other == "swe" else 0.2
                        self.assertEqual(
                            result["score"],
                            weight * result["continuous"][other]["score"],
                        )

    def test_fault_marker_cannot_alias_a_different_group(self):
        values = self.observations()
        values["bfcl/public"] = cpu.RuntimeGroupFailure(
            "reference", "swe/public", tuple(expected_groups("train")["bfcl/public"])
        )
        with self.assertRaises(original.MeasurementError):
            cpu.evaluate_train(self.observations(), self.observations(), values)

    def test_bfcl_model_output_faults_and_normal_limit_all_variants(self):
        group, batch = projection_fixtures.MeasurementProjectionTests().make("bfcl")
        ident = batch["episodes"][0]["instance_id"]
        for variant in ("vanilla", "reference", "candidate"):
            for fault in ("api", "unknown", "normal_limit"):
                altered = copy.deepcopy(batch)
                row = altered["episodes"][0]
                row["episode"].update(output="")
                if fault == "normal_limit":
                    row["episode"].update(exit_status="force_quit", error="max_steps")
                else:
                    row["episode"].update(
                        exit_status="api_error" if fault == "api" else "unknown_error",
                        error="UnknownFailure",
                    )
                    row["requires_error_attribution"] = True
                    row["trace"] = [
                        {
                            "kind": "model",
                            "error_origin": "model_api_response"
                            if fault == "api"
                            else "model_response_decode",
                            "error_type": "UnknownFailure",
                        }
                    ]
                boundary, failed = remote_terminal_classification(
                    "batch", "bfcl/public", row
                )
                with self.subTest(variant=variant, fault=fault):
                    self.assertEqual(failed, fault != "normal_limit")
                    self.assertEqual(
                        boundary.kind,
                        "swe_case" if fault == "unknown" else "candidate_service",
                    )
                    if fault != "api":
                        rows = project_complete_batch(
                            altered,
                            group,
                            variant=variant,
                            model_case_failures=frozenset({ident})
                            if fault == "unknown"
                            else frozenset(),
                        )
                        self.assertFalse(rows[0].correct)
                        self.assertEqual(
                            rows[0].latency_seconds, row["episode"]["latency_seconds"]
                        )

    def test_missing_or_invalid_bfcl_time_is_group_error(self):
        group, batch = projection_fixtures.MeasurementProjectionTests().make("bfcl")
        for key in ("started_monotonic", "ended_monotonic", "latency_seconds"):
            for value in (None, float("nan"), True, -1):
                row = copy.deepcopy(batch["episodes"][0])
                row["episode"][key] = value
                boundary, failed = remote_terminal_classification(
                    "batch", "bfcl/public", row
                )
                self.assertTrue(failed)
                self.assertEqual(boundary.kind, "candidate_service")

    def test_real_bfcl_regression_still_invalidates_unaffected_workload(self):
        values = {
            name: self.observations("eval")
            for name in ("vanilla", "reference", "candidate")
        }
        ids = expected_groups("eval")["bfcl/public"]
        values["candidate"]["bfcl/public"] = cpu.RuntimeGroupFailure(
            "candidate", "bfcl/public", tuple(ids)
        )
        ids = expected_groups("eval")["bfcl/hidden"]
        values["candidate"]["bfcl/hidden"] = [
            original.InstanceObservation(ident, 10.0, False) for ident in ids
        ]
        result = cpu.evaluate_final(
            values["vanilla"], values["candidate"], values["reference"]
        )
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["pass"])

    def test_trusted_preflight_cannot_be_bypassed_by_candidate_runtime_error(self):
        from dataclasses import replace

        for mode in ("train", "eval"):
            for group, ids in expected_groups(mode).items():
                for role in ("vanilla", "reference"):
                    good = self.observations(mode)[group]
                    broken = [
                        None,
                        [],
                        [object()],
                        good[:-1],
                        [replace(row, latency_seconds=float("nan")) for row in good],
                        [
                            replace(row, execution="candidate_error", correct=False)
                            for row in good
                        ],
                        cpu.RuntimeGroupFailure(role, group, tuple(ids)),
                    ]
                    for rows in broken:
                        with self.subTest(
                            mode=mode, group=group, role=role, rows=type(rows).__name__
                        ):
                            values = {
                                name: self.observations(mode)
                                for name in ("vanilla", "reference", "candidate")
                            }
                            values["candidate"][group] = cpu.RuntimeGroupFailure(
                                "candidate", group, tuple(ids)
                            )
                            values[role][group] = rows
                            with self.assertRaises(original.MeasurementError):
                                cpu._evaluate_phase(
                                    values["vanilla"],
                                    values["candidate"],
                                    values["reference"],
                                    mode=mode,
                                )
                    values = {
                        name: self.observations(mode)
                        for name in ("vanilla", "reference", "candidate")
                    }
                    del values[role][group]
                    with self.assertRaises(original.MeasurementError):
                        cpu._evaluate_phase(
                            values["vanilla"],
                            values["candidate"],
                            values["reference"],
                            mode=mode,
                        )

    def test_candidate_runtime_marker_must_bind_candidate_role(self):
        for role in ("vanilla", "reference"):
            values = self.observations()
            group, ids = next(iter(expected_groups("train").items()))
            values[group] = cpu.RuntimeGroupFailure(role, group, tuple(ids))
            with self.assertRaises(original.MeasurementError):
                cpu.evaluate_train(self.observations(), values, self.observations())

    def test_ordinary_trusted_incorrect_measurements_still_score(self):
        from dataclasses import replace

        values = {
            name: self.observations("eval")
            for name in ("vanilla", "reference", "candidate")
        }
        for role in ("vanilla", "reference"):
            values[role]["swe/hidden"] = [
                replace(row, correct=False) for row in values[role]["swe/hidden"]
            ]
        result = cpu.evaluate_final(
            values["vanilla"], values["candidate"], values["reference"]
        )
        self.assertEqual(result["execution"], "complete")
        self.assertIsInstance(result["score"], float)

    def test_sanitized_sol_opus_48_slot_child_exit_never_gets_trusted_censored_time(
        self,
    ):
        from dataclasses import replace

        from cpu_batch_measurements import expected_input_digest

        group, batch = projection_fixtures.MeasurementProjectionTests().make("swe")
        group = replace(group, mode="eval", rate="hidden")
        ids = expected_groups("eval")["swe/hidden"]
        template = batch["episodes"][0]
        rows = []
        for ident in ids:
            row = copy.deepcopy(template)
            row["instance_id"] = row["episode"]["instance_id"] = ident
            rows.append(row)
        batch.update(
            identity=group.identity(),
            mode="eval",
            group="swe/hidden",
            expected_instances=48,
            episodes=rows,
            jps=__import__("batch_coordinator").FINAL_HIDDEN_JPS["swe"],
            input_digest=expected_input_digest(group),
        )
        # Same structural failure found in both archived Sol/Opus signed batches.
        # The injected fixture contains no model patch or private workload text.
        rows[0].update(
            episode=None,
            requires_error_attribution=True,
            execution_error={"origin": "supervised_episode", "type": "child_exit"},
        )
        for variant in ("vanilla", "reference"):
            with (
                self.subTest(variant=variant),
                self.assertRaises(cpu.TrustedMeasurementError) as caught,
            ):
                project_complete_batch(
                    batch,
                    group,
                    variant=variant,
                    swe_receipts={},
                    model_case_failures=frozenset(ids),
                )
            self.assertEqual(
                caught.exception.reason, "missing_or_invalid_timed_episode"
            )
            # Exercise the exact function bound into the live SWE phase, before
            # correctness requests skip CPU-prefailed slots or touch the relay.
            from types import SimpleNamespace

            import correctness_phase
            import phase_run

            self.assertIs(
                phase_run.run_swe_correctness, correctness_phase.run_swe_correctness
            )
            relay = SimpleNamespace(
                session=SimpleNamespace(runner_id=group.operation_id)
            )
            with self.assertRaises(cpu.TrustedMeasurementError) as actual:
                phase_run.run_swe_correctness(
                    batch,
                    group,
                    relay=relay,
                    variant=variant,
                    evidence="/unused-before-trusted-preflight",
                    request_bytes=1000000,
                    timeout=10,
                    model_case_failures=frozenset(ids),
                    model_case_bounds={ids[0]: object()},
                )
            self.assertEqual(
                actual.exception.reason, "missing_or_invalid_timed_episode"
            )

    def test_all_trusted_projection_validation_faults_are_typed(self):
        group, batch = projection_fixtures.MeasurementProjectionTests().make("bfcl")
        for variant in ("vanilla", "reference"):
            for mutation in (
                "missing_row",
                "wrong_identity",
                "missing_output",
                "invalid_time",
            ):
                altered = copy.deepcopy(batch)
                if mutation == "missing_row":
                    altered["episodes"].pop()
                elif mutation == "wrong_identity":
                    altered["identity"] = {}
                elif mutation == "missing_output":
                    altered["episodes"][0]["episode"].pop("output")
                else:
                    altered["episodes"][0]["episode"]["latency_seconds"] = float("nan")
                with (
                    self.subTest(variant=variant, mutation=mutation),
                    self.assertRaises(cpu.TrustedMeasurementError),
                ):
                    project_complete_batch(altered, group, variant=variant)

    def test_malformed_trusted_map_is_typed_unavailable(self):
        for bad in (None, [], object()):
            for role in ("vanilla", "reference"):
                values = {
                    name: self.observations()
                    for name in ("vanilla", "reference", "candidate")
                }
                values[role] = bad
                with (
                    self.subTest(role=role, bad=type(bad).__name__),
                    self.assertRaises(cpu.TrustedMeasurementError),
                ):
                    cpu.evaluate_train(
                        values["vanilla"], values["candidate"], values["reference"]
                    )
