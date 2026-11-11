"""Model-side C2 loss keeps original cases and CPU-provenanced watchdog bounds."""

import deployment_test_guard  # noqa: F401

import copy
import json
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from cryptography.exceptions import InvalidSignature
from vllm_task import TASK_ROOT  # noqa: E402

TASK = TASK_ROOT
sys.path[:0] = [str(TASK / "judge"), str(TASK / "verifier")]
import test_batch_measurements as measurement_fixture
from batch_measurements import project_complete_batch
from candidate_projection import timed_case_bounds, timed_case_failures
from containment_policy import Boundary, child_failure_metadata
from controller_contract_build import emit
from instance_manifest import expected_groups
from phase_controller import PhaseControllerFactory
from reference_verdict import InstanceObservation, MeasurementError, evaluate_train
from serving_eval.accuracy import build_correctness_request
from sidecar_catalog import VARIANTS, check_name, instance_index
from terminal_contract import terminal


class ModelCaseTimingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "judge").mkdir()
        (self.root / "verifier").mkdir()
        (self.root / "judge/source.py").write_text("immutable CPU fixture\n")
        (self.root / "verifier/runtime-plan.json").write_bytes(
            (TASK / "verifier/runtime-plan.json").read_bytes()
        )
        emit(
            self.root,
            ["judge", "verifier"],
            [],
            self.root / "controller-contract.json",
            self.root / "controller-contract-limits.json",
        )
        group, self.batch = measurement_fixture.MeasurementProjectionTests().make("swe")
        self.group = replace(
            group,
            operation_id="parent",
            group_id="parent",
            placement_id="parent",
            model_source_digest="sha256:" + "a" * 64,
        )
        self.batch.update(
            identity=self.group.identity(),
            model_source_digest=self.group.model_source_digest,
        )
        self.ids = expected_groups("train")["swe/public"]
        environment = {
            "FRONTIER_RUN_ID": "run",
            "FRONTIER_ATTEMPT_ID": "attempt",
            "FRONTIER_VERIFICATION_ID": "verification",
            "FRONTIER_TASK_CONFIG_DIGEST": "sha256:" + "b" * 64,
            "FRONTIER_VERIFIER_IMAGE_DIGEST": "sha256:" + "c" * 64,
        }
        session = SimpleNamespace(
            runner_id="parent",
            mode="train",
            client=SimpleNamespace(verification_id="verification"),
        )
        self.factory = PhaseControllerFactory(
            session,
            {variant: self.group for variant in VARIANTS},
            source_root=self.root,
            environment=environment,
            maximum_terminal_bytes=100000,
            maximum_manifest_bytes=100000,
        )

    def case_failure(self, variant, *, cap=10):
        started = time.monotonic()
        controller = self.factory(
            "batch",
            variant,
            {},
            "sha256:" + "d" * 64,
            self.root / (variant + "-" + str(len(self.factory.controllers))),
            started=started,
            deadline=started + cap,
        )
        # Allocation exists before any model input/dispatch observation.
        self.assertTrue((controller.evidence / "timing-allocation.json").is_file())
        self.assertFalse(
            any(controller.ledger.dispatched(slot) for slot in controller.slots)
        )
        for index, slot in enumerate(controller.slots):
            controller.mark_dispatched(slot)

            def observed(_, slot=slot, index=index):
                return terminal(
                    slot,
                    status="runtime_error" if index == 0 else "complete",
                    cause="child_exit" if index == 0 else "normal_completion",
                    elapsed=1,
                    cap=cap,
                    result=None if index == 0 else self.batch["episodes"][index],
                    result_limit_bytes=None if index == 0 else 100000,
                    result_limit_depth=None if index == 0 else 128,
                )

            controller.observe(
                slot,
                b"",
                verify=observed,
                boundary=Boundary("swe_case", "tool"),
                prefix_bytes=2048,
            )
        failed = timed_case_failures(self.factory, self.group, variant=variant)
        bounds = timed_case_bounds(self.factory, self.group, variant=variant)
        self.assertEqual(failed, frozenset(self.ids[:1]))
        return controller, bounds[self.ids[0]]

    def project(
        self,
        variant,
        bound,
        *,
        known_elapsed=False,
        spoof_elapsed=False,
        supervised=None,
        projector=project_complete_batch,
    ):
        batch = copy.deepcopy(self.batch)
        if not known_elapsed and not spoof_elapsed:
            batch["episodes"][0].update(
                episode=None,
                execution_error="model killed its case",
                requires_error_attribution=True,
            )
        if supervised is not None:
            batch["episodes"][0]["supervised_terminal"] = supervised
        receipts = {}
        for ident in self.ids[1:]:
            request = build_correctness_request(
                ident,
                "",
                variant=variant,
                identity={
                    **self.group.identity(),
                    "instance_id": ident,
                    "phase": "train",
                    "child_id": check_name(instance_index(ident)),
                },
            )
            receipts[ident] = {
                "instance_id": ident,
                "request_sha256": request["request_sha256"],
                "execution": "candidate_patch_rejected",
                "candidate_patch_applied": False,
                "patch_apply_log": "unchanged bound receipt",
            }
        return projector(
            batch,
            self.group,
            swe_receipts=receipts,
            variant=variant,
            model_case_failures=frozenset(self.ids[:1]),
            model_case_bounds={} if known_elapsed else {self.ids[0]: bound},
        )

    def test_preserved_supervisor_terminal_uses_actual_elapsed_for_every_role(self):
        from cpu_batch_measurements import project_complete_batch as cpu_project

        for variant in VARIANTS:
            _, bound = self.case_failure(variant)
            record = terminal(
                bound.slot,
                status="runtime_error",
                cause="child_exit",
                elapsed=3.125,
                cap=10,
            )
            record["candidate_input_dispatched"] = True
            record["failure_context"] = child_failure_metadata(
                bound.slot, Boundary("swe_case", "execution"), True
            )
            for projector in (project_complete_batch, cpu_project):
                observed = self.project(
                    variant, bound, supervised=record, projector=projector
                )
                self.assertEqual(observed[0].latency_seconds, 3.125)
                self.assertFalse(observed[0].correct)
                self.assertIsNone(observed[0].timing_bound)
                self.assertEqual(len(observed), len(self.ids))
                for key, value in [
                    ("elapsed_seconds", 0),
                    ("candidate_input_dispatched", False),
                    (
                        "slot",
                        {**record["slot"], "request_digest": "sha256:" + "e" * 64},
                    ),
                ]:
                    malformed = {**record, key: value}
                    with self.subTest(
                        variant=variant, projector=projector.__module__, key=key
                    ):
                        with self.assertRaises((ValueError, MeasurementError)):
                            self.project(
                                variant,
                                bound,
                                supervised=malformed,
                                projector=projector,
                            )

    def test_final_48_slot_supervised_failure_correctness_join_and_scorer_all_roles(
        self,
    ):
        from batch_coordinator import PreparedTestbed
        from batch_measurements import expected_input_digest
        from correctness_phase import run_swe_correctness
        from cpu_batch_measurements import project_complete_batch as cpu_project
        from cpu_reference_verdict import evaluate_final

        self.factory.session.mode = "eval"
        ids = expected_groups("eval")["swe/public"]
        incident = json.loads(
            (
                Path(__file__).parent
                / "fixtures/vllm/qwen-final-hidden-reference-django13710.json"
            ).read_bytes()
        )
        failed_index = ids.index(incident["instance_id"])
        group = replace(
            self.group,
            mode="eval",
            testbeds=tuple(
                PreparedTestbed(
                    ident, "http://fixture:8003", "fixture-" + str(index), "eval"
                )
                for index, ident in enumerate(ids)
            ),
        )
        self.factory.groups = {variant: group for variant in VARIANTS}
        inputs = {
            variant: {
                key: [InstanceObservation(ident, 32, True) for ident in values]
                for key, values in expected_groups("eval").items()
            }
            for variant in VARIANTS
        }

        def case(variant):
            started = time.monotonic()
            controller = self.factory(
                "batch",
                variant,
                {},
                "sha256:" + "d" * 64,
                self.root / ("eval48-" + variant),
                started=started,
                deadline=started + 20,
            )
            records = []
            for index, slot in enumerate(controller.slots):
                controller.mark_dispatched(slot)
                if index == failed_index:
                    observed = terminal(
                        slot,
                        status="runtime_error",
                        cause="child_exit",
                        elapsed=1.25,
                        cap=20,
                    )
                    observed["candidate_input_dispatched"] = True
                    observed["failure_context"] = child_failure_metadata(
                        slot, Boundary("swe_case", "execution"), True
                    )
                    # Reuse the exact incident trace; elapsed is a new local
                    # fixture observation, never a repair of historical timing.
                    row = copy.deepcopy(incident)
                    row["supervised_terminal"] = observed
                else:
                    row = copy.deepcopy(self.batch["episodes"][1])
                    row["instance_id"] = slot.instance_id
                    row["episode"]["instance_id"] = slot.instance_id
                    observed = terminal(
                        slot,
                        status="complete",
                        cause="normal_completion",
                        elapsed=2,
                        cap=20,
                        result=row,
                        result_limit_bytes=100000,
                        result_limit_depth=128,
                    )
                controller.observe(
                    slot,
                    b"",
                    verify=lambda _, observed=observed: observed,
                    boundary=Boundary("swe_case", "execution"),
                    prefix_bytes=2048,
                )
                records.append(row)
            bounds = timed_case_bounds(self.factory, group, variant=variant)
            failed = timed_case_failures(self.factory, group, variant=variant)
            self.assertEqual(failed, frozenset([incident["instance_id"]]))
            batch = {
                **self.batch,
                "mode": "eval",
                "expected_instances": 48,
                "input_digest": expected_input_digest(group),
                "episodes": records,
            }
            receipts = {}

            def correctness(role, requests, **kwargs):
                self.assertEqual(set(requests), set(ids) - failed)
                parsed = {}
                for ident, request in requests.items():
                    receipts[ident] = {
                        "instance_id": ident,
                        "request_sha256": request["request_sha256"],
                        "execution": "candidate_patch_rejected",
                        "candidate_patch_applied": False,
                        "patch_apply_log": "fixture bounded incorrect patch",
                    }
                    parsed[ident] = kwargs["parse"](request, receipts[ident])
                return {
                    "receipts": receipts,
                    "summaries": {},
                    "parsed": parsed,
                    "model_case_failures": failed,
                }

            relay = SimpleNamespace(
                session=SimpleNamespace(runner_id="parent"),
                stream=True,
                correctness=correctness,
            )
            observations = run_swe_correctness(
                batch,
                group,
                relay=relay,
                variant=variant,
                evidence=self.root / ("eval48-correctness-" + variant),
                request_bytes=100000,
                timeout=10,
                model_case_failures=failed,
                model_case_bounds=bounds,
            )
            self.assertEqual(len(observations), 48)
            for projector in (project_complete_batch, cpu_project):
                self.assertEqual(
                    projector(
                        batch,
                        group,
                        swe_receipts=receipts,
                        variant=variant,
                        model_case_failures=failed,
                        model_case_bounds=bounds,
                    ),
                    observations,
                )
            self.assertEqual(
                [row.latency_seconds for row in observations],
                [1.25 if index == failed_index else 32 for index in range(48)],
            )
            self.assertTrue(
                all(
                    row.execution == "complete" and not row.correct
                    for row in observations
                )
            )
            inputs[variant]["swe/public"] = observations

        for variant in VARIANTS:
            case(variant)
        verdict = evaluate_final(
            inputs["vanilla"], inputs["candidate"], inputs["reference"]
        )
        self.assertIn("score", verdict)

    def test_all_variants_missing_time_use_explicit_cpu_bound_and_keep_healthy_siblings(
        self,
    ):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                controller, bound = self.case_failure(variant)
                observations = self.project(variant, bound)
                self.assertEqual(
                    tuple(row.instance_id for row in observations), self.ids
                )
                failed = observations[0]
                self.assertEqual(failed.execution, "model_case_failure")
                self.assertIsNone(failed.latency_seconds)
                self.assertFalse(failed.correct)
                self.assertEqual(failed.timing_bound.bound_seconds, 10)
                proof = failed.timing_bound.provenance()
                self.assertEqual(proof["kind"], "upper_bound")
                self.assertFalse(proof["measured"])
                self.assertEqual(proof["configured_limit_seconds"], 7000)
                self.assertEqual(
                    proof["config_pointer"], "relay.batch_timeout_seconds.train"
                )
                self.assertEqual(
                    proof["controller_context"]["task_config_digest"],
                    "sha256:" + "b" * 64,
                )
                self.assertEqual(proof["slot"]["variant"], variant)
                self.assertTrue(
                    all(
                        row.latency_seconds == 32 and row.execution == "complete"
                        for row in observations[1:]
                    )
                )
                spoofed = self.project(variant, bound, spoof_elapsed=True)[0]
                self.assertIsNone(spoofed.latency_seconds)
                self.assertEqual(spoofed.timing_bound.bound_seconds, 10)
                measured = self.project(variant, bound, known_elapsed=True)[0]
                self.assertEqual(measured.latency_seconds, 32)
                self.assertIsNone(measured.timing_bound)
                self.assertEqual(measured.execution, "complete")

    def test_final_allocation_uses_existing_eval_watchdog_and_all_48_original_cases(
        self,
    ):
        self.factory.session.mode = "eval"
        self.group = replace(self.group, mode="eval")
        self.factory.groups = {variant: self.group for variant in VARIANTS}
        started = time.monotonic()
        controller = self.factory(
            "batch",
            "reference",
            {},
            "sha256:" + "d" * 64,
            self.root / "eval-reference",
            started=started,
            deadline=started + 20,
        )
        self.assertEqual(
            tuple(slot.instance_id for slot in controller.slots),
            expected_groups("eval")["swe/public"],
        )
        self.assertEqual(len(controller.slots), 48)
        self.assertFalse(
            any(controller.ledger.dispatched(slot) for slot in controller.slots)
        )
        controller.fail_unresolved(boundary=Boundary("swe_case", "tool"))
        bounds = timed_case_bounds(self.factory, self.group, variant="reference")
        self.assertEqual(len(bounds), 48)
        for bound in bounds.values():
            proof = bound.provenance()
            self.assertEqual(proof["configured_limit_seconds"], 24000)
            self.assertEqual(
                proof["config_pointer"], "relay.batch_timeout_seconds.eval"
            )
            self.assertEqual(proof["bound_seconds"], 20)
            self.assertFalse(proof["measured"])

    def test_remote_bound_forgery_tampering_and_foreign_slot_cannot_supply_timing(self):
        _, bound = self.case_failure("candidate")
        for fake in (
            {"kind": "upper_bound", "bound_seconds": 1},
            replace(bound, slot=replace(bound.slot, variant="reference")),
        ):
            with (
                self.subTest(fake=type(fake).__name__),
                self.assertRaises((ValueError, MeasurementError)),
            ):
                self.project("candidate", fake)
        payload = bound.allocation.envelope["payload"]
        payload["owner_seconds"] = 1
        payload["deadline_monotonic"] = payload["started_monotonic"] + 1
        with self.assertRaises(InvalidSignature):
            self.project("candidate", bound)

    def test_cpu_source_change_or_unconfigured_owner_blocks_before_dispatch(self):
        raw = self.root / "verifier/runtime-plan.json"
        original = raw.read_bytes()
        raw.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "source"):
            self.case_failure("candidate")
        raw.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "existing configured watchdog"):
            self.case_failure("candidate", cap=7001)

    def test_actual_scorer_fixture_matches_existing_formula_with_explicit_substitution(
        self,
    ):
        baseline = {
            role: {
                group: [InstanceObservation(ident, latency, True) for ident in ids]
                for group, ids in expected_groups("train").items()
            }
            for role, latency in (("vanilla", 4), ("reference", 2), ("candidate", 1))
        }
        healthy = evaluate_train(
            baseline["vanilla"], baseline["candidate"], baseline["reference"]
        )
        self.assertTrue(healthy["pass"])
        self.assertEqual(healthy["continuous"]["swe"]["speedup"], 4)
        for variant in VARIANTS:
            _, bound = self.case_failure(variant)
            inputs, equivalent = copy.deepcopy(baseline), copy.deepcopy(baseline)
            ident = self.ids[0]
            inputs[variant]["swe/public"][0] = InstanceObservation(
                ident,
                None,
                False,
                execution="model_case_failure",
                source_status="cpu_observed_model_case_failure",
                timing_bound=bound,
            )
            equivalent[variant]["swe/public"][0] = InstanceObservation(ident, 10, False)
            actual = evaluate_train(
                inputs["vanilla"], inputs["candidate"], inputs["reference"]
            )
            expected = evaluate_train(
                equivalent["vanilla"], equivalent["candidate"], equivalent["reference"]
            )
            provenance = actual["reference_verdict"]["groups"]["swe/public"].pop(
                "censored_timing"
            )
            with self.subTest(variant=variant):
                self.assertEqual(actual, expected)
                self.assertEqual(actual["continuous"]["swe"]["instance_evaluations"], 5)
                self.assertTrue(
                    actual["reference_verdict"]["groups"]["swe/public"]["complete"]
                )
                self.assertEqual(set(provenance[variant]), {ident})
                self.assertEqual(provenance[variant][ident]["bound_seconds"], 10)
                self.assertFalse(provenance[variant][ident]["measured"])

    def test_c1_or_unclassified_platform_failure_never_gains_c2_timing(self):
        _, bound = self.case_failure("candidate")
        for variant in VARIANTS:
            inputs = {
                role: {
                    group: [InstanceObservation(ident, latency, True) for ident in ids]
                    for group, ids in expected_groups("train").items()
                }
                for role, latency in (
                    ("vanilla", 4),
                    ("reference", 2),
                    ("candidate", 1),
                )
            }
            inputs[variant]["swe/public"][0] = InstanceObservation(
                self.ids[0], None, False, execution="platform_error", timing_bound=bound
            )
            with self.subTest(variant=variant), self.assertRaises(MeasurementError):
                evaluate_train(
                    inputs["vanilla"], inputs["candidate"], inputs["reference"]
                )
