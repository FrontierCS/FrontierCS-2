"""Original CPU parser consumes bound full receipts before durable terminals."""

import deployment_test_guard  # noqa: F401

import hashlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from vllm_task import TASK_ROOT  # noqa: E402

TASK = TASK_ROOT
sys.path[:0] = [str(TASK / "judge"), str(TASK / "verifier")]
import attestation
from batch_coordinator import PreparedGroup
from controller_contract_build import emit
from correctness_stream import collect
from cpu_reference_verdict import TrustedMeasurementError
from cpu_swe_accuracy import (
    CorrectnessRuntimeError,
    TrustedCorrectnessError,
    build_correctness_request,
    parse_test_receipt,
)
from driver_client import RemoteExecutionError, SidecarExecutionError
from instance_manifest import expected_groups
from phase_controller import PhaseControllerFactory
from relay_client import SchedulerRelay, canonical_bytes
from scheduler_relay import CORRECTNESS_VIEW_SCHEMA
from sidecar_catalog import SCHEDULER, VARIANTS, check_name, instance_index
from test_attestation import keypair


class Session:
    runner_id, mode = "parent", "train"

    def __init__(self, signer, ids):
        self.signer, self.ids = signer, tuple(reversed(ids))
        self.client = SimpleNamespace(
            verification_id="verification",
            deadline=time.monotonic() + 30,
            remaining=lambda: 30,
        )
        self.limits, self.operations, self.calls = {"ssh_sessions": 900}, 0, []
        self.doc = None
        self.bad, self.drop_after, self.case_failure = None, None, None

    def run(self, argv, *, stdin, output, **kwargs):
        self.operations += 1
        self.calls.append((argv, stdin, kwargs))
        output = Path(output)
        output.mkdir(parents=True)
        if argv[3] == "job-submit":
            self.doc, self.digest = (
                json.loads(stdin),
                "sha256:" + hashlib.sha256(stdin).hexdigest(),
            )
            (output / "stdout").write_bytes(b"{}")
            (output / "stderr").write_bytes(b"")
            return output / "stdout"
        selector = json.loads(stdin)
        cursor = selector["cursor"]
        if self.drop_after is not None and cursor >= self.drop_after:
            raise RemoteExecutionError(
                "GPU relay died after completed sibling", cause="child_exit"
            )
        ident = self.ids[cursor if self.bad != "duplicate" else 0]
        request = self.doc["requests"].get(ident)
        body, summary = b"", None
        if request is not None:
            receipt = {
                "instance_id": ident,
                "request_sha256": request["request_sha256"],
                "execution": "candidate_patch_rejected",
                "candidate_patch_applied": False,
                "patch_apply_log": "original receipt grammar, no executed test claims",
            }
            if cursor == 0 and self.case_failure == "malformed_receipt":
                receipt.update(
                    testbed=None, stage_exit_codes=[], candidate_syntax_checks=None
                )
            if cursor == 0 and self.case_failure == "surrogate_patch_log":
                receipt["patch_apply_log"] = "\ud800"
            if cursor == 0 and self.case_failure in {"unknown_output", "surrogate_log"}:
                receipt.update(
                    execution="complete",
                    candidate_patch_applied=True,
                    test_patch_applied=True,
                    tests_started=True,
                    tests_completed=True,
                    log="\ud800"
                    if self.case_failure == "surrogate_log"
                    else "ImportError: model-generated unknown dependency or cycle\n",
                )
            if cursor == 0 and self.case_failure in {
                "setup_failed",
                "incomplete_test_stages",
            }:
                receipt["execution"] = self.case_failure
            body = json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode()
            summary = {
                "request_sha256": request["request_sha256"],
                "receipt_sha256": "sha256:" + hashlib.sha256(body).hexdigest(),
                "receipt_bytes": len(body),
                "reset": None
                if self.doc["variant"] == "vanilla"
                else {"head": "fixed"},
                "run_index": VARIANTS.index(self.doc["variant"]),
                "worker": check_name(instance_index(ident)),
            }
        if self.bad == "request":
            summary["request_sha256"] = "sha256:" + "f" * 64
        if cursor == 0 and self.case_failure == "killed":
            body, summary = b"", None
        record = {
            "instance_id": ident,
            "episode": summary,
            "trace": [],
            "correctness_cursor": cursor + (1 if self.bad == "cursor" else 0),
        }
        if request is None or (cursor == 0 and self.case_failure == "killed"):
            record["execution_error"] = {
                "origin": "timed_case_boundary",
                "type": "TimedCaseFailed",
            }
        payload = {
            "variant": self.doc["variant"],
            "event_index": cursor * 2 + 1,
            "event": "terminal",
            "record": record,
        }
        if self.bad in {"record_none", "record_list", "instance_list"}:
            payload["record"] = {
                "record_none": None,
                "record_list": [],
                "instance_list": {**record, "instance_id": []},
            }[self.bad]
        event = self.signer.sign(
            kind="correctness",
            sequence=self.doc["sequence"],
            request_sha256=self.digest,
            payload=payload,
        )
        if self.bad == "signature":
            signature = event["signature"]
            event["signature"] = ("A" if signature[0] != "A" else "B") + signature[1:]
        view = {
            "schema": CORRECTNESS_VIEW_SCHEMA,
            "sequence": self.doc["sequence"],
            "request_sha256": self.digest,
            "state": "complete",
            "cursor": cursor,
            "event": event,
        }
        (output / "stdout").write_bytes(body)
        (output / "stderr").write_bytes(canonical_bytes(view))
        return output / "stdout"


class CorrectnessStreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "judge").mkdir()
        (self.root / "judge/source.py").write_bytes(b"immutable source fixture")
        emit(
            self.root,
            ["judge"],
            [],
            self.root / "controller-contract.json",
            self.root / "controller-contract-limits.json",
        )
        private, public = keypair()
        identity = {
            "run_id": "run",
            "attempt_id": "attempt",
            "operation_id": "parent",
            "phase": "train",
            "name": SCHEDULER,
            "profile_digest": "sha256:" + "a" * 64,
        }
        self.ids = expected_groups("train")["swe/public"]
        self.session = Session(attestation.Signer(private, public, identity), self.ids)
        environment = {
            "FRONTIER_RUN_ID": "run",
            "FRONTIER_ATTEMPT_ID": "attempt",
            "FRONTIER_VERIFICATION_ID": "verification",
            "FRONTIER_TASK_CONFIG_DIGEST": "sha256:" + "b" * 64,
            "FRONTIER_VERIFIER_IMAGE_DIGEST": "sha256:" + "c" * 64,
        }
        self.group = PreparedGroup(
            "parent",
            "parent",
            "parent",
            "train",
            "swe",
            "public",
            "http://main:8000/v1",
            "sha256:" + "d" * 64,
        )
        self.factory = PhaseControllerFactory(
            self.session,
            {variant: self.group for variant in VARIANTS},
            source_root=self.root,
            environment=environment,
            maximum_terminal_bytes=100000,
            maximum_manifest_bytes=100000,
        )
        self.relay = SchedulerRelay(
            self.session,
            public_key=public,
            relay_limits={
                "job_bytes": 100000,
                "response_bytes": 100000,
                "receipt_bytes": 100000,
                "receipt_timeout_seconds": 5,
            },
            max_stderr_bytes=16000,
            evidence=self.root / "evidence",
            stream=True,
            controller_factory=self.factory,
            cleanup_seconds=1,
        )
        self.requests = {
            ident: build_correctness_request(
                ident,
                "",
                variant="candidate",
                identity={
                    **self.group.identity(),
                    "instance_id": ident,
                    "child_id": check_name(instance_index(ident)),
                    "phase": "train",
                },
            )
            for ident in self.ids
        }

    def invoke(self, parse=parse_test_receipt):
        return collect(
            self.relay,
            "candidate",
            self.requests,
            ids=self.ids,
            timeout=10,
            name="correctness",
            parse=parse,
        )

    def test_trusted_preflight_cannot_become_candidate_case_failure(self):
        from unittest.mock import patch

        def case(stage):
            original = self.session.run

            def run(argv, **kwargs):
                if (stage == "submit") == (argv[3] == "job-submit"):
                    raise SidecarExecutionError(
                        "trusted SSH budget exhausted before dispatch"
                    )
                return original(argv, **kwargs)

            with self.subTest(stage=stage), patch.object(self.session, "run", run):
                with self.assertRaisesRegex(
                    SidecarExecutionError, "trusted SSH budget"
                ):
                    collect(
                        self.relay,
                        "candidate",
                        self.requests,
                        ids=self.ids,
                        timeout=10,
                        name="preflight-" + stage,
                        parse=parse_test_receipt,
                    )
            controller = self.factory.controllers[-1]
            self.assertTrue(
                all(controller.ledger.read(slot) is None for slot in controller.slots)
            )

        for stage in ("submit", "receipt"):
            case(stage)

    def test_original_parse_precedes_terminal_and_preserves_completion_order(self):
        seen = []

        def parsed(request, receipt):
            controller = self.factory.controllers[0]
            slot = next(
                slot
                for slot in controller.slots
                if slot.instance_id == request["instance_id"]
            )
            self.assertIsNone(
                controller.ledger.read(slot), "not scientific before original parser"
            )
            seen.append(request["instance_id"])
            return parse_test_receipt(request, receipt)

        result = self.invoke(parsed)
        self.assertEqual(seen, list(reversed(self.ids)))
        self.assertEqual(set(result["receipts"]), set(self.ids))
        self.assertTrue(
            all(row["correct"] is False for row in result["parsed"].values())
        )
        controller = self.factory.controllers[0]
        self.assertEqual(controller.ledger.unresolved(), [])
        self.assertTrue(
            all(
                controller.ledger.read(slot)["status"] == "complete"
                for slot in controller.slots
            )
        )
        self.assertEqual(self.session.operations, 1 + len(self.ids))
        self.assertFalse(
            list((self.root / "evidence/correctness").glob("cursor*/stdout"))
        )
        self.assertTrue(
            all(
                call[2]["max_stdout_bytes"] == 100000
                and call[2]["max_stderr_bytes"] == 16000
                for call in self.session.calls
            )
        )

    def test_stream_death_retains_completed_siblings_and_bounds_total_ssh(self):
        self.session.drop_after = 2
        captured = self.invoke()
        self.assertEqual(set(captured["receipts"]), set(self.ids[-2:]))
        self.assertEqual(captured["model_case_failures"], frozenset(self.ids[:-2]))
        controller = self.factory.controllers[0]
        records = [controller.ledger.read(slot) for slot in controller.slots]
        self.assertEqual(sum(record["status"] == "complete" for record in records), 2)
        self.assertEqual(controller.ledger.unresolved(), [])
        self.assertLessEqual(self.session.operations, len(self.ids) + 3)
        failed_calls = [json.loads(call[1]) for call in self.session.calls[3:]]
        self.assertEqual(
            {selector["cursor"] for selector in failed_calls},
            set(range(2, len(self.ids))),
        )

    def assert_bad_binding(self, kind, completed):
        self.session.bad = kind
        captured = self.invoke()
        self.assertEqual(
            len(captured["model_case_failures"]), len(self.ids) - completed
        )
        controller = self.factory.controllers[0]
        self.assertEqual(
            sum(
                controller.ledger.read(slot)["status"] == "complete"
                for slot in controller.slots
            ),
            completed,
        )

    def test_wrong_cursor_never_becomes_a_measurement(self):
        self.assert_bad_binding("cursor", 0)

    def test_duplicate_instance_never_overwrites_the_first_outcome(self):
        self.assert_bad_binding("duplicate", 1)

    def test_foreign_request_never_becomes_a_measurement(self):
        self.assert_bad_binding("request", 0)

    def test_malformed_remote_records_and_invalid_signatures_close_owned_cases(self):
        for variant in VARIANTS:
            for bad in ("record_none", "record_list", "instance_list", "signature"):
                with self.subTest(variant=variant, malformed=bad):
                    self.session.bad = bad
                    captured = collect(
                        self.relay,
                        variant,
                        self.requests_for(variant),
                        ids=self.ids,
                        timeout=10,
                        name="remote-" + variant + "-" + bad,
                        parse=parse_test_receipt,
                    )
                    self.assertEqual(
                        captured["model_case_failures"], frozenset(self.ids)
                    )
                    self.assertEqual(
                        self.factory.controllers[-1].ledger.unresolved(), []
                    )

    def test_cpu_crypto_verification_failures_propagate_without_scientific_terminals(
        self,
    ):
        from unittest.mock import patch

        for variant in VARIANTS:

            def case(index, error, variant):

                def failed_verify(*args):
                    raise error

                # Run the real envelope/shape/signature path up to the trusted crypto call.
                public = SimpleNamespace(verify=failed_verify)
                with self.subTest(variant=variant, error=type(error).__name__):
                    with patch.object(
                        attestation, "load_ssh_public_key", return_value=public
                    ):
                        with self.assertRaises(type(error)):
                            collect(
                                self.relay,
                                variant,
                                self.requests_for(variant),
                                ids=self.ids,
                                timeout=10,
                                name=f"cpu-crypto-{variant}-{index}",
                                parse=parse_test_receipt,
                            )
                    controller = self.factory.controllers[-1]
                    self.assertEqual(len(controller.ledger.unresolved()), len(self.ids))

            for index, error in enumerate(
                (
                    OSError("CPU crypto device failed"),
                    MemoryError("CPU crypto allocation failed"),
                    RuntimeError("CPU crypto bug"),
                )
            ):
                case(index, error, variant)

    def test_cpu_metadata_bind_and_decode_faults_remain_blockers(self):
        from unittest.mock import patch

        import correctness_stream

        for variant in VARIANTS:
            for operation in ("bind", "decode"):
                for index, error in enumerate(
                    (
                        OSError("CPU metadata device failed"),
                        MemoryError("CPU metadata allocation failed"),
                        RuntimeError("CPU metadata bug"),
                    )
                ):
                    owner = attestation if operation == "bind" else correctness_stream
                    with self.subTest(
                        variant=variant, operation=operation, error=type(error).__name__
                    ):
                        with patch.object(owner, operation, side_effect=error):
                            with self.assertRaises(type(error)):
                                collect(
                                    self.relay,
                                    variant,
                                    self.requests_for(variant),
                                    ids=self.ids,
                                    timeout=10,
                                    name=f"cpu-{operation}-{variant}-{index}",
                                    parse=parse_test_receipt,
                                )
                        self.assertEqual(
                            len(self.factory.controllers[-1].ledger.unresolved()),
                            len(self.ids),
                        )

    def test_unknown_parser_error_is_not_a_trusted_platform_signal(self):
        def broken(*args):
            raise OSError("unknown error in model receipt parser")

        captured = self.invoke(broken)
        self.assertEqual(captured["model_case_failures"], frozenset(self.ids))
        self.assertEqual(self.factory.controllers[0].ledger.unresolved(), [])

    def test_failed_reference_case_preserves_completed_siblings(self):
        self.factory.groups["reference"] = self.group
        requests = {
            ident: build_correctness_request(
                ident,
                "",
                variant="reference",
                identity={
                    **self.group.identity(),
                    "instance_id": ident,
                    "child_id": check_name(instance_index(ident)),
                    "phase": "train",
                },
            )
            for ident in self.ids
        }
        self.session.drop_after = 2
        captured = collect(
            self.relay,
            "reference",
            requests,
            ids=self.ids,
            timeout=10,
            name="reference-correctness",
            parse=parse_test_receipt,
        )
        self.assertEqual(set(captured["receipts"]), set(self.ids[-2:]))
        self.assertEqual(captured["model_case_failures"], frozenset(self.ids[:-2]))
        self.assertEqual(self.factory.controllers[-1].ledger.unresolved(), [])

    def test_trusted_aggregate_loss_never_synthesizes_missing_case_results(self):
        from unittest.mock import patch

        for variant in VARIANTS:

            def case(fault, variant):
                with self.subTest(variant=variant, fault=fault):
                    self.session.bad = (
                        fault
                        if fault
                        in {
                            "cursor",
                            "request",
                            "signature",
                            "record_none",
                            "duplicate",
                        }
                        else None
                    )
                    self.session.drop_after = (
                        0
                        if fault == "zero_receipts"
                        else 2
                        if fault == "partial_loss"
                        else None
                    )
                    original = self.session.run

                    def run(argv, **kwargs):
                        if fault == "submit_loss" and argv[3] == "job-submit":
                            raise RemoteExecutionError(
                                "aggregate relay unavailable", cause="child_exit"
                            )
                        return original(argv, **kwargs)

                    with (
                        patch.object(self.session, "run", run),
                        self.assertRaises(TrustedMeasurementError) as observed,
                    ):
                        collect(
                            self.relay,
                            variant,
                            self.requests_for(variant),
                            ids=self.ids,
                            timeout=10,
                            name=f"trusted-{variant}-{fault}",
                            parse=parse_test_receipt,
                        )
                    error = observed.exception
                    self.assertEqual(
                        (error.role, error.group, error.reason),
                        (variant, "swe/public", "aggregate_correctness_unavailable"),
                    )
                    self.assertFalse(error.private_classification()["score_ready"])
                    controller = self.factory.controllers[-1]
                    completed = (
                        2
                        if fault == "partial_loss"
                        else 1
                        if fault == "duplicate"
                        else 0
                    )
                    records = [
                        controller.ledger.read(slot) for slot in controller.slots
                    ]
                    self.assertEqual(
                        sum(record is not None for record in records), completed
                    )
                    self.assertTrue(
                        all(
                            record["status"] == "complete"
                            for record in records
                            if record is not None
                        )
                    )
                    self.assertEqual(
                        len(controller.ledger.unresolved()), len(self.ids) - completed
                    )

            for fault in ("submit_loss", "zero_receipts"):
                case(fault, variant)

    def test_all_protocol_or_oversize_fetch_failures_are_incorrect_not_infra(self):
        from unittest.mock import patch

        for variant in VARIANTS:

            def case(cause, variant):
                original = self.session.run

                def run(argv, **kwargs):
                    if argv[3] != "job-submit":
                        raise RemoteExecutionError(
                            "unusable per-case response", cause=cause
                        )
                    return original(argv, **kwargs)

                with (
                    self.subTest(variant=variant, cause=cause),
                    patch.object(self.session, "run", run),
                ):
                    result = collect(
                        self.relay,
                        variant,
                        self.requests_for(variant),
                        ids=self.ids,
                        timeout=10,
                        name=f"all-{variant}-{cause}",
                        parse=parse_test_receipt,
                    )
                self.assertEqual(result["model_case_failures"], frozenset(self.ids))
                self.assertEqual(self.factory.controllers[-1].ledger.unresolved(), [])

            for cause in ("protocol_error", "output_limit"):
                case(cause, variant)

    def test_trusted_signed_per_case_failure_preserves_healthy_siblings(self):
        for variant in ("vanilla", "reference"):
            with self.subTest(variant=variant):
                prefailed = frozenset(self.ids[:1])
                requests = {
                    ident: request
                    for ident, request in self.requests_for(variant).items()
                    if ident not in prefailed
                }
                captured = collect(
                    self.relay,
                    variant,
                    requests,
                    ids=self.ids,
                    timeout=10,
                    name="trusted-signed-case-" + variant,
                    parse=parse_test_receipt,
                    prefailed_ids=prefailed,
                )
                self.assertEqual(captured["model_case_failures"], prefailed)
                self.assertEqual(set(captured["receipts"]), set(self.ids) - prefailed)
                self.assertEqual(self.factory.controllers[-1].ledger.unresolved(), [])

    def test_timed_case_failure_has_no_fabricated_patch_or_dispatched_correctness(self):
        prefailed = frozenset(self.ids[:1])
        requests = {
            ident: request
            for ident, request in self.requests.items()
            if ident not in prefailed
        }
        captured = collect(
            self.relay,
            "candidate",
            requests,
            ids=self.ids,
            timeout=10,
            name="prefailed-correctness",
            parse=parse_test_receipt,
            prefailed_ids=prefailed,
        )
        self.assertEqual(captured["model_case_failures"], prefailed)
        self.assertEqual(set(captured["parsed"]), set(self.ids) - prefailed)
        self.assertNotIn(self.ids[0], self.session.doc["requests"])
        controller = self.factory.controllers[0]
        slot = next(
            slot for slot in controller.slots if slot.instance_id == self.ids[0]
        )
        self.assertFalse(controller.ledger.read(slot)["candidate_input_dispatched"])

    def requests_for(self, variant):
        return {
            ident: build_correctness_request(
                ident,
                "",
                variant=variant,
                identity={
                    **self.group.identity(),
                    "instance_id": ident,
                    "child_id": check_name(instance_index(ident)),
                    "phase": "train",
                },
            )
            for ident in self.ids
        }

    def test_unknown_model_grader_output_is_incorrect_for_every_variant(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):

                def parse(request, receipt):
                    if request["instance_id"] == self.ids[-1]:
                        raise CorrectnessRuntimeError(
                            "unknown model-generated import or output failure"
                        )
                    return parse_test_receipt(request, receipt)

                captured = collect(
                    self.relay,
                    variant,
                    self.requests_for(variant),
                    ids=self.ids,
                    timeout=10,
                    name="unknown-" + variant,
                    parse=parse,
                )
                self.assertEqual(
                    captured["model_case_failures"], frozenset(self.ids[-1:])
                )
                self.assertEqual(set(captured["receipts"]), set(self.ids[:-1]))
                controller = self.factory.controllers[-1]
                self.assertEqual(controller.ledger.unresolved(), [])
                self.assertEqual(
                    sum(
                        controller.ledger.read(slot)["status"] == "complete"
                        for slot in controller.slots
                    ),
                    len(self.ids) - 1,
                )

    def test_model_tainted_sidecar_setup_incomplete_and_kill_are_case_failures_all_variants(
        self,
    ):
        for variant in VARIANTS:
            for failure in (
                "surrogate_log",
                "surrogate_patch_log",
                "malformed_receipt",
                "unknown_output",
                "setup_failed",
                "incomplete_test_stages",
                "killed",
            ):
                with self.subTest(variant=variant, failure=failure):
                    self.session.case_failure = failure
                    captured = collect(
                        self.relay,
                        variant,
                        self.requests_for(variant),
                        ids=self.ids,
                        timeout=10,
                        name=variant + "-" + failure,
                        parse=parse_test_receipt,
                    )
                    self.assertEqual(
                        captured["model_case_failures"], frozenset(self.ids[-1:])
                    )
                    self.assertEqual(set(captured["receipts"]), set(self.ids[:-1]))
                    self.assertEqual(
                        self.factory.controllers[-1].ledger.unresolved(), []
                    )

    def test_unknown_receipt_parser_exceptions_fail_only_owned_cases_all_variants(self):
        for variant in VARIANTS:

            def case(index, error, variant):
                with self.subTest(variant=variant, error=type(error).__name__):
                    fault = self.ids[-1]

                    def parse(request, receipt):
                        if request["instance_id"] == fault:
                            raise error
                        return parse_test_receipt(request, receipt)

                    captured = collect(
                        self.relay,
                        variant,
                        self.requests_for(variant),
                        ids=self.ids,
                        timeout=10,
                        name=f"unknown-{variant}-{index}",
                        parse=parse,
                    )
                    self.assertEqual(
                        captured["model_case_failures"], frozenset({fault})
                    )
                    self.assertEqual(set(captured["receipts"]), set(self.ids) - {fault})
                    self.assertEqual(
                        self.factory.controllers[-1].ledger.unresolved(), []
                    )

            for index, error in enumerate(
                (
                    RuntimeError("unknown model output"),
                    ValueError("unknown syntax"),
                    TypeError("unknown field type"),
                    KeyError("unknown field"),
                    RecursionError("unknown nesting"),
                )
            ):
                case(index, error, variant)

    def test_fixed_inputs_and_harness_failures_stay_unscored(self):
        from unittest.mock import patch

        for variant in VARIANTS:

            def case(index, error, variant):
                with self.subTest(variant=variant, error=type(error).__name__):

                    def parse(*args):
                        raise error

                    captured = collect(
                        self.relay,
                        variant,
                        self.requests_for(variant),
                        ids=self.ids,
                        timeout=10,
                        name=f"cpu-{variant}-{index}",
                        parse=parse,
                    )
                    self.assertEqual(
                        captured["model_case_failures"], frozenset(self.ids)
                    )
                    self.assertEqual(
                        self.factory.controllers[-1].ledger.unresolved(), []
                    )

            for index, error in enumerate(
                (
                    TrustedCorrectnessError("fixed data digest mismatch"),
                    TrustedCorrectnessError("supported harness unavailable"),
                )
            ):
                case(index, error, variant)

            def preinput(*args, **kwargs):
                raise SidecarExecutionError(
                    "CPU verified pre-input control unavailable"
                )

            with patch.object(self.session, "run", preinput):
                with self.assertRaises(SidecarExecutionError):
                    collect(
                        self.relay,
                        variant,
                        self.requests_for(variant),
                        ids=self.ids,
                        timeout=10,
                        name="preinput-" + variant,
                        parse=parse_test_receipt,
                    )
            self.assertEqual(
                len(self.factory.controllers[-1].ledger.unresolved()), len(self.ids)
            )

    def test_malformed_trusted_script_return_never_becomes_model_case_failure(self):
        from unittest.mock import patch

        import cpu_swe_accuracy as accuracy

        grading, specs, _ = accuracy._harness()
        for variant in VARIANTS:
            requests = self.requests_for(variant)

            def case(index, commands, requests, variant):
                with self.subTest(variant=variant, commands=commands):
                    with patch.object(
                        accuracy,
                        "_harness",
                        return_value=(grading, specs, lambda *args: commands),
                    ):
                        captured = collect(
                            self.relay,
                            variant,
                            requests,
                            ids=self.ids,
                            timeout=10,
                            name=f"bad-script-{variant}-{index}",
                            parse=parse_test_receipt,
                        )
                    self.assertEqual(
                        captured["model_case_failures"], frozenset(self.ids)
                    )
                    self.assertEqual(
                        self.factory.controllers[-1].ledger.unresolved(), []
                    )

            for index, commands in enumerate((None, [None])):
                case(index, commands, requests, variant)
