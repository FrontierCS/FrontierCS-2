"""Real signatures through fragmented streaming and hostile relay failures."""

import deployment_test_guard  # noqa: F401

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_task import TASK_ROOT  # noqa: E402

TASK = TASK_ROOT
sys.path[:0] = [str(TASK / "judge"), str(TASK / "verifier")]
import attestation
from driver_client import RemoteExecutionError, SidecarExecutionError
from relay_client import SchedulerRelay, canonical_bytes
from scheduler_stream import SCHEMA, Decoder, FrameError, decode, frame
from sidecar_catalog import SCHEDULER
from test_attestation import keypair


class Session:
    runner_id, mode = "runner_one", "eval"

    def __init__(self, signer):
        self.signer = signer
        self.client = SimpleNamespace(
            remaining=lambda: 10, verification_id="verification"
        )
        self.operations, self.calls, self.limits = 0, [], {"ssh_sessions": 10}
        self.drop, self.corrupt, self.bad_final = False, False, False
        self.ids, self.drop_always = ("original",), False

    def run(self, argv, *, stdin, output, on_stdout, **kwargs):
        self.operations += 1
        self.calls.append((argv, stdin, kwargs))
        doc = json.loads(stdin)
        digest = "sha256:" + hashlib.sha256(stdin).hexdigest()

        def signed(payload):
            return self.signer.sign(
                kind=doc["kind"],
                sequence=doc["sequence"],
                request_sha256=digest,
                payload=payload,
            )

        rows = getattr(self, "rows", None) or {
            ident: {
                "instance_id": ident,
                "episode": {
                    "original_value": 7,
                    "instance_id": ident,
                    "started_monotonic": 1.0,
                    "ended_monotonic": 2.0,
                    "latency_seconds": 1.0,
                    "exit_status": "ok",
                    "output": "",
                    "error": "",
                },
                "trace": [],
                "requires_error_attribution": False,
            }
            for ident in self.ids
        }
        for index in range(2 * len(self.ids)):
            ident, event = self.ids[index // 2], "terminal" if index % 2 else "dispatch"
            payload = {
                "variant": doc["variant"],
                "event_index": index,
                "event": event,
                "record": {"instance_id": ident}
                if event == "dispatch"
                else rows[ident],
            }
            envelope = signed(payload)
            if self.corrupt:
                envelope = {}
            raw = frame(
                {
                    "schema": SCHEMA,
                    "event_kind": "progress",
                    "event_index": index,
                    "value": envelope,
                },
                100000,
            )
            # Real callback delivery is arbitrary chunks, including split headers.
            for offset in range(0, len(raw), 13):
                on_stdout(raw[offset : offset + 13])
        if self.drop_always or self.drop and self.operations == 1:
            raise RemoteExecutionError(
                "lost stream after durable sibling", cause="child_exit"
            )
        payload = {
            "variant": doc["variant"],
            "batch": {"original_header": 9},
            "episode_sha256": {
                ident: "sha256:" + hashlib.sha256(canonical_bytes(row)).hexdigest()
                for ident, row in rows.items()
            },
            "episode_instance_ids": list(self.ids),
        }
        if self.bad_final:
            payload["episode_instance_ids"] = [None]
        view = {
            "state": "complete",
            "sequence": doc["sequence"],
            "request_sha256": digest,
            "signed": signed(payload),
        }
        on_stdout(
            frame(
                {
                    "schema": SCHEMA,
                    "event_kind": "complete",
                    "event_index": 2 * len(self.ids),
                    "value": view,
                },
                100000,
            )
        )
        return Path(output) / "stdout"


class SchedulerStreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        private, self.public = keypair()
        identity = {
            "run_id": "run",
            "attempt_id": "attempt",
            "operation_id": "runner_one",
            "phase": "eval",
            "name": SCHEDULER,
            "profile_digest": "sha256:" + "a" * 64,
        }
        self.session = Session(attestation.Signer(private, self.public, identity))

    def relay(self, **kwargs):
        return SchedulerRelay(
            self.session,
            public_key=self.public,
            relay_limits={"job_bytes": 10000, "response_bytes": 100000},
            max_stderr_bytes=512,
            evidence=Path(self.tmp.name) / "relay",
            stream=True,
            cleanup_seconds=1,
            **kwargs,
        )

    def test_lost_stream_replays_once_and_reconstructs_exact_original_rows(self):
        events = []
        self.session.drop = True
        payload = self.relay(on_progress=lambda value, raw: events.append(value)).job(
            "batch", "candidate", {}, timeout=10, name="batch"
        )
        self.assertEqual(len(events), 2)
        self.assertEqual(len(self.session.calls), 2)
        self.assertEqual(self.session.calls[0][1], self.session.calls[1][1])
        self.assertEqual(
            payload["batch"],
            {
                "original_header": 9,
                "episodes": [
                    {
                        "instance_id": "original",
                        "episode": {
                            "original_value": 7,
                            "instance_id": "original",
                            "started_monotonic": 1.0,
                            "ended_monotonic": 2.0,
                            "latency_seconds": 1.0,
                            "exit_status": "ok",
                            "output": "",
                            "error": "",
                        },
                        "trace": [],
                        "requires_error_attribution": False,
                    }
                ],
            },
        )
        self.assertEqual(self.session.calls[0][0][3], "job-stream")

    def test_partial_patch_model_errors_become_cpu_owned_cases_for_every_variant(self):
        from candidate_projection import timed_case_failures
        from test_vllm_correctness_stream import CorrectnessStreamTests

        for variant in ("vanilla", "reference", "candidate"):
            with self.subTest(variant=variant):
                fixture = CorrectnessStreamTests()
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                from controller_contract_build import emit
                from phase_controller import PhaseControllerFactory

                source = fixture.root / "timed-source"
                (source / "verifier").mkdir(parents=True)
                (source / "verifier/runtime-plan.json").write_bytes(
                    (TASK / "verifier/runtime-plan.json").read_bytes()
                )
                emit(
                    source,
                    ["verifier"],
                    [],
                    source / "controller-contract.json",
                    source / "controller-contract-limits.json",
                )
                fixture.factory = PhaseControllerFactory(
                    fixture.session,
                    {v: fixture.group for v in ("vanilla", "reference", "candidate")},
                    source_root=source,
                    environment=fixture.factory.environment,
                    maximum_terminal_bytes=100000,
                    maximum_manifest_bytes=100000,
                )
                fixture.relay.controller_factory = fixture.factory
                session = Session(fixture.session.signer)
                session.runner_id, session.mode = "parent", "train"
                session.ids = fixture.ids
                session.rows = {
                    ident: {
                        "instance_id": ident,
                        "episode": {
                            "patch": "partial",
                            "exit_status": "limit_with_patch",
                            "error": "PreviouslyUnseenModelError",
                        },
                        "requires_error_attribution": True,
                        "trace": [
                            {
                                "kind": "model",
                                "error_origin": "model_api_response",
                                "error_type": "PreviouslyUnseenModelError",
                            }
                        ],
                    }
                    for ident in session.ids
                }
                fixture.relay.session = session
                fixture.factory.session = session
                fixture.relay.job(
                    "batch", variant, {}, timeout=10, name=variant + "-batch"
                )
                self.assertEqual(
                    timed_case_failures(
                        fixture.factory, fixture.group, variant=variant
                    ),
                    frozenset(session.ids),
                )
                batch = fixture.factory.controllers[0]
                self.assertTrue(
                    all(
                        batch.ledger.read(slot)["status"] == "runtime_error"
                        for slot in batch.slots
                    )
                )
                self.assertEqual(batch.ledger.unresolved(), [])

    def test_unsigned_remote_closes_but_cpu_journal_failure_propagates(self):
        self.session.corrupt = True
        closed = []
        controller = SimpleNamespace(fail_unresolved=lambda **kw: closed.append(kw))
        with self.assertRaises(SidecarExecutionError):
            self.relay(controller_factory=lambda *args, **kwargs: controller).job(
                "batch", "candidate", {}, timeout=10, name="batch"
            )
        self.assertEqual(len(closed), 2)

    def test_trusted_progress_callback_error_is_never_remote_retry(self):
        def broken(*args):
            raise ValueError("CPU fsync integrity defect")

        with self.assertRaisesRegex(ValueError, "CPU fsync"):
            self.relay(on_progress=broken).job(
                "batch", "candidate", {}, timeout=10, name="batch"
            )
        self.assertEqual(len(self.session.calls), 1)

    def test_all_case_terminals_survive_missing_aggregate_with_separate_group_failure(
        self,
    ):
        self.assert_owned_protocol_failure("lost")

    def test_authenticated_invalid_batch_header_has_owned_group_failure(self):
        self.assert_owned_protocol_failure("invalid_batch")

    def assert_owned_protocol_failure(self, failure):
        from candidate_projection import runtime_group_failure
        from phase_run import _validate_owned_payload
        from test_vllm_phase_controller import PhaseControllerTests

        fixture = PhaseControllerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        private, public = keypair()
        identity = {
            "run_id": "run",
            "attempt_id": "attempt",
            "operation_id": "parent",
            "phase": "train",
            "name": SCHEDULER,
            "profile_digest": "sha256:" + "a" * 64,
        }
        session = Session(attestation.Signer(private, public, identity))
        session.runner_id, session.mode = "parent", "train"
        session.ids = tuple(
            __import__("instance_manifest").expected_groups("train")["bfcl/public"]
        )
        session.drop_always = failure == "lost"
        factory = fixture.factory()
        relay = SchedulerRelay(
            session,
            public_key=public,
            relay_limits={"job_bytes": 10000, "response_bytes": 1000000},
            max_stderr_bytes=512,
            evidence=Path(self.tmp.name) / "relay",
            stream=True,
            cleanup_seconds=1,
            controller_factory=factory,
            payload_validator=(
                lambda kind, variant, payload: _validate_owned_payload(
                    kind,
                    variant,
                    payload,
                    groups={"candidate": fixture.group},
                    factory=factory,
                    calibration=False,
                )
            )
            if failure == "invalid_batch"
            else None,
        )
        with self.assertRaises(RemoteExecutionError):
            relay.job("batch", "candidate", {}, timeout=10, name="batch")
        batch, control = factory.controllers
        self.assertTrue(
            all(batch.ledger.read(slot)["status"] == "complete" for slot in batch.slots)
        )
        self.assertTrue(
            all(
                control.ledger.read(slot)["status"] == "runtime_error"
                for slot in control.slots
            )
        )
        self.assertTrue(
            all(slot.group.endswith("/group_control") for slot in control.slots)
        )
        rows = runtime_group_failure(factory, fixture.group)
        self.assertEqual(len(rows), len(session.ids))
        self.assertTrue(all(row.execution == "candidate_error" for row in rows))
        self.assertEqual(len(session.calls), 3 if failure == "lost" else 1)

    def test_trusted_payload_validator_fault_does_not_close_candidate_slots(self):
        def case(error_type):
            with self.subTest(error_type=error_type):
                closed = []
                controller = SimpleNamespace(
                    slots=[
                        SimpleNamespace(instance_id="original", group="bfcl/public")
                    ],
                    mark_dispatched=lambda slot: None,
                    observe=lambda *args, **kw: None,
                    maximum_terminal_bytes=100000,
                    fail_unresolved=lambda **kw: closed.append(kw),
                )

                def validate(*args):
                    from relay_client import TrustedRelayEvidenceError

                    raise TrustedRelayEvidenceError(
                        "CPU original-plan validation failed"
                    ) from error_type()

                from relay_client import TrustedRelayEvidenceError

                with self.assertRaisesRegex(
                    TrustedRelayEvidenceError, "CPU original-plan"
                ):
                    SchedulerRelay(
                        self.session,
                        public_key=self.public,
                        relay_limits={"job_bytes": 10000, "response_bytes": 100000},
                        max_stderr_bytes=512,
                        evidence=Path(self.tmp.name) / error_type.__name__,
                        stream=True,
                        cleanup_seconds=1,
                        controller_factory=lambda *args, **kwargs: controller,
                        payload_validator=validate,
                    ).job("batch", "candidate", {}, timeout=10, name="batch")
                self.assertEqual(closed, [])

        for error_type in (OSError, RuntimeError, MemoryError):
            case(error_type)

    def test_trusted_attestation_fault_does_not_close_candidate_slots(self):
        closed = []
        controller = SimpleNamespace(
            slots=[SimpleNamespace(instance_id="original", group="bfcl/public")],
            mark_dispatched=lambda slot: None,
            observe=lambda *args, **kw: None,
            maximum_terminal_bytes=100000,
            fail_unresolved=lambda **kw: closed.append(kw),
        )
        with patch.object(
            attestation, "verify", side_effect=OSError("CPU crypto unavailable")
        ):
            from relay_client import TrustedRelayEvidenceError

            with self.assertRaisesRegex(TrustedRelayEvidenceError, "CPU attestation"):
                self.relay(controller_factory=lambda *args, **kwargs: controller).job(
                    "batch", "candidate", {}, timeout=10, name="batch"
                )
        self.assertEqual(closed, [])

    def test_malformed_final_also_closes_unresolved_cpu_slots(self):
        self.session.bad_final = True
        closed = []
        controller = SimpleNamespace(
            slots=[SimpleNamespace(instance_id="original", group="bfcl/public")],
            mark_dispatched=lambda slot: None,
            observe=lambda *args, **kw: None,
            maximum_terminal_bytes=100000,
            fail_unresolved=lambda **kw: closed.append(kw),
        )
        relay = self.relay(controller_factory=lambda *args, **kwargs: controller)
        with self.assertRaises(SidecarExecutionError):
            relay.job("batch", "candidate", {}, timeout=10, name="batch")
        self.assertEqual(len(closed), 2)

    def test_framing_duplicate_json_and_bounded_partial_records(self):
        accepted = []
        decoder = Decoder(
            maximum_frame_bytes=32, maximum_total_bytes=64, accept=accepted.append
        )
        with self.assertRaises(FrameError):
            decoder.feed(struct.pack("!I", 33))
        decoder = Decoder(
            maximum_frame_bytes=32, maximum_total_bytes=64, accept=accepted.append
        )
        decoder.feed(struct.pack("!I", 5) + b"xx")
        with self.assertRaises(FrameError):
            decoder.finish()
        for raw in [b'{"a":1,"a":2}', b'{"a":NaN}']:
            with self.assertRaises(ValueError):
                decode(raw)
