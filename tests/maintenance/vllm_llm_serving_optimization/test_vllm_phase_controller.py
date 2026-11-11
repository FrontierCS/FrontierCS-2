"""CPU frozen ownership binds the actual Driver grant, request and image inputs."""

import deployment_test_guard  # noqa: F401

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_task import TASK_ROOT  # noqa: E402

TASK = TASK_ROOT
sys.path[:0] = [str(TASK / "judge"), str(TASK / "verifier")]
from batch_coordinator import PreparedGroup
from candidate_projection import runtime_group_failure
from containment_policy import Boundary
from controller_contract_build import emit
from driver_client import RemoteExecutionError, SidecarExecutionError
from gpu_session import GPUSession
from phase_controller import PhaseControllerFactory
from phase_lifecycle import CPULifecycleEvidenceError, owned_lifecycle


class PhaseControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "judge").mkdir()
        (self.root / "judge/source.py").write_bytes(b"immutable CPU source")
        emit(
            self.root,
            ["judge"],
            [],
            self.root / "controller-contract.json",
            self.root / "controller-contract-limits.json",
        )
        self.environment = {
            "FRONTIER_RUN_ID": "run",
            "FRONTIER_ATTEMPT_ID": "attempt",
            "FRONTIER_VERIFICATION_ID": "verification",
            "FRONTIER_TASK_CONFIG_DIGEST": "sha256:" + "c" * 64,
            "FRONTIER_VERIFIER_IMAGE_DIGEST": "sha256:" + "d" * 64,
        }
        self.session = SimpleNamespace(
            runner_id="parent",
            mode="train",
            client=SimpleNamespace(verification_id="verification"),
        )
        self.group = PreparedGroup(
            "parent",
            "parent",
            "parent",
            "train",
            "bfcl",
            "public",
            "http://main:8000/v1",
            "sha256:" + "a" * 64,
        )

    def factory(self, **changes):
        arguments = dict(
            session=self.session,
            groups={"candidate": self.group},
            source_root=self.root,
            environment=self.environment,
            maximum_terminal_bytes=100000,
            maximum_manifest_bytes=100000,
        )
        return PhaseControllerFactory(**dict(arguments, **changes))

    def test_exact_expected_slots_exist_before_dispatch_and_cannot_alias_requests(self):
        factory = self.factory()
        started = time.monotonic()
        controller = factory(
            "batch",
            "candidate",
            {},
            "sha256:" + "b" * 64,
            self.root / "first",
            started=started,
            deadline=started + 10,
        )
        self.assertEqual(len(controller.slots), 8)
        self.assertTrue(
            all(
                slot.run_id == "run"
                and slot.attempt_id == "attempt"
                and slot.request_digest == "sha256:" + "b" * 64
                for slot in controller.slots
            )
        )
        self.assertTrue((controller.evidence / "manifest.json").exists())
        self.assertTrue(
            all(controller.ledger.read(slot) is None for slot in controller.slots)
        )
        self.assertEqual(
            controller.signer.context.executable_image_digest,
            self.environment["FRONTIER_VERIFIER_IMAGE_DIGEST"],
        )

    def test_missing_driver_identity_or_foreign_grant_blocks_before_allocation(self):
        environment = dict(self.environment)
        del environment["FRONTIER_RUN_ID"]
        with self.assertRaises(ValueError):
            self.factory(environment=environment)
        self.session.client.verification_id = "foreign"
        with self.assertRaises(ValueError):
            self.factory()

    def lifecycle(self, factory, invoke, name="startup"):
        return owned_lifecycle(
            factory,
            "start",
            "candidate",
            {"variant": "candidate"},
            self.root / name,
            deadline=time.monotonic() + 10,
            invoke=invoke,
        )

    def test_startup_failure_is_cpu_owned_before_first_input(self):
        def case(cause):
            factory = self.factory()

            def invoke():
                self.assertEqual(len(factory.controllers), 1)
                self.assertTrue(
                    (factory.controllers[0].evidence / "manifest.json").exists()
                )
                raise RemoteExecutionError("bounded failure", cause=cause)

            with self.subTest(cause=cause), self.assertRaises(RemoteExecutionError):
                self.lifecycle(factory, invoke, cause)
            controller = factory.controllers[0]
            self.assertEqual(len(controller.slots), 8)
            for slot in controller.slots:
                record = controller.ledger.read(slot)
                self.assertEqual(record["cause"], cause)
                self.assertEqual(
                    record["failure_context"]["boundary"]["stage"], "startup"
                )
                self.assertEqual(
                    record["failure_context"]["decision"]["disposition"],
                    "runtime_error",
                )
                self.assertEqual(
                    record["failure_context"]["decision"]["dispatch_state"], "unknown"
                )
                self.assertFalse(record["candidate_input_dispatched"])

        for cause in ["child_exit", "deadline", "output_limit", "protocol_error"]:
            case(cause)

    def test_healthy_lifecycle_is_distinct_and_does_not_fabricate_measurements(self):
        factory = self.factory()
        self.lifecycle(
            factory,
            lambda: {
                "state": "ready",
                "model_endpoint": "http://main:8000/v1",
                "pid": 42,
                "process_start_ticks": 123,
            },
        )
        controller = factory.controllers[0]
        for slot in controller.slots:
            self.assertTrue(slot.group.endswith("/startup"))
            record = controller.ledger.read(slot)
            self.assertEqual(record["status"], "complete")
            self.assertEqual(
                record["result"], {"lifecycle": "startup", "state": "ready"}
            )

    def test_missing_or_invalid_start_process_identity_closes_startup_slots(self):
        for key in ("pid", "process_start_ticks"):

            def case(value, key):
                factory = self.factory()
                receipt = {
                    "state": "ready",
                    "model_endpoint": "http://main:8000/v1",
                    "pid": 42,
                    "process_start_ticks": 123,
                }
                receipt[key] = value
                with (
                    self.subTest(key=key, value=value),
                    self.assertRaises(RemoteExecutionError),
                ):
                    self.lifecycle(factory, lambda: receipt, key + str(value))
                self.assertIsNotNone(runtime_group_failure(factory, self.group))

            for value in (None, False, 0, "1"):
                case(value, key)

    def test_remote_swe_shutdown_validation_is_typed_before_group_completion(self):
        from phase_run import _validate_owned_payload
        from test_batch_measurements import MeasurementProjectionTests

        group, batch = MeasurementProjectionTests().make("swe")
        with self.assertRaises(RemoteExecutionError):
            _validate_owned_payload(
                "batch",
                "candidate",
                {"batch": batch, "testbed_shutdown": None},
                groups={"candidate": group},
                factory=None,
                calibration=False,
            )

    def test_typed_cpu_failure_remains_unscored(self):
        factory = self.factory()

        def invoke():
            raise CPULifecycleEvidenceError("CPU evidence filesystem failed")

        with self.assertRaises(CPULifecycleEvidenceError):
            self.lifecycle(factory, invoke)
        controller = factory.controllers[0]
        self.assertTrue(
            all(controller.ledger.read(slot) is None for slot in controller.slots)
        )

    def test_unknown_lifecycle_failure_is_owned_runtime_error_all_variants(self):
        for variant in ("vanilla", "reference", "candidate"):
            for action in ("start", "stop"):
                with self.subTest(variant=variant, action=action):
                    factory = self.factory(groups={variant: self.group})

                    def invoke():
                        raise SidecarExecutionError(
                            "remote claims platform capacity",
                            error_code="driver_platform_capacity",
                            driver_error_code="gpu_create_failed",
                        )

                    with self.assertRaises(RemoteExecutionError):
                        owned_lifecycle(
                            factory,
                            action,
                            variant,
                            {"variant": variant},
                            self.root / (variant + "-" + action),
                            deadline=time.monotonic() + 10,
                            invoke=invoke,
                        )
                    controller = factory.controllers[0]
                    self.assertEqual(controller.ledger.unresolved(), [])
                    self.assertTrue(
                        all(
                            controller.ledger.read(slot)["cause"] == "unknown_exception"
                            for slot in controller.slots
                        )
                    )

    def test_illegal_startup_receipt_is_c1_not_infrastructure(self):
        factory = self.factory()
        with self.assertRaises(RemoteExecutionError):
            self.lifecycle(factory, lambda: {"state": "error"})
        controller = factory.controllers[0]
        self.assertTrue(
            all(
                controller.ledger.read(slot)["cause"] == "protocol_error"
                for slot in controller.slots
            )
        )

    def test_real_owned_process_crash_kill_hang_and_overflow_close_startup_slots(self):
        # This exercises the actual CPU process/pipe owner, not Driver or GPU.
        scripts = [
            ("crash", 'raise RuntimeError("remote worker failed")', "child_exit"),
            (
                "kill",
                "import os,signal;os.kill(os.getpid(),signal.SIGKILL)",
                "child_exit",
            ),
            ("hang", "import time;time.sleep(20)", "deadline"),
            ("overflow", 'import os;os.write(1,b"x"*10000)', "output_limit"),
        ]

        def case(name, script, cause):
            session = GPUSession.__new__(GPUSession)
            session.mode, session.runner_id = "train", "parent"
            session.client = SimpleNamespace(
                verification_id="verification",
                deadline=time.monotonic() + (0.15 if name == "hang" else 5),
            )
            session.client.remaining = lambda: max(
                0.001, session.client.deadline - time.monotonic()
            )
            session.cleanup_deadline = time.monotonic() + 6
            session.operations = 0
            session.limits = {"ssh_sessions": 10, "ssh_bytes_per_direction": 100000}
            session.ssh, session.environment = (
                [sys.executable, "-c", script],
                dict(os.environ),
            )
            factory = self.factory(session=session)

            def invoke():
                return session.run(
                    ["local-fixture-only"],
                    output=self.root / (name + "-pipes"),
                    max_stdout_bytes=1000,
                    max_stderr_bytes=1000,
                )

            with self.subTest(name=name), self.assertRaises(RemoteExecutionError):
                owned_lifecycle(
                    factory,
                    "start",
                    "candidate",
                    {"variant": "candidate"},
                    self.root / (name + "-ownership"),
                    deadline=session.client.deadline,
                    invoke=invoke,
                )
            controller = factory.controllers[0]
            self.assertTrue(
                all(
                    controller.ledger.read(slot)["cause"] == cause
                    for slot in controller.slots
                )
            )

        for name, script, cause in scripts:
            case(name, script, cause)

    def test_c1_projection_preserves_denominator_without_fabricating_latency(self):
        factory = self.factory()
        with self.assertRaises(RemoteExecutionError):
            self.lifecycle(factory, lambda: {"state": "error"})
        projected = runtime_group_failure(factory, self.group)
        self.assertEqual(len(projected), 8)
        self.assertTrue(
            all(
                row.latency_seconds is None
                and row.correct is False
                and row.execution == "candidate_error"
                for row in projected
            )
        )

    def test_reference_and_c2_have_no_group_failure_numeric_convention(self):
        factory = self.factory(groups={"reference": self.group})
        started = time.monotonic()
        controller = factory(
            "startup",
            "reference",
            {},
            "sha256:" + "b" * 64,
            self.root / "reference",
            started=started,
            deadline=started + 5,
        )
        controller.fail_unresolved(boundary=Boundary("candidate_service", "startup"))
        self.assertIsNone(runtime_group_failure(factory, self.group))
        factory = self.factory()
        controller = factory(
            "batch",
            "candidate",
            {},
            "sha256:" + "b" * 64,
            self.root / "case",
            started=started,
            deadline=started + 5,
        )
        controller.fail_unresolved(boundary=Boundary("swe_case", "tool"))
        self.assertIsNone(runtime_group_failure(factory, self.group))

    def test_live_single_group_startup_c1_returns_fixed_candidate_failure_set(self):
        self.live_lifecycle_failure("start")

    def test_live_single_group_shutdown_c1_returns_fixed_candidate_failure_set(self):
        self.live_lifecycle_failure("stop")

    def live_lifecycle_failure(self, action):
        import phase_run
        from test_phase_run import PhaseJoinTests

        fixture = PhaseJoinTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.run_parent_original = fixture.run_parent

        def run_parent(argv, **arguments):
            import json

            if (
                argv[3] == action
                and json.loads(arguments["stdin"])["variant"] == "candidate"
            ):
                raise RemoteExecutionError(
                    "candidate server exited before readiness", cause="child_exit"
                )
            return fixture.run_parent_original(argv, **arguments)

        fixture.session.run = run_parent
        environment = dict(
            self.environment, FRONTIER_VERIFICATION_ID="verification_one"
        )
        factories = []

        def allocate(session, groups, **arguments):
            arguments["source_root"] = self.root
            factory = PhaseControllerFactory(session, groups, **arguments)
            factories.append(factory)
            return factory

        original_relay = fixture.relay

        class Relay(original_relay):
            def __init__(self, *args, **arguments):
                for key in ("stream", "controller_factory", "cleanup_seconds"):
                    arguments.pop(key)
                super().__init__(*args, **arguments)

        with (
            patch.object(phase_run, "validate_projection", lambda config: config),
            patch.object(phase_run, "SchedulerRelay", Relay),
            patch.object(phase_run, "PhaseControllerFactory", allocate),
        ):
            observations = phase_run.run_phase(
                fixture.session,
                fixture.config,
                fixture.plan,
                candidate_patch=b"",
                evidence=fixture.root / "phase",
                startup_seconds=30,
                stop_seconds=5,
                max_stdout_bytes=1000000,
                max_stderr_bytes=1000,
                selected_group="bfcl/public",
                controller_environment=environment,
            )
        rows = observations["candidate"]["bfcl/public"]
        self.assertEqual(len(rows), 8)
        self.assertTrue(
            all(
                row.execution == "candidate_error" and row.latency_seconds is None
                for row in rows
            )
        )
        if action == "start":
            self.assertFalse(
                any(event == "scheduler-bfcl-candidate" for event in fixture.events)
            )
        self.assertFalse((fixture.root / "phase/operator-result.json").exists())
