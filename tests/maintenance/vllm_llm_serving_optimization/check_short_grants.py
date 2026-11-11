"""Localhost-only regression tests; not part of the published task tree."""

import deployment_test_guard  # noqa: F401

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "tasks/problems/MLSys/vllm_llm_serving_optimization"
sys.path[:0] = [str(TASK / "judge"), str(TASK / "verifier")]
from driver_client import DriverWorkIntervalError, SidecarExecutionError
from test_gpu_session import GPUWireTests


class ShortGrantTests(unittest.TestCase):
    def fixture(self):
        owner = GPUWireTests()
        owner.setUp()
        self.addCleanup(owner.doCleanups)
        return owner

    def test_positive_short_grant_installs_work_cleanup_and_capture_deadlines(self):
        self.fixture().test_short_grant_acceptance_persists_identity_and_applies_all_cutoffs()

    def test_write_failures_keep_the_short_grant_cleanup_deadline(self):
        for stage in ("parent", "marker"):
            with self.subTest(stage=stage):
                self.fixture().assert_grant_bound_on_failure(stage)

    def test_invalid_grants_are_rejected_and_only_the_owned_parent_is_cleaned(self):
        for value in (0, -1, 21, True, False, 1.5, "10", None):
            with self.subTest(value=value):
                owner = self.fixture()
                row = owner.runner()
                row["granted_seconds"] = value
                session, clock = owner.calibration_session()
                with patch.object(owner, "runner", return_value=row):
                    with self.assertRaises(SidecarExecutionError):
                        with session:
                            self.fail("invalid grant accepted")
                self.assertEqual(
                    [call[0] for call in owner.calls], ["create", "delete"]
                )
                self.assertIsNone(clock.granted_seconds)

    def test_short_grant_without_its_clock_is_rejected_and_cleaned(self):
        owner = self.fixture()
        row = owner.runner()
        row["granted_seconds"] = 5
        with patch.object(owner, "runner", return_value=row):
            with self.assertRaisesRegex(
                SidecarExecutionError, "bound calibration clock"
            ):
                with owner.session():
                    self.fail("unbound short grant accepted")
        self.assertEqual([call[0] for call in owner.calls], ["create", "delete"])

    def test_accepted_grant_without_work_interval_is_resource_exhaustion_not_infra(
        self,
    ):
        from parallel_phase import infrastructure_signal

        owner = self.fixture()
        session, clock = owner.calibration_session()
        row = owner.runner()
        row["granted_seconds"] = 4
        with patch.object(owner, "runner", return_value=row):
            with self.assertRaises(DriverWorkIntervalError) as raised:
                with session:
                    self.fail("exhausted interval started model work")
        self.assertEqual(clock.granted_seconds, 4)
        self.assertEqual(raised.exception.quota["resource"], "work_interval")
        self.assertIsNone(infrastructure_signal(raised.exception))
        self.assertEqual([call[0] for call in owner.calls], ["create", "delete"])

    def test_retries_keep_their_behavior_and_retain_nonsecret_diagnostics(self):
        owner = self.fixture()
        owner.retry_post, owner.retry_delete = 2, 1
        with owner.session():
            pass
        diagnostic = (owner.root / "evidence/driver-control.json").read_text()
        rows = json.loads(diagnostic)["events"]
        self.assertIn("transport_error", {row["outcome"] for row in rows})
        self.assertIn(425, {row.get("http_status") for row in rows})
        self.assertEqual(sum(row["outcome"] == "accepted" for row in rows), 2)
        self.assertNotIn("synthetic-test-bearer", diagnostic)
        self.assertNotIn(owner.client.driver_url, diagnostic)
        creates = [row for row in owner.calls if row[0] == "create"]
        self.assertEqual(len(creates), 3)
        self.assertEqual(len({row[2] for row in creates}), 1)

    def test_retry_diagnostics_are_bounded_and_keep_first_and_latest_events(self):
        owner = self.fixture()
        path = owner.root / "control.json"
        owner.client.enable_control_diagnostics(path)
        for _ in range(100):
            owner.client._control_event(
                "POST", "http_error", status=503, code="gpu_backend_unavailable"
            )
        value = json.loads(path.read_bytes())
        self.assertEqual(len(value["events"]), 32)
        self.assertEqual(value["omitted_events"], 68)
        self.assertLess(path.stat().st_size, 12000)

    def test_unwritable_diagnostics_do_not_replay_acceptance_or_prevent_cleanup(self):
        owner = self.fixture()
        enable = owner.client.enable_control_diagnostics
        with patch.object(
            owner.client,
            "enable_control_diagnostics",
            side_effect=lambda path: enable(owner.root / "missing" / "control.json"),
        ):
            with owner.session():
                pass
        self.assertEqual([row[0] for row in owner.calls], ["create", "delete"])
        self.assertTrue(owner.client._diagnostics["write_failed"])
        self.assertEqual(
            json.loads((owner.root / "evidence/cleanup.json").read_bytes())["state"],
            "terminated",
        )


if __name__ == "__main__":
    unittest.main()
