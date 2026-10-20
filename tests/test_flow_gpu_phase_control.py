"""Flow's real phase-control path spends grants and retains cleanup evidence."""

import deployment_test_guard  # noqa: F401

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_gpu_remaining_envelopes import PROBLEMS, load


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def assert_grant_cleanup(test, adapter, grant, elapsed):
    clock = FakeClock()
    snapshots = []
    scratch_paths = []

    def create(*args, window):
        test.assertGreater(window.remaining(), 0)
        clock.advance(elapsed)
        return {"runner_id": "owned", "granted_seconds": grant}

    def archive(path, *args):
        scratch_paths.append(path.parent)

    with (
        tempfile.TemporaryDirectory() as temp,
        mock.patch.dict(os.environ, {"FRONTIER_MAX_EVIDENCE_BYTES": "8192",
                                    "FRONTIER_VERIFIER_DEADLINE_UNIX_MS": ""}),
        mock.patch.object(adapter, "_config", return_value=("url", "verification", "token", "gpu", 7200, 1)),
        mock.patch.object(adapter, "_input_archive", side_effect=archive),
        mock.patch.object(adapter.time, "monotonic", side_effect=clock.monotonic),
        mock.patch.object(adapter, "_create_runner", side_effect=create) as provision,
        mock.patch.object(adapter, "_terminate_runner", return_value=True) as delete,
        mock.patch.object(adapter, "_upload_inputs") as upload,
        mock.patch.object(adapter, "_run_remote_verifier") as remote,
    ):
        with test.assertRaises(adapter.VerifierGpuError) as caught:
            adapter.run(payload=Path(temp)/"payload", dataset=Path(temp)/"dataset", mode="eval",
                        result_path=Path(temp)/"result", evidence_path=Path(temp)/"evidence",
                        record_evidence=snapshots.append)
        provision.assert_called_once()
        test.assertEqual(str(caught.exception), "Verifier GPU control failed")
        if grant is True:
            test.assertIn("invalid GPU granted duration", str(caught.exception.__cause__))
        else:
            test.assertIsInstance(caught.exception.__cause__, adapter._PhaseTimeout)
            test.assertEqual(caught.exception.__cause__.boundary, "work_budget_exhausted")
        delete.assert_called_once()
        test.assertEqual(delete.call_args.args, ("url", "verification", "token", "owned"))
        test.assertGreater(delete.call_args.kwargs["window"].remaining(), 0)
        upload.assert_not_called()
        remote.assert_not_called()
    test.assertTrue(scratch_paths)
    test.assertTrue(all(not path.exists() for path in scratch_paths))
    evidence = caught.exception.private_evidence
    test.assertEqual(snapshots[-1], evidence)
    events = {event["phase"]: event for event in evidence["events"]}
    test.assertEqual(events["delete"]["runner_termination"], "confirmed")
    test.assertEqual(events["delete"]["status"], "completed")
    test.assertEqual(events["cleanup"]["status"], "completed")
    test.assertEqual(evidence["evaluator_progress"], "unobserved")
    test.assertFalse(evidence["evidence_write_failed"])
    test.assertLessEqual(len(json.dumps(evidence).encode()), adapter.MAX_PHASE_EVIDENCE_BYTES)


class FlowPhaseControlTests(unittest.TestCase):
    def adapter(self):
        if os.environ.get("FRONTIER_FLOW_OPD_EXPORT"):
            return load(Path(os.environ["FRONTIER_FLOW_OPD_EXPORT"])/"verifier/fcs2_verifier_gpu.py",
                        "flow_phase_adapter")
        paths = list(PROBLEMS.rglob("flow_opd/verifier/fcs2_verifier_gpu.py"))
        self.assertEqual(len(paths), 1)
        return load(paths[0], "flow_phase_adapter")

    def test_invalid_grant_and_exhausted_work_budget_cleanup(self):
        for grant, elapsed in ((True, 0), (100, 50)):
            with self.subTest(grant=grant):
                assert_grant_cleanup(self, self.adapter(), grant, elapsed)

    def test_fully_consumed_grant_reports_unconfirmed_cleanup_without_score(self):
        adapter = self.adapter()
        clock = FakeClock()
        snapshots = []
        scratch_paths = []

        def create(*args, window):
            clock.advance(200)
            return {"runner_id": "owned", "granted_seconds": 100}

        def archive(path, *args):
            scratch_paths.append(path.parent)

        try:
            with (
                tempfile.TemporaryDirectory() as temp,
                mock.patch.dict(os.environ, {"FRONTIER_MAX_EVIDENCE_BYTES": "8192",
                                            "FRONTIER_VERIFIER_DEADLINE_UNIX_MS": ""}),
                mock.patch.object(adapter.time, "monotonic", side_effect=clock.monotonic),
                mock.patch.object(adapter, "_config", return_value=("url", "verification", "token", "gpu", 7200, 1)),
                mock.patch.object(adapter, "_input_archive", side_effect=archive),
                mock.patch.object(adapter, "_create_runner", side_effect=create) as provision,
                mock.patch.object(adapter, "_terminate_runner", return_value=True) as delete,
                mock.patch.object(adapter, "_upload_inputs") as upload,
                mock.patch.object(adapter, "_run_remote_verifier") as remote,
                mock.patch.object(adapter, "_copy_outputs") as copy,
            ):
                result = Path(temp) / "result"
                with self.assertRaises(adapter.VerifierGpuError) as caught:
                    adapter.run(payload=Path(temp)/"payload", dataset=Path(temp)/"dataset", mode="eval",
                                result_path=result, evidence_path=Path(temp)/"evidence",
                                record_evidence=snapshots.append)
                provision.assert_called_once()
                upload.assert_not_called()
                remote.assert_not_called()
                copy.assert_not_called()
                delete.assert_not_called()
                self.assertFalse(result.exists())
                self.assertEqual(str(caught.exception), "Verifier GPU control failed")
                evidence = caught.exception.private_evidence
                events = {event["phase"]: event for event in evidence["events"]}
                for phase in ("upload", "delete", "cleanup"):
                    self.assertEqual(events[phase]["status"], "timed_out")
                self.assertEqual(events["outer_deadline"]["status"], "bound_expired")
                self.assertFalse(events["outer_deadline"]["driver_expiry_observed"])
                self.assertNotIn("runner_termination", events["delete"])
                self.assertEqual(evidence["evaluator_progress"], "unobserved")
                self.assertTrue(evidence["evidence_write_failed"])
                self.assertNotIn("score", evidence)
                self.assertTrue(scratch_paths)
                self.assertTrue(all(path.exists() for path in scratch_paths))
        finally:
            # The adapter truthfully cannot clean after its entire envelope
            # expires. Remove only this test's local temporary directories.
            for path in scratch_paths:
                shutil.rmtree(path)

    def test_request_anchor_and_cleanup_reserve_are_not_renewed(self):
        adapter = self.adapter()
        clock = FakeClock()
        with mock.patch.object(adapter.time, "monotonic", side_effect=clock.monotonic):
            outer = adapter._OuterDeadline(7300, "declared_request_pending_grant")
            anchor = clock.monotonic()
            clock.advance(200)
            deadline = adapter._runner_deadline({"granted_seconds": 1200}, 7200, anchor=anchor)
            self.assertEqual(deadline, 1299)
            bounded = outer.with_grant(deadline)
            self.assertEqual(bounded.end, 1299)
            self.assertEqual(bounded.work_end, 609)
            self.assertEqual(bounded.cleanup_end, 1269)
            self.assertEqual(bounded.window(7200).remaining(), 309)
            clock.advance(310)
            with self.assertRaises(adapter._PhaseTimeout) as caught:
                bounded.window(120).remaining()
            self.assertEqual(caught.exception.boundary, "work_budget_exhausted")
            self.assertEqual(bounded.window(660, cleanup=True).remaining(), 659)
            self.assertEqual(bounded.end, deadline)

    def test_success_records_all_phases_and_confirms_runner_cleanup(self):
        adapter = self.adapter()
        clock = FakeClock()
        snapshots = []

        def create(*args, window):
            self.assertEqual(window.end, 6610)
            clock.advance(200)
            return {"runner_id": "owned", "granted_seconds": 1200}

        def upload(*args, runner_deadline, window):
            self.assertEqual(runner_deadline, 1299)
            self.assertEqual(window.remaining(), 120)
            clock.advance(100)

        def execute(*args, runner_deadline, window, evidence):
            self.assertEqual(runner_deadline, 1299)
            self.assertEqual(window.remaining(), 209)
            evidence.transport_activity()
            clock.advance(100)
            return 0, False

        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.dict(os.environ, {"FRONTIER_MAX_EVIDENCE_BYTES": "8192",
                                        "FRONTIER_VERIFIER_DEADLINE_UNIX_MS": ""}),
            mock.patch.object(adapter.time, "monotonic", side_effect=clock.monotonic),
            mock.patch.object(adapter, "_config", return_value=("url", "verification", "token", "gpu", 7200, 1)),
            mock.patch.object(adapter, "_input_archive"),
            mock.patch.object(adapter, "_create_runner", side_effect=create),
            mock.patch.object(adapter, "_ssh_argv", return_value=["ssh"]),
            mock.patch.object(adapter, "_ssh_environment", return_value={}),
            mock.patch.object(adapter, "_upload_inputs", side_effect=upload),
            mock.patch.object(adapter, "_run_remote_verifier", side_effect=execute),
            mock.patch.object(adapter, "_copy_outputs") as copy,
            mock.patch.object(adapter, "_terminate_runner", return_value=True) as delete,
        ):
            adapter.run(payload=Path(temp)/"payload", dataset=Path(temp)/"dataset", mode="eval",
                        result_path=Path(temp)/"result", evidence_path=Path(temp)/"evidence",
                        record_evidence=snapshots.append)
            copy.assert_called_once()
            delete.assert_called_once()
            self.assertEqual(delete.call_args.args, ("url", "verification", "token", "owned"))
            self.assertEqual(delete.call_args.kwargs["window"].remaining(), 660)
        evidence = snapshots[-1]
        self.assertEqual([e["phase"] for e in evidence["events"]], [
            "configuration", "input_archive", "provisioning", "upload", "remote_execution",
            "output_validation", "delete", "cleanup"])
        self.assertTrue(all(e["status"] == "completed" for e in evidence["events"]))
        remote = evidence["events"][4]
        self.assertTrue(remote["transport_activity_seen"])
        self.assertEqual(remote["transport_outcome"], "zero_exit")
        self.assertEqual(evidence["deadline_basis"], "returned_grant_and_declared_cap")
        self.assertEqual(evidence["evaluator_progress"], "unobserved")
        self.assertEqual(evidence["events"][6]["runner_termination"], "confirmed")
        self.assertFalse(evidence["evidence_write_failed"])


if __name__ == "__main__":
    unittest.main()
