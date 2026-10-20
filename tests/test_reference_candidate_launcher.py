from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import contextlib
import importlib.util
import io
import json
import os
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATH = (
    ROOT
    / "tasks/frontier-cs-2.0-demo/shared/reference-candidate/fcs2_reference_candidate.py"
)
SPEC = importlib.util.spec_from_file_location("reference_candidate_launcher", PATH)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


class ReferenceLauncherTests(unittest.TestCase):
    def response(self, **updates):
        value = {
            "verification_id": "verification_train",
            "mode": "train",
            "state": "passed",
            "raw_score": 12.5,
            "metrics": {
                "latency": 4.0,
                "beat_spec_progress": {
                    "latency": {
                        "passed": 0,
                        "failed": 2,
                        "total": 2,
                        "pass_percent": 0.0,
                        "fail_percent": 100.0,
                    }
                },
            },
        }
        value.update(updates)
        return value

    def test_event_contains_only_approved_train_response_fields(self):
        source = self.response(
            token="must-not-log",
            payload="private-reference",
            config={"secret": "hidden"},
        )
        event = LAUNCHER._train_result_event(source, "verification_train")
        self.assertEqual(set(event), {"event", "verification_id", "score", "metrics"})
        self.assertEqual(event["event"], "reference_train_result")
        self.assertEqual(event["metrics"], source["metrics"])
        source["metrics"]["latency"] = 99
        self.assertEqual(event["metrics"]["latency"], 4.0)
        self.assertNotIn("must-not-log", json.dumps(event))
        self.assertNotIn("private-reference", json.dumps(event))

    def test_authored_driver_deadline_owns_long_train_poll(self):
        client = mock.Mock()
        client.verification.side_effect = [
            {"state": "queued"},
            {"state": "running"},
            self.response(),
        ]
        with (
            mock.patch.dict(os.environ, {"FRONTIER_TRAIN_POLL_TIMEOUT_SECONDS": "1"}),
            mock.patch.object(
                LAUNCHER.time,
                "monotonic",
                side_effect=AssertionError("private deadline"),
            ),
            mock.patch.object(LAUNCHER.time, "sleep") as sleep,
        ):
            event = LAUNCHER._wait_for_train(client, "verification_train")
        self.assertEqual(event["score"], 12.5)
        self.assertEqual(sleep.call_args_list, [mock.call(2), mock.call(2)])
        self.assertEqual(client.verification.call_count, 3)

    def test_rejects_wrong_identity_mode_nonfinite_scores_and_metrics(self):
        for updates in (
            {"verification_id": "other"},
            {"mode": "eval"},
            {"raw_score": True},
            {"raw_score": float("nan")},
            {"raw_score": float("inf")},
            {"raw_score": -1},
            {"metrics": []},
            {"metrics": {"value": float("nan")}},
        ):
            with (
                self.subTest(updates=list(updates)),
                self.assertRaises(LAUNCHER.CandidateError),
            ):
                LAUNCHER._train_result_event(
                    self.response(**updates), "verification_train"
                )

    def test_http_error_does_not_echo_private_response_body(self):
        response = mock.Mock(status=500)
        response.read.return_value = b'{"token":"must-not-log","reference":"private"}'
        with self.assertRaises(LAUNCHER.CandidateError) as caught:
            LAUNCHER.DriverClient._decode(response, {200})
        self.assertEqual(str(caught.exception), "Driver returned HTTP 500")

    def test_main_logs_train_once_and_submits_final_once(self):
        client = mock.Mock()
        client.upload.side_effect = ["payload_train", "payload_final"]
        client.post_json.side_effect = [
            {"verification_id": "verification_train"},
            {"verification_id": "verification_final"},
        ]
        client.verification.return_value = self.response()
        output = io.StringIO()
        with (
            mock.patch.object(
                LAUNCHER, "_package_reference", return_value=(100, "sha256:test")
            ),
            mock.patch.object(LAUNCHER, "_required", return_value="http://driver"),
            mock.patch.object(LAUNCHER, "_driver_token", return_value="secret-token"),
            mock.patch.object(LAUNCHER, "DriverClient", return_value=client),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(LAUNCHER.main(), 0)
        lines = output.getvalue().splitlines()
        self.assertEqual(json.loads(lines[0])["metrics"], self.response()["metrics"])
        self.assertEqual(len([line for line in lines if line.startswith("{")]), 1)
        self.assertEqual(client.post_json.call_args_list[-1].args[0], "/v1/submissions")
        self.assertEqual(client.post_json.call_count, 2)
        self.assertNotIn("secret-token", output.getvalue())


if __name__ == "__main__":
    unittest.main()
