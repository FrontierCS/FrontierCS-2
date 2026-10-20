#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("frontier_submit.py")
SPEC = importlib.util.spec_from_file_location("frontier_submit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


class FakeClient:
    def __init__(self, state: str = "passed", mode: str = "train") -> None:
        self.state = state
        self.mode = mode
        self.payloads = 0
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, path: str, **_: object) -> dict[str, object]:
        self.calls.append((method, path))
        if path == "/v1/payloads":
            self.payloads += 1
            return {"payload_id": f"payload_{self.payloads:026d}"}
        if path == "/v1/verifier-train":
            return {"verification_id": "verification_train", "state": "queued", "remaining_train_calls": 2}
        if path == "/v1/submissions":
            return {"verification_id": "verification_eval"}
        if method == "GET" and path == "/v1/verifications/verification_train":
            return {
                "verification_id": "verification_train",
                "mode": self.mode,
                "payload_id": "payload_00000000000000000000000001",
                "state": self.state,
                "raw_score": 7.5,
                "remaining_train_calls": 2,
                "metrics": {
                    "case_seconds": [1.0, 2.0],
                    "beat_spec_progress": {
                        "case_seconds": {
                            "passed": 1,
                            "failed": 1,
                            "total": 2,
                            "pass_percent": 50.0,
                            "fail_percent": 50.0,
                        }
                    },
                },
            }
        raise AssertionError((method, path))


class SubmitTests(unittest.TestCase):
    def test_response_larger_than_one_mebibyte_and_idempotent_transport_retry(self) -> None:
        data = json.dumps({"detail": "x" * (2 * 1024 * 1024)}).encode()
        client = submit.Client.__new__(submit.Client)
        client.host, client.port, client.auth = "localhost", 8080, "test-token"
        response = mock.Mock(status=200)
        response.read.side_effect = lambda n: data[:n]
        connection = mock.Mock()
        connection.getresponse.return_value = response
        connection.request.side_effect = [OSError("temporary")] * 5 + [None]
        with mock.patch.object(submit.http.client, "HTTPConnection", return_value=connection), mock.patch.object(submit.time, "sleep"):
            self.assertEqual(client.request("POST", "/v1/payloads", body=b"payload", headers={"Idempotency-Key": "same-key"})["detail"], "x" * (2 * 1024 * 1024))
        self.assertEqual(connection.request.call_count, 6)
        self.assertTrue(all(call == connection.request.call_args_list[0] for call in connection.request.call_args_list))

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.token = root / "token"
        self.token.write_text("run-token", encoding="utf-8")
        self.archive = root / "answer.tar"
        with tarfile.open(self.archive, "w") as archive:
            data = b"answer\n"
            info = tarfile.TarInfo("payload/answer.txt")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        self.receipt = root / "receipt.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_store_final_uses_final_payload_and_has_no_final_receipt(self) -> None:
        client = mock.Mock()
        client.request.side_effect = [
            {"payload_id": "payload_checkpoint"},
            {"remaining_runtime_s": 900, "remaining_checkpoint_submissions": 99},
        ]
        with mock.patch.object(submit, "FINAL_RECEIPT", self.receipt):
            result = submit.run_store_final(client, self.archive)
        self.assertEqual(result["state"], "stored")
        self.assertEqual(result["remaining_runtime_s"], 900)
        self.assertEqual(result["remaining_checkpoint_submissions"], 99)
        self.assertFalse(self.receipt.exists())
        self.assertEqual(client.request.call_args_list[0].kwargs["headers"]["X-Frontier-Purpose"], "final")
        self.assertEqual(client.request.call_args.args, ("POST", "/v1/submissions/checkpoint"))
        self.assertEqual(json.loads(client.request.call_args.kwargs["body"]), {"payload_id": "payload_checkpoint"})
        self.assertNotIn("verification_id", result)
        self.assertNotIn("raw_score", result)

    def test_store_final_rejects_missing_or_invalid_counters(self) -> None:
        for view in ({}, {"remaining_runtime_s": 900},
                     {"remaining_runtime_s": True, "remaining_checkpoint_submissions": 99},
                     {"remaining_runtime_s": 0, "remaining_checkpoint_submissions": -1},
                     {"remaining_runtime_s": 0, "remaining_checkpoint_submissions": True}):
            client = mock.Mock()
            client.request.side_effect = [{"payload_id": "payload_checkpoint"}, view]
            with self.assertRaises(submit.SubmitError):
                submit.run_store_final(client, self.archive)

    def test_store_final_accepts_explicit_unlimited_allowance(self) -> None:
        client = mock.Mock()
        client.request.side_effect = [{"payload_id": "payload_checkpoint"},
                                     {"remaining_runtime_s": 900, "remaining_checkpoint_submissions": None}]
        result = submit.run_store_final(client, self.archive)
        self.assertEqual(result["remaining_runtime_s"], 900)
        self.assertIsNone(result["remaining_checkpoint_submissions"])

    def test_store_final_accepts_driver_reported_allowance_above_old_limit(self) -> None:
        client = mock.Mock()
        client.request.side_effect = [{"payload_id": "payload_checkpoint"},
                                     {"remaining_runtime_s": 900, "remaining_checkpoint_submissions": 1024}]
        result = submit.run_store_final(client, self.archive)
        self.assertEqual(result["remaining_checkpoint_submissions"], 1024)

    def test_store_final_cli_returns_counters_at_limit(self) -> None:
        client = mock.Mock()
        client.request.side_effect = [{"payload_id": "payload_checkpoint"},
                                     {"remaining_runtime_s": 0, "remaining_checkpoint_submissions": 0}]
        stdout = io.TextIOWrapper(io.BytesIO())
        with mock.patch.object(submit, "Client", return_value=client), mock.patch.object(submit.sys, "stdout", stdout):
            status = submit.main(["store-final", str(self.archive)])
        self.assertEqual(status, 0)
        result = json.loads(stdout.buffer.getvalue())
        self.assertEqual(result["operation"], "store-final")
        self.assertEqual(result["remaining_checkpoint_submissions"], 0)

    def test_status_returns_score_metrics_and_remaining_calls(self) -> None:
        with mock.patch.object(submit.time, "sleep"):
            result, status = submit.run_status(FakeClient(), "verification_train")
        self.assertEqual(status, 0)
        self.assertEqual(result["raw_score"], 7.5)
        self.assertEqual(result["remaining_train_calls"], 2)
        self.assertEqual(result["metrics"]["case_seconds"], [1.0, 2.0])
        self.assertEqual(
            result["metrics"]["beat_spec_progress"]["case_seconds"]["pass_percent"],
            50.0,
        )

    def test_failed_status_retains_feedback_and_exits_two(self) -> None:
        with mock.patch.object(submit.time, "sleep"):
            result, status = submit.run_status(FakeClient("failed"), "verification_train")
        self.assertEqual(status, 2)
        self.assertEqual(result["remaining_train_calls"], 2)
        self.assertIn("beat_spec_progress", result["metrics"])

    def test_final_writes_create_only_receipt(self) -> None:
        with mock.patch.object(submit, "FINAL_RECEIPT", self.receipt):
            result = submit.run_final(FakeClient(), self.archive)
        self.assertEqual(result["state"], "accepted")
        self.assertEqual(self.receipt.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(self.receipt.read_text())["verification_id"], "verification_eval")

    def test_rejects_escaping_tar_member(self) -> None:
        unsafe = Path(self.temp.name) / "unsafe.tar"
        with tarfile.open(unsafe, "w") as archive:
            info = tarfile.TarInfo("../escape")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        with self.assertRaisesRegex(submit.SubmitError, "unsafe archive member"):
            submit.validate_tar(unsafe)

    def test_rejects_duplicate_tar_member(self) -> None:
        duplicate = Path(self.temp.name) / "duplicate.tar"
        with tarfile.open(duplicate, "w") as archive:
            for value in (b"a", b"b"):
                info = tarfile.TarInfo("payload/answer.txt")
                info.size = 1
                archive.addfile(info, io.BytesIO(value))
        with self.assertRaisesRegex(submit.SubmitError, "duplicate archive member"):
            submit.validate_tar(duplicate)


class AsynchronousTrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.archive = Path(self.temp.name) / "answer.tar"
        with tarfile.open(self.archive, "w") as archive:
            info = tarfile.TarInfo("payload/answer.txt")
            info.size = 7
            archive.addfile(info, io.BytesIO(b"answer\n"))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_cli(self, client: FakeClient, *argv: str) -> tuple[int, dict[str, object]]:
        stdout = io.TextIOWrapper(io.BytesIO())
        with mock.patch.object(submit, "Client", return_value=client), \
                mock.patch.object(submit.sys, "stdout", stdout), mock.patch.object(submit.time, "sleep"):
            status = submit.main(list(argv))
        return status, json.loads(stdout.buffer.getvalue())

    def test_train_returns_the_accepted_verification_without_polling(self) -> None:
        client = FakeClient()
        status, result = self.run_cli(client, "train", str(self.archive))
        self.assertEqual(status, 0)
        self.assertEqual(result["state"], "queued")
        self.assertEqual(result["verification_id"], "verification_train")
        self.assertEqual(result["remaining_train_calls"], 2)
        self.assertNotIn("metrics", result)
        self.assertNotIn("raw_score", result)
        self.assertNotIn(("GET", "/v1/verifications/verification_train"), client.calls)

    def test_train_never_polls_or_sleeps(self) -> None:
        client = FakeClient("running")
        with mock.patch.object(submit.time, "sleep", side_effect=AssertionError("train must not wait")), \
                mock.patch.object(submit.time, "monotonic", side_effect=AssertionError("train has no poll deadline")):
            result, status = submit.run_train(client, self.archive)
        self.assertEqual((status, result["state"]), (0, "queued"))
        self.assertEqual([method for method, _ in client.calls], ["POST", "POST"])

    def test_status_reads_one_train_verification_in_the_train_shape(self) -> None:
        for state, expected in (("passed", 0), ("failed", 2), ("running", 0), ("queued", 0)):
            client = FakeClient(state)
            status, result = self.run_cli(client, "status", "verification_train")
            self.assertEqual(status, expected)
            self.assertEqual(result["operation"], "train")
            self.assertEqual(result["state"], state)
            self.assertEqual(result["payload_id"], "payload_00000000000000000000000001")
            self.assertEqual(result["remaining_train_calls"], 2)
            self.assertEqual(client.calls, [("GET", "/v1/verifications/verification_train")])
            if state in ("queued", "running"):
                self.assertNotIn("raw_score", result)
            else:
                self.assertEqual(result["raw_score"], 7.5)
                self.assertIn("beat_spec_progress", result["metrics"])

    def test_status_rejects_invalid_ids_and_eval_verifications(self) -> None:
        for bad in ("../payloads", "verification_a/b", "payload_00000000000000000000000001", "verification_"):
            client = FakeClient()
            with self.assertRaisesRegex(submit.SubmitError, "invalid verification id"):
                submit.run_status(client, bad)
            self.assertEqual(client.calls, [])
        with self.assertRaisesRegex(submit.SubmitError, "train verifications only"):
            submit.run_status(FakeClient(mode="eval"), "verification_train")
        mismatched = mock.Mock()
        mismatched.request.return_value = {"verification_id": "verification_train", "mode": "train",
                                           "payload_id": "payload_x", "state": "passed", "remaining_train_calls": 2}
        with self.assertRaisesRegex(submit.SubmitError, "train verifications only"):
            submit.run_status(mismatched, "verification_other")

    def test_unknown_state_is_rejected(self) -> None:
        with self.assertRaisesRegex(submit.SubmitError, "invalid state"):
            submit.train_result({"state": "lost"}, payload_id="payload_x", verification_id="verification_x", remaining=0)


class FailureFeedbackTests(unittest.TestCase):
    def test_train_forwards_only_public_failure_fields(self):
        client = FakeClient("failed")
        request = client.request
        def with_failure(method, path, **kwargs):
            reply = request(method, path, **kwargs)
            if method == "GET":
                reply["failure"] = {"code": "verifier_gpu_error", "message": "GPU verification failed", "retryable": False, "private_stderr": "PRIVATE"}
            return reply
        client.request = with_failure
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "payload.tar"
            with tarfile.open(archive, "w") as output:
                info = tarfile.TarInfo("payload/answer.txt")
                info.size = 7
                output.addfile(info, io.BytesIO(b"payload"))
            with mock.patch.object(submit.time, "sleep"):
                result, status = submit.run_status(client, "verification_train")
        self.assertEqual(status, 2)
        self.assertEqual(result["failure"], {"code": "verifier_gpu_error", "message": "GPU verification failed", "retryable": False})
        self.assertNotIn("PRIVATE", json.dumps(result))

if __name__ == "__main__":
    unittest.main()
