#!/usr/bin/env python3

from __future__ import annotations

# Keep tests isolated from live configuration.
import deployment_test_guard  # noqa: F401  # isort: skip
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "candidates/kimi-code/submit/frontier_submit.py"
)
SPEC = importlib.util.spec_from_file_location("frontier_submit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


class FakeClient:
    def __init__(self, state: str = "passed") -> None:
        self.state = state
        self.payloads = 0
        self.reads = 0

    def request(self, method: str, path: str, **_: object) -> dict[str, object]:
        if path == "/v1/payloads":
            self.payloads += 1
            return {"payload_id": f"payload_{self.payloads:026d}"}
        if path == "/v1/verifier-train":
            return {
                "verification_id": "verification_train",
                "state": "queued",
                "remaining_train_calls": 2,
            }
        if path == "/v1/submissions":
            return {"verification_id": "verification_eval"}
        if method == "GET" and path == "/v1/verifications/verification_train":
            self.reads += 1
            return {
                "verification_id": "verification_train",
                "mode": "train",
                "payload_id": "payload_train",
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
        self.assertEqual(
            client.request.call_args_list[0].kwargs["headers"]["X-Frontier-Purpose"],
            "final",
        )
        self.assertEqual(
            client.request.call_args.args, ("POST", "/v1/submissions/checkpoint")
        )
        self.assertEqual(
            json.loads(client.request.call_args.kwargs["body"]),
            {"payload_id": "payload_checkpoint"},
        )
        self.assertNotIn("verification_id", result)
        self.assertNotIn("raw_score", result)

    def test_store_final_accepts_unlimited_saves(self) -> None:
        client = mock.Mock()
        client.request.side_effect = [
            {"payload_id": "payload_checkpoint"},
            {"remaining_runtime_s": 900, "remaining_checkpoint_submissions": None},
        ]
        with mock.patch.object(submit, "FINAL_RECEIPT", self.receipt):
            result = submit.run_store_final(client, self.archive)
        self.assertEqual(result["state"], "stored")
        self.assertIsNone(result["remaining_checkpoint_submissions"])
        self.assertFalse(self.receipt.exists())

    def test_store_final_rejects_missing_or_invalid_counters(self) -> None:
        for view in (
            {},
            {"remaining_runtime_s": 900},
            {"remaining_runtime_s": True, "remaining_checkpoint_submissions": 99},
            {"remaining_runtime_s": 0, "remaining_checkpoint_submissions": -1},
            {"remaining_runtime_s": 0, "remaining_checkpoint_submissions": 100},
        ):
            client = mock.Mock()
            client.request.side_effect = [{"payload_id": "payload_checkpoint"}, view]
            with self.assertRaises(submit.SubmitError):
                submit.run_store_final(client, self.archive)

    def test_store_final_cli_returns_counters_at_limit(self) -> None:
        client = mock.Mock()
        client.request.side_effect = [
            {"payload_id": "payload_checkpoint"},
            {"remaining_runtime_s": 0, "remaining_checkpoint_submissions": 0},
        ]
        stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        self.addCleanup(stdout.close)
        with (
            mock.patch.object(submit, "Client", return_value=client),
            mock.patch.object(submit.sys, "stdout", stdout),
        ):
            status = submit.main(["store-final", str(self.archive)])
        self.assertEqual(status, 0)
        result = json.loads(stdout.buffer.getvalue())
        self.assertEqual(result["operation"], "store-final")
        self.assertEqual(result["remaining_checkpoint_submissions"], 0)

    def test_train_returns_immediately_without_reading_status(self) -> None:
        client = FakeClient()
        result, status = submit.run_train(client, self.archive)
        self.assertEqual(status, 0)
        self.assertEqual(result["state"], "queued")
        self.assertEqual(result["verification_id"], "verification_train")
        self.assertEqual(client.reads, 0)
        self.assertNotIn("metrics", result)

    def test_status_returns_score_metrics_and_remaining_calls(self) -> None:
        client = FakeClient()
        result, status = submit.run_status(client, "verification_train")
        self.assertEqual(client.reads, 1)
        self.assertEqual(status, 0)
        self.assertEqual(result["raw_score"], 7.5)
        self.assertEqual(result["remaining_train_calls"], 2)
        self.assertEqual(result["metrics"]["case_seconds"], [1.0, 2.0])
        self.assertEqual(
            result["metrics"]["beat_spec_progress"]["case_seconds"]["pass_percent"],
            50.0,
        )

    def test_failed_train_retains_feedback_and_exits_two(self) -> None:
        result, status = submit.run_status(FakeClient("failed"), "verification_train")
        self.assertEqual(status, 2)
        self.assertEqual(result["remaining_train_calls"], 2)
        self.assertIn("beat_spec_progress", result["metrics"])

    def test_final_writes_create_only_receipt(self) -> None:
        with mock.patch.object(submit, "FINAL_RECEIPT", self.receipt):
            result = submit.run_final(FakeClient(), self.archive)
        self.assertEqual(result["state"], "accepted")
        self.assertEqual(self.receipt.stat().st_mode & 0o777, 0o600)
        self.assertEqual(
            json.loads(self.receipt.read_text())["verification_id"], "verification_eval"
        )

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


class FailureFeedbackTests(unittest.TestCase):
    def test_train_forwards_only_public_failure_fields(self):
        client = FakeClient("failed")
        request = client.request

        def with_failure(method, path, **kwargs):
            reply = request(method, path, **kwargs)
            if method == "GET":
                reply["failure"] = {
                    "code": "verifier_gpu_error",
                    "message": "GPU verification failed",
                    "retryable": False,
                    "private_stderr": "PRIVATE",
                }
            return reply

        client.request = with_failure
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "payload.tar"
            with tarfile.open(archive, "w") as output:
                info = tarfile.TarInfo("payload/answer.txt")
                info.size = 7
                output.addfile(info, io.BytesIO(b"payload"))
            result, status = submit.run_status(client, "verification_train")
        self.assertEqual(status, 2)
        self.assertEqual(
            result["failure"],
            {
                "code": "verifier_gpu_error",
                "message": "GPU verification failed",
                "retryable": False,
            },
        )
        self.assertNotIn("PRIVATE", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
