"""Run inside the pinned LiteLLM image; no provider or cloud credentials needed."""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

os.environ.setdefault("FRONTIER_MODEL_AUDIT_ROOT", "/tmp/preview-audit-test/artifacts/test-bucket")
from preview_audit import PreviewAuditLogger
from litellm.integrations.s3_v2 import S3Logger
from litellm.responses.utils import ResponsesAPIRequestUtils


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        os.environ["FRONTIER_MODEL_AUDIT_ROOT"] = self.directory.name + "/artifacts/test-bucket"
        self.logger = PreviewAuditLogger()

    def element(self, key="runs/run_test/attempt_test/2026-10-01/call.json", payload=None):
        return SimpleNamespace(s3_object_key=key, payload=payload or {"id": "call", "total_tokens": 42})

    def test_atomic_immutable_bytes_and_retry(self):
        element = self.element()
        self.logger.persist(element)
        self.logger.persist(element)
        path = self.logger.root / element.s3_object_key
        self.assertEqual(json.loads(path.read_bytes()), element.payload)
        with self.assertRaises(ValueError):
            self.logger.persist(self.element(payload={"id": "changed"}))
        self.assertEqual(list(self.logger.spool.iterdir()), [])
        self.assertEqual(len(list(self.logger.root.rglob("*.json"))), 1)

    def test_rejects_traversal_and_symlink(self):
        for key in ["/tmp/leak", "runs/../escape/a.json", "runs/a/b/../../escape", "runs/a//x"]:
            with self.assertRaises(ValueError):
                self.logger.persist(self.element(key))
        outside = Path(self.directory.name) / "outside"
        outside.mkdir()
        (self.logger.root / "runs").mkdir(parents=True)
        (self.logger.root / "runs/a").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.logger.persist(self.element("runs/a/b/c.json"))

    def test_upstream_payload_naming_and_success_failure_callbacks(self):
        self.assertIs(PreviewAuditLogger.create_s3_batch_logging_element, S3Logger.create_s3_batch_logging_element)
        now = datetime.now(timezone.utc)
        payload = {"id": "call-id", "trace_id": "trace-id", "metadata": {
            "user_api_key_team_alias": "run_01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "user_api_key_alias": "attempt_01ARZ3NDEKTSV4RRFFQ69G5FAV"},
            "status": "success", "total_tokens": 42, "response_cost": 0.01}
        kwargs = {"standard_logging_object": payload}
        expected = self.logger.create_s3_batch_logging_element(now, payload)
        asyncio.run(self.logger.async_log_success_event(kwargs, None, now, now))
        self.assertEqual(json.loads((self.logger.root / expected.s3_object_key).read_bytes()), payload)
        payload["id"] = "failed-call"
        payload["status"] = "failure"
        expected = self.logger.create_s3_batch_logging_element(now, payload)
        asyncio.run(self.logger.async_log_failure_event(kwargs, None, now, now))
        self.assertEqual(json.loads((self.logger.root / expected.s3_object_key).read_bytes())["status"], "failure")

    def test_responses_callback_preserves_ids_under_safe_stable_filename(self):
        now = datetime.now(timezone.utc)
        response_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            "openai", "00000000-0000-4000-8000-000000000000", "resp_provider_test1")
        self.assertIn("=", response_id)
        payload = {"id": response_id, "trace_id": "original-driver-call-id", "metadata": {
            "user_api_key_team_alias": "run_test", "user_api_key_alias": "attempt_test"},
            "status": "success", "total_tokens": 42, "response_cost": 0.01}
        expected = self.logger.create_s3_batch_logging_element(now, payload)
        original = Path(expected.s3_object_key)
        filename = "sha256-" + hashlib.sha256(original.name.encode()).hexdigest() + ".json"
        destination = self.logger.root / original.parent / filename
        for _ in range(2):
            asyncio.run(self.logger.async_log_success_event(
                {"standard_logging_object": payload}, None, now, now))
        self.assertEqual(json.loads(destination.read_bytes()), payload)
        self.assertEqual(list(self.logger.root.rglob("*.json")), [destination])
        self.assertEqual(list(self.logger.spool.iterdir()), [])
        with self.assertRaises(ValueError):
            self.logger.persist(self.element("runs/bad=prefix/attempt/call.json"))


if __name__ == "__main__":
    unittest.main()
