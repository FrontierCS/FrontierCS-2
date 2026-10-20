from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "modal_delivery", ROOT / "runtime/storage-controller/modal_credentials.py"
)
delivery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delivery)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        for key in ("token-id", "token-secret"):
            (self.path / key).write_text("fixture-" + key)
        self.namespace = {
            "metadata": {
                "name": "frontier-run-" + "0" * 26,
                "uid": "namespace-uid",
                "labels": {
                    "frontier.dev/owner": "management",
                    "frontier.dev/resource-kind": "run-namespace",
                    "frontier.dev/run-id": "run_" + "0" * 26,
                    "frontier.dev/attempt-id": "attempt_" + "1" * 26,
                },
            }
        }
        labels = dict(
            self.namespace["metadata"]["labels"],
            **{"frontier.dev/resource-kind": "driver-job"},
        )
        self.environment = [
            {"name": "FRONTIER_DRIVER_BACKENDS", "value": "local-k8s"},
            *[
                {
                    "name": variable,
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": "frontier-modal-credentials",
                            "key": key,
                        }
                    },
                }
                for variable, key in (
                    ("MODAL_TOKEN_ID", "token-id"),
                    ("MODAL_TOKEN_SECRET", "token-secret"),
                )
            ],
        ]
        self.job = {
            "metadata": {"uid": "job-uid", "labels": labels},
            "spec": {
                "template": {
                    "metadata": {"labels": {"frontier.dev/network-role": "driver"}},
                    "spec": {
                        "serviceAccountName": "frontier-driver",
                        "containers": [
                            {
                                "image": "registry/driver@sha256:" + "a" * 64,
                                "env": self.environment,
                            }
                        ],
                    },
                }
            },
        }
        self.stored = None
        self.writes = []

    def api(self, method, path, body=None, **kwargs):
        if path.endswith("/jobs"):
            return {"items": [self.job]}
        if method == "POST":
            self.writes.append(copy.deepcopy(body))
            self.stored = copy.deepcopy(body)
        return self.stored

    def reconcile(self):
        delivery.reconcile_modal_credentials(
            self.api, self.namespace, "frontier-modal-credentials", self.path
        )

    def test_gpu_secret_is_immutable_scoped_and_idempotent(self):
        self.reconcile()
        self.reconcile()
        self.assertEqual(len(self.writes), 1)
        self.assertIs(self.stored["immutable"], True)
        self.assertEqual(
            self.stored["metadata"]["ownerReferences"][0]["uid"], "namespace-uid"
        )
        self.assertEqual(set(self.stored["data"]), {"token-id", "token-secret"})

    def test_cpu_agent_foreign_attempt_and_deleting_namespaces_get_no_secret(self):
        for change in ("cpu", "agent", "foreign", "deleting"):
            with self.subTest(change=change):
                original_job, original_namespace = (
                    copy.deepcopy(self.job),
                    copy.deepcopy(self.namespace),
                )
                if change == "cpu":
                    self.job["spec"]["template"]["spec"]["containers"][0]["env"] = (
                        self.environment[:1]
                    )
                elif change == "agent":
                    self.job["spec"]["template"]["spec"]["serviceAccountName"] = (
                        "frontier-agent"
                    )
                elif change == "foreign":
                    self.job["metadata"]["labels"]["frontier.dev/attempt-id"] = (
                        "attempt_" + "2" * 26
                    )
                else:
                    self.namespace["metadata"]["deletionTimestamp"] = (
                        "2026-09-05T00:00:00Z"
                    )
                self.reconcile()
                self.assertEqual(self.writes, [])
                self.job, self.namespace = original_job, original_namespace

    def test_cleanup_driver_gets_the_same_credential(self):
        self.job["metadata"]["labels"]["frontier.dev/resource-kind"] = (
            "driver-cleanup-job"
        )
        self.reconcile()
        self.assertEqual(len(self.writes), 1)

    def test_wrong_secret_reference_is_rejected_without_writing(self):
        self.environment[1]["valueFrom"]["secretKeyRef"]["name"] = "foreign-secret"
        with self.assertRaisesRegex(RuntimeError, "reference"):
            self.reconcile()
        self.assertEqual(self.writes, [])

    def test_existing_different_secret_is_never_overwritten(self):
        self.reconcile()
        self.stored["metadata"]["ownerReferences"][0]["uid"] = "foreign-namespace"
        with self.assertRaisesRegex(RuntimeError, "immutable identity"):
            self.reconcile()
        self.assertEqual(len(self.writes), 1)
