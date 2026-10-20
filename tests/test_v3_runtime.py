"""V3 runtime gates and mutable operation state; no live state or cluster access."""

from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import http.client
import io
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preview import acceptance, cli, infra, publication
from preview import config as site_config
from preview import images as image_ops
from test_frontier_deploy import ROOT, Fixture
from test_frontier_deploy import frontier_deploy as deploy

BASE = "http://127.0.0.1:58080"
CAPACITY_PATH = "/api/v1/infrastructure/scheduler-capacity"
OPEN_HEADERS = {
    "x-management-driver-mode": "kubernetes",
    "x-management-accepting-evaluations": "true",
}


def incident_snapshot():
    return deploy.load_json(ROOT / "tests/fixtures/evidence/v3-runtime-incident-2026-09-12.json")[
        "observed"
    ]["scheduler_capacity_snapshot"]


def available_snapshot():
    snapshot = incident_snapshot()
    snapshot.update(configured_cap=4, available_slots=1)
    return snapshot


class RuntimeFixture(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="frontier-v3-test-")
        self.addCleanup(temp.cleanup)
        self.state = Path(temp.name)
        self.config = deploy.validate_config(
            deploy.load_json(ROOT / "config/site.example.json")
        )
        self.images = Fixture.images(self.config)
        self.kubeconfig = self.state / "kubeconfig"
        self.kubeconfig.write_text("test kubeconfig")
        self.output = io.StringIO()
        self.patch(
            "state_directory",
            modules=(site_config, infra, acceptance, cli, publication, image_ops),
            return_value=self.state,
        )
        self.patch(
            "executable",
            modules=(infra, cli, acceptance, publication, image_ops),
            side_effect=lambda name: name,
        )
        self.forward = self.patch(
            "port_forward", modules=(infra, cli, acceptance, publication, image_ops)
        )
        self.forward.return_value.__enter__.return_value = 58080
        self.enterContext(mock.patch("sys.stdout", self.output))
        # Tests must opt in to every HTTP / subprocess interaction they use.
        self.patch(
            "run_command",
            modules=(infra, cli, acceptance, publication, image_ops),
            side_effect=AssertionError("unexpected command"),
        )
        self.enterContext(
            mock.patch.object(
                infra.urllib_request,
                "urlopen",
                side_effect=AssertionError("unexpected HTTP"),
            )
        )

    def patch(self, name, *, modules=(acceptance,), **kwargs):
        replacement = self.enterContext(mock.patch.object(modules[0], name, **kwargs))
        for module in modules[1:]:
            self.enterContext(mock.patch.object(module, name, replacement))
        return replacement


class CapacityTests(RuntimeFixture):
    def preflight(self, snapshot, **kwargs):
        self.patch(
            "api_json_with_headers", modules=(infra,), return_value=({}, OPEN_HEADERS)
        )
        self.patch("api_json", modules=(infra,), return_value=snapshot)
        return deploy.runtime_admission_preflight(self.config, BASE, **kwargs)

    def test_incident_snapshot_blocked_reservations_fail_before_submission(self):
        snapshot = incident_snapshot()
        with self.assertRaisesRegex(
            deploy.DeployError,
            "raise execution.max_active_drivers above the retained count",
        ) as error:
            self.preflight(snapshot)
        self.assertIn(
            "historical-GC recovery is unsupported for these records at this platform pin",
            str(error.exception),
        )
        self.assertNotIn("wait or raise the cap", str(error.exception))
        for blocker in snapshot["retained_cleanup_blockers"]:
            self.assertIn(blocker["attempt_id"], str(error.exception))
        self.assertIn(
            "capacity: cap=1 occupied=3 available=0 queued=2 active=0 cleanups pending/applying/running=0 blocked=3 released=0",
            self.output.getvalue(),
        )

    def test_temporary_occupancy_warns_and_allows_smoke_to_queue(self):
        snapshot = incident_snapshot()
        snapshot.update(
            configured_cap=6, occupied_slots=6, retained_cleanup_blockers=[]
        )
        snapshot["counts"] = {
            "run_counts": {"QUEUED": 9, "STARTING": 1, "RUNNING": 1, "FINALIZING": 1},
            "cleanup_counts": {"PENDING": 1, "APPLYING": 1, "RUNNING": 1, "BLOCKED": 0},
        }
        self.assertEqual(self.preflight(snapshot), snapshot)
        self.assertIn(
            "capacity temporarily exhausted: 3 active Runs, 3 progressing cleanups, 0 retained; the smoke will queue",
            self.output.getvalue(),
        )
        self.assertNotIn("historical-GC", self.output.getvalue())
        # Seed check retains its informational diagnostic and return value.
        self.assertEqual(self.preflight(snapshot, informational=True), {})
        self.assertIn(
            "runtime preflight (informational): no scheduler capacity: temporary occupancy",
            self.output.getvalue(),
        )

    def test_mixed_occupancy_below_retained_cap_allows_smoke_to_queue(self):
        snapshot = incident_snapshot()
        snapshot["counts"]["run_counts"]["RUNNING"] = 1
        snapshot.update(configured_cap=4, occupied_slots=4)
        self.assertEqual(self.preflight(snapshot), snapshot)
        self.assertIn(
            "capacity temporarily exhausted: 1 active Runs, 0 progressing cleanups, 3 retained; the smoke will queue",
            self.output.getvalue(),
        )

    def test_retained_reservations_at_or_above_cap_fail_even_with_active_runs(self):
        for cap in (1, 3):
            with self.subTest(cap=cap):
                snapshot = incident_snapshot()
                snapshot["counts"]["run_counts"]["RUNNING"] = 1
                snapshot.update(configured_cap=cap, occupied_slots=4)
                with self.assertRaisesRegex(
                    deploy.DeployError, "retained BLOCKED reservations=3"
                ) as error:
                    self.preflight(snapshot)
                self.assertNotIn("wait or raise the cap", str(error.exception))

    def test_blocked_count_uses_counts_and_diagnostics_are_null_safe_and_bounded(self):
        snapshot = incident_snapshot()
        snapshot["counts"]["cleanup_counts"]["BLOCKED"] = 123
        snapshot["blockers_truncated"] = True
        snapshot["retained_cleanup_blockers"] = [
            {
                "attempt_id": f"attempt_{n:03}",
                "cleanup_job_state": "PENDING",
                "diagnostics": {
                    "cleanup_job_last_error": None if n % 2 else {"known_message": None}
                },
            }
            for n in range(12)
        ]
        with self.assertRaises(deploy.DeployError) as error:
            self.preflight(snapshot)
        detail = str(error.exception)
        self.assertIn("reservations=123", detail)
        self.assertIn("attempt_009", detail)
        self.assertNotIn("attempt_010", detail)
        self.assertIn("error=unknown", detail)
        self.assertIn("showing 10 of 12", detail)
        self.assertIn("blockers_truncated=true", detail)

    def test_headroom_allows_queued_work_and_retained_blockers(self):
        snapshot = available_snapshot()
        snapshot["released_historical_cleanups"] = 2
        self.assertEqual(self.preflight(snapshot), snapshot)
        self.assertIn("available=1 queued=2", self.output.getvalue())
        self.assertIn("released=2", self.output.getvalue())

    def test_closed_or_wrong_mode_or_missing_headers_fail_before_capacity(self):
        for headers in (
            {},
            {**OPEN_HEADERS, "x-management-driver-mode": "simulator"},
            {**OPEN_HEADERS, "x-management-accepting-evaluations": "false"},
            {**OPEN_HEADERS, "x-management-accepting-evaluations": "TRUE"},
        ):
            with (
                self.subTest(headers=headers),
                mock.patch.object(
                    infra, "api_json_with_headers", return_value=({}, headers)
                ),
                mock.patch.object(infra, "api_json") as api,
            ):
                with self.assertRaisesRegex(
                    deploy.DeployError, "runtime admission requires"
                ):
                    deploy.runtime_admission_preflight(self.config, BASE)
                api.assert_not_called()

    def test_unavailable_capacity_only_continues_in_informational_mode(self):
        self.patch(
            "api_json_with_headers", modules=(infra,), return_value=({}, OPEN_HEADERS)
        )
        for status in (404, 503):
            with (
                self.subTest(status=status),
                mock.patch.object(
                    infra,
                    "api_json",
                    side_effect=deploy.ManagementHTTPError(
                        status, CAPACITY_PATH, "unavailable"
                    ),
                ),
            ):
                with self.assertRaises(deploy.ManagementHTTPError):
                    deploy.runtime_admission_preflight(self.config, BASE)
                self.assertEqual(
                    deploy.runtime_admission_preflight(
                        self.config, BASE, informational=True
                    ),
                    {},
                )
                self.assertIn(
                    "capacity endpoint unavailable (pre-V3 runtime or non-kubernetes mode)",
                    self.output.getvalue(),
                )

    def test_informational_mode_still_reads_capacity_after_overview_error(self):
        self.patch(
            "api_json_with_headers",
            modules=(infra,),
            side_effect=deploy.DeployError("overview unavailable"),
        )
        api = self.patch(
            "api_json", modules=(infra,), return_value=available_snapshot()
        )
        deploy.runtime_admission_preflight(self.config, BASE, informational=True)
        api.assert_called_once_with(BASE, "GET", CAPACITY_PATH)
        self.assertIn("overview unavailable", self.output.getvalue())
        self.assertIn("capacity: cap=4", self.output.getvalue())

    def test_informational_blockers_and_bad_capacity_never_fail(self):
        for snapshot in (incident_snapshot(), {}, {"counts": []}):
            with self.subTest(snapshot=snapshot):
                self.assertEqual(self.preflight(snapshot, informational=True), {})
        self.assertIn("retained BLOCKED reservations=3", self.output.getvalue())

    def test_header_helper_preserves_normalized_headers_and_payload(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"healthy":true}'
        response.headers = {
            "X-Management-Driver-Mode": "kubernetes",
            "X-Management-Accepting-Evaluations": "true",
        }
        with mock.patch.object(infra.urllib_request, "urlopen", return_value=response):
            self.assertEqual(
                deploy.api_json_with_headers(BASE, "GET", "/api/v1/overview"),
                ({"healthy": True}, OPEN_HEADERS),
            )
            self.assertEqual(
                deploy.api_json(BASE, "GET", "/api/v1/overview"), {"healthy": True}
            )

    def test_header_helper_keeps_http_status_and_wraps_transport_errors(self):
        for status in (404, 503):
            error = HTTPError(
                BASE + CAPACITY_PATH,
                status,
                "unavailable",
                {},
                io.BytesIO(b"not configured"),
            )
            with mock.patch.object(infra.urllib_request, "urlopen", side_effect=error):
                with self.assertRaises(deploy.ManagementHTTPError) as caught:
                    deploy.api_json(BASE, "GET", CAPACITY_PATH)
                self.assertEqual(caught.exception.status, status)
        for error in (
            URLError("connection refused"),
            http.client.IncompleteRead(b"partial", 100),
            http.client.HTTPException("bad response"),
            ConnectionError("connection reset"),
            TimeoutError("timeout"),
            socket.timeout("socket timeout"),
        ):
            for phase in ("open", "read", "close"):
                with self.subTest(error=type(error).__name__, phase=phase):
                    response = mock.MagicMock()
                    response.__enter__.return_value = response
                    response.read.return_value = b"{}"
                    if phase == "read":
                        response.read.side_effect = error
                    elif phase == "close":
                        response.__exit__.side_effect = error
                    with mock.patch.object(
                        infra.urllib_request,
                        "urlopen",
                        side_effect=error if phase == "open" else None,
                        return_value=response,
                    ):
                        with self.assertRaisesRegex(
                            deploy.DeployError, "request failed"
                        ):
                            deploy.api_json_with_headers(BASE, "GET", CAPACITY_PATH)

    def test_header_helper_wraps_invalid_json_and_invalid_encoding(self):
        for payload in (b"{invalid", b"\xff"):
            with self.subTest(payload=payload):
                response = mock.MagicMock()
                response.__enter__.return_value = response
                response.read.return_value = payload
                with mock.patch.object(
                    infra.urllib_request, "urlopen", return_value=response
                ):
                    with self.assertRaisesRegex(
                        deploy.DeployError, "Management returned invalid JSON"
                    ):
                        deploy.api_json_with_headers(BASE, "GET", CAPACITY_PATH)

    def test_unreadable_http_error_body_preserves_status(self):
        for status in (404, 503):
            with self.subTest(status=status):
                body = mock.Mock()
                body.read.side_effect = http.client.IncompleteRead(
                    b"private payload", 100
                )
                error = HTTPError(BASE + CAPACITY_PATH, status, "unavailable", {}, body)
                with mock.patch.object(
                    infra.urllib_request, "urlopen", side_effect=error
                ):
                    with self.assertRaises(deploy.ManagementHTTPError) as caught:
                        deploy.api_json_with_headers(BASE, "GET", CAPACITY_PATH)
                self.assertEqual(caught.exception.status, status)
                self.assertIn(
                    "error body unreadable (IncompleteRead)", str(caught.exception)
                )
                self.assertNotIn("private payload", str(caught.exception))
                body.close.assert_called_once()

    def test_http_error_close_failure_is_a_deploy_error(self):
        body = mock.Mock()
        body.read.return_value = b"unavailable"
        body.close.side_effect = ConnectionError("connection reset")
        error = HTTPError(BASE + CAPACITY_PATH, 503, "unavailable", {}, body)
        with mock.patch.object(infra.urllib_request, "urlopen", side_effect=error):
            with self.assertRaisesRegex(
                deploy.DeployError, "Management request failed"
            ):
                deploy.api_json_with_headers(BASE, "GET", CAPACITY_PATH)


class AdmissionObjectsTests(RuntimeFixture):
    def test_kubernetes_objects_equal_pinned_platform_and_simulator_is_unchanged(self):
        fixture = deploy.load_json(
            ROOT / "tests/fixtures/evidence/v3-admission-control-0cc869e9.json"
        )["objects"]
        maps = [item for item in fixture if item["kind"] == "ConfigMap"]
        static = [item for item in fixture if item["kind"] != "ConfigMap"]
        self.assertEqual(deploy.admission_control_maps(), maps)
        self.assertEqual(deploy.admission_control_rbac(), static)
        self.assertFalse(maps[0]["data"]["command.json"].endswith("\n"))
        ensure_maps = self.patch("ensure_admission_control_maps", modules=(cli,))
        for simulator in (True, False):
            rendered = deploy.render_runtime_manifest(
                self.config,
                Fixture.outputs(self.config),
                self.images,
                ["172.20.0.1/32"],
                simulator_bootstrap=simulator,
            )
            items = json.loads(rendered)["items"]
            names = {item["metadata"]["name"] for item in items}
            self.assertNotIn(deploy.ADMISSION_COMMAND_MAP, names)
            self.assertNotIn(deploy.ADMISSION_RECEIPT_MAP, names)
            for item in static:
                self.assertEqual(item in items, not simulator)
            if simulator:
                self.assertEqual(
                    rendered,
                    (
                        ROOT / "tests/fixtures/golden/cpu/runtime-simulator-true.json"
                    ).read_bytes(),
                )
        ensure_maps.assert_not_called()

    def test_missing_maps_are_created_from_json_stdin_without_apply_annotations(self):
        calls = []

        def command(args, **kwargs):
            calls.append(args)
            if "get" in args:
                self.assertIn("--ignore-not-found", args)
                return ""  # kubectl converts only 404 to empty stdout
            self.assertEqual(
                args, ["kubectl", "-n", "frontier-system", "create", "-f", "-"]
            )
            document = json.loads(kwargs["input_text"])
            self.assertNotIn("annotations", document["metadata"])
            self.assertIn(document, deploy.admission_control_maps())
            return "created"

        self.patch("run_command", modules=(infra,), side_effect=command)
        deploy.ensure_admission_control_maps(self.kubeconfig)
        self.assertEqual(len(calls), 4)

    def test_existing_operator_command_and_management_receipt_are_untouched(self):
        command = self.patch(
            "run_command",
            modules=(infra,),
            return_value='{"data":{"operator":"modified"}}',
        )
        deploy.ensure_admission_control_maps(self.kubeconfig)
        self.assertEqual(command.call_count, 2)
        self.assertTrue(all("get" in call.args[0] for call in command.call_args_list))

    def test_racing_create_already_exists_is_success(self):
        command = self.patch(
            "run_command",
            modules=(infra,),
            side_effect=[
                "",
                deploy.DeployError(
                    "Error from server (AlreadyExists): configmaps already exists"
                ),
                "{}",
            ],
        )
        deploy.ensure_admission_control_maps(self.kubeconfig)
        self.assertEqual(command.call_count, 3)

    def test_other_get_and_create_errors_are_fatal(self):
        for results in (
            [deploy.DeployError("Error from server (Forbidden): denied")],
            ["", deploy.DeployError("Error from server (Forbidden): denied")],
            [deploy.DeployError("Unable to connect to server")],
        ):
            with (
                self.subTest(results=results),
                mock.patch.object(infra, "run_command", side_effect=results) as command,
            ):
                with self.assertRaises(deploy.DeployError):
                    deploy.ensure_admission_control_maps(self.kubeconfig)
                self.assertEqual(command.call_count, len(results))

    def test_run_command_supplies_json_input_without_logging_payload(self):
        # Exercise the real subprocess adapter separately from Kubernetes mocks.
        with mock.patch.object(
            infra.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, "ok", ""),
        ) as process:
            self.real_run_command(
                ["kubectl", "create", "-f", "-"],
                input_text='{"fixture":"body"}',
                capture=True,
            )
        self.assertEqual(process.call_args.kwargs["input"], '{"fixture":"body"}')
        self.assertNotIn("fixture", self.output.getvalue())

    real_run_command = staticmethod(deploy.run_command)

    def test_expected_receipt_error_is_informational(self):
        receipt = {
            "operation": "FENCE",
            "status": "ERROR",
            "error_code": "fence_epoch_or_state_conflict",
            "receipt_revision": "sqlite:1",
        }
        self.patch(
            "run_command",
            return_value=json.dumps({"data": {"receipt.json": json.dumps(receipt)}}),
        )
        deploy.print_admission_receipt(self.kubeconfig)
        self.assertIn(
            "admission receipt: operation=FENCE status=ERROR error_code=fence_epoch_or_state_conflict revision=sqlite:1",
            self.output.getvalue(),
        )
        self.assertIn(
            "expected on a simulator-opened OPEN/1 site", self.output.getvalue()
        )

    def test_absent_or_unreadable_receipts_print_one_payload_free_diagnostic(self):
        for result, message in (
            ("  \n", "absent"),
            (
                deploy.DeployError("Forbidden: private payload"),
                "unreadable (kubectl failed)",
            ),
            (OSError("private payload"), "unreadable (kubectl failed)"),
            (
                UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte"),
                "unreadable (kubectl failed)",
            ),
            (
                subprocess.TimeoutExpired(["kubectl"], 120, output="private payload"),
                "unreadable (kubectl failed)",
            ),
            *(
                (encoded, "unreadable (invalid receipt map or receipt.json)")
                for encoded in (
                    "private payload",
                    "[" * 2000,
                    "null",
                    "[]",
                    "{}",
                    '{"data":null}',
                    '{"data":{"receipt.json":"private payload"}}',
                    '{"data":{"receipt.json":{}}}',
                    '{"data":{"receipt.json":"[]"}}',
                )
            ),
        ):
            with self.subTest(result=result):
                self.output.seek(0)
                self.output.truncate()
                with mock.patch.object(
                    acceptance,
                    "run_command",
                    side_effect=result if isinstance(result, Exception) else None,
                    return_value=result,
                ) as command:
                    deploy.print_admission_receipt(self.kubeconfig)
                self.assertIn("--ignore-not-found", command.call_args.args[0])
                self.assertEqual(
                    self.output.getvalue(), f"admission receipt: {message}\n"
                )

    def test_deploy_ensures_maps_between_apply_and_restart_then_preflights(self):
        events = []
        self.patch("deploy_modal_credentials", modules=(cli,))
        self.patch("verify_api_route_reader_rbac", modules=(cli,))
        self.patch("api_cidrs", modules=(cli,), return_value=["172.20.0.1/32"])

        def apply(path, kube):
            # The full simulator phase must complete without calling ensure.
            ensure_maps.assert_not_called()
            events.append(("apply", path.name))
            self.assertIn(
                "admission control maps: skipped for simulator bootstrap",
                self.output.getvalue(),
            )

        def command(args, **kwargs):
            if ("apply", "executable-preview.json") not in events:
                ensure_maps.assert_not_called()
            events.append(("command", args))

        self.patch("kubectl_apply", modules=(cli,), side_effect=apply)
        self.patch("run_command", modules=(cli,), side_effect=command)
        ensure_maps = self.patch(
            "ensure_admission_control_maps",
            modules=(cli,),
            side_effect=lambda kube: events.append(("maps",)),
        )
        self.patch(
            "runtime_admission_preflight",
            modules=(cli,),
            side_effect=lambda config, base: events.append(("preflight", base)),
        )
        deploy.deploy_runtime(
            self.config,
            self.state,
            Fixture.outputs(self.config),
            self.kubeconfig,
            self.images,
        )
        applied = events.index(("apply", "executable-preview.json"))
        self.assertEqual(events[applied + 1], ("maps",))
        self.assertIn("restart", events[applied + 2][1])
        self.assertIn("status", events[applied + 3][1])
        self.assertEqual(events[applied + 4], ("preflight", BASE))
        ensure_maps.assert_called_once_with(self.kubeconfig)

    def test_verify_calls_preflight_even_without_smoke_receipt(self):
        self.patch("load_task_selection")
        self.patch("verify_litellm")
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"ok"
        for receipt in (
            "",
            "malformed receipt",
            deploy.DeployError("kubectl unavailable"),
        ):
            for blocked in (False, True):
                with self.subTest(receipt=receipt, blocked=blocked):
                    self.output.seek(0)
                    self.output.truncate()

                    def command(args, *, receipt=receipt, **kwargs):
                        if deploy.ADMISSION_RECEIPT_MAP in args:
                            if isinstance(receipt, Exception):
                                raise receipt
                            return receipt
                        return json.dumps(
                            {"status": {"availableReplicas": 1, "succeeded": 1}}
                        )

                    with (
                        mock.patch.object(
                            acceptance, "run_command", side_effect=command
                        ),
                        mock.patch.object(
                            infra.urllib_request, "urlopen", return_value=response
                        ),
                        mock.patch.object(
                            infra,
                            "api_json_with_headers",
                            return_value=({}, OPEN_HEADERS),
                        ) as overview,
                        mock.patch.object(
                            infra,
                            "api_json",
                            return_value=incident_snapshot()
                            if blocked
                            else available_snapshot(),
                        ) as capacity,
                    ):
                        if blocked:
                            with self.assertRaisesRegex(
                                deploy.DeployError, "no scheduler capacity"
                            ):
                                deploy.verify_runtime(
                                    self.config,
                                    Fixture.outputs(self.config),
                                    self.kubeconfig,
                                    require_smoke=False,
                                )
                            self.assertNotIn("readiness: PASS", self.output.getvalue())
                        else:
                            deploy.verify_runtime(
                                self.config,
                                Fixture.outputs(self.config),
                                self.kubeconfig,
                                require_smoke=False,
                            )
                            self.assertIn(
                                "Management readiness: PASS", self.output.getvalue()
                            )
                    overview.assert_called_once_with(BASE, "GET", "/api/v1/overview")
                    capacity.assert_called_once_with(BASE, "GET", CAPACITY_PATH)
                    self.assertIn(
                        "admission receipt: absent"
                        if receipt == ""
                        else "admission receipt: unreadable",
                        self.output.getvalue(),
                    )


class SmokeOperationTests(RuntimeFixture):
    def setUp(self):
        super().setUp()
        self.candidate, self.task = deploy.smoke_documents(self.images)
        self.candidate_id = self.candidate["driver_config"]["candidate_version_id"]
        self.task_id = self.task["driver_config"]["task_version_id"]
        self.operation_path = self.state / "smoke-operation.json"
        self.patch(
            "ensure_version",
            side_effect=lambda base, path, version_id, doc: {"id": version_id},
        )

    def prepare(self, images=None):
        return deploy.prepare_smoke_operation(
            self.config, BASE, images or self.images, self.candidate_id, self.task_id
        )

    def evaluation(self, operation, status="RUNNING", created_at=None):
        return {
            "id": "eval_fixture",
            "created_at": created_at or operation["started_at"],
            "status": status,
            "items": [{"run_id": "run_fixture"}],
        }

    def save(self, operation):
        deploy.atomic_write(
            self.operation_path, deploy.canonical_json(operation), mode=0o600
        )

    def test_new_operation_persisted_before_post_and_lost_post_response_reuses_key(
        self,
    ):
        api = self.patch(
            "api_json", side_effect=deploy.DeployError("lost POST response")
        )
        with self.assertRaisesRegex(deploy.DeployError, "lost POST response"):
            deploy.run_smoke(self.config, self.kubeconfig, self.images)
        operation = deploy.load_json(self.operation_path)
        self.assertEqual(operation["retry"], 0)
        self.assertEqual(
            operation["runtime"],
            {
                "management_digest": self.images["management"]["digest"],
                "driver_digest": self.images["driver"]["digest"],
            },
        )
        self.assertEqual(
            operation["idempotency_key"],
            f"deployment-smoke-preview-{self.candidate_id[-8:].lower()}-{self.task_id[-8:].lower()}-{operation['nonce']}",
        )
        self.assertEqual(self.operation_path.stat().st_mode & 0o777, 0o600)
        api.reset_mock()
        self.assertEqual(self.prepare(), operation)
        api.assert_not_called()

    def test_old_nonterminal_operation_reuses_original_start_nonce_key_and_retry(self):
        operation = self.prepare()
        operation.update(
            evaluation_id="eval_fixture", started_at="2026-09-01T00:00:00Z", retry=2
        )
        operation["idempotency_key"] = deploy.smoke_operation_key(
            self.config, operation
        )
        self.save(operation)
        before = self.operation_path.read_bytes()
        self.patch("api_json", return_value=self.evaluation(operation))
        self.assertEqual(self.prepare(), operation)
        self.assertEqual(self.operation_path.read_bytes(), before)

    def test_every_terminal_outcome_rotates_nonce_and_key(self):
        for status in sorted(deploy.TERMINAL_EVALUATIONS):
            with self.subTest(status=status):
                self.operation_path.unlink(missing_ok=True)
                operation = self.prepare()
                operation["evaluation_id"] = "eval_fixture"
                self.save(operation)
                with mock.patch.object(
                    acceptance,
                    "api_json",
                    return_value=self.evaluation(operation, status),
                ):
                    rotated = self.prepare()
                self.assertNotEqual(rotated["nonce"], operation["nonce"])
                self.assertNotEqual(
                    rotated["idempotency_key"], operation["idempotency_key"]
                )
                self.assertEqual(rotated["retry"], 0)
                self.assertNotIn("evaluation_id", rotated)

    def test_deleted_evaluation_rotates_operation_but_other_http_errors_preserve_it(
        self,
    ):
        for status in (404, 403, 429, 500, 503):
            with self.subTest(status=status):
                self.operation_path.unlink(missing_ok=True)
                operation = self.prepare()
                operation["evaluation_id"] = "eval_archived"
                self.save(operation)
                before = self.operation_path.read_bytes()
                path = "/api/v1/evaluations/eval_archived"
                error = deploy.ManagementHTTPError(status, path, "lookup failed")
                with mock.patch.object(
                    acceptance, "api_json", side_effect=error
                ) as api:
                    if status == 404:
                        rotated = self.prepare()
                        self.assertNotEqual(rotated["nonce"], operation["nonce"])
                        self.assertNotEqual(
                            rotated["idempotency_key"], operation["idempotency_key"]
                        )
                        self.assertEqual(rotated["retry"], 0)
                        self.assertNotIn("evaluation_id", rotated)
                        self.assertEqual(deploy.load_json(self.operation_path), rotated)
                    else:
                        with self.assertRaises(deploy.ManagementHTTPError) as caught:
                            self.prepare()
                        self.assertEqual(caught.exception.status, status)
                        self.assertEqual(self.operation_path.read_bytes(), before)
                api.assert_called_once_with(BASE, "GET", path)

    def test_either_runtime_digest_change_rotates_without_querying_old_evaluation(self):
        for name in ("management", "driver"):
            with self.subTest(name=name):
                self.operation_path.unlink(missing_ok=True)
                operation = self.prepare()
                operation["evaluation_id"] = "eval_old_runtime"
                self.save(operation)
                images = copy.deepcopy(self.images)
                images[name]["digest"] = "sha256:" + "f" * 64
                api = self.patch(
                    "api_json", side_effect=AssertionError("old evaluation queried")
                )
                rotated = self.prepare(images)
                self.assertNotEqual(rotated["nonce"], operation["nonce"])
                self.assertEqual(
                    rotated["runtime"][f"{name}_digest"], images[name]["digest"]
                )
                api.assert_not_called()

    def test_timestamp_tolerance_and_missing_timestamps_fail_closed(self):
        operation = self.prepare()
        start = datetime.fromisoformat(operation["started_at"].replace("Z", "+00:00"))
        for offset in (-60, 0, 600):
            deploy.check_smoke_replay(
                operation,
                self.evaluation(
                    operation,
                    created_at=(start + timedelta(seconds=offset)).isoformat(),
                ),
            )
        for invalid in (None, "not a date", "2026-09-12T00:00:00"):
            evaluation = self.evaluation(operation)
            evaluation["created_at"] = invalid
            with self.assertRaisesRegex(deploy.DeployError, "timestamp with timezone"):
                deploy.check_smoke_replay(operation, evaluation)

    def test_post_replay_is_rejected_before_run_poll_or_success_receipt(self):
        operation = self.prepare()
        start = datetime.fromisoformat(operation["started_at"].replace("Z", "+00:00"))
        api = self.patch(
            "api_json",
            return_value=self.evaluation(
                operation, "COMPLETED", (start - timedelta(seconds=61)).isoformat()
            ),
        )
        with self.assertRaisesRegex(
            deploy.DeployError,
            "smoke evaluation eval_fixture is an idempotent replay created .*; rotate the smoke operation",
        ):
            deploy.run_smoke(self.config, self.kubeconfig, self.images)
        self.assertEqual(api.call_count, 1)
        self.assertFalse((self.state / "smoke.json").exists())

    def test_fast_completion_requires_no_observed_intermediate_states_and_binds_receipt(
        self,
    ):
        operation = self.prepare()

        def api(base, method, path, body=None, headers=None):
            if method == "POST":
                self.assertEqual(
                    headers["Idempotency-Key"], operation["idempotency_key"]
                )
                self.assertEqual(body["labels"]["smoke_nonce"], operation["nonce"])
                return self.evaluation(operation, "COMPLETED")
            if path.startswith("/api/v1/runs/"):
                return {"state": "COMPLETED", "score": 1.0}
            return self.evaluation(operation, "COMPLETED")

        self.patch("api_json", side_effect=api)
        sleep = self.patch("time", wraps=deploy.time)
        receipt = deploy.run_smoke(self.config, self.kubeconfig, self.images)
        sleep.sleep.assert_not_called()
        self.assertEqual(receipt["nonce"], operation["nonce"])
        self.assertEqual(receipt["evaluation_created_at"], operation["started_at"])
        self.assertEqual(receipt["runtime"], operation["runtime"])
        self.assertEqual(deploy.load_json(self.state / "smoke.json"), receipt)
        self.assertEqual(
            deploy.smoke_documents(self.images), (self.candidate, self.task)
        )

    def test_resumed_old_operation_uses_persisted_start_for_replay_check(self):
        operation = self.prepare()
        operation.update(
            started_at="2026-09-01T00:00:00Z", evaluation_id="eval_fixture"
        )
        self.save(operation)
        eval_gets = 0

        def api(base, method, path, body=None, headers=None):
            nonlocal eval_gets
            if path.startswith("/api/v1/runs/"):
                return {"state": "COMPLETED", "score": 1.0}
            if method == "GET":
                eval_gets += 1
            return self.evaluation(
                operation, "RUNNING" if eval_gets == 1 else "COMPLETED"
            )

        self.patch("api_json", side_effect=api)
        receipt = deploy.run_smoke(self.config, self.kubeconfig, self.images)
        self.assertEqual(receipt["nonce"], operation["nonce"])
        self.assertEqual(receipt["evaluation_created_at"], "2026-09-01T00:00:00Z")

    def test_retry_is_persisted_before_post_and_survives_interruption(self):
        operation = self.prepare()
        posted = []

        def api(base, method, path, body=None, headers=None):
            if method == "POST":
                persisted = deploy.load_json(self.operation_path)
                self.assertEqual(
                    persisted["idempotency_key"], headers["Idempotency-Key"]
                )
                posted.append(headers["Idempotency-Key"])
                if len(posted) == 2:
                    self.assertEqual(persisted["retry"], 1)
                    self.assertNotIn("evaluation_id", persisted)
                    raise deploy.DeployError("lost retry response")
                return self.evaluation(operation, "FAILED")
            return {"state": "FAILED"}

        self.patch("api_json", side_effect=api)
        with self.assertRaisesRegex(deploy.DeployError, "lost retry response"):
            deploy.run_smoke(self.config, self.kubeconfig, self.images)
        self.assertEqual(
            posted,
            [operation["idempotency_key"], operation["idempotency_key"] + "-retry-1"],
        )
        resumed = self.prepare()
        self.assertEqual(resumed["nonce"], operation["nonce"])
        self.assertEqual(resumed["retry"], 1)
        self.assertEqual(resumed["idempotency_key"], posted[-1])


if __name__ == "__main__":
    unittest.main()
