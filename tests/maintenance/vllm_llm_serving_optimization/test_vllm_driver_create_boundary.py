"""Owned Driver CREATE exhaustion is infra; child/unknown failures gain no authority."""

import deployment_test_guard  # noqa: F401

import hashlib
import io
import json
import sys
import tempfile
import time
import unittest
import urllib.error
from contextlib import contextmanager
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_task import TASK_ROOT  # noqa: E402

TASK = TASK_ROOT
sys.path[:0] = [str(TASK / "judge"), str(TASK / "verifier")]
import driver_client
from driver_client import (
    DriverClient,
    DriverCreateInfrastructureError,
    DriverQuotaError,
    SidecarExecutionError,
)
from parallel_phase import PhaseInfrastructureError, infrastructure_signal
from provisioning_owner import ProvisioningOwner


class DriverCreateBoundaryTests(unittest.TestCase):
    def test_real_read_http_adapter_maps_only_transport_errors_to_readiness_retry_type(
        self,
    ):
        from unittest.mock import Mock

        import batch_coordinator as batch

        signer = SimpleNamespace(headers=lambda *args: {})
        for error, expected in [
            (urllib.error.URLError("connection refused"), "connection"),
            (TimeoutError(), "timeout"),
            (
                urllib.error.HTTPError("http://fixture/ready", 503, "", {}, None),
                "http_5xx",
            ),
        ]:
            opener = Mock()
            opener.open.side_effect = error
            with patch.object(batch, "build_opener", return_value=opener):
                with self.assertRaises(batch.TestbedTransportError) as raised:
                    batch._request(
                        "http://fixture",
                        "/ready",
                        signer=signer,
                        target_sidecar_id="declared",
                    )
                self.assertEqual(raised.exception.code, expected)
        for error in [
            urllib.error.HTTPError("http://fixture/ready", 401, "", {}, None),
            ValueError("malformed JSON"),
        ]:
            opener = Mock()
            opener.open.side_effect = error
            with patch.object(batch, "build_opener", return_value=opener):
                with self.assertRaises(batch.BackendError) as raised:
                    batch._request(
                        "http://fixture",
                        "/ready",
                        signer=signer,
                        target_sidecar_id="declared",
                    )
                self.assertNotIsInstance(raised.exception, batch.TestbedTransportError)

    def test_readiness_transport_retries_keep_original_deadline_and_first_typed_cause(
        self,
    ):
        from unittest.mock import Mock

        import batch_coordinator as batch

        client = batch.TestbedClient.__new__(batch.TestbedClient)
        client._call = Mock(
            side_effect=[batch.TestbedTransportError("connection"), {"ready": True}]
        )
        clock = [0.0]
        with (
            patch.object(batch.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(
                batch.time,
                "sleep",
                side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            ),
        ):
            client.ready()
            self.assertEqual(client._call.call_count, 2)
            self.assertEqual(clock[0], 0.1)
            client._call = Mock(side_effect=batch.TestbedTransportError("timeout"))
            with self.assertRaises(batch.TestbedTransportError) as raised:
                client.ready(deadline=1.0)
            self.assertEqual(raised.exception.code, "timeout")
            self.assertEqual(clock[0], 1.0)
            client._call = Mock(side_effect=ValueError("bad protocol"))
            with self.assertRaises(ValueError):
                client.ready()
            self.assertEqual(client._call.call_count, 1)

    def setUp(self):
        fixture = json.loads(
            (
                Path(__file__).parent / "fixtures/vllm/verifier-platform-failure.json"
            ).read_bytes()
        )
        self.environment = dict(
            zip(
                (
                    "FRONTIER_RUN_ID",
                    "FRONTIER_ATTEMPT_ID",
                    "FRONTIER_VERIFICATION_ID",
                    "FRONTIER_TASK_CONFIG_DIGEST",
                    "FRONTIER_VERIFIER_IMAGE_DIGEST",
                ),
                (
                    fixture[k]
                    for k in (
                        "run_id",
                        "attempt_id",
                        "verification_id",
                        "task_config_digest",
                        "verifier_image_digest",
                    )
                ),
                strict=True,
            )
        )
        self.request = {
            "gpu_type": "H100",
            "gpu_count": 1,
            "max_duration_seconds": 3600,
        }
        self.key = "owned-create-fixture"
        self.owner = ProvisioningOwner(
            self.environment,
            {"frozen": "plan"},
            [{"instance_id": "original"}],
            self.request,
            self.key,
        )
        self.client = DriverClient(
            "http://driver.invalid",
            fixture["verification_id"],
            "synthetic-token",
            deadline=5000,
        )
        self.now = 0.0
        self.opens = []

    def error(self, status=502, code="gpu_backend_unavailable"):
        headers = Message()
        headers["Content-Type"] = "application/problem+json"
        return urllib.error.HTTPError(
            self.client.url,
            status,
            "fixture",
            headers,
            io.BytesIO(
                json.dumps(
                    {"status": status, "code": code, "title": "capacity fixture"}
                ).encode()
            ),
        )

    def sleep(self, duration):
        self.now += duration

    def invoke(self, fault, *, method="POST", suffix="", owner=True):
        def opened(request, deadline):
            self.opens.append(
                (request.method, request.get_header("Idempotency-key"), request.data)
            )
            raise fault() if callable(fault) else fault

        with (
            patch.object(driver_client.time, "monotonic", side_effect=lambda: self.now),
            patch.object(driver_client.time, "sleep", side_effect=self.sleep),
            patch.object(driver_client, "_open_before_deadline", side_effect=opened),
        ):
            return self.client.request(
                method,
                suffix,
                body=self.request if method == "POST" else None,
                key=self.key if method == "POST" else None,
                provisioning_owner=self.owner if owner else None,
            )

    def test_owned_capacity_timeout_preserves_bound_identity_and_bounded_3720s_cap(
        self,
    ):
        with self.assertRaises(DriverCreateInfrastructureError) as raised:
            self.invoke(self.error)
        signal = raised.exception.signal
        self.assertEqual(self.now, 3720)
        self.assertEqual(driver_client._CREATE_OBSERVATION_SECONDS, 3720)
        self.assertEqual(signal["cause"], "driver_create_capacity_timeout")
        self.assertEqual(signal["driver_error_code"], "gpu_backend_unavailable")
        self.assertEqual(signal["owner_identity"], self.owner.identity)
        self.assertEqual(
            signal["idempotency_sha256"],
            "sha256:" + hashlib.sha256(self.key.encode()).hexdigest(),
        )
        self.assertFalse(signal["candidate_input_forwarded"])
        self.assertFalse(signal["parent_response_accepted"])
        self.assertTrue(signal["cleanup_required"])
        self.assertIsNone(self.owner.operation_id)
        self.assertEqual(len({row[1] for row in self.opens}), 1)
        self.assertEqual(len({row[2] for row in self.opens}), 1)
        self.assertNotIn("synthetic-token", json.dumps(signal))

    def test_cold_import_1861_seconds_succeeds_with_original_key_and_body(self):
        accepted = {
            "runner_id": "gpurunner_00000000000000000000000000",
            "controller_binding": self.owner.claim,
        }

        @contextmanager
        def opened(request, deadline):
            self.opens.append(
                (request.method, request.get_header("Idempotency-key"), request.data)
            )
            self.assertEqual(deadline, 3720)
            if self.now < 1861:
                self.now = min(1861.0, self.now + 120.0)
                raise urllib.error.URLError("cold image import in progress")
            response = io.BytesIO(json.dumps(accepted).encode())
            response.status = 202
            response.geturl = lambda: self.client.url
            response.isclosed = lambda: response.closed
            yield response, SimpleNamespace(settimeout=lambda timeout: None)

        with (
            patch.object(driver_client.time, "monotonic", side_effect=lambda: self.now),
            patch.object(driver_client.time, "sleep", side_effect=self.sleep),
            patch.object(driver_client, "_open_before_deadline", side_effect=opened),
        ):
            result = self.client.request(
                "POST",
                "",
                body=self.request,
                key=self.key,
                provisioning_owner=self.owner,
            )
        self.assertEqual(result, accepted)
        self.assertGreaterEqual(self.now, 1861)
        self.assertLess(self.now, 3720)
        self.assertEqual(len({row[1] for row in self.opens}), 1)
        self.assertEqual(len({row[2] for row in self.opens}), 1)
        self.assertEqual(self.client.deadline, 5000)
        self.assertEqual(self.request["max_duration_seconds"], 3600)

    def test_shorter_absolute_owner_deadline_still_wins_over_create_cap(self):
        self.client.deadline = 1000
        with self.assertRaises(DriverCreateInfrastructureError) as raised:
            self.invoke(urllib.error.URLError("image import still pending"))
        self.assertEqual(self.now, 1000)
        self.assertEqual(raised.exception.signal["observation_seconds"], 1000)
        self.assertEqual(self.client.deadline, 1000)
        self.assertEqual(len({row[1] for row in self.opens}), 1)
        self.assertEqual(len({row[2] for row in self.opens}), 1)

    def test_typed_quota_exhaustion_is_terminal_resource_feedback_not_infra(self):
        def case(quota):
            self.opens.clear()

            def fault():
                headers = Message()
                headers["Content-Type"] = "application/problem+json"
                return urllib.error.HTTPError(
                    self.client.url,
                    429,
                    "fixture",
                    headers,
                    io.BytesIO(
                        json.dumps(
                            {
                                "status": 429,
                                "code": "quota_exhausted",
                                "title": "ledger",
                                "quota_failure": quota,
                            }
                        ).encode()
                    ),
                )

            with self.assertRaises(DriverQuotaError) as raised:
                self.invoke(fault)
            self.assertEqual(raised.exception.quota, quota)
            self.assertIsNone(infrastructure_signal(raised.exception))
            self.assertEqual(len(self.opens), 1)

        for quota in (
            {
                "resource": "gpu_instances",
                "principal": "verifier",
                "limit": 4,
                "current": 4,
                "requested": 1,
            },
            {
                "resource": "gpu_seconds",
                "principal": "verifier",
                "limit": 100,
                "used": 80,
                "reserved": 10,
                "remaining": 10,
                "requested_count": 1,
                "requested_seconds": 20,
            },
        ):
            case(quota)
        for malformed in (
            {
                "resource": "gpu_instances",
                "principal": "verifier",
                "limit": True,
                "current": 0,
                "requested": 1,
            },
            {"resource": "gpu_seconds", "principal": "verifier"},
        ):
            with self.assertRaises(ValueError):
                DriverQuotaError(malformed)

    def test_quota_integer_widths_match_core_and_reject_overflow(self):
        for resource, values in (
            ("gpu_instances", {"limit": 0, "current": 0, "requested": 0}),
            (
                "gpu_seconds",
                {
                    "limit": 0,
                    "used": 0,
                    "reserved": 0,
                    "remaining": 0,
                    "requested_count": 0,
                    "requested_seconds": 0,
                },
            ),
        ):
            for field in values:
                bits = (
                    32
                    if resource == "gpu_instances" or field == "requested_count"
                    else 64
                )
                base = {"resource": resource, "principal": "verifier", **values}
                DriverQuotaError({**base, field: 2**bits - 1})
                with self.assertRaises(ValueError):
                    DriverQuotaError({**base, field: 2**bits})

    def test_only_typed_create_mutation_rate_429_retries_with_identical_request(self):
        @contextmanager
        def opened(request, deadline):
            self.opens.append(
                (request.method, request.get_header("Idempotency-key"), request.data)
            )
            if len(self.opens) == 1:
                raise self.error(429, "mutation_rate_exhausted")
            response = io.BytesIO(b'{"accepted":true}')
            response.status = 202
            response.geturl = lambda: self.client.url
            response.isclosed = lambda: response.closed
            yield response, SimpleNamespace(settimeout=lambda timeout: None)

        with (
            patch.object(driver_client.time, "monotonic", side_effect=lambda: self.now),
            patch.object(driver_client.time, "sleep", side_effect=self.sleep),
            patch.object(driver_client, "_open_before_deadline", side_effect=opened),
        ):
            self.assertEqual(
                self.client.request("POST", "", body=self.request, key=self.key),
                {"accepted": True},
            )
        self.assertEqual(len(self.opens), 2)
        self.assertEqual(self.opens[0], self.opens[1])
        self.assertEqual(self.now, 1.0)
        self.opens.clear()
        with self.assertRaises(SidecarExecutionError):
            self.invoke(lambda: self.error(429, "unknown_rate_limit"))
        self.assertEqual(len(self.opens), 1)

    def test_transport_timeout_only_on_owned_create_not_other_operations(self):
        with self.assertRaises(DriverCreateInfrastructureError) as raised:
            self.invoke(urllib.error.URLError("fixture transport"))
        self.assertEqual(
            raised.exception.signal["cause"], "driver_create_transport_timeout"
        )
        self.now = 0.0
        self.client.deadline = 2
        with self.assertRaises(SidecarExecutionError) as ordinary:
            self.invoke(
                urllib.error.URLError("fixture transport"),
                method="GET",
                suffix="/runner",
                owner=False,
            )
        self.assertIsNone(infrastructure_signal(ordinary.exception))

    def test_unowned_unknown_or_malformed_create_never_acquires_infra_type(self):
        for fault, owner in (
            (self.error, False),
            (lambda: self.error(502, "unknown_failure"), True),
            (LookupError("unknown candidate-like error"), True),
        ):
            self.now = 0.0
            self.client.deadline = 2
            with (
                self.subTest(fault=str(fault), owner=owner),
                self.assertRaises(Exception) as raised,
            ):
                self.invoke(fault, owner=owner)
            self.assertIsNone(infrastructure_signal(raised.exception))

    def test_cleanup_uncertain_is_typed_but_never_recreates_or_claims_compensation(
        self,
    ):
        with self.assertRaises(DriverCreateInfrastructureError) as raised:
            self.invoke(lambda: self.error(500, "gpu_cleanup_uncertain"))
        self.assertEqual(len(self.opens), 1)
        self.assertEqual(
            raised.exception.signal["cause"], "driver_create_cleanup_uncertain"
        )
        self.assertTrue(raised.exception.signal["cleanup_required"])
        self.assertEqual(
            raised.exception.signal["proof_kind"],
            "cpu_owned_control_observation_not_compensated_proof",
        )

    def test_real_session_keeps_driver_owned_uncertain_cleanup_without_new_create(self):
        import gpu_session

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_config = {"sidecars": {"scheduler": {"args": []}}}
            with patch.object(
                gpu_session, "phase_limits", return_value={"gpu_seconds": 3600}
            ):
                session = gpu_session.GPUSession(
                    self.client,
                    legacy_config,
                    "train",
                    gpu_seconds=3600,
                    request_key=self.key,
                    scratch=root / "scratch",
                    evidence=root / "evidence",
                    cleanup_deadline=6000,
                    provisioning_owner=self.owner,
                )
            calls = []

            def unavailable(request, deadline):
                calls.append((request.method, request.get_header("Idempotency-key")))
                raise self.error()

            with (
                patch.object(
                    driver_client.time, "monotonic", side_effect=lambda: self.now
                ),
                patch.object(driver_client.time, "sleep", side_effect=self.sleep),
                patch.object(
                    driver_client, "_open_before_deadline", side_effect=unavailable
                ),
            ):
                with self.assertRaises(DriverCreateInfrastructureError):
                    session.__enter__()
                prior = len(calls)
                session.close()
                self.assertEqual(len(calls), prior)
            cleanup = json.loads((root / "evidence/cleanup-error.json").read_text())
            self.assertTrue(cleanup["cleanup_uncertain"])
            self.assertEqual(cleanup["owner"], "Driver lifecycle")
            self.assertIsNone(session.runner_id)
            self.assertEqual({row[0] for row in calls}, {"POST"})
            self.assertEqual({row[1] for row in calls}, {self.key})

    def test_unknown_child_signals_and_exception_subclasses_are_not_infra(self):
        for error in (
            RuntimeError("gpu_backend_unavailable"),
            SidecarExecutionError(
                "capacity", error_code="driver_create_capacity_timeout"
            ),
            {"cause": "driver_create_capacity_timeout"},
            SimpleNamespace(signal={"candidate_input_forwarded": False}),
        ):
            self.assertIsNone(infrastructure_signal(error))

        class UntrustedSubclass(DriverCreateInfrastructureError):
            pass

        error = UntrustedSubclass(
            cause="driver_create_capacity_timeout",
            verification_id=self.client.verification_id,
            request_key=self.key,
            request_body=b"fixture",
            observation_seconds=1800,
            owner_identity=self.owner.identity,
        )
        self.assertIsNone(infrastructure_signal(error))

    def test_parallel_group_preserves_typed_preinput_failure_and_completed_sibling(
        self,
    ):
        import parallel_phase
        from test_runtime_plan import funded_fixture

        config_plan, config = funded_fixture()
        with self.assertRaises(DriverCreateInfrastructureError) as raised:
            self.invoke(self.error)
        observed_error = raised.exception

        class Client:
            verification_id = "test_verification"

            def __init__(self, *args, **kwargs):
                pass

        class Session:
            def __init__(self, client, runtime_config, mode, **kwargs):
                self.group = kwargs["evidence"].parent.name
                self.clock = kwargs["calibration_clock"]
                self.runner_id = "runner_" + self.group
                if self.group != "swe-public":
                    self.clock.accept_grant(self.runner_id, 8)

            def __enter__(self):
                if self.group == "swe-public":
                    raise observed_error
                return self

            def __exit__(self, *args):
                return False

        def phase(*args, **kwargs):
            if scenario == "quota_trusted":
                from cpu_reference_verdict import TrustedMeasurementError

                raise TrustedMeasurementError(
                    "reference", "bfcl/public", "fixture_missing"
                )
            if scenario == "mixed":
                raise RuntimeError("unknown candidate group exception")
            group = kwargs["selected_group"]
            return {
                variant: {group: [group, variant]}
                for variant in parallel_phase.VARIANTS
            }

        typed_error = observed_error
        for scenario in ("typed", "unknown", "mixed", "quota_trusted"):
            observed_error = (
                RuntimeError("gpu_backend_unavailable unknown candidate exception")
                if scenario == "unknown"
                else typed_error
            )
            if scenario == "quota_trusted":
                observed_error = DriverQuotaError(
                    {
                        "resource": "gpu_instances",
                        "principal": "verifier",
                        "limit": 4,
                        "current": 4,
                        "requested": 1,
                    }
                )
            with (
                self.subTest(scenario=scenario),
                tempfile.TemporaryDirectory() as directory,
                patch.object(parallel_phase, "DriverGPU", Client),
                patch.object(parallel_phase, "GPUSession", Session),
                patch.object(parallel_phase, "PhaseAuditPlan"),
                patch.object(parallel_phase, "for_phase", return_value=self.owner),
                patch.object(parallel_phase, "run_phase", side_effect=phase) as run,
                patch.object(parallel_phase, "evaluate_train") as score,
            ):
                root = Path(directory)
                failure_type = (
                    PhaseInfrastructureError
                    if scenario == "typed"
                    else parallel_phase.PhaseQuotaError
                    if scenario == "quota_trusted"
                    else RuntimeError
                )
                with self.assertRaises(failure_type) as failure:
                    parallel_phase.run_parallel_phase(
                        config,
                        config_plan,
                        "train",
                        environment={
                            "FRONTIER_DRIVER_INTERNAL_URL": "http://driver",
                            "FRONTIER_VERIFICATION_ID": "test_verification",
                            "FRONTIER_VERIFIER_TOKEN": "fixture",
                        },
                        candidate_patch=b"diff",
                        root=root,
                        evidence=root / "evidence",
                        deadline=time.monotonic() + 1000,
                    )
                score.assert_not_called()
                self.assertEqual(
                    [call.kwargs["selected_group"] for call in run.call_args_list],
                    ["bfcl/public"],
                )
                record = json.loads(
                    (root / "evidence/incomplete-groups.json").read_text()
                )
                self.assertEqual(
                    record["completed_groups"],
                    [] if scenario in ("mixed", "quota_trusted") else ["bfcl/public"],
                )
                self.assertFalse(record["score_ready"])
                if scenario == "typed":
                    self.assertTrue(failure.exception.cleanup_required)
                    self.assertEqual(
                        record["infrastructure_signals"]["swe/public"]["cause"],
                        "driver_create_capacity_timeout",
                    )
                elif scenario == "quota_trusted":
                    self.assertEqual(
                        type(failure.exception), parallel_phase.PhaseQuotaError
                    )
                    self.assertEqual(record["infrastructure_signals"], {})
                else:
                    self.assertEqual(type(failure.exception), RuntimeError)
                    if scenario == "unknown":
                        self.assertEqual(record["infrastructure_signals"], {})
                    else:
                        self.assertEqual(
                            set(record["infrastructure_signals"]), {"swe/public"}
                        )

    def test_native_entrypoint_infra_is_unscored_and_requires_cleanup_before_retry(
        self,
    ):
        import os
        import tarfile

        import verify

        def case(mode):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload = root / "payload.tar"
                with tarfile.open(payload, "w") as archive:
                    entry = tarfile.TarInfo("payload/solution.patch")
                    entry.size = 4
                    archive.addfile(entry, io.BytesIO(b"diff"))
                (root / "scratch").mkdir()
                (root / "config.json").write_text("{}")
                (root / "plan.json").write_text("{}")
                (root / "dataset.json").write_text("{}")
                paths = {
                    "/opt/frontier/runtime-config.json": root / "config.json",
                    "/opt/frontier/runtime-plan.json": root / "plan.json",
                }
                env = {
                    "FRONTIER_RESULT_PATH": str(root / "result.json"),
                    "FRONTIER_EVIDENCE_PATH": str(root / "evidence"),
                    "FRONTIER_VERIFICATION_MODE": mode,
                    "FRONTIER_PAYLOAD_PATH": str(payload),
                    "FRONTIER_DATASET_PATH": str(root / "dataset.json"),
                    "FRONTIER_MAX_EVIDENCE_BYTES": "100000",
                }
                plan = {
                    "max_payload_bytes": 100000,
                    "phases": {mode: {"groups": {"g": {"gpu_seconds": 1}}}},
                }
                failure = PhaseInfrastructureError(
                    {
                        "swe/public": {
                            "cleanup_required": True,
                            "cause": "driver_create_capacity_timeout",
                            "candidate_input_forwarded": False,
                        }
                    }
                )
                with (
                    patch.dict(os.environ, env),
                    patch.object(
                        verify,
                        "Path",
                        side_effect=lambda value: Path(paths.get(str(value), value)),
                    ),
                    patch.object(
                        verify.tempfile, "mkdtemp", return_value=str(root / "scratch")
                    ),
                    patch.object(verify, "validate_plan", return_value=plan),
                    patch.object(verify, "load_release", return_value=object()),
                    patch.object(verify, "dataset_selector"),
                    patch.object(
                        verify,
                        "inspect_submission",
                        return_value={"valid_submission": True},
                    ),
                    patch.object(
                        verify, "phase_limits", return_value={"timeout_seconds": 100}
                    ),
                    patch.object(verify, "run_parallel_phase", side_effect=failure),
                    patch.object(verify, "write_evidence"),
                ):
                    self.assertEqual(verify.main(), 0)
                result = json.loads((root / "result.json").read_text())
                self.assertEqual(
                    result["failure"]["code"], "vllm_platform_infrastructure_error"
                )
                self.assertFalse(result["failure"]["retryable"])
                self.assertNotIn("raw_score", result)
                self.assertFalse((root / "scratch/evidence/failure.json").exists())
                self.assertTrue(
                    (root / "scratch/evidence/infrastructure-failure.json").is_file()
                )

        for mode in ("train", "eval"):
            case(mode)
