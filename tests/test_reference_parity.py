from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import frontier_deploy as deploy
from deploy import acceptance as parity
from deploy import infra
from deploy import publication as references
from test_task_seeding import mirrored_images, resolved_catalog


class ParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "receipt.json"
        self.name = "batch_local_counting"
        self.catalog = deploy.validate_task_catalog(
            resolved_catalog(), require_resolved=True
        )
        self.mirrors = mirrored_images(self.catalog)
        self.image = references.load_manifest()[self.name]
        self.image["digest"] = "sha256:" + "d" * 64
        self.mirror = {
            **self.mirrors["candidate"]["overlay_image"],
            "source_sha256": self.image["digest"],
            "source_reference": self.image["repository"] + "@" + self.image["digest"],
        }
        self.document = parity.candidate_document(self.name, self.image, self.mirror)
        task = next(t for t in self.catalog["tasks"] if t["name"] == self.name)
        self.task_document = deploy.task_version_document(task, self.mirrors[self.name])
        self.baseline = references.load_baselines()[self.name]
        # Exercise comparable replay with a synthetic baseline, independently
        # of whether the current release has real acceptance measurements.
        self.baseline["comparable"] = True
        self.receipt = {
            "schema_version": "frontier.reference-parity/v1",
            "management_url": "http://localhost",
            "tasks": {},
        }
        self.evaluation = {
            "id": "evaluation_test",
            "status": "COMPLETED",
            "items": [{"run_id": "run_test"}],
        }
        self.run = {
            "id": "run_test",
            "evaluation_id": self.evaluation["id"],
            "state": "COMPLETED",
            "outcome": "SUCCEEDED",
            "task_version_id": self.task_document["driver_config"]["task_version_id"],
            "candidate_version_id": self.document["driver_config"][
                "candidate_version_id"
            ],
            "score": self.baseline["historical_mean"],
        }
        self.events = [
            {
                "run_id": self.run["id"],
                "source": "DRIVER",
                "kind": "VERIFIER_TRAIN_COMPLETED",
                "payload": {
                    "status": "success",
                    "verification_id": "verification_test",
                    "unused": "judge content",
                },
            }
        ]
        self.stored = []
        self.calls = []
        self.conflict = False
        self.now = 0
        self.clock = mock.patch.object(
            parity.time, "time", side_effect=lambda: self.now
        )
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.sleep = mock.patch.object(
            parity.time, "sleep", side_effect=self.advance_clock
        )
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def advance_clock(self, seconds):
        self.now += seconds

    def http(self, request, timeout):
        path = request.full_url.removeprefix("http://localhost")
        method = request.get_method()
        self.calls.append((method, path, request))
        if path.startswith("/api/v1/task-versions/"):
            result = self.task_document
        elif path == "/api/v1/candidate-versions":
            if method == "POST":
                self.stored = [
                    {**json.loads(request.data), "id": self.run["candidate_version_id"]}
                ]
                if self.conflict:
                    raise HTTPError(
                        request.full_url, 409, "already exists", {}, io.BytesIO(b"{}")
                    )
                result = self.stored[0]
            else:
                self.fail((method, path))
        elif path.startswith("/api/v1/candidate-versions/") and method == "GET":
            version_id = path.removeprefix("/api/v1/candidate-versions/")
            matches = [item for item in self.stored if item["id"] == version_id]
            if not matches:
                raise HTTPError(
                    request.full_url,
                    404,
                    "Not Found",
                    {},
                    io.BytesIO(b'{"code":"not_found"}'),
                )
            (result,) = matches
        elif path.endswith("/cancel"):
            result = {**self.evaluation, "status": "CANCELLED"}
        elif (
            path == "/api/v1/evaluations"
            or path == "/api/v1/evaluations/evaluation_test"
        ):
            result = self.evaluation
        elif path.endswith("/events?limit=500"):
            result = self.events
        elif path == "/api/v1/runs/run_test":
            result = self.run
        else:
            self.fail((method, path))
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(result).encode()
        response.headers = {}
        return response

    def replay(self):
        return parity.replay_task(
            "http://localhost",
            self.name,
            self.document,
            self.task_document,
            self.baseline,
            self.image,
            self.receipt,
            self.path,
            "1",
            0.001,
        )

    def test_completed_success_with_train_and_score_passes_and_receipt_excludes_evidence(
        self,
    ):
        with mock.patch.object(infra.urllib_request, "urlopen", side_effect=self.http):
            result = self.replay()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["train_verification_id"], "verification_test")
        self.assertEqual(json.loads(self.path.read_text()), self.receipt)
        self.assertNotIn("judge content", self.path.read_text())
        self.assertIn("image_digests", result)
        self.assertEqual(result["agreement_rule"], parity.RULE)
        posted = next(
            call[2]
            for call in self.calls
            if call[:2] == ("POST", "/api/v1/evaluations")
        )
        self.assertEqual(
            posted.get_header("Idempotency-key"),
            f"parity-{self.name}-{self.run['candidate_version_id']}-1",
        )
        self.assertEqual(
            json.loads(posted.data)["task_version_ids"], [self.run["task_version_id"]]
        )

    def test_parent_resources_and_sidecars_bind_only_image_digests(self):
        images = self.task_document["driver_config"]["images"]
        images["gpu_parent_resources"] = {
            "agent": {"cpu": 8, "memory_mib": 65536},
            "verifier": {"cpu": 8, "memory_mib": 65536},
        }
        images["gpu_sidecars"] = {
            "scheduler": {
                "image": {"tag": "example/sidecar:t", "digest": "sha256:" + "5" * 64}
            }
        }
        with mock.patch.object(infra.urllib_request, "urlopen", side_effect=self.http):
            result = self.replay()
        self.assertEqual(result["status"], "PASS")
        self.assertNotIn("gpu_parent_resources", result["image_digests"])
        self.assertEqual(
            result["image_digests"]["gpu_sidecar:scheduler"], "sha256:" + "5" * 64
        )
        self.assertEqual(
            result["image_digests"]["verifier"], images["verifier"]["digest"]
        )

    def test_identical_candidate_409_is_idempotent_but_different_digest_is_rejected(
        self,
    ):
        self.conflict = True
        with mock.patch.object(infra.urllib_request, "urlopen", side_effect=self.http):
            self.assertEqual(self.replay()["status"], "PASS")
        self.receipt["tasks"] = {}
        self.stored[0]["definition_digest"] = "sha256:" + "e" * 64
        with (
            mock.patch.object(infra.urllib_request, "urlopen", side_effect=self.http),
            self.assertRaises(deploy.DeployError),
        ):
            self.replay()

    def test_not_comparable_records_evidence_and_is_terminal(self):
        self.baseline.update(comparable=False, comparability_note="earlier revision")
        self.run["score"] = self.baseline["historical_mean"] + 1000
        with mock.patch.object(infra.urllib_request, "urlopen", side_effect=self.http):
            result = self.replay()
        self.assertEqual(result["status"], "NOT_COMPARABLE")
        self.assertEqual(result["failure_summaries"], [])
        self.assertEqual(result["score"], self.run["score"])
        self.assertEqual(result["evaluation_id"], self.evaluation["id"])
        self.assertEqual(result["run_id"], self.run["id"])
        self.assertEqual(result["train_verification_id"], "verification_test")
        self.assertIs(result["comparable"], False)
        self.assertEqual(result["provenance"], "historical-authoring")
        self.assertEqual(result["comparability_note"], "earlier revision")
        self.receipt = json.loads(self.path.read_text())
        with mock.patch.object(infra.urllib_request, "urlopen") as http:
            self.assertEqual(self.replay(), result)
            http.assert_not_called()

    def test_not_comparable_still_fails_invalid_or_unsuccessful_runs(self):
        self.baseline["comparable"] = False
        for field, value in (
            ("score", None),
            ("score", True),
            ("score", float("nan")),
            ("score", float("inf")),
            ("state", "FAILED"),
            ("outcome", "FAILED"),
        ):
            with self.subTest(field=field, value=value):
                failures = parity.verdict(
                    self.evaluation, {**self.run, field: value}, self.baseline, "train"
                )
                self.assertTrue(failures)
                self.assertFalse(any("tolerance" in failure for failure in failures))
        for evaluation_state, outcome, events in (
            ("FAILED", "FAILED", self.events),
            ("COMPLETED", "SUCCEEDED", []),
        ):
            self.receipt["tasks"] = {}
            self.evaluation["status"] = evaluation_state
            self.run["outcome"] = outcome
            self.events = events
            with mock.patch.object(
                infra.urllib_request, "urlopen", side_effect=self.http
            ):
                self.assertEqual(self.replay()["status"], "FAIL")

    def test_classification_binding_compares_all_keys_and_their_presence(self):
        self.interrupt_observation()
        original = copy.deepcopy(self.receipt["tasks"][self.name])
        baseline = copy.deepcopy(self.baseline)
        classification = {
            "provenance": baseline["provenance"],
            "comparable": baseline["comparable"],
        }
        with_note = {**classification, "comparability_note": "earlier revision"}
        cases = [
            (classification, with_note),
            (with_note, classification),
            (with_note, {**with_note, "comparability_note": "another revision"}),
            (classification, {**classification, "comparability_note": None}),
            ({**classification, "comparability_note": None}, classification),
            ({"provenance": baseline["provenance"]}, classification),
            ({"comparable": baseline["comparable"]}, classification),
            ({"comparability_note": "earlier revision"}, with_note),
            (classification, {**classification, "comparable": False}),
            (classification, {**classification, "provenance": "unverified"}),
        ]
        for status in ("RUNNING", "PASS"):
            for previous, current in cases:
                with self.subTest(status=status, previous=previous, current=current):
                    record = {
                        key: value
                        for key, value in original.items()
                        if key not in ("provenance", "comparable", "comparability_note")
                    }
                    record.update(previous, status=status)
                    self.receipt["tasks"][self.name] = record
                    self.baseline = {**baseline, **current}
                    parity.save_receipt(self.path, self.receipt)
                    before = self.path.read_bytes()
                    with mock.patch.object(infra.urllib_request, "urlopen") as http:
                        with self.assertRaisesRegex(
                            ValueError, "use a new receipt and attempt"
                        ):
                            self.replay()
                        http.assert_not_called()
                    self.assertEqual(self.path.read_bytes(), before)

    def test_legacy_unfinished_receipts_acquire_classification_on_resume(self):
        for comparable in (True, False):
            with self.subTest(comparable=comparable):
                self.receipt["tasks"] = {}
                self.interrupt_observation()
                record = self.receipt["tasks"][self.name]
                for key in ("provenance", "comparable", "comparability_note"):
                    record.pop(key, None)
                parity.save_receipt(self.path, self.receipt)
                self.receipt = json.loads(self.path.read_text())
                self.baseline.update(
                    comparable=comparable, comparability_note="classification on resume"
                )
                self.calls = []
                with mock.patch.object(
                    infra.urllib_request, "urlopen", side_effect=self.http
                ):
                    result = self.replay()
                self.assertEqual(
                    result["status"], "PASS" if comparable else "NOT_COMPARABLE"
                )
                self.assertIs(result["comparable"], comparable)
                self.assertEqual(result["provenance"], self.baseline["provenance"])
                self.assertEqual(
                    result["comparability_note"], self.baseline["comparability_note"]
                )
                self.assertEqual(result["evaluation_id"], record["evaluation_id"])
                self.assertEqual(result["run_id"], record["run_id"])
                self.assertEqual(result["train_verification_id"], "verification_test")
                self.assertEqual(result["failure_summaries"], [])
                self.assertFalse(any(method == "POST" for method, _, _ in self.calls))
                self.assertEqual(json.loads(self.path.read_text()), self.receipt)

    def test_unverified_baseline_without_scores_can_replay_and_is_not_comparable(self):
        self.baseline = {"provenance": "unverified", "comparable": False}
        parity.validate_baseline(
            self.name, self.baseline, self.image, deploy.TASK_DEFINITIONS[self.name]
        )
        with mock.patch.object(infra.urllib_request, "urlopen", side_effect=self.http):
            result = self.replay()
        self.assertEqual(result["status"], "NOT_COMPARABLE")
        self.assertIsNone(result["historical_mean"])

    def test_baselines_must_be_classified(self):
        for key in ("provenance", "comparable"):
            changed = dict(self.baseline)
            del changed[key]
            with self.assertRaisesRegex(ValueError, "baselines must be classified"):
                parity.validate_baseline(
                    self.name, changed, self.image, deploy.TASK_DEFINITIONS[self.name]
                )
        for key, value in (
            ("provenance", "unknown"),
            ("provenance", []),
            ("comparable", "false"),
            ("comparable", 0),
            ("comparable", None),
        ):
            with (
                self.subTest(key=key, value=value),
                self.assertRaisesRegex(ValueError, "baselines must be classified"),
            ):
                parity.validate_baseline(
                    self.name,
                    {**self.baseline, key: value},
                    self.image,
                    deploy.TASK_DEFINITIONS[self.name],
                )
        for provenance in ("measured", "historical-authoring", "unverified"):
            for comparable in (False,) if provenance == "unverified" else (True, False):
                parity.validate_baseline(
                    self.name,
                    {
                        **self.baseline,
                        "provenance": provenance,
                        "comparable": comparable,
                    },
                    self.image,
                    deploy.TASK_DEFINITIONS[self.name],
                )

    def test_score_agreement_tolerances_and_all_terminal_conditions(self):
        # Fixed mathematical boundaries keep the rule independent of whichever
        # actual calibration observation is currently published for each task.
        for mean, tolerance in (
            (0.0, 0.01),
            (20.0, 1.01),
            (-20.0, 1.01),
            (100.0, 5.01),
            (1.23, 0.0715),
        ):
            baseline = {**self.baseline, "historical_mean": mean}
            for sign in (-1, 1):
                run = {
                    **self.run,
                    "score": baseline["historical_mean"] + sign * (tolerance - 1e-8),
                }
                self.assertEqual(
                    parity.verdict(self.evaluation, run, baseline, "verification_test"),
                    [],
                )
                run["score"] = baseline["historical_mean"] + sign * (tolerance + 1e-8)
                self.assertTrue(
                    parity.verdict(self.evaluation, run, baseline, "verification_test")
                )
        for field, value in (
            ("score", None),
            ("score", True),
            ("score", float("nan")),
            ("score", float("inf")),
            ("outcome", "FAILED"),
            ("state", "FAILED"),
        ):
            self.assertTrue(
                parity.verdict(
                    self.evaluation,
                    {**self.run, field: value},
                    self.baseline,
                    "verification_test",
                )
            )
        self.assertTrue(
            parity.verdict(
                {**self.evaluation, "status": "PARTIAL"},
                self.run,
                self.baseline,
                "verification_test",
            )
        )
        self.assertTrue(parity.verdict(self.evaluation, self.run, self.baseline, None))

    def test_completed_eval_without_successful_train_fails(self):
        self.events[0]["payload"]["status"] = "fail"
        with mock.patch.object(infra.urllib_request, "urlopen", side_effect=self.http):
            self.assertEqual(self.replay()["status"], "FAIL")
        self.assertIn(
            "No successful train",
            self.receipt["tasks"][self.name]["failure_summaries"][0],
        )

    def test_timeout_cancels_only_its_own_evaluation_and_records_timeout(self):
        self.run["state"] = "RUNNING"
        self.evaluation["status"] = "RUNNING"

        def expired(request, timeout):
            response = self.http(request, timeout)
            if request.get_method() == "POST" and request.full_url.endswith(
                "/evaluations"
            ):
                self.now = 10**9
            return response

        with mock.patch.object(infra.urllib_request, "urlopen", side_effect=expired):
            result = self.replay()
        self.assertEqual(result["status"], "TIMEOUT")
        quota = self.task_document["driver_config"]["quota"]
        self.assertEqual(
            result["deadline_unix"],
            quota["wall_time_seconds"]
            + quota["verifier"]["eval_timeout_seconds"]
            + 900,
        )
        cancels = [call for call in self.calls if call[1].endswith("/cancel")]
        self.assertEqual(len(cancels), 1)
        self.assertEqual(
            cancels[0][:2], ("POST", "/api/v1/evaluations/evaluation_test/cancel")
        )
        self.assertTrue(result["cancellation_requested"])
        self.assertFalse(
            any(
                method == "GET" and ("/runs/" in path or "/evaluations/" in path)
                for method, path, _ in self.calls
            )
        )

    def interrupt_observation(self):
        def interrupted(request, timeout):
            if request.full_url.endswith("/runs/run_test"):
                raise KeyboardInterrupt
            return self.http(request, timeout)

        with (
            mock.patch.object(infra.urllib_request, "urlopen", side_effect=interrupted),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.replay()
        self.receipt = json.loads(self.path.read_text())

    def test_resume_preserves_ids_deadline_and_skips_terminal_tasks(self):
        self.interrupt_observation()
        deadline = self.receipt["tasks"][self.name]["deadline_unix"]
        with mock.patch.object(infra.urllib_request, "urlopen", side_effect=self.http):
            self.assertEqual(self.replay()["status"], "PASS")
        self.assertEqual(self.receipt["tasks"][self.name]["deadline_unix"], deadline)
        self.assertEqual(
            sum(call[:2] == ("POST", "/api/v1/evaluations") for call in self.calls), 1
        )
        with mock.patch.object(infra.urllib_request, "urlopen") as http:
            self.assertEqual(self.replay()["status"], "PASS")
            http.assert_not_called()
        self.image["digest"] = "sha256:" + "e" * 64
        with self.assertRaisesRegex(ValueError, "different inputs"):
            self.replay()

    def test_recoverable_observation_errors_retry_each_endpoint_and_persist_safe_details(
        self,
    ):
        for endpoint in (
            "/evaluations/evaluation_test",
            "/runs/run_test",
            "/events?limit=500",
        ):
            for kind in ("connection", "url_error", "http_503", "timeout", "oversized"):
                with self.subTest(endpoint=endpoint, kind=kind):
                    self.receipt["tasks"] = {}
                    failed = False

                    def flaky(request, timeout, *, endpoint=endpoint, kind=kind):
                        nonlocal failed
                        if request.full_url.endswith(endpoint) and not failed:
                            failed = True
                            if kind == "connection":
                                raise ConnectionError("private error detail")
                            if kind == "url_error":
                                raise URLError("private error detail")
                            if kind == "http_503":
                                raise HTTPError(
                                    request.full_url,
                                    503,
                                    "unavailable",
                                    {},
                                    io.BytesIO(b"private error detail"),
                                )
                            if kind == "timeout":
                                raise TimeoutError("private error detail")
                            response = self.http(request, timeout)
                            response.read.return_value = b"x" * (
                                parity.RESPONSE_LIMIT + 1
                            )
                            return response
                        return self.http(request, timeout)

                    with mock.patch.object(
                        infra.urllib_request, "urlopen", side_effect=flaky
                    ):
                        result = self.replay()
                    self.assertEqual(result["status"], "PASS")
                    self.assertEqual(result["observation_error_count"], 1)
                    self.assertEqual(len(result["observation_errors"]), 1)
                    self.assertNotIn("private error detail", self.path.read_text())
                    self.assertEqual(json.loads(self.path.read_text()), self.receipt)

    def test_persistent_event_failure_expires_and_cancels_with_bounded_history(self):
        self.interrupt_observation()
        record = self.receipt["tasks"][self.name]
        record["deadline_unix"] = 0.025
        timeouts = []

        def unavailable(request, timeout):
            if "/events?" in request.full_url:
                timeouts.append(timeout)
                raise HTTPError(
                    request.full_url,
                    503,
                    "unavailable",
                    {},
                    io.BytesIO(b"private error detail"),
                )
            return self.http(request, timeout)

        with mock.patch.object(
            infra.urllib_request, "urlopen", side_effect=unavailable
        ):
            result = self.replay()
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertGreater(
            result["observation_error_count"], parity.ERROR_HISTORY_LIMIT
        )
        self.assertEqual(len(result["observation_errors"]), parity.ERROR_HISTORY_LIMIT)
        self.assertEqual(result["deadline_unix"], 0.025)
        self.assertTrue(result["cancellation_requested"])
        self.assertTrue(all(0 < timeout <= 0.025 for timeout in timeouts))
        self.assertNotIn("private error detail", self.path.read_text())

    def test_cancel_failure_retries_with_same_identity_then_records_timeout(self):
        self.interrupt_observation()
        self.now = 10**9
        cancels = []

        def flaky_cancel(request, timeout):
            self.assertTrue(request.full_url.endswith("/cancel"))
            cancels.append(request)
            if len(cancels) < 3:
                raise ConnectionError("private cancellation detail")
            return self.http(request, timeout)

        with mock.patch.object(
            infra.urllib_request, "urlopen", side_effect=flaky_cancel
        ):
            result = self.replay()
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertEqual(result["cancellation_attempts"], 3)
        self.assertEqual(result["cancellation_error_count"], 2)
        self.assertEqual(
            len({call.get_header("Idempotency-key") for call in cancels}), 1
        )
        self.assertNotIn("private cancellation detail", self.path.read_text())

    def test_cancel_retry_exhaustion_is_bounded_and_resumable(self):
        self.interrupt_observation()
        self.now = 10**9
        deadline = self.receipt["tasks"][self.name]["deadline_unix"]

        def unavailable(request, timeout):
            self.assertTrue(request.full_url.endswith("/cancel"))
            raise HTTPError(
                request.full_url,
                503,
                "unavailable",
                {},
                io.BytesIO(b"private cancellation detail"),
            )

        for resume in range(5):
            with mock.patch.object(
                infra.urllib_request, "urlopen", side_effect=unavailable
            ) as http:
                result = self.replay()
            self.assertEqual(http.call_count, parity.CANCEL_ATTEMPTS)
            self.assertEqual(result["status"], "CANCEL_PENDING")
            self.assertFalse(result["cancellation_requested"])
            self.assertEqual(
                result["cancellation_attempts"], (resume + 1) * parity.CANCEL_ATTEMPTS
            )
            self.receipt = json.loads(self.path.read_text())
        self.assertEqual(len(result["cancellation_errors"]), parity.ERROR_HISTORY_LIMIT)
        with mock.patch.object(infra.urllib_request, "urlopen", side_effect=self.http):
            result = self.replay()
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertEqual(result["deadline_unix"], deadline)
        self.assertEqual(result["evaluation_id"], "evaluation_test")
        self.assertEqual(
            sum(call[:2] == ("POST", "/api/v1/evaluations") for call in self.calls), 1
        )

    def test_interruption_before_post_rejects_expired_unsubmitted_receipt(self):
        save = parity.save_receipt

        def interrupted(path, receipt):
            save(path, receipt)
            if receipt["tasks"][self.name].get("submission_state") == "UNSUBMITTED":
                raise KeyboardInterrupt

        with (
            mock.patch.object(infra.urllib_request, "urlopen", side_effect=self.http),
            mock.patch.object(parity, "save_receipt", side_effect=interrupted),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.replay()
        self.receipt = json.loads(self.path.read_text())
        record = self.receipt["tasks"][self.name]
        self.assertIsNone(record["evaluation_id"])
        self.assertFalse(
            any(call[:2] == ("POST", "/api/v1/evaluations") for call in self.calls)
        )
        self.now = record["deadline_unix"]
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                if legacy:
                    record.pop("submission_state")
                with mock.patch.object(infra.urllib_request, "urlopen") as http:
                    with self.assertRaisesRegex(
                        ValueError, "unsubmitted parity task deadline expired"
                    ):
                        self.replay()
                    http.assert_not_called()

    def test_lost_post_response_recovers_same_evaluation_and_deadline_after_expiry(
        self,
    ):
        accepted = {}
        posts = []

        def lost_response(request, timeout):
            if request.get_method() == "POST" and request.full_url.endswith(
                "/evaluations"
            ):
                identity = (request.get_header("Idempotency-key"), request.data)
                posts.append(identity)
                if identity not in accepted:
                    accepted[identity] = self.http(request, timeout)
                    raise ConnectionError("response lost after submission")
                return accepted[identity]
            return self.http(request, timeout)

        with mock.patch.object(
            infra.urllib_request, "urlopen", side_effect=lost_response
        ):
            with self.assertRaises(parity.ManagementError):
                self.replay()
            self.receipt = json.loads(self.path.read_text())
            record = self.receipt["tasks"][self.name]
            self.assertEqual(record["submission_state"], "SUBMITTING")
            self.assertIsNone(record["evaluation_id"])
            deadline = record["deadline_unix"]
            self.now = deadline + 1
            result = self.replay()
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertEqual(result["submission_state"], "SUBMITTED")
        self.assertEqual(result["evaluation_id"], "evaluation_test")
        self.assertEqual(result["deadline_unix"], deadline)
        self.assertEqual(result["submission_error_count"], 1)
        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[0], posts[1])
        self.assertEqual(len(accepted), 1)

    def test_resume_recovers_successful_train_outside_first_event_page(self):
        self.interrupt_observation()
        queries = []
        cursor = "older/events+cursor="

        def paginated(request, timeout):
            if "/events?" in request.full_url:
                query = parse_qs(urlsplit(request.full_url).query)
                queries.append(query)
                response = (
                    self.http(request, timeout)
                    if "cursor" not in query
                    else mock.MagicMock()
                )
                response.__enter__.return_value = response
                if "cursor" not in query:
                    response.read.return_value = json.dumps(
                        [{"run_id": "run_test", "kind": "HEARTBEAT"}] * 500
                    ).encode()
                    response.headers = {"X-Next-Cursor": cursor}
                else:
                    self.assertEqual(query["cursor"], [cursor])
                    response.read.return_value = json.dumps(self.events).encode()
                    response.headers = {}
                return response
            return self.http(request, timeout)

        with mock.patch.object(infra.urllib_request, "urlopen", side_effect=paginated):
            result = self.replay()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["train_verification_id"], "verification_test")
        self.assertEqual(len(queries), 2)
        self.assertEqual(
            sum(call[:2] == ("POST", "/api/v1/evaluations") for call in self.calls), 1
        )

    def test_unrelated_run_cannot_be_scored_or_cancelled(self):
        self.run["evaluation_id"] = "unrelated"
        with (
            mock.patch.object(infra.urllib_request, "urlopen", side_effect=self.http),
            self.assertRaisesRegex(ValueError, "does not belong"),
        ):
            self.replay()
        self.assertFalse(any(call[1].endswith("/cancel") for call in self.calls))

    def test_baselines_match_native_evaluator_adapter_dataset_and_reference_inputs(
        self,
    ):
        baselines = references.load_baselines()
        images = references.load_manifest()
        for name in images:
            parity.validate_baseline(
                name, baselines[name], images[name], deploy.TASK_DEFINITIONS[name]
            )
        for field in (
            "reference_patch_sha256",
            "evaluator_sha256",
            "verifier_adapter_sha256",
            "source_set_sha256",
        ):
            changed = {**self.baseline, field: "sha256:" + "e" * 64}
            with self.assertRaises(ValueError):
                parity.validate_baseline(
                    self.name, changed, self.image, deploy.TASK_DEFINITIONS[self.name]
                )

    def main_inputs(
        self, *, modal_enabled=True, schema_version="v1", domains=None, catalog=None
    ):
        directory = Path(self.temp.name)
        raw = resolved_catalog() if catalog is None else copy.deepcopy(catalog)
        if catalog is None:
            next(t for t in raw["tasks"] if t["name"] == self.name)["baseline"] = (
                copy.deepcopy(self.baseline)
            )
        config = deploy.load_json(ROOT / "config/site.example.json")
        config["features"]["modal_gpu"] = modal_enabled
        if domains is not None:
            config["execution"]["model_domains"] = domains
        selection = deploy.select_tasks(config, raw, deploy.TASK_DEFINITIONS)
        (directory / "site.json").write_text(json.dumps(config))
        images = references.load_manifest()
        for image in images.values():
            image["digest"] = "sha256:" + "d" * 64
        for task in raw["tasks"]:
            task["reference"] = images.get(task["name"])
        (directory / "catalog.json").write_text(json.dumps(raw))
        self.mirrors["reference_candidates"] = {self.name: self.mirror}
        (directory / "task-images.json").write_text(json.dumps(self.mirrors))
        seed = {
            "schema_version": "frontier.deployment-task-seed-receipt/v1",
            "site": "preview",
            "model_domains": selection.model_domains,
            "catalog_sha256": selection.effective_catalog_sha256,
            "candidate_version_id": deploy.candidate_version_document(
                self.catalog["candidate"], self.mirrors["candidate"]["overlay_image"]
            )["driver_config"]["candidate_version_id"],
            "task_version_ids": [],
        }
        tasks = {}
        for task in self.catalog["tasks"]:
            name = task["name"]
            if name in selection.selected:
                document = deploy.task_version_document(
                    task, self.mirrors[name], model_domains=selection.model_domains
                )
                tasks[name] = {
                    "task_version_id": document["driver_config"]["task_version_id"],
                    "definition_digest": document["definition_digest"],
                }
                seed["task_version_ids"].append(tasks[name]["task_version_id"])
        if schema_version == "v2":
            seed.update(
                schema_version="frontier.deployment-task-seed-receipt/v2",
                tasks=tasks,
                **{
                    key: getattr(selection, key)
                    for key in (
                        "catalog_sha256",
                        "effective_catalog_sha256",
                        "selection_digest",
                        "selected",
                        "excluded",
                        "model_domains",
                    )
                },
            )
        seed_path = directory / "seed.json"
        seed_path.write_text(json.dumps(seed))
        args = [
            "--task",
            self.name,
            "--management-url",
            "http://localhost",
            "--receipt",
            str(self.path),
            "--seed-receipt",
            str(seed_path),
            "--catalog",
            str(directory / "catalog.json"),
            "--config",
            str(directory / "site.json"),
        ]
        return args, seed_path, seed

    def test_main_reads_seed_and_mirror_receipts_and_rejects_stale_seed(self):
        args, seed_path, seed = self.main_inputs()
        with mock.patch.object(infra.urllib_request, "urlopen", side_effect=self.http):
            self.assertEqual(parity.reference_parity_main(args), 0)
        seed["task_version_ids"] = []
        seed_path.write_text(json.dumps(seed))
        with self.assertRaisesRegex(deploy.DeployError, "absent from the seed receipt"):
            parity.reference_parity_main(args)

    def test_legacy_terminal_pass_fails_cli_gate_without_rewriting_receipt(self):
        args, _, _ = self.main_inputs()
        with (
            mock.patch.object(infra.urllib_request, "urlopen", side_effect=self.http),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            self.assertEqual(parity.reference_parity_main(args), 0)
        receipt = json.loads(self.path.read_text())
        record = receipt["tasks"][self.name]
        for key in ("provenance", "comparable", "comparability_note"):
            record.pop(key, None)
        # Keep deliberately different formatting to detect any rewrite.
        self.path.write_text(json.dumps(receipt, indent=4) + "\n")
        before = self.path.read_bytes()
        baselines = references.load_baselines()
        baseline_path = Path(self.temp.name) / "catalog.json"
        for comparable in (False, True):
            with self.subTest(comparable=comparable):
                baselines[self.name]["comparable"] = comparable
                catalog = deploy.load_json(baseline_path)
                for task in catalog["tasks"]:
                    task["baseline"] = baselines[task["name"]]
                baseline_path.write_text(json.dumps(catalog))
                with (
                    mock.patch.object(infra.urllib_request, "urlopen") as http,
                    mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
                ):
                    self.assertEqual(parity.reference_parity_main(args), 1)
                    http.assert_not_called()
                self.assertEqual(
                    stdout.getvalue(),
                    f"{self.name}: PASS (unclassified legacy receipt — rerun with a new --receipt/--attempt)\n",
                )
                self.assertEqual(self.path.read_bytes(), before)

    def test_all_summary_distinguishes_not_comparable_from_pass(self):
        # --all preserves catalog order, including non-alphabetical imports.
        catalog = resolved_catalog()
        catalog["tasks"].sort(key=lambda task: task["name"], reverse=True)
        args, seed_path, seed = self.main_inputs(catalog=catalog)
        references_path = Path(self.temp.name) / "catalog.json"
        images = references.load_manifest(references_path)
        seed["task_version_ids"] = [
            deploy.task_version_document(task, self.mirrors[task["name"]])[
                "driver_config"
            ]["task_version_id"]
            for task in self.catalog["tasks"]
        ]
        seed_path.write_text(json.dumps(seed))
        self.mirrors["reference_candidates"] = {
            name: {
                **self.mirror,
                "source_reference": image["repository"] + "@" + image["digest"],
            }
            for name, image in images.items()
        }
        (Path(self.temp.name) / "task-images.json").write_text(json.dumps(self.mirrors))
        baselines = references.load_baselines()
        # Explicit mixed classifications; published calibration can change.
        baselines[self.name]["comparable"] = False
        catalog = deploy.load_json(references_path)
        for task in catalog["tasks"]:
            task["baseline"] = baselines[task["name"]]
        references_path.write_text(json.dumps(catalog))

        def replay(base, name, document, task_document, baseline, *args):
            return {
                "status": "PASS" if baseline["comparable"] else "NOT_COMPARABLE",
                "provenance": baseline["provenance"],
                "comparable": baseline["comparable"],
            }

        with (
            mock.patch.object(parity, "replay_task", side_effect=replay),
            mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            # An unmeasured comparison cannot satisfy the all-PASS gate.
            self.assertEqual(parity.reference_parity_main(["--all", *args[2:]]), 1)
        self.assertEqual(
            stdout.getvalue().splitlines(),
            [
                f"{name}: {'PASS' if baselines[name]['comparable'] else 'NOT_COMPARABLE'}"
                for name in images
            ],
        )

    def test_catalog_hash_mismatch_reports_paths_hashes_and_exact_seeding_requirement(
        self,
    ):
        args, seed_path, seed = self.main_inputs()
        catalog_path = Path(self.temp.name) / "catalog.json"
        actual_hash = seed["catalog_sha256"]
        seed["catalog_sha256"] = "sha256:" + "e" * 64
        seed_path.write_text(json.dumps(seed))
        with mock.patch.object(infra.urllib_request, "urlopen") as http:
            with self.assertRaises(ValueError) as raised:
                parity.reference_parity_main(args)
            http.assert_not_called()
        message = str(raised.exception)
        for detail in (
            str(seed_path),
            str(catalog_path),
            seed["catalog_sha256"],
            actual_hash,
            "does not bind current selection",
        ):
            self.assertIn(detail, message)

    def test_cpu_selection_accepts_v1_and_v2_receipts(self):
        for schema in ("v1", "v2"):
            with self.subTest(schema=schema):
                self.path.unlink(missing_ok=True)
                args, _, seed = self.main_inputs(
                    modal_enabled=False, schema_version=schema
                )
                with mock.patch.object(
                    infra.urllib_request, "urlopen", side_effect=self.http
                ):
                    self.assertEqual(parity.reference_parity_main(args), 0)
                expected = [
                    task["name"]
                    for task in self.catalog["tasks"]
                    if not deploy.requirements(deploy.TASK_DEFINITIONS[task["name"]])[
                        "modal_gpu"
                    ]
                ]
                self.assertEqual(len(seed["task_version_ids"]), len(expected))

    def test_v2_rejects_stale_selection_and_task_bindings_before_http(self):
        args, seed_path, seed = self.main_inputs(
            modal_enabled=False, schema_version="v2"
        )
        mutations = {
            "catalog_sha256": "sha256:" + "0" * 64,
            "effective_catalog_sha256": "sha256:" + "0" * 64,
            "selection_digest": "sha256:" + "0" * 64,
            "site": "other",
            "selected": list(reversed(seed["selected"])),
            "excluded": {},
            "model_domains": ["changed.example"],
            "schema_version": "unknown",
        }
        variants = [{**seed, key: value} for key, value in mutations.items()]
        bad_task = copy.deepcopy(seed)
        bad_task["tasks"][self.name]["definition_digest"] = "sha256:" + "0" * 64
        variants.append(bad_task)
        for changed in variants:
            with self.subTest(receipt=changed):
                seed_path.write_text(json.dumps(changed))
                with mock.patch.object(infra.urllib_request, "urlopen") as http:
                    with self.assertRaises((ValueError, deploy.DeployError)):
                        parity.reference_parity_main(args)
                http.assert_not_called()

    def test_excluded_task_refused_before_http_even_after_a_selected_task(self):
        args, _, _ = self.main_inputs(modal_enabled=False)
        with mock.patch.object(infra.urllib_request, "urlopen") as http:
            with self.assertRaisesRegex(
                ValueError,
                "sparse_video_attention_policy is excluded: requires features.modal_gpu",
            ):
                parity.reference_parity_main(
                    [*args, "--task", "sparse_video_attention_policy"]
                )
        http.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_parity_reconstructs_task_with_selection_model_domains(self):
        def replay(base, name, document, task_document, baseline, *args):
            self.assertEqual(
                task_document["driver_config"]["network"]["allowed_domains"],
                ["api.moonshot.ai", "openrouter.ai"],
            )
            self.assertIn(
                task_document["driver_config"]["task_version_id"],
                seed["task_version_ids"],
            )
            return {
                "status": "PASS",
                "provenance": baseline["provenance"],
                "comparable": baseline["comparable"],
            }

        for schema in ("v1", "v2"):
            args, seed_path, seed = self.main_inputs(
                modal_enabled=False,
                schema_version=schema,
                domains=["openrouter.ai", "api.moonshot.ai"],
            )
            with (
                self.subTest(schema=schema),
                mock.patch.object(parity, "replay_task", side_effect=replay) as called,
            ):
                self.assertEqual(parity.reference_parity_main(args), 0)
            called.assert_called_once()
            seed["model_domains"] = ["different.example"]
            seed_path.write_text(json.dumps(seed))
            with mock.patch.object(
                infra.urllib_request, "urlopen", side_effect=self.http
            ) as http:
                # Positive control: the real parity transport reaches this mock.
                parity.api_json(
                    "http://localhost", "GET", "/api/v1/task-versions/probe"
                )
                http.assert_called_once()
                http.reset_mock()
                with self.assertRaisesRegex(ValueError, "model_domains"):
                    parity.reference_parity_main(args)
                http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
