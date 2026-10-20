from __future__ import annotations

# All tests remain offline, including real evidence checks.
import deployment_test_guard  # noqa: F401  # isort: skip
import copy
import io
import json
import math
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import import_task as importer
import test_import_task as import_tests
from preview import measured_baseline as measured


def archive(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, body in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            info.mode = 0o444
            tar.addfile(info, io.BytesIO(body))
    return output.getvalue()


class PacketFixture:
    """Synthetic receipts, never historical author evidence or network responses."""

    def __init__(
        self,
        source,
        definition,
        entry,
        directory,
        launcher,
        *,
        json_payload=False,
        dbscan=False,
        radix=False,
    ):
        self.directory, self.source = directory, source
        self.packet_path = directory / "measurement.json"
        directory.mkdir()
        self.docs, self.artifacts = {}, {}
        self.entry, self.definition = entry, definition
        self.payload = source / (
            "judge/reference.json" if json_payload else "judge/reference.patch"
        )
        self.launcher = launcher
        self.tid, self.cid, self.eid, self.rid, self.aid = [
            p + "_01AAAAAAAAAAAAAAAAAAAAAAAA"
            for p in ("taskver", "candver", "eval", "run", "attempt")
        ]
        self.vids = {
            mode: "verification_01" + letter * 24
            for mode, letter in (("train", "B"), ("eval", "C"))
        }
        self.pids = {
            mode: "payload_01" + letter * 24
            for mode, letter in (("train", "D"), ("final", "E"))
        }
        identity = definition["source"]["source_set_sha256"]
        policy = json.loads((source / "policy.json").read_text())
        s3 = "s3://example-build-bucket/"
        registry = "registry.example.com/build/"
        images = {
            role: {
                "reference": registry + image + ":synthetic",
                "digest": entry[public]["digest"],
            }
            for role, image, public in (
                ("agent_cpu_base", "agent", "agent_image"),
                ("verifier", "verifier", "verifier_image"),
            )
        }
        pins = {
            mode: {
                "uri": s3
                + f"datasets/frontier-cs-2.0/{source.name}/{mode}/sha256/"
                + definition["datasets"][mode + "_sha256"][7:]
                + ".json",
                "body_sha256": definition["datasets"][mode + "_sha256"],
                "size_bytes": (source / "datasets" / (mode + ".json")).stat().st_size,
                "s3_version_id": "synthetic-immutable-version",
            }
            for mode in ("train", "eval")
        }
        config = {
            "schema_version": "frontier.task-config/v1",
            "task_version_id": self.tid,
            "images": copy.deepcopy(images),
            "datasets": {
                mode: {"uri": pin["uri"], "version": pin["body_sha256"]}
                for mode, pin in pins.items()
            },
            "agent": policy["agent"],
            "network": policy["network"],
            "quota": {
                "wall_time_seconds": policy["quota"]["wall_time_seconds"],
                "agent_gpu": measured.GPU["agent"],
                "verifier": {
                    **measured.GPU["verifier"],
                    **{
                        key: policy["quota"][key]
                        for key in (
                            "max_train_calls",
                            "max_train_wall_seconds",
                            "eval_timeout_seconds",
                        )
                    },
                },
            },
            "submission": {
                **policy["submission"],
                "media_type": "application/x-tar",
                "required_sha256": True,
            },
            "verifier": {
                **policy["verifier"],
                "result_path": "/frontier/output/result.json",
                "evidence_path": "/frontier/output/evidence.tar.zst",
            },
        }
        config["images"]["agent_cpu_base"]["platform"] = dict(measured.PLATFORM)
        config_sha = measured.sha(measured.native_bytes(config))
        definition_ref = (
            s3
            + f"definitions/frontier-cs-2.0/{source.name}/sha256/"
            + config_sha[7:]
            + ".json"
        )
        author = {
            "schema_version": "frontier.task-author-release/v1",
            "task_version_id": self.tid,
            "source": {
                "sha256": identity,
                "reference": "https://example.invalid/synthetic-source",
            },
            "build": {
                "inputs_sha256": identity,
                "platform": measured.PLATFORM,
                "tool_versions": {"source-identity": "preview-source-identity/v1"},
            },
            "images": images,
            "datasets": pins,
            "definition": {"ref": definition_ref, "digest": config_sha},
        }
        task = {
            "task_id": source.name,
            "version": "synthetic",
            "definition_ref": definition_ref,
            "definition_digest": config_sha,
            "driver_config": config,
            "author_receipt": author,
            "metadata": {
                "source_digest": identity,
                "source_identity_algorithm": "preview-source-identity/v1",
                "publication_mode": "prepared-exact-preview-image-adoption",
            },
        }
        self.docs["task_create"] = task
        self.docs["task_catalog"] = {
            key: task[key]
            for key in (
                "task_id",
                "version",
                "definition_ref",
                "definition_digest",
                "metadata",
            )
        }
        self.docs["task_catalog"].update(id=self.tid, archived_at=None)
        self.docs["task_publication"] = {
            "task_version_id": self.tid,
            "definition_ref": definition_ref,
            "definition_digest": config_sha,
            "images": images,
            "datasets": pins,
        }
        ref = entry["reference"]
        cconfig = {
            "schema_version": "frontier.candidate-config/v1",
            "candidate_version_id": self.cid,
            "overlay_image": {
                "reference": registry + "candidate@" + ref["digest"],
                "digest": ref["digest"],
                "format": "frontier.oci-overlay/v1",
                "platform": measured.PLATFORM,
            },
            "execution": copy.deepcopy(measured.REFERENCE_EXECUTION),
            "model_api": {"allowed_domains": [], "secret_refs": []},
        }
        csha = measured.sha(measured.native_bytes(cconfig))
        cref = (
            s3
            + f"definitions/frontier-cs-2.0/reference-candidates/{source.name}/sha256/"
            + csha[7:]
            + ".json"
        )
        candidate = {
            "candidate_id": "frontiersmith2-preview-reference-"
            + source.name.replace("_", "-"),
            "version": "synthetic",
            "definition_digest": csha,
            "definition_ref": cref,
            "driver_config": cconfig,
            "metadata": {
                "source": "frontiercs2-preview-reference",
                "model_execution": "deterministic-reference-adopted",
                "profile": "preview-reference-v1",
                "problem_id": source.name,
                "source_sha256": identity,
                "launcher_sha256": ref["launcher_sha256"],
                "payload_sha256": ref["payload_sha256"],
                "public_image_reference": ref["repository"] + "@" + ref["digest"],
            },
        }
        self.docs["candidate_create"] = candidate
        binding = {
            "schema_version": "frontier.frontiersmith-reference-adoption/v1",
            "profile": "preview-reference-v1",
            "source": {"path": "/synthetic/unused", "sha256": identity},
            "request_sha256": measured.sha(measured.native_bytes(candidate)),
            "launcher": {
                "sha256": ref["launcher_sha256"],
                "size_bytes": launcher.stat().st_size,
            },
            "payload": {
                "sha256": ref["payload_sha256"],
                "size_bytes": self.payload.stat().st_size,
            },
            "image_manifest": {"sha256": ref["digest"]},
            "definition": {
                "uri": cref,
                "body_sha256": csha,
                "size_bytes": len(measured.native_bytes(cconfig)),
                "s3_version_id": "synthetic",
            },
        }
        self.docs["candidate_binding"] = binding
        self.docs["candidate_adoption"] = {
            "schema_version": "frontier.frontiersmith-reference-adoption-receipt/v1",
            "candidate_version_id": self.cid,
            "definition_digest": csha,
            "request_sha256": binding["request_sha256"],
            "binding_sha256": measured.sha(measured.native_bytes(binding)),
            "overlay_digest": ref["digest"],
            "profile": "preview-reference-v1",
            "source_sha256": identity,
            "model_execution": "deterministic-reference-adopted",
            "installed_files": {
                "opt/frontier/reference-candidate": {
                    **binding["launcher"],
                    "mode": 0o555,
                },
                "opt/frontier/reference-payload/"
                + policy["submission"]["entrypoint"].split("/")[1]: {
                    **binding["payload"],
                    "mode": 0o555,
                },
            },
            "catalog": {"id": self.cid, **candidate},
        }
        labels = {
            "owner": source.name,
            "purpose": "independent-authoring-reference",
            "phase": "current-reference-synthetic",
        }
        scores = (
            {"train": 49.590244174198695, "eval": 48.21403389012444}
            if radix
            else {"train": 1.0, "eval": 2.0}
        )
        self.docs["evaluation_create"] = {
            "candidate_version_id": self.cid,
            "task_version_ids": [self.tid],
            "max_parallel_runs": 1,
            "labels": labels,
        }
        self.docs["evaluation"] = {
            "id": self.eid,
            "candidate_version_id": self.cid,
            "labels": labels,
            "status": "COMPLETED",
            "total_runs": 1,
            "completed_runs": 1,
            "failed_runs": 0,
            "cancelled_runs": 0,
            "active_runs": 0,
            "queued_runs": 0,
            "items": [
                {
                    "task_version_id": self.tid,
                    "run_id": self.rid,
                    "run_state": "COMPLETED",
                    "outcome": "SUCCEEDED",
                    "score": scores["eval"],
                    "verdict": "FAIL",
                }
            ],
        }
        self.docs["run"] = {
            "id": self.rid,
            "evaluation_id": self.eid,
            "task_version_id": self.tid,
            "candidate_version_id": self.cid,
            "state": "COMPLETED",
            "outcome": "SUCCEEDED",
            "finished_at": "2026-01-01T00:00:00Z",
            "score": scores["eval"],
            "verdict": "FAIL",
        }
        self.submission = archive(
            {policy["submission"]["entrypoint"]: self.payload.read_bytes()}
        )
        self.members = {}
        results = {}
        for mode, purpose in (("train", "train"), ("eval", "final")):
            psha = measured.sha(self.submission)
            score = scores[mode]
            detail = {
                "mode": mode,
                "dataset_sha256": definition["datasets"][mode + "_sha256"],
                "payload_sha256": psha,
                "raw_score": score,
            }
            name = "metrics.json"
            if dbscan:
                detail.pop("raw_score")
                detail.update(
                    schema="parallel-dbscan-evidence/v1", measurement={"score": score}
                )
                detail["dataset_sha256"] = detail["dataset_sha256"][7:]
                detail["payload_sha256"] = psha[7:]
                name = "measurement.json"
            if radix:
                detail.pop("raw_score")
                detail.update(
                    score=score, score_unbounded=score, metrics={"beats_reference": 0}
                )
                detail["dataset_sha256"] = detail["dataset_sha256"][7:]
                detail["payload_sha256"] = psha[7:]
                name = "measurement.json"
            evidence = archive({name: measured.native_bytes(detail)})
            vid, pid = self.vids[mode], self.pids[purpose]
            pin = {
                "key": f"runs/{self.rid}/verifications/{vid}/evidence.tar",
                "sha256": measured.sha(evidence),
                "size_bytes": len(evidence),
            }
            result = {
                "schema_version": "frontier.verifier-result/v1",
                "verification_id": vid,
                "mode": mode,
                "status": "passed",
                "raw_score": score,
                "evidence": {key: pin[key] for key in ("sha256", "size_bytes")},
            }
            results[mode] = (result, pin)
            self.members[f"verifications/{vid}/result.json"] = measured.native_bytes(
                result
            )
            self.members[f"payloads/{pid}/payload.tar"] = self.submission
            self.members[f"payloads/{pid}/metadata.json"] = measured.native_bytes(
                {
                    "payload_id": pid,
                    "purpose": purpose,
                    "sha256": psha,
                    "size_bytes": len(self.submission),
                    "object_key": f"runs/{self.rid}/payloads/{pid}/payload.tar",
                }
            )
            if mode == "train":
                self.artifacts["train_evidence"] = evidence
                self.members[f"verifications/{vid}/evidence.tar"] = evidence
            else:
                self.artifacts["final_evidence"] = evidence
                self.members["verifier/evidence.tar"] = evidence
        self.manifest = {
            "schema_version": "frontier.platform-evidence/v4",
            "run_id": self.rid,
            "model_requests": [],
            "traffic_sessions": 0,
            "verifier_evidence": results["eval"][1],
        }

        def event(kind, payload):
            return {
                "id": kind,
                "kind": kind,
                "source": "DRIVER",
                "evaluation_id": self.eid,
                "run_id": self.rid,
                "attempt_id": self.aid,
                "payload": payload,
            }

        self.docs["events"] = [
            event(
                "VERIFIER_TRAIN_COMPLETED",
                {
                    "status": "success",
                    "raw_score": scores["train"],
                    "verification_id": self.vids["train"],
                    "payload_id": self.pids["train"],
                    "evidence": results["train"][1],
                },
            ),
            event(
                "RESULT_ACCEPTED",
                {
                    "outcome": "completed",
                    "raw_score": scores["eval"],
                    "failure": None,
                    "audit_warnings": [],
                },
            ),
        ]
        self.repack()

    def repack(self):
        self.manifest["members"] = [
            {"path": name, "sha256": measured.sha(body), "size_bytes": len(body)}
            for name, body in self.members.items()
        ]
        outer = archive(
            {**self.members, "manifest.json": measured.native_bytes(self.manifest)}
        )
        self.artifacts["platform_evidence"] = outer
        self.docs["events"][-1]["payload"]["evidence"] = {
            "key": f"runs/{self.rid}/final/evidence.tar.zst",
            "sha256": measured.sha(outer),
            "size_bytes": len(outer),
        }

    def replace_synthetic_measurement(self, mode, members):
        """Rebind every synthetic archive descriptor, not any real receipt."""
        body = archive(members)
        vid = self.vids[mode]
        pin = {
            "key": f"runs/{self.rid}/verifications/{vid}/evidence.tar",
            "sha256": measured.sha(body),
            "size_bytes": len(body),
        }
        result_path = f"verifications/{vid}/result.json"
        result = measured.parse(self.members[result_path])
        result["evidence"] = {key: pin[key] for key in ("sha256", "size_bytes")}
        self.members[result_path] = measured.native_bytes(result)
        if mode == "train":
            self.artifacts["train_evidence"] = body
            self.members[f"verifications/{vid}/evidence.tar"] = body
            self.docs["events"][0]["payload"]["evidence"] = pin
        else:
            self.artifacts["final_evidence"] = body
            self.members["verifier/evidence.tar"] = body
            self.manifest["verifier_evidence"] = pin
        self.repack()

    def write(self):
        packet = {"schema_version": measured.SCHEMA}
        for group, rows in (("documents", self.docs), ("artifacts", self.artifacts)):
            packet[group] = {}
            for name, value in rows.items():
                body = measured.native_bytes(value) if group == "documents" else value
                path = name + (".json" if group == "documents" else ".tar")
                (self.directory / path).write_bytes(body)
                packet[group][name] = {
                    "path": path,
                    "sha256": measured.sha(body),
                    "size_bytes": len(body),
                }
        self.packet_path.write_bytes(measured.native_bytes(packet))
        return self.packet_path

    def rebind_synthetic_candidate(self):
        """Rehash this synthetic fixture, so negative tests reach typed guards.

        This never changes retained author packets or supplies authority for one.
        It deliberately allows internally consistent but inadmissible inputs.
        """
        request = self.docs["candidate_create"]
        config = request["driver_config"]
        old_digest = request["definition_digest"]
        request["definition_digest"] = measured.sha(measured.native_bytes(config))
        request["definition_ref"] = request["definition_ref"].replace(
            old_digest[7:], request["definition_digest"][7:]
        )
        binding = self.docs["candidate_binding"]
        binding["request_sha256"] = measured.sha(measured.native_bytes(request))
        binding["definition"].update(
            uri=request["definition_ref"],
            body_sha256=request["definition_digest"],
            size_bytes=len(measured.native_bytes(config)),
        )
        receipt = self.docs["candidate_adoption"]
        receipt.update(
            candidate_version_id=config["candidate_version_id"],
            definition_digest=request["definition_digest"],
            request_sha256=binding["request_sha256"],
            binding_sha256=measured.sha(measured.native_bytes(binding)),
            overlay_digest=config["overlay_image"]["digest"],
            catalog={"id": config["candidate_version_id"], **copy.deepcopy(request)},
        )


class MeasuredFixturePathTests(unittest.TestCase):
    def test_system_temp_alias_is_resolved_without_accepting_evidence_symlinks(self):
        with tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        ) as tmp:
            root = Path(tmp)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with mock.patch.object(tempfile, "tempdir", str(alias)):
                harness = import_tests.ImportTests()
                try:
                    harness.setUp()
                    fixture_root = Path(harness.temp.name)
                    self.assertEqual(fixture_root.parent, real)
                    evidence = fixture_root / "evidence.json"
                    evidence.write_bytes(b"{}")
                    self.assertEqual(measured.read_regular(evidence, 10), b"{}")
                    with self.assertRaises(ValueError):
                        measured.read_regular(
                            alias / fixture_root.name / evidence.name, 10
                        )
                finally:
                    harness.doCleanups()


class MeasuredBaselineTests(unittest.TestCase):
    def setUp(self):
        self.harness = import_tests.ImportTests()
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)
        self.source = self.harness.source
        (self.source / "judge/result.json").unlink()
        self.make_fixture()

    def make_fixture(self, *, json_payload=False, dbscan=False, radix=False):
        if radix:
            renamed = self.source.with_name("parallel_radix_sort")
            self.source.rename(renamed)
            self.source = self.harness.source = renamed
            for filename in ("policy.json", "upstream.lock.json"):
                path = renamed / filename
                body = json.loads(path.read_text())
                body["problem_id"] = renamed.name
                path.write_text(json.dumps(body))
        if json_payload:
            (self.source / "judge/reference.patch").unlink()
            (self.source / "judge/reference.json").write_text('{"synthetic":true}\n')
            policy = json.loads((self.source / "policy.json").read_text())
            policy["submission"]["entrypoint"] = "payload/solution.json"
            (self.source / "policy.json").write_text(json.dumps(policy))
        definition = importer.validate_export(self.source)
        payload = self.source / (
            "judge/reference.json" if json_payload else "judge/reference.patch"
        )
        entry = importer.catalog_entry(self.source, definition, "joyemang33", payload)
        for image, _ in importer.resolver.task_images(entry):
            image["digest"] = "sha256:" + "c" * 64
        directory = Path(self.harness.temp.name) / (
            "packet-radix"
            if radix
            else "packet-json"
            if json_payload
            else "packet-dbscan"
            if dbscan
            else "packet"
        )
        self.fixture = PacketFixture(
            self.source,
            definition,
            entry,
            directory,
            self.harness.root / importer.LAUNCHER,
            json_payload=json_payload,
            dbscan=dbscan,
            radix=radix,
        )
        self.packet = self.fixture.write()
        images = {
            role: {key: entry[value][key] for key in ("repository", "tag", "digest")}
            for role, value in importer.ROLE_KEYS.items()
            if entry.get(value) is not None
        }
        self.manifest = Path(self.harness.temp.name) / "published.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": "frontier.task-release-images/v1",
                    "problem_id": self.source.name,
                    "source_sha256": definition["source"]["source_set_sha256"],
                    "images": images,
                }
            )
        )

    def qualify(self):
        f = self.fixture
        return measured.baseline_from_measurement(
            f.write(),
            self.source,
            f.definition,
            f.entry,
            f.payload,
            f.launcher,
            self.harness.root,
        )

    def invoke(self, *args):
        return self.harness.invoke(
            "--published-images",
            str(self.manifest),
            "--measured-baseline",
            str(self.packet),
            "--reference-payload",
            str(self.fixture.payload),
            *args,
        )

    def test_real_structure_derives_score_preserving_fail_and_rule(self):
        baseline = self.qualify()
        self.assertEqual(baseline["historical_scores"], [2.0])
        self.assertEqual(baseline["measurement"]["verdict"], "FAIL")
        self.assertEqual(baseline["measurement"]["train_score"], 1.0)
        self.assertNotEqual(
            baseline["reference_patch_sha256"],
            baseline["measurement"]["submission_sha256"]["final"],
        )
        self.assertEqual(
            baseline["agreement_rule"],
            {"name": "cross-site-agreement/v1", "abs": 0.01, "rel": 0.05, "replays": 1},
        )

    def test_18000_authoring_phase_preserves_receipt_and_parity_boundary(self):
        f = self.fixture
        labels = dict(
            f.docs["evaluation_create"]["labels"],
            phase="current-18000-reference-synthetic",
        )
        f.docs["evaluation_create"]["labels"] = labels
        f.docs["evaluation"]["labels"] = dict(labels)
        self.assertEqual(self.qualify()["historical_mean"], 2.0)
        for field, value in [
            ("purpose", "preview-parity"),
            ("owner", "unrelated"),
            ("phase", "current-18000-parity-synthetic"),
        ]:
            changed = dict(labels, **{field: value})
            f.docs["evaluation_create"]["labels"] = changed
            f.docs["evaluation"]["labels"] = dict(changed)
            with self.assertRaisesRegex(ValueError, "independent authoring"):
                self.qualify()
        f.docs["evaluation_create"]["labels"] = labels
        f.docs["evaluation"]["labels"] = dict(labels, phase="current-reference-other")
        with self.assertRaisesRegex(ValueError, "Evaluation must"):
            self.qualify()

    def test_import_plan_check_and_repeated_import_use_same_packet(self):
        before = import_tests.snapshot(Path(self.harness.temp.name))
        self.assertEqual(self.invoke("--plan"), 0)
        self.assertEqual(import_tests.snapshot(Path(self.harness.temp.name)), before)
        with mock.patch.object(importer, "verify_images") as verify:
            self.assertEqual(self.invoke(), 0)
            self.assertEqual(self.invoke("--check"), 0)
            self.assertEqual(self.invoke(), 0)
            self.assertGreaterEqual(verify.call_count, 1)
        self.assertEqual(
            import_tests.snapshot(self.source),
            import_tests.snapshot(
                self.harness.root / importer.PROBLEMS / self.source.name
            ),
        )
        with self.assertRaisesRegex(ValueError, "catalog baseline differs"):
            self.harness.invoke("--check", "--published-images", str(self.manifest))

    def test_json_payload_and_dbscan_measurement_supported(self):
        self.make_fixture(json_payload=True, dbscan=True)
        self.assertEqual(self.qualify()["historical_mean"], 2.0)
        with mock.patch.object(importer, "verify_images"):
            self.invoke()
            self.invoke("--check")

    def test_radix_legacy_measurement_requires_trusted_task_and_exact_audit(self):
        dataset, payload = "sha256:" + "a" * 64, "sha256:" + "b" * 64
        document = {
            "mode": "eval",
            "dataset_sha256": dataset[7:],
            "payload_sha256": payload[7:],
            "score": 48.0,
            "score_unbounded": 48.0,
            "metrics": {"beats_reference": 0},
        }

        def read(detail, task="parallel_radix_sort"):
            return measured.measurement_document(
                {"measurement.json": measured.native_bytes(detail)},
                "eval",
                dataset,
                payload,
                problem_id=task,
            )

        self.assertEqual(read(document), 48.0)
        for score in (0, 100):
            self.assertEqual(
                read({**document, "score": score, "score_unbounded": score}), score
            )
        for task in (None, "another_task"):
            with (
                self.subTest(task=task),
                self.assertRaisesRegex(ValueError, "unsupported measurement schema"),
            ):
                read(document, task)
        for change in (
            {"score_unbounded": 49.0},
            {"score_unbounded": math.nextafter(48.0, math.inf)},
            {"score": True, "score_unbounded": True},
            {"score_unbounded": True},
            {"score": -1.0, "score_unbounded": -1.0},
            {"score": 101.0, "score_unbounded": 101.0},
            {"metrics": None},
            {"schema": "untrusted-format"},
            {"extra": True},
            {"outcome": "candidate_invalid"},
            {"mode": "train"},
            {"dataset_sha256": "0" * 64},
            {"payload_sha256": "0" * 64},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                read({**document, **change})

    def test_radix_native_train_final_packet_qualifies_without_scientific_promotion(
        self,
    ):
        self.make_fixture(radix=True)
        f = self.fixture
        f.docs["candidate_create"]["candidate_id"] = (
            "frontiersmith2-reference-parallel_radix_sort"
        )
        f.rebind_synthetic_candidate()
        baseline = self.qualify()
        self.assertEqual(baseline["measurement"]["train_score"], 49.590244174198695)
        self.assertEqual(baseline["historical_scores"], [48.21403389012444])
        self.assertEqual(baseline["measurement"]["verdict"], "FAIL")
        before = import_tests.snapshot(Path(self.harness.temp.name))
        self.assertEqual(self.invoke("--plan"), 0)
        self.assertEqual(import_tests.snapshot(Path(self.harness.temp.name)), before)

    def test_radix_rehashed_nested_archives_cannot_substitute_another_measurement_contract(
        self,
    ):
        self.make_fixture(radix=True)
        f = self.fixture
        self.qualify()  # Prove the synthetic source/receipt chain before mutations.
        original = copy.deepcopy((f.docs, f.artifacts, f.members, f.manifest))
        for mode, role in (("train", "train_evidence"), ("eval", "final_evidence")):
            detail = measured.parse(
                measured.tar_members(original[1][role])["measurement.json"]
            )
            binding = {
                key: detail[key] for key in ("mode", "dataset_sha256", "payload_sha256")
            }
            cases = [
                (
                    "foreign_dataset",
                    {**detail, "dataset_sha256": "0" * 64},
                    "measurement dataset/payload/mode differs",
                ),
                (
                    "foreign_payload",
                    {**detail, "payload_sha256": "0" * 64},
                    "measurement dataset/payload/mode differs",
                ),
                (
                    "wrong_mode",
                    {**detail, "mode": "eval" if mode == "train" else "train"},
                    "measurement dataset/payload/mode differs",
                ),
                (
                    "score_differs_from_result",
                    {**detail, "score": 20, "score_unbounded": 20},
                    "task measurement differs from verifier score",
                ),
                (
                    "adjacent_score_in_same_body",
                    {
                        **detail,
                        "score_unbounded": math.nextafter(detail["score"], math.inf),
                    },
                    "invalid Radix measurement",
                ),
                (
                    "missing_metrics",
                    {key: value for key, value in detail.items() if key != "metrics"},
                    "Radix measurement fields differ",
                ),
                (
                    "metrics_not_object",
                    {**detail, "metrics": []},
                    "invalid Radix measurement",
                ),
                (
                    "invalid_candidate_audit",
                    {**detail, "outcome": "candidate_invalid"},
                    "Radix measurement fields differ",
                ),
                (
                    "foreign_schema",
                    {**detail, "schema": "unrelated/v1"},
                    "Radix measurement fields differ",
                ),
                (
                    "known_dbscan_schema",
                    {
                        **binding,
                        "schema": "parallel-dbscan-evidence/v1",
                        "measurement": {"score": detail["score"]},
                    },
                    "Radix measurement fields differ",
                ),
            ]
            for label, document, error in cases:
                with self.subTest(mode=mode, case=label):
                    f.docs, f.artifacts, f.members, f.manifest = copy.deepcopy(original)
                    f.replace_synthetic_measurement(
                        mode, {"measurement.json": measured.native_bytes(document)}
                    )
                    with self.assertRaisesRegex(ValueError, error):
                        self.qualify()
            # Generic metrics.json must not bypass this task's exact audit contract.
            with self.subTest(mode=mode, case="generic_metrics_filename"):
                f.docs, f.artifacts, f.members, f.manifest = copy.deepcopy(original)
                f.replace_synthetic_measurement(
                    mode,
                    {
                        "metrics.json": measured.native_bytes(
                            {**binding, "raw_score": detail["score"]}
                        )
                    },
                )
                with self.assertRaisesRegex(
                    ValueError, "Radix measurement requires measurement.json"
                ):
                    self.qualify()

    def test_radix_measurement_rejects_nonobject_duplicate_and_nonfinite_json(self):
        dataset, payload = "a" * 64, "b" * 64
        document = {
            "mode": "eval",
            "dataset_sha256": dataset,
            "payload_sha256": payload,
            "score": 48,
            "score_unbounded": 48,
            "metrics": {},
        }
        bodies = [
            b"null",
            b"[]",
            b"1",
            b'"not an audit"',
            json.dumps(document)
            .encode()
            .replace(b'"score": 48', b'"score": 48, "score": 48'),
        ]
        for value in (math.nan, math.inf, -math.inf):
            bodies.append(json.dumps({**document, "score_unbounded": value}).encode())
        bodies.append(
            json.dumps(document)
            .encode()
            .replace(b'"score_unbounded": 48', b'"score_unbounded": 1e999')
        )
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(ValueError):
                measured.measurement_document(
                    {"measurement.json": body},
                    "eval",
                    "sha256:" + dataset,
                    "sha256:" + payload,
                    problem_id="parallel_radix_sort",
                )

    def test_existing_native_candidate_logical_ids_preserve_full_binding(self):
        f = self.fixture
        slug = self.source.name.replace("_", "-")
        for name, admitted in (
            ("frontiersmith2-reference-" + slug, True),
            ("frontiersmith2-reference-" + self.source.name, True),
            ("frontiersmith2-reference-wrong_task", False),
            ("frontiersmith2-preview-reference-" + self.source.name, False),
            ("reference-" + slug + "-authoring", True),
            ("reference-" + slug, False),
            ("unrelated", False),
        ):
            request = f.docs["candidate_create"]
            request["candidate_id"] = name
            binding = f.docs["candidate_binding"]
            binding["request_sha256"] = measured.sha(measured.native_bytes(request))
            receipt = f.docs["candidate_adoption"]
            receipt["catalog"]["candidate_id"] = name
            receipt["request_sha256"] = binding["request_sha256"]
            receipt["binding_sha256"] = measured.sha(measured.native_bytes(binding))
            if admitted:
                self.assertEqual(
                    self.qualify()["measurement"]["candidate_version_id"], f.cid
                )
            else:
                with self.assertRaisesRegex(
                    ValueError, "not the independent adopted reference"
                ):
                    self.qualify()

    def test_exact_native_underscore_alias_qualifies_without_relabelling_fail(self):
        f = self.fixture
        f.docs["candidate_create"]["candidate_id"] = (
            "frontiersmith2-reference-" + self.source.name
        )
        f.rebind_synthetic_candidate()
        baseline = self.qualify()
        self.assertEqual(baseline["measurement"]["candidate_version_id"], f.cid)
        self.assertEqual(baseline["measurement"]["verdict"], "FAIL")
        self.assertEqual(baseline["historical_scores"], [2.0])
        self.assertEqual(self.invoke("--plan"), 0)

    def test_native_underscore_alias_rejects_self_consistent_foreign_candidate_inputs(
        self,
    ):
        f = self.fixture
        f.docs["candidate_create"]["candidate_id"] = (
            "frontiersmith2-reference-" + self.source.name
        )
        f.rebind_synthetic_candidate()
        original = copy.deepcopy(f.docs)
        changes = [
            (
                "foreign-task",
                ("candidate_id",),
                "frontiersmith2-reference-another_task",
            ),
            (
                "prefix-alias",
                ("candidate_id",),
                "frontiersmith2-preview-reference-" + self.source.name,
            ),
            (
                "case-alias",
                ("candidate_id",),
                "frontiersmith2-reference-" + self.source.name.upper(),
            ),
            (
                "suffix-alias",
                ("candidate_id",),
                "frontiersmith2-reference-" + self.source.name + "-copy",
            ),
            ("typed-prefix", ("driver_config", "candidate_version_id"), f.tid),
            (
                "typed-version",
                ("driver_config", "candidate_version_id"),
                "candver_01ZZZZZZZZZZZZZZZZZZZZZZZZ",
            ),
            ("source", ("metadata", "source_sha256"), "sha256:" + "0" * 64),
            ("payload", ("metadata", "payload_sha256"), "sha256:" + "0" * 64),
            ("launcher", ("metadata", "launcher_sha256"), "sha256:" + "0" * 64),
            (
                "image",
                ("driver_config", "overlay_image", "digest"),
                "sha256:" + "0" * 64,
            ),
            ("execution", ("driver_config", "execution", "entrypoint"), ["/bin/true"]),
            (
                "model-policy",
                ("driver_config", "model_api", "allowed_domains"),
                ["example.invalid"],
            ),
        ]
        for name, path, value in changes:
            with self.subTest(name=name):
                f.docs = copy.deepcopy(original)
                parent = f.docs["candidate_create"]
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = value
                f.rebind_synthetic_candidate()
                with self.assertRaises(ValueError):
                    self.qualify()

    def test_native_underscore_alias_keeps_adoption_and_immutable_receipt_checks(self):
        f = self.fixture
        f.docs["candidate_create"]["candidate_id"] = (
            "frontiersmith2-reference-" + self.source.name
        )
        f.rebind_synthetic_candidate()
        original = copy.deepcopy(f.docs)
        changes = [
            ("request", "candidate_binding", ("request_sha256",), "sha256:" + "0" * 64),
            (
                "binding-source",
                "candidate_binding",
                ("source", "sha256"),
                "sha256:" + "0" * 64,
            ),
            (
                "immutable-version",
                "candidate_binding",
                ("definition", "s3_version_id"),
                None,
            ),
            (
                "receipt-source",
                "candidate_adoption",
                ("source_sha256",),
                "sha256:" + "0" * 64,
            ),
            (
                "receipt-image",
                "candidate_adoption",
                ("overlay_digest",),
                "sha256:" + "0" * 64,
            ),
            (
                "installed-mode",
                "candidate_adoption",
                (
                    "installed_files",
                    "opt/frontier/reference-payload/solution.patch",
                    "mode",
                ),
                0o777,
            ),
            ("catalog-id", "candidate_adoption", ("catalog", "id"), f.tid),
            (
                "catalog-alias",
                "candidate_adoption",
                ("catalog", "candidate_id"),
                "frontiersmith2-reference-another_task",
            ),
        ]
        for name, document, path, value in changes:
            with self.subTest(name=name):
                f.docs = copy.deepcopy(original)
                parent = f.docs[document]
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = value
                # Update the outer binding hash for an invalid binding. A stale
                # digest must not be the only reason these proofs are rejected.
                if document == "candidate_binding":
                    f.docs["candidate_adoption"]["binding_sha256"] = measured.sha(
                        measured.native_bytes(f.docs["candidate_binding"])
                    )
                with self.assertRaises(ValueError):
                    self.qualify()

    def test_existing_immutable_candidate_definition_namespace_is_retained(self):
        f = self.fixture
        request = f.docs["candidate_create"]
        request["definition_ref"] = request["definition_ref"].replace(
            "definitions/frontier-cs-2.0/", "definitions/"
        )
        binding = f.docs["candidate_binding"]
        binding["definition"]["uri"] = request["definition_ref"]
        binding["request_sha256"] = measured.sha(measured.native_bytes(request))
        receipt = f.docs["candidate_adoption"]
        receipt["catalog"]["definition_ref"] = request["definition_ref"]
        receipt["request_sha256"] = binding["request_sha256"]
        receipt["binding_sha256"] = measured.sha(measured.native_bytes(binding))
        self.assertEqual(self.qualify()["measurement"]["candidate_version_id"], f.cid)
        request["definition_ref"] = request["definition_ref"].replace(
            "/sha256/", "/mutable/"
        )
        with self.assertRaises(ValueError):
            self.qualify()

    def test_retained_nested_task_qualification_profile_binds_same_release(self):
        f = self.fixture
        metadata = f.docs["task_create"]["metadata"]
        metadata.pop("publication_mode")
        metadata.pop("source_identity_algorithm")
        metadata["reference_qualification"] = {
            "source_identity_algorithm": "preview-source-identity/v1",
            "no_rebuild": True,
        }
        self.assertEqual(self.qualify()["historical_mean"], 2.0)
        for change in (
            {"no_rebuild": False},
            {"source_identity_algorithm": "unrelated"},
        ):
            original = dict(metadata["reference_qualification"])
            metadata["reference_qualification"].update(change)
            with self.assertRaisesRegex(
                ValueError, "TaskVersion source/profile differs"
            ):
                self.qualify()
            metadata["reference_qualification"] = original
        f.docs["task_create"]["author_receipt"]["source"]["sha256"] = (
            "sha256:" + "0" * 64
        )
        with self.assertRaisesRegex(ValueError, "source binding differs"):
            self.qualify()

    def test_wrong_source_image_task_or_candidate_rejected(self):
        for role, mutate in (
            (
                "task_create",
                lambda d: d["author_receipt"]["source"].update(
                    sha256="sha256:" + "0" * 64
                ),
            ),
            ("task_create", lambda d: d.update(task_id="other_task")),
            (
                "task_create",
                lambda d: d["driver_config"]["images"]["verifier"].update(
                    digest="sha256:" + "0" * 64
                ),
            ),
            ("run", lambda d: d.update(task_version_id="taskver_01" + "Z" * 24)),
            ("run", lambda d: d.update(candidate_version_id="candver_01" + "Z" * 24)),
            (
                "candidate_create",
                lambda d: d["driver_config"]["execution"]["entrypoint"].__setitem__(
                    2, 'print("no guard")'
                ),
            ),
            (
                "candidate_create",
                lambda d: d["driver_config"]["model_api"].update(
                    allowed_domains=["provider.invalid"]
                ),
            ),
            (
                "candidate_create",
                lambda d: d["metadata"].update(payload_sha256="sha256:" + "0" * 64),
            ),
        ):
            with self.subTest(role=role, mutate=mutate):
                original = copy.deepcopy(self.fixture.docs[role])
                mutate(self.fixture.docs[role])
                with self.assertRaises(ValueError):
                    self.qualify()
                self.fixture.docs[role] = original

    def test_nonterminal_parity_missing_or_failed_train_and_false_score_rejected(self):
        for role, mutate in (
            ("run", lambda d: d.update(state="RUNNING")),
            (
                "evaluation_create",
                lambda d: d["labels"].update(purpose="reference-parity"),
            ),
            (
                "candidate_create",
                lambda d: d["metadata"].update(purpose="reference-parity"),
            ),
            ("events", lambda d: d.pop(0)),
            ("events", lambda d: d[0]["payload"].update(status="failure")),
            ("events", lambda d: d[0].update(run_id="run_01" + "Z" * 24)),
            ("run", lambda d: d.update(score=123.0)),
        ):
            with self.subTest(role=role, mutate=mutate):
                original = copy.deepcopy(self.fixture.docs)
                mutate(self.fixture.docs[role])
                with self.assertRaises(ValueError):
                    self.qualify()
                self.fixture.docs = original

    def test_adjacent_transport_value_allowed_arbitrary_score_not(self):
        for score, succeeds in (
            (math.nextafter(2.0, math.inf), True),
            (2.0000001, False),
        ):
            self.fixture.docs["run"]["score"] = score
            self.fixture.docs["evaluation"]["items"][0]["score"] = score
            self.fixture.docs["events"][-1]["payload"]["raw_score"] = score
            if succeeds:
                self.assertEqual(self.qualify()["historical_mean"], 2.0)
            else:
                with self.assertRaisesRegex(ValueError, "score differs"):
                    self.qualify()

    def test_payload_tamper_rejected_even_with_all_tar_and_manifest_pins_updated(self):
        f = self.fixture
        pid = f.pids["final"]
        body = archive({f.definition["submission"]["entrypoint"]: b"tampered"})
        f.members[f"payloads/{pid}/payload.tar"] = body
        metadata = measured.parse(f.members[f"payloads/{pid}/metadata.json"])
        metadata.update(sha256=measured.sha(body), size_bytes=len(body))
        f.members[f"payloads/{pid}/metadata.json"] = measured.native_bytes(metadata)
        f.repack()
        with self.assertRaisesRegex(ValueError, "actual submitted payload differs"):
            self.qualify()

    def test_self_consistent_final_descriptors_cannot_replace_actual_blob(self):
        f = self.fixture
        pin = f.manifest["verifier_evidence"]
        pin.update(sha256="sha256:" + "a" * 64, size_bytes=123)
        key = f"verifications/{f.vids['eval']}/result.json"
        result = measured.parse(f.members[key])
        result["evidence"] = {k: pin[k] for k in ("sha256", "size_bytes")}
        f.members[key] = measured.native_bytes(result)
        f.repack()
        with self.assertRaisesRegex(ValueError, "artifact descriptor differs"):
            self.qualify()

    def test_final_nested_bytes_must_match_original_compressed_blob(self):
        f = self.fixture
        f.members["verifier/evidence.tar"] = archive({"metrics.json": b"{}"})
        f.repack()
        with self.assertRaisesRegex(ValueError, "final archive bytes differ"):
            self.qualify()

    def test_wrong_dataset_and_nonfinite_task_score_rejected(self):
        f = self.fixture
        original = f.members["verifier/evidence.tar"]
        for change, message in (
            ({"dataset_sha256": "0" * 64}, "dataset/payload/mode differs"),
            ({"raw_score": "NaN"}, "task measurement score is not finite"),
        ):
            doc = measured.parse(measured.tar_members(original)["metrics.json"])
            doc.update(change)
            body = archive({"metrics.json": measured.native_bytes(doc)})
            # Keep every artifact descriptor and actual byte binding consistent,
            # so only the task's scientific dataset/finite-score check rejects it.
            f.members["verifier/evidence.tar"] = body
            f.artifacts["final_evidence"] = body
            pin = f.manifest["verifier_evidence"]
            pin.update(sha256=measured.sha(body), size_bytes=len(body))
            key = f"verifications/{f.vids['eval']}/result.json"
            result = measured.parse(f.members[key])
            result["evidence"] = {k: pin[k] for k in ("sha256", "size_bytes")}
            f.members[key] = measured.native_bytes(result)
            f.repack()
            with self.assertRaisesRegex(ValueError, message):
                self.qualify()

    def test_archive_links_duplicate_members_and_traversal_rejected(self):
        for name, kind in (
            ("../escape", tarfile.REGTYPE),
            ("link", tarfile.SYMTYPE),
            ("duplicate", tarfile.REGTYPE),
        ):
            output = io.BytesIO()
            with tarfile.open(fileobj=output, mode="w") as tar:
                info = tarfile.TarInfo(name)
                info.type = kind
                info.linkname = "target"
                tar.addfile(info)
                if name == "duplicate":
                    tar.addfile(info)
            with self.assertRaises(ValueError):
                measured.tar_members(output.getvalue())

    def test_packet_paths_duplicate_keys_and_bounds_rejected(self):
        for body in (
            b'{"schema_version":"a","schema_version":"b"}',
            b'{"x":NaN}',
            b'{"x":1e999}',
        ):
            with self.assertRaises(ValueError):
                measured.parse(body)
        for value in ("../escape", "/absolute", "a//b", "./a", "a\\b"):
            with self.assertRaises(ValueError):
                measured.member_path(value)
        link = self.fixture.directory / "link"
        link.symlink_to(self.fixture.packet_path)
        with self.assertRaises(ValueError):
            measured.read_regular(link, measured.MAX_JSON)
        parent_link = self.fixture.directory / "parent"
        parent_link.symlink_to(self.fixture.directory, target_is_directory=True)
        with self.assertRaises(ValueError):
            measured.read_regular(parent_link / "measurement.json", measured.MAX_JSON)
        with self.assertRaises(ValueError):
            measured.read_regular(self.fixture.packet_path, 10)
        with self.assertRaises(ValueError):
            measured.pinned_file(
                self.fixture.directory,
                {"path": "measurement.json", "size_bytes": 1, "sha256": "0"},
                10,
            )

    def test_compressed_platform_and_train_archives(self):
        import zstandard

        f = self.fixture
        compressed = zstandard.ZstdCompressor().compress(f.artifacts["train_evidence"])
        f.artifacts["train_evidence"] = compressed
        pin = f.docs["events"][0]["payload"]["evidence"]
        pin.update(sha256=measured.sha(compressed), size_bytes=len(compressed))
        key = f"verifications/{f.vids['train']}/result.json"
        result = measured.parse(f.members[key])
        result["evidence"] = {k: pin[k] for k in ("sha256", "size_bytes")}
        f.members[key] = measured.native_bytes(result)
        compressed = zstandard.ZstdCompressor().compress(f.artifacts["final_evidence"])
        f.artifacts["final_evidence"] = compressed
        pin = f.manifest["verifier_evidence"]
        pin.update(sha256=measured.sha(compressed), size_bytes=len(compressed))
        key = f"verifications/{f.vids['eval']}/result.json"
        result = measured.parse(f.members[key])
        result["evidence"] = {k: pin[k] for k in ("sha256", "size_bytes")}
        f.members[key] = measured.native_bytes(result)
        f.repack()
        compressed = zstandard.ZstdCompressor().compress(
            f.artifacts["platform_evidence"]
        )
        f.artifacts["platform_evidence"] = compressed
        f.docs["events"][-1]["payload"]["evidence"].update(
            sha256=measured.sha(compressed), size_bytes=len(compressed)
        )
        self.assertEqual(self.qualify()["historical_mean"], 2.0)

    def test_check_rejects_catalog_score_annotation_and_image_edits(self):
        with mock.patch.object(importer, "verify_images"):
            self.invoke()
            path = self.harness.root / "config/task-catalog.json"
            original = path.read_bytes()
            for mutate in (
                lambda e: e["baseline"].update(comparable=False),
                lambda e: e["baseline"].update(historical_mean=3.0),
                lambda e: e["agent_image"].update(digest="sha256:" + "d" * 64),
            ):
                catalog = json.loads(original)
                mutate(catalog["tasks"][-1])
                path.write_text(json.dumps(catalog))
                with self.assertRaises(ValueError):
                    self.invoke("--check")
                path.write_bytes(original)

    def test_external_packet_requires_published_images_and_excludes_legacy_override(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "requires published images"):
            self.harness.invoke("--measured-baseline", str(self.packet), "--plan")
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                self.invoke("--baseline-comparable", "override", "--plan")


if __name__ == "__main__":
    unittest.main()
