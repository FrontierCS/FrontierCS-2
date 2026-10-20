import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import unittest
from unittest import mock

import test_native_local_publication as fixtures
from preview import infra
from preview import native_definition_publication as definitions
from preview import native_image_publication as images
from preview import native_local_publication as datasets


class FakeS3:
    def __init__(self):
        self.versions = []
        self.objects = {}
        self.puts = 0
        self.deleted = False
        self.truncated = False

    def versioned(self, bucket):
        return True

    def history(self, bucket, key):
        return {
            "Versions": self.versions,
            "IsTruncated": self.truncated,
            "DeleteMarkers": [{"Key": key}] if self.deleted else [],
        }

    def get(self, bucket, key, version):
        return self.objects[version]

    def put(self, bucket, key, body):
        self.puts += 1
        digest = "sha256:" + __import__("hashlib").sha256(body).hexdigest()
        self.objects["v1"] = (
            {
                "VersionId": "v1",
                "ContentLength": len(body),
                "ContentType": "application/json",
                "Metadata": {"sha256": digest[7:]},
            },
            body,
        )
        self.versions.append(
            {"Key": key, "VersionId": "v1", "LastModified": "2026-01-01"}
        )
        return {"VersionId": "v1"}


class NativeDefinitionPublicationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.NativeLocalPublicationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.doc, self.site = self.fixture.doc, self.fixture.site
        self.local = images.plan(self.site, self.doc)["document"]
        self.s3 = FakeS3()

    def test_check_upload_readback_and_idempotence(self):
        self.assertEqual(
            definitions.publish_definition(self.local, self.s3, check=True)["status"],
            "MISSING",
        )
        self.assertEqual(self.s3.puts, 0)
        receipt = definitions.publish_definition(self.local, self.s3, check=False)
        self.assertTrue(receipt["uploaded"])
        self.assertEqual(receipt["body_sha256"], self.local["definition_digest"])
        self.assertFalse(
            definitions.publish_definition(self.local, self.s3, check=False)["uploaded"]
        )
        self.assertEqual(self.s3.puts, 1)

    def test_all_history_versions_must_match(self):
        definitions.publish_definition(self.local, self.s3, check=False)
        meta, body = self.s3.objects["v1"]
        self.s3.objects["v1"] = (meta, body + b" ")
        with self.assertRaises(infra.DeployError):
            definitions.publish_definition(self.local, self.s3, check=False)
        self.assertEqual(self.s3.puts, 1)

    def test_deleted_or_truncated_fail_before_put(self):
        for attribute in ("deleted", "truncated"):
            s3 = FakeS3()
            setattr(s3, attribute, True)
            with self.assertRaises(infra.DeployError):
                definitions.publish_definition(self.local, s3, check=False)
            self.assertEqual(s3.puts, 0)

    def test_conditional_write_failure_no_overwrite_fallback(self):
        with mock.patch.object(
            self.s3, "put", side_effect=infra.DeployError("PreconditionFailed")
        ) as put:
            with self.assertRaises(infra.DeployError):
                definitions.publish_definition(self.local, self.s3, check=False)
        self.assertEqual(put.call_count, 1)

    def test_wrong_version_readback_or_metadata_rejected(self):
        for key, value in [
            ("VersionId", "wrong"),
            ("ContentLength", 0),
            ("ContentType", "text/plain"),
            ("Metadata", {}),
        ]:
            s3 = FakeS3()
            definitions.publish_definition(self.local, s3, check=False)
            s3.objects["v1"][0][key] = value
            with self.assertRaises(infra.DeployError):
                definitions.publish_definition(self.local, s3, check=True)

    def test_pipeline_rebind_same_transport_finalize_and_readback(self):
        original_bundle = self.fixture.bundle()
        rebound = self.fixture.root / "rebound"
        evidence = definitions.rebind_transport(
            self.doc, self.local, original_bundle, rebound
        )
        self.assertFalse(evidence["dataset_content_changed"])
        for blob in (original_bundle / "blobs/sha256").iterdir():
            self.assertEqual(
                blob.read_bytes(), (rebound / "blobs/sha256" / blob.name).read_bytes()
            )
            self.assertEqual(
                blob.stat().st_ino, (rebound / "blobs/sha256" / blob.name).stat().st_ino
            )
        provenance = datasets.verify_transport_bundle(self.local, rebound)
        for mode, body in self.fixture.bodies.items():
            key = datasets._key(self.doc["driver_config"]["datasets"][mode]["uri"])
            target = self.fixture.mount / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        final = datasets.prepare_registration(
            self.local, self.fixture.mount, provenance
        )["document"]
        for key in ("source", "build"):
            self.assertEqual(
                final["author_receipt"][key], self.doc["author_receipt"][key]
            )
        for mode, row in final["author_receipt"]["datasets"].items():
            self.assertEqual(
                row["s3_version_id"],
                "local-" + self.doc["driver_config"]["datasets"][mode]["version"],
            )
        artifact = definitions.publish_definition(final, self.s3, check=False)
        self.assertEqual(artifact["body_sha256"], final["definition_digest"])
        actual = copy.deepcopy(final)
        actual["id"] = final["driver_config"]["task_version_id"]
        actual.pop("author_receipt")
        datasets.validate_registration_readback(final, actual)

    def test_non_transport_change_rejected(self):
        original_bundle = self.fixture.bundle()
        changed = copy.deepcopy(self.local)
        changed["driver_config"]["quota"]["wall_time_seconds"] += 1
        with self.assertRaises(infra.DeployError):
            definitions.rebind_transport(
                self.doc, changed, original_bundle, self.fixture.root / "bad"
            )
        self.assertFalse((self.fixture.root / "bad").exists())

    def test_disabled_bucket_versioning_rejected_before_write(self):
        with mock.patch.object(self.s3, "versioned", return_value=False):
            with self.assertRaises(infra.DeployError):
                definitions.publish_definition(self.local, self.s3, check=False)
        self.assertEqual(self.s3.puts, 0)

    def test_aws_conditional_put_is_account_and_region_bound(self):
        config = {
            "site": {"aws_region": "us-west-1", "expected_account_id": "123456789012"}
        }
        with (
            mock.patch("preview.infra.state_directory", return_value=self.fixture.root),
            mock.patch("preview.infra.executable", side_effect=lambda x: x),
            mock.patch(
                "preview.infra.run_command", return_value='{"VersionId":"v1"}'
            ) as run,
        ):
            definitions.AwsDefinitions(config).put(
                "bucket", "definitions/new.json", b"{}"
            )
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("--if-none-match") + 1], "*")
        self.assertEqual(
            argv[argv.index("--expected-bucket-owner") + 1], "123456789012"
        )
        self.assertEqual(argv[argv.index("--region") + 1], "us-west-1")

    def test_cli_pipeline_definition_check_apply_and_candidate_check_apply(self):
        import argparse
        import json
        from contextlib import nullcontext

        from preview import publication

        candidate = publication.candidate_version_document(
            {
                "name": "reference",
                "overlay_image": {
                    "repository": "123456789012.dkr.ecr.us-west-1.amazonaws.com/example/candidate",
                    "tag": "frozen",
                    "digest": "sha256:" + "e" * 64,
                },
            },
            execution={
                "entrypoint": ["/reference"],
                "working_dir": "/workspace",
                "user": "1000:1000",
                "args": [],
            },
        )
        candidate["definition_ref"] = (
            "s3://synthetic-source/definitions/frontier-cs-2.0/reference/candidate.json"
        )
        frozen = self.fixture.root / "frozen.json"
        cand = self.fixture.root / "candidate.json"
        frozen.write_text(json.dumps(self.doc))
        cand.write_text(json.dumps(candidate))
        args = argparse.Namespace(
            config=self.fixture.root / "site.json",
            document=frozen,
            candidate=cand,
            output=self.fixture.root / "result.json",
            stage="definitions",
            check=True,
        )
        localized = images.plan(self.site, self.doc, candidate)

        def artifact(actual, config):
            for doc in [
                self.doc,
                candidate,
                localized["document"],
                localized["candidate"],
            ]:
                if actual["definition_ref"] == doc["definition_ref"]:
                    return {"driver_config": doc["driver_config"]}
            raise AssertionError("unbound artifact")

        with (
            mock.patch(
                "preview.config.read_and_validate", return_value=(self.site, None)
            ),
            mock.patch(
                "preview.native_image_publication.mirror",
                return_value=[{"status": "MATCH"}],
            ),
            mock.patch.object(definitions, "AwsDefinitions", return_value=self.s3),
            mock.patch(
                "preview.native_local_publication.read_definition", side_effect=artifact
            ),
            mock.patch("preview.infra.state_directory", return_value=self.fixture.root),
            mock.patch("preview.infra.port_forward", return_value=nullcontext(12345)),
        ):
            self.assertEqual(
                definitions.command(args)["definitions"]["document"]["status"],
                "MISSING",
            )
            self.assertEqual(self.s3.puts, 0)
            args.check = False
            result = definitions.command(args)
            self.assertEqual(self.s3.puts, 2)
            self.assertTrue(result["definition_upload_performed"])
            args.stage = "candidate"
            args.check = True
            with (
                mock.patch("preview.publication.find_version", return_value=None),
                mock.patch("preview.publication.ensure_version") as ensure,
            ):
                self.assertEqual(
                    definitions.command(args)["candidate_registration"]["status"],
                    "MISSING",
                )
                ensure.assert_not_called()
            args.check = False
            actual = copy.deepcopy(localized["candidate"])
            actual["id"] = actual["driver_config"]["candidate_version_id"]
            with (
                mock.patch(
                    "preview.publication.find_version", side_effect=[None, actual]
                ),
                mock.patch("preview.publication.ensure_version") as ensure,
            ):
                result = definitions.command(args)
                self.assertTrue(result["registration_performed"])
                ensure.assert_called_once()
            original_bundle = self.fixture.bundle()
            args.stage = "rebind"
            args.bundle = original_bundle
            args.rebound_bundle = self.fixture.root / "cli-rebound"
            result = definitions.command(args)
            self.assertEqual(result["dataset_rebind"]["payload_copy_bytes"], 0)
            self.assertEqual(
                datasets.verify_transport_bundle(
                    result["document"], args.rebound_bundle
                )["transport_manifest_digest"],
                datasets.verify_transport_bundle(self.doc, original_bundle)[
                    "transport_manifest_digest"
                ],
            )
