from __future__ import annotations

# Install isolation before project imports; every transport below is synthetic.
import deployment_test_guard  # noqa: F401  # isort: skip

import ast
import copy
from contextlib import ExitStack, nullcontext, redirect_stdout
import hashlib
import io
import json
import os
import signal
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jsonschema import ValidationError

from preview import cli, config as config_ops, infra, publication
from preview import native_local_publication as native


def digest(body):
    return "sha256:" + hashlib.sha256(body).hexdigest()


class NativeLocalPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mount = self.root / "mount"
        self.mount.mkdir()
        self.bodies = {"train": b"synthetic train\x00bytes\n", "eval": b"distinct eval\x00bytes\n"}
        images = {
            "agent_cpu_base": {"reference": "registry.invalid/agent@sha256:" + "a" * 64,
                               "digest": "sha256:" + "a" * 64,
                               "platform": {"os": "linux", "architecture": "amd64"}},
            "verifier": {"reference": "registry.invalid/verifier@sha256:" + "b" * 64,
                         "digest": "sha256:" + "b" * 64},
        }
        config = {
            "schema_version": "frontier.task-config/v1",
            "task_version_id": "taskver_" + "0" * 26,
            "images": images,
            "agent": {"cpu_millis": 1000, "memory_bytes": 1048576, "workspace_bytes": 1048576},
            "quota": {"wall_time_seconds": 60,
                      "agent_gpu": {"allowed_types": [], "max_instances": 0,
                                    "max_gpu_seconds": 0, "max_instance_seconds": 0},
                      "verifier": {"max_train_calls": 2, "max_train_wall_seconds": 30,
                                   "max_gpu_instances": 0, "max_gpu_seconds": 0,
                                   "eval_timeout_seconds": 30}},
            "submission": {"max_bytes": 1048576, "media_type": "application/x-tar",
                           "required_sha256": True, "entrypoint": "answer/run"},
            "datasets": {},
            "verifier": {"command": ["/opt/verifier"], "result_path": "/frontier/output/result.json",
                         "evidence_path": "/frontier/output/evidence", "cpu_millis": 1000,
                         "memory_bytes": 1048576},
            "network": {"allowed_domains": []},
        }
        original, self.readback = {}, {}
        for mode, body in self.bodies.items():
            sha = digest(body)
            uri = f"s3://synthetic-source/datasets/frontier-cs-2.0/native_fixture/{mode}/sha256/{sha[7:]}.tar"
            config["datasets"][mode] = {"uri": uri, "version": sha}
            original[mode] = {"uri": uri, "body_sha256": sha, "size_bytes": len(body),
                              "s3_version_id": "pinned-version-" + mode}
            self.readback[mode] = {"uri": uri, "body_sha256": sha, "size_bytes": len(body),
                                   "version_id": "pinned-version-" + mode}
        self.doc = {"task_id": "native_fixture", "display_name": "Synthetic native fixture",
                    "family": "synthetic", "version": "v1", "metadata": {},
                    "definition_ref": "s3://synthetic-source/definitions/native_fixture.json",
                    "definition_digest": infra.canonical_object_sha256(config), "driver_config": config}
        receipt_images = copy.deepcopy(images)
        receipt_images["agent_cpu_base"].pop("platform")
        self.doc["author_receipt"] = {
            "schema_version": "frontier.task-author-release/v1",
            "task_version_id": config["task_version_id"],
            "source": {"reference": "git:synthetic", "sha256": "sha256:" + "c" * 64},
            "build": {"platform": {"os": "linux", "architecture": "amd64"},
                      "inputs_sha256": "sha256:" + "d" * 64, "tool_versions": {"synthetic": "1"}},
            "definition": {"ref": self.doc["definition_ref"], "digest": self.doc["definition_digest"]},
            "images": receipt_images, "datasets": original,
        }
        self.site = {"site": {"name": "synthetic", "aws_region": "us-west-1"},
                     "network": {"registry_cluster_ip": "172.20.0.50"}}
        self.transport = native.transport_oci(self.doc, self.bodies, self.readback)
        self.mirror = {"digest": self.transport["digest"],
                       "reference": "172.20.0.50:5000/frontier/tasks/native-fixture/dataset@" + self.transport["digest"]}
        self.runner = "registry.invalid/runner@sha256:" + "e" * 64

    def bundle(self):
        path = self.root / "bundle"
        if path.exists():
            import shutil
            shutil.rmtree(path)
        native.write_transport_bundle(self.doc, self.bodies, self.readback, path)
        return path

    def seed(self, layer=None):
        encoded = native.render_native_seed(self.site, self.doc, self.mirror, self.runner)
        pod = json.loads(encoded)["items"][0]["spec"]["template"]["spec"]
        self.assertFalse(pod["automountServiceAccountToken"])
        self.assertEqual(pod["volumes"][0]["persistentVolumeClaim"]["claimName"], "frontier-object-seed")
        self.assertEqual(pod["containers"][0]["image"], self.runner)
        code = pod["containers"][0]["command"][2].replace("/data/", str(self.mount) + "/")
        def response(request, timeout=None):
            url = getattr(request, "full_url", request)
            if "/manifests/" in url:
                return io.BytesIO(self.transport["manifest"])
            if "/blobs/" in url:
                return io.BytesIO(self.transport["layer"] if layer is None else layer)
            raise AssertionError(url)
        namespace = {}
        try:
            with mock.patch("urllib.request.urlopen", side_effect=response), mock.patch("builtins.print"):
                exec(compile(code, "<synthetic-native-seed>", "exec"), namespace)
        finally:
            # The real one-shot Job exits on assertion. Close its archive here
            # because this harness keeps the interpreter alive for other tests.
            if "archive" in namespace:
                namespace["archive"].close()

    def target(self, mode="train"):
        return self.mount / self.doc["driver_config"]["datasets"][mode]["uri"].split("/", 3)[3]

    def rebound(self, document):
        document["definition_digest"] = infra.canonical_object_sha256(document["driver_config"])
        document["author_receipt"]["definition"]["digest"] = document["definition_digest"]
        return document

    def test_seed_bundle_and_local_receipt_preserve_native_identity_and_provenance(self):
        original = copy.deepcopy(self.doc)
        bundle = self.bundle()
        provenance = native.verify_transport_bundle(self.doc, bundle)
        with mock.patch.object(publication, "load_task_definitions", side_effect=AssertionError("catalog accessed")):
            self.seed()
        result = native.prepare_registration(self.doc, self.mount, provenance)
        expected = copy.deepcopy(self.doc)
        for mode in self.bodies:
            expected["author_receipt"]["datasets"][mode]["s3_version_id"] = "local-" + digest(self.bodies[mode])
            self.assertEqual(self.target(mode).read_bytes(), self.bodies[mode])
            self.assertEqual(self.target(mode).stat().st_mode & 0o777, 0o444)
        self.assertEqual(result["document"], expected)
        self.assertEqual(result["provenance"]["original_receipt"], original["author_receipt"])
        self.assertEqual(result["provenance"]["transport"], provenance)
        self.assertEqual(result["provenance"]["original_document_sha256"], infra.canonical_object_sha256(original))
        self.assertEqual(result["provenance"]["local_document_sha256"], infra.canonical_object_sha256(expected))
        self.assertEqual(self.doc, original)
        self.assertEqual(self.transport, native.transport_oci(self.doc, self.bodies, self.readback))

    def test_seed_rejects_corrupt_body_and_does_not_overwrite_conflict(self):
        bad = self.transport["layer"].replace(self.bodies["train"], b"x" * len(self.bodies["train"]))
        with self.assertRaisesRegex(AssertionError, "sha mismatch"):
            self.seed(bad)
        target = self.target()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"conflicting object")
        with self.assertRaisesRegex(AssertionError, "existing file differs"):
            self.seed()
        self.assertEqual(target.read_bytes(), b"conflicting object")

    def test_sdk_reads_only_exact_pinned_versions_and_closes_streams(self):
        client = mock.Mock(spec=["get_object"])
        streams = []
        def get(**kwargs):
            mode = "train" if "/train/" in kwargs["Key"] else "eval"
            expected = self.doc["author_receipt"]["datasets"][mode]
            self.assertEqual(kwargs, {"Bucket": "synthetic-source", "Key": expected["uri"].split("/", 3)[3],
                                      "VersionId": expected["s3_version_id"]})
            stream = io.BytesIO(self.bodies[mode]); streams.append(stream)
            return {"VersionId": kwargs["VersionId"], "ContentLength": len(self.bodies[mode]), "Body": stream}
        client.get_object.side_effect = get
        self.assertEqual(native.read_pinned_s3(self.doc, client), (self.bodies, self.readback))
        self.assertEqual(client.get_object.call_count, 2)
        self.assertTrue(all(stream.closed for stream in streams))

    def test_sdk_rejects_version_length_hash_truncation_and_overrun(self):
        body = self.bodies["train"]
        cases = [("wrong-version", len(body), body), ("pinned-version-train", len(body) + 1, body),
                 ("pinned-version-train", len(body), b"x" * len(body)),
                 ("pinned-version-train", len(body), body[:-1]),
                 ("pinned-version-train", len(body), body + b"extra")]
        for version, size, data in cases:
            with self.subTest(version=version, size=size, data=data):
                stream = io.BytesIO(data)
                client = mock.Mock(spec=["get_object"])
                client.get_object.return_value = {"VersionId": version, "ContentLength": size, "Body": stream}
                with self.assertRaises(infra.DeployError):
                    native.read_pinned_s3(self.doc, client)
                self.assertTrue(stream.closed)

    def test_transport_rejects_unbound_readback_and_original_body(self):
        for key, value in (("uri", "s3://other/key"), ("version_id", "latest"),
                           ("size_bytes", 1), ("body_sha256", "sha256:" + "0" * 64)):
            bad = copy.deepcopy(self.readback); bad["train"][key] = value
            with self.subTest(key=key), self.assertRaises(infra.DeployError):
                native.transport_oci(self.doc, self.bodies, bad)
        with self.assertRaises(infra.DeployError):
            native.transport_oci(self.doc, {**self.bodies, "train": b"wrong"}, self.readback)

    def test_document_rejects_digest_image_receipt_and_local_version_mismatches(self):
        mutations = [lambda d: d["driver_config"]["agent"].update(cpu_millis=2000),
                     lambda d: d["author_receipt"].update(task_version_id="taskver_" + "1" * 26),
                     lambda d: d["author_receipt"]["definition"].update(ref="s3://other/definition"),
                     lambda d: d["author_receipt"]["images"]["verifier"].update(digest="sha256:" + "f" * 64)]
        for change in mutations:
            bad = copy.deepcopy(self.doc); change(bad)
            with self.subTest(change=change), self.assertRaises(infra.DeployError):
                native.dataset_targets(bad)

    def test_declared_sidecar_receipts_bind_images_only(self):
        # Management (Core dataset_authorization) checks the receipt against the
        # TaskConfig images projected onto tag/digest per sidecar, without the
        # parent resources; native documents must carry exactly that shape.
        doc = copy.deepcopy(self.doc)
        images = doc["driver_config"]["images"]
        images["gpu_sidecars"] = {
            "scheduler": {"image": {"tag": "registry.invalid/scheduler:source-" + "e" * 64,
                                    "digest": "sha256:" + "e" * 64},
                          "args": ["/opt/frontier/scheduler"], "workdir": "/opt/frontier",
                          "phases": ["train", "eval"], "attestation_key": True},
            "swe-000-check": {"image": {"tag": "registry.invalid/swe:source-" + "e" * 64,
                                        "digest": "sha256:" + "f" * 64},
                              "args": ["/opt/frontier/worker"], "workdir": "/opt/frontier",
                              "phases": ["eval"]},
        }
        images["gpu_parent_resources"] = {"verifier": {"cpu_millis": 16000, "memory_bytes": 68719476736}}
        doc["definition_digest"] = infra.canonical_object_sha256(doc["driver_config"])
        doc["author_receipt"]["definition"]["digest"] = doc["definition_digest"]
        doc["author_receipt"]["images"] = publication.receipt_images(images)
        self.assertEqual(doc["author_receipt"]["images"]["gpu_sidecars"],
                         {name: profile["image"] for name, profile in images["gpu_sidecars"].items()})
        self.assertNotIn("gpu_parent_resources", doc["author_receipt"]["images"])
        self.assertEqual(native.dataset_targets(doc), native.dataset_targets(self.doc))
        full_profiles = copy.deepcopy(doc)
        full_profiles["author_receipt"]["images"] = copy.deepcopy(images)
        full_profiles["author_receipt"]["images"]["agent_cpu_base"].pop("platform")
        with self.assertRaisesRegex(infra.DeployError, "receipt image mismatch"):
            native.dataset_targets(full_profiles)
        wrong_digest = copy.deepcopy(doc)
        wrong_digest["author_receipt"]["images"]["gpu_sidecars"]["scheduler"]["digest"] = "sha256:" + "1" * 64
        with self.assertRaisesRegex(infra.DeployError, "receipt image mismatch"):
            native.dataset_targets(wrong_digest)

    def test_document_rejects_secret_fields_before_creating_transport(self):
        bad = copy.deepcopy(self.doc)
        bad["metadata"]["aws_secret_access_key"] = "synthetic-only"
        with self.assertRaises(infra.DeployError):
            native.write_transport_bundle(bad, self.bodies, self.readback, self.root / "rejected")
        self.assertFalse((self.root / "rejected").exists())

    def test_json_dataset_keys_remain_json_without_catalog_translation(self):
        for mode in self.bodies:
            for row in (self.doc["driver_config"]["datasets"][mode],
                        self.doc["author_receipt"]["datasets"][mode], self.readback[mode]):
                row["uri"] = row["uri"].removesuffix(".tar") + ".json"
        self.rebound(self.doc)
        targets = native.dataset_targets(self.doc)
        self.assertTrue(all(row["member"].endswith(".json") and row["target"].endswith(".json") for row in targets))
        native.verify_transport_bundle(self.doc, self.bundle())

    def use_key_shape(self, task_id, legacy):
        self.doc["task_id"] = task_id
        for mode, body in self.bodies.items():
            segment = "" if legacy else mode + "/"
            uri = (f"s3://synthetic-source/datasets/frontier-cs-2.0/{task_id}/"
                   f"{segment}sha256/{digest(body)[7:]}.json")
            for row in (self.doc["driver_config"]["datasets"][mode],
                        self.doc["author_receipt"]["datasets"][mode], self.readback[mode]):
                row["uri"] = uri
        self.rebound(self.doc)
        self.transport = native.transport_oci(self.doc, self.bodies, self.readback)
        self.mirror = {"digest": self.transport["digest"],
                       "reference": "172.20.0.50:5000/frontier/tasks/" + task_id.replace("_", "-")
                                    + "/dataset@" + self.transport["digest"]}

    def test_published_hyphenated_and_legacy_keys_preserve_full_custody(self):
        for task_id, legacy in (("native_fixture-cand_000-preflight-a0b1", False),
                                ("native_fixture", True), ("native-fixture_legacy", True)):
            with self.subTest(task_id=task_id, legacy=legacy):
                self.use_key_shape(task_id, legacy)
                original = copy.deepcopy(self.doc)
                provenance = native.verify_transport_bundle(self.doc, self.bundle())
                self.seed()
                result = native.prepare_registration(self.doc, self.mount, provenance)
                self.assertEqual(result["document"]["driver_config"], original["driver_config"])
                self.assertEqual(result["document"]["definition_digest"], original["definition_digest"])
                self.assertEqual(result["provenance"]["original_receipt"], original["author_receipt"])
                for mode, target in zip(("train", "eval"), native.dataset_targets(self.doc)):
                    uri = original["driver_config"]["datasets"][mode]["uri"]
                    self.assertEqual(target["target"], "/data/" + uri.split("/", 3)[3])
                    self.assertEqual(target["member"], f"datasets/{mode}/sha256/{digest(self.bodies[mode])[7:]}.json")
                    self.assertEqual(self.target(mode).read_bytes(), self.bodies[mode])

    def test_legacy_keys_reject_task_hash_traversal_and_explicit_wrong_mode(self):
        self.use_key_shape("native-fixture_legacy", True)
        good = self.doc["driver_config"]["datasets"]["train"]["uri"]
        uris = [good.replace("native-fixture_legacy", "native-fixture-other"),
                good.replace("/sha256/", "/eval/sha256/"),
                good.replace("/sha256/", "/train/eval/sha256/"),
                good.replace(digest(self.bodies["train"])[7:], "f" * 64),
                good.replace("/sha256/", "/../sha256/"),
                good.replace("/sha256/", "/%2e%2e/sha256/"),
                good.replace("/sha256/", "//sha256/")]
        for uri in uris:
            bad = copy.deepcopy(self.doc)
            bad["driver_config"]["datasets"]["train"]["uri"] = uri
            bad["author_receipt"]["datasets"]["train"]["uri"] = uri
            with self.subTest(uri=uri), self.assertRaises(infra.DeployError):
                native.dataset_targets(self.rebound(bad))
        bad = copy.deepcopy(self.doc)
        bad["driver_config"]["datasets"]["train"], bad["driver_config"]["datasets"]["eval"] = (
            bad["driver_config"]["datasets"]["eval"], bad["driver_config"]["datasets"]["train"])
        with self.assertRaises(infra.DeployError):
            native.dataset_targets(self.rebound(bad))
        targets = native.dataset_targets(self.doc)
        targets[0]["member"] = targets[0]["member"].replace("/train/", "/eval/")
        with self.assertRaises(infra.DeployError):
            native.validate_seed_targets(targets)

    def test_legacy_keys_reject_same_mounted_destination_and_mixed_tasks(self):
        self.use_key_shape("native-fixture_legacy", True)
        bad = copy.deepcopy(self.doc)
        for group in (bad["driver_config"]["datasets"], bad["author_receipt"]["datasets"]):
            group["eval"] = copy.deepcopy(group["train"])
            group["eval"]["uri"] = group["eval"]["uri"].replace("synthetic-source", "another-bucket")
        with self.assertRaisesRegex(infra.DeployError, "aliased destination"):
            native.dataset_targets(self.rebound(bad))
        targets = native.dataset_targets(self.doc)
        targets[1]["target"] = targets[1]["target"].replace("native-fixture_legacy", "another-task")
        with self.assertRaisesRegex(infra.DeployError, "task mismatch"):
            native.validate_seed_targets(targets)

    def test_legacy_keys_keep_symlink_and_fifo_mount_rejection(self):
        self.use_key_shape("native-fixture_legacy", True)
        self.seed()
        target = self.target()
        saved = target.with_name("saved"); target.rename(saved); target.symlink_to(saved)
        with self.assertRaises(infra.DeployError):
            native.inspect_objects(self.doc, self.mount)
        target.unlink(); os.mkfifo(target)
        with self.assertRaises(infra.DeployError):
            native.inspect_objects(self.doc, self.mount)

    def test_original_provenance_rejects_local_version_bad_size_and_uri(self):
        for key, value in (("s3_version_id", "local-sha256:" + "a" * 64), ("s3_version_id", ""),
                           ("size_bytes", True), ("size_bytes", 0), ("uri", "s3://other/key")):
            bad = copy.deepcopy(self.doc); bad["author_receipt"]["datasets"]["train"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(infra.DeployError):
                native.dataset_targets(bad)

    def test_document_and_targets_reject_unsafe_paths_and_mode_swaps(self):
        good = self.doc["driver_config"]["datasets"]["train"]["uri"]
        uris = [good.replace("/train/", "/eval/"), good.replace("native_fixture", "other_task"),
                good.replace("datasets/", "datasets/../"), good.replace("datasets/", "datasets/%2e%2e/"),
                good + "?versionId=untrusted", good + "#fragment", good.replace("s3://", "https://"),
                good.replace("synthetic-source", "user@synthetic-source")]
        for uri in uris:
            bad = copy.deepcopy(self.doc)
            bad["driver_config"]["datasets"]["train"]["uri"] = uri
            bad["author_receipt"]["datasets"]["train"]["uri"] = uri
            with self.subTest(uri=uri), self.assertRaises((infra.DeployError, ValidationError)):
                native.dataset_targets(self.rebound(bad))
        for key, value in (("member", "../escape"), ("target", "/data/datasets/../../escape"),
                           ("sha256", "A" * 64)):
            targets = native.dataset_targets(self.doc); targets[0][key] = value
            with self.subTest(key=key), self.assertRaises(infra.DeployError):
                native.validate_seed_targets(targets)

    def test_bundle_rejects_blob_corruption_size_path_and_symlink(self):
        for kind in ("body", "size", "digest-path", "symlink"):
            bundle = self.bundle()
            index = json.loads((bundle / "index.json").read_text())
            desc = index["manifests"][0]
            path = bundle / "blobs/sha256" / desc["digest"][7:]
            if kind == "body":
                path.write_bytes(path.read_bytes() + b" ")
            elif kind == "size":
                desc["size"] += 1
            elif kind == "digest-path":
                desc["digest"] = "sha256:../../escape"
            else:
                other = self.root / "manifest-copy"; other.write_bytes(path.read_bytes())
                path.unlink(); path.symlink_to(other)
            (bundle / "index.json").write_text(json.dumps(index))
            with self.subTest(kind=kind), self.assertRaises(infra.DeployError):
                native.verify_transport_bundle(self.doc, bundle)

    def test_bundle_rejects_invalid_oci_layout_index_and_descriptor_metadata(self):
        cases = [
            ("oci-layout", lambda value: value.update(imageLayoutVersion="1.1.0")),
            ("oci-layout", lambda value: value.pop("imageLayoutVersion")),
            ("index.json", lambda value: value.update(schemaVersion=1)),
            ("index.json", lambda value: value.pop("schemaVersion")),
            ("index.json", lambda value: value["manifests"][0].update(mediaType="application/json")),
            ("index.json", lambda value: value["manifests"][0].pop("mediaType")),
            ("index.json", lambda value: value["manifests"][0].update(
                annotations={"org.opencontainers.image.ref.name": "other"})),
            ("index.json", lambda value: value["manifests"][0].pop("annotations")),
        ]
        for number, (filename, change) in enumerate(cases):
            bundle = self.bundle(); path = bundle / filename
            value = json.loads(path.read_text()); change(value)
            path.write_text(json.dumps(value))
            with self.subTest(case=number), self.assertRaises(infra.DeployError):
                native.verify_transport_bundle(self.doc, bundle)

    def rewrite_layer(self, bundle, entries):
        """Rehash every descriptor so the archive validator, not hashing, is tested."""
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, kind, body in entries:
                member = tarfile.TarInfo(name); member.type = kind; member.mode = 0o444
                member.size = len(body) if kind == tarfile.REGTYPE else 0
                member.linkname = "outside" if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE) else ""
                archive.addfile(member, io.BytesIO(body) if member.size else None)
        def blob(body):
            (bundle / "blobs/sha256" / digest(body)[7:]).write_bytes(body)
            return {"digest": digest(body), "size": len(body)}
        layer = raw.getvalue()
        config = json.loads(self.transport["config"]); config["rootfs"]["diff_ids"] = [digest(layer)]
        manifest = json.loads(self.transport["manifest"])
        manifest["config"].update(blob(infra.canonical_json(config)))
        manifest["layers"][0].update(blob(layer))
        index = json.loads((bundle / "index.json").read_text())
        index["manifests"][0].update(blob(infra.canonical_json(manifest)))
        (bundle / "index.json").write_text(json.dumps(index))
        provenance = json.loads((bundle / "provenance.json").read_text())
        provenance["transport_manifest_digest"] = index["manifests"][0]["digest"]
        (bundle / "provenance.json").write_text(json.dumps(provenance))

    def test_rehashed_oci_rejects_extra_duplicate_traversal_links_fifo_and_changed_body(self):
        entries = [(row["member"], tarfile.REGTYPE, self.bodies[mode])
                   for mode, row in zip(("train", "eval"), native.dataset_targets(self.doc))]
        cases = [entries + [("extra", tarfile.REGTYPE, b"extra")], entries + [entries[0]],
                 [("../escape", tarfile.REGTYPE, self.bodies["train"]), entries[1]],
                 [(entries[0][0], tarfile.REGTYPE, b"wrong"), entries[1]]]
        cases += [[(entries[0][0], kind, b""), entries[1]]
                  for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE)]
        for number, case in enumerate(cases):
            bundle = self.bundle(); self.rewrite_layer(bundle, case)
            with self.subTest(case=number), self.assertRaises(infra.DeployError):
                native.verify_transport_bundle(self.doc, bundle)

    def test_bundle_rejects_source_receipt_and_readback_tampering(self):
        for key in ("source", "original_receipt", "s3_readback", "transport_manifest_digest"):
            bundle = self.bundle()
            path = bundle / ("source-document.json" if key == "source" else "provenance.json")
            value = json.loads(path.read_text())
            if key == "source": value["version"] = "different"
            elif key == "original_receipt": value[key]["build"]["tool_versions"]["synthetic"] = "2"
            elif key == "s3_readback": value[key]["train"]["version_id"] = "wrong"
            else: value[key] = "sha256:" + "0" * 64
            path.write_text(json.dumps(value))
            with self.subTest(key=key), self.assertRaises(infra.DeployError):
                native.verify_transport_bundle(self.doc, bundle)

    def test_mount_rejects_missing_corrupt_and_symlink_objects(self):
        with self.assertRaises(infra.DeployError): native.inspect_objects(self.doc, self.mount)
        self.seed()
        target = self.target(); target.chmod(0o644); target.write_bytes(b"corrupt")
        with self.assertRaises(infra.DeployError): native.inspect_objects(self.doc, self.mount)
        target.write_bytes(self.bodies["train"])
        other = self.root / "other"; other.write_bytes(self.bodies["train"])
        target.unlink(); target.symlink_to(other)
        with self.assertRaises(infra.DeployError): native.inspect_objects(self.doc, self.mount)
        target.unlink(); target.write_bytes(self.bodies["train"])
        parent = target.parent; moved = parent.with_name("moved")
        parent.rename(moved); parent.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(infra.DeployError): native.inspect_objects(self.doc, self.mount)
        alias = self.root / "alias"; alias.symlink_to(self.mount, target_is_directory=True)
        with self.assertRaises(infra.DeployError): native.inspect_objects(self.doc, alias)

    @unittest.skipUnless(hasattr(os, "mkfifo") and hasattr(signal, "SIGALRM"), "POSIX FIFO check")
    def test_mount_rejects_fifo_without_blocking(self):
        self.seed(); target = self.target(); target.unlink(); os.mkfifo(target)
        def timed_out(signum, frame):
            raise AssertionError("FIFO inspection blocked before regular-file validation")
        previous = signal.signal(signal.SIGALRM, timed_out)
        try:
            signal.setitimer(signal.ITIMER_REAL, 2)
            with self.assertRaises(infra.DeployError): native.inspect_objects(self.doc, self.mount)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    def test_registration_requires_original_provenance(self):
        self.seed()
        with self.assertRaises(infra.DeployError): native.prepare_registration(self.doc, self.mount, {})

    def test_catalog_renderer_keeps_demo_namespace_and_native_renderer_keeps_native_namespace(self):
        task = {"name": "catalog_fixture", "datasets": {
            mode + "_sha256": digest(body) for mode, body in self.bodies.items()}}
        definitions = {"catalog_fixture": {"dataset_extension": "json"}}
        with mock.patch.object(publication, "load_task_definitions", return_value=definitions):
            encoded = publication.render_task_dataset_seed_manifest(self.site, task, {"dataset_image": self.mirror}, self.runner)
        job = json.loads(encoded)["items"][0]
        script = job["spec"]["template"]["spec"]["containers"][0]["command"][2]
        assignment = ast.parse(script).body[1]
        settings = json.loads(ast.literal_eval(assignment.value.args[0]))
        expected = [{"member": f"datasets/{mode}/sha256/{digest(body)[7:]}.json",
                     "target": f"/data/datasets/frontier-cs-2.0-demo/catalog_fixture/{mode}/sha256/{digest(body)[7:]}.json",
                     "sha256": digest(body)[7:]} for mode, body in self.bodies.items()]
        self.assertEqual(settings["targets"], expected)
        self.assertEqual(settings["repository"], "frontier/tasks/catalog-fixture/dataset")
        self.assertEqual(job["metadata"]["name"], publication.task_dataset_seed_job_name(task, self.mirror["digest"]))
        # Verified against the unmodified renderer at this branch's HEAD.
        self.assertEqual(hashlib.sha256(encoded).hexdigest(),
                         "c73e3c9082a99fe7e762ed7a33e4fa330764bae4aab15028e5e83363542dc34d")

    def public_readback(self):
        actual = copy.deepcopy(self.doc)
        actual.pop("author_receipt")
        actual["id"] = self.doc["driver_config"]["task_version_id"]
        return actual

    def test_registration_readback_accepts_exact_identity_and_trusted_full_config(self):
        actual = self.public_readback()
        actual["created_at"] = "2026-01-01T00:00:00Z"
        native.validate_registration_readback(self.doc, actual)

    def test_registration_readback_rejects_missing_config_and_identity_mismatches(self):
        for key in ("id", "task_id", "version", "definition_ref", "definition_digest", "driver_config"):
            actual = self.public_readback(); actual.pop(key)
            with self.subTest(missing=key), self.assertRaises(infra.DeployError):
                native.validate_registration_readback(self.doc, actual)
        for key in ("id", "task_id", "version", "definition_ref", "definition_digest", "display_name", "family"):
            actual = self.public_readback(); actual[key] = "different"
            with self.subTest(changed=key), self.assertRaises(infra.DeployError):
                native.validate_registration_readback(self.doc, actual)

    def test_registration_readback_rejects_changed_config_images_datasets_and_network(self):
        mutations = [lambda c: c["images"]["verifier"].update(reference="other@sha256:" + "b" * 64),
                     lambda c: c["datasets"]["train"].update(uri="s3://other/object"),
                     lambda c: c["network"].update(allowed_domains=["openrouter.ai"]),
                     lambda c: c["quota"].update(wall_time_seconds=61),
                     lambda c: c["agent"].update(cpu_millis=2000),
                     lambda c: c["submission"].update(entrypoint="other"),
                     lambda c: c["verifier"].update(command=["/other"])]
        for change in mutations:
            actual = self.public_readback(); change(actual["driver_config"])
            with self.subTest(change=change), self.assertRaises(infra.DeployError):
                native.validate_registration_readback(self.doc, actual)

    def arguments(self, stage, *extra):
        path = self.root / "document.json"; path.write_text(json.dumps(self.doc))
        return cli.parser().parse_args(["native-publish", "--config", str(self.root / "site.json"),
                                       stage, "--document", str(path), *map(str, extra)])

    def test_cli_plan_is_offline_and_preserves_config(self):
        with mock.patch.object(config_ops, "read_and_validate", return_value=(self.site, None)), \
             mock.patch.object(infra, "run_command", side_effect=AssertionError("external command")), \
             mock.patch.object(publication, "load_task_definitions", side_effect=AssertionError("catalog access")):
            result = native.command(self.arguments("plan"))
        self.assertEqual(result["network_calls"], 0)
        self.assertEqual(result["definition_digest"], self.doc["definition_digest"])
        self.assertEqual(result["task_version_id"], self.doc["driver_config"]["task_version_id"])
        self.assertEqual(result["targets"], native.dataset_targets(self.doc))

    def test_cli_finalize_verifies_bundle_and_actual_mount_then_writes_private_envelope(self):
        bundle = self.bundle(); self.seed(); output = self.root / "finalized.json"
        args = self.arguments("finalize", "--bundle", bundle, "--mounted-root", self.mount, "--output", output)
        with mock.patch.object(config_ops, "read_and_validate", return_value=(self.site, None)), \
             mock.patch.object(publication, "ensure_version", side_effect=AssertionError("registration attempted")):
            result = native.command(args)
        self.assertFalse(result["registration_performed"])
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(output.read_text())["document"]["driver_config"], self.doc["driver_config"])

    def test_cli_prepare_uses_exact_version_gets_and_writes_verifiable_bundle(self):
        output = self.root / "prepared"
        def get_object(command, **kwargs):
            self.assertEqual(command[:3], ["synthetic-aws", "s3api", "get-object"])
            self.assertEqual(command[command.index("--bucket") + 1], "synthetic-source")
            key = command[command.index("--key") + 1]
            mode = "train" if "/train/" in key else "eval"
            version = command[command.index("--version-id") + 1]
            self.assertEqual(version, self.readback[mode]["version_id"])
            Path(command[-1]).write_bytes(self.bodies[mode])
            return json.dumps({"VersionId": version, "ContentLength": len(self.bodies[mode])})
        with mock.patch.object(config_ops, "read_and_validate", return_value=(self.site, None)), \
             mock.patch.object(infra, "executable", return_value="synthetic-aws"), \
             mock.patch.object(infra, "run_command", side_effect=get_object) as calls:
            result = native.command(self.arguments("prepare", "--output", output))
        self.assertEqual(calls.call_count, 2)
        self.assertEqual(result, native.verify_transport_bundle(self.doc, output))

    def test_cli_render_seed_is_offline_and_uses_verified_transport_digest(self):
        bundle = self.bundle(); output = self.root / "seed.json"
        with mock.patch.object(config_ops, "read_and_validate", return_value=(self.site, None)), \
             mock.patch("preview.images.load_runtime_images", return_value={"storage_controller": {"reference": self.runner}}), \
             mock.patch.object(infra, "run_command", side_effect=AssertionError("external command")):
            result = native.command(self.arguments("render-seed", "--bundle", bundle, "--output", output))
        self.assertEqual(result["transport_digest"], self.transport["digest"])
        self.assertEqual(output.read_bytes(), native.render_native_seed(self.site, self.doc, self.mirror, self.runner))
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    def test_cli_registration_requires_an_explicit_exclusive_action(self):
        flags = ["--bundle", self.root / "bundle", "--mounted-root", self.mount]
        for extra in ([], ["--check", "--apply"]):
            with self.subTest(extra=extra), mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                self.arguments("register", *flags, *extra)
        self.assertTrue(self.arguments("register", *flags, "--check").check)
        self.assertTrue(self.arguments("register", *flags, "--apply", "--output", self.root / "receipt.json").apply)

    def definition_get(self, command, **kwargs):
        self.assertEqual(command[:3], ["synthetic-aws", "s3api", "get-object"])
        self.assertEqual(command[command.index("--bucket") + 1], "synthetic-source")
        self.assertEqual(command[command.index("--key") + 1], "definitions/native_fixture.json")
        Path(command[-1]).write_text(json.dumps(self.doc["driver_config"]))
        return "{}"

    def test_definition_readback_fetches_actual_ref_and_requires_canonical_digest(self):
        actual = self.public_readback(); actual.pop("driver_config")
        with mock.patch.object(infra, "executable", return_value="synthetic-aws"), \
             mock.patch.object(infra, "run_command", side_effect=self.definition_get):
            verified = native.read_definition(actual, self.site)
            self.assertEqual(verified["driver_config"], self.doc["driver_config"])
            self.assertNotIn("driver_config", actual)
            for change in (lambda a: a.update(definition_digest="sha256:" + "0" * 64),
                           lambda a: a.update(driver_config={"different": True})):
                bad = copy.deepcopy(actual); change(bad)
                with self.subTest(change=change), self.assertRaises(infra.DeployError):
                    native.read_definition(bad, self.site)
        for ref in ("https://example.com/config", "s3://bucket/key?query=1", "s3://bucket/key#fragment"):
            with self.subTest(ref=ref), mock.patch.object(infra, "run_command") as calls, self.assertRaises(infra.DeployError):
                native.read_definition({**actual, "definition_ref": ref}, self.site)
            calls.assert_not_called()

    def registration_context(self, responses):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(config_ops, "read_and_validate", return_value=(self.site, None)))
        stack.enter_context(mock.patch.object(infra, "state_directory", return_value=self.root / "state"))
        stack.enter_context(mock.patch.object(infra, "port_forward", return_value=nullcontext(12345)))
        stack.enter_context(mock.patch.object(infra, "executable", return_value="synthetic-aws"))
        stack.enter_context(mock.patch.object(infra, "run_command", side_effect=self.definition_get))
        stack.enter_context(mock.patch.object(publication, "find_version", side_effect=responses))
        ensure = stack.enter_context(mock.patch.object(publication, "ensure_version"))
        return stack, ensure

    def test_cli_register_check_never_posts_and_apply_records_exact_readback(self):
        bundle = self.bundle(); self.seed()
        actual = self.public_readback(); actual.pop("driver_config")
        for existing, status in ((None, "MISSING"), (actual, "MATCH")):
            context, ensure = self.registration_context([existing])
            with self.subTest(status=status), context:
                result = native.command(self.arguments("register", "--bundle", bundle,
                                                       "--mounted-root", self.mount, "--check"))
            self.assertEqual(result["status"], status)
            self.assertFalse(result["registration_performed"])
            ensure.assert_not_called()
        output = self.root / "registration.json"
        context, ensure = self.registration_context([None, actual])
        with context:
            result = native.command(self.arguments("register", "--bundle", bundle, "--mounted-root", self.mount,
                                                   "--apply", "--output", output))
        self.assertTrue(result["registration_performed"])
        ensure.assert_called_once()
        submitted = ensure.call_args.args[3]
        self.assertEqual(submitted["driver_config"], self.doc["driver_config"])
        for mode in self.bodies:
            self.assertEqual(submitted["author_receipt"]["datasets"][mode]["s3_version_id"], "local-" + digest(self.bodies[mode]))
        saved = json.loads(output.read_text())["registration"]["readback"]
        self.assertEqual(saved["driver_config"], self.doc["driver_config"])

    def test_cli_register_rejects_conflict_before_post_and_bad_readback_after_post(self):
        bundle = self.bundle(); self.seed(); output = self.root / "rejected-registration.json"
        actual = self.public_readback(); actual["version"] = "conflict"
        for responses, posted in (([actual], False), ([None, actual], True), ([None, None], True)):
            context, ensure = self.registration_context(responses)
            with self.subTest(responses=responses), context, self.assertRaises(infra.DeployError):
                native.command(self.arguments("register", "--bundle", bundle, "--mounted-root", self.mount,
                                              "--apply", "--output", output))
            self.assertEqual(ensure.called, posted)
            self.assertFalse(output.exists())

    def inspector_pod(self):
        pod = json.loads(native.render_inspector(self.site, self.doc, self.mirror, self.runner))
        pod["metadata"]["uid"] = "synthetic-pod-uid"
        pod["status"] = {"phase": "Running"}
        return pod

    def inspector_context(self, before, after=None, wrong_sizes=False):
        pods = iter((before, before if after is None else after))
        def run(command, **kwargs):
            self.assertEqual(command[:3], ["synthetic-kubectl", "-n", "frontier-system"])
            if command[3] == "get":
                return json.dumps(next(pods))
            self.assertEqual(command[3:10], ["exec", "-i", self.inspector_pod()["metadata"]["name"], "-c", "inspect", "--", "/usr/local/bin/python3"])
            self.assertEqual(command[10], "-")
            code = kwargs["input_text"].replace('"/data"', repr(str(self.mount)))
            captured = io.StringIO()
            with mock.patch("sys.argv", ["-", command[11]]), redirect_stdout(captured):
                exec(compile(code, "<synthetic-pvc-inspector>", "exec"), {})
            return "{}" if wrong_sizes else captured.getvalue()
        stack = ExitStack()
        stack.enter_context(mock.patch.object(infra, "executable", return_value="synthetic-kubectl"))
        stack.enter_context(mock.patch.object(infra, "state_directory", return_value=self.root / "state"))
        stack.enter_context(mock.patch.object(infra, "kubernetes_environment", return_value={}))
        stack.enter_context(mock.patch.object(infra, "run_command", side_effect=run))
        stack.enter_context(mock.patch("preview.images.load_runtime_images", return_value={"storage_controller": {"reference": self.runner}}))
        return stack

    def test_inspector_name_binds_task_and_frozen_document(self):
        pod = self.inspector_pod()
        name = pod["metadata"]["name"]
        self.assertLessEqual(len(name), 63)
        self.assertIn("native-fixture", name)
        self.assertEqual(name, self.inspector_pod()["metadata"]["name"])
        changed = copy.deepcopy(self.doc)
        changed["version"] = "another-publication"
        other = json.loads(native.render_inspector(self.site, changed, self.mirror, self.runner))
        self.assertNotEqual(name, other["metadata"]["name"])

    def test_readonly_inspector_runs_shared_checker_against_actual_bytes(self):
        self.seed(); pod = self.inspector_pod()
        self.assertEqual(pod["kind"], "Pod")
        self.assertEqual(pod["spec"]["restartPolicy"], "Never")
        self.assertFalse(pod["spec"]["automountServiceAccountToken"])
        self.assertTrue(pod["spec"]["containers"][0]["volumeMounts"][0]["readOnly"])
        self.assertTrue(pod["spec"]["volumes"][0]["persistentVolumeClaim"]["readOnly"])
        provenance = native.verify_transport_bundle(self.doc, self.bundle())
        with self.inspector_context(pod):
            result = native.prepare_registration(self.doc, None, provenance,
                                                 config=self.site, pvc_pod=self.inspector_pod()["metadata"]["name"])
        self.assertEqual(result["provenance"]["mounted_observed_sizes"],
                         {digest(body)[7:]: len(body) for body in self.bodies.values()})
        self.assertEqual(result["document"]["driver_config"], self.doc["driver_config"])

    def test_inspector_rejects_untrusted_runtime_wrong_mount_and_pod_identity_races(self):
        self.seed()
        mutations = [lambda p: p["status"].update(phase="Pending"),
                     lambda p: p["spec"]["containers"][0].update(image="untrusted"),
                     lambda p: p["spec"]["containers"][0]["volumeMounts"][0].update(readOnly=False),
                     lambda p: p["spec"]["containers"][0]["volumeMounts"][0].update(subPath="datasets"),
                     lambda p: p["spec"]["volumes"][0]["persistentVolumeClaim"].update(claimName="other")]
        for change in mutations:
            pod = self.inspector_pod(); change(pod)
            with self.subTest(change=change), self.inspector_context(pod), self.assertRaises(infra.DeployError):
                native.inspect_pvc_objects(self.doc, self.site, self.inspector_pod()["metadata"]["name"])
        for change in (lambda p: p["metadata"].update(uid="replaced"),
                       lambda p: p["metadata"].update(deletionTimestamp="now"),
                       lambda p: p["spec"].update(restartPolicy="Always")):
            before = self.inspector_pod(); after = copy.deepcopy(before); change(after)
            with self.subTest(change=change), self.inspector_context(before, after), self.assertRaises(infra.DeployError):
                native.inspect_pvc_objects(self.doc, self.site, self.inspector_pod()["metadata"]["name"])
        with self.inspector_context(self.inspector_pod(), wrong_sizes=True), self.assertRaises(infra.DeployError):
            native.inspect_pvc_objects(self.doc, self.site, self.inspector_pod()["metadata"]["name"])


if __name__ == "__main__":
    unittest.main()
