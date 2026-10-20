from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import base64
import copy
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import resolve_catalog_digests as resolver


def sha(body):
    return "sha256:" + hashlib.sha256(body).hexdigest()


def image_manifest(*, os="linux", architecture="amd64", layers=1):
    config = json.dumps(
        {
            "os": os,
            "architecture": architecture,
            "config": {},
            "rootfs": {"type": "layers", "diff_ids": [sha(b"layer")] * layers},
        }
    ).encode()
    document = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "digest": sha(config),
            "size": len(config),
        },
        "layers": [
            {
                "mediaType": "application/vnd.oci.image.layer.v1.tar",
                "digest": sha(b"layer"),
                "size": len(b"layer"),
            }
        ]
        * layers,
    }
    return json.dumps(document).encode(), config


class DigestResolverTests(unittest.TestCase):
    def setUp(self):
        self.auth_env = mock.patch.dict(
            os.environ,
            {"FRONTIER_DOCKERHUB_USERNAME": "", "FRONTIER_DOCKERHUB_TOKEN": ""},
        )
        self.auth_env.start()
        self.addCleanup(self.auth_env.stop)

    def test_pull_credentials_only_go_to_token_service(self):
        body, config = image_manifest()
        responses = [
            self.response(b'{"token":"scoped-pull-grant"}'),
            self.response(body),
            self.response(config),
        ]
        with (
            mock.patch.dict(
                os.environ,
                {
                    "FRONTIER_DOCKERHUB_USERNAME": "operator",
                    "FRONTIER_DOCKERHUB_TOKEN": "test-token",
                },
            ),
            mock.patch.object(
                resolver.request, "urlopen", side_effect=responses
            ) as http,
        ):
            self.assertEqual(
                resolver.resolve_digest("docker.io/joyemang33/test", "v1"), sha(body)
            )
        calls = [c.args[0] for c in http.call_args_list]
        self.assertEqual(
            calls[0].get_header("Authorization"),
            "Basic " + base64.b64encode(b"operator:test-token").decode(),
        )
        for call in calls[1:]:
            self.assertEqual(
                call.get_header("Authorization"), "Bearer scoped-pull-grant"
            )
        self.assertNotIn("test-token", calls[0].full_url)

    def test_incomplete_pull_credentials_fail_before_network(self):
        with (
            mock.patch.dict(os.environ, {"FRONTIER_DOCKERHUB_USERNAME": "operator"}),
            mock.patch.object(resolver.request, "urlopen") as http,
        ):
            with self.assertRaisesRegex(ValueError, "both username and token"):
                resolver.resolve_digest("docker.io/joyemang33/test", "v1")
            http.assert_not_called()

    def response(self, body, headers=None):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = body
        response.headers = (
            headers if headers is not None else {"Docker-Content-Digest": sha(body)}
        )
        return response

    def test_anonymous_registry_v2_direct_manifest(self):
        body, config = image_manifest()
        responses = [
            self.response(b'{"token":"anonymous-test-grant"}'),
            self.response(body),
            self.response(config),
        ]
        with mock.patch.object(
            resolver.request, "urlopen", side_effect=responses
        ) as http:
            self.assertEqual(
                resolver.resolve_digest(
                    "docker.io/joyemang33/test", "source-abc", single_layer=True
                ),
                sha(body),
            )
        calls = [call.args[0] for call in http.call_args_list]
        self.assertTrue(calls[0].full_url.startswith("https://auth.docker.io/token?"))
        self.assertIn("repository%3Ajoyemang33%2Ftest%3Apull", calls[0].full_url)
        self.assertFalse(calls[0].has_header("Authorization"))
        self.assertEqual(
            calls[1].full_url,
            "https://registry-1.docker.io/v2/joyemang33/test/manifests/source-abc",
        )
        self.assertIn(
            "application/vnd.oci.image.manifest.v1+json", calls[1].get_header("Accept")
        )
        self.assertEqual(
            calls[2].full_url,
            "https://registry-1.docker.io/v2/joyemang33/test/blobs/" + sha(config),
        )

    def test_index_selects_exactly_one_amd64_leaf_and_verifies_its_hash(self):
        body, config = image_manifest()
        child = {
            "digest": sha(body),
            "size": len(body),
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "platform": {"os": "linux", "architecture": "amd64"},
        }
        for children, succeeds in (
            ([child], True),
            ([child, child], False),
            ([], False),
        ):
            index = json.dumps({"schemaVersion": 2, "manifests": children}).encode()
            responses = [
                self.response(b'{"token":"anonymous-test-grant"}'),
                self.response(index),
                self.response(body),
                self.response(config),
            ]
            with mock.patch.object(resolver.request, "urlopen", side_effect=responses):
                if succeeds:
                    self.assertEqual(
                        resolver.resolve_digest("docker.io/joyemang33/test", "v1"),
                        sha(body),
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "one linux/amd64"):
                        resolver.resolve_digest("docker.io/joyemang33/test", "v1")

    def test_corrupt_header_failed_http_and_multilayer_dataset_fail_closed(self):
        body, _ = image_manifest(layers=2)
        for response in (
            self.response(body, {"Docker-Content-Digest": "sha256:" + "b" * 64}),
            HTTPError("https://registry-1.docker.io/", 404, "missing", {}, None),
            self.response(body),
        ):
            with mock.patch.object(
                resolver.request,
                "urlopen",
                side_effect=[
                    self.response(b'{"token":"anonymous-test-grant"}'),
                    response,
                ],
            ):
                with self.assertRaises((ValueError, HTTPError)):
                    resolver.resolve_digest(
                        "docker.io/joyemang33/test", "v1", single_layer=True
                    )

    def test_direct_manifest_rejects_wrong_platform(self):
        for platform in ({"architecture": "arm64"}, {"os": "windows"}):
            with self.subTest(platform=platform):
                body, config = image_manifest(**platform)
                responses = [
                    self.response(b'{"token":"anonymous-test-grant"}'),
                    self.response(body),
                    self.response(config),
                ]
                with mock.patch.object(
                    resolver.request, "urlopen", side_effect=responses
                ):
                    with self.assertRaisesRegex(ValueError, "linux/amd64"):
                        resolver.resolve_digest("docker.io/joyemang33/test", "v1")

    def test_config_blob_hash_and_size_must_match_descriptor(self):
        original, config = image_manifest()
        for field, value in (
            ("digest", "sha256:" + "b" * 64),
            ("size", len(config) + 1),
        ):
            with self.subTest(field=field):
                document = json.loads(original)
                document["config"][field] = value
                body = json.dumps(document).encode()
                responses = [
                    self.response(b'{"token":"anonymous-test-grant"}'),
                    self.response(body),
                    self.response(config),
                ]
                with mock.patch.object(
                    resolver.request, "urlopen", side_effect=responses
                ):
                    with self.assertRaisesRegex(ValueError, "config blob differs"):
                        resolver.resolve_digest("docker.io/joyemang33/test", "v1")

    def test_manifest_requires_valid_config_and_layer_descriptors(self):
        body, _ = image_manifest()
        missing_config = json.loads(body)
        del missing_config["config"]
        documents = [missing_config]
        for kind in ("config", "layer"):
            for field, value in (
                ("digest", "invalid"),
                ("size", -1),
                ("size", True),
                ("mediaType", ""),
            ):
                document = json.loads(body)
                descriptor = (
                    document["config"] if kind == "config" else document["layers"][0]
                )
                descriptor[field] = value
                documents.append(document)
        for document in documents:
            with self.subTest(document=document):
                responses = [
                    self.response(b'{"token":"anonymous-test-grant"}'),
                    self.response(json.dumps(document).encode()),
                ]
                with mock.patch.object(
                    resolver.request, "urlopen", side_effect=responses
                ):
                    with self.assertRaisesRegex(ValueError, "image descriptor"):
                        resolver.resolve_digest("docker.io/joyemang33/test", "v1")

    def test_layer_member_verification_rejects_drift_links_traversal_duplicates_and_modes(
        self,
    ):
        def layer(name, body=b"expected", mode=0o555, kind=tarfile.REGTYPE):
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.type = len(body), mode, kind
                archive.addfile(info, io.BytesIO(body))
            return stream.getvalue()

        expected = {"opt/frontier/reference-candidate": b"expected"}
        valid = layer(next(iter(expected)))
        resolver.verify_layer_members(
            [
                layer("opt", kind=tarfile.DIRTYPE),
                layer("opt/frontier", kind=tarfile.DIRTYPE),
                valid,
            ],
            expected,
            executable=True,
        )
        cases = [
            [layer("../escape")],
            [layer(next(iter(expected)), b"changed!")],
            [layer(next(iter(expected)), kind=tarfile.SYMTYPE)],
            [layer(next(iter(expected)), mode=0o644)],
            [valid, valid],
            [],
        ]
        for layers in cases:
            with self.subTest(layers=len(layers)), self.assertRaises(ValueError):
                resolver.verify_layer_members(layers, expected, executable=True)

    def test_reference_layers_verify_parents_and_reject_later_conflicts_and_extras(
        self,
    ):
        expected = {
            "opt/frontier/reference-candidate": b"launcher",
            "opt/frontier/reference-payload/nested/solution.py": b"payload",
        }
        parents = [
            "opt",
            "opt/frontier",
            "opt/frontier/reference-payload",
            "opt/frontier/reference-payload/nested",
        ]

        def layer(entries):
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                for name, kind, mode, body in entries:
                    info = tarfile.TarInfo(name)
                    info.type, info.mode, info.size = kind, mode, len(body)
                    archive.addfile(info, io.BytesIO(body))
            return stream.getvalue()

        entries = [(name, tarfile.DIRTYPE, 0o755, b"") for name in parents]
        entries += [
            (name, tarfile.REGTYPE, 0o555, body) for name, body in expected.items()
        ]
        valid = layer(entries)
        resolver.verify_layer_members([valid], expected, executable=True)
        cases = {
            "absolute root": [valid, layer([("/", tarfile.DIRTYPE, 0o755, b"")])],
            "missing parent": [layer(entries[1:])],
            "owner-only parent": [
                layer([(parents[0], tarfile.DIRTYPE, 0o700, b""), *entries[1:]])
            ],
            "parent lacks read": [
                layer([(parents[0], tarfile.DIRTYPE, 0o711, b""), *entries[1:]])
            ],
            "later parent mode": [
                valid,
                layer([(parents[1], tarfile.DIRTYPE, 0o700, b"")]),
            ],
            "extra directory": [valid, layer([("tmp", tarfile.DIRTYPE, 0o755, b"")])],
            "extra file": [
                valid,
                layer([("opt/frontier/extra", tarfile.REGTYPE, 0o555, b"x")]),
            ],
            "whiteout": [
                valid,
                layer(
                    [
                        (
                            "opt/frontier/.wh.reference-candidate",
                            tarfile.REGTYPE,
                            0o555,
                            b"",
                        )
                    ]
                ),
            ],
            "parent symlink": [
                valid,
                layer([(parents[2], tarfile.SYMTYPE, 0o755, b"")]),
            ],
        }
        for name, body in expected.items():
            cases[name + " later directory"] = [
                valid,
                layer([(name, tarfile.DIRTYPE, 0o755, b"")]),
            ]
            cases[name + " later file"] = [
                valid,
                layer([(name, tarfile.REGTYPE, 0o555, body)]),
            ]
        for variant, layers in cases.items():
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                resolver.verify_layer_members(layers, expected, executable=True)

    def test_file_updates_cover_catalog_gpu_and_reference_images_and_keep_identity_fields(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.json"
            catalog.write_bytes((ROOT / "config/task-catalog.json").read_bytes())
            original = json.loads(catalog.read_text())
            artifact_digests = {
                (image["repository"], image["tag"]): image["digest"]
                for task in original["tasks"]
                for image, _ in resolver.task_images(task)
                if "-image-" in image["tag"]
            }

            def fixture_digest(repository, tag, **kwargs):
                # Immutable artifact tags resolve to the digest they already bind.
                return artifact_digests.get((repository, tag), "sha256:" + "a" * 64)

            with mock.patch.object(
                resolver, "resolve_digest", side_effect=fixture_digest
            ) as resolve:
                resolver.resolve_files(catalog)
            self.assertEqual(
                resolve.call_count,
                1 + sum(len(resolver.task_images(task)) for task in original["tasks"]),
            )
            self.assertEqual(
                sum(call.kwargs["single_layer"] for call in resolve.call_args_list),
                len(original["tasks"]),
            )
            expected = copy.deepcopy(original)
            expected["candidate"]["overlay_image"]["digest"] = "sha256:" + "a" * 64
            for task in expected["tasks"]:
                for image, _ in resolver.task_images(task):
                    image["digest"] = fixture_digest(image["repository"], image["tag"])
            self.assertEqual(json.loads(catalog.read_text()), expected)

    def test_failed_resolution_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task-catalog.json"
            path.write_bytes((ROOT / "config/task-catalog.json").read_bytes())
            before = path.read_bytes()
            with mock.patch.object(
                resolver,
                "resolve_digest",
                side_effect=["sha256:" + "a" * 64, ValueError("missing")],
            ):
                with self.assertRaises(ValueError):
                    resolver.resolve_files(path)
            self.assertEqual(before, path.read_bytes())

    def test_task_filter_leaves_candidate_and_other_tasks_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task-catalog.json"
            path.write_bytes((ROOT / "config/task-catalog.json").read_bytes())
            original = json.loads(path.read_text())
            name = original["tasks"][0]["name"]
            with mock.patch.object(
                resolver, "resolve_digest", return_value="sha256:" + "a" * 64
            ) as resolve:
                resolver.resolve_files(path, name)
            self.assertEqual(resolve.call_count, 4)
            updated = json.loads(path.read_text())
            self.assertEqual(updated["candidate"], original["candidate"])
            self.assertEqual(updated["tasks"][1:], original["tasks"][1:])


if __name__ == "__main__":
    unittest.main()
