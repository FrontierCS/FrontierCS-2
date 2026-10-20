# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

from preview import images as image_ops

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "registry_reuse_deploy", ROOT / "scripts/frontier_deploy.py"
)
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


class RegistryReuseTests(unittest.TestCase):
    def test_candidate_rejects_a_cached_index_resolved_to_another_manifest(self):
        index = "sha256:" + "a" * 64
        leaf = "sha256:" + "b" * 64
        catalog = {"candidate": {"overlay_image": {
            "reference": "docker.io/example/image@" + index, "digest": index,
        }}, "tasks": []}
        selection = d.TaskSelection([], {}, {}, "", {}, [], ["openrouter.ai"], "", "")
        with (
            mock.patch.object(image_ops, "port_forward") as forward,
            mock.patch.object(image_ops, "executable", side_effect=lambda name: name),
            mock.patch.object(image_ops, "cached_registry_image", return_value=leaf),
            mock.patch.object(image_ops, "run_command") as run,
        ):
            forward.return_value.__enter__.return_value = 5001
            with self.assertRaisesRegex(d.DeployError, "manifest, not its index"):
                d.mirror_task_images(
                    {"network": {"registry_cluster_ip": "172.20.0.50"}},
                    Path("kube"), catalog, selection=selection,
                )
            run.assert_not_called()

    def test_candidate_rejects_index_after_copy_without_writing_mirror_state(self):
        index = "sha256:" + "a" * 64
        leaf = "sha256:" + "b" * 64
        catalog = {"candidate": {"overlay_image": {
            "reference": "docker.io/example/image@" + index, "digest": index,
        }}, "tasks": []}
        selection = d.TaskSelection([], {}, {}, "", {}, [], ["openrouter.ai"], "", "")
        with (
            mock.patch.dict(image_ops.os.environ, {"FRONTIER_REGISTRY_COPY_TOOL": "/tmp/crane"}),
            mock.patch.object(image_ops, "port_forward") as forward,
            mock.patch.object(image_ops, "executable", side_effect=lambda name: name),
            mock.patch.object(image_ops, "cached_registry_image", return_value=None),
            mock.patch.object(image_ops, "registry_digest", return_value=leaf),
            mock.patch.object(image_ops, "run_command") as run,
            mock.patch.object(image_ops, "state_directory") as state,
        ):
            forward.return_value.__enter__.return_value = 5001
            with self.assertRaisesRegex(d.DeployError, "manifest, not its index"):
                d.mirror_task_images(
                    {"network": {"registry_cluster_ip": "172.20.0.50"}},
                    Path("kube"), catalog, selection=selection,
                )
            self.assertEqual(run.call_count, 1)
            state.assert_not_called()

    def test_stream_copy_must_preserve_the_source_digest(self):
        digest = "sha256:" + "a" * 64
        source = {"reference": "docker.io/example/image@" + digest, "digest": digest}
        catalog = {"candidate": {"overlay_image": source}, "tasks": []}
        selection = d.TaskSelection([], {}, {}, "", {}, [], ["openrouter.ai"], "", "")
        for preserved in (True, False):
            with (
                self.subTest(preserved=preserved),
                tempfile.TemporaryDirectory() as directory,
                mock.patch.dict(
                    image_ops.os.environ, {"FRONTIER_REGISTRY_COPY_TOOL": "/tmp/crane"}
                ),
                mock.patch.object(image_ops, "port_forward") as forward,
                mock.patch.object(
                    image_ops, "executable", side_effect=lambda name: name
                ),
                mock.patch.object(image_ops, "run_command") as run,
                mock.patch.object(image_ops, "registry_digest", return_value=digest),
                mock.patch.object(
                    image_ops,
                    "cached_registry_image",
                    side_effect=[None, digest if preserved else None],
                ),
                mock.patch.object(
                    image_ops, "state_directory", return_value=Path(directory)
                ),
            ):
                forward.return_value.__enter__.return_value = 5001
                if preserved:
                    result = d.mirror_task_images(
                        {"network": {"registry_cluster_ip": "172.20.0.50"}},
                        Path("kube"),
                        catalog,
                        selection=selection,
                    )
                    self.assertEqual(
                        result["candidate"]["overlay_image"]["digest"], digest
                    )
                else:
                    with self.assertRaisesRegex(d.DeployError, "preserve"):
                        d.mirror_task_images(
                            {"network": {"registry_cluster_ip": "172.20.0.50"}},
                            Path("kube"),
                            catalog,
                            selection=selection,
                        )
                self.assertEqual(run.call_count, 1)
                self.assertEqual(
                    run.call_args.args[0][:3], ["/tmp/crane", "copy", "--platform"]
                )

    def test_exact_index_leaf_and_all_blobs_are_verified(self):
        blobs = [{"digest": "sha256:" + value * 64, "size": 10} for value in ("a", "b")]
        leaf = {
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": blobs[0],
            "layers": blobs[1:],
        }
        leaf_body = json.dumps(leaf).encode()
        leaf_digest = d.sha256_bytes(leaf_body)
        index = {
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "digest": leaf_digest,
                    "platform": {"os": "linux", "architecture": "amd64"},
                }
            ],
        }
        index_body = json.dumps(index).encode()
        index_digest = d.sha256_bytes(index_body)
        calls = []

        def response(call, timeout):
            calls.append(call)
            digest = call.full_url.rsplit("/", 1)[1]
            body = index_body if digest == index_digest else leaf_body
            result = mock.MagicMock()
            result.__enter__.return_value = result
            result.read.return_value = body
            result.headers = {"Docker-Content-Digest": digest, "Content-Length": "10"}
            return result

        with mock.patch.object(
            image_ops.urllib_request, "urlopen", side_effect=response
        ):
            self.assertEqual(
                d.cached_registry_image(5000, "task", index_digest), leaf_digest
            )
        self.assertEqual(
            [call.get_method() for call in calls], ["GET", "GET", "HEAD", "HEAD"]
        )

    def test_missing_source_is_the_only_cache_miss(self):
        for status in (404, 403, 500):
            with mock.patch.object(
                image_ops.urllib_request,
                "urlopen",
                side_effect=HTTPError("url", status, "error", {}, None),
            ):
                if status == 404:
                    self.assertIsNone(
                        d.cached_registry_image(5000, "task", "sha256:" + "a" * 64)
                    )
                else:
                    with self.assertRaises(d.DeployError):
                        d.cached_registry_image(5000, "task", "sha256:" + "a" * 64)

    def test_digest_header_does_not_hide_corrupt_manifest_bytes(self):
        result = mock.MagicMock()
        result.__enter__.return_value = result
        result.headers = {"Docker-Content-Digest": "sha256:" + "a" * 64}
        result.read.return_value = b"{}"
        with mock.patch.object(
            image_ops.urllib_request, "urlopen", return_value=result
        ):
            with self.assertRaisesRegex(d.DeployError, "identity differs"):
                d.cached_registry_image(5000, "task", "sha256:" + "a" * 64)

    def test_bad_blob_metadata_is_not_a_cache_miss(self):
        blobs = [{"digest": "sha256:" + value * 64, "size": 10} for value in ("a", "b")]
        body = json.dumps(
            {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "config": blobs[0],
                "layers": blobs[1:],
            }
        ).encode()
        digest = d.sha256_bytes(body)
        for headers in (
            {"Docker-Content-Digest": blobs[0]["digest"], "Content-Length": "9"},
            {"Docker-Content-Digest": blobs[1]["digest"], "Content-Length": "10"},
        ):

            def response(call, timeout, *, headers=headers):
                result = mock.MagicMock()
                result.__enter__.return_value = result
                result.read.return_value = body
                result.headers = (
                    {"Docker-Content-Digest": digest}
                    if call.get_method() == "GET"
                    else headers
                )
                return result

            with mock.patch.object(
                image_ops.urllib_request, "urlopen", side_effect=response
            ):
                with self.assertRaisesRegex(
                    d.DeployError, "blob identity or size differs"
                ):
                    d.cached_registry_image(5000, "task", digest)

    def test_ambiguous_platform_is_rejected(self):
        child = {
            "digest": "sha256:" + "a" * 64,
            "platform": {"os": "linux", "architecture": "amd64"},
        }
        body = json.dumps(
            {
                "mediaType": "application/vnd.oci.image.index.v1+json",
                "manifests": [child, child],
            }
        ).encode()
        digest = d.sha256_bytes(body)
        result = mock.MagicMock()
        result.__enter__.return_value = result
        result.read.return_value = body
        result.headers = {"Docker-Content-Digest": digest}
        with mock.patch.object(
            image_ops.urllib_request, "urlopen", return_value=result
        ):
            with self.assertRaisesRegex(d.DeployError, "one linux/amd64 child"):
                d.cached_registry_image(5000, "task", digest)
