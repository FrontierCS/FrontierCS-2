import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import test_native_local_publication as fixtures
from preview import infra, publication
from preview import native_image_publication as images


class NativeImagePublicationTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.NativeLocalPublicationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.doc, self.site = fixture.doc, fixture.site
        self.candidate = publication.candidate_version_document(
            {
                "name": "reference",
                "overlay_image": {
                    "repository": "private.invalid/reference",
                    "tag": "frozen",
                    "digest": "sha256:" + "e" * 64,
                },
            },
            execution={
                "entrypoint": ["/reference"],
                "args": ["--frozen"],
                "working_dir": "/workspace",
                "user": "1000:1000",
                "env": {"FIXED": "yes"},
            },
        )

    def test_only_references_and_identity_bindings_change(self):
        original = copy.deepcopy(self.doc)
        result = images.plan(self.site, self.doc, self.candidate)
        self.assertEqual(self.doc, original)
        for kind, before, after in [
            ("task", self.doc, result["document"]),
            ("candidate", self.candidate, result["candidate"]),
        ]:
            cfg = copy.deepcopy(after["driver_config"])
            key = kind + "_version_id"
            self.assertNotEqual(cfg[key], before["driver_config"][key])
            cfg[key] = before["driver_config"][key]
            if kind == "task":
                for role in ("agent_cpu_base", "verifier"):
                    cfg["images"][role]["reference"] = before["driver_config"][
                        "images"
                    ][role]["reference"]
            else:
                cfg["overlay_image"]["reference"] = before["driver_config"][
                    "overlay_image"
                ]["reference"]
            self.assertEqual(cfg, before["driver_config"])
            self.assertEqual(
                after["definition_digest"],
                infra.canonical_object_sha256(after["driver_config"]),
            )
            self.assertNotEqual(after["version"], before["version"])
            self.assertNotEqual(after["definition_ref"], before["definition_ref"])
        for key in ("source", "build", "datasets"):
            self.assertEqual(
                result["document"]["author_receipt"][key],
                self.doc["author_receipt"][key],
            )
        self.assertEqual(result, images.plan(self.site, self.doc, self.candidate))

    def test_idempotence_and_existing_local_candidate(self):
        result = images.plan(self.site, self.doc, self.candidate)
        again = images.plan(self.site, result["document"], result["candidate"])
        self.assertEqual(again["document"], result["document"])
        self.assertEqual(again["candidate"], result["candidate"])
        self.assertEqual(
            images.plan(self.site, self.doc, result["candidate"])["candidate"],
            result["candidate"],
        )

    def test_pinned_digest_over_mutable_tag(self):
        result = images.plan(self.site, self.doc, self.candidate)
        self.assertEqual(
            result["images"][-1]["source_reference"],
            "private.invalid/reference@sha256:" + "e" * 64,
        )
        with self.assertRaises(infra.DeployError):
            images._pinned(
                {
                    "reference": "registry.invalid/image@sha256:" + "a" * 64,
                    "digest": "sha256:" + "b" * 64,
                }
            )

    def test_receipt_or_definition_tampering(self):
        bad = copy.deepcopy(self.doc)
        bad["definition_digest"] = "sha256:" + "f" * 64
        with self.assertRaises(infra.DeployError):
            images.plan(self.site, bad)
        bad = copy.deepcopy(self.doc)
        bad["author_receipt"]["images"]["verifier"]["digest"] = "sha256:" + "f" * 64
        with self.assertRaises(infra.DeployError):
            images.plan(self.site, bad)

    def test_gpu_references_untouched(self):
        self.doc["driver_config"]["images"]["verifier"].update(
            gpu_tag="private.invalid/gpu:frozen", gpu_digest="sha256:" + "f" * 64
        )
        self.doc["author_receipt"]["images"]["verifier"] = copy.deepcopy(
            self.doc["driver_config"]["images"]["verifier"]
        )
        self.doc["definition_digest"] = infra.canonical_object_sha256(
            self.doc["driver_config"]
        )
        self.doc["author_receipt"]["definition"]["digest"] = self.doc[
            "definition_digest"
        ]
        after = images.plan(self.site, self.doc)["document"]["driver_config"]["images"][
            "verifier"
        ]
        self.assertEqual(after["gpu_tag"], "private.invalid/gpu:frozen")
        self.assertEqual(after["gpu_digest"], "sha256:" + "f" * 64)

    def test_read_only_check_and_exact_copy(self):
        prepared = images.plan(self.site, self.doc)
        a, b = [r["digest"] for r in prepared["images"]]
        with (
            mock.patch("preview.infra.port_forward", return_value=nullcontext(12345)),
            mock.patch(
                "preview.infra.state_directory", return_value=Path("/synthetic")
            ),
            mock.patch("preview.images.cached_registry_image", return_value=None),
            mock.patch("preview.infra.run_command") as run,
        ):
            result = images.mirror(self.site, self.doc, check=True)
        run.assert_not_called()
        self.assertEqual([r["status"] for r in result], ["MISSING", "MISSING"])
        with (
            mock.patch("preview.infra.port_forward", return_value=nullcontext(12345)),
            mock.patch(
                "preview.infra.state_directory", return_value=Path("/synthetic")
            ),
            mock.patch(
                "preview.images.cached_registry_image", side_effect=[None, a, b]
            ),
            mock.patch("preview.infra.executable", side_effect=lambda x: x),
            mock.patch("preview.infra.run_command") as run,
            mock.patch.dict(os.environ, {"FRONTIER_REGISTRY_COPY_TOOL": "crane"}),
        ):
            result = images.mirror(self.site, self.doc, check=False)
        self.assertEqual(run.call_count, 1)
        self.assertIn(prepared["images"][0]["source_reference"], run.call_args.args[0])
        self.assertEqual([r["status"] for r in result], ["MATCH", "MATCH"])

    def test_index_to_leaf_substitution_rejected(self):
        with (
            mock.patch("preview.infra.port_forward", return_value=nullcontext(12345)),
            mock.patch(
                "preview.infra.state_directory", return_value=Path("/synthetic")
            ),
            mock.patch(
                "preview.images.cached_registry_image",
                return_value="sha256:" + "f" * 64,
            ),
            mock.patch("preview.infra.run_command") as run,
        ):
            with self.assertRaises(infra.DeployError):
                images.mirror(self.site, self.doc, check=False)
        run.assert_not_called()

    def test_official_native_dataset_precheck_accepts_localized_document(self):
        from preview.native_local_publication import dataset_targets

        result = images.plan(self.site, self.doc)
        self.assertEqual(dataset_targets(result["document"]), dataset_targets(self.doc))

    def test_declared_sidecar_receipts_localize_in_management_shape(self):
        # Declared sidecars and parent resources are copied unchanged; the
        # original and localized receipts both carry Management's projection
        # (sidecar tag/digest by name, no parent resources).
        doc = copy.deepcopy(self.doc)
        cfg_images = doc["driver_config"]["images"]
        cfg_images["gpu_sidecars"] = {
            "scheduler": {"image": {"tag": "registry.invalid/scheduler:source-" + "e" * 64,
                                    "digest": "sha256:" + "e" * 64},
                          "args": ["/opt/frontier/scheduler"], "workdir": "/opt/frontier",
                          "phases": ["train", "eval"], "attestation_key": True},
        }
        cfg_images["gpu_parent_resources"] = {"verifier": {"cpu_millis": 16000, "memory_bytes": 68719476736}}
        doc["definition_digest"] = infra.canonical_object_sha256(doc["driver_config"])
        doc["author_receipt"]["definition"]["digest"] = doc["definition_digest"]
        doc["author_receipt"]["images"] = publication.receipt_images(cfg_images)
        result = images.plan(self.site, doc, self.candidate)
        localized = result["document"]
        self.assertEqual(localized["driver_config"]["images"]["gpu_sidecars"], cfg_images["gpu_sidecars"])
        self.assertEqual(localized["driver_config"]["images"]["gpu_parent_resources"],
                         cfg_images["gpu_parent_resources"])
        self.assertEqual(localized["author_receipt"]["images"],
                         publication.receipt_images(localized["driver_config"]["images"]))
        self.assertNotIn("gpu_parent_resources", localized["author_receipt"]["images"])
        self.assertEqual(localized["author_receipt"]["images"]["gpu_sidecars"]["scheduler"],
                         cfg_images["gpu_sidecars"]["scheduler"]["image"])
        full = copy.deepcopy(doc)
        full["author_receipt"]["images"] = copy.deepcopy(cfg_images)
        full["author_receipt"]["images"]["agent_cpu_base"].pop("platform")
        # The dataset precheck (native_local_publication) rejects the full-profile
        # receipt first; the localization binding check guards the same shape.
        with self.assertRaisesRegex(infra.DeployError, "receipt (image mismatch|binding differs)"):
            images.plan(self.site, full, self.candidate)
        with self.assertRaisesRegex(infra.DeployError, "receipt binding differs"):
            with mock.patch.object(fixtures.native, "dataset_targets", return_value=[]):
                images.plan(self.site, full, self.candidate)

    def test_source_build_evidence_required(self):
        for key in ("source", "build"):
            bad = copy.deepcopy(self.doc)
            bad["author_receipt"][key] = {}
            with self.assertRaises(infra.DeployError):
                images.plan(self.site, bad)

    def test_mirror_rejects_unvalidated_input_before_network(self):
        bad = images.plan(self.site, self.doc)
        bad["images"][0]["repository"] = "outside/approved/repository"
        with mock.patch("preview.infra.port_forward") as forward:
            with self.assertRaises(KeyError):
                images.mirror(self.site, bad, check=False)
        forward.assert_not_called()
