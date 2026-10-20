"""A reviewed task source must not claim images that have not been published."""
import deployment_test_guard  # noqa: F401  # isort: skip

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from preview import publication


class PendingVllmPublicationTests(unittest.TestCase):
    def check_catalog(self, raw):
        entry = next(row for row in raw["tasks"] if row["name"] == "vllm_llm_serving_optimization")
        policy = publication.load_task_definitions()[entry["name"]]
        images = [value for key, value in entry.items() if key.endswith("_image")]
        images += list(entry["sidecar_images"].values()) + [entry["reference"]]
        self.assertEqual(len(images), 55)
        self.assertEqual(entry["datasets"], policy["datasets"])
        roles = {profile.get("image", name) for name, profile in policy["sidecars"].items()}
        self.assertEqual(set(entry["sidecar_images"]), roles)
        self.assertEqual(len(roles), 49)
        self.assertNotIn("swe-018", roles)
        self.assertNotIn("swe-039", roles)
        for image in images:
            # Official catalog validation also permits the immutable legacy v1
            # dataset tag. Every other role must bind the reviewed source.
            self.assertTrue(
                (image is entry["dataset_image"] and image["tag"] == "v1")
                or publication.source_image_tag_matches(
                    image["tag"], policy["image_tag"], image["digest"]
                )
            )
        pending = [image for image in images if image["digest"] == publication.TASK_PLACEHOLDER_DIGEST]
        # Future complete official publication replaces all placeholders atomically.
        # Partial publication remains explicit and cannot seed a TaskVersion.
        if pending:
            self.assertEqual(len(pending), 55)
            result = publication.validate_task_catalog(raw)
            unresolved = [path for path in result["unresolved_digests"]
                          if path.startswith(entry["name"] + ".")]
            self.assertEqual(len(unresolved), len(pending))
            with self.assertRaisesRegex(publication.DeployError, "placeholder image digests"):
                publication.validate_task_catalog(raw, require_resolved=True)
        else:
            publication.validate_task_catalog(raw)

    def test_pending_images_bind_reviewed_source_and_refuse_seeding(self):
        self.check_catalog(json.loads((ROOT / "config/task-catalog.json").read_text()))

    def test_unrelated_pending_publication_is_not_counted_as_vllm(self):
        raw = json.loads((ROOT / "config/task-catalog.json").read_text())
        other = next(row for row in raw["tasks"] if row["name"] != "vllm_llm_serving_optimization")
        policy = publication.load_task_definitions()[other["name"]]
        other["agent_image"].update(tag=policy["image_tag"], digest=publication.TASK_PLACEHOLDER_DIGEST)
        self.check_catalog(raw)
