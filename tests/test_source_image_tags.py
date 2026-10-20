from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from preview.infra import DeployError, source_image_tag_matches, validate_image


class SourceImageTagTests(unittest.TestCase):
    source = "source-" + "a" * 64
    digest = "sha256:" + "b" * 64

    def test_exact_source_and_content_addressed_revision(self):
        for tag in (self.source, self.source + "-image-" + "b" * 32):
            image = validate_image(
                {
                    "repository": "docker.io/owner/task",
                    "tag": tag,
                    "digest": self.digest,
                },
                "image",
                expected_tag=self.source,
            )
            self.assertEqual(image["reference"], "docker.io/owner/task@" + self.digest)

    def test_revision_does_not_extend_commit_tags_or_legacy_tags(self):
        for expected in ("sha-" + "a" * 40, "v1"):
            tag = expected + "-image-" + "b" * 32
            with self.subTest(expected=expected), self.assertRaises(DeployError):
                validate_image(
                    {
                        "repository": "docker.io/owner/task",
                        "tag": tag,
                        "digest": self.digest,
                    },
                    "image",
                    expected_tag=expected,
                )

    def test_non_string_inputs_are_rejected_without_exceptions(self):
        for values in (
            (None, self.source, self.digest),
            (self.source, None, self.digest),
            (self.source, self.source, None),
            (123, 123, self.digest),
        ):
            with self.subTest(values=values):
                self.assertFalse(source_image_tag_matches(*values))

    def test_placeholder_is_not_a_content_addressed_artifact(self):
        self.assertFalse(
            source_image_tag_matches(
                self.source + "-image-" + "0" * 32, self.source, "sha256:" + "0" * 64
            )
        )

    def test_full_digest_is_still_validated_for_exact_and_revision_tags(self):
        for digest in (
            "sha256:" + "B" * 64,
            "sha256:" + "b" * 63,
            "sha512:" + "b" * 64,
        ):
            for tag in (self.source, self.source + "-image-" + "b" * 32):
                with (
                    self.subTest(digest=digest, tag=tag),
                    self.assertRaises(DeployError),
                ):
                    validate_image(
                        {
                            "repository": "docker.io/owner/task",
                            "tag": tag,
                            "digest": digest,
                        },
                        "image",
                        expected_tag=self.source,
                    )


if __name__ == "__main__":
    unittest.main()
