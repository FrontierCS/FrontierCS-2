"""Offline checks for the Candidate harness recipes under candidates/."""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import json
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROSTER = json.loads((ROOT / "config/candidates.json").read_text())
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
EFFORTS = {"low", "medium", "high", "xhigh", "max"}
# Node tests that need only the recipe directory (no installed package or binary).
NODE_TESTS = {
    "muse-code": ["test_launcher.mjs", "test_official_host.mjs", "test_transport.mjs"],
    "zcode": ["test-fetch-sanitize.cjs"],
}


def node_major() -> int:
    node = shutil.which("node")
    if node is None:
        return 0
    version = subprocess.run([node, "--version"], capture_output=True, text=True, check=False)
    match = re.match(r"v(\d+)\.", version.stdout)
    return int(match.group(1)) if match else 0


class RosterTests(unittest.TestCase):
    def test_every_candidate_names_a_recipe_and_pinned_image(self):
        harnesses = ROSTER["harnesses"]
        self.assertEqual(
            sorted(path.name for path in (ROOT / "candidates").iterdir() if path.is_dir()),
            sorted(harnesses),
        )
        self.assertIn(ROSTER["builtin_candidate"], harnesses)
        ids = [candidate["id"] for candidate in ROSTER["candidates"]]
        self.assertEqual(len(ids), len(set(ids)))
        for candidate in ROSTER["candidates"]:
            with self.subTest(candidate=candidate["id"]):
                harness = harnesses[candidate["harness"]]
                self.assertEqual(candidate["harness_version"], harness["version"])
                self.assertIn(candidate["reasoning_effort"], EFFORTS)
                image = candidate["overlay_image"]
                self.assertTrue(image["repository"].startswith("docker.io/"))
                self.assertRegex(image["digest"], DIGEST)
                self.assertRegex(candidate["image_source_commit"], r"^[0-9a-f]{40}$")
        for name, harness in harnesses.items():
            with self.subTest(harness=name):
                self.assertEqual(harness["recipe"]["path"], f"candidates/{name}")
                self.assertTrue((ROOT / harness["recipe"]["path"] / "Dockerfile").is_file())
                self.assertTrue((ROOT / harness["recipe"]["path"] / "README.md").is_file())
                self.assertTrue(harness["upstream"].startswith("https://"))

    def test_attested_kimi_image_is_the_mirrored_recipe(self):
        receipt = json.loads((ROOT / "config/kimi-candidate-image.json").read_text())
        recipe = ROSTER["harnesses"]["kimi-code"]
        self.assertEqual(recipe["attestation"], "config/kimi-candidate-image.json")
        self.assertEqual(receipt["source_commit"], recipe["recipe"]["source_commit"])
        self.assertEqual(receipt["source"]["provenance"], "candidates/kimi-code/source.json")


class HarnessUnitTests(unittest.TestCase):
    def test_python_harness_tests(self):
        for name in ROSTER["harnesses"]:
            directory = ROOT / "candidates" / name
            tests = sorted(path.name for path in directory.glob("test_*.py"))
            if not tests:
                continue
            with self.subTest(harness=name):
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
                    cwd=directory, capture_output=True, text=True, timeout=600, check=False,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-2000:])

    @unittest.skipUnless(node_major() >= 18, "Requires Node.js 18 or newer")
    def test_node_harness_tests(self):
        for name, tests in NODE_TESTS.items():
            with self.subTest(harness=name):
                result = subprocess.run(
                    [shutil.which("node"), "--test", *tests],
                    cwd=ROOT / "candidates" / name, capture_output=True, text=True,
                    timeout=600, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
