from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sync_candidate_overlay as sync


def recipe_for(harness: str, sources: list[str]) -> bytes:
    lines = ["FROM scratch AS builder"]
    lines += [f"COPY --chmod=0444 {source} /out/opt/frontier/candidate/{index}"
              for index, source in enumerate(sources)]
    lines += ["FROM scratch", "COPY --from=builder /out/ /", ""]
    return "\n".join(lines).encode()


class SubmitMirrorTests(unittest.TestCase):
    def fake_git(self, recipe: bytes, contents: dict[str, bytes]):
        def git(_source, *args):
            self.assertEqual(args[0], "show")
            path = args[1].split(":", 1)[1]
            if path.endswith("/Dockerfile"):
                return recipe
            return contents[path.removeprefix(sync.SOURCE_PREFIX)]
        return git

    def test_exact_mirror_and_receipt_reject_byte_drift(self):
        sources = [path.removeprefix(sync.SOURCE_PREFIX) for path in sync.FILES.values()]
        contents = {source: (source + "\n").encode() for source in sources}
        recipe = recipe_for(sync.KIMI, sources)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (mock.patch.object(sync, "require_commit"),
                  mock.patch.object(sync, "git", side_effect=self.fake_git(recipe, contents))):
                sync.synchronize(root, "a" * 40, check=False, root=root)
                sync.synchronize(root, "a" * 40, check=True, root=root)
                destination = root / "candidates/kimi-code"
                receipt = json.loads((destination / "source.json").read_text())
                self.assertEqual(receipt["platform_source_commit"], "a" * 40)
                self.assertEqual(set(receipt["files"]), set(sync.FILES))
                skill = contents["kimi-code/skills/frontier-submit/SKILL.md"]
                self.assertEqual(receipt["files"]["submit/SKILL.md"]["sha256"],
                                 "sha256:" + hashlib.sha256(skill).hexdigest())
                local = (destination / "Dockerfile").read_text()
                self.assertIn(" submit/SKILL.md ", local)
                self.assertNotIn("codex/", local)
                (destination / "submit/frontier_submit.py").write_bytes(b"drift\n")
                with self.assertRaisesRegex(ValueError, "mirror differs"):
                    sync.synchronize(root, "a" * 40, check=True, root=root)

    def test_shared_clients_and_harness_files_map_into_one_context(self):
        self.assertEqual(sync.local_name("zcode", "codex/frontier_gpu.py"),
                         "gpu-helper/frontier_gpu.py")
        self.assertEqual(sync.local_name("zcode", "zcode/fetch-sanitize.cjs"),
                         "fetch-sanitize.cjs")
        self.assertEqual(sync.local_name("codex", "codex/skills/frontier-gpu/SKILL.md"),
                         "skills/frontier-gpu/SKILL.md")
        with self.assertRaisesRegex(ValueError, "foreign source"):
            sync.local_name("zcode", "kimi-code/launcher.py")
        recipe = b"COPY a/x.json a/y.json ./\nCOPY --from=b /out/ /\n"
        self.assertEqual(sync.copy_sources("a", recipe), ["a/x.json", "a/y.json"])
        self.assertEqual(sync.adapted_dockerfile(recipe, "a"),
                         b"COPY x.json y.json ./\nCOPY --from=b /out/ /\n")

    def test_checked_in_recipes_match_their_source_receipts(self):
        roster = json.loads((ROOT / sync.ROSTER).read_text())
        for harness, entry in roster["harnesses"].items():
            with self.subTest(harness=harness):
                directory = ROOT / entry["recipe"]["path"]
                receipt = json.loads((directory / "source.json").read_text())
                self.assertEqual(receipt["platform_source_commit"],
                                 entry["recipe"]["source_commit"])
                self.assertEqual(receipt["repository"], sync.SOURCE_REPOSITORY)
                dockerfile = (directory / "Dockerfile").read_bytes()
                self.assertEqual(receipt["dockerfile"]["local_sha256"], sync.sha256(dockerfile))
                # The local Dockerfile copies exactly the receipt's files.
                local_sources = {source.removeprefix(harness + "/")
                                 for source in sync.copy_sources(harness, dockerfile)}
                self.assertEqual(local_sources, set(receipt["files"]))
                for name, value in receipt["files"].items():
                    self.assertEqual(sync.sha256((directory / name).read_bytes()),
                                     value.get("local_sha256", value["sha256"]), name)
                for name, value in sync.receipts_for(receipt).items():
                    self.assertEqual(json.loads((directory / name).read_text()), value)

    def test_unmerged_or_abbreviated_commit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "full commit"):
            sync.require_commit(Path("/source"), "main")
        with mock.patch.object(sync.subprocess, "run",
                               side_effect=sync.subprocess.CalledProcessError(1, "git")):
            with self.assertRaises(sync.subprocess.CalledProcessError):
                sync.require_commit(Path("/source"), "a" * 40)


if __name__ == "__main__":
    unittest.main()
