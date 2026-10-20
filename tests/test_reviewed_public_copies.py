from __future__ import annotations

import deployment_test_guard  # noqa: F401

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_task_layout as layout


class ReviewedPublicCopyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "semantic_robot_world_model"
        self.public = self.root / "runtime-context/runtime"
        self.public.mkdir(parents=True)
        (self.public / "tool.py").write_text("print('public runtime')\n")
        (self.public / "tool.py").chmod(0o644)
        private = self.root / "runtime-context/actor-weights/private.pt"
        private.parent.mkdir()
        private.write_bytes(b"private fixture")
        (self.root / "policy.json").write_text('{"problem_id":"semantic_robot_world_model"}')
        self.recipe = self.root / "agent-gpu/Dockerfile"
        self.recipe.parent.mkdir()
        self.recipe.write_text("FROM scratch\nCOPY runtime-context/runtime/ /runtime/\n")
        files = [
            {"path": p.relative_to(self.root).as_posix(), "bytes": p.stat().st_size,
             "sha256": layout.digest(p)}
            for p in sorted(self.root.rglob("*")) if p.is_file()
        ]
        (self.root / "upstream.lock.json").write_text(json.dumps({
            "problem_id": self.root.name, "file_count": len(files), "files": files,
        }))
        self.review = {
            "schema": "preview.reviewed-public-copies/v1",
            "problem_id": self.root.name, "role": "agent-gpu",
            "bindings": {name: layout.digest(self.root / name) for name in (
                "upstream.lock.json", "policy.json", "agent-gpu/Dockerfile",
            )},
            "copies": {"runtime-context/runtime": "directory"},
            "directories": ["runtime-context/runtime"],
            "files": {"runtime-context/runtime/tool.py": {
                "bytes": (self.public / "tool.py").stat().st_size,
                "sha256": layout.digest(self.public / "tool.py"), "executable": False,
            }},
        }
        self.review_path = Path(self.temp.name) / "review.json"
        self.write_review()
        self.mapping = patch.dict(layout.PUBLIC_COPY_REVIEWS, {
            (self.root.name, "agent-gpu"): self.review_path,
        }, clear=True)
        self.mapping.start()
        self.addCleanup(self.mapping.stop)

    def write_review(self):
        self.review_path.write_text(json.dumps(self.review))

    def check(self):
        layout.check_dockerfile(self.root, "agent-gpu")

    def test_reviewed_directory_and_complete_inventory_pass(self):
        self.check()
        layout.check_inventory(self.root)
        (self.public / "tool.py").chmod(0o664)
        self.check()  # Git-normalized mode, not author umask.

    def test_unreviewed_task_or_agent_role_is_rejected(self):
        other = self.root.with_name("other_task")
        self.root.rename(other)
        with self.assertRaisesRegex(layout.LayoutFailure, "non-public"):
            layout.check_dockerfile(other, "agent-gpu")
        other.rename(self.root)
        (self.root / "agent-base").mkdir()
        (self.root / "agent-base/Dockerfile").write_text(
            "# agent-base/frontier/task\nCOPY runtime-context/runtime/ /runtime/\n"
        )
        with self.assertRaisesRegex(layout.LayoutFailure, "non-public"):
            layout.check_dockerfile(self.root, "agent-base")

    def test_changed_lock_policy_or_recipe_requires_new_review(self):
        for name in self.review["bindings"]:
            path = self.root / name
            original = path.read_bytes()
            with self.subTest(path=name):
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(layout.LayoutFailure, "binding mismatch"):
                    self.check()
                path.write_bytes(original)

    def test_body_size_and_executable_drift_are_rejected(self):
        path = self.public / "tool.py"
        original = path.read_bytes()
        for changed in (b"X" * len(original), original + b"extra"):
            with self.subTest(body=changed):
                path.write_bytes(changed)
                with self.assertRaisesRegex(layout.LayoutFailure, "bytes or executable"):
                    self.check()
        path.write_bytes(original)
        path.chmod(0o755)
        with self.assertRaisesRegex(layout.LayoutFailure, "bytes or executable"):
            self.check()

    def test_extra_missing_and_empty_directory_members_are_rejected(self):
        for name, directory in (("private.pt", False), ("unreviewed", True)):
            path = self.public / name
            path.mkdir() if directory else path.write_bytes(b"unexpected")
            with self.subTest(member=name), self.assertRaisesRegex(
                layout.LayoutFailure, "inventory mismatch"
            ):
                self.check()
            path.rmdir() if directory else path.unlink()
        (self.public / "tool.py").unlink()
        with self.assertRaisesRegex(layout.LayoutFailure, "inventory mismatch"):
            self.check()

    def test_symlink_inside_reviewed_directory_is_rejected(self):
        path = self.public / "tool.py"
        path.unlink()
        path.symlink_to(self.root / "runtime-context/actor-weights/private.pt")
        with self.assertRaisesRegex(layout.LayoutFailure, "non-regular"):
            self.check()

    def test_private_weight_copy_is_not_an_alias_for_reviewed_runtime(self):
        # Even updating only the recipe binding cannot widen the reviewed paths.
        self.recipe.write_text(
            "FROM scratch\nCOPY runtime-context/actor-weights/private.pt /runtime/tool.py\n"
        )
        self.review["bindings"]["agent-gpu/Dockerfile"] = layout.digest(self.recipe)
        self.write_review()
        with self.assertRaisesRegex(layout.LayoutFailure, "source set"):
            self.check()

    def test_review_schema_task_role_and_required_bindings_are_checked(self):
        original = copy.deepcopy(self.review)
        for key, value in (("schema", "other"), ("problem_id", "other"),
                           ("role", "verifier-gpu"), ("bindings", {})):
            self.review = copy.deepcopy(original)
            self.review[key] = value
            self.write_review()
            with self.subTest(key=key), self.assertRaises(layout.LayoutFailure):
                self.check()

    def test_private_source_inventory_still_runs_after_copy_review(self):
        (self.root / "runtime-context/actor-weights/private.pt").write_bytes(b"drift")
        self.check()
        with self.assertRaisesRegex(layout.LayoutFailure, "source lock mismatch"):
            layout.check_inventory(self.root)

    def test_directory_slash_accepts_only_directory_and_rejects_unsafe_paths(self):
        self.assertEqual(layout.copy_source(self.root, "runtime-context/runtime/"),
                         "runtime-context/runtime")
        for source in ("runtime-context/runtime/tool.py/", "runtime-context/runtime//",
                       "runtime-context/../runtime-context/runtime", "./runtime-context/runtime",
                       "/runtime-context/runtime", "runtime-context//runtime", "/"):
            with self.subTest(source=source), self.assertRaises(layout.LayoutFailure):
                layout.copy_source(self.root, source)
        link = self.root / "alias"
        link.symlink_to(self.public, target_is_directory=True)
        with self.assertRaisesRegex(layout.LayoutFailure, "symlink"):
            layout.copy_source(self.root, "alias/tool.py")

    def test_nonexceptional_public_tree_and_verifier_directory_remain_supported(self):
        (self.root / "agent-base/frontier/task").mkdir(parents=True)
        (self.root / "agent-base/Dockerfile").write_text(
            "FROM scratch\nCOPY agent-base/frontier/task/ /frontier/task/\n"
        )
        layout.check_dockerfile(self.root, "agent-base")
        (self.root / "verifier").mkdir()
        (self.root / "verifier/Dockerfile").write_text(
            "FROM scratch\nCOPY runtime-context/actor-weights/ /private/\n"
        )
        layout.check_dockerfile(self.root, "verifier")

    def test_shipped_review_binds_original_freeze_without_private_copy_roots(self):
        review = json.loads((ROOT / "tests/fixtures/reviewed/semantic-public-runtime-reviewed-v1.json").read_text())
        self.assertEqual(review["bindings"]["upstream.lock.json"],
                         "sha256:de8f662e557b753545ed284527977227c8f83e0271894a188d8eb0256ab6255f")
        self.assertEqual(len(review["files"]), 128)
        for name in [*review["copies"], *review["files"]]:
            self.assertTrue(name.startswith("runtime-context/"))
            self.assertNotIn(name.split("/")[1], {"actor-weights", "qualification"})
        self.assertEqual(set(review["copies"]), {
            "runtime-context/" + name for name in (
                "wheelhouse", "runtime", "runtime/requirements.cuda.lock.txt",
                "os-packages/bubblewrap_0.6.1-1ubuntu0.1_amd64.deb", "os-packages/bwrap-netlink-readback",
                "model-assets/semantic-wm/README.md", "model-assets/semantic-wm/webssl/DiT-S_D96.pt",
                "model-assets/semantic-wm/webssl/adapter_webssl_96.pt",
                "model-assets/webssl-dino300m-full2b-224/README.md",
                "model-assets/webssl-dino300m-full2b-224/config.json",
                "model-assets/webssl-dino300m-full2b-224/model.safetensors",
                "model-assets/webssl-dino300m-full2b-224/preprocessor_config.json",
                "debug/episode_0/bridge/test/000000000.mp4", "debug/episode_0/bridge/test/000000000.npz",
                "debug/episode_1/bridge/test/000000000.mp4", "debug/episode_1/bridge/test/000000000.npz",
                "metric-assets/alexnet-owt-7be5be79.pth",
            )
        })


if __name__ == "__main__":
    unittest.main()
