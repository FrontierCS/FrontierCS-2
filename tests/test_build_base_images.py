"""Shared base image tags are derived from their sources (scripts/build_base_images.py).

A role's tag is src-<12 hex>, the git tree hash of exactly the files its image is built from:
the same sources always give the same tag, a change to any of them gives a new one, and a change
elsewhere in tasks/shared gives none. The recipes may copy nothing outside their SOURCES.
"""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_base_images  # noqa: E402
import render_task_definition as renderer  # noqa: E402
import task_manifest  # noqa: E402

TEMPLATE = REPO / "tasks/template/Theory/template_task"


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


class DerivedTags(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="base-tags-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.shared = self.root / "tasks/shared"
        shutil.copytree(
            build_base_images.SHARED,
            self.shared,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        git(self.root, "init", "-q")
        git(self.root, "add", "-A")
        git(
            self.root,
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "sources",
        )
        patch = mock.patch.object(build_base_images, "SHARED", self.shared)
        patch.start()
        self.addCleanup(patch.stop)

    def tags(self) -> dict[str, str]:
        return {
            role: build_base_images.source_tag(role) for role in build_base_images.ROLES
        }

    def test_the_tag_is_the_tree_hash_of_the_sources_only(self):
        before = self.tags()
        for tag in before.values():
            self.assertRegex(tag, r"^src-[0-9a-f]{12}$")
        self.assertEqual(len(set(before.values())), 3)
        self.assertEqual(self.tags(), before)
        (self.shared / "agent/README.platform.md").write_text("not an image input\n")
        (self.shared / "task-defaults.toml").write_text("# not an image input\n")
        self.assertEqual(self.tags(), before)

    def test_a_source_change_moves_only_its_roles(self):
        before = self.tags()
        adapter = self.shared / "verifier-gpu/fcs2_verifier_gpu.py"
        adapter.write_text(adapter.read_text() + "\n# change\n")
        after = self.tags()
        self.assertNotEqual(after["gpu-tools"], before["gpu-tools"])
        self.assertEqual(
            {r: after[r] for r in ("agent", "verifier")},
            {r: before[r] for r in ("agent", "verifier")},
        )
        self.assertIn(
            "tasks/shared/verifier-gpu/fcs2_verifier_gpu.py",
            build_base_images.uncommitted_sources("gpu-tools"),
        )
        self.assertEqual(build_base_images.uncommitted_sources("agent"), [])
        shared = self.shared / "lib/edit_scope.py"
        shared.write_text(shared.read_text() + "\n# change\n")
        moved = self.tags()
        self.assertNotEqual(moved["agent"], before["agent"])
        self.assertNotEqual(moved["verifier"], before["verifier"])

    def test_the_tag_matches_plain_git_on_the_committed_tree(self):
        # Anyone can recompute a tag with git alone: build the same tree from HEAD.
        environment_index = self.root / "index"
        env = {"GIT_INDEX_FILE": str(environment_index), "PATH": "/usr/bin:/bin"}

        def run(*a):
            return subprocess.run(
                ["git", "-C", str(self.root), *a],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.strip()

        paths = [
            f"tasks/shared/{path}" for path in build_base_images.SOURCES["gpu-tools"]
        ]
        run("read-tree", "HEAD")  # all of HEAD, then keep only the sources
        run("rm", "-r", "-q", "--cached", "--", ".")
        run("reset", "-q", "HEAD", "--", *paths)
        self.assertEqual(
            build_base_images.source_tag("gpu-tools"), "src-" + run("write-tree")[:12]
        )


class RecipesUseOnlyTheirSources(unittest.TestCase):
    def test_every_copy_source_is_a_declared_source(self):
        for role, recipe in build_base_images.RECIPES.items():
            sources = build_base_images.SOURCES[role]
            self.assertIn(f"images/{recipe}", sources)
            text = (build_base_images.SHARED / "images" / recipe).read_text()
            for line in re.findall(r"^(?:COPY|ADD)\s+(.*)$", text, re.MULTILINE):
                words = [w for w in line.split() if not w.startswith("--")]
                for source in words[:-1]:
                    with self.subTest(role=role, source=source):
                        if role == "gpu-tools" and source == ".root/":
                            continue  # staged by gpu_tools_context (gpu-tools/, verifier-gpu/, lib/)
                        self.assertTrue(
                            any(
                                source == s or source.startswith(s.rstrip("/") + "/")
                                for s in sources
                            ),
                            source,
                        )


class TaskTagsBindResolvedBases(unittest.TestCase):
    """A recipe that takes a base by build argument no longer pins it in its text, so the task's
    source identity (and image tag) binds the bases.json reference it resolves to."""

    def task(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="base-bound-")) / "Theory" / "template_task"
        self.addCleanup(shutil.rmtree, root.parents[1], True)
        shutil.copytree(TEMPLATE, root)
        return root

    def test_a_base_bump_moves_only_tasks_that_take_it_by_argument(self):
        root = self.task()
        pinned = task_manifest.base_reference("verifier")
        self.assertEqual(
            task_manifest.resolved_bases(root),
            {"agent": task_manifest.base_reference("agent"), "verifier": pinned},
        )
        before = renderer.source_sha256(root)
        recipe = root / "docker/verifier.Dockerfile"
        text = recipe.read_text()
        self.assertIn("ARG VERIFIER_BASE\nFROM ${VERIFIER_BASE}", text)
        recipe.write_text(
            text.replace("ARG VERIFIER_BASE\nFROM ${VERIFIER_BASE}", f"FROM {pinned}")
        )
        self.assertNotIn("verifier", task_manifest.resolved_bases(root))
        self.assertNotEqual(renderer.source_sha256(root), before)
        recipe.write_text(text)
        self.assertEqual(renderer.source_sha256(root), before)  # stable
        bumped = dict(
            task_manifest.bases(),
            verifier=pinned.replace("sha256:", "sha256:0", 1)[:-1],
        )
        with mock.patch.object(task_manifest, "bases", return_value=bumped):
            self.assertNotEqual(renderer.source_sha256(root), before)
        # A base the task does not name leaves its tag alone.
        other = dict(
            task_manifest.bases(), **{"gpu-tools": "docker.io/x/y@sha256:" + "1" * 64}
        )
        with mock.patch.object(task_manifest, "bases", return_value=other):
            self.assertEqual(renderer.source_sha256(root), before)


if __name__ == "__main__":
    unittest.main()
