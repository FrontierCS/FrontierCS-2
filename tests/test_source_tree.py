"""Source identity and build contexts use only git-tracked files (scripts/source_tree.py):
an ignored __pycache__ or an untracked scratch file under a task fails instead of
changing the image tag or reaching an image, and a worktree computes the identity a clean
checkout computes."""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

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
import build_task_images  # noqa: E402
import render_task_definition as renderer  # noqa: E402
import source_tree  # noqa: E402
import task_manifest  # noqa: E402

TEMPLATE = REPO / "tasks/template/Theory/template_task"


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.email=t@example.com",
                           "-c", "user.name=t", *arguments],
                          check=True, capture_output=True, text=True).stdout.strip()


def files_under(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if not p.is_dir()}


class TaskSource(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp(prefix="source-tree-"))
        self.addCleanup(shutil.rmtree, self.repo, True)
        self.task = self.repo / "problems/Theory/template_task"
        shutil.copytree(TEMPLATE, self.task, ignore=shutil.ignore_patterns("__pycache__"))
        (self.repo / ".gitignore").write_text("__pycache__/\n*.pyc\n")
        git(self.repo, "init", "-q")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "task")

    def test_clean_worktree_identity_equals_a_clean_checkout_and_the_disk_tree(self):
        identity = renderer.source_sha256(self.task)
        clone = Path(tempfile.mkdtemp(prefix="source-clone-"))
        self.addCleanup(shutil.rmtree, clone, True)
        git(self.repo, "worktree", "add", "-q", "--detach", str(clone / "checkout"))
        cloned = clone / "checkout/problems/Theory/template_task"
        self.assertEqual(renderer.source_sha256(cloned), identity)
        # An exported tree (no git) with the same files hashes the same.
        exported = clone / "exported/Theory/template_task"
        shutil.copytree(cloned, exported)
        self.assertEqual(renderer.source_sha256(exported), identity)

    def test_ignored_or_untracked_files_fail_the_identity(self):
        for relative in ("judge/__pycache__/evaluator.cpython-312.pyc", "judge/scratch.txt"):
            with self.subTest(path=relative):
                stray = self.task / relative
                stray.parent.mkdir(parents=True, exist_ok=True)
                stray.write_bytes(b"junk")
                with self.assertRaisesRegex(source_tree.UntrackedFiles, "judge/"):
                    renderer.source_sha256(self.task)
                shutil.rmtree(stray.parent) if "__pycache__" in relative else stray.unlink()
        renderer.source_sha256(self.task)

    def test_an_empty_untracked_directory_changes_nothing(self):
        identity = renderer.source_sha256(self.task)
        (self.task / "judge/empty").mkdir()
        self.assertEqual(renderer.source_sha256(self.task), identity)

    def test_setgid_directories_do_not_change_the_identity(self):
        identity = renderer.source_sha256(self.task)
        for directory in (self.task, self.task / "judge"):
            directory.chmod(directory.stat().st_mode | 0o2000)
        exported = self.repo.parent / f"{self.repo.name}-sg" / "Theory/template_task"
        self.addCleanup(shutil.rmtree, exported.parents[1], True)
        shutil.copytree(self.task, exported)  # copies the setgid bits; outside git: hashed as is
        self.assertTrue(exported.stat().st_mode & 0o2000)
        self.assertEqual(renderer.source_sha256(exported), identity)

    def test_build_context_holds_tracked_and_generated_files_only(self):
        tracked = set(git(self.task, "ls-files").splitlines())
        generated = set(task_manifest.generated_files(self.task))
        scratch = self.repo.parent / f"{self.repo.name}-context"
        self.addCleanup(shutil.rmtree, scratch, True)
        (self.task / "judge/empty").mkdir()
        context = build_task_images.build_context(
            self.task, task_manifest.layout(self.task), scratch)
        self.assertEqual(files_under(context), tracked | generated)
        (self.task / "judge/__pycache__").mkdir()
        (self.task / "judge/__pycache__/x.pyc").write_bytes(b"junk")
        with self.assertRaises(source_tree.UntrackedFiles):
            build_task_images.build_context(self.task, task_manifest.layout(self.task), scratch)


class BaseContext(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="base-context-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.shared = self.root / "tasks/shared"
        shutil.copytree(build_base_images.SHARED, self.shared,
                        ignore=shutil.ignore_patterns("__pycache__"))
        (self.root / ".gitignore").write_text("__pycache__/\n")
        git(self.root, "init", "-q")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-qm", "sources")
        patch = mock.patch.object(build_base_images, "SHARED", self.shared)
        patch.start()
        self.addCleanup(patch.stop)

    def test_context_is_the_roles_tracked_sources_and_refuses_ignored_files(self):
        context = build_base_images.source_context("verifier", self.root / "scratch")
        expected = {f for f in git(self.shared, "ls-files").splitlines()
                    if any(f == s or f.startswith(s + "/")
                           for s in build_base_images.SOURCES["verifier"])}
        self.assertEqual(files_under(context), expected)
        (self.shared / "verifier/__pycache__").mkdir()
        (self.shared / "verifier/__pycache__/fcs2_verifier.cpython-312.pyc").write_bytes(b"x")
        with self.assertRaises(source_tree.UntrackedFiles):
            build_base_images.source_context("verifier", self.root / "scratch2")


if __name__ == "__main__":
    unittest.main()
