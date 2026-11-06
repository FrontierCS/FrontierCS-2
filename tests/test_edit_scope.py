"""tasks/shared/lib/edit_scope.py: the one edit-scope check."""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tasks/shared/lib"))

import edit_scope  # noqa: E402
from edit_scope import Rules  # noqa: E402


def git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@invalid"}
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True).stdout


@unittest.skipUnless(shutil.which("git"), "needs git")
class EditScope(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(tempfile.mkdtemp(prefix="edit-scope-test-"))
        self.pristine = self.scratch / "pristine"
        (self.pristine / "src").mkdir(parents=True)
        (self.pristine / "tests").mkdir()
        (self.pristine / "src/a.c").write_text("int a(void) { return 1; }\n")
        (self.pristine / "src/b.c").write_text("int b(void) { return 2; }\n")
        (self.pristine / "tests/t.c").write_text("int main(void) { return 0; }\n")
        (self.pristine / "README.md").write_text("starter\n")
        # The published pristine tree is read-only; the module must cope.
        for path in sorted(self.pristine.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        self.work = self.scratch / "work"
        shutil.copytree(self.pristine, self.work)
        for path in self.work.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        self.work.chmod(0o755)
        git(self.work, "init", "-q")
        git(self.work, "add", "-A")
        git(self.work, "commit", "-qm", "base")

    def tearDown(self):
        for path in self.scratch.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)
        shutil.rmtree(self.scratch, ignore_errors=True)

    def diff(self) -> bytes:
        git(self.work, "add", "-A")
        # Bytes, not text: universal newlines would turn CRLF content into LF.
        return subprocess.run(["git", "diff", "--cached", "--no-renames", "--binary", "HEAD"],
                              cwd=self.work, check=True, capture_output=True).stdout

    def rename_diff(self) -> bytes:
        """The working tree as a patch with git's rename detection (allow_rename tasks' tool)."""
        git(self.work, "add", "-A")
        return subprocess.run(["git", "diff", "--cached", "--find-renames", "--binary", "HEAD"],
                              cwd=self.work, check=True, capture_output=True).stdout

    def run_check(self, patch: bytes, **rules) -> edit_scope.Result:
        rules.setdefault("allowed_paths", ("src/",))
        result = edit_scope.check(patch, self.pristine, Rules(**rules))
        self.addCleanup(result.dispose)
        return result

    def test_allowed_edit_builds_pristine_plus_allowed_files(self):
        (self.work / "src/a.c").write_text("int a(void) { return 3; }\n")
        (self.work / "src/new.c").write_text("int n;\n")
        result = self.run_check(self.diff())
        self.assertTrue(result.ok, result.message)
        self.assertEqual(sorted((c.status, c.path) for c in result.changes),
                         [("A", "src/new.c"), ("M", "src/a.c")])
        self.assertIn("int a(void) { return 3; }", dict((c.path, c.added) for c in result.changes)["src/a.c"])
        self.assertEqual((result.tree / "src/a.c").read_text(), "int a(void) { return 3; }\n")
        self.assertEqual((result.tree / "tests/t.c").read_text(), "int main(void) { return 0; }\n")
        self.assertFalse((result.tree / ".git").exists())

    def test_out_of_scope_edit_is_rejected_with_the_path(self):
        (self.work / "tests/t.c").write_text("int main(void) { return 1; }\n")
        result = self.run_check(self.diff())
        self.assertEqual(result.category, "out_of_scope")
        self.assertIn("tests/t.c", result.message)
        self.assertIsNone(result.tree)

    def test_deletion_is_out_of_scope_unless_allowed(self):
        (self.work / "src/b.c").unlink()
        self.assertEqual(self.run_check(self.diff()).category, "out_of_scope")
        self.assertTrue(self.run_check(self.diff(), allow_delete=True).ok)

    def test_globs_suffixes_and_dot_directories(self):
        (self.work / "src/a.c").write_text("y\n")
        self.assertTrue(self.run_check(self.diff(), allowed_paths=("src/*.c",)).ok)
        self.assertTrue(self.run_check(self.diff(), allowed_paths=("**/a.c",)).ok)
        self.assertEqual(self.run_check(self.diff(), allowed_suffixes=(".py",)).category,
                         "out_of_scope")
        self.assertFalse(edit_scope.matches("github/x", (".github/",)))
        self.assertTrue(edit_scope.matches(".github/x", (".github/",)))

    def test_forbidden_content_names_paths_not_patterns(self):
        (self.work / "src/a.c").write_text("int a(void) { return SECRET_ORACLE(); }\n")
        result = self.run_check(self.diff(), forbidden_added=("SECRET_ORACLE",))
        self.assertEqual(result.category, "content_policy")
        self.assertIn("src/a.c", result.message)
        self.assertNotIn("SECRET_ORACLE", result.message)
        self.assertTrue(self.run_check(self.diff(), forbidden_added=("SECRET_ORACLE",)).ok
                        is False)
        pre = edit_scope.check(self.diff(), self.pristine, Rules(allowed_paths=("src/",),
                               forbidden_added=("SECRET_ORACLE",)), content_rules=False,
                               build_tree=False)
        self.assertTrue(pre.ok)

    def test_rename_counts_both_paths(self):
        git(self.work, "mv", "src/b.c", "tests/b.c")
        result = self.run_check(self.diff())
        self.assertEqual(result.category, "out_of_scope")
        self.assertIn("tests/b.c", result.message)

    def test_deny_paths_win_over_allowed_directories(self):
        (self.work / "src/b.c").write_text("x\n")
        result = self.run_check(self.diff(), deny_paths=("src/b.c",))
        self.assertEqual(result.category, "out_of_scope")

    def test_too_many_files(self):
        for name in ("a", "b"):
            (self.work / f"src/{name}.c").write_text("y\n")
        result = self.run_check(self.diff(), max_changed_files=1)
        self.assertEqual(result.category, "too_many_files")

    def test_mode_change_symlink_and_binary_are_forbidden(self):
        (self.work / "src/a.c").chmod(0o755)
        self.assertEqual(self.run_check(self.diff()).category, "forbidden_file_type")
        git(self.work, "reset", "-q", "--hard")
        (self.work / "src/new.sh").write_text("#!/bin/sh\n")
        (self.work / "src/new.sh").chmod(0o755)
        self.assertEqual(self.run_check(self.diff()).category, "forbidden_file_type")
        self.assertTrue(self.run_check(self.diff(), allow_executable=True).ok)
        (self.work / "src/new.sh").unlink()
        (self.work / "src/a.c").chmod(0o644)
        (self.work / "src/link").symlink_to("../tests/t.c")
        self.assertEqual(self.run_check(self.diff()).category, "forbidden_file_type")
        (self.work / "src/link").unlink()
        (self.work / "src/blob.bin").write_bytes(b"\x00\x01\x02")
        self.assertEqual(self.run_check(self.diff()).category, "forbidden_file_type")

    def test_patch_that_does_not_apply_is_malformed_not_out_of_scope(self):
        patch = (b"--- a/src/a.c\n+++ b/src/a.c\n@@ -1 +1 @@\n-int nothing;\n+int x;\n")
        self.assertEqual(self.run_check(patch).category, "malformed")
        # Patch text that names an allowed file but escapes through "..": git refuses it.
        escape = (b"--- a/src/../../escape\n+++ b/src/../../escape\n@@ -0,0 +1 @@\n+x\n")
        self.assertIn(self.run_check(escape).category, ("malformed", "out_of_scope"))

    def test_empty_and_oversize(self):
        self.assertEqual(self.run_check(b"").category, "empty")
        self.assertTrue(self.run_check(b"", allow_empty=True).ok)
        self.assertEqual(self.run_check(b"  \n", allow_empty=True).category, "malformed")
        (self.work / "src/a.c").write_text("z" * 5000 + "\n")
        self.assertEqual(self.run_check(self.diff(), max_patch_bytes=1000).category, "too_large")

    def test_crlf_content_is_kept_byte_for_byte(self):
        (self.work / "src/a.c").write_bytes(b"int a(void) {\r\n  return 4;\r\n}\r\n")
        result = self.run_check(self.diff())
        self.assertTrue(result.ok, result.message)
        self.assertEqual((result.tree / "src/a.c").read_bytes(),
                         b"int a(void) {\r\n  return 4;\r\n}\r\n")

    def test_dispose_removes_the_scratch_tree(self):
        (self.work / "src/a.c").write_text("q\n")
        result = edit_scope.check(self.diff(), self.pristine, Rules(allowed_paths=("src/",)))
        tree = result.tree
        result.dispose()
        self.assertFalse(tree.exists())

    # allow_rename (vllm_llm_serving_optimization): a rename is one change whose old and new
    # paths must both be editable; without it a rename stays a delete plus an add.
    def big_file(self, name: str) -> None:
        """A tracked file large enough for git's rename detection to pair an edited rename."""
        for root in (self.pristine, self.work):
            path = root / name
            if root == self.pristine:
                path.parent.chmod(0o755)
            path.write_text("".join(f"int f{i}(void) {{ return {i}; }}\n" for i in range(40)))
            if root == self.pristine:
                path.chmod(0o444)
                path.parent.chmod(0o555)
        git(self.work, "add", "-A")
        git(self.work, "commit", "-qm", "big")

    def test_rename_is_a_delete_and_an_add_by_default(self):
        self.big_file("src/big.c")
        git(self.work, "mv", "src/big.c", "src/moved.c")
        patch = self.rename_diff()
        self.assertIn(b"rename from src/big.c", patch)
        result = self.run_check(patch)
        self.assertEqual(result.category, "out_of_scope")
        self.assertIn("src/big.c", result.message)
        allowed = self.run_check(patch, allow_delete=True)
        self.assertTrue(allowed.ok, allowed.message)
        self.assertEqual(sorted((c.status, c.path) for c in allowed.changes),
                         [("A", "src/moved.c"), ("D", "src/big.c")])

    def test_allowed_rename_with_an_edit_is_one_change(self):
        self.big_file("src/big.c")
        git(self.work, "mv", "src/big.c", "src/moved.c")
        with (self.work / "src/moved.c").open("a") as stream:
            stream.write("int extra;\n")
        result = self.run_check(self.rename_diff(), allow_rename=True, max_changed_files=1)
        self.assertTrue(result.ok, result.message)
        self.assertEqual([(c.status, c.old_path, c.path, c.added) for c in result.changes],
                         [("R", "src/big.c", "src/moved.c", ("int extra;",))])
        self.assertEqual(result.metrics["changed_files"], 1)
        self.assertFalse((result.tree / "src/big.c").exists())
        self.assertTrue((result.tree / "src/moved.c").read_text().endswith("int extra;\n"))
        # The same tree as a hand-made delete + add is detected as the same rename.
        self.assertTrue(self.run_check(self.diff(), allow_rename=True).ok)

    def test_allowed_rename_needs_both_paths_in_scope(self):
        self.big_file("src/big.c")
        git(self.work, "mv", "src/big.c", "tests/big.c")
        result = self.run_check(self.rename_diff(), allow_rename=True)
        self.assertEqual((result.category, result.paths), ("out_of_scope", ("tests/big.c",)))
        git(self.work, "reset", "-q", "--hard")
        self.big_file("tests/big2.c")
        git(self.work, "mv", "tests/big2.c", "src/big2.c")
        result = self.run_check(self.rename_diff(), allow_rename=True)
        self.assertEqual((result.category, result.paths), ("out_of_scope", ("tests/big2.c",)))
        git(self.work, "reset", "-q", "--hard")
        git(self.work, "mv", "src/big.c", "src/big.h")
        result = self.run_check(self.rename_diff(), allow_rename=True, allowed_suffixes=(".c",))
        self.assertEqual((result.category, result.paths), ("out_of_scope", ("src/big.h",)))
        result = self.run_check(self.rename_diff(), allow_rename=True, deny_paths=("src/big.c",))
        self.assertEqual((result.category, result.paths), ("out_of_scope", ("src/big.c",)))

    def test_allowed_rename_keeps_mode_and_content_rules(self):
        self.big_file("src/big.c")
        git(self.work, "mv", "src/big.c", "src/moved.c")
        (self.work / "src/moved.c").chmod(0o755)
        self.assertEqual(self.run_check(self.rename_diff(), allow_rename=True).category,
                         "forbidden_file_type")
        (self.work / "src/moved.c").chmod(0o644)
        with (self.work / "src/moved.c").open("a") as stream:
            stream.write("int SECRET_ORACLE;\n")
        result = self.run_check(self.rename_diff(), allow_rename=True,
                                forbidden_added=("SECRET_ORACLE",))
        self.assertEqual((result.category, result.paths), ("content_policy", ("src/moved.c",)))


@unittest.skipUnless(shutil.which("git"), "needs git")
class SubmissionTool(unittest.TestCase):
    """tasks/shared/agent/bin/fcs2_submission.py (make_submission.sh) with and without
    allow_rename."""

    def setUp(self):
        sys.path.insert(0, str(REPO / "tasks/shared/agent/bin"))
        import fcs2_submission
        self.tool = fcs2_submission
        self.scratch = Path(tempfile.mkdtemp(prefix="submission-test-"))
        self.addCleanup(shutil.rmtree, self.scratch, ignore_errors=True)
        self.starter = self.scratch / "starter"
        (self.starter / "src").mkdir(parents=True)
        (self.starter / "src/big.c").write_text(
            "".join(f"int f{i}(void) {{ return {i}; }}\n" for i in range(40)))
        self.workspace = self.scratch / "workspace"
        self.repo = self.workspace / "repo"
        shutil.copytree(self.starter, self.repo)
        git(self.repo, "init", "-q")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "baseline")
        git(self.repo, "update-ref", "refs/frontier/baseline", "HEAD")
        git(self.repo, "mv", "src/big.c", "src/moved.c")
        with (self.repo / "src/moved.c").open("a") as stream:
            stream.write("int extra;\n")

    def submit(self, **rules) -> bytes:
        policy = {"member": "solution.patch",
                  "edit_scope": {"allowed_paths": ["src/"], "max_patch_bytes": 100000, **rules}}
        path = self.tool.make_submission(self.repo, policy, starter=self.starter,
                                         workspace=self.workspace)
        return path.read_bytes()

    def test_rename_is_refused_without_allow_rename(self):
        with self.assertRaisesRegex(self.tool.Refused, "src/big.c"):
            self.submit()

    def test_rename_patch_with_allow_rename(self):
        patch = self.submit(allow_rename=True)
        self.assertIn(b"rename from src/big.c\nrename to src/moved.c", patch)
        result = edit_scope.check(patch, self.starter,
                                  Rules(allowed_paths=("src/",), allow_rename=True))
        self.addCleanup(result.dispose)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(sorted(p.name for p in (result.tree / "src").iterdir()), ["moved.c"])

    def test_without_allow_rename_the_patch_is_a_delete_and_an_add(self):
        patch = self.submit(allow_delete=True)
        self.assertNotIn(b"rename from", patch)
        self.assertIn(b"deleted file mode", patch)



@unittest.skipUnless(shutil.which("git") and shutil.which("sh"), "needs git and sh")
class SubmissionBaseline(unittest.TestCase):
    """The patch is taken against the starter commit prepare_workspace.sh records
    (refs/frontier/baseline), so committed and uncommitted changes both count."""

    def setUp(self):
        sys.path.insert(0, str(REPO / "tasks/shared/agent/bin"))
        import fcs2_submission
        self.tool = fcs2_submission
        self.scratch = Path(tempfile.mkdtemp(prefix="submission-baseline-"))
        self.addCleanup(shutil.rmtree, self.scratch, ignore_errors=True)
        self.starter = self.scratch / "starter"
        (self.starter / "src").mkdir(parents=True)
        (self.starter / "src/a.c").write_text("int a;\n")
        (self.starter / "src/b.c").write_text("int b;\n")
        (self.starter / "README.md").write_text("starter\n")
        self.workspace = self.scratch / "workspace"
        self.workspace.mkdir()
        self.repo = self.workspace / "repo"
        # prepare_workspace.sh itself, with its fixed paths pointed at this scratch tree.
        script = (REPO / "tasks/shared/agent/bin/prepare_workspace.sh").read_text()
        script = script.replace("/frontier/task/repo", str(self.starter))
        script = script.replace("/workspace", str(self.workspace))
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
        subprocess.run(["sh", "-c", script, "prepare", str(self.repo)], env=env, check=True,
                       capture_output=True)

    def submit(self, **rules) -> bytes:
        policy = {"member": "solution.patch",
                  "edit_scope": {"allowed_paths": ["src/"], "max_patch_bytes": 100000,
                                 "allow_delete": True, **rules}}
        return self.tool.make_submission(self.repo, policy, starter=self.starter,
                                         workspace=self.workspace).read_bytes()

    def changed(self, patch: bytes) -> set[str]:
        return {line.split(" b/", 1)[1] for line in patch.decode().splitlines()
                if line.startswith("diff --git ")}

    def test_prepare_records_the_baseline_ref(self):
        self.assertEqual(git(self.repo, "rev-parse", "refs/frontier/baseline"),
                         git(self.repo, "rev-parse", "HEAD"))

    def test_commit_then_submit(self):
        (self.repo / "src/a.c").write_text("int a = 1;\n")
        git(self.repo, "commit", "-qam", "edit")
        patch = self.submit()
        self.assertEqual(self.changed(patch), {"src/a.c"})
        self.assertIn(b"+int a = 1;", patch)

    def test_amend(self):
        (self.repo / "src/a.c").write_text("int a = 1;\n")
        git(self.repo, "commit", "-qam", "edit")
        (self.repo / "src/a.c").write_text("int a = 2;\n")
        git(self.repo, "commit", "-qa", "--amend", "--no-edit")
        patch = self.submit()
        self.assertIn(b"+int a = 2;", patch)
        self.assertNotIn(b"int a = 1;", patch)

    def test_multiple_commits_with_new_and_deleted_files(self):
        (self.repo / "src/a.c").write_text("int a = 1;\n")
        git(self.repo, "commit", "-qam", "one")
        (self.repo / "src/new.c").write_text("int n;\n")
        git(self.repo, "add", "src/new.c")
        git(self.repo, "commit", "-qm", "two")
        git(self.repo, "rm", "-q", "src/b.c")
        git(self.repo, "commit", "-qm", "three")
        patch = self.submit()
        self.assertEqual(self.changed(patch), {"src/a.c", "src/new.c", "src/b.c"})
        self.assertIn(b"deleted file mode", patch)
        self.assertIn(b"new file mode", patch)

    def test_rebase_squash(self):
        base = git(self.repo, "rev-parse", "HEAD").strip()
        (self.repo / "src/a.c").write_text("int a = 1;\n")
        git(self.repo, "commit", "-qam", "one")
        (self.repo / "src/b.c").write_text("int b = 1;\n")
        git(self.repo, "commit", "-qam", "two")
        git(self.repo, "reset", "-q", "--soft", base)
        git(self.repo, "commit", "-qm", "squashed")
        self.assertEqual(self.changed(self.submit()), {"src/a.c", "src/b.c"})

    def test_rebase_dropping_a_commit(self):
        (self.repo / "src/a.c").write_text("int a = 1;\n")
        git(self.repo, "commit", "-qam", "one")
        (self.repo / "src/b.c").write_text("int b = 1;\n")
        git(self.repo, "commit", "-qam", "two")
        git(self.repo, "rebase", "-q", "--onto", "refs/frontier/baseline", "HEAD~1")
        self.assertEqual(self.changed(self.submit()), {"src/b.c"})

    def test_committed_and_uncommitted_mix(self):
        (self.repo / "src/a.c").write_text("int a = 1;\n")
        git(self.repo, "commit", "-qam", "one")
        (self.repo / "src/b.c").write_text("int b = 1;\n")  # not committed
        (self.repo / "src/untracked.c").write_text("int u;\n")  # new, allowed path
        (self.repo / "build.log").write_text("noise\n")  # new, outside: left out
        self.assertEqual(self.changed(self.submit()), {"src/a.c", "src/b.c", "src/untracked.c"})

    def test_committed_out_of_scope_edit_is_refused(self):
        (self.repo / "README.md").write_text("changed\n")
        git(self.repo, "commit", "-qam", "outside")
        with self.assertRaisesRegex(self.tool.Refused, "README.md"):
            self.submit()

    def test_reset_back_to_baseline_is_empty(self):
        (self.repo / "src/a.c").write_text("int a = 1;\n")
        git(self.repo, "commit", "-qam", "one")
        git(self.repo, "reset", "-q", "--hard", "refs/frontier/baseline")
        with self.assertRaisesRegex(self.tool.Refused, edit_scope.CATEGORIES["empty"]):
            self.submit()


if __name__ == "__main__":
    unittest.main()
