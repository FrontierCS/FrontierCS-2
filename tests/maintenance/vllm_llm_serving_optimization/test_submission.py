"""The Agent's shared submission tool (tasks/shared/agent/bin, with this task's generated
submission policy) against the task's own judge-side patch policy (judge/submission.py)."""
from vllm_task import REPO_ROOT, TASK_ROOT  # noqa: E402
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path[:0] = [str(REPO_ROOT / 'scripts'), str(REPO_ROOT / 'tasks/shared/lib'),
                str(REPO_ROOT / 'tasks/shared/agent/bin')]
import fcs2_submission  # noqa: E402
import task_manifest  # noqa: E402
from patch_policy import validate_patch  # noqa: E402
from submission import inspect_submission  # noqa: E402

POLICY = task_manifest.submission_policy(task_manifest.read_task(TASK_ROOT))


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # The starter (/frontier/task/repo) and the judge's clean git checkout of it.
        self.starter = self.root / 'starter'
        (self.starter / 'vllm/v1/core/sched').mkdir(parents=True)
        (self.starter / 'vllm/v1/core/sched/scheduler.py').write_text('VALUE = 1\n')
        self.clean = self.root / 'clean'
        shutil.copytree(self.starter, self.clean)
        self.commit(self.clean)
        # What the shared prepare_workspace.sh makes.
        self.work = self.root / 'repo'
        shutil.copytree(self.starter, self.work)
        self.commit(self.work)
        self.patch = self.root / 'payload/solution.patch'

    def git(self, where, *args):
        return subprocess.run(['git', '-C', str(where), *args], check=True,
                              capture_output=True, text=True).stdout

    def commit(self, where):
        self.git(where, 'init', '--quiet')
        self.git(where, 'add', '--all')
        self.git(where, '-c', 'user.name=t', '-c', 'user.email=t@invalid',
                 'commit', '--quiet', '-m', 'baseline')

    def submit(self):
        return fcs2_submission.make_submission(self.work, POLICY, starter=self.starter,
                                               workspace=self.root)

    def test_empty_patch_is_original_noop(self):
        self.assertEqual(self.submit(), self.patch)
        self.assertEqual(self.patch.read_bytes(), b'')
        result = inspect_submission(self.patch, self.clean)
        self.assertTrue(result['valid_submission'])
        self.assertNotIn('score', result)
        self.assertEqual(self.patch.stat().st_mode & 0o777, 0o600)

    def test_edits_and_new_allowlisted_files_are_included(self):
        (self.work / 'vllm/v1/core/sched/scheduler.py').write_text('VALUE = 2\n')
        (self.work / 'vllm/v1/core/sched/new_queue.py').write_text('QUEUE = []\n')
        self.submit()
        result = inspect_submission(self.patch, self.clean)
        self.assertTrue(result['valid_submission'], result)
        self.assertEqual(result['patch_metrics']['changed_files'], 2)
        self.git(self.clean, 'apply', '--check', str(self.patch))

    def test_tool_refuses_what_the_judge_rejects(self):
        (self.work / 'tests').mkdir()
        (self.work / 'tests/answer.py').write_text('ANSWER = 1\n')
        # New files outside the allowlist never enter the patch.
        self.submit()
        self.assertEqual(self.patch.read_bytes(), b'')
        (self.work / 'vllm/v1/core/sched/scheduler.py').unlink()
        with self.assertRaises(fcs2_submission.Refused):  # deletion
            self.submit()
        (self.work / 'vllm/v1/core/sched/scheduler.py').write_text('VALUE = 1\n')
        (self.work / 'vllm/v1/core/sched/binary.py').write_bytes(b'a\x00b')
        with self.assertRaises(fcs2_submission.Refused):  # binary
            self.submit()
        (self.work / 'vllm/v1/core/sched/binary.py').unlink()
        (self.work / 'vllm/v1/core/sched/notes.txt').write_text('text\n')
        with self.assertRaises(fcs2_submission.Refused):  # not .py/.pyi
            self.submit()

    def test_judge_rejects_out_of_scope_and_binary_changes(self):
        self.patch.parent.mkdir()
        self.patch.write_text('diff --git a/tests/answer.py b/tests/answer.py\nnew file mode 100644\n'
                              '--- /dev/null\n+++ b/tests/answer.py\n@@ -0,0 +1 @@\n+ANSWER = 1\n')
        self.assertFalse(validate_patch(self.patch)[0])
        self.assertEqual(inspect_submission(self.patch, self.clean)['category'], 'out_of_scope')
        self.patch.write_text('diff --git a/vllm/v1/core/sched/b.py b/vllm/v1/core/sched/b.py\n'
                              'GIT binary patch\nliteral 3\n')
        self.assertIn('binary', validate_patch(self.patch)[1])
        self.assertEqual(inspect_submission(self.patch, self.clean)['category'],
                         'forbidden_file_type')

    def test_malformed_patch_and_symlink_are_not_accepted(self):
        self.patch.parent.mkdir()
        self.patch.write_text('not a patch\n')
        result = inspect_submission(self.patch, self.clean)
        self.assertFalse(result['valid_submission'])
        self.assertEqual(result['category'], 'malformed')
        self.patch.unlink()
        self.patch.symlink_to(self.clean / 'vllm/v1/core/sched/scheduler.py')
        result = inspect_submission(self.patch, self.clean)
        self.assertFalse(result['valid_submission'])
        self.assertEqual(result['category'], 'malformed')

    def test_missing_backend_is_an_error_not_a_smoke_score(self):
        (self.work / 'vllm/v1/core/sched/scheduler.py').write_text('VALUE = 2\n')
        self.submit()
        with self.assertRaises(RuntimeError):
            inspect_submission(self.patch, self.root / 'missing')

    def test_oversize_patch_is_refused_and_preserves_previous_payload(self):
        self.submit()
        (self.work / 'vllm/v1/core/sched/large.py').write_text('x' * 1500001 + '\n')
        with self.assertRaises(fcs2_submission.Refused):
            self.submit()
        self.assertEqual(self.patch.read_bytes(), b'')


if __name__ == '__main__':
    unittest.main()
