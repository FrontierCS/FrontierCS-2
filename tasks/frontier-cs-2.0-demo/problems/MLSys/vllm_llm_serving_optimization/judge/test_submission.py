import subprocess
import tempfile
import unittest
from pathlib import Path

from patch_policy import validate_patch
from submission import inspect_submission

BIN = Path(__file__).resolve().parents[1] / 'agent-base/frontier/task/bin'


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clean = self.root / 'clean'
        self.clean.mkdir()
        self.git(self.clean, 'init', '--quiet')
        self.git(self.clean, 'config', 'user.name', 'submission-test')
        self.git(self.clean, 'config', 'user.email', 'test@invalid')
        (self.clean / 'vllm/v1/core/sched').mkdir(parents=True)
        (self.clean / 'vllm/v1/core/sched/scheduler.py').write_text('VALUE = 1\n')
        self.git(self.clean, 'add', '--all')
        self.git(self.clean, 'commit', '--quiet', '-m', 'base')
        self.work = self.root / 'work'
        self.patch = self.root / 'payload/solution.patch'
        self.run_script('prepare_workspace.sh', self.work, self.clean)

    def git(self, where, *args):
        return subprocess.run(['git', '-C', str(where), *args], check=True, capture_output=True, text=True).stdout

    def run_script(self, script, *args):
        return subprocess.run([str(BIN / script), *map(str, args)], check=True, capture_output=True, text=True).stdout.strip()

    def test_empty_patch_is_original_noop(self):
        self.assertEqual(self.run_script('make_submission.sh', self.work, self.patch), str(self.patch))
        self.assertEqual(self.patch.read_bytes(), b'')
        result = inspect_submission(self.patch, self.clean)
        self.assertTrue(result['valid_submission'])
        self.assertNotIn('score', result)
        self.assertEqual(self.patch.stat().st_mode & 0o777, 0o600)

    def test_commits_and_new_files_are_included_against_original_base(self):
        (self.work / 'vllm/v1/core/sched/scheduler.py').write_text('VALUE = 2\n')
        self.git(self.work, 'add', '--all')
        self.git(self.work, 'commit', '--quiet', '-m', 'optimization')
        (self.work / 'vllm/v1/core/sched/new_queue.py').write_text('QUEUE = []\n')
        self.run_script('make_submission.sh', self.work, self.patch)
        result = inspect_submission(self.patch, self.clean)
        self.assertTrue(result['valid_submission'], result)
        self.assertEqual(result['patch_metrics']['changed_files'], 2)
        applied = self.root / 'applied'
        self.run_script('prepare_workspace.sh', applied, self.clean)
        self.git(applied, 'apply', str(self.patch))
        self.assertEqual((applied / 'vllm/v1/core/sched/scheduler.py').read_text(), 'VALUE = 2\n')
        self.assertTrue((applied / 'vllm/v1/core/sched/new_queue.py').exists())
        self.assertEqual((self.clean / 'vllm/v1/core/sched/scheduler.py').read_text(), 'VALUE = 1\n')

    def test_policy_rejects_out_of_scope_and_binary_changes(self):
        (self.work / 'tests').mkdir()
        (self.work / 'tests/answer.py').write_text('ANSWER = 1\n')
        self.run_script('make_submission.sh', self.work, self.patch)
        self.assertFalse(validate_patch(self.patch)[0])
        (self.work / 'tests/answer.py').unlink()
        (self.work / 'vllm/v1/core/sched/binary.py').write_bytes(b'a\x00b')
        self.run_script('make_submission.sh', self.work, self.patch)
        self.assertIn('binary', validate_patch(self.patch)[1])

    def test_malformed_patch_and_symlink_are_not_accepted(self):
        self.patch.parent.mkdir()
        self.patch.write_text('not a patch\n')
        self.assertFalse(inspect_submission(self.patch, self.clean)['valid_submission'])
        self.patch.unlink()
        self.patch.symlink_to(self.clean / 'vllm/v1/core/sched/scheduler.py')
        self.assertFalse(inspect_submission(self.patch, self.clean)['valid_submission'])

    def test_missing_backend_is_an_error_not_a_smoke_score(self):
        self.run_script('make_submission.sh', self.work, self.patch)
        with self.assertRaises(RuntimeError):
            inspect_submission(self.patch, self.root / 'missing')

    def test_failed_oversize_update_preserves_previous_payload(self):
        self.run_script('make_submission.sh', self.work, self.patch)
        (self.work / 'vllm/v1/core/sched/large.py').write_text('x' * 1500001 + '\n')
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_script('make_submission.sh', self.work, self.patch)
        self.assertEqual(self.patch.read_bytes(), b'')
        self.assertEqual(list(self.patch.parent.glob('*.tmp.*')), [])


if __name__ == '__main__':
    unittest.main()
