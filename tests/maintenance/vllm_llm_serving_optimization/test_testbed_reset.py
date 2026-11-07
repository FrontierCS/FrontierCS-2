"""Real Git worktree resets between synthetic correctness runs; no container."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from serving_eval.testbed_reset import ResetError, TestbedReset


class TestbedResetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / 'testbed'
        self.repo.mkdir()
        (self.repo / 'module.py').write_text('VALUE = 0\n')
        (self.repo / '.gitignore').write_text('*.so\n__pycache__/\nbuild/\n')
        (self.repo / 'build').mkdir()
        (self.repo / 'build' / 'ext.so').write_bytes(b'image-built extension')
        (self.repo / 'ignored-link.so').symlink_to('build/ext.so')
        self.git('init', '-q')
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'base')
        self.head = self.git('rev-parse', 'HEAD').strip()
        self.reset = TestbedReset(self.repo)

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True)

    def test_snapshot_then_reset_restores_tracked_files_and_image_ignored_state(self):
        snapshot = self.reset.snapshot()
        self.assertEqual(snapshot, {'head': self.head, 'ignored_files': 2})
        # A run: patch applied, caches and a rebuilt extension left behind, an
        # untracked file created, the image extension modified in place.
        (self.repo / 'module.py').write_text('VALUE = 1\n')
        (self.repo / 'untracked.txt').write_text('left behind\n')
        (self.repo / '__pycache__').mkdir()
        (self.repo / '__pycache__' / 'module.cpython-312.pyc').write_bytes(b'stale bytecode')
        (self.repo / 'build' / 'ext.so').write_bytes(b'rebuilt from patched sources')
        (self.repo / 'build' / 'new.so').write_bytes(b'new artifact')
        receipt = self.reset.reset()
        self.assertEqual((self.repo / 'module.py').read_text(), 'VALUE = 0\n')
        self.assertFalse((self.repo / 'untracked.txt').exists())
        self.assertFalse((self.repo / '__pycache__' / 'module.cpython-312.pyc').exists())
        self.assertFalse((self.repo / 'build' / 'ext.so').exists(), 'changed image artifact is removed, not kept')
        self.assertFalse((self.repo / 'build' / 'new.so').exists())
        self.assertTrue((self.repo / 'ignored-link.so').is_symlink())
        self.assertEqual(receipt['head'], self.head)
        self.assertEqual(receipt['removed_changed_or_new_ignored'], 3)
        self.assertEqual(receipt['missing_inventory_files'], 0)
        self.assertEqual(receipt['verified_ignored_files'], 1)
        self.assertEqual(self.git('status', '--porcelain', '--untracked-files=all'), '')
        second = self.reset.reset()
        self.assertEqual(second['removed_changed_or_new_ignored'], 0)
        self.assertEqual(second['missing_inventory_files'], 1)

    def test_unclean_image_or_missing_snapshot_fail_closed(self):
        with self.assertRaisesRegex(ResetError, 'prior snapshot'):
            self.reset.reset()
        (self.repo / 'dirty.txt').write_text('dirty\n')
        with self.assertRaisesRegex(ResetError, 'not clean'):
            self.reset.snapshot()
        (self.repo / 'dirty.txt').unlink()
        self.reset.snapshot()
        with self.assertRaisesRegex(ResetError, 'already taken'):
            self.reset.snapshot()

    def test_stale_index_lock_is_cleared_and_head_moves_are_refused(self):
        self.reset.snapshot()
        (self.repo / '.git' / 'index.lock').touch()
        self.reset.reset()
        self.assertFalse((self.repo / '.git' / 'index.lock').exists())
        (self.repo / 'module.py').write_text('VALUE = 2\n')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qam', 'moved')
        with self.assertRaisesRegex(ResetError, 'HEAD changed'):
            self.reset.reset()


if __name__ == '__main__':
    unittest.main()
