"""Real Git repositories exercise fixed source and declared checkout identity."""
import deployment_test_guard  # noqa: F401  # isort: skip

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/frontier-cs-2.0-demo/problems/MLSys/vllm_llm_serving_optimization'
sys.path.insert(0, str(TASK / 'judge'))

from serving_eval.accuracy import CorrectnessRuntimeError
from serving_eval.swe_test_worker import inspect_testbed


class SourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.new_repo('testbed')
        (self.repo / 'plain.txt').write_bytes(b'fixed\nsource\n')
        (self.repo / 'script.bat').write_bytes(b'one\ntwo\n')
        (self.repo / 'link').symlink_to('plain.txt')
        self.base = self.commit(self.repo)

    def git(self, repo, *args):
        return subprocess.check_output(
            ['git', '-c', 'protocol.file.allow=always', '-c', 'core.fsmonitor=false',
             '-c', 'user.name=Identity fixture',
             '-c', 'user.email=identity@example.invalid', '-C', str(repo), *args],
            stderr=subprocess.PIPE, env={**os.environ, 'GIT_CONFIG_GLOBAL': '/dev/null',
                                        'GIT_CONFIG_NOSYSTEM': '1'}).decode().strip()

    def new_repo(self, name):
        repo = self.root / name
        repo.mkdir()
        self.git(repo, 'init', '-q')
        return repo

    def commit(self, repo):
        self.git(repo, 'add', '.')
        self.git(repo, 'commit', '-qm', 'fixture')
        return self.git(repo, 'rev-parse', 'HEAD')

    def inspect(self):
        return inspect_testbed(self.repo, self.base, 1048576)

    def rejected(self):
        with self.assertRaises(CorrectnessRuntimeError):
            self.inspect()

    def fixed_crlf(self, attrs='*.bat text eol=crlf\n'):
        (self.repo / '.gitattributes').write_text(attrs)
        self.base = self.commit(self.repo)
        (self.repo / 'script.bat').write_bytes(b'one\r\ntwo\r\n')

    def submodule(self, nested=False):
        child = self.new_repo('upstream')
        (child / 'child.txt').write_text('child\n')
        self.commit(child)
        if nested:
            leaf = self.new_repo('leaf')
            (leaf / 'leaf.txt').write_text('leaf\n')
            self.commit(leaf)
            self.git(child, 'submodule', 'add', '-q', str(leaf), 'leaf')
            self.commit(child)
        self.git(self.repo, 'submodule', 'add', '-q', str(child), 'deps/child')
        self.base = self.commit(self.repo)
        if nested:
            self.git(self.repo, 'submodule', 'update', '--init', '--recursive')
        return self.repo / 'deps/child'

    def test_plain_and_symlink_and_packaging_mode_commit(self):
        self.assertTrue(self.inspect()['fresh_worktree'])
        (self.repo / 'plain.txt').chmod(0o755)
        self.commit(self.repo)
        self.assertEqual(self.inspect()['regular_executable_mode_differences'], 1)

    def test_changed_head_and_staged_content_rejected(self):
        (self.repo / 'plain.txt').write_text('changed\n')
        self.commit(self.repo)
        self.rejected()
        self.git(self.repo, 'reset', '--hard', self.base)
        (self.repo / 'plain.txt').write_text('staged\n')
        self.git(self.repo, 'add', '.')
        (self.repo / 'plain.txt').write_bytes(b'fixed\nsource\n')
        self.rejected()

    def test_assume_unchanged_does_not_hide_dirty_bytes(self):
        self.git(self.repo, 'update-index', '--assume-unchanged', 'plain.txt')
        stat = (self.repo / 'plain.txt').stat()
        (self.repo / 'plain.txt').write_bytes(b'other\nbytes!\n')
        os.utime(self.repo / 'plain.txt', ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.rejected()

    def test_untracked_content_rejected_even_if_local_exclude_hides_it(self):
        (self.repo / '.git/info/exclude').write_text('hidden\n')
        (self.repo / 'hidden').write_text('not fixed\n')
        self.rejected()

    def test_symlink_type_or_target_change_rejected(self):
        (self.repo / 'link').unlink()
        (self.repo / 'link').symlink_to('script.bat')
        self.rejected()
        (self.repo / 'link').unlink()
        (self.repo / 'link').write_bytes(b'plain.txt')
        self.rejected()

    def test_explicit_fixed_crlf_accepted(self):
        self.fixed_crlf()
        (self.repo / '.git/info/attributes').write_text('*.bat -text eol=lf\n')
        receipt = self.inspect()
        self.assertEqual(receipt['declared_checkout_representations'][0]['path'], 'script.bat')

    def test_nested_fixed_attribute_precedence(self):
        (self.repo / '.gitattributes').write_text('*.bat text eol=lf\n')
        nested = self.repo / 'nested'
        nested.mkdir()
        (nested / '.gitattributes').write_text('*.bat text eol=crlf\n')
        (nested / 'other.bat').write_bytes(b'nested\n')
        self.base = self.commit(self.repo)
        (nested / 'other.bat').write_bytes(b'nested\r\n')
        self.assertEqual(self.inspect()['declared_checkout_representations'][0]['path'], 'nested/other.bat')
        (self.repo / 'script.bat').write_bytes(b'one\r\ntwo\r\n')
        self.rejected()

    def test_changed_fixed_attributes_rejected(self):
        self.fixed_crlf()
        (self.repo / '.gitattributes').write_text('*.bat text eol=lf\n')
        self.rejected()

    def test_undeclared_or_extra_crlf_or_changed_text_rejected(self):
        (self.repo / 'script.bat').write_bytes(b'one\r\ntwo\r\n')
        self.rejected()
        self.fixed_crlf()
        for changed in (b'one\r\r\ntwo\r\n', b'one\r\ntwo\n', b'new\r\ntwo\r\n'):
            with self.subTest(changed=changed):
                (self.repo / 'script.bat').write_bytes(changed)
                self.rejected()

    def test_mutable_info_or_global_attributes_cannot_authorize_crlf(self):
        (self.repo / 'script.bat').write_bytes(b'one\r\ntwo\r\n')
        (self.repo / '.git/info/attributes').write_text('*.bat text eol=crlf\n')
        global_attrs = self.root / 'global-attributes'
        global_attrs.write_text('*.bat text eol=crlf\n')
        config = self.root / 'global-config'
        config.write_text('[core]\nattributesFile = ' + str(global_attrs) + '\n')
        with patch.dict(os.environ, {'GIT_CONFIG_GLOBAL': str(config)}):
            self.rejected()

    def test_unsupported_declared_conversions_rejected_without_running_filters(self):
        marker = self.root / 'executed'
        for declaration in ('filter=evil', 'working-tree-encoding=UTF-16', 'ident'):
            with self.subTest(declaration=declaration):
                # Commit the declaration without invoking a configured filter.
                (self.repo / 'script.bat').write_bytes(b'one\ntwo\n')
                (self.repo / '.gitattributes').write_text('*.bat text eol=crlf ' + declaration + '\n')
                # Encoding declarations need an exact blob index, not checkout conversion.
                attrs = self.git(self.repo, 'hash-object', '-w', '--no-filters', '.gitattributes')
                self.git(self.repo, 'update-index', '--add', '--cacheinfo', '100644,' + attrs + ',.gitattributes')
                tree = self.git(self.repo, 'write-tree')
                self.base = self.git(self.repo, 'commit-tree', tree, '-p', 'HEAD', '-m', 'fixed attributes')
                self.git(self.repo, 'update-ref', 'HEAD', self.base)
                self.git(self.repo, 'config', 'filter.evil.clean', 'touch ' + str(marker))
                self.git(self.repo, 'config', 'core.fsmonitor', 'touch ' + str(marker))
                (self.repo / 'script.bat').write_bytes(b'one\r\ntwo\r\n')
                self.rejected()
                self.assertFalse(marker.exists())

    def test_local_filter_and_fsmonitor_never_execute(self):
        marker = self.root / 'executed'
        (self.repo / '.git/info/attributes').write_text('* filter=evil\n')
        self.git(self.repo, 'config', 'filter.evil.clean', 'touch ' + str(marker))
        self.git(self.repo, 'config', 'core.fsmonitor', 'touch ' + str(marker))
        self.assertTrue(self.inspect()['fresh_worktree'])
        self.assertFalse(marker.exists())

    def test_mutable_git_template_cannot_seed_private_filter_config(self):
        marker = self.root / 'executed'
        (self.repo / '.gitattributes').write_text('plain.txt filter=evil\n')
        self.base = self.commit(self.repo)
        template = self.root / 'template'
        template.mkdir()
        (template / 'config').write_text('[filter "evil"]\nclean = touch ' + str(marker) + '\n')
        self.git(self.repo, 'config', 'init.templateDir', str(template))
        self.assertTrue(self.inspect()['fresh_worktree'])
        self.assertFalse(marker.exists())

    def test_exact_initialized_nested_submodules_accepted(self):
        self.submodule(nested=True)
        self.assertEqual(len(self.inspect()['submodules_verified']), 2)

    def test_changed_gitlink_in_root_head_rejected(self):
        child = self.submodule()
        (child / 'child.txt').write_text('new child\n')
        self.commit(child)
        self.commit(self.repo)
        self.rejected()

    def test_child_wrong_head_even_if_content_matches_rejected(self):
        child = self.submodule()
        self.git(child, 'commit', '--allow-empty', '-qm', 'wrong commit')
        self.rejected()

    def test_uninitialized_submodule_rejected(self):
        self.submodule()
        self.git(self.repo, 'submodule', 'deinit', '-f', '--all')
        self.rejected()

    def test_dirty_untracked_or_staged_child_rejected(self):
        child = self.submodule()
        (child / 'child.txt').write_text('changed\n')
        self.rejected()
        self.git(child, 'checkout', '--', 'child.txt')
        (child / 'extra').write_text('untracked\n')
        self.rejected()
        (child / 'extra').unlink()
        (child / 'child.txt').write_text('staged\n')
        self.git(child, 'add', '.')
        (child / 'child.txt').write_text('child\n')
        self.rejected()

    def test_submodule_gitdir_escape_rejected(self):
        child = self.submodule()
        external = self.root / 'external-git'
        gitdir = Path(self.git(child, 'rev-parse', '--absolute-git-dir'))
        gitdir.rename(external)
        (child / '.git').write_text('gitdir: ' + str(external) + '\n')
        self.rejected()

    def test_submodule_ancestor_symlink_rejected(self):
        self.submodule()
        deps = self.repo / 'deps'
        moved = self.root / 'moved-deps'
        deps.rename(moved)
        deps.symlink_to(moved, target_is_directory=True)
        self.rejected()

    def test_regular_file_ancestor_symlink_inside_repo_rejected(self):
        nested = self.repo / 'nested'
        nested.mkdir()
        (nested / 'file').write_text('fixed\n')
        self.base = self.commit(self.repo)
        nested.rename(self.repo / 'moved')
        nested.symlink_to('moved', target_is_directory=True)
        self.rejected()


if __name__ == '__main__':
    unittest.main()
