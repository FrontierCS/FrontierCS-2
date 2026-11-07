"""Actual local subprocess fixtures, distinct from real SWE/model acceptance."""
import copy
import contextlib
import io
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from serving_eval import swe_test_worker as worker
from serving_eval.accuracy import CorrectnessRuntimeError, build_test_request, load_swe_cases
from swebench.harness.constants import START_TEST_OUTPUT, END_TEST_OUTPUT
from swebench.harness.run_evaluation import GIT_APPLY_CMDS


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class PreparedWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / 'testbed'
        self.repo.mkdir()
        (self.repo / 'answer.py').write_text('VALUE = 0\n')
        (self.repo / 'test_answer.py').write_text("print('unpatched tests')\n")
        self.git('init', '-q')
        self.git('add', '.')
        self.git('-c', 'user.name=Worker fixture', '-c', 'user.email=worker@example.invalid', 'commit', '-qm', 'base')
        self.base = self.git('rev-parse', 'HEAD').strip()
        gold = ("from answer import VALUE\nimport sys\n"
                "print(('PASSED' if VALUE == 1 else 'FAILED') + ' test_answer')\n"
                "sys.exit(0 if VALUE == 1 else 1)\n")
        self.test_patch = self.diff('test_answer.py', gold)
        self.candidate = self.diff('answer.py', 'VALUE = 1\n')
        self.row = {'instance_id': 'local-fixture', 'repo': 'fixture/repo', 'version': 'fixture',
                    'base_commit': self.base, 'test_patch': self.test_patch}
        reset = f'git checkout {self.base} test_answer.py'
        self.commands = [f'cd {shlex.quote(str(self.repo))}', reset,
                         "git apply -v - <<'EOF_TEST'\n" + self.test_patch + '\nEOF_TEST',
                         f": '{START_TEST_OUTPUT}'", 'python3 test_answer.py',
                         f": '{END_TEST_OUTPUT}'", reset]
        for name, value in [('_case', lambda _: self.row),
                            ('build_test_request', self.request),
                            ('_harness', lambda: (None, {'fixture/repo': {'fixture': {}}}, lambda *a: self.commands))]:
            context = patch.object(worker, name, value)
            context.start()
            self.addCleanup(context.stop)

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True)

    def diff(self, name, changed):
        path = self.repo / name
        original = path.read_text()
        path.write_text(changed)
        result = self.git('diff', '--', name)
        path.write_text(original)
        return result

    def request(self, ident='local-fixture', candidate=None):
        if ident != self.row['instance_id']:
            raise CorrectnessRuntimeError('wrong fixture')
        script = '\n'.join(['#!/bin/bash', 'set -uxo pipefail'] + self.commands) + '\n'
        request = {'instance_id': ident, 'candidate_patch': self.candidate if candidate is None else candidate,
                   'eval_script': script, 'eval_script_sha256': sha(script),
                   'timeout_seconds': 1200, 'harness_version': '4.1.0'}
        request['request_sha256'] = sha(json.dumps(request, sort_keys=True))
        return request

    def backend(self, limit=65536):
        return worker.PreparedSWEWorker(instance_id='local-fixture', testbed=self.repo, max_output_bytes=limit,
                                        testbed_python=sys.executable)

    def test_real_candidate_then_gold_tests_and_cleanup(self):
        backend = self.backend()
        receipt = backend.run_correctness(self.request())
        self.assertEqual(receipt['execution'], 'complete')
        self.assertEqual(receipt['stage_exit_codes'], {'setup_0': 0, 'test_reset': 0, 'test_patch': 0, 'tests': 0, 'test_cleanup': 0})
        self.assertIn('PASSED test_answer', receipt['log'])
        self.assertEqual(len(receipt['eval_execution']['executed_script_sha256']), 64)
        self.assertNotEqual(receipt['eval_execution']['executed_script_sha256'], receipt['eval_script_sha256'])
        self.assertEqual((self.repo / 'test_answer.py').read_text(), "print('unpatched tests')\n")
        self.assertEqual((self.repo / 'answer.py').read_text(), 'VALUE = 1\n')
        with self.assertRaisesRegex(CorrectnessRuntimeError, 'consumed'):
            backend.run_correctness(self.request())

    def test_stdio_cli_runs_actual_worker_and_emits_one_bound_json_receipt(self):
        request = self.request()
        source = io.TextIOWrapper(io.BytesIO(json.dumps(request).encode()))
        output = io.StringIO()
        with patch.object(worker, 'PreparedSWEWorker', return_value=self.backend()), \
                patch.object(worker.sys, 'stdin', source), contextlib.redirect_stdout(output):
            worker.main(['--instance-id', 'local-fixture', '--request', '-', '--receipt', '-',
                         '--max-request-bytes', '65536', '--max-output-bytes', '65536'])
        receipt = json.loads(output.getvalue())
        self.assertEqual(receipt['request_sha256'], request['request_sha256'])
        self.assertEqual(receipt['execution'], 'complete')
        self.assertEqual(receipt['stage_exit_codes']['tests'], 0)
        self.assertIn('PASSED test_answer', receipt['log'])
        self.assertNotIn('score', receipt)

    def test_stdio_input_bound_rejects_before_worker_execution(self):
        source = io.TextIOWrapper(io.BytesIO(b'x' * 65))
        with patch.object(worker, 'PreparedSWEWorker') as backend, \
                patch.object(worker.sys, 'stdin', source), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                worker.main(['--instance-id', 'local-fixture', '--request', '-', '--receipt', '-',
                             '--max-request-bytes', '64', '--max-output-bytes', '65536'])
        backend.assert_not_called()

    def test_candidate_test_edit_is_reset_and_failing_tests_remain_complete(self):
        tamper = self.diff('test_answer.py', "print('PASSED test_answer')\n")
        receipt = self.backend().run_correctness(self.request(candidate=tamper))
        self.assertEqual(receipt['execution'], 'complete')
        self.assertEqual(receipt['stage_exit_codes']['tests'], 1)
        self.assertIn('FAILED test_answer', receipt['log'])
        self.assertTrue(receipt['tests_completed'])
        self.assertNotIn('correct', receipt)
        self.assertNotIn('score', receipt)

    def test_bad_patch_records_all_original_fallbacks(self):
        receipt = self.backend().run_correctness(self.request(candidate='not a patch\n'))
        self.assertEqual(receipt['execution'], 'candidate_patch_rejected')
        self.assertEqual([r['command'] for r in receipt['patch_attempts']],
                         [shlex.split(command) + ['-'] for command in GIT_APPLY_CMDS])
        self.assertFalse(receipt['candidate_patch_applied'])
        self.assertTrue(receipt['patch_apply_log'])
        self.assertNotIn('eval_execution', receipt)

    def test_failed_setup_is_evidenced_and_stops_before_gold_patch_or_tests(self):
        self.commands.insert(1, "printf 'offline dependency unavailable\\n'; false")
        receipt = self.backend().run_correctness(self.request())
        self.assertEqual(receipt['execution'], 'setup_failed')
        self.assertEqual(receipt['stage_exit_codes'], {'setup_0': 0, 'setup_1': 1})
        self.assertFalse(receipt['tests_started'])
        self.assertFalse(receipt['test_patch_applied'])
        self.assertNotIn('PASSED test_answer', receipt['log'])

    def test_candidate_syntax_proof_uses_baseline_and_actual_target_interpreter(self):
        invalid = self.diff('answer.py', 'def broken():\ndef unindented():\n    pass\n')
        receipt = self.backend().run_correctness(self.request(candidate=invalid))
        self.assertEqual(receipt['execution'], 'complete')
        evidence, = receipt['candidate_syntax_evidence']
        self.assertEqual(evidence['path'], 'answer.py')
        self.assertTrue(evidence['baseline_compiles'])
        self.assertEqual(evidence['error_type'], 'IndentationError')
        self.assertEqual(evidence['lineno'], 2)
        self.assertIn('IndentationError', receipt['log'])

    def test_new_python_files_are_compiled_and_audited_without_baseline_blame(self):
        new_file = ('diff --git a/extra.py b/extra.py\nnew file mode 100644\n'
                    '--- /dev/null\n+++ b/extra.py\n@@ -0,0 +1 @@\n+def invalid():\n')
        receipt = self.backend().run_correctness(self.request(candidate=self.candidate + new_file))
        checks = {check['path']: check for check in receipt['candidate_syntax_checks']}
        self.assertTrue(checks['answer.py']['patched_compiles'])
        self.assertFalse(checks['extra.py']['baseline_exists'])
        self.assertFalse(checks['extra.py']['patched_compiles'])
        self.assertEqual(receipt['candidate_syntax_evidence'], [])
        self.assertEqual(receipt['stage_exit_codes']['tests'], 0)

    def test_legal_syntax_warning_does_not_corrupt_probe_json(self):
        candidate = self.diff('answer.py', 'VALUE = 1 if 1 is 1 else 0\n')
        receipt = self.backend().run_correctness(self.request(candidate=candidate))
        self.assertEqual(receipt['execution'], 'complete')
        self.assertEqual(receipt['candidate_syntax_evidence'], [])
        self.assertTrue(receipt['candidate_syntax_checks'][0]['patched_compiles'])

    def test_preexisting_invalid_fixture_does_not_establish_candidate_blame(self):
        (self.repo / 'parser_fixture.py').write_text('def invalid():\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Worker fixture', '-c', 'user.email=worker@example.invalid', 'commit', '-qm', 'invalid fixture')
        self.row['base_commit'] = self.git('rev-parse', 'HEAD').strip()
        candidate = self.diff('parser_fixture.py', 'def still_invalid():\n')
        receipt = self.backend().run_correctness(self.request(candidate=candidate))
        self.assertEqual(receipt['candidate_syntax_evidence'], [])

    def test_request_tampering_is_rejected_before_execution(self):
        for field, value in [('eval_script', 'echo bypass'), ('timeout_seconds', 1), ('instance_id', 'other')]:
            request = copy.deepcopy(self.request())
            request[field] = value
            with self.subTest(field=field), self.assertRaises(CorrectnessRuntimeError):
                self.backend().run_correctness(request)
        self.assertEqual((self.repo / 'answer.py').read_text(), 'VALUE = 0\n')

    def test_exact_content_accepts_packaging_mode_commit_and_rejects_dirty_content(self):
        (self.repo / 'answer.py').chmod(0o755)
        self.git('add', '.')
        self.git('-c', 'user.name=Worker fixture', '-c', 'user.email=worker@example.invalid', 'commit', '-qm', 'SWE-bench')
        identity = worker.inspect_testbed(self.repo, self.base, 65536)
        self.assertEqual(identity['regular_executable_mode_differences'], 1)
        self.assertNotEqual(identity['observed_head'], self.base)
        (self.repo / 'answer.py').write_text('VALUE = 9\n')
        with self.assertRaises(CorrectnessRuntimeError):
            worker.inspect_testbed(self.repo, self.base, 65536)

    def test_gold_patch_failure_does_not_erase_continuing_test_execution(self):
        self.commands[-5] = "git apply -v - <<'EOF_TEST'\nnot a patch\nEOF_TEST"
        receipt = self.backend().run_correctness(self.request())
        self.assertEqual(receipt['execution'], 'incomplete_test_stages')
        self.assertFalse(receipt['test_patch_applied'])
        self.assertTrue(receipt['tests_started'])
        self.assertTrue(receipt['tests_completed'])
        self.assertIn('unpatched tests', receipt['log'])

    def test_test_timeout_keeps_stage_and_original_timeout_without_attribution(self):
        self.commands[-3] = 'sleep 10'
        original_run = worker._run
        seen = []
        def short_test_clock(argv, **kwargs):
            if kwargs.get('script'):
                seen.append(kwargs['timeout'])
                kwargs['timeout'] = 0.05
            return original_run(argv, **kwargs)
        with patch.object(worker, '_run', short_test_clock):
            receipt = self.backend().run_correctness(self.request())
        self.assertEqual(len(seen), 1)
        self.assertGreater(seen[0], 0)
        self.assertLess(seen[0], 1200, 'patch/setup time consumes the original case deadline')
        self.assertEqual(receipt['execution'], 'timeout')
        self.assertEqual(receipt['stopped_stage'], 'eval_script')
        self.assertTrue(receipt['test_patch_applied'])
        self.assertFalse(receipt['tests_completed'])
        self.assertNotIn('platform_error', receipt)

    def test_output_flood_is_bounded_and_not_complete(self):
        self.commands[-3] = "python3 -c 'print(\"x\" * 100000)'"
        receipt = self.backend(limit=4096).run_correctness(self.request())
        self.assertEqual(receipt['execution'], 'output_limit')
        total = sum(r['output_bytes_retained'] + r['stage_bytes_retained'] for r in receipt['patch_attempts'])
        total += receipt['syntax_probe_execution']['output_bytes_retained'] + receipt['syntax_probe_execution']['stage_bytes_retained']
        total += receipt['eval_execution']['output_bytes_retained'] + receipt['eval_execution']['stage_bytes_retained']
        self.assertLessEqual(total, 4096)
        self.assertFalse(receipt['tests_completed'])


class OriginalScriptTests(unittest.TestCase):
    def test_all48_wrappers_preserve_every_original_command_and_shell_flags(self):
        for row in load_swe_cases('eval'):
            request = build_test_request(row['instance_id'], '')
            wrapper = worker.instrument_script(request, row)(3)
            original = '\n'.join(line for line in wrapper.splitlines() if not line.startswith("printf 'test") and not line.startswith("printf 'tests=") and not line.startswith('frontier_setup_status=')) + '\n'
            self.assertEqual(original, request['eval_script'])
            self.assertNotIn('set -e', wrapper)
            self.assertNotEqual(sha(wrapper), request['eval_script_sha256'])


if __name__ == '__main__':
    unittest.main()
