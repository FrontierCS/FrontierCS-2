"""Original harness parser/script tests; synthetic receipts are not real tests."""
import copy
import json
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "verifier"))

from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT, MAP_REPO_VERSION_TO_SPECS
from swebench.harness.test_spec.python import make_eval_script_list_py

from cpu_swe_accuracy import TrustedCorrectnessError, CorrectnessRuntimeError, build_test_request, load_swe_cases, parse_test_receipt, summarize_swe_correctness, verify_swe_case


class FixtureBackend:
    def __init__(self, log):
        self.log = log
        self.request = None

    def run_correctness(self, request):
        self.request = request
        return {'instance_id': request['instance_id'], 'request_sha256': request['request_sha256'],
                'execution': 'complete', 'candidate_patch_applied': True, 'test_patch_applied': True,
                'tests_started': True, 'tests_completed': True, 'log': self.log}


class SWEAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.row = next(row for row in load_swe_cases('train') if row['repo'] == 'pytest-dev/pytest')
        self.request = build_test_request(self.row['instance_id'], '')
        self.tests = self.row['FAIL_TO_PASS'] + self.row['PASS_TO_PASS']
        self.lines = ['PASSED ' + test for test in self.tests]

    def log(self, lines=None):
        return START_TEST_OUTPUT + '\n' + '\n'.join(self.lines if lines is None else lines) + '\n' + END_TEST_OUTPUT + '\n'

    def receipt(self, lines=None):
        return FixtureBackend(self.log(lines)).run_correctness(self.request)

    def test_fixed_all48_scripts_use_original_test_patch_and_commands(self):
        self.assertEqual(len(load_swe_cases('train')), 5)
        self.assertEqual(len(load_swe_cases('eval')), 48)
        for row in load_swe_cases('eval'):
            request = build_test_request(row['instance_id'], '')
            commands = make_eval_script_list_py(row, MAP_REPO_VERSION_TO_SPECS[row['repo']][row['version']], 'testbed', '/testbed', row['base_commit'], row['test_patch'])
            self.assertEqual(request['eval_script'], '\n'.join(['#!/bin/bash', 'set -uxo pipefail'] + commands) + '\n')
            self.assertIn(row['test_patch'], request['eval_script'])
            self.assertEqual(request['timeout_seconds'], 1200)

    def test_missing_backend_does_not_fall_back_to_patch_validity(self):
        with self.assertRaises(CorrectnessRuntimeError):
            verify_swe_case(self.row['instance_id'], 'diff --git a/x b/x\n', None)

    def test_original_all_pass_log_produces_instance_bound_report(self):
        backend = FixtureBackend(self.log())
        result = verify_swe_case(self.row['instance_id'], '', backend)
        self.assertTrue(result['result']['correct'])
        self.assertEqual(result['result']['instance_id'], self.row['instance_id'])
        self.assertEqual(result['result']['missing_test_statuses'], [])
        self.assertIn(self.row['test_patch'], backend.request['eval_script'])
        self.assertNotIn('score', result)

    def test_original_failed_or_missing_test_is_not_a_smaller_denominator(self):
        for lines in ([self.lines[0].replace('PASSED ', 'FAILED ', 1)] + self.lines[1:], self.lines[1:]):
            with self.subTest():
                result = parse_test_receipt(self.request, self.receipt(lines))
                self.assertFalse(result['correct'])
                group = result['report']['tests_status']['FAIL_TO_PASS']
                self.assertEqual(len(group['success']) + len(group['failure']), len(self.row['FAIL_TO_PASS']))

    def test_real_recorded_quiet_output_does_not_claim_all_tests_passed(self):
        raw = (Path(__file__).parent / 'tests/fixtures/swe_pytest_quiet_observed.txt').read_text()
        self.assertIn('1 passed', raw)
        for log in (raw, START_TEST_OUTPUT + '\n' + raw + END_TEST_OUTPUT):
            receipt = FixtureBackend(log).run_correctness(self.request)
            with self.assertRaises(CorrectnessRuntimeError):
                parse_test_receipt(self.request, receipt)

    def test_only_test_traceback_matching_baseline_compile_proof_is_candidate_incorrect(self):
        receipt = self.receipt(['  File "/testbed/src/_pytest/assertion/util.py", line 257',
                                '    def broken():', '    ^',
                                'IndentationError: expected an indented block'])
        proof = {'path': 'src/_pytest/assertion/util.py', 'lineno': 257, 'baseline_compiles': True,
                 'baseline_sha256': 'a' * 64, 'patched_sha256': 'b' * 64,
                 'error_type': 'IndentationError', 'interpreter': '3.9.19'}
        receipt['candidate_syntax_evidence'] = [proof]
        receipt['stage_exit_codes'] = {'setup_0': 0, 'test_reset': 0, 'test_patch': 0,
                                       'tests': 1, 'test_cleanup': 0}
        result = parse_test_receipt(self.request, receipt)
        self.assertFalse(result['correct'])
        self.assertEqual(result['reason'], 'candidate_syntax_error')
        for field, value in [('path', 'unrelated.py'), ('lineno', 258), ('baseline_compiles', False),
                             ('error_type', 'SyntaxError'), ('baseline_sha256', '')]:
            changed = copy.deepcopy(receipt)
            changed['candidate_syntax_evidence'][0][field] = value
            with self.subTest(field=field), self.assertRaises(CorrectnessRuntimeError):
                parse_test_receipt(self.request, changed)
        for status in ('setup_failed', 'timeout', 'incomplete_test_stages'):
            changed = copy.deepcopy(receipt)
            changed['execution'] = status
            with self.subTest(status=status), self.assertRaises(CorrectnessRuntimeError):
                parse_test_receipt(self.request, changed)
        dependency = copy.deepcopy(receipt)
        dependency['log'] = self.log(['ModuleNotFoundError: No module named setuptools'])
        with self.assertRaises(CorrectnessRuntimeError):
            parse_test_receipt(self.request, dependency)
        before_tests = copy.deepcopy(receipt)
        before_tests['log'] = receipt['log'].replace(START_TEST_OUTPUT, '').replace(END_TEST_OUTPUT, '') + self.log(['ModuleNotFoundError: unrelated_dependency'])
        with self.assertRaises(CorrectnessRuntimeError):
            parse_test_receipt(self.request, before_tests)

    def test_wrong_instance_request_or_generic_count_is_rejected(self):
        bad_instance = self.receipt()
        bad_instance['instance_id'] = 'different-case'
        bad_request = self.receipt()
        bad_request['request_sha256'] = 'different-request'
        for receipt in (bad_instance, bad_request, {'resolved_instances': 50}):
            with self.subTest(), self.assertRaises(CorrectnessRuntimeError):
                parse_test_receipt(self.request, receipt)
        request = copy.deepcopy(self.request)
        request['eval_script'] = 'echo bypass'
        with self.assertRaises(CorrectnessRuntimeError):
            parse_test_receipt(request, self.receipt())

    def test_incomplete_platform_and_candidate_patch_failures_stay_distinct(self):
        for status in ('platform_error', 'timeout', 'unclassified_error'):
            receipt = self.receipt()
            receipt['execution'] = status
            with self.subTest(), self.assertRaises(CorrectnessRuntimeError):
                parse_test_receipt(self.request, receipt)
        receipt = self.receipt()
        receipt['tests_completed'] = False
        with self.assertRaises(CorrectnessRuntimeError):
            parse_test_receipt(self.request, receipt)
        rejected = {'instance_id': self.row['instance_id'], 'request_sha256': self.request['request_sha256'], 'execution': 'candidate_patch_rejected', 'candidate_patch_applied': False, 'patch_apply_log': 'Explicit fixture: candidate patch does not apply.'}
        self.assertFalse(parse_test_receipt(self.request, rejected)['correct'])

    def test_skipped_gold_test_remains_explicit_unresolved_runtime_error(self):
        lines = [self.lines[0].replace('PASSED ', 'SKIPPED ', 1)] + self.lines[1:]
        with self.assertRaises(CorrectnessRuntimeError):
            parse_test_receipt(self.request, self.receipt(lines))

    def test_summary_requires_every_original_instance_once(self):
        results = [{'instance_id': row['instance_id'], 'correct': True} for row in load_swe_cases('train')]
        self.assertEqual(summarize_swe_correctness('train', results)['accuracy'], 1)
        for records in (results[:-1], results + results[:1]):
            with self.subTest(), self.assertRaises(CorrectnessRuntimeError):
                summarize_swe_correctness('train', records)


    def test_cpu_requests_remain_byte_identical_to_published_sidecar_contract(self):
        from serving_eval import accuracy as sidecar
        import cpu_swe_accuracy as cpu
        for row in load_swe_cases('train'):
            for patch in ('', 'diff --git a/model.py b/model.py\n', 'x' * 100000):
                identity = {'run_id': 'parity', 'instance_id': row['instance_id']}
                for variant in ('vanilla', 'reference', 'candidate'):
                    self.assertEqual(cpu.build_correctness_request(row['instance_id'], patch,
                        identity=identity, variant=variant), sidecar.build_correctness_request(
                        row['instance_id'], patch, identity=identity, variant=variant))

    def test_model_log_grader_exceptions_are_case_failures(self):
        from unittest.mock import patch
        import cpu_swe_accuracy as accuracy
        grading, _, _ = accuracy._harness()
        for operation in ('get_logs_eval', 'get_eval_report'):
            for error in (ValueError('CPU grader bug'), TypeError('CPU grader type defect'),
                          OSError('CPU file failure'), RuntimeError('CPU runtime failure')):
                with self.subTest(operation=operation, error=type(error).__name__):
                    with patch.object(grading, operation, side_effect=error):
                        with self.assertRaises(CorrectnessRuntimeError):
                            parse_test_receipt(self.request, self.receipt())

    def test_malformed_model_log_grader_return_is_case_failure(self):
        from unittest.mock import patch
        import cpu_swe_accuracy as accuracy
        grading, _, _ = accuracy._harness()
        for result in (None, {}, {self.row['instance_id']: {'resolved': None}}):
            with self.subTest(result=result):
                with patch.object(grading, 'get_eval_report', return_value=result):
                    with self.assertRaises(CorrectnessRuntimeError):
                        parse_test_receipt(self.request, self.receipt())
        for value in (None, 1, [1]):
            report = {'resolved': False, 'patch_successfully_applied': True,
                'tests_status': {field: {'success': [], 'failure': []}
                    for field in ('FAIL_TO_PASS', 'PASS_TO_PASS')}}
            report['tests_status']['FAIL_TO_PASS']['success'] = value
            with self.subTest(status_list=value):
                with patch.object(grading, 'get_eval_report', return_value={self.row['instance_id']: report}):
                    with self.assertRaises(CorrectnessRuntimeError):
                        parse_test_receipt(self.request, self.receipt())
        for result in (None, (None, False), ({}, None)):
            with self.subTest(result=result):
                with patch.object(grading, 'get_logs_eval', return_value=result):
                    with self.assertRaises(CorrectnessRuntimeError):
                        parse_test_receipt(self.request, self.receipt())
        with patch.object(accuracy, '_load_swe_cases', side_effect=RecursionError('corrupt fixed JSON')):
            with self.assertRaises(TrustedCorrectnessError):
                accuracy.load_swe_cases('eval')

    def test_corrupt_cpu_fixed_data_is_not_model_output(self):
        import tempfile
        from unittest.mock import patch
        import cpu_swe_accuracy as accuracy
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'instances.json').write_text('{}')
            (root / 'provenance.json').write_text('{"native_instances_sha256": "corrupt"}')
            with patch.object(accuracy, 'DATA_DIR', root):
                with self.assertRaises(TrustedCorrectnessError):
                    parse_test_receipt(self.request, self.receipt())


if __name__ == '__main__':
    unittest.main()
