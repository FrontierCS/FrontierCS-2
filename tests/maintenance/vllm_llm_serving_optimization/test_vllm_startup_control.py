"""Offline attribution and budget tests; no serving or provider resources."""
from vllm_task import TASK_ROOT  # noqa: E402
import deployment_test_guard  # noqa: F401

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ('swebench', 'cryptography')):
    raise unittest.SkipTest('startup-control tests require the task SWE/cryptography dependencies')

TASK = TASK_ROOT
sys.path.insert(0, str(TASK / 'judge'))
from serving_eval import accuracy
from serving_eval import swe_correctness_service as service
from serving_eval import swe_test_worker as worker
from serving_eval import testbed_reset as reset_module
from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT


class StartupControlTests(unittest.TestCase):
    def setUp(self):
        self.row = next(row for row in accuracy.load_swe_cases('train') if row['repo'] == 'pytest-dev/pytest')
        self.request = accuracy.build_test_request(self.row['instance_id'], '# synthetic prediction')
        self.baseline_request = accuracy.build_test_request(self.row['instance_id'], '')
        self.base = self.row['base_commit']
        self.identity = {'base_commit': self.base, 'observed_head': self.base,
                         'content_matches_fixed_base': True, 'fresh_worktree': True}
        self.receipt = self.complete(self.request, [
            'Traceback (most recent call last):',
            '  File "/testbed/src/_pytest/assertion/util.py", line 257, in broken',
            '    return method()', "TypeError: 'classmethod' object is not callable"])
        self.receipt['stage_exit_codes'] = {'setup_0': 0, 'test_reset': 0, 'test_patch': 0,
                                             'tests': 1, 'test_cleanup': 0}
        self.receipt['candidate_syntax_checks'] = [
            {'path': 'src/_pytest/assertion/util.py', 'baseline_exists': True,
             'baseline_compiles': True, 'patched_compiles': True,
             'baseline_sha256': 'a' * 64, 'patched_sha256': 'b' * 64}]
        tests = self.row['FAIL_TO_PASS'] + self.row['PASS_TO_PASS']
        self.baseline = self.complete(self.baseline_request, ['PASSED ' + test for test in tests])
        self.baseline['fixed_base_control'] = True

    def complete(self, request, lines):
        return {'instance_id': request['instance_id'], 'request_sha256': request['request_sha256'],
                'execution': 'complete', 'candidate_patch_applied': True, 'test_patch_applied': True,
                'tests_started': True, 'tests_completed': True,
                'stage_exit_codes': {'setup_0': 0, 'test_reset': 0, 'test_patch': 0,
                                     'tests': 0, 'test_cleanup': 0},
                'testbed': copy.deepcopy(self.identity),
                'log': START_TEST_OUTPUT + '\n' + '\n'.join(lines) + '\n' + END_TEST_OUTPUT + '\n'}

    def controlled(self, variant='candidate'):
        receipt = copy.deepcopy(self.receipt)
        receipt['startup_control'] = {
            'schema': 'frontier.vllm-fixed-base-startup-control/v1',
            'request_sha256': self.request['request_sha256'], 'variant': variant, 'state': 'complete',
            'elapsed_seconds': 2, 'reset': {'head': self.base}, 'receipt': copy.deepcopy(self.baseline)}
        return receipt

    def test_each_variant_is_incorrect_only_with_positive_control(self):
        for variant in service.VARIANT_ORDER:
            with self.subTest(variant=variant):
                result = accuracy.parse_test_receipt(self.request, self.controlled(variant))
                self.assertFalse(result['correct'])
                self.assertEqual(result['reason'], 'candidate_test_startup_error')
                self.assertIn('fixed_base_control_sha256', result)
        baseline = accuracy.parse_test_receipt(self.baseline_request, self.baseline)
        for field in ('FAIL_TO_PASS', 'PASS_TO_PASS'):
            group = baseline['report']['tests_status'][field]
            self.assertEqual(len(group['success']) + len(group['failure']), len(self.row[field]))

    def test_name_error_requires_the_same_strict_positive_control(self):
        for variant in service.VARIANT_ORDER:
            receipt = self.controlled(variant)
            receipt['log'] = receipt['log'].replace("TypeError: 'classmethod' object is not callable",
                                                   "NameError: name '_multi_decorate' is not defined")
            parsed = accuracy.parse_test_receipt(self.request, receipt)
            self.assertFalse(parsed['correct'])
            self.assertEqual(parsed['startup_evidence']['error_type'], 'NameError')
            for mutation in ('missing_control', 'failing_control', 'unchanged_file', 'unrelated_frame'):
                broken = copy.deepcopy(receipt)
                if mutation == 'missing_control':
                    broken.pop('startup_control')
                elif mutation == 'failing_control':
                    broken['startup_control']['receipt']['log'] = receipt['log']
                elif mutation == 'unchanged_file':
                    broken['candidate_syntax_checks'][0]['patched_sha256'] = 'a' * 64
                else:
                    broken['log'] = broken['log'].replace('src/_pytest/assertion/util.py', 'environment.py')
                with self.subTest(variant=variant, mutation=mutation), self.assertRaises(accuracy.CorrectnessRuntimeError):
                    accuracy.parse_test_receipt(self.request, broken)

    def test_missing_control_unknown_error_and_provider_failure_stay_runtime_errors(self):
        receipts = [self.receipt]
        for key, value in [('execution', 'timeout'), ('execution', 'setup_failed')]:
            changed = self.controlled(); changed[key] = value; receipts.append(changed)
        for error in ['ModuleNotFoundError: dependency missing', 'MemoryError: exhausted',
                      'RuntimeError: unknown worker failure']:
            changed = self.controlled()
            changed['log'] = changed['log'].replace("TypeError: 'classmethod' object is not callable", error)
            receipts.append(changed)
        changed = self.controlled(); changed['stage_exit_codes']['setup_0'] = 1; receipts.append(changed)
        changed = self.controlled(); changed['candidate_syntax_checks'][0]['path'] = 'unrelated.py'; receipts.append(changed)
        for receipt in receipts:
            with self.subTest(receipt=receipt.get('execution')), self.assertRaises(accuracy.CorrectnessRuntimeError):
                accuracy.parse_test_receipt(self.request, receipt)

    def test_incomplete_or_unbound_control_never_converts_to_incorrect(self):
        mutations = [
            lambda c: c.update(state='unavailable'),
            lambda c: c.update(request_sha256='another-request'),
            lambda c: c.update(variant='unknown'),
            lambda c: c.update(elapsed_seconds=1201),
            lambda c: c.update(elapsed_seconds=float('nan')),
            lambda c: c.update(reset={'head': 'wrong'}),
            lambda c: c['receipt'].update(execution='timeout'),
            lambda c: c['receipt'].update(execution='setup_failed'),
            lambda c: c['receipt']['stage_exit_codes'].update(setup_0=1),
            lambda c: c['receipt']['stage_exit_codes'].update(test_reset=1),
            lambda c: c['receipt']['stage_exit_codes'].update(test_cleanup=1),
            lambda c: c['receipt'].update(startup_control={}),
            lambda c: c['receipt']['testbed'].update(content_matches_fixed_base=False),
            lambda c: c['receipt']['testbed'].update(base_commit='wrong'),
            lambda c: c['receipt'].update(log=START_TEST_OUTPUT + '\nTypeError: same failure\n' + END_TEST_OUTPUT),
            lambda c: c['receipt'].update(log=START_TEST_OUTPUT + '\nPASSED unrelated_test\n' + END_TEST_OUTPUT),
        ]
        for mutate in mutations:
            receipt = self.controlled(); mutate(receipt['startup_control'])
            with self.subTest(mutation=mutate), self.assertRaises(accuracy.CorrectnessRuntimeError):
                accuracy.parse_test_receipt(self.request, receipt)

    def test_control_requires_usable_dispositions_but_need_not_pass(self):
        receipt = self.controlled()
        test_name = self.row['FAIL_TO_PASS'][0]
        receipt['startup_control']['receipt']['log'] = self.baseline['log'].replace(
            'PASSED ' + test_name, 'FAILED ' + test_name)
        self.assertFalse(accuracy.parse_test_receipt(self.request, receipt)['correct'])

    def fake_service(self, *, reset_error=False):
        instance = object.__new__(service.CorrectnessService)
        instance.identity = {'instance_id': self.row['instance_id']}
        instance.testbed = Path('/unused-testbed')
        instance.max_output_bytes = 65536
        instance.process_cleanup = lambda: None
        instance.reset_calls = []
        instance.worker_calls = []
        test = self

        class Reset:
            def reset(self, *, deadline):
                instance.reset_calls.append(deadline)
                if reset_error:
                    raise reset_module.ResetError('fixture reset failure')
                return {'head': test.base}

        class Worker:
            def __init__(self, **kwargs):
                instance.worker_calls.append(kwargs)

            def run_fixed_base_control(self, request, *, deadline):
                instance.worker_calls.append({'deadline': deadline, 'request': request})
                return copy.deepcopy(test.baseline)

        instance.reset = Reset()
        instance.worker_factory = Worker
        return instance

    def test_service_shares_original_deadline_and_byte_allocation_for_all_variants(self):
        for variant in service.VARIANT_ORDER:
            instance = self.fake_service()
            receipt = copy.deepcopy(self.receipt)
            receipt['eval_execution'] = {'output_bytes_retained': 1234, 'stage_bytes_retained': 100}
            with patch.object(service.time, 'monotonic', return_value=100):
                instance._startup_control(self.request, receipt, variant, started=99)
            self.assertEqual(instance.reset_calls, [1299])
            self.assertEqual(instance.worker_calls[0]['max_output_bytes'], 65536 - 1334)
            self.assertEqual(instance.worker_calls[1]['deadline'], 1299)
            self.assertEqual(instance.worker_calls[1]['request']['candidate_patch'], '')
            self.assertEqual(receipt['startup_control']['state'], 'complete')

    def test_exhausted_deadline_output_and_reset_failure_never_create_valid_control(self):
        instance = self.fake_service(); receipt = copy.deepcopy(self.receipt)
        with patch.object(service.time, 'monotonic', return_value=1300):
            instance._startup_control(self.request, receipt, 'candidate', started=99)
        self.assertEqual(instance.reset_calls, [])
        self.assertEqual(instance.worker_calls, [])
        self.assertEqual(receipt['startup_control']['state'], 'unavailable')
        instance = self.fake_service(); receipt = copy.deepcopy(self.receipt)
        receipt['eval_execution'] = {'output_bytes_retained': 65536}
        instance._startup_control(self.request, receipt, 'candidate', started=time.monotonic())
        self.assertEqual(instance.worker_calls, [])
        instance = self.fake_service(reset_error=True); receipt = copy.deepcopy(self.receipt)
        instance._startup_control(self.request, receipt, 'candidate', started=time.monotonic())
        self.assertEqual(receipt['startup_control']['error_type'], 'ResetError')
        self.assertEqual(instance.worker_calls, [])

    def test_worker_clamps_each_subprocess_and_rejects_deadline_exhaustion(self):
        instance = worker.PreparedSWEWorker(instance_id=self.row['instance_id'], max_output_bytes=65536)
        instance._control_deadline = 100
        calls = []
        with patch.object(worker.time, 'monotonic', return_value=99.5):
            instance._bounded(lambda **kw: calls.append(kw), timeout=1200)
        self.assertEqual(calls, [{'timeout': 0.5}])
        with patch.object(worker.time, 'monotonic', return_value=100), self.assertRaises(accuracy.CorrectnessRuntimeError):
            instance._bounded(lambda **kw: calls.append(kw), timeout=1200)
        self.assertEqual(len(calls), 1)

    def test_control_deadline_exhaustion_during_reset_or_worker_stays_unavailable(self):
        for during in ('reset', 'worker'):
            instance = self.fake_service()
            receipt = copy.deepcopy(self.receipt)
            clock = [100]

            if during == 'reset':
                def expired_reset(*, deadline):
                    clock[0] = deadline + 1
                    raise reset_module.ResetError('deadline exhausted during reset')
                instance.reset.reset = expired_reset
            else:
                original_factory = instance.worker_factory

                def expired_worker(**kwargs):
                    result = original_factory(**kwargs)
                    original_run = result.run_fixed_base_control

                    def run(request, *, deadline):
                        value = original_run(request, deadline=deadline)
                        clock[0] = deadline + 1
                        return value
                    result.run_fixed_base_control = run
                    return result
                instance.worker_factory = expired_worker
            with self.subTest(during=during), patch.object(service.time, 'monotonic', side_effect=lambda: clock[0]):
                instance._startup_control(self.request, receipt, 'candidate', started=99)
            self.assertEqual(receipt['startup_control']['state'], 'unavailable')
            self.assertNotIn('receipt', receipt['startup_control'])

    def test_serialized_response_overflow_discards_control(self):
        instance = self.fake_service()
        receipt = copy.deepcopy(self.receipt)
        receipt['unmeasured_metadata'] = 'x' * instance.max_output_bytes
        with patch.object(service.time, 'monotonic', return_value=100):
            instance._startup_control(self.request, receipt, 'candidate', started=99)
        self.assertEqual(receipt['startup_control']['state'], 'unavailable')
        self.assertNotIn('receipt', receipt['startup_control'])

    def test_internal_worker_entrypoint_requires_unused_empty_prediction(self):
        instance = worker.PreparedSWEWorker(instance_id=self.row['instance_id'], max_output_bytes=65536)
        with self.assertRaises(accuracy.CorrectnessRuntimeError):
            instance.run_fixed_base_control(self.request, deadline=time.monotonic() + 1)
        instance.used = True
        with self.assertRaises(accuracy.CorrectnessRuntimeError):
            instance.run_fixed_base_control(self.baseline_request, deadline=time.monotonic() + 1)
        instance.used = False
        with patch.object(instance, 'run_correctness', return_value={'fixture': True}) as run:
            result = instance.run_fixed_base_control(self.baseline_request, deadline=time.monotonic() + 1)
        self.assertEqual(result, {'fixture': True})
        run.assert_called_once_with(self.baseline_request)
        self.assertTrue(instance._fixed_base_control)


class ResetDeadlineTests(unittest.TestCase):
    def test_actual_reset_uses_remaining_deadline_and_expired_reset_does_not_run_git(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(['git', 'init', '-q', directory], check=True)
            (repo / 'value').write_text('base')
            subprocess.run(['git', '-C', directory, 'add', '.'], check=True)
            subprocess.run(['git', '-C', directory, '-c', 'user.name=Fixture', '-c',
                            'user.email=fixture@example.invalid', 'commit', '-qm', 'base'], check=True)
            calls = []

            def execute(*args, **kwargs):
                calls.append(kwargs['timeout'])
                return subprocess.run(*args, **kwargs)

            reset = reset_module.TestbedReset(repo, run=execute)
            reset.snapshot(); calls.clear()
            (repo / 'value').write_text('changed')
            reset.reset(deadline=time.monotonic() + 5)
            self.assertEqual((repo / 'value').read_text(), 'base')
            self.assertTrue(calls and all(0 < value <= 5 for value in calls))
            calls.clear()
            with self.assertRaises(reset_module.ResetError):
                reset.reset(deadline=time.monotonic() - 1)
            self.assertEqual(calls, [])
            self.assertIsNone(reset._deadline)


if __name__ == '__main__':
    unittest.main()
