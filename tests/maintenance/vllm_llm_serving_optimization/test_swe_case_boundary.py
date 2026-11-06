"""Fault injection at the real authenticated SWE HTTP and worker boundaries."""
import sys
import time
import unittest
from unittest import mock

import test_swe_correctness_service as correctness_fixture
import test_swe_tool_service as tool_fixture
from serving_eval import swe_correctness_service, swe_test_worker
from serving_eval.testbed_reset import ResetError, TestbedReset
from serving_eval.swe_case_failure import SCHEMA, failure, validate_failure


class UnanticipatedFailure(BaseException):
    pass


class SWECaseBoundaryTests(unittest.TestCase):
    def tool(self):
        fixture = tool_fixture.ToolServiceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def correctness(self):
        fixture = correctness_fixture.CorrectnessServiceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.service.prepare()
        return fixture

    def test_closed_failure_schema_rejects_forged_infra_and_untyped_values(self):
        value = failure('correctness', UnanticipatedFailure())
        self.assertEqual(value['schema'], SCHEMA)
        self.assertEqual(value['disposition'], 'incorrect')
        for field, bad in [('stage', []), ('stage', 'platform'), ('reason', 'infra'),
                           ('disposition', 'retry'), ('error_type', 'secret\nvalue')]:
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                validate_failure({**value, field: bad})
        with self.assertRaises(ValueError):
            validate_failure({**value, 'platform_signal': 'gpu_backend_unavailable'})

    def test_unknown_worker_baseexception_becomes_case_failure_and_releases_lock(self):
        fixture = self.correctness()
        def unknown(**kwargs):
            raise UnanticipatedFailure()
        fixture.service.worker_factory = unknown
        status, result = fixture.request('POST', '/correctness', fixture.body('vanilla'))
        self.assertEqual(status, 422)
        self.assertEqual(validate_failure(result['case_failure'])['disposition'], 'incorrect')
        self.assertEqual(result['case_failure']['error_type'], 'UnanticipatedFailure')
        self.assertTrue(fixture.service.command_lock.acquire(blocking=False))
        fixture.service.command_lock.release()

    def test_cleanup_failure_cannot_erase_primary_failure(self):
        fixture = self.correctness()
        fixture.service.process_cleanup = lambda: (_ for _ in ()).throw(UnanticipatedFailure())
        status, result = fixture.request('POST', '/correctness', fixture.body('vanilla', patch='reject'))
        self.assertEqual(status, 422)
        self.assertEqual(result['case_failure']['error_type'], 'CorrectnessRuntimeError')
        self.assertEqual(result['cleanup_failure']['stage'], 'cleanup')

    def test_cleanup_failure_cannot_publish_a_success(self):
        fixture = self.correctness()
        fixture.service.process_cleanup = lambda: (_ for _ in ()).throw(UnanticipatedFailure())
        status, result = fixture.request('POST', '/correctness', fixture.body('vanilla'))
        self.assertEqual(status, 422)
        self.assertEqual(result['case_failure']['stage'], 'cleanup')
        self.assertEqual(fixture.service.completed, [])

    def test_nonserializable_receipt_is_a_bounded_case_failure(self):
        fixture = self.correctness()
        class Worker:
            def __init__(self, **kwargs):
                pass
            def run_correctness(self, request):
                return {'invalid': object()}
        fixture.service.worker_factory = Worker
        status, result = fixture.request('POST', '/correctness', fixture.body('vanilla'))
        self.assertEqual(status, 422)
        self.assertEqual(result['case_failure']['error_type'], 'TypeError')

    def test_all_correctness_stages_share_one_absolute_deadline(self):
        worker = object.__new__(swe_test_worker.PreparedSWEWorker)
        worker._request_deadline = time.monotonic() + 0.01
        def operation(*args, **kwargs):
            time.sleep(0.02)
        with self.assertRaises(swe_correctness_service.CorrectnessRuntimeError):
            worker._bounded(operation, timeout=1200)
        with self.assertRaises(swe_correctness_service.CorrectnessRuntimeError):
            worker._bounded(lambda **kwargs: self.fail('deadline restarted'), timeout=1200)

    def test_unknown_tool_and_patch_errors_are_case_failures_over_real_http(self):
        fixture = self.tool()
        def unknown(*args, **kwargs):
            raise UnanticipatedFailure()
        fixture.service.shell_runner = unknown
        fixture.service.patch_reader = unknown
        for route, body, stage in (
                ('/execute', fixture.body(), 'execute'),
                ('/patch', {'identity': fixture.client.identity, 'timeout_seconds': 1}, 'patch')):
            with self.subTest(route=route):
                status, result = fixture.request(route, body)
                self.assertEqual(status, 422)
                self.assertEqual(validate_failure(result['case_failure'])['stage'], stage)
        self.assertTrue(fixture.service.command_lock.acquire(blocking=False))
        fixture.service.command_lock.release()

    def test_unknown_dispatch_and_invalid_serialization_are_bounded_over_real_http(self):
        fixture = self.tool()
        for result in ({'output': object()}, {'output': float('nan')},
                       {'output': 'x' * (fixture.service.max_request_bytes +
                                         fixture.service.max_output_bytes + 8192)}):
            with self.subTest(kind=type(result['output']).__name__):
                with mock.patch.object(fixture.service, 'handle', return_value=(200, result)):
                    status, response = fixture.request('/execute', fixture.body())
                self.assertEqual(status, 422)
                self.assertEqual(validate_failure(response['case_failure'])['stage'], 'serialize')
        with mock.patch.object(fixture.service, 'handle', side_effect=UnanticipatedFailure()):
            status, response = fixture.request('/execute', fixture.body())
        self.assertEqual(status, 422)
        self.assertEqual(response['case_failure']['error_type'], 'UnanticipatedFailure')

    def test_worker_closing_output_pipes_does_not_escape_its_watchdog(self):
        started = time.monotonic()
        result = swe_test_worker._run(
            [sys.executable, '-c', 'import os,time; os.closerange(1,256); time.sleep(30)'],
            cwd='/', timeout=0.15, output_limit=4096)
        self.assertEqual(result.stop_reason, 'timeout')
        self.assertLess(time.monotonic() - started, 2)

    def test_reset_uses_one_deadline_and_bounded_git_output(self):
        fixture = self.tool()
        reset = TestbedReset(fixture.root, output_limit=4096)
        reset.snapshot()
        with self.assertRaises(ResetError):
            reset.reset(deadline=time.monotonic() - 1)
        self.assertIsNone(reset._deadline)
        with mock.patch.object(swe_test_worker, '_run', return_value=
                swe_test_worker.Execution(0, b'bounded', b'', 0, 'output_limit', None)):
            with self.assertRaises(ResetError):
                reset._git('status')


if __name__ == '__main__':
    unittest.main()
