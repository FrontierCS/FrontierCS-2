"""CPU-only local boundary fixtures; no provider, GPU or live platform calls."""
import deployment_test_guard  # noqa: F401
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import signal
import selectors
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

JUDGE = Path(__file__).resolve().parents[1] / 'tasks/frontier-cs-2.0-demo/problems/MLSys/vllm_llm_serving_optimization/judge'
sys.path.insert(0, str(JUDGE))
from instance_supervisor import Limits, run
from terminal_contract import Slot, terminal
from terminal_ledger import Ledger
from containment_policy import Boundary, LocalPlatformSignal, classify_failure
from terminal_contract import validate_record


def framed(result):
    raw = json.dumps({'schema': 'frontier.vllm-child-result-draft/v1', 'result': result}).encode()
    return struct.pack('!I', len(raw)) + raw


def output_code(raw):
    return 'import os;os.write(1,' + repr(raw) + ')'


class TerminalContainmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'ledger.sqlite'
        self.ledger = Ledger(self.path)
        self.slot = Slot('run1', 'attempt1', 'verification1', 'operation1', 'eval', 'swe/public',
                         'candidate', 'instance1', 'sha256:' + 'a' * 64, 'sha256:' + 'b' * 64)
        self.ledger.register(self.slot, 'operation1/testbed1')
        self.limits = Limits(1, 0.3, 4096, 2048, 128)

    def execute(self, code, slot=None, limits=None):
        return run(self.ledger, slot or self.slot, [sys.executable, '-B', '-c', code], limits or self.limits)

    def assert_failure(self, record, cause=None):
        self.assertIn(record['status'], {'runtime_error', 'timeout'})
        self.assertNotIn('infra', record)
        if cause:
            self.assertEqual(record['cause'], cause)
        self.assertTrue(record['candidate_input_dispatched'])
        self.assertTrue(self.ledger.cleanup(Slot(**record['slot']))['confirmed'])

    def test_candidate_service_fault_matrix_never_retries(self):
        faults = [
            ('startup', 'raise RuntimeError("service cannot start")'),
            ('health', 'raise RuntimeError("health probe failed")'),
            ('execution', 'import os,signal;os.kill(os.getpid(),signal.SIGKILL)'),
            ('execution', 'raise MemoryError("injected service memory exhaustion")'),
            ('request', 'import time;time.sleep(10)'),
            ('response', 'import os;os.write(1,b"not a framed response")'),
        ]
        for index, (stage, code) in enumerate(faults):
            with self.subTest(stage=stage, fault=index):
                slot = replace(self.slot, instance_id='service-fault-' + str(index))
                self.ledger.register(slot, 'service-fault-' + str(index))
                record = run(self.ledger, slot, [sys.executable, '-B', '-c', code],
                             replace(self.limits, total_seconds=0.15),
                             boundary=Boundary('candidate_service', stage))
                self.assert_failure(record)
                decision = record['failure_context']['decision']
                self.assertEqual(decision['disposition'], 'runtime_error')
                self.assertEqual(decision['affected_scope'], 'workload_group')
                self.assertEqual(decision['scientific_effect'], 'evaluation_fail_affected_scope_no_credit')
                self.assertFalse(decision['retry_eligible'])
                validate_record(record)

    def test_model_output_sidecar_faults_are_case_local(self):
        for index, code in enumerate([
                'raise RuntimeError("model command killed SWE worker")',
                'raise OSError(28,"injected SWE quota exhaustion")',
                'import time;time.sleep(10)']):
            with self.subTest(fault=index):
                slot = replace(self.slot, instance_id='swe-fault-' + str(index))
                self.ledger.register(slot, 'swe-fault-' + str(index))
                record = run(self.ledger, slot, [sys.executable, '-B', '-c', code],
                             replace(self.limits, total_seconds=0.15),
                             boundary=Boundary('swe_case', 'tool'))
                self.assert_failure(record)
                self.assertEqual(record['failure_context']['decision']['disposition'], 'case_fail')
                self.assertEqual(record['failure_context']['decision']['affected_scope'], 'case')
                self.assertFalse(record['failure_context']['decision']['retry_eligible'])
        sibling = replace(self.slot, instance_id='unaffected-swe-case')
        self.ledger.register(sibling, 'unaffected-swe-case')
        good = run(self.ledger, sibling, [sys.executable, '-B', '-c', output_code(framed({'ok': True}))],
                   self.limits, boundary=Boundary('swe_case', 'tool'))
        self.assertEqual(good['status'], 'complete')
        self.assertNotIn('failure_context', good)

    def test_only_bound_typed_candidate_independent_platform_signals_are_infra(self):
        signals = [('gpu_create_failed', 'driver_gpu_provisioning', 'provisioning'),
                   ('sandbox_create_failed', 'driver_sandbox_provisioning', 'provisioning'),
                   ('driver_transport_failed', 'driver_control', 'control'),
                   ('network_control_failed', 'driver_control', 'control'),
                   ('sidecar_unavailable', 'sidecar_preflight', 'sidecar_preflight')]
        for code, channel, stage in signals:
            with self.subTest(code=code):
                boundary = Boundary('platform', stage, 'swe_case' if stage == 'sidecar_preflight' else 'candidate_service')
                signal = LocalPlatformSignal(self.slot.run_id, self.slot.attempt_id,
                                             self.slot.operation_id, code, channel, True,
                                             preinput_never_forwarded=stage == 'sidecar_preflight')
                result = classify_failure(self.slot, boundary, candidate_input_dispatched=False, platform_signal=signal,
                                          dispatch_state='never_forwarded')
                self.assertEqual(result['disposition'], 'infrastructure')
                self.assertTrue(result['retry_eligible'])
                for invalid in [None, {'code': code}, replace(signal, attempt_id='another-attempt'),
                                replace(signal, operation_id='another-operation'),
                                replace(signal, candidate_independent=False), replace(signal, code='new_unknown')]:
                    decision = classify_failure(self.slot, boundary, candidate_input_dispatched=False, platform_signal=invalid)
                    self.assertEqual(decision['attribution'], 'candidate')
                    self.assertFalse(decision['retry_eligible'])

    def test_candidate_startup_health_before_first_input_stays_runtime_error(self):
        signal = LocalPlatformSignal(self.slot.run_id, self.slot.attempt_id, self.slot.operation_id,
                                     'sidecar_unavailable', 'sidecar_preflight', True)
        for stage in ['startup', 'health']:
            result = classify_failure(self.slot, Boundary('candidate_service', stage),
                                      candidate_input_dispatched=False, platform_signal=signal)
            self.assertEqual(result['disposition'], 'runtime_error')
            self.assertFalse(result['retry_eligible'])

    def test_tainted_sidecar_readiness_falls_back_to_its_explicit_domain(self):
        signal = LocalPlatformSignal(self.slot.run_id, self.slot.attempt_id, self.slot.operation_id,
                                     'sidecar_unavailable', 'sidecar_preflight', True)
        for domain, expected in [('swe_case', 'case_fail'), ('candidate_service', 'runtime_error')]:
            boundary = Boundary('platform', 'sidecar_preflight', domain)
            result = classify_failure(self.slot, boundary, candidate_input_dispatched=True, platform_signal=signal)
            self.assertEqual(result['disposition'], expected)
            self.assertFalse(result['retry_eligible'])
        with self.assertRaises(ValueError):
            Boundary('platform', 'sidecar_preflight')

    def test_child_cannot_forge_boundary_or_platform_decision(self):
        record = run(self.ledger, self.slot, [sys.executable, '-B', '-c', 'raise Exception()'],
                     self.limits, boundary=Boundary('swe_case', 'tool'))
        record['failure_context']['decision']['attribution'] = 'platform'
        record['failure_context']['decision']['retry_eligible'] = True
        with self.assertRaises(ValueError):
            validate_record(record)

    def test_unknown_platform_outcome_defaults_to_affected_candidate_domain(self):
        for domain, expected in [('swe_case', 'case_fail'), ('candidate_service', 'runtime_error')]:
            result = classify_failure(self.slot, Boundary('platform', 'control', domain),
                                      candidate_input_dispatched=True)
            self.assertEqual(result['disposition'], expected)
            self.assertFalse(result['retry_eligible'])

    def test_false_legacy_dispatch_flag_is_not_clean_sidecar_proof(self):
        boundary = Boundary('platform', 'sidecar_preflight', 'swe_case')
        signal = LocalPlatformSignal(self.slot.run_id, self.slot.attempt_id, self.slot.operation_id,
                                     'sidecar_unavailable', 'sidecar_preflight', True,
                                     preinput_never_forwarded=True)
        unknown = classify_failure(self.slot, boundary, candidate_input_dispatched=False, platform_signal=signal)
        self.assertEqual(unknown['dispatch_state'], 'unknown')
        self.assertEqual(unknown['disposition'], 'case_fail')
        clean = classify_failure(self.slot, boundary, candidate_input_dispatched=False, platform_signal=signal,
                                 dispatch_state='never_forwarded')
        self.assertEqual(clean['disposition'], 'infrastructure')
        tainted = classify_failure(self.slot, boundary, candidate_input_dispatched=True, platform_signal=signal,
                                   dispatch_state='never_forwarded')
        self.assertEqual(tainted['dispatch_state'], 'observed_dispatched')
        self.assertEqual(tainted['disposition'], 'case_fail')
        unproven = classify_failure(self.slot, boundary, candidate_input_dispatched=False,
                                    platform_signal=replace(signal, preinput_never_forwarded=False),
                                    dispatch_state='never_forwarded')
        self.assertEqual(unproven['disposition'], 'case_fail')

    def test_success_is_execution_only_and_persisted(self):
        result = self.execute(output_code(framed({'answer': 'OK'})))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['result'], {'answer': 'OK'})
        self.assertEqual(result['authority'], 'local-controller-draft-unattested')
        self.assertNotIn('score', result)
        self.assertEqual(Ledger(self.path).read(self.slot), result)
        self.assertEqual(self.ledger.unresolved(), [])

    def test_taint_is_committed_before_child_can_read_it(self):
        code = ('import sqlite3,os;db=sqlite3.connect(' + repr(str(self.path)) + ');'
                'assert db.execute("SELECT dispatched FROM lifecycles").fetchone()==(1,);'
                'assert db.execute("SELECT started FROM slots").fetchone()==(1,);'
                'os.write(1,' + repr(framed({'seen': True})) + ')')
        self.assertEqual(self.execute(code)['result'], {'seen': True})

    def test_sticky_taint_survives_reopen_and_other_variant(self):
        self.assertTrue(self.ledger.mark_dispatched(self.slot))
        reopened = Ledger(self.path)
        sibling = replace(self.slot, variant='reference')
        reopened.register(sibling, 'operation1/testbed1')
        self.assertTrue(reopened.dispatched(sibling))
        with self.assertRaises(ValueError):
            reopened.register(self.slot, 'replacement-lifecycle')

    def test_dispatched_unresolved_slot_is_not_replayed(self):
        self.ledger.mark_dispatched(self.slot)
        with self.assertRaisesRegex(RuntimeError, 'reconcile'):
            self.execute('raise AssertionError("must not execute")')

    def test_first_terminal_wins_under_competing_threads(self):
        barrier = threading.Barrier(2)
        def commit(status):
            barrier.wait()
            record = terminal(self.slot, status=status,
                              cause='normal_completion' if status == 'complete' else 'deadline',
                              elapsed=1, cap=1, result={} if status == 'complete' else None,
                              result_limit_bytes=32 if status == 'complete' else None,
                              result_limit_depth=128 if status == 'complete' else None)
            return self.ledger.commit_once(self.slot, record)
        with ThreadPoolExecutor(max_workers=2) as pool:
            winners = list(pool.map(commit, ('complete', 'timeout')))
        self.assertEqual(sum(won for won, _ in winners), 1)
        self.assertEqual(winners[0][1], winners[1][1])

    def test_late_child_result_cannot_replace_timeout(self):
        first = terminal(self.slot, status='timeout', cause='deadline', elapsed=1, cap=1)
        first = self.ledger.commit_once(self.slot, first)[1]
        late = terminal(self.slot, status='complete', cause='normal_completion', elapsed=2, cap=1, result={}, result_limit_bytes=32, result_limit_depth=128)
        self.assertEqual(self.ledger.commit_once(self.slot, late), (False, first))
        with patch('instance_supervisor.subprocess.Popen') as spawn:
            self.assertEqual(self.execute('ignored'), first)
            spawn.assert_not_called()

    def test_unknown_child_exception_is_scientific_failure(self):
        record = self.execute('raise LookupError("new uninventoried failure")')
        self.assert_failure(record, 'child_exit')
        self.assertEqual(record['failure_context']['decision']['disposition'], 'runtime_error')
        self.assertFalse(record['failure_context']['decision']['retry_eligible'])

    def test_baseexception_systemexit_and_abrupt_exit_remain_candidate_failures(self):
        for index, code in enumerate(('raise BaseException("uninventoried")',
                                      'raise SystemExit(7)', 'import os;os._exit(19)')):
            slot = replace(self.slot, instance_id='nonexception-' + str(index))
            self.ledger.register(slot, 'operation1/nonexception-' + str(index))
            result = self.execute(code, slot=slot)
            self.assert_failure(result, 'child_exit')
            self.assertFalse(result['failure_context']['decision']['retry_eligible'])

    def test_child_ignoring_graceful_cleanup_cannot_hold_controller(self):
        code = ('import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                'os.fork();time.sleep(60)')
        started = time.monotonic()
        result = self.execute(code, limits=replace(self.limits, total_seconds=0.15))
        self.assert_failure(result, 'deadline')
        self.assertLess(time.monotonic() - started, 1.5)

    def test_unknown_parent_spawn_exception_is_scientific_failure(self):
        with patch('instance_supervisor.subprocess.Popen', side_effect=LookupError('new failure')):
            self.assert_failure(self.execute('unused'), 'unknown_exception')

    def test_child_sigkill_and_exit137_do_not_mint_infra(self):
        for code in ('import os,signal;os.kill(os.getpid(),signal.SIGKILL)', 'import os;os._exit(137)'):
            slot = replace(self.slot, instance_id=code)
            self.ledger.register(slot, 'operation1/' + str(len(code)))
            self.assert_failure(self.execute(code, slot=slot), 'child_exit')

    def test_entire_mock_scheduler_domain_dies_controller_survives(self):
        code = 'import os,signal;os.fork();os.killpg(os.getpgrp(),signal.SIGKILL)'
        result = self.execute(code)
        self.assert_failure(result, 'child_exit')
        self.assertEqual(result['exit_code'], -signal.SIGKILL)
        sibling = replace(self.slot, instance_id='sibling')
        self.ledger.register(sibling, 'operation1/sibling')
        self.assertEqual(self.execute(output_code(framed({'alive': True})), slot=sibling)['status'], 'complete')
        self.assertEqual(self.ledger.unresolved(), [])

    def test_hang_has_hard_total_deadline(self):
        started = time.monotonic()
        record = self.execute('import time;time.sleep(60)', limits=Limits(0.15, 0.3, 4096, 2048, 128))
        self.assert_failure(record, 'deadline')
        self.assertLess(time.monotonic() - started, 1.5)

    def test_slow_output_does_not_reset_total_deadline(self):
        code = 'import os,time;os.write(1,b"\\x00\\x00\\x00\\x64");[(os.write(1,b"a"),time.sleep(.04)) for _ in range(100)]'
        record = self.execute(code, limits=Limits(0.15, 0.3, 4096, 2048, 128))
        self.assert_failure(record, 'deadline')

    def test_excessive_stderr_is_bounded_and_prefix_only(self):
        record = self.execute('import os;\nwhile True: os.write(2,b"x"*65536)')
        self.assert_failure(record, 'output_limit')
        stream = record['streams']['stderr']
        self.assertEqual(stream['retained_bytes'], 2048)
        self.assertLessEqual(stream['observed_bytes'], 2049)
        self.assertTrue(stream['truncated'])
        self.assertFalse(stream['stream_fully_observed'])
        self.assertNotIn('full_sha256', stream)

    def test_oversize_frame_length_is_rejected_before_body(self):
        self.assert_failure(self.execute(output_code(struct.pack('!I', 10**8))), 'output_limit')

    def test_garbage_nonutf8_duplicate_and_forged_frames(self):
        malformed = [b'garbage', b'\x00\x00\x00\x01\xff', framed({}) + framed({}),
                     struct.pack('!I', 3) + b'NaN',
                     struct.pack('!I', 40) + b'{"status":"infra","code":"unavailable"}']
        for index, raw in enumerate(malformed):
            with self.subTest(index=index):
                slot = replace(self.slot, instance_id='bad' + str(index))
                self.ledger.register(slot, 'operation1/bad' + str(index))
                self.assert_failure(self.execute(output_code(raw), slot=slot))

    def test_candidate_claims_remain_untrusted_data(self):
        result = self.execute(output_code(framed({'code': 'gpu_backend_unavailable', 'status': 'infra'})))
        self.assertEqual(result['status'], 'complete')
        self.assertNotIn('platform_signal', result)
        self.assertNotIn(result['status'], {'infra', 'retry'})

    def test_environment_does_not_inherit_credentials(self):
        with patch.dict('os.environ', {'FRONTIER_ATTESTATION_PRIVATE_KEY': 'fixture-not-a-real-key'}):
            code = 'import os;assert "FRONTIER_ATTESTATION_PRIVATE_KEY" not in os.environ;' + output_code(framed({}))
            self.assertEqual(self.execute(code)['status'], 'complete')

    def test_deleted_workspace_and_full_disk_exception_are_contained(self):
        for index, code in enumerate(('import tempfile,os;p=tempfile.mkdtemp();os.rmdir(p);os.chdir(p)',
                                      'import errno;raise OSError(errno.ENOSPC,"fixture full disk")')):
            slot = replace(self.slot, instance_id='io' + str(index))
            self.ledger.register(slot, 'operation1/io' + str(index))
            self.assert_failure(self.execute(code, slot=slot), 'child_exit')

    def test_record_cannot_forge_infra_or_cross_slot(self):
        record = terminal(self.slot, status='runtime_error', cause='child_exit', elapsed=0, cap=1)
        record['status'] = 'infra'
        with self.assertRaises(ValueError):
            self.ledger.commit_once(self.slot, record)
        record['status'] = 'runtime_error'
        record['slot']['instance_id'] = 'other'
        with self.assertRaises(ValueError):
            self.ledger.commit_once(self.slot, record)

    def test_limits_and_digest_validation_reject_invalid_contract(self):
        for seconds in (0, -1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                Limits(seconds, 1, 1, 1, 1)
        with self.assertRaises(ValueError):
            replace(self.slot, request_digest='not-a-digest')

    def test_attempt_is_part_of_slot_identity(self):
        self.assertNotEqual(self.slot.key(), replace(self.slot, attempt_id='attempt2').key())

    def test_terminal_metadata_rejects_unbounded_or_forged_fields(self):
        original = self.execute(output_code(framed({})))
        from terminal_contract import validate_record
        for update in ({'exit_code': True}, {'platform_signal': 'gpu_backend_unavailable'},
                       {'authority': 'scheduler-signed'}, {'result_limit_bytes': 1}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                candidate = json.loads(json.dumps(original))
                candidate.update(update)
                validate_record(candidate)
        for field, bad in (('observed_bytes', 10**9), ('retained_bytes', -1),
                           ('truncated', 1), ('retained_prefix_sha256', 'fake')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                candidate = json.loads(json.dumps(original))
                candidate['streams']['stdout'][field] = bad
                validate_record(candidate)

    def test_terminal_cannot_clear_persisted_dispatch_taint(self):
        self.ledger.mark_dispatched(self.slot)
        record = terminal(self.slot, status='runtime_error', cause='child_exit', elapsed=0, cap=1)
        record['candidate_input_dispatched'] = False
        with self.assertRaises(ValueError):
            self.ledger.commit_once(self.slot, record)

    def test_selector_creation_failure_still_finalizes(self):
        with patch('instance_supervisor.selectors.DefaultSelector', side_effect=LookupError('unknown')):
            self.assert_failure(self.execute('unused'), 'unknown_exception')

    def test_primary_commits_before_cleanup_and_cleanup_cannot_replace_it(self):
        import instance_supervisor
        actual_reap = instance_supervisor._reap
        def inspect_primary(process, seconds):
            self.assertEqual(self.ledger.read(self.slot)['status'], 'complete')
            time.sleep(0.15)
            return actual_reap(process, seconds)
        started = time.monotonic()
        with patch('instance_supervisor._reap', side_effect=inspect_primary):
            result = self.execute(output_code(framed({})))
        self.assertGreaterEqual(time.monotonic() - started - result['elapsed_seconds'], 0.14)
        self.ledger.record_cleanup(self.slot, {'confirmed': False, 'errors': ['reap']})
        self.assertTrue(self.ledger.cleanup(self.slot)['confirmed'])
        self.assertEqual(self.ledger.read(self.slot), result)

    def test_reap_failure_is_separate_from_primary(self):
        import instance_supervisor
        actual_reap = instance_supervisor._reap
        def fail_reap(process, seconds):
            self.addCleanup(actual_reap, process, seconds)
            raise OSError('fixture cleanup failure')
        with patch('instance_supervisor._reap', side_effect=fail_reap):
            result = self.execute(output_code(framed({})))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(self.ledger.cleanup(self.slot), {'confirmed': False, 'errors': ['reap']})

    def test_selector_close_failure_is_separate_from_primary(self):
        actual = selectors.DefaultSelector()
        self.addCleanup(actual.close)
        class BrokenClose:
            def __getattr__(self, name):
                return getattr(actual, name)
            def close(self):
                raise LookupError('fixture close failure')
        with patch('instance_supervisor.selectors.DefaultSelector', return_value=BrokenClose()):
            result = self.execute(output_code(framed({})))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(self.ledger.cleanup(self.slot), {'confirmed': False, 'errors': ['selector_close']})

    def test_pipe_close_failure_cannot_prevent_terminal_commit(self):
        import instance_supervisor
        actual_spawn = instance_supervisor.subprocess.Popen
        def spawn(*args, **kwargs):
            process = actual_spawn(*args, **kwargs)
            actual_pipe = process.stdout
            self.addCleanup(actual_pipe.close)
            class BrokenClose:
                def fileno(self):
                    return actual_pipe.fileno()
                def close(self):
                    raise OSError('fixture pipe close failure')
            process.stdout = BrokenClose()
            return process
        with patch('instance_supervisor.subprocess.Popen', side_effect=spawn):
            result = self.execute(output_code(framed({})))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(self.ledger.cleanup(self.slot), {'confirmed': False, 'errors': ['stdout_close']})

    def test_external_timeout_wins_over_late_actual_child_result(self):
        def close_pending():
            time.sleep(0.05)
            record = terminal(self.slot, status='timeout', cause='deadline', elapsed=0.05, cap=0.05)
            self.ledger.commit_once(self.slot, record)
        thread = threading.Thread(target=close_pending)
        thread.start()
        try:
            result = self.execute('import time;time.sleep(.15);' + output_code(framed({'late': True})))
        finally:
            thread.join(timeout=1)
        self.assertEqual(result['status'], 'timeout')
        self.assertIsNone(result['result'])

    def test_child_terminal_encoding_failure_falls_back_without_losing_slot(self):
        import instance_supervisor
        actual = instance_supervisor.terminal
        def fail_child_wrapper(*args, **kwargs):
            if kwargs.get('result') is not None:
                raise RecursionError('fixture nested child encoding')
            return actual(*args, **kwargs)
        with patch('instance_supervisor.terminal', side_effect=fail_child_wrapper):
            self.assert_failure(self.execute(output_code(framed({'nested': []}))), 'protocol_error')
        self.assertEqual(self.ledger.unresolved(), [])

    def test_deep_valid_json_near_recursion_bound_does_not_escape(self):
        for depth in (900, 970, 990):
            slot = replace(self.slot, instance_id='nested' + str(depth))
            self.ledger.register(slot, 'operation1/nested' + str(depth))
            body = (b'{"schema":"frontier.vllm-child-result-draft/v1","result":{"nested":'
                    + b'[' * depth + b'0' + b']' * depth + b'}}')
            record = self.execute(output_code(struct.pack('!I', len(body)) + body), slot=slot)
            self.assertIn(record['status'], {'complete', 'runtime_error'})
            if record['status'] == 'runtime_error':
                self.assertEqual(record['cause'], 'protocol_error')
            self.assertIsNotNone(self.ledger.read(slot))

    def test_child_frame_at_exact_limit_has_bounded_result(self):
        raw = framed({'padding': 'x' * 256})
        result = self.execute(output_code(raw), limits=Limits(1, .3, len(raw) - 4, 2048, 128))
        self.assertEqual(result['status'], 'complete')
        self.assertLessEqual(len(json.dumps(result['result']).encode()), result['result_limit_bytes'])

    def test_collector_catch_all_closes_only_owned_unresolved_slots(self):
        first = self.execute(output_code(framed({})))
        pending = replace(self.slot, instance_id='pending')
        unrelated = replace(self.slot, group='bfcl/public', instance_id='unrelated')
        for slot in (pending, unrelated):
            self.ledger.register(slot, 'operation1/' + slot.instance_id)
        rows = self.ledger.close_unresolved((self.slot, pending), elapsed=1, cap=1)
        self.assertEqual(rows[0], first)
        self.assertEqual(rows[1]['status'], 'runtime_error')
        self.assertEqual(rows[1]['cause'], 'collector_failure')
        self.assertFalse(rows[1]['candidate_input_dispatched'])
        self.assertEqual(self.ledger.unresolved(), [json.loads(unrelated.key())])

    def test_trusted_persistence_failure_is_not_candidate_zero(self):
        with patch.object(self.ledger, 'commit_once', side_effect=OSError('fixture trusted disk failure')):
            with self.assertRaises(OSError):
                self.execute(output_code(framed({})))
        self.assertIsNone(self.ledger.read(self.slot))


if __name__ == '__main__':
    unittest.main()
