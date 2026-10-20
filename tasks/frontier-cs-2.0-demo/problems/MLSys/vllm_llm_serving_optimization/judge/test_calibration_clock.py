"""Local filesystem and synthetic clock proofs; no Driver/provider execution."""
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'verifier'))
import calibration_clock as clock


def binding():
    return {'driver_origin': 'http://driver.fixture:8081',
            'verification_id': 'verification_fixture', 'request_key': 'create_fixture',
            'request': {'gpu_type': 'H100', 'gpu_count': 1, 'max_duration_seconds': 500},
            'config_sha256': 'sha256:' + 'a' * 64, 'plan_sha256': 'sha256:' + 'b' * 64}


class CalibrationClockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'clock.json'
        self.now = 1000.0
        self.domain = {'boot_id': 'synthetic_boot', 'time_namespace': 'time:[123]',
                       'monotonic_offset_ns': 0, 'implementation': 'synthetic_monotonic'}
        for replacement in (
            patch.object(clock.time, 'monotonic', side_effect=lambda: self.now),
            patch.object(clock, '_clock_domain', side_effect=lambda: copy.deepcopy(self.domain)),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        self.budgets = {'cap_seconds': 500, 'cleanup_seconds': 20, 'capture_seconds': 10}

    def begin(self, **kwargs):
        return clock.begin(kwargs.pop('binding', binding()), kwargs.pop('path', self.path),
                           **{**self.budgets, **kwargs})

    def test_begin_fsyncs_private_file_and_directory_before_return(self):
        events = []
        actual_fsync = os.fsync
        def fsync(descriptor):
            mode = os.fstat(descriptor).st_mode
            events.append('directory' if stat.S_ISDIR(mode) else 'file')
            actual_fsync(descriptor)
        with patch.object(clock.os, 'fsync', side_effect=fsync):
            timer = self.begin()
        self.assertEqual(events, ['file', 'directory'])
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        marker = json.loads(self.path.read_bytes())
        self.assertEqual(marker['t0'], 1000.0)
        self.assertEqual(marker['binding'], binding())
        self.assertEqual(marker['clock_domain'], self.domain)
        self.assertEqual(timer.initial_deadlines, {'work_deadline': 1470.0,
                         'delete_deadline': 1490.0, 'capture_deadline': 1500.0})

    def test_delayed_response_consumes_budget_without_extending_deadlines(self):
        timer = self.begin()
        self.now = 1300.0
        deadlines = timer.accept_grant('gpurunner_fixture', 450)
        self.assertEqual(deadlines, {'work_deadline': 1419.0,
                         'delete_deadline': 1439.0, 'capture_deadline': 1449.0})
        self.assertEqual(timer.runner_id, 'gpurunner_fixture')
        self.assertEqual(timer.granted_seconds, 450)
        self.now = 1450.0
        self.assertEqual(timer.accept_grant('gpurunner_fixture', 450), deadlines)
        self.assertLess(deadlines['capture_deadline'], self.now)

    def test_short_grant_retained_even_when_no_work_budget_remains(self):
        timer = self.begin()
        self.now = 1002.0
        deadlines = timer.accept_grant('gpurunner_fixture', 25)
        self.assertLess(deadlines['work_deadline'], timer.t0)
        self.assertGreater(deadlines['delete_deadline'], self.now)
        self.assertEqual(json.loads(self.path.read_bytes())['granted_seconds'], 25)

    def test_resume_keeps_earliest_pregrant_time_and_accepted_grant(self):
        first = self.begin()
        self.now = 1100.0
        resumed = self.begin(resume=True)
        self.assertEqual(resumed.t0, first.t0)
        self.assertEqual(resumed.initial_deadlines, first.initial_deadlines)
        expected = resumed.accept_grant('gpurunner_fixture', 400)
        self.now = 1250.0
        again = self.begin(resume=True)
        self.assertEqual(again.t0, 1000.0)
        self.assertEqual(again.local_cap_deadline, 1500.0)
        self.assertEqual(again.initial_deadlines, expected)
        self.assertEqual(again.accept_grant('gpurunner_fixture', 400), expected)
        with self.assertRaises(clock.CalibrationClockError):
            self.begin()

    def test_stale_object_cannot_replace_another_accepted_runner_or_grant(self):
        first = self.begin()
        second = self.begin(resume=True)
        first.accept_grant('gpurunner_first', 400)
        for runner, grant in [('gpurunner_other', 400), ('gpurunner_first', 399)]:
            with self.subTest(runner=runner, grant=grant), self.assertRaises(clock.CalibrationClockError):
                second.accept_grant(runner, grant)
        self.assertEqual(second.accept_grant('gpurunner_first', 400), first.deadlines)

    def test_binding_and_budget_changes_never_rewrite_marker(self):
        self.begin()
        original = self.path.read_bytes()
        for key, value in [('driver_origin', 'http://other.fixture'),
                           ('verification_id', 'verification_other'), ('request_key', 'other'),
                           ('config_sha256', 'sha256:'+'c'*64), ('plan_sha256', 'sha256:'+'d'*64),
                           ('request', {'gpu_type': 'H100', 'gpu_count': 1, 'max_duration_seconds': 499})]:
            changed = binding(); changed[key] = value
            with self.subTest(key=key), self.assertRaises(clock.CalibrationClockError):
                self.begin(binding=changed, resume=True)
            self.assertEqual(self.path.read_bytes(), original)
        for key, value in [('cap_seconds', 499), ('cleanup_seconds', 21),
                           ('capture_seconds', 11), ('margin_seconds', 2)]:
            with self.subTest(key=key), self.assertRaises(clock.CalibrationClockError):
                self.begin(resume=True, **{key: value})
            self.assertEqual(self.path.read_bytes(), original)

    def test_boot_namespace_offset_or_implementation_drift_refuses_resume_and_accept(self):
        timer = self.begin()
        domain = copy.deepcopy(self.domain)
        for key, value in [('boot_id', 'rebooted'), ('time_namespace', 'time:[456]'),
                           ('monotonic_offset_ns', 1), ('implementation', 'other_clock')]:
            self.domain = {**domain, key: value}
            with self.subTest(key=key):
                with self.assertRaises(clock.CalibrationClockError):
                    self.begin(resume=True)
                with self.assertRaises(clock.CalibrationClockError):
                    timer.accept_grant('gpurunner_fixture', 400)
        self.domain = domain
        self.assertEqual(self.begin(resume=True).t0, 1000.0)

    def test_missing_corrupt_and_future_markers_refuse_explicit_resume(self):
        with self.assertRaises(clock.CalibrationClockError):
            self.begin(resume=True)
        self.assertFalse(self.path.exists())
        self.begin()
        valid = self.path.read_bytes()
        for raw in [b'{', b'{}', b'x' * 65537]:
            self.path.write_bytes(raw)
            with self.assertRaises(clock.CalibrationClockError):
                self.begin(resume=True)
        marker = json.loads(valid)
        marker.update(t0=2000, local_cap_deadline=2500)
        self.path.write_text(json.dumps(marker))
        with self.assertRaises(clock.CalibrationClockError):
            self.begin(resume=True)

    def test_nonblocking_lock_refuses_competing_marker_update(self):
        with open(str(self.path) + '.lock', 'w') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(clock.CalibrationClockError):
                self.begin()
            self.assertFalse(self.path.exists())
        self.begin()

    def test_marker_boolean_type_drift_is_rejected(self):
        self.begin()
        marker = json.loads(self.path.read_bytes())
        marker['binding']['request']['gpu_count'] = True
        self.path.write_text(json.dumps(marker))
        with self.assertRaises(clock.CalibrationClockError):
            self.begin(resume=True)

    def test_marker_symlink_is_not_followed(self):
        other = self.root/'other'; other.write_text('unchanged')
        self.path.symlink_to(other)
        with self.assertRaises(OSError):
            self.begin(resume=True)
        self.assertEqual(other.read_text(), 'unchanged')

    def test_fsync_failure_returns_no_ready_clock_and_recovery_resyncs(self):
        with patch.object(clock.os, 'fsync', side_effect=OSError('synthetic fsync failure')):
            with self.assertRaises(OSError):
                self.begin()
        resumed = self.begin(resume=True)
        self.assertEqual(resumed.t0, 1000.0)
        with patch.object(clock, '_sync_directory', side_effect=OSError('synthetic directory failure')):
            with self.assertRaises(OSError):
                resumed.accept_grant('gpurunner_fixture', 400)
        # Replacement happened; recover the original durable identity, never a
        # second runner after an ambiguous persistence response.
        recovered = self.begin(resume=True)
        self.assertEqual(recovered.granted_seconds, 400)
        with self.assertRaises(clock.CalibrationClockError):
            recovered.accept_grant('gpurunner_other', 400)

    def test_secrets_and_unbounded_metadata_are_not_accepted_binding_fields(self):
        for kind in ('token', 'request_token', 'url_password', 'url_query', 'url_fragment'):
            changed = binding()
            if kind == 'token': changed['token'] = 'synthetic_bearer'
            elif kind == 'request_token': changed['request']['token'] = 'synthetic_bearer'
            elif kind == 'url_password': changed['driver_origin'] = 'http://user:synthetic@driver.fixture'
            elif kind == 'url_query': changed['driver_origin'] += '?token=synthetic'
            else: changed['driver_origin'] += '#synthetic'
            with self.subTest(kind=kind), self.assertRaises(clock.CalibrationClockError):
                self.begin(binding=changed)
        self.assertFalse(self.path.exists())

    def test_inadequate_or_nonfinite_reserves_and_invalid_grants(self):
        for kwargs in [{'cap_seconds': 31}, {'cleanup_seconds': 0}, {'capture_seconds': 0},
                       {'capture_seconds': True}, {'margin_seconds': 0.999},
                       {'margin_seconds': math.inf}, {'cap_seconds': math.nan}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(clock.CalibrationClockError):
                self.begin(**kwargs)
        timer = self.begin()
        for runner, grant in [('bad/runner', 400), ('runner', 0), ('runner', True),
                              ('runner', 400.0), ('runner', 501)]:
            with self.subTest(runner=runner, grant=grant), self.assertRaises(clock.CalibrationClockError):
                timer.accept_grant(runner, grant)
        self.assertIsNone(timer.granted_seconds)

    def test_explicit_extra_clock_error_margin_shortens_cutoffs(self):
        timer = self.begin(margin_seconds=3.5)
        expected = timer.accept_grant('gpurunner_fixture', 450)
        self.assertEqual(expected['capture_deadline'], 1446.5)
        self.assertEqual(self.begin(margin_seconds=3.5, resume=True).deadlines, expected)

    def test_synthetic_train_and_eval_integer_clamps_are_before_owner_endpoints(self):
        # Driver epoch differs arbitrarily from client epoch; only grant-event
        # ordering and comparable elapsed rates are assumed. No cloud timing.
        cases = 0
        for mode in ('train', 'eval'):
            for elapsed in (0.0, 0.125, 1.999, 399.875):
                for delay in (0.0, 3.25):
                    remaining_verification = 400 - elapsed
                    remaining_answer = 40.75
                    grant = min(500, 400 - math.floor(elapsed))
                    if mode == 'train':
                        grant = min(grant, math.floor(remaining_answer))
                    timer = self.begin(path=self.root/f'math-{cases}.json')
                    self.now = 1000 + delay + 0.5
                    cutoffs = timer.accept_grant('gpurunner_fixture', grant)
                    verification_end = 1000 + delay + remaining_verification
                    self.assertLess(cutoffs['capture_deadline'], verification_end)
                    if mode == 'train':
                        self.assertLess(cutoffs['capture_deadline'], 1000 + delay + remaining_answer)
                    self.assertLess(cutoffs['work_deadline'], cutoffs['delete_deadline'])
                    self.assertLess(cutoffs['delete_deadline'], cutoffs['capture_deadline'])
                    self.now = 1000.0
                    cases += 1


if __name__ == '__main__':
    unittest.main()
