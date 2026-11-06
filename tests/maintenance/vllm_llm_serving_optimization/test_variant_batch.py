"""Local parent lifecycle checks; no vLLM, GPU, provider or model proof."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from variant_batch import (_validate_start, failure_diagnostics, source_digest,
                           stop_group, stop_group_id)


class VariantProcessTests(unittest.TestCase):
    def test_failure_diagnostics_identify_stage_and_bound_log_tails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'setup.log').write_text('old-prefix-' + 'x' * 1000)
            (root / 'server.log').write_text('server-tail')
            record = failure_diagnostics(
                root, 'engine_readiness', RuntimeError('startup expired'), 600, 12.5)
            encoded = __import__('json').dumps(
                record, allow_nan=False, separators=(',', ':')).encode()
            self.assertLessEqual(len(encoded), 600)
            self.assertEqual(record['stage'], 'engine_readiness')
            self.assertEqual(record['error_type'], 'RuntimeError')
            self.assertEqual(record['error_message'], 'startup expired')
            self.assertTrue(record['diagnostics']['setup.log']['truncated'])
            self.assertEqual(record['diagnostics']['server.log']['tail'], 'server-tail')
            long_record = failure_diagnostics(
                root, 'prepare', RuntimeError('e' * 10000), 600, 1.0)
            long_encoded = __import__('json').dumps(
                long_record, allow_nan=False, separators=(',', ':')).encode()
            self.assertLessEqual(len(long_encoded), 600)
            self.assertLess(len(long_record['error_message']), 10000)

    def test_exited_leader_does_not_leave_term_ignoring_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'child'
            child_script = ('import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                'open(' + repr(str(path)) + ',"w").write(str(os.getpid())); time.sleep(30)')
            leader_script = ('import subprocess,sys,time,pathlib; '
                'subprocess.Popen([sys.executable,"-c",' + repr(child_script) + ']); '
                'p=pathlib.Path(' + repr(str(path)) + ');\n'
                'while not p.exists(): time.sleep(.01)')
            leader = subprocess.Popen([sys.executable, '-c', leader_script],
                                      start_new_session=True)
            try:
                leader.wait(timeout=5)
                child = int(path.read_text())
                self.assertTrue(Path(f'/proc/{child}').exists())
                stop_group(leader, 1)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        state = Path(f'/proc/{child}/stat').read_text().split()[2]
                    except FileNotFoundError:
                        break
                    if state == 'Z':
                        break
                    time.sleep(.01)
                else:
                    self.fail('child remains alive after leader exited')
            finally:
                try:
                    os.killpg(leader.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                leader.wait()

    def test_exact_detached_generation_is_terminated(self):
        process = subprocess.Popen([sys.executable, '-c',
            'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'],
            start_new_session=True)
        try:
            time.sleep(0.1)
            stat = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
            ticks = int(stat[19])
            receipt = stop_group_id(process.pid, ticks, 1)
            self.assertTrue(receipt['term_sent'])
            self.assertTrue(receipt['kill_sent'])
            process.wait(timeout=2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def test_exact_group_is_terminated_after_leader_exits(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'child'
            child_script = ('import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                'open(' + repr(str(path)) + ',\"w\").write(str(os.getpid())); time.sleep(30)')
            leader_script = ('import subprocess,sys,time,pathlib; '
                'subprocess.Popen([sys.executable,\"-c\",' + repr(child_script) + ']); '
                'p=pathlib.Path(' + repr(str(path)) + ');\n'
                'while not p.exists(): time.sleep(.01)')
            leader = subprocess.Popen([sys.executable, '-c', leader_script],
                                      start_new_session=True)
            try:
                start_ticks = int(
                    Path(f'/proc/{leader.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
                leader.wait(timeout=5)
                child = int(path.read_text())
                receipt = stop_group_id(leader.pid, start_ticks, 1)
                self.assertTrue(receipt['term_sent'])
                self.assertTrue(receipt['kill_sent'])
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        state = Path(f'/proc/{child}/stat').read_text().split()[2]
                    except FileNotFoundError:
                        break
                    if state == 'Z':
                        break
                    time.sleep(.01)
                else:
                    self.fail('child remains alive after exact group cleanup')
            finally:
                try:
                    os.killpg(leader.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                leader.wait()

    def test_closed_start_input_rejects_overrides_and_source_mixup(self):
        for document in ({}, {'command': ['arbitrary']}, {'token': 'synthetic'}):
            with self.assertRaises(ValueError):
                _validate_start(document)
        with self.assertRaises(ValueError):
            _validate_start({'variant': 'vanilla', 'patch': 'x',
                             'model_source_digest': source_digest(b'x')})
        with self.assertRaises(ValueError):
            _validate_start({'variant': 'candidate', 'patch': 'x',
                             'model_source_digest': source_digest(b'y')})


if __name__ == '__main__':
    unittest.main()
