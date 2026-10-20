"""Protect current Polygon rules when importing the disambiguation repair."""
import deployment_test_guard  # noqa: F401

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/frontier-cs-2.0-demo/problems/Programming Language/polygon'
sys.path.insert(0, str(TASK / 'judge'))
spec = importlib.util.spec_from_file_location('polygon_v4_evaluator', TASK / 'judge/evaluator.py')
evaluator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluator
spec.loader.exec_module(evaluator)


class PolygonV4ReconciliationTests(unittest.TestCase):
    def test_runner_descendants_are_stopped_after_success_and_timeout(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / evaluator.RUNNER).write_text(
                    'import json, subprocess, time\n'
                    'from pathlib import Path\n'
                    'p = subprocess.Popen(["sleep", "60"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n'
                    'Path("child.pid").write_text(str(p.pid))\n'
                    'Path("oom_priority").write_text(Path("/proc/self/oom_score_adj").read_text())\n'
                    + ('time.sleep(60)\n' if timeout else 'print(json.dumps({"ok": True, "refuted": True, "time_seconds": 0.1, "iterations": 1, "backtracks": 0}))\n')
                )
                workload = evaluator.Workload('cleanup', 'pair', 1, {}, evaluator.HARD_STRATUM)
                created = []
                real_mkdtemp = evaluator.tempfile.mkdtemp

                def make_directory(*args, **kwargs):
                    path = real_mkdtemp(*args, **kwargs)
                    created.append(Path(path))
                    return path

                with mock.patch.object(evaluator, 'HARD_SUBPROCESS_MARGIN_SECONDS', 0), mock.patch.object(evaluator, 'SUBPROCESS_TIMEOUT_SECONDS', 0), mock.patch.object(evaluator.tempfile, 'mkdtemp', make_directory):
                    result = evaluator._run_case(root, workload, 'unused')
                self.assertEqual(result.valid, not timeout)
                self.assertEqual((root / 'oom_priority').read_text().strip(), '1000')
                self.assertTrue(created)
                self.assertTrue(all(not path.exists() for path in created))
                pid = int((root / 'child.pid').read_text())
                for _ in range(50):
                    status = Path(f'/proc/{pid}/stat')
                    if not status.exists() or status.read_text().split()[2] == 'Z':
                        break
                    time.sleep(0.01)
                else:
                    self.fail('Runner descendant remained active after case cleanup')

    def test_both_roles_preserve_hard_pairs_and_add_disambiguation(self):
        suites = []
        for role in ('dev', 'final'):
            with mock.patch.dict('os.environ', {'FRONTIER_SUBMISSION_ROLE': role, 'FRONTIER_FAST_TEST': '0'}):
                rows = evaluator._load_suite()
            self.assertEqual(len(rows), 21)
            self.assertEqual(sum(row.kind == 'disambiguation' for row in rows), 3)
            self.assertEqual(sum(row.stratum == evaluator.HARD_STRATUM for row in rows), 2)
            self.assertTrue(all(row.timeout_seconds == 20 for row in rows if row.stratum == evaluator.HARD_STRATUM))
            suites.append({(row.payload['problem_id'], row.payload['qid']) for row in rows if row.kind == 'pair'})
        self.assertFalse(suites[0] & suites[1])

    def test_hard_timing_remains_strict_and_score_excludes_hard_cases(self):
        with mock.patch.dict('os.environ', {'FRONTIER_SUBMISSION_ROLE': 'dev', 'FRONTIER_FAST_TEST': '0'}):
            workloads = evaluator._load_suite()
        hard = next(row for row in workloads if row.stratum == evaluator.HARD_STRATUM)
        ref = evaluator.Observation(True, 4.0, 1, 0, hard.timeout_seconds)
        for value, expected in ((4.0, False), (2.0, False), (1.99, True)):
            candidate = evaluator.Observation(True, value, 1, 0, hard.timeout_seconds)
            result = evaluator._comparison_decisions((hard,), [candidate], [ref])
            self.assertEqual(result[hard.identifier]['wall_time_seconds'], expected)
        rows = [evaluator.Observation(True, 0.1, 1, 0, row.timeout_seconds) for row in workloads]
        base = evaluator.score_observations(evaluator._original(workloads, rows))
        for index, row in enumerate(workloads):
            if row.stratum == evaluator.HARD_STRATUM:
                rows[index] = evaluator.Observation(False, 20.0, 0, 0, 20)
        self.assertEqual(base, evaluator.score_observations(evaluator._original(workloads, rows)))

    def test_reference_failure_never_becomes_candidate_zero(self):
        with mock.patch.dict('os.environ', {'FRONTIER_FAST_TEST': '1'}), \
                mock.patch.object(evaluator, '_build'), \
                mock.patch.object(evaluator, '_measure_pair', return_value=([], [evaluator.Observation(False, 2.0, 0, 0, 2, False)])):
            with self.assertRaises(evaluator.ReferenceFailure):
                evaluator.evaluate(str(TASK / 'judge/reference.patch'))

    def test_reference_failure_persists_private_case_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = {}
            for side, body in (
                ('candidate', 'print(json.dumps({"ok": True, "refuted": True, "time_seconds": 0.1, "iterations": 1, "backtracks": 0}))\n'),
                ('reference', 'sys.stderr.write("forced reference diagnostic\\n")\nraise SystemExit(3)\n'),
            ):
                roots[side] = Path(directory) / side
                roots[side].mkdir()
                (roots[side] / evaluator.RUNNER).write_text('import json, sys\nsys.stdin.read()\n' + body)

            class Validator:
                starts = 1
                socket = 'unused'

                def reset(self):
                    return {}

                def freeze(self):
                    pass

            class Validators:
                def __enter__(self):
                    return {evaluator.CANDIDATE_SIDE: Validator(), evaluator.REFERENCE_SIDE: Validator()}

                def __exit__(self, *_args):
                    return False

            real_measure = evaluator._measure_pair
            with mock.patch.dict('os.environ', {'FRONTIER_FAST_TEST': '1'}), \
                    mock.patch.object(evaluator, '_build'), \
                    mock.patch.object(evaluator, '_Validators', Validators), \
                    mock.patch.object(evaluator, '_measure_pair', lambda c, r, w, records: real_measure(roots['candidate'], roots['reference'], w, records)):
                with self.assertRaises(evaluator.ReferenceFailure) as raised:
                    evaluator.evaluate(str(TASK / 'judge/reference.patch'))
        evidence = raised.exception.private_execution_evidence
        reference = [row for row in evidence['cases'] if row['side'] == evaluator.REFERENCE_SIDE]
        candidate = [row for row in evidence['cases'] if row['side'] == evaluator.CANDIDATE_SIDE]
        self.assertEqual(len(reference), 1)
        self.assertEqual(reference[0]['returncode'], 3)
        self.assertFalse(reference[0]['valid'])
        self.assertIn('forced reference diagnostic', reference[0]['stderr_tail'])
        self.assertTrue(candidate[0]['valid'])
        self.assertNotIn('stderr_tail', candidate[0])
        self.assertNotIn('forced', str(raised.exception))

    def test_only_the_reference_gets_a_larger_budget(self):
        workload = evaluator.Workload('budget', 'pair', 2, {'timeout_seconds': 2})
        self.assertEqual(evaluator._reference_budget_seconds(workload), 12)
        self.assertEqual(evaluator._reference_budget_seconds(evaluator.Workload('d100', 'pair', 12, {})), 24)
        self.assertEqual(evaluator._reference_budget_seconds(evaluator.Workload('hard', 'pair', 20, {}, evaluator.HARD_STRATUM)), 40)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / evaluator.RUNNER).write_text(
                'import json, sys, time\n'
                'payload = json.loads(sys.stdin.read())\n'
                'time.sleep(2.3)\n'
                'print(json.dumps({"ok": True, "refuted": True, "time_seconds": 2.3, "iterations": 1, "backtracks": 0}))\n'
            )
            candidate_record, reference_record = {}, {}
            candidate = evaluator._run_case(root, workload, 'unused', candidate_record)
            reference = evaluator._run_case(root, workload, 'unused', reference_record, evaluator._reference_budget_seconds(workload))
        # 2.3 s exceeds the candidate's 2 s budget but not the reference's 12 s.
        self.assertFalse(candidate.valid)
        self.assertEqual(candidate_record['budget_seconds'], 2)
        self.assertTrue(reference.valid and reference.refuted)
        self.assertEqual(reference.timeout_seconds, 2)
        self.assertEqual(reference_record['budget_seconds'], 12)

    def test_current_operational_limits_and_hygiene(self):
        policy = json.loads((TASK / 'policy.json').read_text())
        self.assertEqual(policy['quota']['eval_timeout_seconds'], 330)
        self.assertEqual(policy['quota']['wall_time_seconds'], 18000)
        self.assertEqual(policy['network']['audit']['max_bundle_bytes'], 34359738368)
        self.assertEqual(policy['network']['audit']['max_model_sessions'], 1024)
        self.assertEqual(policy['metadata']['runtime']['submission_kind'], 'patch')
        for name in ('DESIGN.md', 'NOTES.md', 'REVIEW_DOSSIER.md', 'test_evaluator.py'):
            self.assertFalse((TASK / 'judge' / name).exists())
