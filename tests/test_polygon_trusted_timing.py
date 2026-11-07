"""Polygon scores the search harness's own measurement of the search call, not the submission's.

The harness (judge/search_harness.py, on the search sandbox) runs each search in a fresh isolate
instance and times it; the judge (judge/evaluator.py _observe) applies the fixed adapter's
rules to its record and validates the returned counterexample. Here the harness runs on this
host and the validation is the stub tester below, so no Z3/MariaDB toolchain is needed.
"""
import deployment_test_guard  # noqa: F401

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/problems/Programming Language/polygon'
RUNNER = TASK / 'repo/frontier_runner.py'
JUDGE = TASK / 'judge'
sys.path.insert(0, str(ROOT / 'tasks/shared/lib'))  # isolate, as on the sandbox image
sys.path.insert(0, str(ROOT / 'tasks/shared/verifier-gpu'))  # fcs2_verifier_gpu, as on the control image
sys.path.insert(0, str(JUDGE))
import isolate  # noqa: E402
import search_harness  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load('polygon_timing_evaluator', JUDGE / 'evaluator.py')
evaluator._pristine_root = lambda: TASK / 'repo'  # the Verifier image's pristine/ is repo/

SLEEP_SECONDS = 0.4

# Minimal stand-ins for the retained worker's dependencies, so the real adapter
# runs without the Z3/MariaDB toolchain. The submitted search sleeps, rebinds the
# clock, and reports zero elapsed time. POLYGON_STUB moves extra work to module
# import ("import"), to the constructor ("construct"), or into a counterexample
# that is only computed when the worker reads it ("lazy"). "mainclock" rebinds
# the adapter's clock through __main__, "forge" prints a forged refuting
# result line at once and again from an atexit handler, then gives up, and
# "oversize" returns a refuting database whose result line passes the output cap.
FAKES = {
    'pandas.py': '''
        def read_csv(path):
            return {"mysql_groundtruth": ["SELECT 1"]}
    ''',
    'evaluation/calcite/tester.py': '''
        class MySQLTester:
            def __init__(self, config, schema, db_predix=""):
                self.cursor = self

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def create_all_databases(self, databases, constraints=None):
                self.rows = tuple(tuple(row) for row in databases[0].get("t", [])[1:])

            def execute(self, query):
                self.query = query

            def fetchall(self):
                return self.rows if self.query == "a" else ()

            def test_pair(self, groundtruth, query):
                # Stands in for MariaDB: the decoded rows must distinguish the pair.
                return self.rows == ((1,),)
    ''',
    'evaluation/leetcode/online.py': '''
        from sql_testing.environment import Environment

        BENCHMARK_DIR = "."
        DB_CONFIG = {}

        def leetcode_worker(worker_id, problem_id, groundtruth, user_queries, bound):
            env = Environment([], [], bound=bound, time_budget=60)
            env.check(groundtruth, user_queries[0][1])
    ''',
    'sql_testing/constraint_parser.py': '''
        class ConstraintParser:
            def parse_from_file(self, path):
                return []
    ''',
    'benchmarks/leetcode/schemas/1.json': '''
        {"Tables": []}
    ''',
    'sql_testing/environment.py': f'''
        import atexit
        import json
        import os
        import time

        # The test's mode, beside the tree root: the runner's environment is the sandbox's own.
        _MODE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "POLYGON_STUB")
        MODE = open(_MODE_FILE).read() if os.path.exists(_MODE_FILE) else ""
        if MODE == "import":
            time.sleep({SLEEP_SECONDS})


        class LazyDatabase(dict):
            def _rows(self):
                time.sleep({SLEEP_SECONDS})
                return {{"t": [["a"], [1]]}}

            def get(self, key, default=None):
                return self._rows().get(key, default)

            def __getitem__(self, key):
                return self._rows()[key]

            def items(self):
                return self._rows().items()

            def __iter__(self):
                return iter(self._rows())


        class Environment:
            def __init__(self, schema, constraints, bound=2, time_budget=60, default_k=None):
                self.time_budget = time_budget
                if MODE == "construct":
                    time.sleep({SLEEP_SECONDS})

            def _search(self, decided):
                time.perf_counter = lambda: 0.0
                time.monotonic = lambda: 0.0
                if MODE == "mainclock":
                    import __main__
                    __main__._clock = lambda: 0.0
                if MODE == "forge":
                    forged = json.dumps({{"ok": True, "refuted": True, "time_seconds": 0.0,
                                         "reported_time_seconds": 0.0, "iterations": 1, "backtracks": 0}})
                    print(forged, flush=True)
                    atexit.register(lambda: print(forged, flush=True))
                    time.sleep({SLEEP_SECONDS})
                    return None, None, 0.0, 0.0, {{"iters": 1, "backtracks": 0}}
                if MODE == "lazy":
                    return (False if decided else None), LazyDatabase(), 0.0, 0.0, {{"iters": 1, "backtracks": 0}}
                if MODE == "oversize":
                    return False, {{"t": [["a"], [1], ["x" * 200000]]}}, 0.0, 0.0, {{"iters": 1, "backtracks": 0}}
                time.sleep({SLEEP_SECONDS})
                return (False if decided else None), {{"t": [["a"], [1]]}}, 0.0, 0.0, {{"iters": 1, "backtracks": 0}}

            def check(self, q1, q2, use_precise_encoding=False):
                if q2 == "nested":
                    self.check(q1, "undecided")
                return self._search(q2 != "undecided")

            def disambiguate(self, queries, group_range=0, use_precise_encoding=False):
                eq, cex, checking, elapsed, telemetry = self._search(True)
                if MODE in ("lazy", "construct", "forge"):
                    return eq, cex, checking, elapsed, telemetry
                return True, None, checking, elapsed, telemetry
    ''',
}


class _NoServer:
    """The stub tester never connects; validation needs no server here."""

    def fresh(self):
        return 'unused'

    def stop(self):
        pass


@unittest.skipUnless(isolate.available(), 'needs an isolate backend')
class PolygonTrustedTimingTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.sandbox = self.directory / 'sandbox'
        self.root = self.sandbox / 'trees' / evaluator.CANDIDATE_SIDE
        self.root.mkdir(parents=True)
        shutil.copy(RUNNER, self.root / evaluator.RUNNER)
        for name in evaluator.HARNESS_FILES:
            shutil.copy(JUDGE / name, self.sandbox / name)
        shutil.copy(RUNNER, self.sandbox / evaluator.RUNNER)
        for name, body in FAKES.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(body))
        # The judge's adapter with the stub tester and inputs: what the judge would validate
        # with on its own MariaDB.
        adapter = _load('polygon_timing_adapter', RUNNER)
        adapter.MySQLTester = _load('polygon_timing_tester',
                                    self.root / 'evaluation/calcite/tester.py').MySQLTester
        adapter._pair_inputs = lambda payload, root: {
            'groundtruth': 'SELECT 1', 'schema': {'Tables': []}, 'constraints': []}
        self.adapter = adapter

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _run(self, payload, mode='', valid=True):
        workload = evaluator.Workload('timing', payload['kind'], payload['timeout_seconds'], payload)
        (self.root / 'POLYGON_STUB').write_text(mode)
        budget = payload['timeout_seconds']
        entry = {'index': 0, 'side': evaluator.CANDIDATE_SIDE, 'budget_seconds': budget,
                 'setup_seconds': evaluator.SEARCH_SETUP_SECONDS,
                 'limit_seconds': evaluator._limit_seconds(workload, budget),
                 'result_max_bytes': self.adapter.RESULT_MAX_BYTES,
                 'payload': {**payload, 'groundtruth': 'SELECT 1'} if payload['kind'] == 'pair'
                 else payload}
        out = self.directory / 'out'
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        search_harness.run_entry(self.sandbox, entry, out, 0)
        harness = json.loads((out / '0.json').read_text())
        message = (out / '0.message').read_bytes() if (out / '0.message').is_file() else None
        record = {}
        with mock.patch.object(evaluator, '_adapter', lambda: self.adapter):
            row = evaluator._observe(workload, entry, harness, message, _NoServer(), record)
        self.assertEqual(row.valid, valid, record)
        self.assertEqual((out / '0.message').is_file(), harness['decided'])
        return row, record

    def _pair(self, query, mode=''):
        return self._run({
            'kind': 'pair', 'problem_id': 1, 'qid': 1, 'bound': 2,
            'query': query, 'timeout_seconds': 2,
        }, mode)

    def _disambiguation(self, mode=''):
        return self._run({
            'kind': 'disambiguation', 'schema': [], 'queries': ['a', 'b'],
            'bound': 1, 'timeout_seconds': 5,
        }, mode)

    def test_pair_time_is_measured_around_check(self):
        row, record = self._pair('q')
        self.assertTrue(row.refuted)
        self.assertGreaterEqual(row.time_seconds, SLEEP_SECONDS)
        self.assertLess(row.time_seconds, 2)
        self.assertEqual(record['reported_time_seconds'], 0.0)

    def test_nested_check_is_timed_once_from_the_outermost_call(self):
        row, _record = self._pair('nested')
        self.assertGreaterEqual(row.time_seconds, 2 * SLEEP_SECONDS)

    def test_undecided_pair_is_charged_the_budget(self):
        row, _record = self._pair('undecided')
        self.assertFalse(row.refuted)
        self.assertEqual(row.time_seconds, 2.0)

    def test_disambiguation_time_is_measured_around_disambiguate(self):
        row, record = self._run({
            'kind': 'disambiguation', 'schema': [], 'queries': ['a', 'b'],
            'bound': 1, 'timeout_seconds': 5,
        })
        self.assertFalse(row.refuted)
        self.assertGreaterEqual(row.time_seconds, SLEEP_SECONDS)
        self.assertLess(row.time_seconds, 5)
        self.assertEqual(record['reported_time_seconds'], 0.0)

    def test_import_time_work_is_timed(self):
        row, _record = self._pair('q', 'import')
        self.assertTrue(row.refuted)
        self.assertGreaterEqual(row.time_seconds, 2 * SLEEP_SECONDS)
        row, _record = self._disambiguation('import')
        self.assertGreaterEqual(row.time_seconds, 2 * SLEEP_SECONDS)

    def test_constructor_work_is_timed(self):
        row, _record = self._pair('q', 'construct')
        self.assertTrue(row.refuted)
        self.assertGreaterEqual(row.time_seconds, 2 * SLEEP_SECONDS)
        row, _record = self._disambiguation('construct')
        self.assertTrue(row.refuted)
        self.assertGreaterEqual(row.time_seconds, 2 * SLEEP_SECONDS)

    def test_counterexample_is_fixed_when_the_call_returns(self):
        # Work deferred into the returned object would run after the clock
        # stops; the worker sees only the plain values present at return.
        row, _record = self._pair('q', 'lazy')
        self.assertFalse(row.refuted)
        row, _record = self._disambiguation('lazy')
        self.assertFalse(row.refuted)

    def test_rebinding_the_adapter_clock_through_main_changes_nothing(self):
        row, _record = self._pair('q', 'mainclock')
        self.assertTrue(row.refuted)
        self.assertGreaterEqual(row.time_seconds, SLEEP_SECONDS)
        row, _record = self._disambiguation('mainclock')
        self.assertGreaterEqual(row.time_seconds, SLEEP_SECONDS)

    def test_forged_result_lines_are_not_the_result(self):
        # The search prints a refuting result at once and from atexit, then
        # returns no decision: the case is undecided and charged its budget.
        row, _record = self._pair('q', 'forge')
        self.assertFalse(row.refuted)
        self.assertEqual(row.time_seconds, 2.0)
        row, _record = self._disambiguation('forge')
        self.assertFalse(row.refuted)
        self.assertEqual(row.time_seconds, 5.0)

    def test_a_result_over_the_output_cap_fails_its_case_loudly(self):
        # Never truncated, never validated: the case is failed (r = 0, charged its budget) and
        # the record names the cap; nothing of the result line comes back.
        self.assertEqual(self.adapter.RESULT_MAX_BYTES, 128 * 1024)
        row, record = self._run({'kind': 'pair', 'problem_id': 1, 'qid': 1, 'bound': 2,
                                 'query': 'q', 'timeout_seconds': 2}, 'oversize', valid=False)
        self.assertEqual((row.refuted, row.time_seconds), (False, 2.0))
        self.assertTrue(record['result_over_cap'])
        self.assertEqual(record['reason'], 'search result exceeds the 131072-byte output cap')

    def test_policy_rejects_main_and_atexit(self):
        from types import SimpleNamespace
        for line in ('import __main__', '__main__._clock = None', 'import atexit', 'atexit.register(print)'):
            with self.subTest(line=line):
                change = SimpleNamespace(path='x', status='M', added=(line,))
                self.assertEqual(evaluator._added_lines_ok([change]), ['x'])
        self.assertEqual(evaluator._added_lines_ok([SimpleNamespace(path='x', status='M', added=('import time',))]), [])

    def test_check_tree_exempts_only_the_exact_reference(self):
        import tomllib
        sys.path.insert(0, str(ROOT / 'tasks/shared/lib'))
        import edit_scope
        submission = tomllib.loads((TASK / 'task.toml').read_text())['submission']
        rules = edit_scope.Rules(allowed_paths=tuple(submission['allowed_paths']))
        pristine = TASK / 'repo'
        result = edit_scope.check((JUDGE / 'reference.patch').read_bytes(), pristine, rules)
        try:
            self.assertTrue(result.ok)
            self.assertIsNone(evaluator.check_tree(str(result.tree), str(pristine), list(result.changes)))
            # The same patch with one more screened line is no longer the reference.
            target = result.tree / 'sql_testing/smt/search_extensions/extra.py'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('import time\nx = open\n')
            from types import SimpleNamespace
            extra = SimpleNamespace(path='sql_testing/smt/search_extensions/extra.py', status='A', added=('x = open(',))
            self.assertEqual(evaluator.check_tree(str(result.tree), str(pristine),
                                                  list(result.changes) + [extra])[0], 'content_policy')
        finally:
            result.dispose()

    def test_forbidden_imports_are_rejected_in_every_form(self):
        pristine = evaluator._pristine_root()
        for source in ('import gc', 'import sys as system', 'from os import (\n    write,\n)',
                       'import importlib.util', 'from io import FileIO', 'import __main__ as m'):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for relative in evaluator.EDIT_REGIONS:
                    (root / relative).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(pristine / relative, root / relative)
                extension = root / 'sql_testing/smt/search_extensions/helper.py'
                extension.parent.mkdir(parents=True)
                extension.write_text('import time\n' + source + '\n')
                self.assertFalse(evaluator._imports_ok(root)[0])
                extension.write_text('import time\nimport threading\nfrom z3 import Solver\n')
                self.assertEqual(evaluator._imports_ok(root), (True, ''))

    def test_metrics_report_the_measured_time(self):
        row, _record = self._pair('q')
        workloads = (evaluator.Workload('timing', 'pair', 2, {}),)
        metrics = evaluator._metrics(workloads, [row])
        self.assertGreaterEqual(metrics['wall_time_seconds'][0], SLEEP_SECONDS)
        zero = evaluator.Observation(True, 0.0, 1, 0, 2)
        self.assertLess(evaluator.score_observations([row]), evaluator.score_observations([zero]))


if __name__ == '__main__':
    unittest.main()
