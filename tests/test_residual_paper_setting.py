"""Residual State Advancement keeps the paper's in-database setting (#351).

The trusted executor caps rows returned to Python and rejects statements that
change database settings; the evaluator screens the patch for routes around it
and reports violations as invalid submissions with closed reasons. These tests
need neither DuckDB nor the scientific stack; the workload tests at the end run
only where DuckDB, pandas and scikit-learn are installed.
"""

# Install isolation before runtime imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import ast
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK = Path(
    os.environ.get(
        "RESIDUAL_PAPER_SETTING_EXPORT",
        str(
            ROOT
            / "tasks/frontier-cs-2.0-demo/problems/Database/residual_state_advancement"
        ),
    )
)
REPO = TASK / "agent-base/frontier/task/repo"
EXECUTOR = "src/joinboost/executor.py"
EDITABLE = [
    "src/joinboost/app.py",
    "src/joinboost/semiring.py",
    "src/joinboost/aggregator.py",
    "src/joinboost/cjt.py",
    "src/joinboost/joingraph.py",
]
FACT_SHA256 = {
    "sales-dev-0.parquet": "cbb42ddc6b237eb4332431e7ce81c62a10f5b709bce1f4f28dfe2c62fee32dc5",
    "sales-dev-1.parquet": "77572db9fe8163196a3cf0df3f70a77c55949f9eaf5eff9d0ad28a0f3ec35dd0",
    "sales-dev-2.parquet": "29df15bc172aff64cd122ac401e6311bba36e23ad25c730471cc4614e5a86ba3",
    "sales-dev-3.parquet": "8b38d9a0ebe580cd3e8f5530f2f858ff8a819459ccb23cf7fd34394fdb12eaef",
    "sales-final-0.parquet": "2d3b2de7837d90ae2cdf5ae894e006088157a21ee7bc4381144a4433a71d9719",
    "sales-final-1.parquet": "ee4bed24691e3669836abd5f8fd552670890f3a7fa6f07c8196ccf79dfb5469c",
    "sales-final-2.parquet": "463b4c1d3ea8911aa5fd6cc3626cb41a04cac950c710f6c257cc0394ad40ea4f",
    "sales-final-3.parquet": "a034320758b2481b46488facfc8db2ff7c8646e53692f84467c48404527998af",
}
SCIENTIFIC_STACK = all(
    importlib.util.find_spec(name) for name in ("duckdb", "pandas", "sklearn", "yaml")
)


LOCAL_MODULES = (
    "joinboost",
    "benchmark",
    "submission_patch",
    "_local_submission_patch",
    "residual_paper_setting_evaluator",
)


def _restore_modules(case, stubs=()):
    """Drop task-local modules (and stand-in stubs) again; keep real libraries."""
    path = list(sys.path)

    def restore():
        sys.path[:] = path
        for name in list(sys.modules):
            if name.split(".")[0] in LOCAL_MODULES or name in stubs:
                del sys.modules[name]

    restore()
    sys.path[:] = path
    case.addCleanup(restore)


def _stage(case):
    """A judge-shaped task directory outside any Git checkout."""
    work = tempfile.mkdtemp(prefix="residual-paper-setting-", dir="/tmp")
    case.addCleanup(shutil.rmtree, work, True)
    task = Path(work) / "task"
    shutil.copytree(TASK / "judge", task)
    shutil.copytree(REPO, task / "harbor" / "app")
    return task


def _load_evaluator(case, task):
    _restore_modules(case)
    sys.path.insert(0, str(task))
    spec = importlib.util.spec_from_file_location(
        "residual_paper_setting_evaluator", task / "evaluator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_adding(lines, path="src/joinboost/app.py", extra=None, rename=None):
    """A real git patch that inserts ``lines`` into GradientBoosting._update_error.

    With ``rename`` the reference body becomes that method, and ``_update_error``
    consists of ``lines`` only; ``extra`` is appended to the class.
    """
    work = Path(tempfile.mkdtemp(prefix="residual-patch-", dir="/tmp"))
    try:
        tree = work / "tree"
        shutil.copytree(REPO, tree)
        git = ["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run([*git, "init", "-q"], check=True)
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", "base"], check=True)
        subprocess.run(
            [*git, "apply", "--whitespace=nowarn", str(TASK / "judge/reference.patch")],
            check=True,
        )
        target = tree / path
        text = target.read_text()
        if path.endswith("app.py"):
            anchor = "        target_relation = self.cjt.target_relation\n"
            assert text.count(anchor) == 1
            body = "".join(f"        {line}\n" for line in lines)
            if rename is None:
                text = text.replace(anchor, anchor + body)
            else:
                head = "    def _update_error(self, leaf_nodes, learning_rate):\n"
                assert text.count(head) == 1
                first, second = text.rsplit(head, 1)
                end = second.index("        return self.cjt\n") + len(
                    "        return self.cjt\n"
                )
                text = (
                    first
                    + head
                    + body
                    + "\n"
                    + head.replace("_update_error", rename)
                    + second[:end]
                    + "\n"
                    + (extra or "")
                    + second[end:]
                )
        else:
            text += "".join(f"{line}\n" for line in lines)
        target.write_text(text)
        return subprocess.run(
            [*git, "diff", "--no-color", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        shutil.rmtree(work, ignore_errors=True)


class FakeConnection:
    """Stands in for DuckDB: rows plus a DB-API description of column types."""

    def __init__(self):
        self.executed = []
        self.result = []
        self.types = None
        self.error = None

    @property
    def description(self):
        if self.types is None:
            width = len(self.result[0]) if self.result else 1
            return [("c", "BIGINT")] * width
        return [("c", kind) for kind in self.types]

    def execute(self, sql):
        if self.error is not None:
            raise self.error
        self.executed.append(sql)

    def fetchmany(self, size):
        return self.result[:size]


class TrustedExecutor(unittest.TestCase):
    def setUp(self):
        stubs = [n for n in ("pandas", "numpy") if importlib.util.find_spec(n) is None]
        _restore_modules(self, stubs)
        for name in stubs:
            sys.modules[name] = types.ModuleType(name)
        sys.path.insert(0, str(REPO / "src"))
        self.module = importlib.import_module("joinboost.executor")
        self.raw = FakeConnection()
        self.executor = self.module.DuckdbExecutor(self.raw)
        self.guard = self.executor.conn

    def test_measured_limits(self):
        # Measured on the eight development workloads (reference, per-leaf
        # baseline, three replayed in-database designs): at most 6 rows, 6
        # columns, 36 values and 141 bytes per query, 12 characters per string;
        # 3,043 rows, 8,666 values and 66,332 bytes per fit. The smallest fact
        # table has 1,027,072 rows.
        m = self.module
        limits = {
            "MAX_ROWS_PER_QUERY": (10_000, 6),
            "MAX_ROWS_PER_SESSION": (250_000, 3_043),
            "MAX_COLUMNS_PER_QUERY": (32, 6),
            "MAX_CELLS_PER_QUERY": (40_000, 36),
            "MAX_CELLS_PER_SESSION": (250_000, 8_666),
            "MAX_BYTES_PER_QUERY": (200_000, 141),
            "MAX_BYTES_PER_SESSION": (1_000_000, 66_332),
            "MAX_TEXT_LENGTH": (256, 12),
        }
        for name, (limit, measured) in limits.items():
            self.assertEqual(getattr(m, name), limit, name)
            self.assertGreaterEqual(limit, 5 * measured, name)
        # Far below fact scale: fewer values than a quarter of the smallest
        # fact table's rows, and under a tenth of its 14 MB of key and residual.
        self.assertLessEqual(m.MAX_CELLS_PER_SESSION * 4, 1_027_072)
        self.assertLessEqual(m.MAX_BYTES_PER_SESSION * 10, 1_027_072 * 14)

    def _rejected(self, types, rows, sql="SELECT 1"):
        raw = FakeConnection()
        raw.types, raw.result = types, rows
        executor = self.module.DuckdbExecutor(raw)
        with self.assertRaises(self.module.ExecutorPolicyError):
            executor._execute_query(sql)
        self.assertEqual(executor.conn.stats()["violation"], "data_cap")
        return executor.conn.stats()

    def test_nested_and_binary_result_types_are_refused_before_fetching(self):
        big = list(range(250_001))
        for kind, value in (
            ("BIGINT[]", big),  # list(), array_agg: the reviewer's query shape
            ("DOUBLE[3]", [1.0, 2.0, 3.0]),  # fixed-size ARRAY
            ("STRUCT(a BIGINT, b DOUBLE)", {"a": 1, "b": 2.0}),
            ("MAP(BIGINT, UBIGINT)", {1: 2}),  # histogram, map_from_entries
            ("STRUCT(a INTEGER[])[]", [{"a": [1, 2]}]),  # nested combination
            ("UNION(k INTEGER)", 1),
            ("BLOB", b"x" * 1000),  # encode
            ("BIT", "1010"),  # bitstring_agg
            ("JSON", "[1,2]"),
            ("ENUM('a', 'b')", "a"),
            ("UUID", "u"),
            ("DATE", "d"),
        ):
            with self.subTest(kind=kind):
                raw = FakeConnection()
                raw.types, raw.result = [kind], [(value,)]
                fetched = []
                raw.fetchmany = lambda size, fetched=fetched: fetched.append(size)
                executor = self.module.DuckdbExecutor(raw)
                with self.assertRaises(self.module.ExecutorPolicyError):
                    executor._execute_query("SELECT 1")
                self.assertEqual(fetched, [])
                self.assertEqual(executor.conn.stats()["violation"], "data_cap")

    def test_unbounded_values_in_allowed_types_are_refused(self):
        # A value that is not a scalar even though the column claims to be one.
        self._rejected(["BIGINT"], [([1, 2, 3],)])
        self._rejected(["VARCHAR"], [(b"bytes",)])
        # string_agg, group_concat, repeat, to_json, CAST(list AS VARCHAR), format.
        limit = self.module.MAX_TEXT_LENGTH
        self._rejected(["VARCHAR"], [("x" * (limit + 1),)])
        raw = FakeConnection()
        raw.types, raw.result = ["VARCHAR"], [("x" * limit,)]
        executor = self.module.DuckdbExecutor(raw)
        self.assertEqual(executor._execute_query("SELECT 1"), [("x" * limit,)])
        self.assertEqual(executor.conn.stats()["bytes"], limit)
        self.assertEqual(executor.conn.stats()["max_text"], limit)

    def test_wide_results_are_refused(self):
        columns = self.module.MAX_COLUMNS_PER_QUERY
        raw = FakeConnection()
        raw.result = [tuple(range(columns))]
        executor = self.module.DuckdbExecutor(raw)
        self.assertEqual(len(executor._execute_query("SELECT 1")[0]), columns)
        self._rejected(["BIGINT"] * (columns + 1), [tuple(range(columns + 1))])

    def test_cells_and_bytes_are_capped_not_only_rows(self):
        m = self.module
        # Fewer rows than the row cap, more values than the cell cap.
        width = 8
        rows = m.MAX_CELLS_PER_QUERY // width + 1
        self.assertLess(rows, m.MAX_ROWS_PER_QUERY)
        self._rejected(["BIGINT"] * width, [tuple(range(width))] * rows)
        # Bit-packing into wide numbers is charged by size: 16 bytes per value.
        packed = (1 << 127) - 1
        self.assertEqual(m._value_bytes(packed), 16)
        self.assertEqual(m._value_bytes(1), 8)
        self.assertEqual(m._value_bytes(1.5), 8)
        self.assertEqual(m._value_bytes(None), 1)
        rows = m.MAX_BYTES_PER_QUERY // 16 + 1
        self.assertLess(rows, m.MAX_ROWS_PER_QUERY * 2)
        stats = self._rejected(
            ["HUGEINT", "HUGEINT"], [(packed, packed)] * (rows // 2 + 1)
        )
        self.assertGreater(stats["max_bytes"], m.MAX_BYTES_PER_QUERY)

    def test_session_budget_stops_paging_of_packed_values(self):
        m = self.module
        packed = (1 << 127) - 1
        self.raw.types = ["HUGEINT"]
        self.raw.result = [(packed,)] * 9_000
        returned = 0
        with self.assertRaises(m.ExecutorPolicyError):
            for _ in range(1_000):
                returned += len(self.executor._execute_query("SELECT 1"))
        self.assertEqual(self.guard.stats()["violation"], "data_cap")
        self.assertLessEqual(returned * 16, m.MAX_BYTES_PER_SESSION)

    def test_error_messages_are_bounded_charged_and_detached(self):
        m = self.module
        self.raw.error = ValueError("Conversion Error: " + "9" * 100_000)
        with self.assertRaises(m.ExecutorException) as caught:
            self.executor._execute_query("SELECT error(string_agg(s, ','))")
        self.assertNotIsInstance(caught.exception, m.ExecutorPolicyError)
        self.assertLessEqual(len(str(caught.exception)), m.MAX_TEXT_LENGTH)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(self.guard.stats()["bytes"], m.MAX_TEXT_LENGTH)
        with self.assertRaises(m.ExecutorPolicyError):
            for _ in range(m.MAX_BYTES_PER_SESSION // m.MAX_TEXT_LENGTH + 1):
                try:
                    self.executor._execute_query("SELECT error('x')")
                except m.ExecutorPolicyError:
                    raise
                except m.ExecutorException:
                    pass
        self.assertEqual(self.guard.stats()["violation"], "data_cap")

    def test_only_execute_and_capped_fetch_are_reachable(self):
        self.assertIsNot(self.guard, self.raw)
        for name in ("fetchnumpy", "df", "arrow", "fetchmany", "sql", "cursor"):
            self.assertFalse(hasattr(self.guard, name), name)
        self.assertFalse(hasattr(self.guard, "__dict__"))
        self.assertNotIn(self.raw, vars(self.executor).values())

    def test_row_cap_per_query(self):
        cap = self.module.MAX_ROWS_PER_QUERY
        self.raw.result = [(i,) for i in range(cap)]
        self.assertEqual(len(self.executor._execute_query("SELECT 1")), cap)
        self.assertIsNone(self.guard.stats()["violation"])
        self.raw.result = [(i,) for i in range(cap + 1)]
        with self.assertRaises(self.module.ExecutorPolicyError):
            self.executor._execute_query("SELECT 1")
        self.assertEqual(self.guard.stats()["violation"], "data_cap")
        self.assertEqual(self.guard.stats()["max_rows"], cap + 1)

    def test_row_budget_per_session_stops_paging(self):
        m = self.module
        self.raw.result = [(None,)] * m.MAX_ROWS_PER_QUERY  # one byte per value
        for _ in range(m.MAX_ROWS_PER_SESSION // m.MAX_ROWS_PER_QUERY):
            self.executor._execute_query("SELECT * FROM sales LIMIT 10000 OFFSET 0")
        self.assertIsNone(self.guard.stats()["violation"])
        with self.assertRaises(m.ExecutorPolicyError):
            self.executor._execute_query("SELECT * FROM sales LIMIT 10000 OFFSET 0")
        self.assertEqual(self.guard.stats()["violation"], "data_cap")

    def test_setting_statements_are_rejected_before_execution(self):
        for sql in (
            "SET threads=1",
            "set memory_limit='64GB'",
            "RESET threads",
            "PRAGMA threads=8",
            "PRAGMA enable_profiling",
            "SELECT 1; SET threads=1",
            "/* tune */ SET threads=1",
            "-- tune\nSET threads=1",
            "ATTACH 'x.db' AS x",
            "COPY sales TO 'sales.csv'",
            "INSTALL httpfs",
            "LOAD httpfs",
            "CALL pragma_version()",
            "EXPORT DATABASE 'd'",
        ):
            with self.subTest(sql=sql):
                raw = FakeConnection()
                executor = self.module.DuckdbExecutor(raw)
                with self.assertRaises(self.module.ExecutorPolicyError):
                    executor._execute_query(sql)
                self.assertEqual(raw.executed, [])
                self.assertEqual(executor.conn.stats()["violation"], "settings")

    def test_training_statements_are_accepted(self):
        for sql in (
            "SELECT SUM(s) FROM sales",
            "(SELECT 1)",
            "WITH t AS (SELECT 1) SELECT * FROM t",
            "CREATE OR REPLACE TABLE joinboost_tmp_1 AS SELECT * FROM sales",
            "UPDATE joinboost_tmp_1 SET s=s-1 \n",
            "DROP TABLE IF EXISTS joinboost_tmp_1;\n",
            "ALTER TABLE sales RENAME COLUMN a TO b;",
            "PRAGMA table_info(sales)",
        ):
            with self.subTest(sql=sql):
                self.executor._execute_query(sql)
        self.assertIsNone(self.guard.stats()["violation"])
        self.assertEqual(self.guard.stats()["queries"], 8)

    def test_violation_survives_a_swallowed_exception(self):
        try:
            self.executor._execute_query("SET threads=1")
        except Exception:
            pass
        self.executor._execute_query("SELECT 1")
        self.assertEqual(self.guard.stats()["violation"], "settings")


class EditableSurface(unittest.TestCase):
    def test_executor_is_not_editable_anywhere(self):
        policy = json.loads((TASK / "policy.json").read_text())
        self.assertEqual(policy["metadata"]["runtime"]["allowed_paths"], EDITABLE)
        config = (TASK / "judge/config.yaml").read_text()
        self.assertNotIn("executor.py", config)
        self.assertIn("max_changed_files: 5", config)
        for path in EDITABLE:
            self.assertIn("    - " + path + "\n", config)
        self.assertNotIn("executor.py", (REPO / "make_submission.sh").read_text())
        self.assertNotIn(
            "executor.py`", (TASK / "agent-base/frontier/task/README.md").read_text()
        )
        statement = (TASK / "judge/readme").read_text()
        listed = statement.split("The only files that may change are:")[1].split("```")[
            1
        ]
        self.assertEqual(listed.split(), EDITABLE)
        self.assertEqual(
            statement,
            (TASK / "agent-base/frontier/task/guide/README.md").read_text(),
        )

    def test_statement_states_the_enforced_limits(self):
        statement = (TASK / "judge/readme").read_text()
        for fragment in (
            "MAX_ROWS_PER_QUERY = 10,000",
            "MAX_ROWS_PER_SESSION = 250,000",
            "MAX_COLUMNS_PER_QUERY = 32",
            "MAX_CELLS_PER_SESSION = 250,000",
            "MAX_BYTES_PER_SESSION = 1,000,000",
            "MAX_TEXT_LENGTH = 256",
            "`LIST`, `ARRAY`, `STRUCT`, `MAP`, `UNION`, `BLOB`, `BIT`",
            "`submission returned more data to Python than the trusted executor allows`",
            "`threads=2`",
            "`memory_limit='10GB'`",
            "Joins and aggregations run in the DBMS.",
            "no fact row is repeated",
        ):
            self.assertIn(fragment, statement)
        self.assertNotIn("with replacement", statement)

    def test_loader_copy_is_the_judge_loader(self):
        def parts(path):
            tree = ast.parse(path.read_text())
            found = {}
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in (
                    "_lookup",
                    "load_sales",
                ):
                    found[node.name] = ast.dump(node)
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "FULL_FEATURES"
                    for t in node.targets
                ):
                    found["FULL_FEATURES"] = ast.dump(node)
            return found

        judge = parts(TASK / "judge/benchmark.py")
        public = parts(REPO / "data/favorita/load_sales.py")
        self.assertEqual(set(judge), {"_lookup", "load_sales", "FULL_FEATURES"})
        self.assertEqual(judge, public)

    def test_guard_thresholds_are_in_one_place_and_stated(self):
        tree = ast.parse((TASK / "judge/benchmark.py").read_text())
        constants = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id
            in ("FACT_SCALE_FRACTION", "FACT_EXTRA_COLUMNS", "STATE_TOLERANCE")
        }
        self.assertEqual(
            constants,
            {
                "FACT_SCALE_FRACTION": 0.10,
                "FACT_EXTRA_COLUMNS": 4,
                "STATE_TOLERANCE": 1.0e-6,
            },
        )
        statement = (TASK / "judge/readme").read_text()
        for fragment in (
            "at least 10% of the fact table's rows",
            "more than 4 columns beyond the fact table's own 5",
            "`submission materialised the join into a fact-scale relation`",
            "includes one trusted full read of the updated target",
        ):
            self.assertIn(fragment, statement)
        spec = json.loads((TASK / "judge/beat_spec.json").read_text())
        self.assertIn("trusted full read", spec["metrics"][1]["means"])
        self.assertIn("non-materialised-join rule", spec["beats_reference"])

    def test_no_resampled_private_join_is_shipped(self):
        self.assertFalse((TASK / "judge/judge_data/favorita/train_small.csv").exists())

    def test_fact_slices_are_placed_by_suite(self):
        """Development rows are agent-visible; graded rows are judge-only."""
        public = sorted(p.name for p in (REPO / "data/favorita").glob("*.parquet"))
        private = sorted(p.name for p in (TASK / "judge").rglob("*.parquet"))
        self.assertEqual(public, [f"sales-dev-{i}.parquet" for i in range(4)])
        self.assertEqual(private, [f"sales-final-{i}.parquet" for i in range(4)])
        agent_tree = TASK / "agent-base"
        self.assertFalse(list(agent_tree.rglob("sales-final-*")))
        lock = json.loads((TASK / "upstream.lock.json").read_text())
        recorded = {row["path"]: row["sha256"] for row in lock["files"]}
        for name, digest in FACT_SHA256.items():
            root = (
                "judge/judge_data/favorita/"
                if "final" in name
                else "agent-base/frontier/task/repo/data/favorita/"
            )
            self.assertEqual(recorded[root + name], "sha256:" + digest, name)

    def test_statement_records_the_data_source(self):
        statement = (TASK / "judge/readme").read_text()
        for fragment in (
            "github.com/bgautijonsson/groceries",
            "f79ac95b1c61ca9767f4e7d379bc2229fd8938fc",
            "86276d82a921ac0415d95a776dcf5122b17ed03a58eb8d72d6f9f09b1eef02a5",
            "carries no licence statement",
            "2017-02-02 to 2017-04-30",
            "2017-05-01 to 2017-07-30",
            "(Apache-2.0)",
        ):
            self.assertIn(fragment, statement)


@unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is not installed")
class EvaluatorPolicy(unittest.TestCase):
    def setUp(self):
        self.task = _stage(self)
        self.evaluator = _load_evaluator(self, self.task)
        archive = tempfile.mkdtemp(prefix="residual-archive-", dir="/tmp")
        self.addCleanup(shutil.rmtree, archive, True)
        environment = patch.dict(os.environ, {"FRONTIER_PATCH_ARCHIVE_DIR": archive})
        environment.start()
        self.addCleanup(environment.stop)

    def _write(self, text):
        path = self.task.parent / "candidate.patch"
        path.write_text(text)
        return path

    def test_suites_use_disjoint_real_row_slices(self):
        e = self.evaluator
        self.assertEqual(
            [case["fact"] for case in e.DEV_SUITE],
            [f"sales-dev-{i}.parquet" for i in range(4)] * 2,
        )
        self.assertEqual(
            [case["fact"] for case in e.FINAL_SUITE],
            [f"sales-final-{i}.parquet" for i in range(4)] * 2,
        )
        profile = ("depth", "estimators", "learning_rate", "topology")
        self.assertEqual(
            [[case[k] for k in profile] for case in e.DEV_SUITE],
            [[case[k] for k in profile] for case in e.FINAL_SUITE],
        )
        executor = (REPO / EXECUTOR).read_text()
        self.assertIn("MAX_ROWS_PER_QUERY = 10_000", executor)
        for case in e.DEV_SUITE + e.FINAL_SUITE:
            self.assertNotIn("limit", case)
            self.assertGreaterEqual(case["rows"], 100 * 10_000)
            self.assertLessEqual(case["rows"], 2_200_000)

    def test_eight_workloads_spread_depth_rounds_and_topology(self):
        e = self.evaluator
        spec = json.loads((self.task / "beat_spec.json").read_text())
        for suite in (e.DEV_SUITE, e.FINAL_SUITE):
            self.assertEqual(len(suite), 8)
            self.assertEqual(spec["workload_count"], len(suite))
            # Every workload is a distinct (slice, profile); every slice twice.
            keys = [(c["fact"], c["depth"], c["estimators"]) for c in suite]
            self.assertEqual(len(set(keys)), 8)
            self.assertEqual({keys.count(k) for k in keys}, {1})
            facts = [c["fact"] for c in suite]
            self.assertEqual({facts.count(f) for f in facts}, {2})
            self.assertEqual({c["depth"] for c in suite}, {1, 2, 3, 4, 5})
            self.assertGreaterEqual(sum(c["estimators"] >= 20 for c in suite), 2)
            self.assertEqual(
                {c["topology"] for c in suite},
                {"core_four_table", "five_table_oil", "six_table"},
            )
            # Same slices and row counts as the four original workloads.
            self.assertEqual(
                [c["rows"] for c in suite[:4]], [c["rows"] for c in suite[4:]]
            )
        self.assertEqual(len(e.PAIR_ORDER), 4)
        config = (self.task / "config.yaml").read_text()
        self.assertIn("timeout_seconds: 7200", config)
        policy = json.loads((TASK / "policy.json").read_text())
        self.assertEqual(policy["quota"]["eval_timeout_seconds"], 7200)
        self.assertEqual(
            policy["quota"]["max_train_wall_seconds"],
            policy["quota"]["max_train_calls"] * 7200,
        )
        statement = (TASK / "judge/readme").read_text()
        self.assertIn("each contain eight deterministic cases", statement)
        self.assertIn("evaluation timeout is 7,200 seconds", statement)

    def test_bundled_patches_pass_the_static_screen(self):
        for name in ("reference.patch", "baseline_naive.patch"):
            self.assertEqual(
                self.evaluator._patch_policy(self.task / name), (True, "ok"), name
            )

    def test_static_screen_rejects_routes_around_the_executor(self):
        e = self.evaluator
        for line in (
            "rows = self.cjt.exe.conn.execute('select 1')",
            "import duckdb",
            "data = handle.fetchnumpy()",
            "data = handle.fetchall()",
            "frame = relation.df()",
            "table = relation.arrow()",
            "raw = guard.execute.__closure__[0].cell_contents",
            "raw = getattr(self.cjt.exe, 'conn')",
            "state = vars(self.cjt.exe)",
            "state = self.cjt.exe.__dict__",
        ):
            with self.subTest(line=line):
                path = self._write(_patch_adding([line]))
                self.assertEqual(e._patch_policy(path), (False, e.STATIC_BYPASS_REASON))
                self.assertEqual(
                    e.evaluate(str(path))[:3], (0.0, 0.0, e.STATIC_BYPASS_REASON)
                )

    def test_patch_touching_the_executor_is_outside_the_surface(self):
        path = self._write(_patch_adding(["# tuned"], EXECUTOR))
        score, unbounded, message, metrics = self.evaluator.evaluate(str(path))
        self.assertEqual((score, unbounded), (0.0, 0.0))
        self.assertIn("outside the editable", message)
        self.assertEqual(metrics["beats_reference"], 0)

    def test_executor_bytes_are_locked_in_the_patched_tree(self):
        e = self.evaluator
        tree = self.task.parent / "tree"
        shutil.copytree(self.task / "harbor" / "app", tree)
        self.assertEqual(e._structural_policy(tree), (True, "ok"))
        with (tree / EXECUTOR).open("a") as handle:
            handle.write("\n")
        self.assertEqual(
            e._structural_policy(tree),
            (False, "the trusted executor must remain unchanged"),
        )

    def test_worker_policy_exit_maps_to_closed_reasons(self):
        e = self.evaluator
        self.assertEqual(
            set(e.POLICY_REASONS),
            {"data_cap", "settings", "executor", "materialised_join"},
        )
        self.assertEqual(
            e.POLICY_REASONS["materialised_join"],
            "submission materialised the join into a fact-scale relation",
        )
        for code, expected in (
            ("data_cap", "data_cap"),
            ("settings", "settings"),
            ("executor", "executor"),
            ("materialised_join", "materialised_join"),
            ("anything else", "executor"),
        ):
            done = types.SimpleNamespace(
                returncode=e.POLICY_EXIT_CODE,
                stdout=json.dumps({"policy_violation": code}) + "\n",
            )
            with patch.object(e.subprocess, "run", return_value=done):
                with self.assertRaises(e.PolicyRejection) as caught:
                    e._run_once(Path("tree"), e.FAST_SUITE[0])
            self.assertEqual(caught.exception.code, expected)

    def _evaluate_with_violation(self, code, role):
        e = self.evaluator
        sample = {
            "elapsed_seconds": 1.0,
            "rows": 1000,
            "estimators": 2,
            "rmse_error": 0.0,
            "prediction_l1": 0.0,
            "residual_update_seconds": 0.1,
        }
        reference_tree = []

        def run_once(tree, case):
            if not reference_tree:
                reference_tree.append(str(tree))  # the untimed reference warm-up
            is_reference = str(tree) == reference_tree[0]
            if (role == "candidate") != is_reference:
                raise e.PolicyRejection(code)
            return dict(sample)

        candidate = self._write(_patch_adding(["# byte-different reference"]))
        with (
            patch.dict(os.environ, {"FRONTIER_EVALUATOR_FAST": "1"}),
            patch.object(e, "_prepare_oracle", return_value=None),
            patch.object(e, "_baseline_seconds", return_value=2.0),
            patch.object(e, "_run_once", side_effect=run_once),
        ):
            return e.evaluate(str(candidate))

    def test_candidate_violation_is_an_invalid_submission(self):
        e = self.evaluator
        for code in ("data_cap", "settings", "executor", "materialised_join"):
            with self.subTest(code=code):
                score, unbounded, message, metrics = self._evaluate_with_violation(
                    code, "candidate"
                )
                self.assertEqual((score, unbounded), (0.0, 0.0))
                self.assertEqual(message, e.POLICY_REASONS[code])
                self.assertEqual(metrics["beats_reference"], 0)

    def test_reference_violation_is_a_failed_verification(self):
        with self.assertRaises(self.evaluator.ReferenceVerificationError):
            self._evaluate_with_violation("data_cap", "reference")

    def test_l1_bound_is_per_row(self):
        e = self.evaluator
        self.assertEqual(e.PREDICTION_L1_TOLERANCE_PER_ROW, 1.0e-6)
        run = {
            "elapsed_seconds": 1.0,
            "rmse_error": 0.0,
            "prediction_l1": 1.9,
            "prediction_l1_per_row": 1.9 / 2_000_000,
        }
        self.assertGreater(e._score_run(run, 10.0)[0], 0.0)
        run["prediction_l1_per_row"] = 2.1 / 2_000_000
        self.assertEqual(e._score_run(run, 10.0), (0.0, 0.0))


@unittest.skipUnless(
    SCIENTIFIC_STACK, "DuckDB, pandas and scikit-learn are not installed"
)
class RealWorkload(unittest.TestCase):
    """End to end on the small local suite, with real DuckDB."""

    @classmethod
    def setUpClass(cls):
        cls.environment = patch.dict(os.environ, {"FRONTIER_EVALUATOR_FAST": "1"})
        cls.environment.start()

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()

    def setUp(self):
        self.task = _stage(self)
        self.evaluator = _load_evaluator(self, self.task)
        archive = tempfile.mkdtemp(prefix="residual-archive-", dir="/tmp")
        self.addCleanup(shutil.rmtree, archive, True)
        environment = patch.dict(os.environ, {"FRONTIER_PATCH_ARCHIVE_DIR": archive})
        environment.start()
        self.addCleanup(environment.stop)

    def _evaluate(self, lines, **options):
        path = self.task.parent / "candidate.patch"
        path.write_text(_patch_adding(lines, **options))
        return self.evaluator.evaluate(str(path))

    def test_in_database_solutions_pass(self):
        e = self.evaluator
        for name in ("reference.patch", "baseline_naive.patch"):
            score, _, message, metrics = e.evaluate(str(self.task / name))
            self.assertIn("correct ensemble", message, name)
            self.assertEqual(metrics["correct"], 1)
        score, _, message, metrics = self._evaluate(["# byte-different reference"])
        self.assertIn("correct ensemble", message)
        self.assertEqual(metrics["correct"], 1)

    def test_real_rows_are_unique_join_fully_and_suites_are_disjoint(self):
        import duckdb

        sys.path.insert(0, str(self.task))
        benchmark = importlib.import_module("benchmark")
        e = self.evaluator
        data = self.task / "harbor/app/data/favorita"
        private = self.task / "judge_data/favorita"
        con = duckdb.connect()
        for name in ("items", "transactions"):
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM '{data}/{name}.csv'")
        dates = {}
        for suite, root in ((e.DEV_SUITE, data), (e.FINAL_SUITE, private)):
            for case in suite:
                path = root / case["fact"]
                rows, keys, items, tids, low, high = con.execute(
                    "SELECT count(*), count(DISTINCT (s.item_nbr, s.tid)), count(i.item_nbr),"
                    " count(t.tid), min(t.date), max(t.date) FROM read_parquet(?) s"
                    " LEFT JOIN items i USING (item_nbr) LEFT JOIN transactions t USING (tid)",
                    [str(path)],
                ).fetchone()
                self.assertEqual((rows, keys, items, tids), (case["rows"],) * 4, path)
                dates.setdefault(case["fact"].split("-")[1], set()).update((low, high))
        # Date codes are days in order: 2017-02-02 is 1489, 2017-07-30 is 1667.
        self.assertEqual((min(dates["dev"]), max(dates["dev"])), (1489, 1576))
        self.assertEqual((min(dates["final"]), max(dates["final"])), (1577, 1667))
        keys = set()
        for fact in sorted({case["fact"] for case in e.DEV_SUITE}):
            found = con.execute(
                "SELECT item_nbr::BIGINT * 100000 + tid FROM read_parquet(?)",
                [str(data / fact)],
            ).fetchnumpy()
            fresh = set(next(iter(found.values())).tolist())
            self.assertFalse(keys & fresh)
            keys |= fresh
        sales, joined = benchmark.load_sales(data, data / "sales-dev-2.parquet", 50_000)
        self.assertEqual(
            list(sales.columns), ["item_nbr", "unit_sales", "onpromotion", "tid", "Y"]
        )
        self.assertEqual(set(sales["onpromotion"]), {2})
        self.assertEqual(len(joined), 50_000)

    def test_fetching_the_fact_table_is_invalid(self):
        e = self.evaluator
        score, _, message, _ = self._evaluate(
            ["self.cjt.exe._execute_query('SELECT * FROM sales')"]
        )
        self.assertEqual((score, message), (0.0, e.POLICY_REASONS["data_cap"]))

    def test_caught_violation_is_still_invalid(self):
        e = self.evaluator
        score, _, message, _ = self._evaluate(
            [
                "try:",
                "    self.cjt.exe._execute_query('SELECT * FROM sales')",
                "except Exception:",
                "    pass",
            ]
        )
        self.assertEqual((score, message), (0.0, e.POLICY_REASONS["data_cap"]))

    def test_changing_database_settings_is_invalid(self):
        e = self.evaluator
        for statement in ("SET threads=1", "PRAGMA threads=1"):
            score, _, message, _ = self._evaluate(
                [f"self.cjt.exe._execute_query('{statement}')"]
            )
            self.assertEqual((score, message), (0.0, e.POLICY_REASONS["settings"]))

    def test_every_export_family_is_invalid_on_real_duckdb(self):
        """Each family moves fact-scale data in few rows; the guard refuses it."""
        e = self.evaluator
        families = {
            "reviewer": "SELECT list(i) FROM range(250001) AS r(i)",
            "list": "SELECT list(unit_sales) FROM sales",
            "array_agg": "SELECT array_agg(tid) FROM sales",
            "struct": "SELECT {{a: item_nbr, b: tid}} FROM sales LIMIT 5",
            "map": "SELECT histogram(tid) FROM sales",
            "nested": "SELECT list({{a: item_nbr, b: [tid]}}) FROM sales",
            "string_agg": "SELECT string_agg(tid::VARCHAR, chr(44)) FROM sales",
            "json": "SELECT to_json(list(tid))::VARCHAR FROM sales",
            "list_as_text": "SELECT CAST(list(tid) AS VARCHAR) FROM sales",
            "blob": "SELECT encode(string_agg(tid::VARCHAR, chr(44))) FROM sales",
            "bit": "SELECT bitstring_agg(tid) FROM sales",
            "wide": "SELECT unnest({{"
            + ", ".join(f"c{i}: {i}" for i in range(40))
            + "}})",
            "wide_numbers": "SELECT sum(tid::HUGEINT << (32 * (item_nbr % 4)))"
            " FROM sales GROUP BY item_nbr // 4, tid",
        }
        for name, sql in families.items():
            with self.subTest(family=name):
                score, _, message, _ = self._evaluate(
                    [f'self.cjt.exe._execute_query("{sql.format()}")']
                )
                self.assertEqual(
                    (score, message), (0.0, e.POLICY_REASONS["data_cap"]), name
                )

    def test_database_errors_do_not_carry_data(self):
        """A caught error leaves a bounded message; the fit stays valid."""
        score, _, message, metrics = self._evaluate(
            [
                "try:",
                "    self.cjt.exe._execute_query("
                '"SELECT error(string_agg(tid::VARCHAR, chr(44))) FROM sales")',
                "except Exception as leaked:",
                "    assert len(str(leaked)) <= 256",
                "    assert leaked.__context__ is None",
            ]
        )
        self.assertIn("correct ensemble", message)
        self.assertEqual(metrics["correct"], 1)

    def test_no_file_or_metadata_side_channel(self):
        e = self.evaluator
        for sql, reason in (
            ("COPY sales TO 'sales.csv'", "settings"),
            ("EXPORT DATABASE 'dump'", "settings"),
            ("PRAGMA database_list", "settings"),
            ("CALL pragma_version()", "settings"),
            ("INSTALL httpfs", "settings"),
            ("EXPLAIN SELECT * FROM sales JOIN items USING (item_nbr)", "data_cap"),
        ):
            with self.subTest(sql=sql):
                score, _, message, _ = self._evaluate(
                    [f'self.cjt.exe._execute_query("{sql}")']
                )
                self.assertEqual((score, message), (0.0, e.POLICY_REASONS[reason]))
        # Reading files is refused by DuckDB itself once external access is off:
        # the statement fails, nothing is returned, and the fit goes on.
        score, _, message, metrics = self._evaluate(
            [
                "for sql in (\"SELECT * FROM read_csv('/etc/passwd')\",",
                "            \"SELECT content FROM read_text('/etc/hostname')\",",
                "            \"CREATE TABLE joinboost_tmp_f AS SELECT * FROM read_blob('/etc/hostname')\"):",
                "    try:",
                "        self.cjt.exe._execute_query(sql)",
                "        raise SystemExit('file access was possible')",
                "    except Exception as refused:",
                "        assert 'Permission' in str(refused), str(refused)",
            ]
        )
        self.assertIn("correct ensemble", message)
        self.assertEqual(metrics["correct"], 1)

    def test_reference_and_baseline_run_on_all_eight_development_workloads(self):
        e = self.evaluator
        executor = (REPO / EXECUTOR).read_text()
        for name in ("reference.patch", "baseline_naive.patch"):
            applied = e.apply_submission_patch(
                str(self.task / name), self.task, e.CONFIG
            )
            self.assertTrue(applied.ok, name)
            self.addCleanup(shutil.rmtree, Path(applied.workdir).parent, True)
            for index, case in enumerate(e.DEV_SUITE):
                with self.subTest(patch=name, workload=index):
                    result = e._run_once(Path(applied.workdir), case)
                    self.assertLessEqual(result["rmse_error"], e.RMSE_TOLERANCE)
                    self.assertIsNotNone(result["residual_update_seconds"])
                    # Generous headroom: at least 10x below every limit.
                    for key, limit in (
                        ("rows_returned_max", "MAX_ROWS_PER_QUERY"),
                        ("rows_returned_total", "MAX_ROWS_PER_SESSION"),
                        ("cells_returned_max", "MAX_CELLS_PER_QUERY"),
                        ("cells_returned_total", "MAX_CELLS_PER_SESSION"),
                        ("bytes_returned_max", "MAX_BYTES_PER_QUERY"),
                        ("bytes_returned_total", "MAX_BYTES_PER_SESSION"),
                        ("text_returned_max", "MAX_TEXT_LENGTH"),
                    ):
                        value = int(
                            re.search(limit + r" = ([\d_]+)", executor).group(1)
                        )
                        self.assertLessEqual(result[key] * 10, value, key)
                    self.assertLessEqual(result["columns_returned_max"] * 5, 32)

    def test_materialised_join_is_invalid(self):
        e = self.evaluator
        wide = (
            "SELECT s.*, i.family, i.class, i.perishable, i.f1, t.transactions"
            " FROM sales s JOIN items i USING (item_nbr) JOIN transactions t USING (tid)"
        )
        for kind in ("TABLE", "VIEW"):
            score, _, message, _ = self._evaluate(
                [
                    "self.cjt.exe._execute_query("
                    f"'CREATE OR REPLACE {kind} joinboost_tmp_wide AS {wide}')"
                ]
            )
            self.assertEqual(
                (score, message), (0.0, e.POLICY_REASONS["materialised_join"]), kind
            )

    def test_narrow_fact_scale_copies_are_allowed(self):
        score, _, message, metrics = self._evaluate(
            [
                "self.cjt.exe._execute_query('CREATE OR REPLACE TABLE joinboost_tmp_copy AS"
                " SELECT s.*, i.f1, i.family FROM sales s JOIN items i USING (item_nbr)')"
            ]
        )
        self.assertIn("correct ensemble", message)
        self.assertEqual(metrics["correct"], 1)

    def test_deferred_update_is_unmeasured(self):
        """The trusted read finds stale residuals where the next tree reads."""
        # Stale: the update is applied after _update_error returned.
        score, _, message, metrics = self._evaluate(
            [
                "self._pending_update = (leaf_nodes, learning_rate)",
                "return None",
            ],
            extra=(
                "    def _clean_messages(self):\n"
                "        pending = getattr(self, '_pending_update', None)\n"
                "        self._pending_update = None\n"
                "        if pending is not None:\n"
                "            self._apply_update(*pending)\n"
                "        super()._clean_messages()\n"
            ),
            rename="_apply_update",
        )
        self.assertIn("correct ensemble", message)
        self.assertTrue(
            all(
                row["candidate_residual_update_seconds"] is None
                for row in metrics["workload_measurements"]
            )
        )
        self.assertFalse(
            any(
                row["metrics"]["residual_update_seconds"]
                for row in metrics["beat_spec_workload_passes"]
            )
        )
        # In place (the reference itself): measured.
        _, _, _, metrics = self._evaluate(["# byte-different reference"])
        self.assertTrue(
            all(
                row["candidate_residual_update_seconds"] > 0
                for row in metrics["workload_measurements"]
            )
        )

    def test_replacing_executor_methods_is_invalid(self):
        e = self.evaluator
        score, _, message, _ = self._evaluate(
            [
                "cls = type(self.cjt.exe)",
                "original = cls.get_schema",
                "cls.get_schema = lambda this, table: original(this, table)",
            ]
        )
        self.assertEqual((score, message), (0.0, e.POLICY_REASONS["executor"]))


if __name__ == "__main__":
    unittest.main()
