#!/usr/bin/env python3
"""Small adapter around the retained LeetCode worker."""

from __future__ import annotations

import builtins
import csv
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path


_real_open = builtins.open


def _import_open(path, *args, **kwargs):
    # The retained entrypoint eagerly reads a 32 MB result log used only for
    # cross-tool reporting.  The benchmark below does not consume that report.
    if str(path).endswith("leetcode_RQ1_2023_03_27_23_07_03.out"):
        return io.StringIO("")
    return _real_open(path, *args, **kwargs)


builtins.open = _import_open
try:
    from evaluation.leetcode import online
finally:
    builtins.open = _real_open

from evaluation.calcite.tester import MySQLTester
from sql_testing.environment import Environment


def _as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _pair(payload: dict, socket: str) -> dict:
    problem_id = int(payload["problem_id"])
    qid = int(payload["qid"])
    query = str(payload["query"])
    bound = int(payload.get("bound", 2))
    timeout_seconds = int(payload.get("timeout_seconds", 4))
    original_environment = online.Environment

    class BoundedEnvironment(original_environment):
        def __init__(self, *args, **kwargs):
            kwargs["time_budget"] = timeout_seconds
            super().__init__(*args, **kwargs)

    online.Environment = BoundedEnvironment
    online.DB_CONFIG = {"unix_socket": socket, "user": "root"}

    benchmark_root = Path(online.BENCHMARK_DIR) / "leetcode"
    import pandas as pd

    groundtruth = pd.read_csv(
        benchmark_root / "groundtruths" / f"{problem_id}.csv"
    )["mysql_groundtruth"][0]

    with tempfile.TemporaryDirectory(prefix="polygon-worker-") as directory:
        previous = Path.cwd()
        os.chdir(directory)
        try:
            online.leetcode_worker(0, problem_id, groundtruth, [(qid, query)], bound)
            output = Path(f"experiment_result_lc_{problem_id}_0.csv")
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        finally:
            os.chdir(previous)

    if len(rows) != 1:
        raise ValueError("retained worker did not produce one result row")
    row = rows[0]
    refuted = str(row.get("?Refuted", "")).strip().lower() == "true"
    try:
        elapsed = float(row.get("Time", ""))
    except (TypeError, ValueError):
        elapsed = float(timeout_seconds)
    if not math.isfinite(elapsed) or elapsed < 0:
        elapsed = float(timeout_seconds)
    elapsed = min(float(timeout_seconds), elapsed)
    return {
        "ok": True,
        "refuted": refuted,
        "time_seconds": elapsed,
        "iterations": _as_int(row.get("#Iters")),
        "backtracks": _as_int(row.get("#Backtracking")),
    }


def _disambiguation(payload: dict, socket: str) -> dict:
    schema = list(payload["schema"])
    queries = [str(query) for query in payload["queries"]]
    bound = int(payload.get("bound", 1))
    timeout_seconds = int(payload.get("timeout_seconds", 10))
    default_k = {name: bound for name in (
        "filter", "project", "union", "inner join", "left join",
        "right join", "full join", "product", "order by",
    )}
    environment = Environment(
        schema, [], bound=bound, default_k=default_k, time_budget=timeout_seconds
    )
    eq, counterexample, _checking, elapsed, telemetry = environment.disambiguate(
        queries, group_range=0, use_precise_encoding=False
    )

    refuted = False
    if eq is False and counterexample is not None:
        with MySQLTester(
            {"unix_socket": socket, "user": "root"},
            schema,
            db_predix="frontier_disambiguation",
        ) as tester:
            tester.create_all_databases([counterexample], [])
            tester.cursor.execute("USE frontier_disambiguation_0")
            clusters = {}
            for query in queries:
                try:
                    tester.cursor.execute(query)
                    key = tuple(tester.cursor.fetchall())
                except Exception:
                    key = "query_not_executable"
                clusters[key] = clusters.get(key, 0) + 1
        counts = sorted(int(count) for key, count in clusters.items()
                        if key != "query_not_executable")
        refuted = (
            "query_not_executable" not in clusters
            and len(counts) == 2
            and counts == [len(queries) // 2, len(queries) // 2]
        )

    elapsed = float(elapsed)
    if not math.isfinite(elapsed) or elapsed < 0:
        elapsed = float(timeout_seconds)
    return {
        "ok": True,
        "refuted": refuted,
        "time_seconds": min(float(timeout_seconds), elapsed),
        "iterations": _as_int((telemetry or {}).get("iters")),
        "backtracks": _as_int((telemetry or {}).get("backtracks")),
    }


def main() -> int:
    payload = json.loads(sys.stdin.read())
    socket = os.environ["FRONTIER_MYSQL_SOCKET"]
    if payload.get("kind") == "disambiguation":
        result = _disambiguation(payload, socket)
    else:
        result = _pair(payload, socket)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        raise SystemExit(1)
