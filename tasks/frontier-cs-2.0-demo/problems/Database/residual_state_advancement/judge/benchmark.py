#!/usr/bin/env python3
"""Private benchmark worker for the JoinBoost residual-update task."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Bound before any candidate import, so later rebinding of time.perf_counter
# cannot reach the residual-update clock below.
_CLOCK = time.perf_counter

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


FULL_FEATURES = [
    "htype", "locale", "locale_name", "transferred", "f2",
    "dcoilwtico", "f3", "transactions", "f5", "city", "state",
    "stype", "cluster", "f4", "family", "class", "perishable", "f1",
]

CORE_FEATURES = [
    "transactions", "f5", "city", "state", "stype", "cluster", "f4",
    "family", "class", "perishable", "f1",
]

TOPOLOGY_FEATURES = {
    "six_table": FULL_FEATURES,
    "five_table_oil": ["dcoilwtico", "f3"] + CORE_FEATURES,
    "core_four_table": CORE_FEATURES,
}


# Database settings are fixed here and locked before the candidate runs.
DB_THREADS = 2
DB_MEMORY_LIMIT = "10GB"
POLICY_EXIT_CODE = 3
# Materialised-join guard. Training runs over the non-materialised join, so no
# relation with at least FACT_SCALE_FRACTION of the fact rows may carry more
# than FACT_EXTRA_COLUMNS columns beyond the fact table as loaded. Measured: the
# supplied implementation and the per-leaf baseline create no fact-scale
# relation wider than the fact table itself (+0) on any development workload,
# nor did the in-graph designs replayed for this revision; a materialised join
# over the six-table graph has +17.
FACT_SCALE_FRACTION = 0.10
FACT_EXTRA_COLUMNS = 4
# Relative tolerance of the residual-state consistency value (see _probe_state).
STATE_TOLERANCE = 1.0e-6


FACT_FILES = tuple(f"sales-{suite}-{index}.parquet" for suite in ("dev", "final") for index in range(4))


class PolicyViolation(RuntimeError):
    """The fit broke a trusted-executor rule; ``code`` is a closed identifier."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _lookup(frame: pd.DataFrame, key: str, column: str) -> np.ndarray:
    keys = frame[key].to_numpy()
    if len(np.unique(keys)) != len(keys) or keys.min() < 0:
        raise RuntimeError("dimension key is not unique")
    table = np.zeros(int(keys.max()) + 1, dtype=frame[column].dtype)
    table[keys] = frame[column].to_numpy()
    return table


def load_sales(public_data: Path, fact_file: Path, limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fact table of real Favorita sales rows over the shipped dimension tables.

    ``fact_file`` is a Parquet slice of Corporacion Favorita sales rows with the
    columns ``item_nbr`` and ``tid`` (the keys of ``items.csv`` and
    ``transactions.csv``) and the real ``unit_sales``. Every row is a distinct
    (item_nbr, tid) pair, i.e. a distinct item, store and date. ``onpromotion``
    is not available for these rows and is set to 2, the code the artifact uses
    for a promotion flag that was not recorded. ``Y`` is the target formula of
    the upstream JoinBoost artifact (exact on its 10,000-row sample):
    f1 + log2(f2) - 10*f3 - 10*f4 + f5*log2(f5). ``limit`` keeps the first rows
    in file order. Returns the fact table and the aligned joined feature table.
    """
    items = pd.read_csv(public_data / "items.csv")
    transactions = pd.read_csv(public_data / "transactions.csv")
    stores = pd.read_csv(public_data / "stores.csv")
    holidays = pd.read_csv(public_data / "holidays.csv")
    oil = pd.read_csv(public_data / "oil.csv")
    query = "SELECT item_nbr, tid, unit_sales FROM read_parquet(?)"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    reader = duckdb.connect(database=":memory:")
    try:
        reader.execute("SET threads=1")
        columns = reader.execute(query, [str(fact_file)]).fetchnumpy()
    finally:
        reader.close()
    item_nbr = np.asarray(columns["item_nbr"]).astype(np.int64)
    tid = np.asarray(columns["tid"]).astype(np.int64)
    rows = len(item_nbr)
    if rows == 0 or len(np.unique(item_nbr * (int(tid.max()) + 1) + tid)) != rows:
        raise RuntimeError("fact rows are not unique")

    date = _lookup(transactions, "tid", "date")[tid]
    store = _lookup(transactions, "tid", "store_nbr")[tid]
    joined = {}
    for frame, key, index in ((items, "item_nbr", item_nbr), (transactions, "tid", tid),
                              (stores, "store_nbr", store), (holidays, "date", date), (oil, "date", date)):
        for column in frame.columns:
            if column not in ("item_nbr", "tid", "store_nbr", "date"):
                joined[column] = _lookup(frame, key, column)[index]
    target = (joined["f1"] + np.log2(joined["f2"]) - 10.0 * joined["f3"] - 10.0 * joined["f4"]
              + joined["f5"] * np.log2(joined["f5"]))
    sales = pd.DataFrame({
        "item_nbr": item_nbr,
        "unit_sales": np.asarray(columns["unit_sales"], dtype=float),
        "onpromotion": np.full(rows, 2, dtype=np.int64),
        "tid": tid,
        "Y": target,
    })
    features = pd.DataFrame({name: joined[name].astype(np.int32) for name in FULL_FEATURES})
    features["Y"] = target
    return sales, features


def _oracle(joined: pd.DataFrame, features: list[str], depth: int, estimators: int,
            learning_rate: float, cache: Path | None) -> tuple[np.ndarray, float]:
    """Sorted materialized-join oracle predictions and RMSE, cached per case."""
    if cache is not None and cache.is_file():
        with np.load(cache) as stored:
            return stored["prediction"], float(stored["rmse"])
    oracle = GradientBoostingRegressor(
        max_depth=depth,
        learning_rate=learning_rate,
        n_estimators=estimators,
        random_state=0,
    ).fit(joined[features], joined["Y"])
    prediction = np.asarray(oracle.predict(joined[features]), dtype=float)
    rmse = math.sqrt(mean_squared_error(joined["Y"], prediction))
    prediction = np.sort(prediction)
    if cache is not None:
        temporary = cache.with_name(cache.name + f".{os.getpid()}.tmp.npz")
        np.savez(temporary, prediction=prediction, rmse=np.float64(rmse))
        os.replace(temporary, cache)
    return prediction, rmse


def _class_snapshot(module) -> dict:
    snapshot = {("module", name): getattr(module, name) for name in (
        "MAX_ROWS_PER_QUERY", "MAX_ROWS_PER_SESSION", "MAX_COLUMNS_PER_QUERY", "MAX_CELLS_PER_QUERY",
        "MAX_CELLS_PER_SESSION", "MAX_BYTES_PER_QUERY", "MAX_BYTES_PER_SESSION", "MAX_TEXT_LENGTH",
        "_ALLOWED_RESULT_TYPES", "_ALLOWED_STATEMENTS", "_SQL_COMMENT", "_TABLE_INFO",
        "_check_statements", "_value_bytes", "_guard_connection", "ExecutorPolicyError",
        "ExecutorException", "re")}
    for cls in (module.Executor, module.DuckdbExecutor):
        for name, value in vars(cls).items():
            snapshot[(cls.__name__, name)] = value
    return snapshot


def _validate_fitted_model(model, estimators: int, max_leaves: int, audit: dict[str, int]) -> None:
    if audit != {"train_one": estimators, "build_tree": estimators}:
        raise RuntimeError("fit bypassed the generic tree-growth path")
    if "train_one" in model.__dict__ or "_build_tree" in model.__dict__:
        raise RuntimeError("fit replaced a generic tree-growth entry point")
    if not isinstance(model.model_def, list) or len(model.model_def) != estimators:
        raise RuntimeError("fit did not finish the requested ensemble")
    for tree in model.model_def:
        if getattr(getattr(tree, "agg", None), "name", "") != "CASE":
            raise RuntimeError("fit did not produce an ordinary tree model")
        leaves = getattr(tree, "para", None)
        if not isinstance(leaves, list) or not 1 <= len(leaves) <= max_leaves:
            raise RuntimeError("fit produced an invalid leaf collection")
        for prediction, predicates in leaves:
            if not math.isfinite(float(prediction)) or not isinstance(predicates, list) or not predicates:
                raise RuntimeError("fit produced an incomplete leaf")
    if not hasattr(model, "cjt") or not hasattr(model, "constant_") or not math.isfinite(float(model.constant_)):
        raise RuntimeError("fit did not materialize reusable training state")


def run(tree: Path, fact: str, rows: int, depth: int, estimators: int, learning_rate: float,
        topology: str, limit: int | None = None, oracle_cache: Path | None = None,
        oracle_only: bool = False) -> dict:
    public_data = tree / "data" / "favorita"
    features = TOPOLOGY_FEATURES[topology]
    # Fact slices are read from the judge's own copies, never from the patched
    # tree: development slices from the pristine tree, graded slices from the
    # private judge data.
    task_dir = Path(__file__).resolve().parent
    if fact not in FACT_FILES:
        raise RuntimeError("unknown fact slice")
    fact_root = (task_dir / "judge_data" / "favorita" if fact.startswith("sales-final-")
                 else task_dir / "harbor" / "app" / "data" / "favorita")
    sales, joined = load_sales(public_data, fact_root / fact, limit)
    if len(sales) != rows:
        raise RuntimeError("fact slice does not have the expected number of rows")
    if oracle_only:
        _oracle(joined, features, depth, estimators, learning_rate, oracle_cache)
        return {"oracle": "ready", "rows": rows, "topology": topology}

    sys.path.insert(0, str(tree / "src"))
    import joinboost.executor as executor_module
    from joinboost.app import GradientBoosting
    from joinboost.executor import DuckdbExecutor
    from joinboost.joingraph import JoinGraph

    audit = {"train_one": 0, "build_tree": 0}
    original_train_one = GradientBoosting.train_one
    original_build_tree = GradientBoosting._build_tree

    def audited_train_one(self, *args, **kwargs):
        audit["train_one"] += 1
        return original_train_one(self, *args, **kwargs)

    def audited_build_tree(self, *args, **kwargs):
        audit["build_tree"] += 1
        return original_build_tree(self, *args, **kwargs)

    GradientBoosting.train_one = audited_train_one
    GradientBoosting._build_tree = audited_build_tree

    # Beat spec v2: the paper's residual update time (JoinBoost Section 5.3/5.4,
    # Figs. 5 and 14). The fixed train_one calls _update_error once per tree.
    update = {"calls": 0, "seconds": 0.0}
    original_update_error = GradientBoosting._update_error

    judge = {"con": None, "fact_rows": 0, "fact_columns": 0, "sse": None, "consistent": True,
             "violation": None, "guard_seconds": 0.0, "probe_seconds": 0.0, "widest": 0}

    def _sse(relation: str, g_col: str, materialise: bool) -> tuple[float, int]:
        con_ = judge["con"]
        source = relation
        if materialise:
            # A relation that is not a base table is a deferred computation: the
            # trusted read computes it in full, all columns, and discards the copy.
            con_.execute(f"CREATE OR REPLACE TEMP TABLE judge_state_probe AS SELECT * FROM {relation}")
            source = "judge_state_probe"
        try:
            total, count = con_.execute(
                f"SELECT SUM(CAST({g_col} AS DOUBLE) * CAST({g_col} AS DOUBLE)), COUNT(*) FROM {source}").fetchone()
        finally:
            if materialise:
                con_.execute("DROP TABLE IF EXISTS judge_state_probe")
        return float(total), int(count)

    def _is_base_table(relation: str) -> bool:
        return bool(judge["con"].execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name = ?",
            [str(relation)]).fetchone()[0])

    def _probe_state(model_, state, leaf_nodes, rate, before) -> None:
        """Trusted full read of the updated target state, inside the update clock.

        Reads the residual column of exactly the relation the next tree will
        consume (the returned state's, or else the model's, target relation).
        The sum of squared residuals must equal the value before the update
        minus (2*rate - rate^2) * sum_leaf(g^2 / h); otherwise the update did
        not happen where the next tree reads it and the clock is not trusted.
        """
        try:
            cjt = state if state is not None else model_.cjt
            relation = cjt.target_relation
            g_col, _ = model_.semi_ring.get_columns_name()
            after, count = _sse(relation, g_col, not _is_base_table(relation))
            gain = 0.0
            for leaf in leaf_nodes:
                g, h = leaf.get_semi_ring().get_value()
                gain += float(g) * float(g) / float(h)
            expected = before - (2.0 * rate - rate * rate) * gain
            if count != judge["fact_rows"] or not math.isfinite(after) or (
                    abs(after - expected) > STATE_TOLERANCE * max(abs(before), 1.0)):
                judge["consistent"] = False
            judge["sse"] = after
        except Exception:
            judge["consistent"] = False
            judge["sse"] = None

    def _check_relations() -> None:
        """Fail if a fact-scale relation is wider than the fact table allows."""
        started_check = _CLOCK()
        con_ = judge["con"]
        limit = judge["fact_columns"] + FACT_EXTRA_COLUMNS
        floor = FACT_SCALE_FRACTION * judge["fact_rows"]
        wide = False
        for name, rows, columns in con_.execute(
                "SELECT table_name, estimated_size, column_count FROM duckdb_tables() WHERE NOT internal").fetchall():
            if rows >= floor:
                judge["widest"] = max(judge["widest"], int(columns))
                wide = wide or columns > limit
        # A view is only counted (one COUNT(*)) when it is wider than the limit.
        for name, columns in con_.execute(
                "SELECT view_name, column_count FROM duckdb_views() WHERE NOT internal AND column_count > ?",
                [limit]).fetchall():
            if con_.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0] >= floor:
                judge["widest"] = max(judge["widest"], int(columns))
                wide = True
        judge["guard_seconds"] += _CLOCK() - started_check
        if wide:
            judge["violation"] = "materialised_join"
            raise PolicyViolation("materialised_join")

    def timed_update_error(self, *args, **kwargs):
        update["calls"] += 1
        leaf_nodes = args[0] if args else kwargs.get("leaf_nodes", [])
        rate = float(args[1] if len(args) > 1 else kwargs.get("learning_rate", 0.0))
        before = judge["sse"]
        if before is None:
            # First round (or after an unreadable state): the value before the
            # update, read outside the update clock.
            try:
                relation = self.cjt.target_relation
                before, _ = _sse(relation, self.semi_ring.get_columns_name()[0], not _is_base_table(relation))
            except Exception:
                before = float("nan")
        started_update = _CLOCK()
        try:
            state = original_update_error(self, *args, **kwargs)
            started_probe = _CLOCK()
            _probe_state(self, state, leaf_nodes, rate, before)
            judge["probe_seconds"] += _CLOCK() - started_probe
        finally:
            update["seconds"] += _CLOCK() - started_update
        _check_relations()
        return state

    GradientBoosting._update_error = timed_update_error

    spill = tempfile.mkdtemp(prefix="joinboost-spill-")
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads={DB_THREADS}")
    con.execute(f"SET memory_limit='{DB_MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory='{spill}'")
    con.register("bench_sales", sales)
    con.execute("CREATE TABLE sales AS SELECT * FROM bench_sales")
    con.unregister("bench_sales")
    del sales

    judge["con"] = con
    judge["fact_rows"] = int(con.execute("SELECT count(*) FROM sales").fetchone()[0])
    judge["fact_columns"] = len(con.execute("SELECT * FROM sales LIMIT 0").description)
    exe = DuckdbExecutor(con, debug=False)
    guard = exe.conn
    dataset = JoinGraph(exe=exe)
    dataset.add_relation("sales", [], y="Y")
    dataset.add_relation("transactions", ["transactions", "f5"], relation_address=str(public_data / "transactions.csv"))
    dataset.add_relation("stores", ["city", "state", "stype", "cluster", "f4"], relation_address=str(public_data / "stores.csv"))
    dataset.add_relation("items", ["family", "class", "perishable", "f1"], relation_address=str(public_data / "items.csv"))
    dataset.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
    dataset.add_join("sales", "transactions", ["tid"], ["tid"])
    dataset.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
    if topology == "six_table":
        dataset.add_relation("holidays", ["htype", "locale", "locale_name", "transferred", "f2"], relation_address=str(public_data / "holidays.csv"))
        dataset.add_relation("oil", ["dcoilwtico", "f3"], relation_address=str(public_data / "oil.csv"))
        dataset.add_join("transactions", "holidays", ["date"], ["date"])
        dataset.add_join("holidays", "oil", ["date"], ["date"])
    elif topology == "five_table_oil":
        dataset.add_relation("oil", ["dcoilwtico", "f3"], relation_address=str(public_data / "oil.csv"))
        dataset.add_join("transactions", "oil", ["date"], ["date"])

    # Every relation is loaded: from here the database cannot read or write
    # files, load extensions or attach databases, and no setting can change.
    con.execute("SET enable_external_access=false")
    con.execute("SET lock_configuration=true")
    trusted = _class_snapshot(executor_module)

    model = GradientBoosting(
        learning_rate=learning_rate,
        num_leaves=2**depth,
        max_depth=depth,
        n_estimators=estimators,
        partition_early=True,
    )
    started = time.perf_counter()
    try:
        model.fit(dataset)
    except Exception:
        if judge["violation"]:
            raise PolicyViolation(judge["violation"]) from None
        if guard.stats()["violation"]:
            raise PolicyViolation(guard.stats()["violation"]) from None
        raise
    _validate_fitted_model(model, estimators, 2**depth, audit)
    _check_relations()
    elapsed = time.perf_counter() - started
    if judge["violation"]:
        raise PolicyViolation(judge["violation"])
    usage = guard.stats()
    if usage["violation"]:
        raise PolicyViolation(usage["violation"])
    if (
        type(exe) is not DuckdbExecutor
        or exe.conn is not guard
        or dataset.exe is not exe
        or _class_snapshot(executor_module) != trusted
        or any(callable(value) for value in vars(exe).values())
    ):
        raise PolicyViolation("executor")
    # The update clock is trusted only if every round went through the fixed
    # call site and the timed class attribute was neither replaced nor shadowed.
    update_measured = (
        update["calls"] == estimators
        and GradientBoosting.__dict__.get("_update_error") is timed_update_error
        and "_update_error" not in model.__dict__
        and judge["consistent"]
    )

    # The candidate cannot access the materialized oracle during timed fitting.
    con.register("bench_joined", joined)
    con.execute("CREATE TABLE train AS SELECT * FROM bench_joined")
    con.unregister("bench_joined")
    dataset.target_relation = "train"
    # The judge reads the predictions on its own connection: the executor's row
    # cap applies to everything that goes through the executor, the judge included.
    view = model.predict(joingraph=dataset, input_mode="FULL_JOIN_JG", output_mode="WRITE_TO_TABLE")
    candidate_prediction = np.sort(np.asarray(
        con.execute(f"SELECT prediction FROM {view}").fetchnumpy()["prediction"], dtype=float))
    candidate_rmse = float(model.compute_rmse("train")[0])
    if guard.stats()["violation"]:
        raise PolicyViolation(guard.stats()["violation"])

    oracle_prediction, oracle_rmse = _oracle(joined, features, depth, estimators, learning_rate, oracle_cache)
    if candidate_prediction.shape != oracle_prediction.shape:
        raise RuntimeError("prediction row count differs from the materialized oracle")
    prediction_l1 = float(np.sum(np.abs(candidate_prediction - oracle_prediction)))
    return {
        "elapsed_seconds": elapsed,
        "rows": rows,
        "estimators": estimators,
        "topology": topology,
        "rmse_error": abs(candidate_rmse - oracle_rmse),
        "prediction_l1": prediction_l1,
        "residual_update_seconds": update["seconds"] if update_measured else None,
        "residual_update_calls": update["calls"],
        "rows_returned_max": usage["max_rows"],
        "rows_returned_total": usage["rows"],
        "cells_returned_max": usage["max_cells"],
        "cells_returned_total": usage["cells"],
        "bytes_returned_max": usage["max_bytes"],
        "bytes_returned_total": usage["bytes"],
        "columns_returned_max": usage["max_columns"],
        "text_returned_max": usage["max_text"],
        "database_queries": usage["queries"],
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "state_probe_seconds": judge["probe_seconds"],
        "state_consistent": judge["consistent"],
        "relation_guard_seconds": judge["guard_seconds"],
        "fact_scale_max_columns": judge["widest"],
        "fact_columns": judge["fact_columns"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--fact", required=True, choices=FACT_FILES)
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--depth", required=True, type=int)
    parser.add_argument("--estimators", required=True, type=int)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--topology", required=True, choices=sorted(TOPOLOGY_FEATURES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--oracle-cache", type=Path, default=None)
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.tree, args.fact, args.rows, args.depth, args.estimators, args.learning_rate,
                             args.topology, args.limit, args.oracle_cache, args.oracle_only), sort_keys=True))
        return 0
    except PolicyViolation as violation:
        print(json.dumps({"policy_violation": violation.code}, sort_keys=True))
        return POLICY_EXIT_CODE
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
