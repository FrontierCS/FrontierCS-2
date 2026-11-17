# Residual State Advancement

Implement `GradientBoosting._update_error` in JoinBoost, which advances the residual training
state after each boosting round, and make the complete `fit` as fast as possible while training
stays inside DuckDB over the normalized relations. A valid fit matches the materialized-join
oracle's RMSE and predictions on every run.

## Task

JoinBoost (Huang et al., "JoinBoost: Grow Trees over Normalized Data Using Only SQL", PVLDB 16)
trains trees and ensembles as SQL over normalized relations, so the join is never exported or
materialized. Here it trains RMSE gradient boosting on real Corporacion Favorita sales rows over
three join-graph topologies. The body of `GradientBoosting._update_error` has been removed.

## What you may change

- The only files that may change are:

  ```
  src/joinboost/app.py
  src/joinboost/semiring.py
  src/joinboost/aggregator.py
  src/joinboost/cjt.py
  src/joinboost/joingraph.py
  ```

  No file may be added or deleted.
- Fixed (a change is rejected as `out_of_scope`): class inheritance and `fit`, `_fit`, `predict`
  of `DummyModel`; `fit`, `_fit`, `train_one`, `_build_model`, `compute_rmse`, `predict`,
  `_build_tree`, `_get_best_split` of `DecisionTree`; `fit`, `_fit`, `train_one`,
  `_build_model`, `compute_rmse`, `predict`, `_build_tree` of `GradientBoosting` and
  `RandomForest`; and `src/joinboost/executor.py`, the trusted executor, which must not be
  replaced, wrapped or bypassed at run time either. Each estimator must go through one ordinary
  `train_one`/`_build_tree` call; instance-level replacements are rejected at run time.
- Added lines must not contain, case-insensitively, `benchmark.py`, `perf_counter`,
  `monotonic(`, `atexit`, `sys.modules`, `import benchmark`, `__main__`, `builtins`, `inspect`,
  `importlib`, `ctypes`, `.fit =`, `.fit=`, `setattr(`, `_getframe`, `currentframe`, `f_back`,
  `f_globals`, `f_locals`, `traceback`, `gc.get_objects`.
- Added lines must not match (case-sensitive) `duckdb`, `vars(`, `eval(`, `exec(`, `compile(`
  or `gc.` as whole words, `.conn` not followed by a letter, digit or `_`, `'conn'`/`"conn"`,
  `fetch` alone or followed by `all`, `one`, `many`, `numpy`, `df` or `_`, `.df(`,
  `.arrow(`, `.pl(`, `.to_df(`, `.torch(`, `.tf(`, `record_batch`, `__closure__`,
  `cell_contents`, `__globals__`, `__code__`, `__dict__`, `__import__`, `__builtins__`, `__subclasses__`, `__reduce`, `_guard_connection`,
  `GuardedConnection`, `MAX_ROWS_PER`, `_check_statements`. Both lists are rejected as
  `content_policy`.

## Interface

```python
def _update_error(self, leaf_nodes, learning_rate):
    """Advance training state for the next boosting round."""
```

`leaf_nodes` is the finished tree's list of `CJT` leaves, `learning_rate` its multiplier. It is
called after every tree, including the last. It may mutate `self.cjt` and return `None`, or
return the state that replaces it:

```python
residual_state = self._update_error(self.leaf_nodes, self.learning_rate)
if residual_state is not None:
    self.cjt = residual_state
```

Each later round must consume state that includes every preceding tree, and `cjt` must still
exist after `fit`. Talk to the database only through the given executor `self.cjt.exe`
(`DuckdbExecutor`) and its existing methods (`execute_spja_query`, `update_query`,
`_execute_query`, `case_query`, `delete_table`, `rename_column`, `get_schema`, ...).
Joins and aggregations run in the DBMS. Messages, split statistics, partitions and residual
updates are SQL over the normalized relations, and only small aggregated results return to
Python. Inside the database the data may be reorganized freely (tables, views, columns,
indexes).

Data (`data/favorita/`, headered CSVs, first record shown):

```
sales_small.csv   item_nbr=451, unit_sales=4.0, onpromotion=2, tid=17298, Y=-5739.103112581849
items.csv         item_nbr=1192, family=12, class=28, perishable=0, f1=958
transactions.csv  date=229, store_nbr=7, transactions=4, tid=10574, f5=955
stores.csv        store_nbr=38, city=3, state=0, stype=1, cluster=5, f4=243
holidays.csv      date=1040, htype=4, locale=2, locale_name=15, transferred=1, f2=612
oil.csv           date=1506, dcoilwtico=5, f3=450
```

The fact table `sales` has the schema of `sales_small.csv` (target `Y`). Joins:
`sales.item_nbr = items.item_nbr`, `sales.tid = transactions.tid`, `transactions.store_nbr =
stores.store_nbr`; `six_table` adds `transactions.date = holidays.date` and `holidays.date =
oil.date`, `five_table_oil` adds only `transactions.date = oil.date`. Features, in order:
`core_four_table`: transactions, f5, city, state, stype, cluster, f4, family, class, perishable,
f1; `five_table_oil` prepends dcoilwtico, f3; `six_table` prepends htype, locale, locale_name,
transferred, f2, dcoilwtico, f3.

## Correctness

The evaluation scores 0 unless, on all 32 candidate runs, the package imports, each process
finishes within 600 s with finite metrics, `model.model_def` holds exactly `n_estimators`
ordinary `CASE` trees of 1 to `num_leaves` leaves (finite predictions, nonempty predicates),
`model.constant_` is finite, `model.cjt` exists, the RMSE (`compute_rmse("train")[0]`) is
within `1e-3` of the oracle's, and the sorted predictions (`predict(..., input_mode="FULL_JOIN_JG",
output_mode="WRITE_TO_TABLE")`, one per fact row) differ from the oracle's by at most
`rows * 1e-6` in L1. It also scores 0 when a fit breaks a trusted-executor rule (even if the
exception is caught):

- Only these column types return to Python: `BOOLEAN`, integers up to `HUGEINT`/`UHUGEINT`,
  `FLOAT`, `DOUBLE`, `DECIMAL`, `VARCHAR`. Everything else is refused before fetching, including
  `DATE`, `TIMESTAMP` and `LIST`, `ARRAY`, `STRUCT`, `MAP`, `UNION`, `BLOB`, `BIT`, `JSON`, `ENUM`. Limits (`executor.py`): `MAX_ROWS_PER_QUERY = 10,000`, `MAX_COLUMNS_PER_QUERY = 32`, `MAX_CELLS_PER_QUERY = 40,000`, `MAX_BYTES_PER_QUERY = 200,000`, `MAX_TEXT_LENGTH = 256`; per fit `MAX_ROWS_PER_SESSION = 250,000`, `MAX_CELLS_PER_SESSION = 250,000`, `MAX_BYTES_PER_SESSION = 1,000,000`.
- Statements must begin with `SELECT`, `WITH`, `FROM`, `VALUES`, `CREATE`, `DROP`, `UPDATE`,
  `INSERT`, `DELETE`, `ALTER`, `BEGIN`, `COMMIT`, `ROLLBACK`, `EXPLAIN` or `DESCRIBE`; settings
  (`threads=2`, `memory_limit='10GB'`, no external access) are fixed.
- The executor is not replaced or bypassed.
- No relation (table or view) that holds at least 10% of the fact table's rows has more than 4 columns beyond the fact table's own 5 (checked after every round and at the end).

## Workload

The train and final suites each contain eight deterministic cases: each of four fact slices under two profiles `(slice, rows, max_depth,
n_estimators, learning_rate, topology)`, with `num_leaves = 2 ** max_depth` and
`partition_early = True`:

| | slice | train rows | depth | trees | lr | topology |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 2,054,144 | 2 | 8 | 0.5 | core_four_table |
| 2 | 1 | 1,540,608 | 3 | 7 | 0.1 | five_table_oil |
| 3 | 2 | 1,027,072 | 4 | 6 | 1.0 | six_table |
| 4 | 3 | 2,054,144 | 2 | 12 | 0.25 | six_table |
| 5 | 0 | 2,054,144 | 1 | 24 | 0.5 | core_four_table |
| 6 | 1 | 1,540,608 | 2 | 20 | 0.1 | five_table_oil |
| 7 | 2 | 1,027,072 | 5 | 3 | 0.5 | six_table |
| 8 | 3 | 2,054,144 | 3 | 4 | 0.5 | six_table |

Final uses the same profiles on disjoint rows (2017-05-01 to 2017-07-30; slices of 2,142,914,
1,607,195, 1,071,466 and 2,142,929 rows; not distributed). Train slices are
`data/favorita/sales-dev-0.parquet` to `-3` (2017-02-02 to 2017-04-30; `load_sales.py` is the
judge's loader). Fact rows are real sales: every `(item_nbr, tid)` pair is distinct (no fact row is repeated),
`onpromotion` is the constant 2, and `Y = f1 + log2(f2) - 10*f3 - 10*f4 + f5*log2(f5)`. The
data come from the JoinBoost artifact (Apache-2.0) and `train_set_2017.csv.7z` of
`github.com/bgautijonsson/groceries` (commit `f79ac95b1c61ca9767f4e7d379bc2229fd8938fc`,
sha256 `86276d82a921ac0415d95a776dcf5122b17ed03a58eb8d72d6f9f09b1eef02a5`), which
carries no licence statement.

Each case runs four times for the candidate and four for the reference, each in a fresh
process, back to back: reference first in repetitions 1 and 3, candidate first in 2 and 4.
The fit time covers `model.fit` and model validation.

## Scoring

```text
T     = sum_i min(t_i1, t_i2)     (candidate's first two fit times per case)
B     = the same sum for baseline_naive (one UPDATE per leaf), measured on this suite
L     = max(0, log10(B / T))
score = 100 * L / (2 + L)
```

Matching or running slower than the per-leaf baseline scores 0; 2x, 10x and 100x score 13.1,
33.3 and 50; no finite speedup reaches 100. The baseline takes about 87 s and the batched
reference about 82 s on the train suite.

**Beats the reference** when, on every one of the eight workloads, it has: median fit time
(`elapsed_seconds`) strictly below the reference's median over the four repetitions; median time
inside `_update_error` (`residual_update_seconds`, which includes one trusted full read of the updated target state
per round, which must find the residuals updated where the next tree reads them: the sum
of squared residuals must drop by `(2*lr - lr^2) * sum over leaves of g^2/h` to relative `1e-6`)
strictly below the reference's (it counts only if `_update_error` runs once per tree through the
fixed `train_one`, is not replaced on the class or instance at run time, and the read relation
holds every fact row; otherwise the metric fails, without affecting the score); and RMSE within `1e-3` of the oracle on every repetition
(`rmse_error`). Equality or a loss anywhere fails.

## Running locally

```sh
cd /workspace/repo && pip install -e . && python3 -c 'import joinboost'
```

DuckDB 1.5.1, pandas 2.3.3 and scikit-learn 1.7.2 are installed (Python 3.12).

## Notes

The workloads are representative of residual-state advancement: do not special-case slices,
profiles or row counts.
