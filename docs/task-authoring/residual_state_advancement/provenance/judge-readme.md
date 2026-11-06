# Residual State Advancement

## Problem

JoinBoost is a Python library for training decision trees and tree ensembles inside a database. It expresses training as SQL over normalized relations, allowing a model to use their logical join without first exporting a denormalized training table. This task uses the reduced relations derived from the 2017 Corporacion Favorita grocery-sales forecasting data and distributed with the authors' JoinBoost artifact: five dimension tables and a 10,000-row sales fact sample. The fact table of each workload holds one to two million real Corporacion Favorita sales rows, keyed to those dimension tables, and the evaluator trains RMSE gradient-boosting ensembles on three join-graph topologies inside DuckDB; the aligned materialized join used to check the result is private and is unavailable during timed fitting. A correct fit must produce the requested number of ordinary tree definitions and match the RMSE and prediction multiset of the materialized-join oracle. As in the paper, training runs in the database: joins and aggregations are executed by the DBMS through the supplied executor, and only small aggregated results return to Python. The training state observed by each tree must represent the residual target after all preceding trees, and the fitted model must retain reusable training state. The objective is to maximize the score by minimizing the total measured `fit` time while satisfying every model, numerical, and submission-validity requirement. The body of `GradientBoosting._update_error`, which advances training state after a completed tree, has been removed and must be supplied.

## Your workspace

`/app` is a git repository containing the JoinBoost Python package, packaging files, a submission helper, and the reduced Favorita data. Source is under `src/joinboost/`, and data plus the fact-table loader are under `data/favorita/`. The only files that may change are:

```
src/joinboost/app.py
src/joinboost/semiring.py
src/joinboost/aggregator.py
src/joinboost/cjt.py
src/joinboost/joingraph.py
```

All other files, package interfaces, input data, and harness behavior must remain unchanged. In particular `src/joinboost/executor.py` is the trusted executor: it is not editable, the judge checks that the patched tree still carries its exact bytes, and it must not be replaced, wrapped or bypassed at run time. The judge verifies the submitted patch against this literal editable surface.

## Design space

The task covers the residual-state transition performed by `GradientBoosting._update_error` and supporting changes within the five editable files. The public fitting, prediction, and RMSE interfaces, class inheritance, and generic tree-growth entry points are fixed. The input schemas, topology definitions, prediction shape, and numerical meaning are also fixed. Each boosting round after the first must consume state incorporating every preceding tree, and `cjt` must remain present after `fit`.

## Database rules

The paper's setting is part of the task. Solver code talks to the database only through the executor object it is given (`self.cjt.exe`, a `DuckdbExecutor`), using its existing methods (`execute_spja_query`, `update_query`, `_execute_query`, `case_query`, `delete_table`, `rename_column`, `get_schema`, ...).

- **Joins and aggregations run in the DBMS.** Messages, split statistics, partitions and residual updates are SQL over the normalized relations. The data may be reorganized freely inside the database (new tables, views, columns, indexes).
- **Only aggregated results return to Python.** What is limited is the amount of data a fit brings back, not only the number of rows. All limits are constants in `executor.py`:
  - Every fetched column must be a plain scalar: `BOOLEAN`, an integer type up to `HUGEINT`/`UHUGEINT`, `FLOAT`, `DOUBLE`, `DECIMAL`, or `VARCHAR`. `LIST`, `ARRAY`, `STRUCT`, `MAP`, `UNION`, `BLOB`, `BIT`, `JSON`, `ENUM` and every other type are refused before any row is fetched.
  - One query may return at most `MAX_ROWS_PER_QUERY = 10,000` rows, `MAX_COLUMNS_PER_QUERY = 32` columns, `MAX_CELLS_PER_QUERY = 40,000` values and `MAX_BYTES_PER_QUERY = 200,000` bytes. A returned string may have at most `MAX_TEXT_LENGTH = 256` characters.
  - All queries issued through one executor during a fit may return at most `MAX_ROWS_PER_SESSION = 250,000` rows, `MAX_CELLS_PER_SESSION = 250,000` values and `MAX_BYTES_PER_SESSION = 1,000,000` bytes in total. A value is charged 8 bytes (16 for a wide integer or a decimal, 1 for `NULL` or a boolean) and a string its UTF-8 length.
  - A database error reaches solver code as an `ExecutorException` whose message is cut to 256 characters and charged to the byte budget.

  The supplied code returns at most 6 rows, 6 columns, 36 values and 141 bytes per query, strings of at most 12 characters, and fewer than 3,100 rows, 8,700 values and 67,000 bytes per fit on these workloads; the fact tables hold 1,027,072 to 2,142,929 rows. Exceeding a limit raises `ExecutorPolicyError` and makes the submission invalid, also when the exception is caught.
- **Database settings are fixed.** The judge sets `threads=2`, `memory_limit='10GB'` and a spill directory, disables external access (no file reads or writes, no `ATTACH`, no extensions) after the relations are loaded, and locks the configuration. The executor accepts only statements that begin with `SELECT`, `WITH`, `FROM`, `VALUES`, `CREATE`, `DROP`, `UPDATE`, `INSERT`, `DELETE`, `ALTER`, `BEGIN`, `COMMIT`, `ROLLBACK`, `EXPLAIN` or `DESCRIBE` (plus its own `PRAGMA table_info(<table>)`); `SET`, `RESET`, other `PRAGMA`, `ATTACH`, `COPY`, `INSTALL`, `LOAD`, `CALL` and every other statement kind raise `ExecutorPolicyError` and make the submission invalid.
- **The join is not materialised.** Training runs over the normalized relations. No relation that holds at least 10% of the fact table's rows (table or view, whenever it exists at a check) may have more than 4 columns beyond the fact table's own 5, that is, more than 9 columns. The judge checks its catalog after every boosting round and at the end of the fit. The supplied implementation never creates a fact-scale relation wider than the fact table.
- **The residual update is complete when `_update_error` returns.** Before the update clock stops, the judge reads the updated target state in full: the residual (gradient) column of the target relation of the state the next tree will consume (the returned state's, or else `self.cjt`'s). If that relation is not a table (a view, for example), the trusted read first computes all of it into a temporary table and discards the copy. The sum of squared residuals it reads must equal the value before the update minus `(2*lr - lr^2) * sum over leaves of g^2/h`, to a relative `1e-6`, and the relation must hold every fact row; otherwise the update did not happen where the next tree reads it and the update time is unmeasured.
- **No other route to the data.** The executor holds the DuckDB connection behind a guard that exposes only `execute` and a data-capped `fetchall`. Added code must not open or reach a DuckDB connection, call result-export APIs (`fetch*`, `.df()`, `.arrow()`, ...), or use reflection to uncover the guarded connection; the patch text is screened for these and the fitted run is checked for a replaced or modified executor.

## Workload

The development and graded suites each contain eight deterministic cases: each of the four fact slices is measured under two profiles. Their `(fact slice, rows, max_depth, n_estimators, learning_rate, topology)` profiles are (development row counts; the graded slices are about 4% larger):

```
(0, 2054144, 2,  8, 0.5,  core_four_table)
(1, 1540608, 3,  7, 0.1,  five_table_oil)
(2, 1027072, 4,  6, 1.0,  six_table)
(3, 2054144, 2, 12, 0.25, six_table)
(0, 2054144, 1, 24, 0.5,  core_four_table)
(1, 1540608, 2, 20, 0.1,  five_table_oil)
(2, 1027072, 5,  3, 0.5,  six_table)
(3, 2054144, 3,  4, 0.5,  six_table)
```

For every profile, `num_leaves = 2 ** max_depth` and `partition_early = True`. Depth runs from 1 to 5 (2 to 32 leaves) and the number of boosting rounds from 3 to 24. `five_table_oil` is the graph where `oil` joins directly to `transactions` on `date`; `six_table` is the full graph. These profiles, topology assignments, and parameter values are challenge constants (not from materials).

The fact rows are real sales rows of the Corporacion Favorita Grocery Sales Forecasting data (the Kaggle competition data the paper uses, its reference [3]), taken from a public copy: `train_set_2017.csv.7z` in `github.com/bgautijonsson/groceries` at commit `f79ac95b1c61ca9767f4e7d379bc2229fd8938fc` (sha256 `86276d82a921ac0415d95a776dcf5122b17ed03a58eb8d72d6f9f09b1eef02a5`; 18,887,094 rows dated 2017-02-02 to 2017-07-30). That repository carries no licence statement. The dimension tables, their imputed features `f1`..`f5`, and the target formula come from the JoinBoost artifact (Apache-2.0).

Every fact row is a distinct `(item_nbr, tid)` pair, that is, a distinct item, store and date; no fact row is repeated. Keys are translated to the artifact's codes exactly as the artifact encodes them: `item_nbr` is the item's code in `items.csv`, and `tid` is the `transactions.csv` row of the sale's store and date. All 54 stores, all 179 dates and all 9,589 store-days of the source rows exist in the dimension tables; 3,969 of the 3,970 items do, and the 305 rows of the remaining item are left out. `unit_sales` is the real value. The source has no promotion flag, so `onpromotion` is the constant `2`, the artifact's code for a flag that was not recorded; it is not a feature. The target is the formula of the artifact's own sample (it reproduces `sales_small.csv` to within `1e-11`): `Y = f1 + log2(f2) - 10*f3 - 10*f4 + f5*log2(f5)` over the joined `items.f1`, `holidays.f2`, `oil.f3`, `stores.f4` and `transactions.f5`.

The development suite uses rows dated 2017-02-02 to 2017-04-30 and the graded suite rows dated 2017-05-01 to 2017-07-30, so the two suites share no row and no store-day. Within a suite, the four fact slices take disjoint rows: those whose source row id modulo 18 is in `{0,1,2,3}`, `{4,5,6}`, `{7,8}` and `{9,10,11,12}` respectively, in source order. The development fact rows are in `data/favorita/sales-dev-0.parquet` to `sales-dev-3.parquet` (columns `item_nbr`, `tid`, `unit_sales`; 2,054,144, 1,540,608, 1,027,072 and 2,054,144 rows), and `data/favorita/load_sales.py` is the judge's loader, verbatim. The graded fact rows are not distributed; their slices have 2,142,914, 1,607,195, 1,071,466 and 2,142,929 rows.

Each case runs four times for the candidate and four times for the reference, each time in a fresh process. Within each of the four repetitions, every case runs one reference process and one candidate process back to back, alternating in a fixed pre-recorded order: reference first in the first and third repetitions, candidate first in the second and fourth. Correctness is checked on every run. The workload is representative of the residual-state contract rather than a set of records to recognize.

## Interface

The public data consists of six headered CSV files. Counts exclude the header. Below, each schema is followed by the first real record, with fields labelled in CSV column order. The fact table of every workload has exactly the schema of `sales_small.csv`:

```
sales_small.csv — 10,000 rows
item_nbr=451, unit_sales=4.0, onpromotion=2, tid=17298, Y=-5739.103112581849

items.csv — 4,034 rows
item_nbr=1192, family=12, class=28, perishable=0, f1=958

transactions.csv — 83,606 rows
date=229, store_nbr=7, transactions=4, tid=10574, f5=955

stores.csv — 54 rows
store_nbr=38, city=3, state=0, stype=1, cluster=5, f4=243

holidays.csv — 1,684 rows
date=1040, htype=4, locale=2, locale_name=15, transferred=1, f2=612

oil.csv — 1,684 rows
date=1506, dcoilwtico=5, f3=450
```

`sales_small.csv` has the schema of the fact relation `sales`, and `Y` is its regression target. All values in these reduced files are numeric. The common four-table joins are:

```
sales.item_nbr        = items.item_nbr
sales.tid             = transactions.tid
transactions.store_nbr = stores.store_nbr
```

The `six_table` graph additionally joins `transactions.date = holidays.date` and `holidays.date = oil.date`. The `five_table_oil` graph instead adds only `transactions.date = oil.date`. The feature lists, in evaluator order, are:

```
core_four_table:
transactions, f5, city, state, stype, cluster, f4,
family, class, perishable, f1

five_table_oil:
dcoilwtico, f3, transactions, f5, city, state, stype, cluster, f4,
family, class, perishable, f1

six_table:
htype, locale, locale_name, transferred, f2, dcoilwtico, f3,
transactions, f5, city, state, stype, cluster, f4,
family, class, perishable, f1
```

The evaluator loads the fact rows into the table `sales` of an in-memory DuckDB database and keeps the aligned joined table private. Implement the existing method:

```python
def _update_error(self, leaf_nodes, learning_rate):
    """Advance training state for the next boosting round."""
```

`leaf_nodes` is the completed tree's list of `CJT` leaves. `learning_rate` is that tree's configured multiplier. The method is called after every tree, including the final tree. It may mutate `self.cjt` and return `None`, or return the state that must replace `self.cjt`; its caller is:

```python
residual_state = self._update_error(self.leaf_nodes, self.learning_rate)
if residual_state is not None:
    self.cjt = residual_state
```

For each case, the evaluator invokes the private worker with every flag below. Angle-bracketed values are selected from the corresponding workload profile:

```bash
python3 <private-worker-path> \
  --tree <patched-tree> \
  --fact <sales-dev-0.parquet-to-sales-dev-3.parquet-or-a-graded-slice> \
  --rows <row-count-of-the-slice> \
  --depth <1-to-5> \
  --estimators <3-to-24> \
  --learning-rate <0.5-or-0.1-or-1.0-or-0.25> \
  --topology <core_four_table-or-five_table_oil-or-six_table> \
  --oracle-cache <private-oracle-file>
```

After fitting, `model.model_def` must be a list of exactly `n_estimators` ordinary `CASE` tree definitions. Each tree must contain between one and `num_leaves` leaves; every leaf must have a finite prediction and a nonempty predicate list. `model.constant_` must be finite, and `model.cjt` must exist as reusable training state. On the private joined table, the judge calls the supplied `predict(..., input_mode="FULL_JOIN_JG", output_mode="WRITE_TO_TABLE")` and reads the resulting `prediction` column on its own connection; it must hold one prediction per fact row. The first value returned by `compute_rmse("train")` is the model's RMSE.

The worker writes one JSON object to standard output. A real-shaped record is:

```json
{"database_queries": 1876, "elapsed_seconds": 8.3, "estimators": 7, "peak_rss_mb": 6411.0, "prediction_l1": 0.004, "residual_update_calls": 7, "residual_update_seconds": 0.43, "rmse_error": 6e-10, "rows": 1540608, "rows_returned_max": 6, "rows_returned_total": 1276, "cells_returned_max": 36, "cells_returned_total": 3668, "bytes_returned_max": 134, "bytes_returned_total": 26936, "columns_returned_max": 6, "text_returned_max": 12, "topology": "five_table_oil"}
```

`elapsed_seconds` covers `model.fit` and fitted-model validation; the private joined table is registered only after the timer stops. `residual_update_seconds` is the part of that interval spent inside `_update_error` calls, each including one trusted full read of the updated target state (null when the update clock is not trusted, see the separate reference verdict), and `residual_update_calls` counts those calls. `rmse_error` is the absolute difference between candidate and oracle RMSE. `prediction_l1` is the sum of absolute elementwise differences between the sorted candidate and oracle prediction arrays. Sorting ignores row order but does not remove equal predictions. `rows_returned_max` and `rows_returned_total` are the largest single result and the total number of rows the executor returned to Python during the fit; the `cells_`, `bytes_`, `columns_` and `text_` fields report values, bytes, result width and string length the same way. `database_queries` counts executed statements, and `peak_rss_mb` is the worker's peak resident memory. All of these are diagnostics.

## Validity

The judge archives the submitted artifact, then requires a nonempty, readable UTF-8 git patch that applies to the pristine package. A patch is invalid if it exceeds 131,072 bytes, changes more than five files, changes a path outside the editable list, contains binary content or a path escape, fails to apply, adds prohibited judge-introspection code, or adds code that reaches the database outside the trusted executor (see Database rules). Added code must not inspect evaluator arguments, environment or judge paths, clocks, runtime modules, frames, stacks, object registries, or filesystem contents to identify judge artifacts, and it must not dynamically replace `fit`.

Class inheritance and the following supplied methods are structurally fixed: `fit`, `_fit`, and `predict` on `DummyModel`; `fit`, `_fit`, `train_one`, `_build_model`, `compute_rmse`, `predict`, `_build_tree`, and `_get_best_split` on `DecisionTree`; and `fit`, `_fit`, `train_one`, `_build_model`, `compute_rmse`, `predict`, and `_build_tree` on both `GradientBoosting` and `RandomForest`, including inherited definitions. The worker requires exactly one call through the ordinary `train_one` and `_build_tree` path for each requested estimator and rejects instance-level replacements of either method.

The patched package must import, every fit must complete with the model structure described above, and each benchmark process must finish within 600 seconds with finite output metrics. On each of the thirty-two ordinary-suite candidate runs, the candidate prediction array must have the oracle's length, `rmse_error` must be at most `1e-3`, and `prediction_l1` must be at most `rows * 1e-6` (a mean absolute difference of `1e-6` per row; the supplied implementation's own floating-point noise is below `1e-8` per row). Every prediction value must be preserved even though comparison is order-independent. A fit that exceeds a data limit or fetches a refused type, issues a disallowed statement, replaces or bypasses the trusted executor, or materialises the join is invalid and scores zero with one of four fixed reasons: `submission returned more data to Python than the trusted executor allows`, `submission issued a statement that the trusted executor does not allow; database settings are fixed`, `submission replaced or bypassed the trusted executor`, or `submission materialised the join into a fact-scale relation` (no relation with at least 10% of the fact rows may carry more than 4 columns beyond the fact table's own). Patch or import failures, crashes, timeouts, policy violations, malformed output, invalid fitted models, and numerical mismatches score zero before performance is considered.

## Scoring

The goal is to maximize a score from 0 to 100. Index the eight cases by `i`. Let `t_{i,1}` and `t_{i,2}` be the candidate's first two measured fit times for case `i` (from the first two paired repetitions), and let `r_{i,1}` and `r_{i,2}` be the paired comparator's first two measured fit times for that case on the same suite and host. Define

```
T = sum_i min(t_{i,1}, t_{i,2})
B = the same sum for the supplied per-leaf baseline, measured on this suite
L = max(0, log10(B / T))
score = 100 * L / (2 + L)
```

`B` is the straightforward correct implementation shipped as `baseline_naive.patch`:
it issues one `UPDATE` per leaf where the withheld version builds a single batched
`CASE` statement. Same SQL, same semantics, `N` queries instead of one. It is
the zero of the scale, so matching it or running slower scores `0`. Two
improvement decades are the half-saturation point: a hundred times faster scores
`50`. Every finite improvement remains below `100`, with no finite upper
clipping; larger speedups remain strictly ordered as the score approaches
`100`. The legacy `score_unbounded` output carries the same naturally bounded
score. This formula applies when `T > 0`, `B > 0`, and every correctness and
validity gate passes; otherwise the score is `0`. Because `0` is what a correct
submission matching the baseline earns, correctness is reported separately in the
`correct` metric from the RMSE and L1 tolerances rather than inferred from the
score. The two-decade half-saturation is a challenge definition (not from
materials). There is no passing-score or early-stopping threshold beyond the
validity gates.

### Separate reference verdict

A separate `beats_reference` verdict accompanies the continuous score. For each
of the eight selected workloads, the candidate must satisfy every existing model
and numerical validity gate and must clear three per-workload metrics:

- `elapsed_seconds`: its median complete-fit time over the four interleaved
  repetitions must be strictly less than the optimized reference's median over
  its four repetitions in the same schedule;
- `residual_update_seconds`: its median residual update time over the same four
  repetitions (the total time spent inside `GradientBoosting._update_error`
  across all boosting rounds of one fit, measured by the judge around the fixed
  `train_one` call site) must be strictly less than the reference's. The
  residual-update time includes one trusted full read of the updated target
  state per round, charged identically to candidate and reference (see Database
  rules). It counts only when every round goes through that call site exactly
  once, the method is not replaced on the class or the instance, and every
  trusted read finds the updated residuals; otherwise it is unmeasured and
  fails, without affecting validity or the score;
- `rmse_error`: the RMSE difference from the materialized-join oracle must be at
  most `1e-3` on every repetition.

Equality or a loss on any workload or metric makes the overall verdict fail.
Improvements on other workloads or metrics, the summed runtime, and the
continuous score cannot compensate. Reference measurements, workload identities
attached to decisions, and final-suite measurements remain private; development
feedback reports only the aggregate passed/failed counts and percentages for
each metric.

## Calibration

The supplied incomplete program raises `NotImplementedError` during its first call to the removed method, so it cannot finish a benchmark fit and scores `0`. The first meaningful calibration point is validity: all thirty-two candidate runs must finish, every fit must contain the requested ensemble and reusable state, and both numerical errors must pass on every run.

The zero of the scale is `baseline_naive.patch`, a complete and correct implementation of the removed method: it issues one `UPDATE` per leaf where a batched form would build a single `CASE`. On the development suite it trains in about `87` seconds across the eight cases on the authoring host (2 pinned cores), against about `82` seconds for the batched reference (on the two depth-1 cases the per-leaf form is not slower); in the reference, the residual update, including the trusted read, is about 10%, 6%, 5%, 9%, 21%, 10%, 3% and 6% of the fit on the eight cases. It is a real submission, not a stub — it passes every validity and numerical gate — so `0` means "no faster than the obvious way", not "broken".

For orientation on the scale rather than as targets, a `2x` speedup over the per-leaf baseline scores about `13.1`, `5x` about `25.9`, `10x` about `33.3`, `50x` about `45.9`, and `100x` exactly `50`. Evaluator metrics report `baseline_seconds`, `speedup_over_baseline` and `speedup_over_reference` beside the score.

The evaluator defines no fixed timing noise floor. The continuous score limits noise by taking the lower of the first two repetitions of each case; the reference comparison interleaves candidate and reference processes and compares medians of four repetitions, so slow drift on the host affects both sides equally. Treat a timing change as meaningful only when it persists across repeated full-suite submissions; correctness tolerances are hard gates rather than score terms after they pass.

## Feedback and grading

The judge returns a score after every submission. At most 100 development submissions are allowed. **The graded suite is not the suite you receive feedback on**, and its fact rows are not distributed. Development and graded suites have the same four workload profiles on disjoint date ranges of the sales data.

## Submission

Submit a nonempty git patch at `/app/solution.patch`. After editing the package, run:

```bash
bash /app/make_submission.sh
bash /app/submit.sh
```

`make_submission.sh` writes the binary-safe tracked diff against the supplied base for the five editable files. It rejects any tracked change outside that surface rather than silently omitting it, and it rejects an empty patch.

## Resources

The environment provides 2 vCPUs, 16 GiB of memory, and 4 GiB of storage. It uses Python 3.12, DuckDB 1.5.1, pandas 2.3.3, and scikit-learn 1.7.2. DuckDB runs with `threads=2` and `memory_limit='10GB'`; the supplied implementation peaks at about 7.5 GB of resident memory on its most demanding case. The overall evaluation timeout is 7,200 seconds, the build/setup timeout is 300 seconds, and each benchmark process has a 600-second timeout.
