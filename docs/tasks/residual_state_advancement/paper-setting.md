# Residual State Advancement: paper setting

Version `r2-paper-setting-20261011`, source identity
`sha256:073ce8fe932d4b77d93ec501bda9892e9fd39c91fdd8a07421f3e8c934fdb65c`
(previous `native-standardization-1`, `sha256:5da70410...c8b517`). In-place
revision of the same `problem_id`. The three metrics, the strict per-workload
conjunction, the paired schedule, the per-leaf baseline, the reference patch and
the continuous score are unchanged. Old results stay as they are; comparisons on
this version need whole-task reruns.

## What changed

| Area | Before | Now |
| --- | --- | --- |
| Editable files | six, including `src/joinboost/executor.py` | five; the executor is trusted and byte-locked |
| Data returned to Python | unlimited | scalar types only; per query 10,000 rows, 32 columns, 40,000 values, 200,000 bytes; per fit 250,000 rows, 250,000 values, 1,000,000 bytes; strings up to 256 characters |
| Database settings | solver-controlled | `threads=2`, `memory_limit=10GB`, external access off, configuration locked |
| Fact table | 25,000-70,000 rows drawn with replacement from the 10,000-row sample | 1.03-2.14 million real, unique Favorita sales rows |
| Suites | independent seeds | disjoint date ranges: development 2017-02-02..2017-04-30, graded 2017-05-01..2017-07-30 |
| Materialised join | allowed | invalid: no relation with >= 10% of the fact rows may exceed the fact table's 5 columns by more than 4 |
| Update clock | time inside `_update_error` | plus one trusted full read of the updated target state per round, with a consistency value |
| Prediction L1 bound | `1e-3` absolute | `rows * 1e-6` |
| Oracle | refitted in every timed process | fitted once per case in an untimed process |
| Workloads per suite | 4 | 8: each fact slice under two profiles |
| Resources | 2 vCPU, 4 GiB, 600 s evaluation, 180 s per process | 2 vCPU, 16 GiB, 7,200 s evaluation, 600 s per process |

## Enforcement

Run time, in `executor.py` (not editable; the evaluator compares its bytes in
the patched tree with the pristine file):

- `DuckdbExecutor` keeps the connection behind a guard exposing only `execute`
  and `fetchall`. `fetchall` first checks the result description: more than 32
  columns, or any column that is not `BOOLEAN`, an integer up to
  `HUGEINT`/`UHUGEINT`, `FLOAT`, `DOUBLE`, `DECIMAL` or `VARCHAR`, is refused
  before a row is fetched. It then reads at most `MAX_ROWS_PER_QUERY + 1` rows
  and charges every value (8 bytes; 16 for a wide integer or decimal; 1 for
  `NULL` or a boolean; UTF-8 length for a string of at most 256 characters; any
  other Python value is refused). Limits: 10,000 rows, 40,000 values and
  200,000 bytes per query; 250,000 rows, 250,000 values and 1,000,000 bytes
  per executor. The violation is recorded, so catching the exception does not
  help.
- Database errors are re-raised as `ExecutorException` with the message cut to
  256 characters, outside the original `except` block so that neither
  `__cause__` nor `__context__` carries the original, and the bytes are charged.

The earlier guard counted rows only: `SELECT list(i) FROM range(250001) AS r(i)`
returned one row holding 250,001 values (a review finding). Export
families checked against the new guard on real DuckDB, all refused:
`list`/`array_agg`, fixed-size arrays, `STRUCT`, `MAP` (`histogram`,
`map_from_entries`), nested combinations, `UNION`, `string_agg`/`group_concat`,
`repeat`, `to_json` and the `JSON` type, `CAST(list AS VARCHAR)`, `format`,
`BLOB` (`encode`), `BIT` (`bitstring_agg`), `ENUM`, `UUID`, dates, wide results
(`PIVOT`, `unnest` of a struct), `EXPLAIN` and `DESCRIBE` output longer than
256 characters, and `error()` or conversion errors that quote data (bounded and
charged). Bit-packing into `HUGEINT` values is not refused but charged at 16
bytes per value, so it is bounded by the byte budget: at most 1,000,000 bytes
per fit, about 7% of the smallest fact table's key and residual columns.
`COPY`, `EXPORT`, `PRAGMA`, `CALL`, `INSTALL` and `LOAD` are refused by the
statement screen; `read_csv`, `read_text`, `read_blob` and `glob` fail inside
DuckDB with a permission error because external access is disabled; Python
UDFs cannot be registered because the raw connection is not reachable.

Measured maxima behind the limits (eight development workloads; reference,
per-leaf baseline, Astra, Kimi without its executor hunk, Muse): 6 rows, 6
columns, 36 values and 141 bytes per query; 12 characters per string; 3,043
rows, 8,666 values and 66,332 bytes per fit.
- `execute` accepts statements beginning with `SELECT`, `WITH`, `FROM`,
  `VALUES`, `CREATE`, `DROP`, `UPDATE`, `INSERT`, `DELETE`, `ALTER`, `BEGIN`,
  `COMMIT`, `ROLLBACK`, `EXPLAIN`, `DESCRIBE` and `PRAGMA table_info(<name>)`.

Run time, in `judge/benchmark.py`: DuckDB `threads`, `memory_limit` and
`temp_directory` are set, the relations are loaded, then
`enable_external_access=false` and `lock_configuration=true`. After the fit the
worker checks the recorded violation, that the executor is the original
`DuckdbExecutor` with the original guard, and that no attribute of the executor
module's limits or of the `Executor`/`DuckdbExecutor` classes was replaced. The
judge reads predictions on its own connection, so no exemption from the cap
exists.

Static, in `judge/evaluator.py`: the patch may not touch `executor.py`
(`allow_paths`), and added lines are screened for `duckdb`, `.conn`, `fetch*`,
`.df(`, `.arrow(`, `.pl(`, `__closure__`, `cell_contents`, `__globals__`,
`__code__`, `__dict__`, `vars(`, `eval(`, `exec(`, `compile(`, `__import__`,
`gc.` and the guard's own names.

Materialised-join guard, in `judge/benchmark.py` (`FACT_SCALE_FRACTION = 0.10`,
`FACT_EXTRA_COLUMNS = 4`, both at the top of the file): after every boosting
round and at the end of the fit the judge lists `duckdb_tables()` on its own
connection and fails the fit if a table with at least 10% of the fact rows has
more than 5 + 4 columns. Views are listed from `duckdb_views()` by column count
only; a view is counted (`COUNT(*)`) only when it is wider than the limit, so
the check costs nothing extra for narrow views. Measured width of the widest
fact-scale relation: 5 columns (+0) for the reference and the per-leaf baseline
on every development workload; 5 for Astra, Kimi and Muse and 22 (+17) for the
DeepSeek port on a 200,000-row six-table case (the three in-graph designs also
stayed within the limit on every development workload). Cost on the reference and baseline: 13-30 ms per fit (0.1-0.2% of
the fit), so it runs every round.

Update-clock probe, same file: inside the clock, after the submitted
`_update_error` returns, the trusted wrapper reads `SUM(g*g), COUNT(*)` over the
residual column of the target relation of the state the next tree consumes. A
relation that is not a table is first computed in full (`CREATE TEMP TABLE ...
AS SELECT *`) and the copy dropped. The value must equal the previous one minus
`(2*lr - lr^2) * sum_leaf(g^2/h)` to a relative 1e-6 and the count must be the
fact row count; otherwise the update time is unmeasured (the fit stays valid).
Cost on the reference and baseline: 46-107 ms per fit, inside the update time
of both sides (5-8% of the reference's update time, under 1% of its fit).

Not caught by the join guard: a materialised join split vertically into
several fact-scale tables of at most 9 columns each and read back by row
position or key, and a wide relation created and dropped inside one round. A
determined evader has to give up the single wide scan, which is what made the
design fast. Not charged by the probe: work other than producing the updated
residual column (dropping superseded tables, rebuilding auxiliary structures)
moved out of `_update_error`; it counts in the fit time only.

Violations score zero with one of five fixed reasons (four run time, one
static). The cap numbers come from measurement: on every development workload
the reference and the per-leaf baseline return at most 6 rows per query, and
2,878 and 2,968 rows per fit.

Not closed: Python has no hard privacy, so a patch that reaches the guard's
closure through a reflection route the screen does not list would bypass the
run-time cap.

## Data

Dimension tables, imputed features `f1`..`f5` and the target formula are the
JoinBoost artifact's: the task's dimension CSVs are byte-identical to
`data/favorita/` of `github.com/JoinBoost/JoinBoost` (commit `6423ce0`,
Apache-2.0 `LICENSE` at the repository root). That repository publishes only a
10,000-row fact sample.

Fact rows are real Corporación Favorita sales rows (Kaggle competition data,
reference [3] of the paper) from a public copy:

- repository `github.com/bgautijonsson/groceries`, commit
  `f79ac95b1c61ca9767f4e7d379bc2229fd8938fc`, file `train_set_2017.csv.7z`
- 71,930,146 bytes, sha256
  `86276d82a921ac0415d95a776dcf5122b17ed03a58eb8d72d6f9f09b1eef02a5`
- unpacks to `train_set_2017.csv`: 18,887,094 rows, columns
  `id,date,store_nbr,item_nbr,unit_sales,perishable`, dates 2017-02-02 to
  2017-07-30, every `(store_nbr, item_nbr, date)` unique

That repository carries no licence statement (GitHub reports no licence), and
the Kaggle competition rules were not retrievable without an account. Using
this copy is a maintainer decision.

### Key translation

The artifact dictionary-encodes its keys by sorted rank. This was established on
the artifact's own files and verified on its sample:

- `tid` equals the 0-based rank of the `transactions.csv` row ordered by
  `(date, store_nbr)`: 83,606 of 83,606 rows.
- The date code is the 0-based rank of the calendar date among the 1,684 sales
  dates from 2013-01-01 (25 December is absent each year), so 2017-02-02 is 1489
  and 2017-07-30 is 1667. The store code is `store_nbr - 1`.
- The item code is the rank of the raw `item_nbr` within the artifact's 4,034
  items, a subset of the competition's 4,100. The subset was recovered by
  aligning the artifact's `(family, class, perishable)` sequence with a public
  copy of the competition's `items.csv` (used for this analysis only, not
  shipped); family and class codes are sorted ranks too. 4,028 codes align
  uniquely. `item-code-map.csv` beside this page lists the code of each of the
  3,969 source items that has one.
- Check on the sample: of the 1,443 `sales_small.csv` rows dated strictly inside
  the source range, 1,440 are found in the source with identical store, date,
  item and `unit_sales`; the other three differ only in the last printed digit
  of a fractional `unit_sales`. Shifting the date by one day or the store by one
  finds at most 156.

Coverage of the 18,887,094 source rows by the shipped dimension tables: 54 of
54 stores, 179 of 179 dates (in `holidays.csv` and `oil.csv`), 9,589 of 9,589
store-days (in `transactions.csv`), 3,969 of 3,970 items. The one item without
a unique code (`item_nbr` 2124052) is left out: 305 rows.

`unit_sales` is the source value, including 1,263 negative (return) rows in the
source. The source has no promotion flag; `onpromotion` is the constant 2, the
code the artifact's sample carries on rows from the period before the flag was
recorded. `Y = f1 + log2(f2) - 10*f3 - 10*f4 + f5*log2(f5)` over the joined
features; the formula was recovered from the artifact's sample and reproduces
all 10,000 `Y` values to within `7.3e-12`.

### Slices

Selection rule, over the covered rows of `train_set_2017.csv` in `id` order:

| File | Date range | `id % 18` in | Rows | Bytes | sha256 |
| --- | --- | --- | ---: | ---: | --- |
| `sales-dev-0.parquet` | 2017-02-02..2017-04-30 | 0,1,2,3 | 2,054,144 | 3,617,621 | `cbb42ddc…e32dc5` |
| `sales-dev-1.parquet` | same | 4,5,6 | 1,540,608 | 3,054,505 | `77572db9…c35dd0` |
| `sales-dev-2.parquet` | same | 7,8 | 1,027,072 | 1,996,001 | `29df15bc…a86ba3` |
| `sales-dev-3.parquet` | same | 9,10,11,12 | 2,054,144 | 3,626,685 | `8b38d9a0…12eaef` |
| `sales-final-0.parquet` | 2017-05-01..2017-07-30 | 0,1,2,3 | 2,142,914 | 3,858,465 | `2d3b2de7…a9d719` |
| `sales-final-1.parquet` | same | 4,5,6 | 1,607,195 | 3,189,228 | `ee4bed24…b5469c` |
| `sales-final-2.parquet` | same | 7,8 | 1,071,466 | 2,140,588 | `463b4c1d…0ea4f` |
| `sales-final-3.parquet` | same | 9,10,11,12 | 2,142,929 | 3,902,046 | `a0343207…7998af` |

Full hashes are in `upstream.lock.json`. Each file holds `item_nbr` (artifact
code, `USMALLINT`), `tid` (`UINTEGER`) and `unit_sales` (`DOUBLE`), written by
DuckDB 1.5.1 as Parquet with ZSTD level 19 and 1,000,000-row groups. The row
selection and values are reproducible from the source file with this query;
byte-identical Parquet additionally needs the same DuckDB version:

```sql
SELECT m.item_code::USMALLINT AS item_nbr, t.tid::UINTEGER AS tid, r.unit_sales
FROM train_set_2017 r
JOIN item_code_map m ON m.item_nbr = r.item_nbr
JOIN transactions t
  ON t.store_nbr = r.store_nbr - 1
 AND t.date = 1489 + (r.date - DATE '2017-02-02')
WHERE r.date BETWEEN :first_day AND :last_day AND r.id % 18 IN (:residues)
ORDER BY r.id;
```

Development slices are under `agent-base/frontier/task/repo/data/favorita/`
(12,294,812 bytes; in the agent image and, through the pristine tree, in the
verifier image). Graded slices are under `judge/judge_data/favorita/`
(13,090,327 bytes; verifier image only). Nothing is fetched at image build.
The files are below the repository's Git LFS threshold and are committed
directly. `judge/judge_data/favorita/train_small.csv` (931,276 bytes) is no
longer used and is removed.

## Measurements

Authoring host, AMD EPYC 7R13, each run pinned to two cores, DuckDB 1.5.1 with
`threads=2`, 2026-10-11.

Reference, one run per development case, `memory_limit=10GB`, alone on its cores:

| Case | Rows | Fit s | Update s | Update share | Peak RSS MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0: d2 e8 four-table | 2,054,144 | 5.43 | 0.53 | 9.7% | 4,663 |
| 1: d3 e7 five-table | 1,540,608 | 8.33 | 0.43 | 5.2% | 6,411 |
| 2: d4 e6 six-table | 1,027,072 | 12.79 | 0.55 | 4.3% | 7,518 |
| 3: d2 e12 six-table | 2,054,144 | 8.61 | 0.74 | 8.6% | 6,588 |

The real rows are faster than uniformly generated keys of the same size (the
earlier generated-key variant of this revision measured 7.7 / 12.0 / 17.7 /
12.6 s): rows arrive in store-day order and a slice touches 4,675 store-days
rather than all 83,606. The update share is correspondingly a little lower. The
sizes were kept in the 1-2 million range.

Development suite through the revised evaluator with both guards (six
evaluations running concurrently on separate core pairs, host otherwise
lightly loaded): reference median fit 5.8-6.0 / 8.8-9.1 / 13.5-13.9 / 9.2-9.3 s;
update including the trusted read 0.60-0.63 / 0.50-0.51 / 0.62-0.63 /
0.86-0.88 s, that is about 10.5% / 5.6% / 4.5% / 9.4% of the fit (previously
1-3%); baseline total 38.7-40.1 s. One evaluation took 538-655 s wall against
the 2,400 s timeout of that suite (3.7x slack or more). Peak resident memory 7.5 GB against
`memory_limit=10GB` and 16 GiB containers. A first guarded run made while other
jobs drove the host to a load of about 50 took up to 769 s and is not used.

Reference self-noise (a byte-different copy of the reference as candidate, two
evaluations): fit ratio per workload 0.936-1.031, update ratio 0.998-1.019; the
copy cleared 7 of 8 and 6 of 8 timing cells and not the conjunction. A chance
pass of an equivalent implementation is therefore possible.

Sorted-prediction L1 per row: reference 8.5e-9, other correct patches up to
1.3e-8, against the previous bound's 1.4e-8 per row; hence `rows * 1e-6`.

Row-cap maxima (measured on the generated-key variant at 250k-4M rows; the
reference again here): at most 6 rows per query; at most 2,878 (reference) and 2,968
(per-leaf baseline) rows per fit.

## Eight workloads per suite

The four-workload replay further down showed near-parity
designs being decided by a few percent on a handful of cells. Each suite now
measures each of its four fact slices under two profiles. No data file or
dataset selector changed.

| # | Slice | Rows (dev / graded) | Depth | Rounds | Learning rate | Topology |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | 0 | 2,054,144 / 2,142,914 | 2 | 8 | 0.5 | `core_four_table` |
| 1 | 1 | 1,540,608 / 1,607,195 | 3 | 7 | 0.1 | `five_table_oil` |
| 2 | 2 | 1,027,072 / 1,071,466 | 4 | 6 | 1.0 | `six_table` |
| 3 | 3 | 2,054,144 / 2,142,929 | 2 | 12 | 0.25 | `six_table` |
| 4 | 0 | 2,054,144 / 2,142,914 | 1 | 24 | 0.5 | `core_four_table` |
| 5 | 1 | 1,540,608 / 1,607,195 | 2 | 20 | 0.1 | `five_table_oil` |
| 6 | 2 | 1,027,072 / 1,071,466 | 5 | 3 | 0.5 | `six_table` |
| 7 | 3 | 2,054,144 / 2,142,929 | 3 | 4 | 0.5 | `six_table` |

Profiles 4-7 were chosen for spread in depth (1 to 5), rounds (3 to 24) and
topology, subject to the reference's memory: it keeps every partition table of
earlier rounds, so depth-3 with 20 rounds, depth-5, and depth-2 with 24 rounds
on the 2-million-row slices reached the 10 GB DuckDB limit (peak RSS 9.5-10.1
GB) and were rejected. Measured peak RSS of the chosen new profiles: 4.7 / 7.6 /
7.7 / 5.4 GB for the reference and 4.5 / 7.4 / 7.5 / 5.1 GB for the baseline;
both pass the oracle on all of them (L1 per row at most 8.8e-9).

Repetitions: the paired scheme always ran four repetitions per workload
(`PAIR_ORDER`), as documented. The `2` in `_run_paired` and `_run_suite` is
the number of repetitions the continuous score and the sequential baseline use
(the lower of the first two), also as documented. Four are kept: an even count
keeps the alternation balanced and a median of four tolerates one stalled run.

Timing on the eight-workload development suite (two pinned cores per
evaluation; host load average 5-29 during these runs): reference median fit
about 6.3 / 9.8 / 14.9 / 9.9 / 7.8 / 13.4 / 15.4 / 7.2 s; update share 10.5% /
5.7% / 4.5% / 9.3% / 21% / 9.7% / 3.4% / 5.5%; baseline total 87-90 s. A
reference-versus-reference evaluation took 1,460-1,466 s wall (two running
concurrently), and 2,366 s on cores that turned out to be shared with another
job; evaluations of the faster replayed patches took 1,290-1,338 s. The
evaluation timeout is 7,200 s (4.9x the clean measurement, 3.0x the contended
one). The longest single process is about 50 s against the 600 s per-process
limit.

Reference self-noise on eight workloads (two evaluations, concurrent with each
other): fit ratio 0.944-1.094, update ratio 0.919-1.146; 9 of 16 and 8 of 16
timing cells cleared.

| Patch | Outcome | Fit speedup (workloads 0-7) | Update speedup (workloads 0-7) |
| --- | --- | --- | --- |
| GPT-6 Astra | valid, fails 4 update cells | 1.55 / 1.80 / 1.84 / 1.56 / 1.27 / 1.52 / 2.03 / 1.70 | 0.96 / 0.99 / 1.35 / 0.67 / 0.86 / 1.05 / 1.36 / 1.03 |
| Kimi K3, executor part dropped | valid, fails 1 update cell | 2.58 / 2.56 / 2.24 / 2.01 / 1.80 / 2.30 / 2.38 / 2.56 | 1.20 / 1.09 / 1.06 / 0.99 / 1.09 / 1.11 / 1.02 / 1.10 |
| Muse Spark 1.3 | valid, fails 1 update cell | 1.42 / 1.66 / 1.65 / 1.47 / 1.13 / 1.41 / 1.68 / 1.72 | 0.97 / 1.06 / 1.45 / 1.03 / 1.19 / 1.01 / 1.35 / 1.13 |

Sol and the DeepSeek variants are invalid as before and were not re-run on the
eight-workload suite. The four-workload replay below is kept as the record of
the previous measurement.

## Replay of five earlier final patches (four-workload suite)

Development suite, revised evaluator, real rows. Speedups are reference median
over candidate median per workload.

| Patch | Outcome | Fit speedup | Update speedup |
| --- | --- | --- | --- |
| GPT-6.1 Sol | invalid: touches `executor.py`; with that part dropped, invalid by the static screen (`.conn`, `fetchnumpy`) | - | - |
| DeepSeek | invalid: touches `executor.py` (`SET threads=1`, a batch-drop helper) | - | - |
| DeepSeek, helper moved into `app.py`, no `SET` | invalid: `submission materialised the join into a fact-scale relation` (22 columns) | - | - |
| GPT-6 Astra | valid, fails the update metric on three workloads | 1.52 / 1.76 / 1.95 / 1.52 | 0.95 / 0.95 / 1.39 / 0.64 |
| Kimi K3 | invalid: touches `executor.py` (a set of created view names) | - | - |
| Kimi K3, executor part dropped | valid, passes | 2.41 / 2.65 / 2.25 / 2.30 | 1.14 / 1.14 / 1.05 / 1.13 |
| Muse Spark 1.3 | valid, passes | 1.51 / 1.62 / 1.63 / 1.47 | 1.10 / 1.04 / 1.44 / 1.02 |

Muse's lazy union view is now computed inside the clock, so its update
advantage fell from 43-56x to 1.02-1.44x; it still passes because its view
over the leaf partitions plus the forced copy is slightly cheaper than the
reference's in-place `UPDATE`. Kimi (minus the executor hunk) failed one update
cell by 4% before the guards and clears all by 5-14% now; that margin is close
to the noise level above. Thresholds were set from the reference-side maxima
before these runs and were not adjusted afterwards.

## Publication

Agent and verifier images change (source, statement, data) and the reference
image is rebuilt for the new source tag; their catalog digests are
placeholders. The agent image grows by the development slices (12.3 MB); the
verifier image grows by development plus graded slices (25.4 MB) and loses the
0.9 MB `train_small.csv`. No image build needs network access for data: the
slices are in the build context. The dataset selector bodies
(`datasets/train.json`, `datasets/eval.json`) are unchanged, so no new dataset
image is needed: re-tag the existing dataset digest as `source-<new identity>`
and adopt all four with `--published-images`. The placeholder row shows the
builder's default `v1` dataset tag; the existing remote `v1` tag must not be
overwritten.
