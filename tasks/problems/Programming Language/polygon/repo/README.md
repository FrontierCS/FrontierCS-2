# SQL Counterexample Search

Reconstruct the counterexample search of Polygon, a symbolic SQL reasoning engine: given two SQL
queries, find a small database on which they return different results. Maximize the 0–100 score:
validated refutations first, quantized total search time second.

## Task

Polygon (Zhao, Wang and Wang, PLDI 2025) encodes SQL queries over bounded symbolic tables into
SMT and asks Z3 for a model; the model is decoded into a database, and a fixed MariaDB check runs
both queries on it. A refutation counts only when MariaDB sees different results. The parser,
operator encodings, bounded-table representation, SMT AST, `SMTLIBv2` interface to Z3 and the
MariaDB validation are retained. The bodies of `Environment.check` and `Environment.disambiguate`,
the exhaustive, precise and conflict-driven bodies of `FormulaManager`, and the search-state bodies
of `KnowledgeBase` have been removed.

## What you may change

- Only the `FRONTIER_EDIT_BEGIN`/`FRONTIER_EDIT_END` regions of `sql_testing/environment.py`,
  `sql_testing/smt/formula.py` and `sql_testing/smt/knowledgebase.py`; every marker must stay
  exactly once and the text outside the regions must match the starter.
- New or changed Python files under `sql_testing/smt/search_extensions/` (used only if imported
  from an editable region).
- At most 20 changed files; no deletions. Everything else is fixed (class names, signatures,
  parser and encoders, SMT AST, `SMTLIBv2`, telemetry contract, runner, MariaDB validation).
- Added lines may not contain, case-insensitively, any of: `__import__`, `__main__`, `atexit`,
  `builtins`, `compile(`, `eval(`, `exec(`, `inspect`, `open(`, `os.`, `pathlib`, `pickle`,
  `resource`, `socket`, `subprocess`, `sys.`, `/proc`, `/sys`.
- No editable Python file, as applied, may import (in any form or alias) `__main__`, `_io`,
  `atexit`, `builtins`, `ctypes`, `gc`, `importlib`, `inspect`, `io`, `marshal`, `mmap`, `os`,
  `pickle`, `posix`, `resource`, `socket`, `subprocess` or `sys`.

## Interface

For every case the judge runs `python3 frontier_runner.py` (no flags) and sends one UTF-8 JSON
object on standard input. A pair record (`public_smoke.json`):

```json
{
  "problem_id": 1581,
  "qid": 0,
  "bound": 2,
  "timeout_seconds": 2,
  "query": "SELECT V.CUSTOMER_ID, COUNT(V.CUSTOMER_ID) AS COUNT_NO_TRANS FROM VISITS V LEFT JOIN TRANSACTIONS T ON V.VISIT_ID = T.VISIT_ID WHERE AMOUNT IS NULL GROUP BY V.CUSTOMER_ID"
}
```

- `problem_id` selects `benchmarks/leetcode/{schemas/<id>.json, constraints/<id>.yml,
  groundtruths/<id>.csv}` (CSV layout `qid,mysql_groundtruth`); `qid` is copied into the CSV row;
- `bound`: maximum symbolic rows per base table; `timeout_seconds`: the budget `b_w`;
- `query`: compared with the first `mysql_groundtruth` row.

A disambiguation record (`public_disambiguation_smoke.json`) asks `Environment.disambiguate` for
one database whose MariaDB results split the queries into exactly two equal, nonempty clusters:

```json
{
  "kind": "disambiguation",
  "bound": 2,
  "timeout_seconds": 5,
  "schema": [{"TableName": "employees", "PKeys": [{"Name": "emp_id", "Type": "int"}], "FKeys": [], "Others": [{"Name": "age", "Type": "int"}]}],
  "queries": [
    "SELECT emp_id FROM employees WHERE age > 30",
    "SELECT emp_id FROM employees WHERE age >= 30"
  ]
}
```

`Environment.check(self, q1, q2, use_precise_encoding=False)` must parse and encode the pair,
search for a distinguishing bounded database, decode a satisfying model, and return Polygon's
original five-value result tuple consumed by the retained worker. The returned counterexample must
be a `dict` of lists of plain values (`None`, `bool`, `int`, `float`, `str`, `bytes`, `Decimal`,
`datetime`). `FormulaManager.search(self, outputs, ret)` is the internal search hook; it normally
returns a live prover positioned at a satisfying model. The other editable entry points:

```python
FormulaManager.search_naive(self, outputs, ret)
FormulaManager.solve_precise(self, ret)
FormulaManager.backtrack(self, unsat_core, ret)
FormulaManager.add_kb(self, unsat_core)
KnowledgeBase.add_conflict(self, conflict, labels)
```

`FormulaManager.dump()` (fixed) returns one SMT-LIB string with the currently selected named
assertions plus any assertions added in its marked region. The telemetry mapping must contain:

```text
iters                  nonnegative integer
backtracks             nonnegative integer
type_2_backtracks      nonnegative integer
unsat_core_sizes       list
M_sizes                list
solving_time_per_iter  list
num_nodes_changed      list
ast_size               nonnegative integer
```

The runner prints one JSON object, for example:

```json
{"backtracks": 1, "iterations": 3, "ok": true, "refuted": true, "reported_time_seconds": 0.091873, "time_seconds": 0.092906}
```

The search runs in its own process without database access; it ends when the outermost
`Environment.check` (`disambiguate`) returns, and MariaDB validates the returned database
afterwards. `refuted` means MariaDB accepted it. `time_seconds` is measured by the adapter
from the import of the patched modules until the result arrives (it includes module import and
`Environment` construction); `reported_time_seconds` is your own and is not scored. A runner
failure prints `ok: false` and exits nonzero.

## Correctness

- A patch that breaks a rule above is rejected (score 0).
- The build check (30-second limit) requires the retained worker and adapter, successful Python
  byte-compilation, a Z3 4.13 executable and the MariaDB server tools; a build failure scores 0.
- Per case, the runner must exit successfully and emit one record with `ok: true`, finite
  `time_seconds` in `[0, b_w]` for the case budget `b_w`, and nonnegative integer iterations and
  backtracks. A timeout (budget + 12 s), crash, nonzero exit, missing row, `ok: false`,
  malformed JSON or invalid metric, a result that arrives after `b_w`, a result line over
  131,072 bytes (`RESULT_MAX_BYTES`; never truncated), or a call that raises or returns no
  decision, counts as `refuted = false`, `time_seconds = b_w` for that case. A decoded database
  on which MariaDB sees equal results is a valid non-refutation with its measured time.

## Workload

| record kind | per suite | budget `b_w` |
| --- | --- | --- |
| LeetCode query pair, row bounds 2, 3, 4, 5 (four each) | 16 | 2 s |
| D-50 / D-100 disambiguation stress record (50 or 100 queries, two equal clusters) | 3 (two D-50, one D-100) | 5–12 s |
| paper-scale hard pair (PASS only, not in the score) | 2 | 20 s (34 s process ceiling) |

The development (train) and final suites have the same composition and distribution and share no
item; together their 32 pairs span 14 problem schemas. The workloads exercise projection,
filtering, joins, grouping and aggregation, subqueries, distinctness, ordering, common table
expressions and union. Scored query text, identifiers and suite membership are private; the two
public smoke inputs are not scored. Do not recognize workloads by identifier or query text; the
final suite is different.

## Scoring

Let `W` be the 19 non-hard workloads (16 pairs, 3 disambiguation records), `r_w` = 1 if workload
`w` is validated as refuted, else 0, `t_w` its measured `time_seconds`, and `u_w = 2 *
min(b_w, max(0, t_w)) / b_w` (a failed case has `r_w = 0`, `u_w = 2`):

```text
R = sum_w r_w
T = 0.05 * ceil((sum_w u_w) / 0.05)
E = exp(-T / (|W| * 0.25))
score = max(0, min(100, 100 * (R + E) / (|W| + 1)))
```

Each validated refutation adds 5 points; `0 < E <= 1`. All 19 refuted in zero time scores 100;
nothing refuted with every case at its budget scores about 0.002.

**Beats the reference** when, on all 21 workloads: refutation `r_c >= r_ref`, and for the 19 non-hard
workloads `t_c <= t_ref + max(0.03, 0.20 * t_ref)`, for the two hard pairs `t_c < 0.5 * t_ref`
(no allowance). The reference has a larger process budget than the candidate and refutes every
workload (otherwise the evaluation fails without a score), so beating it requires refuting all 21.

## Running locally

```sh
cd /workspace/repo
bash frontier-build.sh   # the judge's build check
bash frontier-bench.sh   # runs the public pair through the scored worker, decoder and MariaDB validation
```

## Environment

CPU only, 2 vCPUs and 8 GiB of memory. Python 3.12, Z3 4.13 (`z3` executable and `z3-solver`
4.13.3.0), and an ephemeral MariaDB instance on a local socket with networking disabled and
case-insensitive table names.
