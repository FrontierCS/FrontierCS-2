# SQL Counterexample Search

## Problem
Polygon is a symbolic SQL reasoning engine that searches for a small input
database on which two SQL queries return different results. Such a database is
a counterexample to query equivalence and is useful for checking submitted SQL
answers and validating query rewrites. This task is reduced from Polygon's
official software artifact and retains its SQL parser, operator encodings,
bounded-table representation, SMT-LIB interface to Z3, and MariaDB validation
boundary. The fixed runner reads a schema, integrity constraints, a ground-truth
SQL query, a candidate SQL query, a row bound, and a time limit. Each suite also retains two paper-scale hard pairs with 20-second budgets. Each suite
contains 16 query-pair records from the artifact's LeetCode
equivalence-refutation corpus plus three query-disambiguation stress records
from D-50/D-100. The original corpus contains pairs of a
problem's ground-truth query and alternative submissions, including submissions
that passed the problem's test cases but are not equivalent. For each record,
the engine either supplies a Z3 model that is decoded into a database or reports
no counterexample from that run. A positive refutation is correct only when the
fixed MariaDB check executes both queries on the decoded database and observes
different results. The objective is to maximize the 0--100 score, with the
number of validated refutations as the primary term and quantized total search
time as the secondary term.

The complete bodies of `Environment.check` and the parallel
`Environment.disambiguate` path have been removed: parsing queries,
building their encodings, driving counterexample discovery, decoding a satisfying
model, and returning the result are solver-owned. The exhaustive and precise
fallbacks and the conflict-driven bodies of `FormulaManager` have also been
removed, together with the search-state bodies in `KnowledgeBase`. The public
classes, helper encoders, SMT AST, prover/model API, and external MariaDB
validation boundary remain so the task reconstructs Polygon's core pipeline
rather than rewriting its SQL frontend or validator.

## Design space
The editable surface owns the complete SQL-to-counterexample discovery spine
behind `Environment.check`: parser/encoder orchestration, search scheduling,
under-approximation growth, conflict learning and recovery, satisfying-model
decoding, and telemetry. `FormulaManager.search`, its exhaustive and precise
fallbacks, and the knowledge-base hooks are supporting pieces of that spine,
not independent scoring hooks. Public class names, method signatures, edit
markers, parser and encoder helpers, the SMT AST, `SMTLIBv2` model interface,
telemetry contract, and MariaDB validation behavior are fixed. Modules added under
`search_extensions/` take effect only when imported by code in an editable
region.

## Workload
The development suite contains 16 inequivalent LeetCode SQL query pairs, and
the distinct graded suite contains 16 more. These 32 records span 14 retained
problem schemas. Each suite has four pair records at each base-table row bound
2, 3, 4, and 5; every pair has a two-second symbolic-search budget. In addition,
each suite has two D-50 and one D-100 stress records. A stress record asks for
one database that splits 50 or 100 related SQL queries into two equally sized
result clusters and has a case-specific 5--12 second budget. The workloads
collectively exercise projection, filtering,
joins, grouping and aggregation, subqueries, distinctness, ordering, common
table expressions, and union.

The public smoke inputs contain one non-scored query pair and one two-query
disambiguation record. Scored query
text, record identifiers, and suite membership are held by the judge. The
workload represents SQL equivalence refutation rather than identifiers or query
strings to recognize. The development and graded suites are stratified so they
carry the same distribution, and they share no item.

## Interface
For every case the evaluator runs this complete command, with no command-line
flags, and sends exactly one UTF-8 JSON object on standard input:

```text
python3 frontier_runner.py
```

The bundled public smoke record is a real input:

```json
{
  "problem_id": 1581,
  "qid": 0,
  "bound": 2,
  "timeout_seconds": 2,
  "query": "SELECT V.CUSTOMER_ID, COUNT(V.CUSTOMER_ID) AS COUNT_NO_TRANS FROM VISITS V LEFT JOIN TRANSACTIONS T ON V.VISIT_ID = T.VISIT_ID WHERE AMOUNT IS NULL GROUP BY V.CUSTOMER_ID"
}
```

Its fields mean:

- `problem_id`: integer selecting
  `benchmarks/leetcode/schemas/<problem_id>.json`,
  `constraints/<problem_id>.yml`, and `groundtruths/<problem_id>.csv`;
- `qid`: integer identifier copied into the retained worker's CSV row;
- `bound`: maximum number of symbolic rows in each base table;
- `timeout_seconds`: integer time budget passed to `Environment` for this
  symbolic check;
- `query`: the candidate SQL query compared with the first
  `mysql_groundtruth` row for the selected problem.

`public_disambiguation_smoke.json` exercises the second workload shape:

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

For this shape, `Environment.disambiguate` must produce one database whose
MariaDB results partition the supplied queries into exactly two equal, nonempty
clusters. Scored stress records contain 50 or 100 queries; their schemas,
queries, identities, and budgets are private to the selected suite.

For that input, the associated schema record has the following actual layout
(the string `"0"` is the type token present in the retained artifact):

```json
{
  "Problem Number": "1581",
  "Tables": [
    {
      "TableName": "Visits",
      "PKeys": [],
      "FKeys": [],
      "Others": [
        {"Name": "visit_id", "Type": "int"},
        {"Name": "customer_id", "Type": "int"}
      ]
    },
    {
      "TableName": "Transactions",
      "PKeys": [],
      "FKeys": [],
      "Others": [
        {"Name": "transaction_id", "Type": "int"},
        {"Name": "visit_id", "Type": "0"},
        {"Name": "amount", "Type": "int"}
      ]
    }
  ],
  "Tuples": []
}
```

The corresponding constraint file is:

```yaml
row_dep: unique(Visits.visit_id);
         unique(Transactions.transaction_id)
col_dep: Transactions.visit_id<-Visits.visit_id
```

The ground-truth CSV layout is `qid,mysql_groundtruth`; its first data record is:

```csv
1581,"SELECT CUSTOMER_ID, COUNT(*) AS COUNT_NO_TRANS FROM VISITS WHERE VISIT_ID NOT IN (SELECT DISTINCT VISIT_ID FROM TRANSACTIONS) GROUP BY CUSTOMER_ID"
```

The fixed adapter writes exactly one JSON object to standard output. One
successful output for the public record has this shape:

```json
{"backtracks": 1, "iterations": 3, "ok": true, "refuted": true, "reported_time_seconds": 0.091873, "time_seconds": 0.092906}
```

The adapter runs the submitted search in a separate search process and never
imports submitted code itself. That process loads the third-party libraries and
builds the SQL parser's grammar, then imports the patched modules and runs the
retained worker until the outermost `Environment.check` returns
(`Environment.disambiguate` for a disambiguation record). The returned
counterexample is copied at that moment, so it must be a `dict` of lists holding
plain values (`None`, `bool`, `int`, `float`, `str`, `bytes`, `Decimal`, or
`datetime` values), and the search process then ends. It has no database access.
The adapter validates the counterexample with MariaDB afterwards.

`ok` means the adapter completed the case. `refuted` means MariaDB accepted the
decoded database as distinguishing the query pair or evenly disambiguating the
query set. `time_seconds` is wall time measured by the adapter from the import of
the patched modules until it receives the returned result, so it includes module
import and `Environment` construction. A result that does not arrive within the
case budget, or a call that returns no decision, is charged the full budget. The
duration returned by the submitted code is echoed as `reported_time_seconds` and
is not scored. `iterations` and `backtracks`
are the submitted search's counters. The numeric values vary between runs. A
runner failure instead prints an object with `ok: false` and exits nonzero.

The removed outer Python entry point is:

```python
Environment.check(self, q1, q2, use_precise_encoding=False)
```

It must parse and encode the SQL pair, search for a distinguishing bounded
database, decode a satisfying model when one is found, and return the original
five-value Polygon result tuple consumed by the retained worker. The preserved
parser, `QueryEncoder`, SMT AST, `SMTLIBv2`, schema database, and string-decoding
helpers are the intended building blocks. `FormulaManager.search(self, outputs,
ret)` remains the internal search hook; it normally returns a live prover
positioned at a satisfying model. The retained worker expects the telemetry
mapping to contain:

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

The editable supporting signatures are:

```python
FormulaManager.search_naive(self, outputs, ret)
FormulaManager.solve_precise(self, ret)
FormulaManager.backtrack(self, unsat_core, ret)
FormulaManager.add_kb(self, unsat_core)
KnowledgeBase.add_conflict(self, conflict, labels)
```

Their results are consumed only by editable search code. The fixed
`FormulaManager.dump()` contract is to return one SMT-LIB string containing the
currently selected named assertions and any assertions added in its marked
region.

From `/workspace/repo`, `bash /workspace/repo/frontier-build.sh` performs the public source/runtime
check and `bash /workspace/repo/frontier-bench.sh` runs the record above through the same
worker, decoder, and MariaDB validation path used for scored cases.

## Validity
The evaluator first validates and applies the submitted text patch. It must be
nonempty, no larger than 262,144 bytes, change at most 20 files, contain no NUL
or Git binary content, use no absolute or parent-traversal path, and touch only
the four editable paths named in `/frontier/task/README.md`. Every edit marker
must remain present exactly once, and the three existing modules must match the supplied
scaffold outside the marked regions.

Added lines are checked case-insensitively and may not contain any of these
substrings:

```text
__import__
__main__
atexit
builtins
compile(
eval(
exec(
frontier_
task-internal adapter prefix
inspect
open(
os.
pathlib
pickle
resource
socket
subprocess
sys.
/proc
/sys
```

No Python file in the editable surface, as applied, may import any of these
modules, in any import form or under any alias: `__main__`, `_io`, `atexit`,
`builtins`, `ctypes`, `gc`, `importlib`, `inspect`, `io`, `marshal`, `mmap`,
`os`, `pickle`, `posix`, `resource`, `socket`, `subprocess`, `sys`.

A patch-policy failure, patch-application failure, immutable-scaffold change,
or build failure makes the whole submission invalid and returns score zero
before workload scoring. The build requires the retained worker and adapter,
successful Python byte-compilation, a Z3 4.13 executable, and the MariaDB
server tools.

For each case, the runner must exit successfully, emit one JSON record with
`ok: true`, report finite `time_seconds` in `[0, b_w]` for that workload's
disclosed budget `b_w`, and report nonnegative
integer iteration and backtrack counts. A case timeout, crash, nonzero exit,
missing row, `ok: false`, malformed JSON, or invalid metric is recorded as
`refuted = false`, `time_seconds = b_w`, `iterations = 0`, and `backtracks = 0` for
that case. A decoded database on which MariaDB observes equal results is a valid
non-refutation with its measured time. These case-level failures are not global
patch failures; they remain in the 19-case score exactly as specified below.

## Scoring
The goal is to maximize the returned score on the 0--100 scale. Let `W` be the
19 non-hard workloads in the selected suite (16 pairs and three disambiguation cases; the two additional hard pairs affect PASS only), let `r_w` be 1 when workload `w` is
validated as refuted and 0 otherwise, let `t_w` be its measured `time_seconds`, and let
`b_w` be its budget. Define normalized time `u_w = 2 * min(b_w, max(0, t_w)) / b_w`.
A failed case has `r_w = 0` and `u_w = 2`. The evaluator computes exactly:

```text
R = sum_{w in W} r_w
T = 0.05 * ceil((sum_{w in W} u_w) / 0.05)
E = exp(-T / (|W| * 0.25))
score = max(0.0, min(100.0, 100 * (R + E) / (|W| + 1)))
```

The 0.05 normalized-time quantum and 0.25 time scale are challenge constants
(not from materials). With `|W| = 19`, each additional validated refutation adds
5 points for fixed `E`; `0 < E <= 1` supplies the quantized total-time term.

A patch that passes patch and build checks but fails every case remains a valid
scored submission with a near-zero efficiency term rather than receiving a
global invalid-submission zero. The score's time component changes when the sum
of normalized case times crosses a 0.05 boundary.

## Feedback and grading
The judge returns a score after every submission. The submission budget is in `/frontier/task/README.md`. **The graded suite is not the suite you receive
feedback on**, and neither its membership nor its query text is disclosed. The
two suites are stratified so they carry the same distribution, and they share
no item.

Alongside the continuous score, evaluation applies a separate withheld
criterion on every workload. A refutation outcome passes when it is non-inferior
to the paired measurement. For the 19 non-hard workloads, a wall-time outcome passes when

```text
candidate_time <= paired_time + max(0.03, 0.20 * paired_time)
```

The two retained paper-scale hard pairs instead require `candidate_time < 0.5 * paired_time`, with no allowance. They each have a 20-second symbolic budget and 34-second process ceiling. Both metrics must pass on all 21 workloads.

The 0.03-second absolute and 20% relative allowances are challenge constants
(not from materials). Both quantities must pass on every workload to clear the
separate criterion. The continuous score remains the optimization objective,
and a submission that does not clear the criterion is still scored normally.

Development feedback contains the submission's absolute `refuted_pairs`,
`wall_time_seconds`, `search_iterations`, and `backtracks` arrays. For each
quantity in the separate criterion, feedback also gives anonymous pass and fail
counts and percentages over the development workloads. It does not identify
the corresponding workloads or disclose the paired measurements.

## Resources
Evaluation is CPU-only and provides 2 vCPUs, 8,192 MiB of memory, and 4,096 MiB
of workspace storage. The environment supplies Python 3.11, the Z3 4.13 command
line executable and Python package 4.13.3.0, and an ephemeral MariaDB instance
using a local socket, disabled networking, and case-insensitive table names.

The configured workspace build timeout is 120 seconds. Within evaluation, the
fixed source/runtime build check has a 30-second limit. Pair checks have a
two-second budget; disambiguation stress checks have 5--12 seconds. Each
`frontier_runner.py` process receives twelve seconds of setup, validation and
cleanup headroom beyond its case budget. MariaDB initialization has a 60-second subprocess limit and a
twenty-second readiness window. One complete evaluator call has a 330-second outer
timeout.
