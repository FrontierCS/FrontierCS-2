# Reusable Library Candidates

## Optimization priority

Your primary objective is **PASS on every required workload and every required
metric**, with all correctness and validity requirements satisfied. The continuous
score is a secondary optimization objective; a higher score cannot compensate for
any failed PASS condition. Do not replace a solution known to satisfy all tested
PASS conditions with one known to fail them merely to increase its score.
Exploratory trials may regress; if no solution passes yet, work on the unmet
conditions and retain a valid submission as a fallback.

Successful verifier execution or a positive score does not mean scientific PASS.
Use the reported per-metric pass/fail progress to assess the tested workloads.
Train feedback covers only its evaluated workloads; passing train does not prove
final PASS. Final independently evaluates the submitted artifact on its specified
suite. The scoring formula, validity rules, and PASS comparisons below are unchanged.

## Problem
BABBLE is a Rust library-learning system that refactors a corpus of related programs by introducing reusable function definitions and replacing program fragments with calls to those definitions. Library learning measures a refactored corpus by counting abstract-syntax-tree (AST) nodes in its programs, library definitions, and calls. BABBLE stores expressions that are equivalent under supplied equations in an e-graph, where one e-class can contain several forms of the same expression; this task retains the program-compression benchmark from the BABBLE project. The input corpora are JSON search-frontier exports from DreamCoder, a program-synthesis system: each frontier contains alternative programs for one typed synthesis problem, and the benchmark processes all frontiers in a corpus together. Evaluation covers list, physics, text, logo, and towers corpora, both with and without domain equations. The benchmark writes one CSV row containing the corpus's initial and final AST costs, their ratio, elapsed time, and number of selected libraries. Every returned library candidate must preserve the meaning of the program fragment it replaces; a correct run must also satisfy the candidate constant and arity contracts, terminate successfully, and produce exactly one internally consistent CSV record. The secondary continuous objective is to maximize a 0--100 score computed only from corpus AST-cost reduction; end-to-end runtime is reported and used independently in the reference-comparison gate. The candidate-construction body of `LearnedLibrary::new` and the paired-state-construction body of `Dfta::cross_over` have been removed and must be supplied.

## Design space
The editable area consists of the two removed bodies and supporting Rust code under `src/learn_extensions/`. `LearnedLibrary::new` receives the saturated e-graph, the constant and arity settings, and the precomputed co-occurrence relation, and it returns the fixed candidate-set representation. `Dfta::cross_over` receives an immutable tree automaton and returns the fixed paired-operation, paired-state automaton representation. Public types, function signatures, marker-delimited boundaries, constructor checks, and the fields and indexes of `LearnedLibrary` and `Dfta` are fixed. Parsing, equation saturation, co-occurrence analysis, candidate deduplication, library selection, extraction, cost measurement, timing, and CSV reporting are outside the editable surface. Files under `src/learn_extensions/` do not enlarge the two marker-delimited regions or make any other path mutable.

## Workload
The development suite has five artifact corpora, one from each domain. The distinct graded suite has ten, two from each domain. The list, text, logo, and towers inputs are unchanged bundled files. Physics uses a fixed 12-frontier prefix of each bundled file to keep repeated timing bounded; each retained frontier and program record is unchanged. All 15 evaluation inputs have different SHA-256 content hashes.

| Suite/domain | Corpora | JSON bytes | Frontiers | Total programs | DSL productions |
| --- | ---: | ---: | ---: | ---: | ---: |
| development/list | 1 | 14,560 | 8 | 14 | 22 |
| development/physics | 1 | 17,463 | 12 | 60 | 15 |
| development/text | 1 | 16,059 | 7 | 14 | 24 |
| development/logo | 1 | 34,634 | 28 | 56 | 27 |
| development/towers | 1 | 16,695 | 15 | 24 | 14 |
| graded/list | 2 | 16,307--21,531 | 9--14 | 17--26 | 22--23 |
| graded/physics | 2 | 17,717--22,237 | 12 | 59--60 | 15--23 |
| graded/text | 2 | 13,790--28,388 | 5--14 | 10--28 | 24--26 |
| graded/logo | 2 | 28,826--63,154 | 23--48 | 44--95 | 27--33 |
| graded/towers | 2 | 7,735--45,828 | 4--36 | 8--72 | 14--16 |

The judge evaluates corpora sequentially. Each corpus is placed at `<case-root>/<domain>_eval/input.json` and is run three times; the discrete outputs must agree and runtime is the median. The fixed settings are the domain's artifact equation set when one exists (list and physics; the other domains intentionally have none), beam size 64, one library per step, BABBLE mode, two rounds, maximum library arity 2, all programs from every frontier, a library-rewrite iteration limit of 1, and one Rayon worker. The corpora are representative program-synthesis inputs, not identifiers or strings to recognize. Development and grading use content-disjoint source files with the same five-domain coverage; suite membership and source-run identity are not disclosed to the solver.

## Interface
The first removed entry point is:

```rust
pub fn new<A: Analysis<AstNode<Op>>>(
    egraph: &EGraph<AstNode<Op>, A>,
    learn_constants: bool,
    max_arity: Option<usize>,
    co_occurrences: CoOccurrences,
) -> LearnedLibrary<Op, (Id, Id)>
```

It must return the existing `LearnedLibrary` representation, whose candidates are `PartialExpr<Op, Var>` values. A `Hole(var)` is a candidate argument, and repeated occurrences of the same `Var` denote the same argument. If `learn_constants` is false, every candidate must contain at least one hole; if `max_arity` is `Some(n)`, a candidate may contain at most `n` distinct holes. The fixed wrapper checks both rules after the editable body returns. In the evaluated benchmark, `learn_constants` is true and `max_arity` is `Some(2)`. Returned candidates must be valid inputs to the fixed `size`, `deduplicate`, `rewrites`, and `libs` methods. The entry point must terminate on cyclic e-graphs and handle operators with no children.

`egraph` is the saturated expression e-graph. `co_occurrences` is a symmetric query relation on its e-class IDs: `may_co_occur(a, b)` reports whether the two classes can occur in one extracted program; the relation is not generally reflexive or transitive.

The second removed entry point is:

```rust
pub(crate) fn cross_over(&self) -> Dfta<(Op, Op), (S, S)>
```

`Dfta<Op, S>` stores rules of the form `op(s1, ..., sn) -> s0`. This function must return a `Dfta<(Op, Op), (S, S)>` whose `by_operation` and `by_output` indexes describe exactly the same stored transitions. It must terminate when the source automaton is cyclic and handle source rules whose operation has no children. Its result is consumed by `LearnedLibrary::new`.

Each input file is one UTF-8 DreamCoder JSON object. Its top level contains a `DSL` object and a `frontiers` array. The loader reads these fields:

- `DSL.logVariable`: the grammar's floating-point log weight for variables.
- `DSL.productions`: grammar entries with a textual `expression` and floating-point `logProbability`.
- `frontiers`: the corpus's array of typed program frontiers.
- `frontiers[].request`: a recursive type object with string `constructor` and array `arguments`.
- `frontiers[].programs`: candidate programs for the same typed problem, each with a textual `program` and floating-point `logLikelihood`. An optional frontier `task`/`name` is accepted by the loader, but is absent from all 15 evaluated corpora.

The evaluated files also contain DreamCoder-run metadata named `verbose`, `arity`, `topK`, `pseudoCounts`, `aic`, `bs`, `topI`, `structurePenalty`, `CPUs`, `lc_score`, `iterations`, and `info`. The benchmark deserializes `DSL` and `frontiers`; Serde accepts and ignores those other top-level fields. At the top level, an evaluated record has this field layout (the bracketed labels stand for the records expanded immediately below):

```json
{
  "verbose": false,
  "arity": 3,
  "topK": 5,
  "pseudoCounts": 10.0,
  "aic": 1.0,
  "bs": 1000000,
  "topI": 300,
  "structurePenalty": 0.5,
  "CPUs": 8,
  "lc_score": 0.0,
  "DSL": {
    "logVariable": -0.503725,
    "productions": ["<production record>"]
  },
  "iterations": 20,
  "frontiers": ["<frontier record>"],
  "info": {
    "iteration": 1,
    "num_learned": 2,
    "new_grammar": {
      "logVariable": -0.466587,
      "productions": ["<production record>"]
    }
  }
}
```

The values shown are from development corpus 1; array lengths vary as listed under **Workload**. The following complete, unabridged frontier and production records from that corpus show the actual nested records represented by the labels.

The frontier record has a recursive `request` type and a `programs` array (line breaks are formatting only):

```json
{
  "request": {
    "constructor": "->",
    "arguments": [
      {
        "constructor": "list",
        "arguments": [
          {"constructor": "real", "arguments": []}
        ]
      },
      {
        "constructor": "->",
        "arguments": [
          {
            "constructor": "list",
            "arguments": [
              {"constructor": "real", "arguments": []}
            ]
          },
          {
            "constructor": "list",
            "arguments": [
              {"constructor": "real", "arguments": []}
            ]
          }
        ]
      }
    ]
  },
  "programs": [
    {
      "program": "(lambda (lambda (zip $0 $1 (lambda (lambda (+. $0 $1))))))",
      "logLikelihood": 0.0
    },
    {
      "program": "(lambda (lambda (zip $0 $1 (lambda (lambda (+. $1 $0))))))",
      "logLikelihood": 0.0
    },
    {
      "program": "(lambda (lambda (zip $1 $0 (lambda (lambda (+. $0 $1))))))",
      "logLikelihood": 0.0
    },
    {
      "program": "(lambda (lambda (zip $1 $0 (lambda (lambda (+. $1 $0))))))",
      "logLikelihood": 0.0
    },
    {
      "program": "(lambda (lambda (map (lambda $0) (zip $0 $1 (lambda (lambda (+. $0 $1)))))))",
      "logLikelihood": 0.0
    }
  ]
}
```

Here `->` constructs a function type, `list` constructs a list type, each leaf `real` has no type arguments, `$0` and `$1` are de Bruijn-indexed variables in the program text, and the five programs belong to the same frontier. The matching record shape in `DSL.productions` is:

```json
{"expression": "zip", "logProbability": -1.035973}
```

For each corpus and each of three repetitions the judge runs the release binary with this complete invocation and with `RAYON_NUM_THREADS=1`:

```text
<cargo-target>/release/benchmark <case-root> --domain=<list|physics|text|logo|towers> --beam-size=64 --lps=1 --mode=babble --rounds=2 --max-arity=2 --use-all=1 --lib-iter-limit=1 --output=<case-root>/result-<repeat>.csv
```

The fixed printer produces exactly one data row. This record was produced for
the bundled public smoke corpus by the empty-candidate implementation:

```csv
name,iter,initial cost,final cost,compression,total time,num libs
physics_eval,input.json,153,107,1.4299065420560748,0.005296935,0
```

Here `initial cost` is 153 AST nodes, `final cost` is 107 AST nodes including
the shared root, `compression` is exactly `153 / 107 = 1.4299065420560748`,
`total time` is 0.005296935 seconds for the complete
library-learning run, and `num libs` is zero because this smoke run selected no
library. For a physics file, the fixed printer writes `physics_eval` as `name`
and `input.json` as `iter`; other domains use the corresponding
`<domain>_eval` name. The initial cost is the sum,
over the 12 frontiers, of the smallest candidate-program AST in that frontier,
plus one shared root node; the final cost is the node count of the extracted
combined expression, including any library definitions and calls. The output
path is the value passed to `--output`; the evaluator uses
`<case-root>/result-<repeat>.csv`.

## Validity
The submitted text patch must apply cleanly, be nonempty, be no larger than 262,144 bytes, change no more than 20 files, and touch only the three listed paths. NUL bytes, Git binary patches, and binary-difference markers are invalid. Both marked regions must remain present exactly once, and all bytes before and after each marked body must match the supplied scaffold.

Added lines are checked case-insensitively and may not contain any of these substrings:

```text
std::env
std::fs
std::process
std::net
std::time
option_env!
env!
include!
include_bytes!
unsafe
libc
asm!
/proc
/sys
frontier_
h + arbor_ (concatenated, with no spaces)
command::new
process::exit
tcpstream
udpstream
```

These restrictions prohibit workload, environment, timing, filesystem, process, and network introspection from the editable code.

The build must retain `src/bin/benchmark/main.rs` and produce the release `benchmark` binary. For every corpus, the process must exit successfully and write exactly one CSV row. `initial cost` and `final cost` must be positive integers; `total time` must be finite and nonnegative; and `compression` must equal `initial cost / final cost` within relative and absolute tolerance `1e-12`. An invalid or empty patch, immutable-scaffold change, constructor-contract violation, policy violation, build failure, timeout, crash, missing or malformed CSV, invalid metric, or inconsistent compression scores zero before the optimization formula is applied.

## Scoring
The secondary continuous objective is to maximize the returned score on the 0--100 scale. For workload `w`, let

```text
r_w = initial_cost_w / final_cost_w
S_w = 100 * (1 - 1 / r_w)
U = (1 / |W|) * sum_{w in W} S_w
score = min(100.0, max(0.0, U))
```

`W` is the five-corpus development suite or the ten-corpus graded suite. Thus
`S_w` is the percentage reduction from initial to final AST cost. Runtime
cannot compensate for worse compression. Every workload must succeed; failed
workloads are not omitted from the mean.

## Calibration
A contract-valid empty-candidate smoke implementation builds, runs every public
smoke check, and gives a nonzero compression score. It is deliberately weaker
than the withheld source-locked implementation.

Compression is determined by integer AST costs and must repeat exactly across
the three runs. Runtime is wall-clock time, so the judge uses a three-run median
for each workload. Workload runtime comparisons are strict: equality does not
pass. Aggregate runtime is reported as diagnostic evidence only. It does not decide
PASS. The continuous compression score itself has no timing component or
tolerance.

## Feedback and grading
The judge returns a score after every submission. The submission budget is in `/frontier/task/README.md`. The development and graded suites are content-disjoint, five-domain stratified samples drawn from the same distribution. **The graded suite is not the suite you receive feedback on**, and neither its membership nor its source-run identity is disclosed.

Alongside the continuous score, evaluation applies a separate reference criterion. A compression comparison passes when the submission is at least the withheld value minus `1e-12`, and every workload's compression comparison must pass. A runtime comparison passes only when the submission's median runtime is strictly lower, and every workload's runtime comparison must pass; equality on any workload fails. PASS is the conjunction of both comparisons across every workload. An improvement on one workload cannot compensate for a tie or regression on another. Speed cannot compensate for worse compression. Aggregate score and aggregate runtime are diagnostics only. The withheld measurements and workload-specific comparison outcomes are not disclosed. The continuous score remains a secondary optimization objective and has no scoring tolerance.

Development feedback includes the submission's absolute `compression_ratio`
and median `runtime_seconds` for each corpus. For each metric in the stricter
criterion, it also gives aggregate pass and fail counts and percentages without
identifying the corresponding corpora. No measurements of the withheld
implementation or workload-specific comparison results are returned. The
development and graded suites are content-disjoint and share the same
five-domain coverage.

## Resources
Evaluation provides 2 vCPUs, 2,048 MiB of memory, and 4,096 MiB of storage. The scoring environment is CPU-only Rust 1.64. The solver workspace uses Rust 1.85 tooling but submitted source must remain Rust 1.64-compatible because the verifier performs the authoritative build. Preparation of an already-populated Cargo target allows 60 seconds for package cleanup; the release build has a 300-second timeout, each workload repetition has a 30-second timeout, and the complete evaluation has a 600-second timeout.
