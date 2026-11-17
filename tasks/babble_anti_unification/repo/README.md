# Reusable Library Candidates

Make BABBLE's library learning shrink program corpora further and run faster by supplying the
two removed bodies: candidate generation in `LearnedLibrary::new` and paired-state construction
in `Dfta::cross_over`. The score rewards only AST-cost reduction; a run is valid when every
corpus builds, runs and reports consistent costs.

## Task

BABBLE (POPL 2023, "babble: Learning Better Abstractions with E-Graphs and Anti-unification") is a
Rust library-learning system. It refactors a corpus of related programs by adding reusable
function definitions and replacing program fragments with calls to them. A refactored corpus is
measured by counting AST nodes in its programs, library definitions and calls. BABBLE stores
expressions that are equal under the supplied equations in an e-graph, where one e-class holds
several forms of one expression. The inputs are DreamCoder search-frontier exports: each frontier
holds alternative programs for one typed synthesis problem, and all frontiers of a corpus are
processed together. This tree keeps BABBLE's program-compression benchmark. Every library
candidate must preserve the meaning of the fragment it replaces.

## What you may change

- `src/learn.rs`: only between `// FRONTIER_EDIT_BEGIN: candidate generation` and
  `// FRONTIER_EDIT_END: candidate generation`.
- `src/dfta.rs`: only between `// FRONTIER_EDIT_BEGIN: paired-state construction` and
  `// FRONTIER_EDIT_END: paired-state construction`.
- `src/learn_extensions/**`: new supporting Rust files. They do not enlarge the two regions or
  make any other path editable.

Each marker must appear exactly once, and every byte before and after each marked body must match
the shipped file. At most 20 files may change. Everything else is fixed: public types and
signatures, the constructor contract checks, the fields and indexes of `LearnedLibrary` and
`Dfta`, parsing, equation saturation, co-occurrence analysis, deduplication, library selection,
extraction, cost measurement, timing and CSV output.

Added lines must not contain any of these substrings, compared case-insensitively:

```text
std::time
option_env!
env!
unsafe
libc
asm!
frontier_
h + arbor_ (concatenated, with no spaces)
process::exit
tcpstream
udpstream
```

## Interface

```rust
pub fn new<A: Analysis<AstNode<Op>>>(
    egraph: &EGraph<AstNode<Op>, A>,
    learn_constants: bool,
    max_arity: Option<usize>,
    co_occurrences: CoOccurrences,
) -> LearnedLibrary<Op, (Id, Id)>
```

Candidates are `PartialExpr<Op, Var>` values; a `Hole(var)` is an argument, and repeated
occurrences of one `Var` are the same argument. If `learn_constants` is false, every candidate needs at least one hole. If `max_arity`
is `Some(n)`, a candidate has at most `n` distinct holes. The fixed wrapper checks both rules after
the body returns; in the benchmark `learn_constants` is true and `max_arity` is `Some(2)`.
Candidates must be valid inputs to the fixed `size`, `deduplicate`, `rewrites` and `libs` methods.
`co_occurrences.may_co_occur(a, b)` (symmetric, not generally reflexive or transitive) says
whether two e-classes can occur in one extracted program.

```rust
pub(crate) fn cross_over(&self) -> Dfta<(Op, Op), (S, S)>
```

`Dfta<Op, S>` stores rules `op(s1, ..., sn) -> s0`. The result's `by_operation` and `by_output`
indexes must describe exactly the same transitions. `LearnedLibrary::new` consumes the result.

Both functions must terminate on cyclic inputs and handle operators with no children.

**Input.** Each corpus is one UTF-8 DreamCoder JSON object. The loader reads:

`DSL.logVariable` (float), `DSL.productions` (records below), and `frontiers`: each with a
recursive type `request` (`constructor`, `arguments`) and `programs` (`program` text,
`logLikelihood` float). Top-level layout (values from train corpus 1):

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

A complete frontier record from that corpus:

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

A production record is:

```json
{"expression": "zip", "logProbability": -1.035973}
```

**Command.** For each corpus and each of three repetitions the judge runs, with
`RAYON_NUM_THREADS=1`:

```text
<cargo-target>/release/benchmark <case-root> --domain=<list|physics|text|logo|towers> --beam-size=64 --lps=1 --mode=babble --rounds=2 --max-arity=2 --use-all=1 --lib-iter-limit=1 --output=<case-root>/result-<repeat>.csv
```

The corpus is at `<case-root>/<domain>_eval/input.json`. Output (one data row; this one is an
empty-candidate run on the smoke corpus):

```csv
name,iter,initial cost,final cost,compression,total time,num libs
physics_eval,input.json,153,107,1.4299065420560748,0.005296935,0
```

`initial cost` is the sum over frontiers of the smallest program AST in each, plus one shared root
node. `compression` is `initial cost / final cost`.

## Correctness

Any of these on any corpus makes the whole evaluation score 0:
- a constructor-contract violation, panic or crash, or a nonzero exit;
- the release `benchmark` binary does not build;
- a timeout: release build 300 s, each benchmark repetition 30 s;
- the CSV is missing or does not hold exactly one data row;
- `initial cost` or `final cost` is not a positive integer, `num libs` is not a nonnegative
  integer, or `total time` is negative or non-finite;
- `compression` differs from `initial cost / final cost` beyond relative and absolute tolerance
  `1e-12`;
- the three repetitions disagree on `initial cost`, `final cost` or `num libs`;
- `initial cost` differs from the reference's for that corpus (it depends only on the input).

## Workload

One workload is one corpus, run under the fixed command above. Settings: the domain's equation
set where one exists (list and physics; the other domains have none), beam size 64, one library
per step, BABBLE mode, two rounds, maximum arity 2, all programs of every frontier, library-rewrite
iteration limit 1, and one Rayon worker.

| suite/domain | corpora | JSON bytes | frontiers | total programs | DSL productions |
| --- | ---: | ---: | ---: | ---: | ---: |
| train/list | 1 | 14,560 | 8 | 14 | 22 |
| train/physics | 1 | 17,463 | 12 | 60 | 15 |
| train/text | 1 | 16,059 | 7 | 14 | 24 |
| train/logo | 1 | 34,634 | 28 | 56 | 27 |
| train/towers | 1 | 16,695 | 15 | 24 | 14 |
| final/list | 2 | 16,307--21,531 | 9--14 | 17--26 | 22--23 |
| final/physics | 2 | 17,717--22,237 | 12 | 59--60 | 15--23 |
| final/text | 2 | 13,790--28,388 | 5--14 | 10--28 | 24--26 |
| final/logo | 2 | 28,826--63,154 | 23--48 | 44--95 | 27--33 |
| final/towers | 2 | 7,735--45,828 | 4--36 | 8--72 | 14--16 |

- Train has 5 corpora (one per domain), final has 10 (two per domain). The suites are
  content-disjoint samples from the same distribution; all 15 files differ. Neither suite's files
  are in the repo, and final membership is withheld.
- List, text, logo and towers corpora are unchanged DreamCoder files. Physics corpora are a fixed
  12-frontier prefix of a DreamCoder file, with records unchanged.
- Corpora run one after another, each three times. Runtime is the median of the three
  `total time` values.
- Only the library-learning run inside the binary is timed (`total time`); the build is not.

## Scoring

For workload `w`:

```text
r_w   = initial_cost_w / final_cost_w
S_w   = 100 * (1 - 1 / r_w)
U     = (1 / |W|) * sum_{w in W} S_w
score = min(100.0, max(0.0, U))
```

`W` is the 5 train or the 10 final corpora. `S_w` is the percentage reduction in AST cost; it is
negative when the final cost exceeds the initial one. Runtime does not enter the score. Anchors:
no compression (`r_w = 1`) gives 0; `r_w = 2` on every corpus gives 50; the maximum is 100. The
untouched tree panics at the removed bodies and scores 0.

**Beats the reference** when, on every one of the 10 final workloads, both hold against the
withheld reference implementation:
- `compression_ratio`: at least the reference's minus `1e-12`;
- `runtime_seconds`: median runtime strictly lower (equality fails).

A win on one corpus cannot offset a tie or loss on another, and speed cannot offset worse
compression. Aggregate runtime is a diagnostic only. This criterion does not change the score.

## Running locally

In `/workspace/repo`:
- `bash frontier-build.sh` builds the release `benchmark` as the judge does.
- `bash frontier-bench.sh [out.csv]` runs the judge's command on the public physics smoke corpus
  under `harness/data/dreamcoder-benchmarks/benches` and writes the CSV (default
  `sample-results.csv`).

## Environment

The judge is CPU-only with 2 CPUs and 2 GiB of memory (the Agent machine has 8 GiB), and builds
with Rust 1.64.0 (`cargo build --release --locked --offline`).
The Agent image has the same toolchain (`RUSTUP_TOOLCHAIN=1.64.0`); code must compile with Rust
1.64.

## Notes

The corpora are representative program-synthesis inputs, not identifiers or strings to recognize:
solutions must not special-case corpora, domains or the train/final phase. The forbidden
substrings rule out reading the time or the environment from your code.
Files below `src/learn_extensions/` are not compiled automatically: no fixed file declares that module, so include them from a marked body (for example with a `#[path = "learn_extensions/x.rs"] mod x;` item).
