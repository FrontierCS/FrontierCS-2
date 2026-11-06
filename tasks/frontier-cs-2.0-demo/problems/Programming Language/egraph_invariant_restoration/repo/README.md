# E-Graph Invariant Restoration

Restore the removed rebuilding pipeline of the `egg` e-graph library and make equality
saturation fast: lower candidate wall-clock time gives a higher score. A submission is valid
when it builds, passes the correctness guards and runs every workload correctly.

## Task

`egg` (POPL 2021, "egg: Fast and Extensible Equality Saturation") is a Rust library for e-graphs
and equality saturation. An e-graph groups equivalent expressions into e-classes; rewrite rules
grow it, and an extractor picks a represented expression of minimum cost. In this pinned source
tree the bodies of `EGraph::process_unions`, `EGraph::rebuild_classes` and `EGraph::rebuild` in
`src/egraph.rs` are `todo!`. Supply them. The implementation must work for any client-defined
`Language` and `Analysis`, with and without explanations enabled.

## What you may change

- `src/egraph.rs`
- `src/tutorials/_01_background.rs`, `src/tutorials/mod.rs`
- new or changed files under `src/solver_extensions/` (compiled only if `src/egraph.rs` includes
  them, e.g. with a `#[path = "solver_extensions/x.rs"] mod x;` item: `src/lib.rs` declares no
  module for them)

`src/lib.rs`, tests, manifests, the lockfile, vendored dependencies, scripts and the public
`EGraph<L, N>` API are fixed. At most 20 files may change.

Lines you add to `src/egraph.rs` or the tutorial files, and every `*.rs` file under
`src/solver_extensions/` in full, must not contain (case-insensitive, comments included, also
inside longer names; `::` may have spaces around it) `unsafe` (unless a letter, digit or `_`
follows it), `std::fs`, `std::process`, `Command::` or `process::exit`. A match is rejected as
`content_policy`.

## Interface

The removed methods keep these signatures:

```rust
fn process_unions(&mut self) -> usize
fn rebuild_classes(&mut self) -> usize
pub fn rebuild(&mut self) -> usize
```

`process_unions` consumes pending union and analysis work and returns the number of successful
congruence unions. `rebuild_classes` restores e-class records and returns the number of duplicate
e-nodes found. `rebuild` consumes all pending work, restores the invariants, sets `self.clean =
true`, and returns the union count that `Runner` reports as `n_rebuilds` (not scored).

After `rebuild` returns, for any supported sequence of additions, unions and analysis changes:
every e-class ID and e-node child ID is canonical; congruent e-nodes share a class; memoized lookup
and re-adding a node resolve to that class; analysis changes are propagated through parents;
explanations stay valid when enabled; extraction returns a represented minimum-cost expression.

There is no input file. Each test in `tests/math.rs` builds `RecExpr` prefix s-expressions, rules
and `Runner` limits through the Rust API, for example from `math_ematching_bench`:

```text
(d x (- (pow x 3) (* 7 (pow x 2))))
 ^ ^  ^  ^   ^ ^   ^ ^ ^  ^   ^ ^
 | |  |  |   | |   | | |  |   | exponent atom
 | |  |  |   | |   | | |  base atom
 | |  |  |   | |   | | multiplication operator
 | |  |  |   | |   | coefficient atom
 | |  |  |   | |   multiplication operator
 | |  |  |   | exponent atom
 | |  |  |   base atom
 | |  |  power operator
 | |  subtraction operator
 | differentiation variable atom
 differentiation operator
```

Worked example: from `(+ 0 (* 1 foo))` with rules `(+ 0 ?x) => ?x` and `(* 1 ?x) => ?x`, a
minimum-`AstSize` extraction is `foo`.

The judge builds:

```bash
cargo test --release --locked --offline --no-run --test simple --test math --test frontier_correctness
```

and runs every guard or workload process as (`<target>` is `simple`, `math` or
`frontier_correctness`):

```bash
target/release/deps/<target>-<hash> --exact <test-name> --nocapture --test-threads=1
```

Each process must exit 0 and print exactly one executed test:

```text
running 1 test
test math_associate_adds ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; <N> filtered out; finished in <S>s
```

Each workload process must also print, on stdout and stderr combined, exactly one report from the
fixed harness (`src/test.rs`, `src/run.rs`):

```text
Runner report
=============
  Stop reason: <reason>
  Iterations: <iterations>
  Egraph size: <nodes> nodes, <classes> classes, <memo> memo
  Rebuilds: <unions returned by rebuild>
  Total time: <seconds>
    Search:  (<fraction>) <seconds>
    Apply:   (<fraction>) <seconds>
    Rebuild: (<fraction>) <seconds>
```

A process's congruence-maintenance time is its `Apply` + `Rebuild` seconds. Do not print the text
`Runner report` yourself.

## Correctness

The evaluation scores 0 if any of these happens:
- the release build fails or exceeds 240 s, or a test binary cannot be identified uniquely;
- one of the three guards (`frontier_invariants_and_extraction`, `frontier_analysis_propagation`,
  `frontier_explanations`: canonicalization and extraction, analysis propagation, explanations) or
  upstream `simple_tests` fails;
- any guard or workload invocation, including every repeated timed one, exits non-zero, panics,
  fails an assertion, or does not print exactly one passing named test;
- a workload process prints no, two or a malformed `Runner report`, or a negative or non-finite
  Search/Apply/Rebuild time, or a workload's median congruence time is not positive;
- a process exceeds 30 s, or the whole evaluation exceeds its 1,700 s internal deadline.

The untouched tree scores 0: the `todo!` bodies panic.

## Workload

One workload is one upstream `tests/math.rs` test, run as one process with `RUST_TEST_THREADS=1`
and `RAYON_NUM_THREADS=1`.

| suite | workloads |
|---|---|
| train (3, public) | `math_ematching_bench` (`EGG_ITER_LIMIT=6`, `EGG_SAMPLES=20`), `diff_power_harder`, `math_associate_adds` |
| final (6) | the three train workloads, measured afresh, plus three withheld math tests: one integration and two algebraic simplifications (radicals, factorization) |

`math_ematching_bench` runs seven expressions for six iterations with 20 timing samples per
pattern; `diff_power_harder` uses explanations with a 10 s, 60-iteration, 100,000-node limit;
`math_associate_adds` adds the constants 1 to 7 in seven iterations and asserts 127 e-classes.
These are representative library workloads, not records to recognize: do not special-case tests,
expressions or the train/final phase.

Per workload: 1 warm-up, then 9 timed samples, each paired with a reference sample in a
deterministic order that may put either first. A sample repeats complete process invocations
until at least 3 have run and at least 4.0 s have elapsed; its time is the mean judge wall clock
per invocation, and its congruence time is the mean Apply + Rebuild per invocation. Compilation and
the guards are not timed.

## Scoring

With `C_w` the median over the 9 samples of the candidate's wall-clock seconds on workload `w`:

```text
G_C   = exp(mean_w(ln(C_w)))        # geometric mean over the suite's workloads
S     = 0.020                       # seconds, fixed
score = 100 * S / (S + G_C)
```

Only candidate times enter the score; it is not clipped. Anchors: `G_C = 20 ms` scores 50,
10 ms about 66.67, 40 ms about 33.33; 100 is approached only as `G_C → 0`.

**Beats the reference** when every workload of the suite (3 in train, 6 in final) passes
Correctness and both of:
- `wall_clock_seconds`: `C_w <= 0.95 * R_w`, with `R_w` the paired reference median wall clock;
- `congruence_seconds`: `K_w <= 0.95 * Q_w`, with `K_w`, `Q_w` the candidate and reference medians of per-sample
  congruence time.

Equality at 0.95 passes. This criterion does not change the score.

## Running locally

From `/workspace/repo`:

```bash
bash ./frontier-build.sh    # release build of the simple tests
bash ./frontier-bench.sh    # runs tests/simple.rs
```

Time `math` workloads with the judge commands above; the guard tests are added by the judge. Create the patch with
`/frontier/task/make_submission.sh /workspace/repo`; see `/frontier/task/README.md`.

## Environment

CPU-only x86_64 Linux, 8 vCPUs, 4 GiB memory. Rust is exactly
`1.94.1-x86_64-unknown-linux-gnu`, vendored/offline Cargo with the supplied lockfile and
`CARGO_BUILD_JOBS=8`.
