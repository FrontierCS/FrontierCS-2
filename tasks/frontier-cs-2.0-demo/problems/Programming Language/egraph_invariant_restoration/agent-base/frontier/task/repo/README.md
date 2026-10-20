# E-Graph Invariant Restoration

## Problem

`egg` is an open-source Rust library for e-graphs and equality saturation, used in program optimization, synthesis, and verification. An e-graph compactly stores many symbolic expressions by grouping equivalent expressions into equivalence classes; equality saturation grows that structure with user-supplied rewrite rules, then an extractor selects a represented expression of minimum user-defined cost. The supplied project is a pinned `egg` source tree whose workload is drawn from its upstream arithmetic integration tests. Those tests provide prefix-form symbolic expressions, rewrite systems, and run limits; a test contains between one and seven starting expressions, and the largest explicit node limit is 100,000. The program produces ordinary Rust test-harness output for each selected test. Correctness requires the selected test to pass and the rebuilt e-graph to have canonical identifiers, congruence closure, consistent memoization, propagated analysis data, valid explanations, and correct minimum-cost extraction. The primary objective is PASS: satisfy every correctness gate and the paired performance criterion on every selected workload. The continuous runtime score is secondary. Development PASS on the three public workloads is provisional; final independently re-evaluates those workloads together with three hidden workloads, and all six must PASS. Exploration is allowed, but do not sacrifice a known PASS solely to increase the continuous score. The implementations of `EGraph::process_unions`, `EGraph::rebuild_classes`, and `EGraph::rebuild` have been removed and must be supplied.

## Your workspace

/frontier/task/repo contains the immutable Rust source tree; `/frontier/task/prepare_workspace.sh` creates its writable Git copy at `/workspace/repo`, which contains the Cargo manifests and lockfile, vendored dependencies, upstream tests, and build and submission helpers. The literal paths admitted by the evaluator configuration are:

```
src/egraph.rs
src/lib.rs
src/tutorials/_01_background.rs
src/tutorials/mod.rs
src/solver_extensions/*
```

`src/lib.rs` is admitted for patch validation but is source-locked and must remain byte-identical; the submission helper therefore rejects a change to it. The tutorial paths are documentation-only. Do not change tests, manifests, dependencies, scripts, generated files, or harness data. Patch size, changed-file count, path boundaries, the locked source file, and forbidden verifier-access capabilities are checked before any performance measurement.

## Design space

The public generic `EGraph<L, N>` API and the behavior of `Language`, `Analysis`, rewriting, `Runner`, explanations, and extraction remain part of the contract. The implementation must work for client-defined languages and analyses, with and without explanations enabled. The three removed methods retain their existing return values and `clean` state. Tests, evaluator data, dependencies, manifests, scripts, and public API definitions are outside the editable task surface.

## Workload

Development feedback measures three upstream `tests/math.rs` tests. `math_ematching_bench` starts from seven arithmetic, differentiation, integration, and simplification expressions; it runs six equality-saturation iterations and takes 20 timing samples per pattern because the evaluator sets `EGG_ITER_LIMIT=6` and `EGG_SAMPLES=20`. `diff_power_harder` differentiates a cubic expression with explanations enabled and has a 10-second runner limit, a 60-iteration limit, and a 100,000-node limit. `math_associate_adds` starts from a nested addition of the seven constants 1 through 7, uses seven iterations, and asserts that the completed graph has 127 e-classes.

Final evaluation freshly measures all three public workloads above together with three additional upstream tests from the same math-test family: one integration shape and two algebraic-simplification shapes whose exact identities and inputs are withheld. The six-workload final suite jointly covers e-matching, explanations, associativity, integration, radicals, factorization, unit analysis, and constant folding. These are representative library workloads, not records to recognize; correctness is required for general `egg` use. Before timing, the evaluator runs three independent invariant guards and upstream `simple_tests`. For each workload it runs one warmup and nine timed samples. A sample averages at least three complete test-process invocations and continues until at least 4.0 seconds of judge time have elapsed; the same sample also averages each invocation's congruence-maintenance time from its Runner report. Every process uses one Rust test thread and one Rayon thread; invocation order within paired measurements is deterministic but may put either executable first.

## Interface

The removed methods retain these signatures:

```rust
fn process_unions(&mut self) -> usize
fn rebuild_classes(&mut self) -> usize
pub fn rebuild(&mut self) -> usize
```

`process_unions` consumes pending union and analysis work and returns the number of successful congruence unions. `rebuild_classes` restores e-class records and returns the number of duplicate e-nodes found. `rebuild` consumes all pending work, restores the required invariants, sets `self.clean` to `true`, and returns the union count exposed by `Runner` as `n_rebuilds`.

After `rebuild` returns, every e-class ID and every e-node child ID must be canonical; congruent e-nodes must belong to the same class; memoized lookup and re-adding a node must resolve to that class; analysis changes must be propagated through parent nodes; explanations must remain valid when enabled; and extraction must return a represented expression with minimum cost under the supplied cost function. These requirements hold after arbitrary supported additions, unions, and analysis changes.

There is no input-file or standard-input protocol. Each test is compiled into the `math` test binary and supplies `RecExpr` records, rewrite records, and `Runner` limits through the Rust API. A `RecExpr` is a prefix s-expression whose recursive record is either a leaf atom or an operator followed by its ordered operand records. This workload record appears in `math_ematching_bench`:

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

For the concrete expression `(+ 0 (* 1 foo))`, the atom records are `0`, `1`, and `foo`; `*` and `+` are binary operator fields whose following records are their ordered operands. With rules `(+ 0 ?x) => ?x` and `(* 1 ?x) => ?x`, a minimum-`AstSize` extraction is the leaf record `foo`.

The evaluator builds the exact targets with:

```bash
cargo test --release --locked --offline --no-run --test simple --test math --test frontier_correctness
```

Every guard or workload process is then invoked in this form, with `<target>` equal to `simple`, `math`, or `frontier_correctness` and `<test-name>` naming exactly one test:

```bash
target/release/deps/<target>-<hash> --exact <test-name> --nocapture --test-threads=1
```

A successful invocation must report exactly one executed test. Its output has this record layout and concrete values:

```text
running 1 test
test math_associate_adds ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; <N> filtered out; finished in <S>s
```

Here `<N>` is the number of other compiled tests filtered out and `<S>` is the harness-reported duration. The evaluator measures process wall-clock time itself; it does not parse `<S>` as the scored timing.

Every selected workload process also prints exactly one report from the fixed upstream harness (`src/test.rs` prints `Runner::report()` on standard output for `test_fn!` tests and on standard error for `math_ematching_bench`). Its timing lines are measured by `Runner` in `src/run.rs`, outside the editable surface:

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

The evaluator defines one process's congruence-maintenance time as the `Apply` seconds plus the `Rebuild` seconds, that is, the time `Runner` spends applying rewrite matches (unions) and inside `EGraph::rebuild` over all iterations. A workload process whose combined output does not contain exactly one well-formed report fails like any other protocol failure. The `Rebuilds` count is the value returned by your own `rebuild` and is not scored.

## Validity

Validity is decided before performance. The patch must be nonempty, apply cleanly, contain no binary hunks or path escapes, change no more than 20 files, be at most 262,144 bytes, and stay within the configured path list. `src/lib.rs` must remain byte-identical to the locked supplied version. Added Rust source in the executable editable paths must not use unsafe code, process or filesystem access, environment access, compile-time file inclusion, evaluator-only paths, or process termination; failure of the evaluator's source checks scores zero.

Rust 1.94.1 must compile the locked dependency set in release mode without network access. Landlock ABI 3 or newer must be available for the evaluator sandbox. A build failure, missing or ambiguous test binary, failed guard, assertion failure, crash, failure to execute exactly the named test, malformed timing, per-process timeout, or overall evaluator deadline scores zero before timing comparisons are considered. The candidate must pass three invariant guards covering canonicalization and extraction, analysis propagation, and explanations, followed by upstream `simple_tests`. During measurement every repeated workload invocation must pass independently. These rules apply within the CPU, memory, storage, and timeout limits in `## Resources`.

## Scoring

Subject to the primary all-workload PASS objective, maximize the secondary score on a 0-to-100 scale. For each selected workload `w`, let `C_w > 0` be the median candidate wall-clock seconds from the nine timed samples, and define the equal-weight geometric mean candidate time:

```text
G_C = exp(mean_w(ln(C_w)))
```

The evaluator uses the fixed calibration constant `S = 0.020` seconds:

```text
score = 100 * S / (S + G_C)
```

Only candidate measurements enter this formula. Lower `G_C` always gives a strictly higher score; `G_C = S` maps to 50 points. Every finite valid runtime scores below 100, and the score approaches 100 only as runtime approaches zero, so there is no finite upper clipping or plateau. A build, validity, correctness, timeout, or protocol failure scores zero.

Development scoring uses the three public workloads. Final scoring freshly uses all six public-plus-hidden workloads. The fixed 20 ms scale was rounded from an exact Rust 1.94.1 six-workload reference calibration whose candidate-only geometric mean was `0.02008310432925681` seconds. Reference measurements never normalize the continuous score.

## Calibration

There is no valid naive runtime score for the untouched workspace: each removed method reaches `todo!` when the library attempts to rebuild, so correctness fails before timing. The fixed 20 ms scale comes from a valid reference implementation measured under the declared Rust 1.94.1 environment and complete six-workload final protocol. It is a frozen score anchor, not a per-submission comparator.

For interpretation, a geometric-mean candidate time of 20 ms scores 50. A 10 ms result scores about 66.67, a 40 ms result scores about 33.33, and every finite speedup strictly increases the score. Individual upstream workload processes observed during source calibration ranged from approximately 0.0053 to 0.389 seconds; the sample loop amortizes short processes by running at least three invocations and continuing for at least four seconds.

## Feedback and grading

The judge returns a score after every submission. At most 100 development submissions are allowed. Development feedback reports `candidate_dev_wall_clock_seconds_median`, the median of the three public per-workload candidate medians, and `candidate_dev_congruence_seconds_median`, the median of the three public per-workload candidate median congruence-maintenance times; lower is faster for both. It also reports only anonymous aggregate progress for each private comparison criterion (`wall_clock_seconds` and `congruence_seconds`) in `beat_spec_progress`. It does not disclose reference values, workload-keyed timing rows, hidden identities, or final measurements.

The primary PASS criterion is judged per workload on two metrics. Every selected workload must pass all correctness checks; its candidate median wall-clock time `C_w` must be at most `0.95 * R_w`, where `R_w` is the paired reference median measured under the same conditions; and its candidate median congruence-maintenance time `K_w` must be at most `0.95 * Q_w`, where `Q_w` is the paired reference median congruence-maintenance time from the same samples (paper Section 3.4 reports congruence-maintenance time separately from total equality-saturation time). A faster total time cannot compensate for slower congruence maintenance, or the reverse. This is a conjunction: all three public comparisons must pass during development, and all six freshly measured public-plus-hidden comparisons must pass during final evaluation. The criterion does not change the continuous score, and a valid submission that misses it is still scored by the candidate-only formula.

## Submission

From the prepared `/workspace/repo`, create `/workspace/payload/solution.patch` with the native wrapper:

```bash
/frontier/task/make_submission.sh /workspace/repo
```

The helper stages tracked and untracked changes, rejects any staged path outside its editable surface, and writes a binary-capable staged diff. It does not silently discard disallowed changes. The helper's effective editable surface excludes the source-locked `src/lib.rs`, while otherwise matching the executable paths listed in `## Your workspace`. Empty patches are invalid. Public local checks are:

```bash
cd /workspace/repo
bash ./frontier-build.sh
bash ./frontier-bench.sh
```

Do not include compiled objects, generated outputs, or timing files in the patch.

## Resources

The task provides 8 vCPUs, 4,096 MiB of memory, and 4,096 MiB of Agent workspace in a CPU-only x86_64 Linux environment. Rust is exactly `1.94.1-x86_64-unknown-linux-gnu`; Cargo uses the supplied lockfile and vendored dependencies with network access disabled and builds with up to eight jobs. Every measured candidate and reference test process uses `RUST_TEST_THREADS=1` and `RAYON_NUM_THREADS=1`.

The release-build timeout is 240 seconds and each test process has a 30-second timeout. The complete evaluator has a 1,700-second internal deadline. Formal Agent answering lasts 18,000 seconds; short acceptance trials use separately rendered 1,800-second answering budgets. Independent final has its own 1,800-second budget and does not consume Agent answering time. Train permits 100 accepted calls and 120,000 cumulative verifier seconds.
