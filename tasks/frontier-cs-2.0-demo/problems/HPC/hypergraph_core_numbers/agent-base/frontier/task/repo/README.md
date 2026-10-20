# Hypergraph Core Numbers

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

This program is a hypergraph k-core application in a retained Hygra/Ligra C++ substrate. A hypergraph generalizes an ordinary graph by allowing one hyperedge to contain any number of vertices; k-core numbers summarize how strongly each vertex remains connected through those hyperedges. The application reads unweighted, undirected incidence data produced by the substrate's `randHypergraph` utility, including retained performance graphs with hundreds of thousands of vertices and hyperedges. It returns one unsigned core number for every vertex. A correct result gives vertex `v` the largest `k` for which `v` belongs to a remaining vertex set in which every vertex has at least `k` incident hyperedges, with a hyperedge removed in full when any of its member vertices is removed. Isolated vertices therefore have core number zero, and duplicate hyperedges remain distinct incidences. The optimization objective is complete-process runtime: lower median elapsed time produces a higher score. The body of `KCore` in `apps/hyper/HyperKCore-Efficient.C` has been removed and must be supplied.

## Your workspace

The substrate at `/app` is the retained Hygra/Ligra C++ source tree: its hypergraph loader represents both vertex-to-hyperedge and hyperedge-to-vertex incidence directions, and its original driver loads a graph and calls the selected application. You may change only:

```text
apps/hyper/HyperKCore-Efficient.C
apps/hyper/frontier_solution/**
```

The first path contains the required `KCore` template and its unchanged `Compute` caller. New solver-authored headers or implementation fragments may be added only under the second path and included from the editable C++ file. The driver entry point in `ligra/ligra.h`, `Compute`, framework and loader headers, Makefile, license, graph inputs, and harness are outside the editable surface; the evaluator verifies the original `Compute` wiring. The evaluator force-includes the framework headers (`hygra.h`, `index_map.h`, `bucket.h`, `edgeMapReduce.h`, with `HYPER` defined) before the editable file, so solver code cannot change framework or loader code. The program also may not define any symbol exported by the libraries it links against (other than the replaceable global `operator new`/`operator delete`); such a build is rejected. Submitted binaries, logs, caches, inputs, and other build products are not part of the source patch.

## Design space

The editable contribution is the implementation of the declared `KCore` function and any source helpers placed under `apps/hyper/frontier_solution/`. The function may use the C++14 language, the types and facilities already present in the retained Hygra/Ligra headers, OpenMP on the eight declared execution threads, and solver-authored code within those two allowed paths. Its externally fixed contract is the function signature, the `GA` hypergraph supplied by the loader, the `num_buckets` value supplied by `Compute`, and an owning `array_imap<uintE>` result indexed by vertex ID. File organization inside the helper directory is unrestricted subject to the patch limits. The command-line interface, input representation, call count, observation boundary, build flags, and process-level measurement boundary are fixed parts of the task and are not editable.

## Workload

Every evaluation first runs three small correctness guards. They cover five vertices with one two-vertex hyperedge and three isolates; five vertices with two overlapping three-vertex hyperedges plus a third, duplicate-but-distinct copy; and six vertices with singleton edges and a cascade of overlapping edges. Guards are checked for correctness only and add no performance samples.

The active performance suite contains two retained, complete outputs of the original `randHypergraph` utility. The development input `feedback-a` has 220,000 vertices, 360,000 hyperedges, 3,599,999 incidences, and requested cardinality 10. `feedback-b` has 300,000 vertices, 480,000 hyperedges, 6,719,996 incidences, and requested cardinality 14. The held-out suite follows the same two-stratum distribution—hundreds of thousands of vertices and hyperedges with millions of incidences and cardinality in the low double digits—but its exact sizes and incidence counts are private. The utility deterministically generates membership positions and removes repeated membership of the same vertex within one hyperedge; separate hyperedges with identical member sets are still separate.

These fixed files are representative workloads, not inputs to recognize. Feedback and graded files have distinct contents, the two suites carry the same distribution, and execution order may vary. Inputs are unweighted and use symmetric incidence semantics. Weighted or compressed records, directed variants, clique-expanded graphs, and other Hygra applications are outside this task.

## Interface

Preserve this entry point exactly:

```cpp
template <class vertex>
array_imap<uintE> KCore(hypergraph<vertex>& GA, size_t num_buckets=128);
```

`GA.nv` is the number of vertices, `GA.nh` is the number of hyperedges, and `GA.mv` and `GA.mh` are the respective incidence-array lengths. `GA.V[v]` lists the hyperedges incident to vertex `v`, while `GA.H[e]` lists the vertices belonging to hyperedge `e`. The returned map must own exactly `GA.nv` `uintE` values; element `v` is vertex `v`'s core number. `num_buckets` is 128 in judged executions.

An input is a whitespace-separated token stream with this exact layout:

```text
AdjacencyHypergraph
nv
mv
nh
mh
nv vertex-incidence offsets
mv incident hyperedge IDs, grouped by vertex
nh hyperedge-incidence offsets
mh member vertex IDs, grouped by hyperedge
```

For either offset array, record `i` begins at offset `i` and ends at the next offset, or at the corresponding incidence count for the final record. IDs are zero-based. Valid retained files have `mv = mh`, because both arrays encode the same memberships in opposite directions.

Here is a complete input representing vertices `0,1,2,3`, hyperedges `{0,1}` and `{1,2}`, and isolated vertex `3`:

```text
AdjacencyHypergraph
4
4
2
4
0 1 3 4
0 0 1 1
0 2
0 1 1 2
```

Thus `nv=4`, `mv=4`, `nh=2`, and `mh=4`; the vertex offsets select incident-edge lists `[0]`, `[0,1]`, `[1]`, and `[]`, while the hyperedge offsets select member lists `[0,1]` and `[1,2]`. The matching return value is:

```text
[1, 1, 1, 0]
```

The judge sets `OMP_DYNAMIC=FALSE` and invokes the executable with every argument shown here (`<input.adj>` is the final positional filename):

```text
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE ./HyperKCore-Efficient -s -rounds 0 -nb 128 <input.adj>
```

For the separate sequential comparison described under Feedback and grading, the judge also runs the same executable and arguments with `OMP_NUM_THREADS=1`.

`-s` selects the symmetric loader, `-rounds 0` leaves only the driver's initial call to `Compute`, and `-nb 128` supplies the bucket-count argument. `Compute` calls `KCore` once. The required program output is the owning map returned by `KCore`; the application does not normally print it. For verification, the evaluator compiles a private observation copy that writes one ASCII `FRONTIER_CORES` record to standard error, followed by the `GA.nv` returned values in vertex-ID order. Standard output is discarded. That complete record is required once per process. The evaluator times the process from launch through graph loading, computation, observation serialization, and exit; compilation is excluded.

## Validity

The evaluator first rejects a missing, empty, oversized, binary, inapplicable, or out-of-scope patch. A patch may contain at most 262,144 bytes and change at most 20 files, all within the two listed paths. It then requires the original `Compute` body and its call boundary to remain unchanged and compiles the submitted source with GNU C++14, `-O3`, OpenMP, and the retained include paths. The platform allows 180 seconds for the build stage; within evaluation, this compiler invocation has a 90-second timeout.

Each resulting process must exit successfully within 40 seconds, emit exactly one bounded ASCII core record through the private adapter, and return exactly one oracle-validated number per vertex. The same requirements apply on every guard and every measured eight-thread performance execution; one-thread executions affect only the separate sequential comparison described below. Patch-policy violations, build failure, timeout, crash, malformed or excessive output, a missing result allocation for a nonempty graph, or any wrong core number score zero before performance is considered. Printed application timing or score claims do not affect evaluation; elapsed time is measured by the evaluator's parent process. The evaluation environment provides 8 vCPUs, 4096 MiB memory, and 2048 MiB storage, with a 300-second overall evaluation timeout.

## Scoring

The secondary continuous objective is to maximize the score by minimizing verifier-measured complete-process runtime. For each performance workload `w`, let `t_{w,1},...,t_{w,7}` be the seven elapsed times in seconds and define

```text
m_w = median(t_{w,1}, t_{w,2}, ..., t_{w,7})
t   = median({m_w : w is in the active two-workload suite})
score = 100 * exp(-t / 0.25)
```

The `0.25`-second scale is a challenge constant (not from materials). A valid submission receives this continuous score in `(0,100]`; an invalid or incorrect submission receives `0`. Candidate and withheld-method executions alternate order across the seven fresh-process repetitions for each workload. Guards do not enter `t`, and failing the separate per-workload criterion described below does not replace or zero the continuous score.

## Calibration

The shipped `KCore` body returns an empty map and is invalid on a nonempty judged graph. No valid non-reference baseline has been measured, so no such baseline score is reported. The original author timing calibration used two OpenMP threads and seven interleaved fresh-process samples per implementation and workload. Repeated evaluation of identical valid code showed a worst absolute per-workload median ratio difference of 1.636%; the separate criterion therefore requires a 3.0% difference to count as an improvement. That threshold is preserved. A later staging scaling diagnostic established that the retained reference benefits materially from eight workers, so the production evaluator now runs both candidate and reference with eight OpenMP threads under the same declared resources. The original two-thread full development evaluation took 42.08 seconds and peaked at 731,744 KiB resident memory. Scores and runtimes are recomputed from the submitted executable on every evaluation.

## Feedback and grading

The judge returns a score after every submission. At most 100 development submissions are allowed. The independently re-evaluated final suite contains both feedback workloads plus the separate held-out graded workloads. The held-out graded subset is not the suite you receive feedback on, and neither its membership nor its exact generator parameters are disclosed. The two suites are stratified to carry the same distribution and share no item.

In addition to the continuous score, a separate stricter criterion is judged per workload and per metric, not in aggregate. A submission clears it only if all core numbers are correct and, on every final performance workload, covering the two public feedback inputs and both held-out graded inputs, both of the following hold. `runtime_seconds`: its seven-run median elapsed time with eight threads is at least 3.0% lower than the withheld method's median. `sequential_runtime_seconds`: its seven-run median elapsed time with one thread (`OMP_NUM_THREADS=1`, the paper's sequential time T1, which measures work efficiency) is at least 3.0% lower than the withheld method's one-thread median. One-thread executions are timed and oracle-checked exactly like the eight-thread ones, in a separate pass after them; a one-thread execution that fails, exceeds 40 seconds or returns any incorrect core number fails `sequential_runtime_seconds` for that workload. The score remains a secondary optimization objective and uses only the eight-thread times: a correct submission that does not clear this criterion is still scored normally.

Development feedback exposes the candidate-only median `t` as `candidate_runtime_seconds_median` and the candidate-only median of the one-thread workload medians as `candidate_sequential_runtime_seconds_median` (null if any one-thread execution failed). It also reports aggregate pass and fail counts and percentages for each of the two stricter runtime comparisons across feedback workloads, without identifying individual workloads or disclosing withheld per-workload times. No other per-workload evaluator measurements are solver-visible.

## Submission

From `/app`, run:

```bash
./make_submission.sh
```

The script stages tracked and untracked changes, then writes the Git diff for the two editable paths to `/app/solution.patch`. Submit that patch:

```text
/app/solution.patch
```

Changes outside the editable surface are rejected rather than silently omitted by evaluation. The script rejects an empty patch, and empty submissions are invalid. Source-only patches must also satisfy the 262,144-byte and 20-file limits stated above.

## Resources

Each evaluation runs in Ubuntu 24.04 with GNU C++ 13, GNU C++14 language mode, OpenMP, and eight execution threads. It provides 8 vCPUs, 4096 MiB memory, and 2048 MiB storage. The platform build stage has a 180-second limit, the evaluator's compiler invocation has a 90-second timeout, each judged executable process has a 40-second limit, and the complete evaluation has a 300-second timeout. Available system packages include Bash, `build-essential`, Git, Python 3, and the standard timing utility. The retained substrate contains the Hygra/Ligra framework and loader headers needed by the application; no weighted, compressed, or alternate application inputs are used.
