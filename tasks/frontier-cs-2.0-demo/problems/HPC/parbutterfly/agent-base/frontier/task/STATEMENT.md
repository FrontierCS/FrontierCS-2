# Exact Butterfly Counting

## Problem

ParButterfly is a multicore C++/OpenMP system for computations on bipartite graphs, including global, per-vertex, and per-edge butterfly counting and graph decompositions. A butterfly is a complete two-by-two biclique: two vertices in each bipartition joined by all four possible cross-partition edges. This task uses the project's exact global-count path. It reads deterministic, generated, unweighted simple bipartite graphs stored as two reciprocal compressed-sparse-row incidence lists; the eight defined graphs have 18,000–23,000 vertices in the first partition, 20,000–25,000 in the second, and 1,584,000–2,484,000 undirected edges; two of them have heavily skewed (power-law) degrees on both sides. It produces one non-negative integer giving the total number of distinct butterflies, and that integer must equal an independently computed serial count on every graph and every repetition. Your primary objective is to pass the all-workload criterion: preserve exact counts and beat the fixed comparison by more than 5% on every graph. The continuous score is secondary and rewards speed relative to the independent serial baseline; do not sacrifice a known all-workload pass merely to raise it. The implementation of `CountTotal` has been removed and must be supplied.

## Your workspace

After running `/frontier/task/prepare_workspace.sh`, `/workspace/repo` contains the writable ParButterfly C++14/OpenMP repository substrate. You may change only:

```text
apps/hyper/butterfly_count_total.h
apps/hyper/butterfly_count.h
apps/hyper/butterfly_ecount.h
apps/hyper/butterfly_peel.h
apps/hyper/butterfly_epeel.h
apps/hyper/butterfly_putils.h
apps/hyper/butterfly_utils.h
apps/hyper/solver_extensions/**
```

The application entrypoint, graph loader, graph-ranking code, build wiring, and harness adapters must remain unchanged and are verified. The submission limit is 524288 bytes across at most 20 changed files. New files may exist only under the listed `solver_extensions` prefix; generated binaries, object files, graph data, and build-system changes are outside the editable surface.

## Design space

The permitted design space is the exact implementation of the existing `CountTotal` interface, the seven editable headers, and supporting C++14 code under `apps/hyper/solver_extensions/`. The alternative vertex and edge counting entrypoints retain callable stubs so the original application still builds; they provide no counting implementation. The fixed interface supplies the bipartite CSR graph, the selected side and wedge count, two capacity limits, the requested count type, and the vertex-ranking mode. The implementation must support the fixed `BATCHS`, `ACOCORE`, and `TOTAL` evaluation path on every valid graph accepted by the loader, using at most the declared process resources. The `ACOCORE` argument is part of the fixed command line, not a required algorithm: only the exact total is checked, so an implementation is free to choose its own vertex ordering and wedge aggregation. The comparison does the same. For this path it applies the source paper's runtime ordering rule (Shi and Shun, Section 6.2.2): it counts the wedges `w_r` that degree ordering would process, compares them with the supplied side-order count `w_s`, and uses side ordering when `(w_s - w_r) / w_s < 0.1` and degree ordering otherwise, followed by the authors' simple-batching aggregation. It must not depend on graph filenames, suite membership, evaluation order, undisclosed seeds, or state retained between evaluator calls.

## Workload

The evaluator defines eight distinct deterministic generated simple bipartite graphs, split into four development workloads and four disjoint graded workloads. Each suite has three uniform-degree graphs and one skewed-degree graph. The uniform graphs' `(first-partition vertices, second-partition vertices, degree, undirected edges)` shapes are:

```text
(18000, 20000,  88, 1584000)
(19000, 21000,  92, 1748000)
(20000, 22000,  96, 1920000)
(21000, 23000, 100, 2100000)
(22000, 24000, 104, 2288000)
(23000, 25000, 108, 2484000)
```

For each first-partition vertex of a uniform graph, a seeded linear-congruential generator selects `degree` distinct second-partition neighbors; neighbor identifiers are sorted, and the reverse incidence lists are then derived from those edges.

The skewed graphs follow the heavy-tailed degree distributions of the real-world bipartite graphs in the source paper's evaluation. Their `(first-partition vertices, second-partition vertices, undirected edges)` shapes are `(20000, 22000, 1900000)` and `(21000, 23000, 2000000)`. Each edge endpoint is drawn independently with probability proportional to `(rank + 1)^-a` on its side, with exponents `a` between 0.75 and 0.85, until the stated number of distinct edges exists; vertex identifiers are then randomly permuted. The highest-degree vertices on both sides have roughly 14,000–19,000 neighbors, so the number of wedges depends strongly on the processing order. Seeds and the allocation of shapes to the two suites are not disclosed. The two suites have the same stratified distribution over vertex count, degree, and aspect ratio and share no graph. These are representative workloads rather than inputs to recognize; execution order and repetition conditions may vary within this regime.

Each graph is first checked with the fixed serial count path. The fixed BATCHS author-parallel path is then run once for stable candidate, source-locked parallel comparison, and frozen serial baseline counts. Timing uses 31 candidate/comparison pairs plus 31 separate candidate/baseline pairs, so the candidate has 62 timed invocations per graph. Pair order alternates independently in both series by repetition. The evaluator invokes every run with four worker threads, `-m 200000000`, and `-a 200000000`.

## Interface

Keep this callable contract:

```cpp
long CountTotal(bipartiteCSR& GA, bool use_v, long num_wedges,
                long max_wedges, long max_array_size,
                CountType type, RankType tw);
```

`GA` is the loaded bipartite CSR graph. `use_v` identifies the partition selected by the fixed preprocessing step, `num_wedges` is that partition's wedge count, `max_wedges` and `max_array_size` are the two command-line capacities, and `type` and `tw` carry the count and ranking modes. The return value is the exact total number of distinct butterflies as a `long`; it must be non-negative and deterministic for repeated calls on the same input.

The evaluator runs this complete command, with the final argument replaced by the generated graph's path:

```bash
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ./apps/hyper/BipartiteButterflies -countType BATCHS -rankType ACOCORE -per TOTAL -r 0 -m 200000000 -a 200000000 graph.adj
```

The input is whitespace-delimited ASCII in `AdjacencyHypergraph` format. The fields occur in this exact order:

```text
AdjacencyHypergraph
n_first
m_first
n_second
m_second
first_offsets[0] ... first_offsets[n_first-1]
first_neighbors[0] ... first_neighbors[m_first-1]
second_offsets[0] ... second_offsets[n_second-1]
second_neighbors[0] ... second_neighbors[m_second-1]
```

Offsets and vertex identifiers are zero-based decimal integers. Each offset starts that vertex's contiguous neighbor block; the next offset, or the incidence-array length for the final vertex, ends it. The two incidence arrays must describe the same edges in opposite directions, so the generated inputs have `m_first == m_second`.

Here is the complete file for `K2,2`, with two vertices per side and all four cross edges:

```text
AdjacencyHypergraph
2
4
2
4
0
2
0
1
0
1
0
2
0
1
0
1
```

The two first-side offsets are `0,2`, followed by first-side neighbor identifiers `0,1,0,1`; the two reverse offsets and four reverse neighbors have the same values. The fixed application prints:

```text
number of butterflies: 1
```

because this graph contains one distinct butterfly. During judging, the fixed parent also prints exactly one trusted timing record of the form `FS2_TRUSTED_SECONDS=<positive finite seconds>`. The evaluator requires exactly one count record and one timing record; submitted code supplies the return value and must not create additional matching records.

## Validity

Patch validation happens before build or performance measurement. A submission must be a non-empty text patch of at most 524,288 bytes, change at most 20 files, contain no binary patch data or symlinks, apply cleanly, and touch only the listed path patterns. Files outside the editable surface are compared with the pristine substrate.

A rejected patch, build failure, missing or failed authentic entrypoint, 45-second counting-process timeout, crash, malformed output, negative count, non-finite or non-positive timing, nondeterministic repeated count, isolation failure, or evaluation failure scores zero. A graph is exact only when the candidate, baseline, and comparison counts equal the independent serial count. Inexact graphs score zero and cannot pass the separate per-workload criterion. The worker must obey the 8,192 MiB address-space limit, four-CPU affinity, fixed thread count, and fixed capacities. Network access, new processes, access to private evaluator files, and persistent state are unavailable inside the isolated worker.

## Pass criterion and scoring

First, satisfy the separate stricter criterion on every graph in the active suite. On each graph, both candidate and comparison must return the exact serial total, and the median of the 31 adjacent paired comparison-time / candidate-time ratios must be strictly greater than 1.05 (a speedup exceeding 5%; equality fails). This is an all-workload, all-metric conjunction. A development pass is provisional because development covers only the four public workloads; final independently evaluates those workloads together with four hidden same-distribution workloads.

Subject to that primary objective, improve the returned continuous score on a 0–100 scale. For every graph `i` in the active suite, let

```text
q_i = 1  if candidate_count_i, baseline_count_i, and comparison_count_i
          all equal serial_count_i,
      0  otherwise

s_i = median(baseline_time_ij / candidate_time_ij over the 31 paired runs)
S = geometric mean of s_i over every active uniform-degree workload
score = 0                                      if any q_i == 0 or S <= 1
score = 100 * ln(S) / (ln(10) + ln(S))         otherwise
```

The continuous score's speedup term uses only the uniform-degree graphs, as before the skewed graphs were added; exactness on every graph, including the skewed ones, remains required for a nonzero score, and the skewed graphs are graded by the separate criterion. The serial baseline is frozen independently of the pass/fail comparison. Changing the comparison implementation cannot change the continuous score. A tenfold speedup over the serial baseline maps to 50; every finite speedup maps below 100, with no artificial finite upper cutoff. The time for one run is trusted-parent monotonic wall-clock time from worker release through library loading and symbol lookup, constructors, cold worker/OpenMP initialization, `CountTotal` including any vertex ranking it performs, result delivery, and worker termination. It excludes graph-file input, single-threaded wedge preprocessing, process creation, and isolation setup. No overhead is subtracted.

A submission that does not clear the pass criterion is still scored normally, so exploratory development submissions remain useful. Once a submission has a known development pass, prefer changes that preserve it unless the evidence supports a stronger all-workload result. Comparison values, per-graph rows, and every graded-suite measurement remain private.

## Calibration

The material-win criterion requires `median(comparison_i / candidate_i) > 1.05` over 31 adjacent alternating pairs. This fixed engineering threshold is not a statistical confidence guarantee. On an AMD EPYC 7R13 calibration host, the earlier comparison, which always computed the `ACOCORE` ranking, took 0.517132435 seconds with one worker and 0.246523506 seconds with four on a `(20000,22000,96)` graph; the current comparison, which selects its ordering as described under Design space, took a four-worker median of about 0.14 seconds on that graph and about 0.29 seconds on the `(20000, 22000, 1900000)` skewed graph. On `(10000,11000,64)`, the earlier comparison's 1/2/4/8-worker medians were 0.126569477, 0.093923388, 0.068519593, and 0.074213514 seconds. Four workers were selected because they were fastest in the bounded scaling diagnostic; the larger same-generator suite also moves the experiment away from the source task's reported startup-dominated regime. These are environment calibration results, not solver acceptance evidence.

## Feedback and grading

The judge returns a score after every submission. At most 100 development submissions are allowed. Development evaluates four public workload shapes. Final independently re-evaluates those four public workloads together with four hidden same-distribution workloads; it never reuses development measurements. Hidden seeds and workload membership are not disclosed.

Development feedback returns the approved absolute candidate metrics `dev_counting_seconds_median` (the median of the four per-graph medians) and `dev_counting_seconds_p90` (their nearest-rank 90th percentile), plus privacy-safe `beat_spec_progress` aggregates for each metric: `{passed, failed, total, pass_percent, fail_percent}`. These aggregates contain no workload identifiers or per-workload rows. Reference values, margins, the private overall `beats_reference` flag, and all final-suite metrics are withheld.

## Submission

The submitted artifact is the non-empty text patch:

```text
/workspace/payload/solution.patch
```

From the prepared workspace, inspect your changes and create the patch artifact:

```bash
cd /workspace/repo
git diff --stat
git diff
/frontier/task/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 -cf /workspace/train.tar payload
frontier-submit train /workspace/train.tar
```

For a local compile check, run:

```bash
bash /workspace/repo/frontier-build.sh /workspace/repo
```

The judge rejects changes outside the editable surface rather than silently dropping them. It applies the patch to a pristine copy, performs a clean build, and evaluates that rebuilt program. An empty patch is invalid. Unstaged files, copied source trees, executables, generated inputs, build products, timing logs, and other binary content are not included as valid source changes.

## Resources

The evaluation environment provides exactly 4 vCPUs, 8,192 MiB of memory, and 4,096 MiB of storage. It uses Ubuntu 24.04 with a C++14/OpenMP toolchain. The build timeout is 180 seconds, each counting command has a 45-second wall-clock timeout and a 180-second CPU limit, and independent final evaluation has a 1,800-second budget. Worker computation uses four OpenMP threads; trusted preprocessing before worker creation uses one thread.
