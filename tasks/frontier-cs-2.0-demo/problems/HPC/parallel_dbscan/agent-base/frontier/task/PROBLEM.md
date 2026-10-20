# Exact DBSCAN Connectivity

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

DBSCAN is a density-based clustering algorithm for points in a metric space: it finds dense regions without requiring a preset number of clusters and treats isolated points as noise. A point is *core* when its closed `epsilon`-neighborhood contains at least `minPts` points, including the point itself; core points belong to the same cluster exactly when a chain of core points links them with every consecutive distance at most `epsilon`. A non-core point is a border point when it lies within `epsilon` of a core point, and it belongs to every core component that touches it; otherwise it is noise. The supplied C++ project is a shared-memory, exact, higher-dimensional DBSCAN implementation derived from the Problem Based Benchmark Suite codebase. It reads whitespace-delimited Euclidean point sets together with `epsilon` and `minPts`; the scored generated sets contain 12,000 or 14,000 points in 3 or 7 dimensions, plus 200,000-point seed-spreader sets in 5 and 7 dimensions. It writes each clustered point's zero-based input index and component membership, omitting noise points. Correct output must preserve the exact core/border/noise classification, the partition of core points, and all memberships of border points; component identifiers themselves may be consistently renamed. The secondary continuous objective is to maximize the score by minimizing the geometric mean of elapsed computation time over the original small workloads after exact correctness is established (four development workloads; all eight public-plus-hidden small workloads in final). The strict reference comparison covers every workload, including the 200,000-point sets (six development workloads; twelve in final).

The implementation of `ClusterCoreBCPGridParallelBucketingND`, the core-connectivity stage called between retained core detection and border assignment, and the bodies of its declared connectivity helpers have been removed and must be supplied.

## Your workspace

The retained project is available read-only at `/frontier/task/repo`; prepare your writable copy at `/workspace/parallel_dbscan`. It contains the DBSCAN command-line program, point loader, grid construction, core-point detection, border assignment, output writer, build files, headers, compatibility utilities, and the reconstruction hook in `code/DBSCAN.C`. Only the following paths may change:

```text
code/DBSCAN.C
code/frontier_solution/**
```

`code/DBSCAN.C` may be changed only within its public reconstruction region; the evaluator verifies the retained orchestration surrounding that region. New supporting files may be placed below `code/frontier_solution/`. All other retained files, the authentic `code/DBSCANTime.C` entrypoint, and harness-owned files must remain present and byte-for-byte unchanged. Symlinks in submitted source are rejected. The evaluator applies the submitted patch to a fresh copy of the retained project, so unsubmitted build products and workspace files have no effect on grading.

## Design space

The editable component assigns exact connected-component identifiers to the core points already marked by the retained program. At its boundary it receives the complete point sequence, DBSCAN parameters, the constructed grid representation, a flag for each grid indicating whether it contains a core point, and a pair containing the core-point count and per-point core markers. It must return one entry for every input point for use by the retained border-membership stage. Changes may include code in the designated reconstruction region and new C++ source or header files under `code/frontier_solution/`; the surrounding loader, grid builder, core classifier, border classifier, output representation, command-line entrypoint, and timing boundary are fixed. Any implementation within this surface is permitted if it satisfies exact DBSCAN semantics, the source policy, and the resource limits.

## Workload

Each partition contains six generated point sets: the four original small sets below and two 200,000-point seed-spreader sets. Development uses the public partition; final independently re-evaluates that partition together with the hidden partition, for twelve workloads. The hidden partition has the same shapes with distinct seeds.

Original small sets:

| family | dimensions | points | `epsilon` | `minPts` |
|---|---:|---:|---:|---:|
| SS similar-density | 3 | 12,000 | 0.56 | 8 |
| SS variable-density | 3 | 14,000 | 0.62 | 10 |
| UniformFill | 7 | 12,000 | 0.78 | 5 |
| SS variable-density | 7 | 14,000 | 0.92 | 8 |

For each SS set, points cycle among 12 deterministic centers. Similar-density coordinates use Gaussian noise with standard deviation `0.22`; variable-density center `k` uses standard deviation `0.08 + 0.035*k`, for `k` from 0 through 11. The generator replaces points 1, 3, ..., 23 with copies of points 0, 2, ..., 22, producing 12 duplicate pairs. UniformFill independently samples every coordinate uniformly from `[0, 1.15 * n^(1/d))`, where `n` is the point count and `d` is the dimension.

Seed-spreader sets (the generator family of the source paper's SS datasets, at 200,000 points instead of the paper's 10 million; `minPts` 10 is the paper's default for both dataset families):

| family | dimensions | points | `epsilon` | `minPts` |
|---|---:|---:|---:|---:|
| seed-spreader variable-density | 5 | 200,000 | 800 | 10 |
| seed-spreader similar-density | 7 | 200,000 | 1600 | 10 |

A seed spreader is a random walk in `[0, 100000)^d`. It emits points uniformly in an axis-aligned hypercube of half-width `r` around its current location and moves `50*d` in a uniformly random direction after every 100 points. Before each point after the first, it instead restarts at a uniformly random location with probability `10 / (n - n/10000)`. Similar density keeps `r = 100`; variable density draws `r` from {100, 200, 400} at every restart. The last `n/10000` points are uniform noise. Nearly every point is a core point and each set has roughly ten walk clusters, so the core-connectivity stage processes a large cell graph of adjacent core cells.

Before timing, five development correctness probes (seven in final) cover duplicate and boundary-distance points, shared border membership, a sparse chain with noise, and seeded clustered sets in 3 and 7 dimensions. The 200,000-point sets are checked against a second independent exact oracle. For every timing workload, both the warm-up and every measured execution must also produce the exact full output. Each implementation receives one `-r 1` warm-up and eight measured `-r 5` executions per workload. Candidate and comparison executions alternate order; each execution contributes the median of its five reported DBSCAN times, and the workload time is the median of those eight execution medians. The public and hidden partitions use distinct undisclosed seeds; final freshly evaluates both partitions, including the original public seeds. They represent the stated distributions rather than records to recognize; suite membership, order, repetition, and scale may differ in later task revisions, while the declared evaluation uses the values above.

## Interface

Supply the implementation of:

```cpp
GdLongDataType* ClusterCoreBCPGridParallelBucketingND(
    _seq<pointNd>* points,
    DbscanParams* params,
    Grids* grids,
    bool* isCoreGrid,
    pair<PtLongDataType, intT*> corePointMarkersGrid);
```

Return a newly allocated array of length `points->n`. For every core point, the corresponding entry must be a nonnegative identifier smaller than `points->n`; two core points must have the same identifier if and only if they are connected by a chain of core points whose consecutive Euclidean distances are at most `params->m_epsilon`. Every non-core entry must be `-1`. The retained code subsequently assigns border memberships and writes the result.

The evaluator builds the project with:

```text
cd code && make clean
cd code && make OPENMP=1 -j8
```

Every warm-up and correctness execution uses the complete command below, with `-r 1`; timed executions use the same command with `-r 5`:

```text
OMP_NUM_THREADS=8 OMP_PROC_BIND=true OMP_PLACES=cores ./DBSCAN -eps <epsilon> -minpts <minPts> -r <rounds> -o <output> <input>
```

The input file begins with the two fields `dim D`, followed by one point per line with exactly `D` floating-point coordinates. For example, this is a real five-point, two-dimensional record set:

```text
dim 2          # header: the following records have two coordinates
0 0            # point 0: x=0, y=0
0 0            # point 1: x=0, y=0
1 0            # point 2: x=1, y=0
2 0            # point 3: x=2, y=0
9 9            # point 4: x=9, y=9
```

With `-eps 1 -minpts 2`, one valid output file is:

```text
0 3
1 3
2 3
3 3
```

Each output record has layout `point_index component_id [component_id ...]`. Here points 0 through 3 are core points in one component labelled `3`, while point 4 is noise and therefore has no record. A core-point record has exactly one identifier; a border-point record contains every distinct component identifier to which it belongs; noise has no record. Point indices must be in `[0,n)`, may occur at most once, and records may appear in any order. Component identifiers must be nonnegative; consistent renaming is accepted.

For every round, standard output must contain exactly one positive, finite `>totaltime*seconds*` record. It must also report `>numcore*count*`, `>numborder*count*`, and `>numnoise*count*` values agreeing with the output classification. The retained entrypoint produces these records, and each `totaltime` measures the complete `DBSCAN(...)` call, including grid construction, core detection, core connectivity, border assignment, and cleanup.

## Validity

Validity is decided before performance. The submitted patch must be nonempty, no larger than 262,144 bytes, change no more than 20 files, apply cleanly, and modify only `code/DBSCAN.C` or files under `code/frontier_solution/`. It must preserve all retained source, the authentic entrypoint, and the orchestration outside the reconstruction region; it may not contain source-tree symlinks. The project must build successfully with the declared commands within 90 seconds.

Every process must exit successfully within its 45-second execution timeout. Each execution must create a parseable output file, emit the required finite positive timing records and correct classification counts, and match an independent exact oracle. Exactness includes the closed distance test `distance <= epsilon`, counting the query point toward `minPts`, correct core status and core connected components, correct border/noise status, and every membership of a border point adjacent to multiple components. Build failure, an invalid patch or source tree, a missing file, timeout, crash, nonzero exit, malformed output, invalid timing, count disagreement, or any correctness mismatch yields score zero before timing is considered. The whole train evaluator is additionally limited to 240 seconds; final is limited to 600 seconds and runs with 8 vCPUs, 4096 MiB memory, and 2048 MiB storage.

## Scoring

The secondary continuous objective is to maximize the score on a 0–100 scale. For workload `w`, let `C_w` be the median of the candidate's eight execution medians and `R_w` the corresponding paired comparison median, where each execution median is taken over five valid `totaltime` measurements. For the selected original small-workload set W (four in development, eight in final), define

```text
C = exp((1/|W|) * sum_w ln(C_w))
R = exp((1/|W|) * sum_w ln(R_w))
score = 100 * 0.020 / (0.020 + C)
```

All times are in seconds, and every workload has equal weight in log space. The fixed challenge anchor is 0.020 seconds (20 ms); it is never recomputed or normalized from reference measurements. For finite positive C, the score strictly improves as C decreases, approaches 100 as C approaches zero, and has no finite upper clipping or lower plateau. For example, C = 10, 20, and 40 ms yields approximately 66.6667, 50, and 33.3333 points. Invalid submissions score zero. There is no scientific score minimum.

The 200,000-point sets do not enter this continuous score. The reference does not enter it either. Separately, reference PASS requires exact correctness and strictly C_w < R_w on every workload, small and large (six in development, twelve in final); ties fail, and there is no 5% margin or compensation across workloads. Private reference_score applies the same fixed formula to R, and private margin is score minus reference_score; these comparison diagnostics do not normalize the candidate score.

## Calibration

There is no meaningful runnable naive baseline because the supplied core-connectivity hook returns `NULL`; the retained caller requires a valid per-point array before it can complete border assignment and output. A first valid exact implementation therefore establishes the solver's own timing baseline. The two solver-visible timing summaries are absolute elapsed seconds measured in the evaluation container, and small changes can be affected by normal scheduling and system noise. Treat an isolated small movement as inconclusive and use repeated development results to distinguish persistent changes from noise. Correctness is an exact hard gate rather than a noisy metric: a single semantic mismatch produces zero, regardless of elapsed time.

## Feedback and grading

The judge returns a score after every submission. At most 100 development submissions are allowed. For a valid development submission it also exposes `candidate_elapsed_seconds_median`, the median of the six candidate workload times (small and large), and `candidate_elapsed_seconds_p90`, the nearest-rank 90th percentile of those six times (the largest value when there are six). These summaries contain candidate timings only; workload identities, workload-level measurements, comparison measurements, ratios, and graded measurements are withheld. Invalid submissions receive score zero and no timing metrics.

Final independently evaluates the union of the six development instances and the six hidden instances. The hidden partition uses the same shapes and distributions with distinct undisclosed seeds. Every public and hidden result is freshly measured; train measurements are never reused. Continuous scoring covers the eight small final workloads; reference PASS covers all twelve. Reference PASS requires exact correctness and strictly lower candidate elapsed time than the reference for every workload (C_w < R_w), with no numeric tolerance or compensation across workloads.

## Submission

The task payload is the nonempty source patch `/workspace/submission.patch`. From your prepared workspace, run:

```text
/frontier/task/bin/make_submission.sh
```

The script stages the workspace changes and writes a source patch and packages it as `payload/solution.patch` in `/workspace/submission.tar` containing only `code/DBSCAN.C` and `code/frontier_solution/`. Changes outside that editable surface are rejected by evaluation rather than silently included or ignored. The evaluator applies the patch to a clean retained tree, validates it, builds it, and runs the correctness and timing suites.

## Resources

The runtime provides 8 vCPUs, 4096 MiB of memory, and 2048 MiB of storage in a prebuilt Linux container with Python 3.11, `g++`, and OpenMP. OpenMP executions use eight workers with `OMP_PROC_BIND=true` and `OMP_PLACES=cores`. The build timeout is 90 seconds, each DBSCAN process has a 45-second timeout, and the train evaluation has a 240-second timeout and final evaluation has a 600-second timeout. The scored executable is built with `OPENMP=1` and `make -j8`; no accelerator is used.

See [the platform guide](bin/README.md) for the identical train/final tar layout, generic submission commands and final-selection lifecycle.
