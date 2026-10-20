# In-place Integer Redistribution

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

This C++ system is Regions Sort, an open-source shared-memory, most-significant-digit radix sorter for integer keys; its source tree also contains support code from the Problem Based Benchmark Suite. Radix sorting classifies values by a fixed-width portion of each key, starting with the most significant portion in this system, and then sorts the resulting key groups by the remaining portions. Regions Sort divides an array into contiguous blocks, determines the final contiguous interval for each radix bucket, represents misplaced ranges as a region graph, and uses `Triangle` records to request exchanges of equal-length ranges before recursing on the bucket intervals. It uses Cilk Plus for shared-memory parallel execution. The evaluator supplies six public deterministic timed arrays generated in memory from C++ `std::mt19937`: four containing 2,400,000–3,200,000 unsigned 32-bit integers or unsigned 32-bit key/payload records and two containing 16,000,000 unsigned 32-bit integers, as well as three smaller guard arrays and a separate 24,000-element integer sequence file for the project's command-line entrypoint. The sort writes its result back into the supplied array, while the timed benchmark writes the median sort-call time and peak live auxiliary allocation to standard output. A correct result is nondecreasing by integer value or record key and contains exactly the input integers or complete key/payload records; records with equal keys may appear in any relative order. The secondary continuous objective is to maximize the score computed from sort-call latency across all timed workloads in the selected suite. The bodies of `EdgeListGraph::matchEdgesToTriangle`, its three `extractTriangles` forms, `extractNode2`, and `consumeTriangle` have been removed and must be supplied.

## Your workspace

The retained C++ project is at `/workspace/parallel_radix_sort`. You may change only:

```
RadixSort/edgelistgraph.h
RadixSort/frontier_solution/**
```

The entrypoint, radix sorter, exchange executor, fallback sorting code, build scripts, and evaluation harness are part of the task and must not change; harness files are verified. `RadixSort/edgelistgraph.h` retains the required class interface, graph construction, and node deletion. Text source headers may be added under `RadixSort/frontier_solution/` and included by `RadixSort/edgelistgraph.h`. Generated objects, executables, and changes to any other path are outside the submission boundary.

## Design space

The editable surface is the region graph's exchange-planning and graph-update component: the required operation bodies, implementation-owned state and helper declarations, and C++ headers below `RadixSort/frontier_solution/`. The caller-facing operation signatures and the `Edge` and `Triangle` record contracts are fixed. Submitted code must be C++14 and compile with the evaluator command described below. The evaluator observes the completed array, elapsed sort-call time, live auxiliary heap allocation, and, in a separate build of the same harness, the time spent inside `EdgeListGraph` operations during each sort call. The radix sorter, its executor for returned `Triangle` records, recursive control flow, input generation, correctness checks, timing boundaries, allocation instrumentation, and command-line programs are outside the editable surface.

## Workload

Feedback contains the following six public timed cases. Independent final re-evaluates these same six public cases and six hidden-seed cases of the same shapes:

| Feedback identifier | Element type | Distribution | Elements | Feedback seed |
| --- | --- | --- | ---: | ---: |
| `d-int-uniform` | unsigned integer | uniform over the 32-bit generator output | 3,000,000 | 12011 |
| `d-pair-zipf` | key/payload record | skewed (`zipf`): a generator word `x` becomes `x & 31` when `x % 100 < 70`, otherwise `x & 65535` | 2,600,000 | 12037 |
| `d-int-duplicates` | unsigned integer | repeated keys (`duplicates`): keys are in 0–1023 | 3,200,000 | 12041 |
| `d-pair-almost` | key/payload record | nearly sorted (`almost`): initially increasing keys, followed by `n/100` random pairwise swaps | 2,400,000 | 12049 |
| `d-int-zipf075` | unsigned integer | Zipf-like (`zipf075`): `u = (x + 0.5) / 2^32` for a generator word `x`, `t = 1 + 176.82794100389228 * u`, key `= floor(t^4)` computed as `(t*t)*(t*t)` and capped at 1,000,000,000, so `P(key >= v)` grows like `v^0.25` over [1, 10^9] | 16,000,000 | 12059 |
| `d-int-uniform16m` | unsigned integer | uniform over the 32-bit generator output | 16,000,000 | 12071 |

The last two rows (beat spec v3) follow the source paper's `zipf_0.75(10^9, n)` and uniform inputs at 16,000,000 keys (64 MiB per array). They are graded by the strict reference comparisons only; the continuous score below still uses the first four rows (eight in final).

The arrays are generated in memory by the evaluator from C++ `std::mt19937` output. An integer element is one generated key, for example `uint32_t{7}` (`key = 7`). A record element is a complete pair, for example `pair<uint32_t,uint32_t>{7, 18}` (`key = 7`, `payload = 18`); the payload is the element's original zero-based position before any whole-record swaps used to form the input distribution. The evaluator also runs three untimed correctness guards before the timed suite: 24,000 equal integers whose value is 7 (seed 7103), 19,999 already sorted key/payload records (seed 7121), and a one-element repeated-key integer array (seed 7127). Each guard is sorted once. Feedback uses the six table rows in order. Final runs the four original public rows, then the four original hidden cases in matching distribution order, then the two 16,000,000-key public rows and their two hidden-seed counterparts, at the same listed sizes, with 21 repetitions per timed case throughout. Hidden identifiers and seeds are undisclosed. Every final measurement is fresh; no train timings are reused. These workloads are representative inputs, not records or identifiers to recognize.

Only unsigned 32-bit keys and payloads and array sizes representable by the retained 32-bit `sizeT` are in scope. The configured radix is eight bits, giving 256 buckets per radix level.

## Interface

The timed benchmark constructs arrays in memory. Their concrete element layouts, with fields labelled, are:

```
integer element: uint32_t{7}                         // key = 7
record element:  pair<uint32_t,uint32_t>{7, 18}     // key = 7, payload = 18
```

The benchmark calls `parallelIntegerSort(array.data(), size, key_function)`. The call returns no value. On return, the same `size` array slots must be ordered by `key_function`, and their contents must be a permutation of the input's complete elements.

The retained caller uses these operations in `EdgeListGraph`:

- `void matchEdgesToTriangle(Edge *fromEdge, Edge *toEdge, Triangle *result)` writes one exchange derived from the two supplied graph edges.
- `template<class T> int extractTriangles(vector<T> &fromRegions, int fromSize, vector<T> &toRegions, int toSize, Triangle *triangles, int triangles_count)` appends exchanges to `triangles` beginning at `triangles_count` and returns the resulting count.
- The two non-template `extractTriangles` overloads provide the same count-and-append contract for vectors of `Edge` and `Edge *`.
- `void extractNode2(int node, Triangle *triangles, int *triangles_count)` writes the exchanges for `node` beginning at `triangles[0]` and sets `*triangles_count` to the number written.
- `void consumeTriangle(Triangle *triangle)` updates the remaining graph state to reflect a completed exchange.

An edge is constructed as `Edge(start, amount, from, to)`: `start` is a zero-based array offset, `amount` is a contiguous element count, `from` is the bucket currently occupying that range, and `to` is its destination bucket. A `Triangle` is `{from, to, offset1, offset2, amount}` and instructs the retained executor to swap the two equal-length, disjoint ranges `[offset1, offset1 + amount)` and `[offset2, offset2 + amount)`. Both ranges must lie within the current array segment. This concrete record:

```
Triangle{from: 7, to: 18, offset1: 120, offset2: 900, amount: 3}
```

has `from = 7`, `to = 18`, `offset1 = 120`, `offset2 = 900`, and `amount = 3`; it exchanges elements 120–122 with elements 900–902. All exchanges returned by one `extractNode2` call are executed concurrently; across that returned set, every array position must occur in at most one exchange range. After execution, the caller deletes the completed node and invokes `consumeTriangle` once for every returned exchange. The count must not exceed the retained caller's allocated triangle capacity, `(K * 256 + 256 + 1) + 256`, where `K` is the block count for that sort level.

The evaluator builds its benchmark and invokes every generated case as:

```
CILK_NWORKERS=4 taskset -c 0-3 <temporary-directory>/candidate TYPE DISTRIBUTION N SEED REPETITIONS
```

`TYPE` is `int` or `pair`; `DISTRIBUTION` is `uniform`, `zipf`, `duplicates`, `almost`, `sorted`, `equal`, or `zipf075`; and the remaining fields are the element count, unsigned generator seed, and repetition count. A successful benchmark leaves the sorted elements in the supplied array and writes exactly one three-field, whitespace-delimited record to standard output:

```
OK 1842093 116240
```

Here `OK` is the validity marker, `1842093` is the median elapsed time in nanoseconds around `parallelIntegerSort`, and `116240` is the maximum concurrently live auxiliary heap allocation in bytes across the repetitions. Any other record is invalid.

The candidate benchmark is compiled with the pinned compiler as:

```
./frontier-cxx -I <patched-tree>/RadixSort -std=c++14 <evaluator-source> -DCILKP -DBITS_HACK -O3 -flto -march=native -fcilkplus -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=free -lcilkrts -o <temporary-directory>/candidate
```

The evaluator also compiles the same benchmark source with `-DFRONTIER_COMPONENT_TIMER` added (otherwise the identical command) and runs it with the identical invocation. That build includes `RadixSort/edgelistgraph.h` first, derives `TimedEdgeListGraph` from `EdgeListGraph`, and makes the retained sorter construct it in place of `EdgeListGraph`. The wrapper times the constructor, `createParallelGraph`, `extractNode2`, `deleteNode`, `consumeTriangle` and the destructor with `std::chrono::steady_clock`. It sums those durations over all Cilk workers for each sort call. The second field of its `OK` record is then the median of that region-graph time over the repetitions instead of the sort time. Your `EdgeListGraph` must therefore remain a non-final class with the retained constructor and public method signatures.

The evaluator also builds the project's authentic entrypoint and runs this smoke test:

```
CILK_NWORKERS=4 taskset -c 0-3 ./radixSort -r 1 -c entry.seq
```

That entrypoint is compiled as:

```
./frontier-cxx -I <patched-tree>/RadixSort -std=c++14 <patched-tree>/RadixSort/example.cpp -DCILKP -DBITS_HACK -O3 -flto -march=native -fcilkplus -lcilkrts -o <temporary-directory>/radixSort
```

An integer sequence file has the literal header `sequenceInt` and one integer per subsequent record; the loader stores these as the project's 32-bit `intT` values. For example, this valid file contains the two values 7 and 18:

```
sequenceInt
7
18
```

A record sequence has the literal header `sequenceIntPair`, followed by whitespace-delimited key/payload pairs. For example, this valid file contains the records `(key = 7, payload = 18)` and `(key = 3, payload = 41)`:

```
sequenceIntPair
7 18
3 41
```

The authentic command sorts the array in place, prints one `radix sort time: M ms` line for its single `-r 1` round, and must print `array is sorted!` because `-c` requests checking. The evaluator's smoke input contains 24,000 integers generated as `(i * 2654435761) & 1023` for zero-based position `i`.

## Validity

Patch-policy failures, compilation failures, any configured process or evaluation timeout, crashes, malformed benchmark output, a failed authentic-entrypoint smoke test, an incorrectly ordered result, or any change to the input integer/record multiset scores zero before performance is considered. Every guard and all timed cases in the selected suite must pass. For records, ordering is checked only by key, while preservation is checked on the complete `(key, payload)` pair.

Auxiliary allocation during each `parallelIntegerSort` call may not exceed 67,108,864 bytes (64 MiB), a challenge constant not from the source materials. The benchmark counts allocations made through C and C++ heap allocation entry points while the sort call is active and reports the greatest concurrently live byte count across all threads and repetitions. Input generation, source and working-array construction, restoration before a repetition, and result checking occur outside the measured allocation interval. An allocation made during the interval remains live in the counter until it is freed. Exceeding the cap on any timed workload makes the entire submission invalid.

The patch must contain text source changes, apply cleanly, change at most 20 files, be no larger than 262,144 bytes, and touch only the editable surface. Empty patches, binary content, and absolute or parent-traversing paths are rejected.

## Scoring

Let `W` be the original timed workloads in the selected suite: the four original public workloads for feedback and the fresh union of four original public plus four original hidden workloads for final (the 16,000,000-key rows are not part of `W`). Let `C_w > 0` be the candidate's median sort-call nanoseconds and `B_w > 0` the independently measured `std::sort` baseline median on the identical array for workload `w`. Define the equal-weight geometric-mean speedup

```
S = exp((1 / |W|) * sum_{w in W}(ln(B_w / C_w)))
score = 0                                      if S <= 1
score = 100 * ln(S) / (ln(10) + ln(S))          if S > 1
```

Invalid submissions score0. A valid submission no faster than the independent baseline in geometric mean also scores0. Baseline parity maps to0;10x speedup maps to50; larger finite speedups increase strictly toward100 without upper clipping. The implementation uses unrounded log-speedup for numerical stability and retains raw timings and geometric speedup in private audit evidence. The optimized reference is used only by the separate strict per-workload PASS comparison; it does not normalize continuous score.

The baseline independently calls `std::sort` over the same contiguous array with comparator `key(left) < key(right)`, for both unsigned integers and unsigned key/payload records. Equal-key record order need not be stable. It shares the unchanged benchmark's exact generation, array restoration, sortedness/permutation checks, allocation instrumentation,21 repetitions and sort-call timing boundary. It compiles with the same pinned GCC7.5/C++14 optimization/Cilk flags and runs within the same four pinned CPU/four-worker environment,2GiB RAM and2GiB storage. `std::sort` itself is the standard sequential sorting algorithm; the same resources are available to all implementations. No optimized-reference patch is applied to the baseline. The original three untimed correctness guards also validate the baseline. A broken baseline is a verifier runtime failure, not a candidate penalty.

For each workload, retain the original candidate/reference execution order: candidate first on odd-numbered rows and reference first on even-numbered rows, followed by the independent baseline on that same row (original rows only), then the candidate and reference region-graph timer builds in the same order. Every timed process performs21 rounds, measures only its sort call with `std::chrono::steady_clock` in nanoseconds, and performs checking outside that interval. Each call's auxiliary allocation is measured using the same exclusions and64MiB cap. All final measurements are newly executed; no train or baseline timing is cached across evaluations.

## Calibration

The supplied implementation, with the required operation bodies empty, scores 0 because it fails the correctness checks. Correct submissions receive the continuous score defined above. Timings use the repeated sort calls and paired execution order described above; repeated development measurements can help assess changes. There is no minimum score-improvement guarantee, timing tolerance, or required percentage speedup. A higher aggregate score alone does not establish that every workload passed its comparison. The separate per-workload criterion is the strict comparison stated below.

## Feedback and grading

The judge returns a score after every submission. At most 100 development submissions are allowed. **Final independently re-evaluates the six public feedback workloads plus six hidden-seed workloads.** Hidden cases use the same six shapes/distributions with independent fixed seeds. Continuous scoring covers the eight original final workloads; the reference comparison covers all twelve.

Development feedback includes `dev_candidate_median_sort_time_ns`, the median of the candidate's six per-workload median times, `dev_candidate_max_auxiliary_bytes`, the greatest candidate auxiliary-byte measurement in the suite, and `dev_candidate_median_region_graph_ns`, the median of the candidate's six per-workload median region-graph times. A separate, stricter per-workload criterion is evaluated in addition to the continuous score. It requires correct sorting, exact key or record preservation, compliance with the auxiliary-memory and process-resource limits, and, for every timed graded workload `w`, all three of

```
C_w < R_w                       (median sort-call time, sort_time_ns)
G_w < Q_w                       (median region-graph time, region_graph_ns)
A_w <= 0.05 * input_bytes_w     (peak auxiliary bytes, auxiliary_bytes)
```

where `R_w` and `Q_w` are the reference's medians under identical pinned CPU and Cilk-worker settings, `A_w` is the candidate's peak live auxiliary allocation on `w`, and `input_bytes_w` is the element count times 4 (integers) or 8 (records). The 5% bound is the source paper's statement that its in-place sorts use less than 5% auxiliary memory relative to the input; the reference meets it on every workload. Equal median times do not clear the strict criteria; every strictly lower median time clears, subject to the validity gates. The comparison is made independently for each input and metric; results are never averaged across inputs, and every input and metric must pass. Withheld measurements, per-input decisions, and all final-suite measurements are private. The judge reports, per metric, the number and percentage of feedback workloads that pass and fail without identifying the workloads. The continuous score remains a secondary optimization objective; a submission that does not pass the separate criterion is still scored normally.

## Submission

Use `/frontier/task/bin/prepare_workspace.sh` to create `/workspace/parallel_radix_sort`, then edit the allowed surface. `/frontier/task/bin/make_submission.sh` writes and returns the patch path; it does not submit it. Package that patch as the sole regular file `payload/solution.patch` in an uncompressed POSIX tar (an optional `payload/` directory member is allowed). The patch limit remains 262,144 bytes; the tar limit is 276,480 bytes including headers/padding. No executable bit is required. Train, store-final and final use identical archives. See `/frontier/task/README.md` and `/frontier/task/bin/README.md` for exact commands and selected-final behavior.


## Resources

Evaluation runs on Ubuntu 24.04 with GCC 7.5 and Cilk Plus, four pinned vCPUs, 2,048 MiB RAM, 2,048 MiB storage, and no GPU. `CILK_NWORKERS` is 4. The configured build timeout is 120 seconds and the complete evaluation timeout is 900 seconds. Within that evaluation, each benchmark process has a 35-second timeout; the authentic smoke-test process has a 30-second timeout. Each C++ compilation command has a 90-second subprocess timeout.

The supplied `./frontier-cxx` wrapper invokes the pinned GCC 7.5/Cilk toolchain for custom development builds. `./frontier-build.sh` uses it automatically. The compiler dependency and license provenance are recorded in `frontier-toolchain.json`.
