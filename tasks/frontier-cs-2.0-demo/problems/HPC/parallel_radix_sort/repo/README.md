# In-place Integer Redistribution

Make Regions Sort, a parallel in-place radix sorter, faster than `std::sort`. You supply its
region-graph exchange planner; every sort must stay exact, and the score rises with the
geometric-mean speedup over `std::sort`. **First priority: beat the optimized reference on every
workload and every metric (see Scoring); the continuous score is secondary** and never
compensates for a failed comparison.

## Task

Regions Sort (Obeya et al., SPAA 2019) is a shared-memory most-significant-digit radix sorter
for unsigned 32-bit keys, built on the Problem Based Benchmark Suite and Cilk Plus. It splits the
array into blocks, finds each 8-bit bucket's final interval, records misplaced ranges as a region
graph, and asks the retained executor to swap equal-length ranges (`Triangle` records) before
recursing. The bodies of `EdgeListGraph::matchEdgesToTriangle`, its three `extractTriangles`
forms, `extractNode2` and `consumeTriangle` have been removed; supply them.

## What you may change

- `RadixSort/edgelistgraph.h`: the operation bodies, implementation-owned state and helpers. The
  class interface, graph construction, node deletion, the public method signatures and the `Edge`
  and `Triangle` record contracts stay as they are; `EdgeListGraph` stays a non-final class with
  its constructor.
- New C++ headers under `RadixSort/frontier_solution/`, included from `edgelistgraph.h`.
- At most 20 changed files; text only (no binary files, symlinks or deletions); C++14. Everything
  else (sorter, executor, recursion, fallback sorts, entrypoint, build scripts, harness) is fixed.

## Interface

`parallelIntegerSort(array.data(), size, key_function)` sorts in place: on return the array is
ordered by key and is a permutation of the input's complete elements (records with equal keys in
any order). Elements are `uint32_t` keys or `pair<uint32_t,uint32_t>` key/payload records. The
caller uses these `EdgeListGraph` operations:

- `void matchEdgesToTriangle(Edge *fromEdge, Edge *toEdge, Triangle *result)`: write one exchange
  derived from the two edges.
- `template<class T> int extractTriangles(vector<T> &fromRegions, int fromSize, vector<T> &toRegions, int toSize, Triangle *triangles, int triangles_count)`
  and the two overloads for `vector<Edge>` and `vector<Edge *>`: append exchanges at
  `triangles_count` and return the new count.
- `void extractNode2(int node, Triangle *triangles, int *triangles_count)`: write the exchanges for
  `node` from `triangles[0]` and set the count.
- `void consumeTriangle(Triangle *triangle)`: update the graph for one completed exchange.

`Edge(start, amount, from, to)` is a range `[start, start+amount)` currently holding bucket `from`
that belongs to bucket `to`. `Triangle{from, to, offset1, offset2, amount}` swaps the disjoint
ranges `[offset1, offset1+amount)` and `[offset2, offset2+amount)` inside the current segment;
e.g. `Triangle{7, 18, 120, 900, 3}` swaps elements 120–122 with 900–902. All exchanges from one
`extractNode2` call run concurrently, so no array position may appear in two of them; the caller
then deletes the node and calls `consumeTriangle` once per exchange. The count may not exceed
`(K * 256 + 256 + 1) + 256`, with `K` the block count of that sort level.

The judge compiles its benchmark with the pinned GCC 7.5/Cilk Plus compiler:

```text
./frontier-cxx -I <tree>/RadixSort -std=c++14 <benchmark> -DCILKP -DBITS_HACK -O3 -flto -march=native -fcilkplus -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=free -lcilkrts -o candidate
CILK_NWORKERS=4 taskset -c 0-3 ./candidate TYPE DISTRIBUTION N SEED REPETITIONS     # prints: OK <median_ns> <peak_aux_bytes>
```

A second build adds `-DFRONTIER_COMPONENT_TIMER`: it derives `TimedEdgeListGraph` from
`EdgeListGraph`, times every `EdgeListGraph` method call in the sort summed over workers, and its
`OK` record reports that region-graph time instead. The judge also builds the authentic
entrypoint `RadixSort/example.cpp` the same way (without the `--wrap` flags) and runs
`CILK_NWORKERS=4 taskset -c 0-3 ./radixSort -r 1 -c entry.seq` on 24,000 integers
`(i * 2654435761) & 1023`; it must print `array is sorted!`. A sequence file starts with
`sequenceInt` (one integer per line) or `sequenceIntPair` (key/payload pairs).

## Correctness

Every guard and every timed case must produce `OK`, a correctly ordered result, and exactly the
input integers or complete records (order checked by key, preservation on whole pairs).
Compilation failures (90 seconds per compile), benchmark processes over 35 seconds, the smoke test
failing or over 30 seconds, crashes, malformed output, or a peak live auxiliary allocation above
67,108,864 bytes (64 MiB) on any timed case score zero. Auxiliary bytes count C/C++ heap memory
made during any sort call and still live (memory kept from an earlier repetition still counts);
input generation, restoring and checking are outside.
An evaluation still running after 840 seconds scores zero.

## Workload

| identifier | element | distribution | elements | seed |
| --- | --- | --- | ---: | ---: |
| `d-int-uniform` | integer | uniform 32-bit | 3,000,000 | 12011 |
| `d-pair-zipf` | record | `x & 31` if `x % 100 < 70`, else `x & 65535` | 2,600,000 | 12037 |
| `d-int-duplicates` | integer | keys 0–1023 | 3,200,000 | 12041 |
| `d-pair-almost` | record | increasing keys, then `n/100` random swaps | 2,400,000 | 12049 |
| `d-int-zipf075` | integer | `floor(t^4)`, `t = 1 + 176.82794100389228·(x+0.5)/2^32`, capped at 10^9 | 16,000,000 | 12059 |
| `d-int-uniform16m` | integer | uniform 32-bit | 16,000,000 | 12071 |

Arrays come from `std::mt19937` words `x`; a record's payload is its original position. Train
uses these six; final re-measures them and adds six hidden-seed cases of the same shapes
(twelve). Hidden seeds are undisclosed. Do not special-case workloads by size, distribution, seed,
identifier or phase (train/final). Before timing, three untimed guards run once: 24,000 equal integers (7, seed 7103),
19,999 sorted records (seed 7121) and a one-element integer array (seed 7127). Each timed case
runs 21 repetitions in one process and reports the median sort-call time. Candidate and reference
alternate order (candidate first on odd-numbered rows, reference first on even-numbered rows),
then the `std::sort` baseline runs on the same row (first four shapes only), then the two timer
builds in the same order.

## Scoring

```text
S = exp(mean over the original shapes w of ln(B_w / C_w))   # 4 in train, 8 in final
score = 0                              if S <= 1
score = 100 * ln(S) / (ln(10) + ln(S))  otherwise
```

`C_w` is the candidate's median sort time and `B_w` the median of `std::sort` with
`key(left) < key(right)` on the same array, same harness and flags. Parity scores 0, a 10x
speedup 50, and larger speedups approach 100. The 16,000,000-key shapes and the reference do not
enter the score.

**Beats the reference** when, on every timed case (six in train, twelve in final), all three
hold: `sort_time_ns` `C_w < R_w`; `region_graph_ns` `G_w < Q_w`; `auxiliary_bytes`
`A_w <= 0.05 * input_bytes_w` (4 bytes per integer, 8 per record). `R_w` and `Q_w` are the
optimized reference's medians. Tolerance: none; ties fail, and every case and metric must pass.

## Running locally

```sh
cd /workspace/repo && ./frontier-build.sh        # builds ./radixSort with ./frontier-cxx
./frontier-bench.sh -r 1 -c entry.seq            # CILK_NWORKERS=4 taskset -c 0-3 ./radixSort ...
```

## Environment

Ubuntu 24.04, GCC 7.5 with Cilk Plus through the shipped `./frontier-cxx` (provenance in
`frontier-toolchain.json`), four pinned CPUs (`taskset -c 0-3`) and `CILK_NWORKERS=4`.
