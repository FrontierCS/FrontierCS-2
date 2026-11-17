# Exact Butterfly Counting

Implement a fast, exact `CountTotal` for ParButterfly. It returns the total number of
butterflies in a bipartite graph, and lower time is better. Every count must equal an
independent serial count on every graph.

## Task

ParButterfly (Shi and Shun, "Parallel Algorithms for Butterfly Computations", APOCS 2020) is a
C++14/OpenMP system for butterfly computations on bipartite graphs. A butterfly is a complete
2 × 2 biclique: two vertices on each side joined by all four cross edges. This task uses only
the exact global-count path. The body of `CountTotal` has been removed, and the stub returns
`-1`. The vertex and edge counting entrypoints keep callable stubs so that the application
builds. Any vertex ordering and wedge aggregation is allowed, because only the exact total is
checked. The `ACOCORE` argument on the fixed command line does not require a particular
algorithm.

## What you may change

- `butterfly_count_total.h`, `butterfly_count.h`, `butterfly_ecount.h`, `butterfly_peel.h`,
  `butterfly_epeel.h`, `butterfly_putils.h` and `butterfly_utils.h`, all in `apps/hyper/`.
  You may also delete them.
- C++14 sources under `apps/hyper/solver_extensions/`. New files are allowed only there.
- A submission may change at most 20 files.

Everything else is fixed: `apps/hyper/BipartiteButterflies.C`, the graph loader, the Makefile
and build wiring, `ligra/`, `lib/` and the `frontier-*.sh` scripts.

Solutions must not depend on graph file names, suite membership, evaluation order, seeds, or
state kept between runs.

## Interface

Keep this signature in `butterfly_count_total.h`:

```cpp
long CountTotal(bipartiteCSR& GA, bool use_v, long num_wedges,
                long max_wedges, long max_array_size,
                CountType type, RankType tw);
```

The arguments are:
- `GA` is the loaded CSR graph.
- `use_v` is the side chosen by the fixed preprocessing, and `num_wedges` is that side's wedge
  count.
- `max_wedges` and `max_array_size` are the `-m` and `-a` capacities.
- `type` is `BATCHS` and `tw` is `ACOCORE`.

Return the exact number of distinct butterflies. It must be non-negative and the same on every
call.

Under judging, your `CountTotal` is compiled into a separate library that receives the
pristine parent's `bipartiteCSR`, so keep the layout of `bipartiteCSR` and the `CountType`/`RankType` enums in
`butterfly_utils.h` compatible with the pristine ones.

The judge runs this command line with the last argument replaced by the generated graph's path. The
pristine parent loads and preprocesses the graph with `OMP_NUM_THREADS=1`; the isolated worker that calls `CountTotal` uses 4
threads with `OMP_DYNAMIC=FALSE`. Locally, run it on a `frontier-build.sh` build as:

```bash
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ./apps/hyper/BipartiteButterflies -countType BATCHS -rankType ACOCORE -per TOTAL -r 0 -m 200000000 -a 200000000 graph.adj
```

The input is whitespace-delimited ASCII in `AdjacencyHypergraph` format, with the fields in
this order:

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

Offsets and vertex identifiers are zero-based decimal integers. Each offset starts a vertex's
neighbor block, and the next offset (or the array length, for the last vertex) ends it. The two
arrays hold the same edges in opposite directions, so `m_first == m_second`.

Example: the complete file for `K2,2`, with two vertices per side and all four edges.

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

The application prints:

```text
number of butterflies: 1
```

Under judging, the trusted parent also prints exactly one `FS2_TRUSTED_SECONDS=<positive finite
seconds>` record. In the
judged worker, the stdout and stderr of your code go to `/dev/null`.

## Correctness

A graph is exact when the candidate's count equals the serial count. Any of these makes the whole
evaluation score 0:
- a changed fixed file or a missing entrypoint;
- a build failure, or a compile step that exceeds 180 s wall or CPU time, 8 GiB of address space
  or 64 MiB per output file;
- a counting run that fails, crashes, or exceeds 45 s wall or 180 s CPU time;
- output without exactly one count record, a negative count, or a timing that is non-finite or
  not positive;
- counts that differ between repeated runs on one graph;
- an inexact count on any graph.

Each run gets 8 GiB of address space and 4 CPUs. `CountTotal` runs with 4 OpenMP threads and
may not create processes, use the network, or read files other than system libraries.

## Workload

One workload is one generated simple bipartite graph.

| suite | uniform-degree graphs | skewed-degree graphs |
|---|---|---|
| train (4) | 3 | 1 |
| final (8) | the 3 train graphs + 3 held-out | the train graph + 1 held-out |

The six uniform graphs have these `(first-partition vertices, second-partition vertices, degree,
undirected edges)` shapes:

```text
(18000, 20000,  88, 1584000)
(19000, 21000,  92, 1748000)
(20000, 22000,  96, 1920000)
(21000, 23000, 100, 2100000)
(22000, 24000, 104, 2288000)
(23000, 25000, 108, 2484000)
```

For each first-side vertex, a seeded linear-congruential generator picks `degree` distinct
neighbors, which are stored sorted.

The two skewed graphs have the `(first-partition vertices, second-partition vertices, undirected
edges)` shapes `(20000, 22000, 1900000)` and `(21000, 23000, 2000000)`. Each edge endpoint is
drawn with probability proportional to `(rank + 1)^-a` on its side, with exponents `a` between
0.75 and 0.85. Drawing stops at that many distinct edges; identifiers are then randomly
permuted. The largest hubs have about 14,000–19,000 neighbors, so the wedge count depends
strongly on the processing order.

Seeds and the assignment of shapes to the suites are withheld. Held-out graphs are not in train
and come from the same distribution.

Per graph, the candidate runs once for its count, then in 31 candidate/comparison and 31
candidate/baseline pairs, with the order within a pair alternating. The trusted parent's
monotonic clock times worker release, library loading, constructors, OpenMP start-up,
`CountTotal` (including any ranking it does), result delivery and worker exit. Graph input,
preprocessing, process creation and isolation setup are not timed.

## Scoring

For each graph `i` in the suite:

```text
q_i   = 1 if candidate_count_i == serial_count_i, else 0
s_i   = median over the 31 pairs j of (baseline_time_ij / candidate_time_ij)
S     = geometric mean of s_i over the uniform-degree graphs only
score = 0                                if any q_i == 0 or S <= 1
score = 100 * ln(S) / (ln(10) + ln(S))   otherwise
```

The baseline is a frozen serial implementation. The skewed graphs count only for exactness and
the beats-reference test, not for `S`. Anchors: the untouched tree returns `-1` and scores 0,
`S = 10` scores 50, and every finite `S` scores below 100.

**Beats the reference** when two conditions hold on every graph in the suite (all 8 in final):
- `butterfly_count`: the candidate and comparison counts both equal the serial count;
- `counting_seconds`: the median over the 31 pairs of `comparison_time / candidate_time` is greater than 1.05.

Equality fails. The comparison follows the paper's ordering rule (Section 6.2.2). With `w_s` the
side-order wedge count and `w_r` the degree-order count, it uses side order when
`(w_s - w_r) / w_s < 0.1` and degree order otherwise, followed by simple batching. This criterion
does not change the score.

## Running locally

Compile check:

```bash
bash /workspace/repo/frontier-build.sh /workspace/repo
```

This builds `apps/hyper/BipartiteButterflies` directly, without the trusted worker.

## Environment

The judge runs on Ubuntu 24.04 with g++ (C++14, OpenMP) and the flags `-std=c++14 -fopenmp
-march=native -O3 -DOPENMP -DBYTERLE -include cstdint`, on 4 CPUs with 8 GiB of memory.

## Notes

The graphs represent a regime and are not inputs to recognize.
