# Hypergraph Core Numbers

Implement `KCore` in the Hygra/Ligra C++ tree to return the exact core number of every vertex of
a hypergraph, as fast as possible: lower verifier-measured runtime scores higher. A submission
is valid only if every checked core number is exact.

## Task

Hygra (Shun, "Practical Parallel Hypergraph Algorithms", PPoPP 2020) extends Ligra to
hypergraphs, where one hyperedge may contain any number of vertices. The core number of vertex
`v` is the largest `k` such that `v` belongs to a remaining vertex set in which every vertex has
at least `k` incident hyperedges, where removing a vertex removes every hyperedge containing it
in full. Isolated vertices have core number 0, and duplicate hyperedges count as distinct
incidences. The body of `KCore` in `apps/hyper/HyperKCore-Efficient.C` has been removed; the
shipped stub returns an empty map.

## What you may change

- `apps/hyper/HyperKCore-Efficient.C`: the `KCore` body and any code outside `Compute`.
- New files below `apps/hyper/frontier_solution/`, included from that file.

Everything else is fixed (driver `ligra/ligra.h`, framework and loader headers, Makefile,
license, inputs). At most 20 changed files; no deletions, no executable files. Also enforced:
- The `Compute` template, from `template <class vertex>` / `void Compute(` to the end of the
  shipped file, appears unchanged exactly once, and `  auto cores = KCore(GA, num_buckets);`
  appears exactly once; otherwise `out_of_scope`.
- The program defines no symbol exported by a linked shared library or the dynamic loader,
  except the replaceable global `operator new` / `operator delete`; otherwise `content_policy`.

## Interface

Keep this entry point exactly:

```cpp
template <class vertex>
array_imap<uintE> KCore(hypergraph<vertex>& GA, size_t num_buckets=128);
```

`GA.nv` and `GA.nh` are the vertex and hyperedge counts, and `GA.mv` and `GA.mh` the two
incidence-array lengths. `GA.V[v]` lists the hyperedges incident to vertex `v`; `GA.H[e]` lists
the members of hyperedge `e`. The returned map must own `GA.nv` `uintE` values; element `v` is
the core number of vertex `v`. `Compute` calls `KCore` once with `num_buckets = 128`.

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

Record `i` of an offset array runs from offset `i` to the next offset, or to the incidence count
for the last record. IDs are zero-based, and `mv = mh`. Example: vertices `0..3`, hyperedges
`{0,1}` and `{1,2}`, vertex `3` isolated:

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

The matching return value is `[1, 1, 1, 0]`.

The judge runs every eight-thread execution as (`<input.adj>` is the final positional argument):

```text
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE ./HyperKCore-Efficient -s -rounds 0 -nb 128 <input.adj>
```

The one-thread pass runs the same command with `OMP_NUM_THREADS=1`. `-s` selects the symmetric
loader and `-rounds 0` leaves only the driver's initial `Compute` call. The judge compiles a
private copy that writes the returned map to standard error as one line beginning
`FRONTIER_CORES`; standard output is discarded.

## Correctness

Each execution is checked against an exact oracle. Any of these on a guard or on an eight-thread
performance execution makes the whole evaluation score 0:
- compilation or linking fails, or either step exceeds 90 s;
- the process exits non-zero, or runs longer than 40 s;
- `KCore` returns no allocation for a non-empty graph;
- standard error exceeds max(65,536, 16·nv + 8,192) bytes, is not ASCII, or has other than exactly
  one line beginning `FRONTIER_CORES`;
- any core number is wrong.

On the one-thread pass, a candidate execution that fails in any of these ways fails only
`sequential_runtime_seconds` for that workload; the score is unchanged.

## Workload

Every evaluation first runs three correctness guards (not timed): 5 vertices with one 2-vertex
hyperedge and three isolates; 5 vertices with two overlapping 3-vertex hyperedges plus a
duplicate of the first; 6 vertices with a singleton edge and a cascade of overlapping edges.

A performance workload is one unweighted, symmetric file from the upstream `randHypergraph`
utility, which removes repeated membership within a hyperedge but keeps hyperedges with
identical member sets distinct.

| suite | workloads |
|---|---|
| train (2) | `feedback-a`: 220,000 vertices, 360,000 hyperedges, 3,599,999 incidences, cardinality 10; `feedback-b`: 300,000, 480,000, 6,719,996, cardinality 14 |
| final (4) | both train workloads plus held-out `graded-a` and `graded-b` |

The held-out files differ in content from the train files and follow the same two strata
(hundreds of thousands of vertices and hyperedges, millions of incidences, cardinality in the low
double digits); their exact sizes are private. The workload files are not in the repo.

For each workload, the candidate and the withheld reference each run seven fresh processes at
eight threads, alternating which goes first; then seven each at one thread. The judge times each
execution from process launch to exit: loading, computation, output and exit. Compilation is not
timed.

## Scoring

With `m_w` the median of the seven eight-thread elapsed times (seconds) on workload `w`:

```text
t     = median(m_w over the active suite)    # 2 train or 4 final workloads: mean of the middle two
score = 100 * exp(-t / 0.25)
```

Anchors: `t = 0.25` s scores 36.79; `t = 1` s scores 1.83; the score approaches 100 as `t → 0`.
The untouched tree scores 0. One-thread times do not enter the score.

**Beats the reference** when every core number is exact and, on every one of the 4 final
workloads, both hold, each comparing seven-run medians with the withheld reference's on the same
workload:
- `runtime_seconds` (eight threads): candidate ≤ 0.97 × reference;
- `sequential_runtime_seconds` (one thread, the paper's T1): candidate ≤ 0.97 × reference, with
  all seven one-thread runs successful and exact.

Equality at 0.97 passes. This criterion does not change the score.

## Running locally

The judge's build, from `/workspace/repo/apps/hyper` (unlike the Makefile, no `-march=native`):

```sh
g++ -std=c++14 -fopenmp -O3 -DOPENMP -DBYTERLE -DHYPER=1 -include cstdint \
  -include ../../ligra/hygra.h -include ../../ligra/index_map.h \
  -include ../../ligra/bucket.h -include ../../ligra/edgeMapReduce.h \
  -I ../../ligra -I . -o HyperKCore-Efficient HyperKCore-Efficient.C
```

Then run the Interface command on your own inputs in that format.

## Environment

Ubuntu 24.04, GCC 13, GNU C++14 with OpenMP. The judge runs with 8 vCPUs and 4 GiB of memory.

## Notes

The workloads are representative inputs, not files to recognize: do not special-case their
contents, sizes or the train/final phase. Times the program prints are ignored. Keep binaries
and other build outputs out of `apps/hyper/frontier_solution/`, since new files there go into
the patch.
