# Exact DBSCAN Connectivity

Make exact parallel DBSCAN faster. You supply the core-connectivity stage of a shared-memory
DBSCAN implementation; it must reproduce exact DBSCAN output on every input, and the score rises
as the measured clustering time falls. **First priority: beat the reference on every workload
(see Scoring); the continuous score is secondary** and never compensates for a failed comparison.

## Task

DBSCAN clusters points by density. A point is *core* when its closed `epsilon`-neighborhood holds
at least `minPts` points, counting itself. Core points are in one cluster exactly when a chain of
core points links them with every step at most `epsilon` apart. A non-core point within `epsilon`
of a core point is a *border* point and belongs to every cluster it touches; any other point is
noise. The project is the exact, higher-dimensional, grid-based DBSCAN of Wang, Gu and Shun
(SIGMOD 2020), built on the Problem Based Benchmark Suite. The function
`ClusterCoreBCPGridParallelBucketingND`, called between the retained core detection and border
assignment, and the bodies of its connectivity helpers have been removed; supply them.

## What you may change

- `code/DBSCAN.C`, only inside its public reconstruction region (from the
  `// Public reconstruction interfaces.` comment to the comment that starts
  `Assigns border points after assigning core points to clusters.`).
- New files under `code/frontier_solution/` (C++ sources and headers).
- At most 20 changed files. Everything else (the loader, grid builder, core and border
  classifiers, output writer, `code/DBSCANTime.C` entrypoint, `Makefile`, headers) must stay
  byte-for-byte unchanged; no file may be deleted, and no symlinks or binary files.

## Interface

```cpp
GdLongDataType* ClusterCoreBCPGridParallelBucketingND(
    _seq<pointNd>* points,
    DbscanParams* params,
    Grids* grids,
    bool* isCoreGrid,
    pair<PtLongDataType, intT*> corePointMarkersGrid);
```

It receives the full point sequence, the DBSCAN parameters, the constructed grids, a per-grid flag
saying whether the grid holds a core point, and the core-point count with per-point core markers.
Return a newly allocated array of length `points->n`: for every core point a nonnegative identifier
smaller than `points->n`, equal for two core points exactly when they are connected by a chain of
core points with consecutive Euclidean distances at most `params->m_epsilon`; `-1` for every
non-core point. The retained code then assigns border memberships and writes the output.

The judge builds with `cd code && make clean && make OPENMP=1 -j8` and runs every execution as

```text
OMP_NUM_THREADS=8 OMP_PROC_BIND=true OMP_PLACES=cores ./DBSCAN -eps <epsilon> -minpts <minPts> -r 1 -o <output> <input>
```

The input starts with `dim D`, then one point per line with `D` floating-point coordinates. The
output has one record per clustered point, `point_index component_id [component_id ...]`: one
identifier for a core point, every distinct component for a border point, no record for noise;
indices in `[0,n)`, each at most once, any order; identifiers nonnegative and consistently
renamable. Example: `dim 2` with points `(0,0) (0,0) (1,0) (2,0) (9,9)` and `-eps 1 -minpts 2`
gives `0 3`, `1 3`, `2 3`, `3 3` (one component; point 4 is noise). Standard output must contain,
per round, exactly one positive finite `>totaltime*seconds*` record and `>numcore*`,
`>numborder*`, `>numnoise*` counts that match the output; the retained entrypoint prints them, and
`totaltime` covers the whole `DBSCAN(...)` call.

## Correctness

Every execution, including warm-ups, must exit successfully within 45 seconds, write a parseable
output file, print valid timing and counts, and match an independent exact oracle (the
200,000-point sets: a second, grid-based exact oracle): exact core/border/noise status, exact core
components and every membership of a border point, with the closed test `distance <= epsilon` and
the query point counted toward `minPts`. The build must finish within 90 seconds. The build and every
execution run isolated: an execution reads only the binary and its input and writes only its output
and a private temporary directory that is discarded, so no state carries between rounds or
executions. Any failure scores zero. A train
evaluation that runs longer than 240 seconds is stopped and reported as failed.

## Workload

Each partition has four small sets and two 200,000-point seed-spreader sets. Train uses the public
partition (six workloads, after five correctness probes); final freshly re-measures the public
partition and adds a hidden one with the same shapes and distinct seeds (twelve workloads, seven
probes). Seeds are undisclosed; the sets stand for the stated distributions. Do not special-case
workloads by input, size, seed, order or phase (train/final). The probes cover duplicates, boundary distances, shared border points, a sparse chain
with noise, and seeded clustered sets in 3 and 7 dimensions.

| family | d | points | `epsilon` | `minPts` |
|---|---:|---:|---:|---:|
| SS similar-density | 3 | 12,000 | 0.56 | 8 |
| SS variable-density | 3 | 14,000 | 0.62 | 10 |
| UniformFill | 7 | 12,000 | 0.78 | 5 |
| SS variable-density | 7 | 14,000 | 0.92 | 8 |
| seed-spreader variable-density | 5 | 200,000 | 800 | 10 |
| seed-spreader similar-density | 7 | 200,000 | 1600 | 10 |

SS sets cycle among 12 fixed centers with Gaussian noise (σ = 0.22; variable density
σ = 0.08 + 0.035·k for center k = 0..11); points 1, 3, ..., 23 copy points 0, 2, ..., 22.
UniformFill samples each coordinate from `[0, 1.15·n^(1/d))`. A seed spreader walks in
`[0, 100000)^d`, emitting points uniformly within half-width `r` of its location, moving `50·d`
in a random direction every 100 points and restarting uniformly with probability
`10 / (n − n/10000)` per point; `r = 100` (similar) or drawn from {100, 200, 400} at each restart
(variable); the last `n/10000` points are uniform noise. Each implementation gets one warm-up and
eight repeats per workload; a repeat is five fresh `-r 1` processes, candidate and reference
alternate, and the workload time is the median of the eight repeat medians of `totaltime`.

## Scoring

```text
C = exp(mean over the small workloads w of ln(C_w))      # 4 in train, 8 in final
score = 100 * 0.020 / (0.020 + C)                         # times in seconds
```

`C_w` is the candidate's workload time. The 20 ms anchor is fixed: C = 10, 20 and 40 ms give
about 66.67, 50 and 33.33, and the score approaches 100 as C falls, with no cap. The
200,000-point sets and the reference do not enter the score.

**Beats the reference** when the submission is exactly correct and, on every workload, small and
large (six in train, twelve in final), `elapsed_seconds` is strictly lower: `C_w < R_w`, with
`R_w` the reference's time from the same paired runs. Tolerance: none; ties fail, and there is
no compensation across workloads.

## Running locally

```sh
cd /workspace/repo/code && make clean && make OPENMP=1 -j8
OMP_NUM_THREADS=8 OMP_PROC_BIND=true OMP_PLACES=cores ./DBSCAN -eps 1 -minpts 2 -r 1 -o out.txt exampleData/tiny.txt
```

## Environment

8 vCPUs with eight OpenMP workers and GCC 12 (`g++`), as the judge uses. The shipped hook returns
`NULL`, so the starter does not run until you implement it. Timings are absolute seconds and
include scheduling noise.
