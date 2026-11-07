# Batch Local Counting

Supply the batch (`-b -share`) path of the SCOPE exact local subgraph counter and make the
complete six-case batch as fast as possible on one core. Every output file must equal the
independent counting path's output byte for byte; the score rewards the median speedup over that
independent path.

## Task

SCOPE (Li and Yu, "Fast Local Subgraph Counting", PVLDB 17) counts, for every vertex of a data
graph, the matches of a small pattern graph that place a chosen pattern orbit at that vertex. In
this C++17/CMake tree the batch implementation selected by `-share` has been removed; the
independent (non-shared) counting path is intact and defines the expected output.

## What you may change

- Only these ten files: `counting/forest.h`, `counting/forest.cpp`, `counting/execution.h`,
  `counting/execution.cpp`, `counting/aggregation.h`, `counting/aggregation.cpp`,
  `executable/scope.cpp`, `executable/command.cpp`, `plan/tree.h`, `plan/tree.cpp`. Any
  declaration or implementation in them is editable.
- No file may be added or deleted. Graph, pattern and triangle loading, the result file format,
  dependencies, build configuration and the data files are fixed (`saveCount` in
  `counting/execution.cpp` is editable as long as the format stays).
- Added lines must not contain, case-insensitively, `web-spam`, `rec-movielens`,
  `bio-grid-yeast`, `4voc` or `5voc` (rejected as `content_policy`).

## Interface

The judge builds `scope.out` and runs, from the repository root, once per graph and query
collection:

```text
.frontier-build/executable/scope.out -q <query-dir> -d exp/data_graph/<graph>.txt \
  -t <triangle-file> -r <empty-result-dir> -b -share
```

`-q` is a directory of query files, `-d` the text data graph, `-t` the binary
triangle-intersection file, `-r` an existing empty output directory; `-b` makes `-q` and `-r`
directories and `-share` selects the batch path. Paths are absolute; no other flags are passed.

- Data graph: `num_vertices num_undirected_edges`, then one `smaller_id larger_id` line per edge;
  IDs are consecutive from 0 (`web-spam.txt` starts `4767 37375`, `0 3015`).
- Query file: `num_pattern_vertices num_undirected_edges`, the edges, then `1 orbit_vertex_id`.
  `15.txt` is `5 4` / `0 2` / `0 4` / `1 3` / `1 4` / `1 3` (the orbit of vertex 3).
- Triangle file (little-endian, read by the supplied loader): `offset1`, `offset2`, `offset3`
  (each `m + 1` uint32), one int64 `triangle_count`, then `triangle_count` uint32 endpoint IDs.
- Output: for each query file `<name>`, the file `<result-dir>/<name>` with one
  newline-terminated signed 64-bit decimal per data vertex (line `i + 1` for vertex `i`, no
  header). The local count is `|ISO_p,o(v)| * |theta| / |Aut(p)|`: injective edge-preserving
  mappings of pattern `p` with representative `o` at `v`, times the size of `o`'s automorphism
  orbit `theta`, divided by the size of the automorphism group. For `15.txt` on `web-spam` the
  file begins `37589`, `342369`, `81862`.

## Correctness

A run scores 0 if nauty or `scope.out` does not build, a counting process times out, crashes or
exits nonzero, writes no result files, or any filename or byte differs from the independent
path's output in any case or round, or changes between rounds. Limits: nauty `./configure`
120 s and build 300 s (one job), CMake configure 120 s, `scope.out` build 600 s (Release, one
job), each counting invocation 180 s.

## Workload

One case is one graph with one query batch; a timing round runs all six cases:

| Graph | Vertices | Undirected edges | Triangle file in `exp/data_graph/` |
| --- | ---: | ---: | --- |
| `web-spam` | 4,767 | 37,375 | `web-spamt.bin` |
| `rec-movielens` | 7,601 | 55,384 | `rec-movielenst.bin.gz` (expanded by the judge) |
| `bio-grid-yeast` | 6,008 | 156,945 | `bio-grid-yeastt.bin.gz` (expanded by the judge) |

Each graph runs a five-vertex batch of 24 queries from `exp/pattern_graph/5voc` and a batch of
four queries from `exp/pattern_graph/4voc`. The usable records are the 48 `5voc` queries and
eight `4voc` queries (one 2-, one 3- and six 4-vertex patterns) on which the two counting paths
agree; alternating records form the train and final halves, which are disjoint and use the same
graphs and protocol. Neither half's membership is disclosed. Five output-producing rounds run the
independent path, the reference and the candidate in rotating order. A round time (`B_r`,
`C_r`) runs from the first case to the last: the six complete `scope.out` processes (loading,
planning, counting, writing) plus the judge's fixed hashing and deletion of each case's output.
The per-case times used for beating the reference cover the `scope.out` process only. Building
and triangle expansion are not timed.

## Scoring

With `B_r` and `C_r` the round times of the independent path and the candidate in round `r`:

```text
L     = max(0, median_{r=1..5} log2(B_r / C_r))
score = 100 * L / (2 + L)
```

A 1x, 2x, 4x and 16x median speedup score 0, 33.3, 50 and 66.7; no finite speedup reaches 100.

**Beats the reference** when, on each of the six cases, the candidate has exact output (`exact_counts`) and a
candidate case time strictly below the reference's in all five rounds (`batch_latency_seconds`);
equality is not a win. A candidate whose source equals the reference's is timed once with it,
so it ties.

## Running locally

nauty's `configure` rewrites tracked files (`utility/automorphism/nauty.h`), which the patch may
not change, so build in a copy of the working repo:

```sh
rm -rf /workspace/build && cp -a /workspace/repo /workspace/build
cd /workspace/build/utility/automorphism && sh ./configure \
  && sed -i 's/^CFLAGS=/CFLAGS= -fPIC /' makefile && make -j1 nauty.a && mv nauty.a libnauty.a
cd /workspace/build && cmake -S . -B .frontier-build -DCMAKE_BUILD_TYPE=Release \
  && cmake --build .frontier-build --target scope.out -j1
```

Run once with and once without `-share` on the same query directory and compare the result
directories.

## Notes

The queries are a representative batch of exact local counting queries: do not recognise or
special-case individual queries, graphs or row counts.
