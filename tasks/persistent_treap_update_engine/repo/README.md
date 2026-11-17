# Persistent Treap Update Engine

Make concurrent transactional updates to Contreap, a persistent treap index, faster while every
version stays correct. Higher mixed-workload throughput gives a higher score. A valid submission
builds, passes the concurrent snapshot checks, and exits normally on every case and
configuration.

## Task

Contreap (Wang et al., "Efficient Concurrent Updates to Persistent Randomized Binary Search Trees",
PVLDB 2025) is an in-memory ordered multiset index. Each treap node stores an integer key, its
multiplicity, and the total multiplicity of its subtree, which supports range sums. Every
transaction gets its own read-only root, so an earlier version can be queried after later updates.
In this tree the update scheduler and the insert/delete routines have been replaced by a serial
functional scaffold. Replace them with a concurrent engine that keeps the version contract below.

## What you may change

- `lib/treap/scheduler.hpp`
- `lib/treap/search_insert.hpp`
- `lib/treap/search_delete.hpp`

These three files are the only ones you may change. You may not add, delete or rename files.
Declarations, helper types and bodies inside them may change, as long as they compile as C++20
with the fixed translation units. Everything else is fixed: trace loading, initial-tree
construction, transaction and query dispatch, timing and reporting, `treap::node_ptr`, the node
fields, the subtree augmentation and `range_estimate`.

The edited headers must not contain any of these strings: `clock_gettime(`, `std::chrono`,
`steady_clock`, `high_resolution_clock`, `rdtsc` or `rdtscp`.

## Interface

The fixed adapter (`main.cpp`, `db/contreap.hpp`) drives `treap::scheduler`:

- `treap::scheduler(size_t num_threads, size_t num_tasks, size_t block_size, node_ptr root0)`
  starts version 0 at `root0`. `num_tasks` is the exact number of transactions that will be
  submitted.
- `insert_elem(key)` submits the next version and adds one occurrence of `key`.
- `delete_elem(key)` submits the next version and removes one occurrence of `key` (always live).
- `nop()` submits the next version unchanged and returns its number (the query's version).
- `get_snapshot(version)` returns the complete immutable root of that exact version. Query-client
  threads may call it while submission or processing is still running, and it must never expose
  an incomplete tree.
- `process()` returns only after every submitted version is complete.
- `last_version()` returns the last submitted version number.

Version `v` must hold exactly the initial multiset followed by transactions 1 through `v`. Earlier
roots never change after later transactions, and no root contains a node from a later version.

The benchmark is invoked as:

```
BUILD_DIR/main-candidate contreap DATA_DIR/case-I.data -threads THREADS -clients CLIENTS -batchsize 1000
```

The trace file uses the native x86-64 (little-endian) C++ ABI:

```
offset  size  field
0       8     initial_count: native size_t
8       8     transaction_count: native size_t
16      8*n   initial_keys[n]: native uint64_t values
16+8*n  24*m  transactions[m], each record:
               +0   1 byte   type: uint8_t (0=query, 1=insert, 2=delete)
               +1   7 bytes  native structure padding
               +8   8 bytes  arg0: uint64_t
               +16  8 bytes  arg1: uint64_t
```

For insert and delete, `arg0` is the key and `arg1` repeats it. For a query, they are the
inclusive left and right endpoints. Example with `n=2`, `m=3`:

```
header:       initial_count=2, transaction_count=3
initial_keys: [4, 9]
record 1:     type=1 (insert), arg0=7, arg1=7
record 2:     type=2 (delete), arg0=4, arg1=4
record 3:     type=0 (query),  arg0=5, arg1=10
```

Versions 0 to 3 are `{4,9}`, `{4,7,9}`, `{7,9}` and `{7,9}`. The version-3 sum of multiplicities
over keys 5 through 10 is 2. The judge reads throughput from the last stdout line, which has this
shape:

```
# Loaded records:	2
# Loaded transactions:	3
# Time Used (S)
contreap	case.data	3	0.002
# Transaction throughput (KTPS)
contreap	case.data	3	1.5
```

Throughput is `m / elapsed_seconds / 1000` KTPS.

## Correctness

Before timing, a fixed harness runs once per case and configuration on its own trace (same `n`,
`m`, profile). After all operations are submitted and before `process()`, `CLIENTS` threads
concurrently fetch version 0, the final version and every `max(1, floor(m/16))`-th version; each
root must have strict BST order, exact multiplicities and total, `node.ver <= version` and a
correct inclusive range sum. Then `process()` must complete.

The evaluation scores 0 if: a build fails (300 s per compile); a harness, benchmark or
historical-query run exits nonzero, crashes, deadlocks or fails a snapshot check; a run times out
at 30 s twice (only a timeout is retried, once); the throughput line is missing or not finite and
positive, or no positive peak memory is recorded; or the historical-query program prints no
result or a non-positive time.

A wrong historical answer does not zero the score. It fails `historical_query_us` for that pair.

## Workload

A suite is six traces with profiles update-heavy, hotspot, balanced, read-heavy,
hotspot, update-heavy. A trace with seed `s` has `n = 80,000 + (s mod 10,000)` shuffled initial
keys `2, 4, ..., 2n` and `m = 450,000 + (s mod 90,000)` transactions. Insert keys lie in `[1, 4n]` (duplicates raise multiplicity); a delete
removes one occurrence of a present key (a query if the multiset is empty). A query is
`[left, left + 127]` with `left` in `[1, 4n - 128]`. In hotspot traces, 98% of keys and query
starts fall in `[1, max(128, floor(n/128))]`.

| Profile | Insert | Delete | Range query |
| --- | ---: | ---: | ---: |
| balanced | 42% | 25% | 33% |
| read-heavy | 18% | 10% | 72% |
| update-heavy | 60% | 28% | 12% |
| hotspot | 55% | 35% | 10% |

Each trace runs with `(server threads, query clients)` = `(4,1)`, `(5,1)`, `(5,2)`, `(6,2)`: 24
pairs. Train and final use separate seeds of the same procedure (one distribution, no shared
trace; final seeds withheld).

Per pair, candidate, reference and scaffold run the benchmark five times each; the
judge records throughput and each process's peak resident memory (`ru_maxrss`). In each
repetition the candidate and reference also run the judge's historical-query program (built from
your headers), which submits the whole trace, calls `process()`, then times only 100,000
single-threaded range sums over a uniform version in `0..m`, width `w` in `0..floor(4n/10)`
and left endpoint in `1..4n-w`. The scaffold supplies the expected answer checksum.

## Scoring

With `C[i,q,r]`, `R[i,q,r]` and `N[i,q,r]` the candidate, reference and scaffold throughputs (KTPS)
for case `i`, configuration `q` and repetition `r = 1..5`:

```
G[i,q] = median(C[i,q,1]/N[i,q,1], ..., C[i,q,5]/N[i,q,5])
Q      = exp((1/24) * sum_i sum_q ln(G[i,q]))
score  = 0                                      if Q <= 1
score  = 100 * ln(Q) / (ln(10) + ln(Q))         if Q > 1
```

The score uses throughput only; the reference is not part of it. Anchors: the scaffold (`Q = 1`)
scores 0, `Q = 10` scores 50, `Q = 100` scores about 66.7, and 100 is approached only
asymptotically.

**Beats the reference** when all 24 pairs clear all three metrics, compared on five-sample medians
with no tolerance (equality fails):
- `case_throughput_ktps`: the candidate's median throughput is strictly higher than the reference's.
- `peak_memory_mib`: the candidate's median peak resident memory is strictly lower.
- `historical_query_us`: all five candidate checksums match the scaffold's, and the candidate's
  median time per query is strictly lower.

The score is still returned when this verdict fails.

## Running locally

Build like the judge, from `/workspace/repo` (write a `case.data` trace in the format above;
none ships):

```
g++ -std=c++20 -O3 -DNDEBUG -march=native -msse4 -maes main.cpp -I. -lpthread -lmimalloc -o main-candidate
./main-candidate contreap case.data -threads 4 -clients 1 -batchsize 1000
```

## Environment

The judge uses GCC with C++20, pthreads, mimalloc and AES-NI on Ubuntu 24.04 x86-64, with 8 vCPUs
and 16 GiB of memory.

## Notes

Do not special-case seeds or traces. Timing is native and varies with host contention. Repeat a measurement before you trust a small
gain.
