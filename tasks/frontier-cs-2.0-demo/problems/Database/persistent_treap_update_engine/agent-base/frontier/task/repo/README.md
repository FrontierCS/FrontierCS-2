# Persistent Treap Update Engine

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
Contreap is an in-memory ordered index based on a persistent randomized binary search tree, or treap. Each node stores an integer key, that key's multiplicity, and the total multiplicity in its subtree; the subtree total supports range aggregation. Persistence here means that every transaction has a read-only tree snapshot, so callers can query an earlier state after later updates have completed. This supports historical range statistics while an ordered stream of inserts, deletes, and queries changes the current multiset.

For each case, the evaluator generates a deterministic binary trace from a pseudorandom suite seed, following the artifact's mixed insert, delete, and range-query workload model. A trace contains 80,000–89,999 initial integer keys and 450,000–539,999 ordered transactions. The program produces an immutable root for version 0 and for every transaction version, answers inclusive range-sum queries from the corresponding roots, and prints elapsed time and transaction throughput to standard output. For every version `v`, the root must represent exactly the initial multiset followed by transactions 1 through `v`: insert adds one occurrence, delete removes one occurrence, and query leaves the multiset unchanged. Earlier roots must remain complete and unchanged after later transactions, with strict binary-search ordering, the correct multiplicity for every key, correct inclusive range sums, and no node created by a future version. The secondary continuous objective is to maximize the reported score derived from mixed-workload transaction throughput; higher KTPS is better.

The artifact's update scheduler and update routines in the three editable headers have been removed and replaced by a serial functional scaffold. You must supply their replacement while preserving the observable version contract.

## Design space
The contents of the three editable headers are the implementation surface. Declarations, helper types, and function bodies may be changed there, provided the resulting source remains C++20, compiles with the fixed translation units, and obeys every scheduler and snapshot requirement below. The surrounding program continues to own trace loading, initial-tree construction, transaction dispatch, query dispatch, timing, and reporting. Its calls into `treap::scheduler`, and its use of `treap::node_ptr`, node fields, key multiplicities, subtree augmentation, and the two `search_*` entry points, remain fixed. No separately loaded component or alternate executable is part of the submission. All observable versions, results, termination behavior, and stdout consumed by the judge must continue to satisfy the fixed program contract for every evaluated trace and thread count.

## Workload
The evaluator generates six files named `case-0.data` through `case-5.data` in a temporary evaluation directory. Each is deterministic for its suite seed. A case begins with `n = 80,000 + (seed mod 10,000)` distinct even keys `2, 4, ..., 2n`, shuffled before initial-tree construction. It then contains `m = 450,000 + (seed mod 90,000)` transactions. Generated insert keys lie in `[1, 4n]`; duplicates are allowed and increase a key's multiplicity. A delete selects a currently present key and removes one occurrence. A range query selects `left` from `[1, 4n - 128]` and uses the inclusive endpoints `[left, left + 127]`.

Profiles are selected by case position and by a uniform integer draw from 0 through 99 for each transaction. The table gives the branch probabilities while the live multiset is nonempty; if it were empty, a selected delete would instead become a query.

| Profile | Insert | Delete | Range query |
| --- | ---: | ---: | ---: |
| balanced | 42% | 25% | 33% |
| read-heavy | 18% | 10% | 72% |
| update-heavy | 60% | 28% | 12% |
| hotspot | 55% | 35% | 10% |

Both roles use six cases in update-heavy, hotspot, balanced, read-heavy, hotspot, and update-heavy order, with separately derived development and final seeds. Hotspot cases direct 98% of generated update/query keys to a narrow interval. Each performance trace uses `(server threads, query clients)` configurations `(4,1)`, `(5,1)`, `(5,2)`, and `(6,2)`, always with batch-size argument 1000. Every case/configuration pair is timed five times for the candidate, the private optimized reference, and the fixed serial naive baseline, using a rotating execution order; the evaluator also records the peak resident memory of each of those timed processes. In each of the five repetitions, the candidate and the private reference also run the fixed historical range-query benchmark described below on the same trace and server-thread count, in alternating order. Private correctness traces use separate derived seeds but the same `n`, `m`, profiles, server-thread/query-client configurations, and block size 1000.

These are representative generated workloads, not fixed records to recognize. The final suite has undisclosed seeds and transaction order and is distinct from the development-feedback suite. Only the stated generated key domain, sizes, profiles, version operations, thread counts, client count, and block sizes are in scope.

## Interface
The fixed candidate benchmark invocation is:

```
BUILD_DIR/main-candidate contreap DATA_DIR/case-I.data -threads THREADS -clients CLIENTS -batchsize 1000
```

`BUILD_DIR` and `DATA_DIR` are evaluator-created temporary directories, and `I` is the case index from 0 through 5. All three flags and their values are passed explicitly.

Before timing, the evaluator compiles the fixed correctness harness with the candidate headers and invokes it once for every case and configuration as follows:

```
BUILD_DIR/check-candidate CHECK_SEED INITIAL_COUNT TRANSACTION_COUNT PROFILE THREADS 1000 CLIENTS
```

The seven arguments after the executable are, in order, a separately derived unsigned 64-bit seed, `n`, `m`, one of the four profile names, the server-thread count, block size 1000, and the query-client count. The harness constructs its operation stream from those values; it does not read `case.data` or produce a scored throughput.

The workload file uses the judge machine's native C++ ABI (x86-64 and therefore little-endian). It has this byte layout, with no file-level padding other than the padding inside each transaction record:

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

For insert and delete, `arg0` is the key and generated traces repeat that key in `arg1`. For query, `arg0` and `arg1` are the inclusive left and right endpoints. For example, this decoded record sequence is a complete file with `n=2` and `m=3`:

```
header:       initial_count=2, transaction_count=3
initial_keys: [4, 9]
record 1:     type=1 (insert), arg0=7, arg1=7
record 2:     type=2 (delete), arg0=4, arg1=4
record 3:     type=0 (query),  arg0=5, arg1=10
```

Version 0 is `{4,9}`; versions 1, 2, and 3 are `{4,7,9}`, `{7,9}`, and `{7,9}`. The version-3 query's inclusive sum of multiplicities over keys 5 through 10 is 2.

The fixed adapter uses the following scheduler contract:

- `treap::scheduler(size_t num_threads, size_t num_tasks, size_t block_size, node_ptr root0)` initializes version 0 from `root0` and is told the exact number of transactions that will be submitted.
- `insert_elem(key)` submits the next version and adds one occurrence of `key`.
- `delete_elem(key)` submits the next version and removes one occurrence of `key`; evaluated deletes always have a matching live occurrence.
- `nop()` submits the next version without changing the multiset and returns that version number. The adapter uses this version for the corresponding query.
- `get_snapshot(version)` returns the complete immutable root for that exact version. It may be called by a query-client thread while submissions or processing are still in progress and must not expose an incomplete tree.
- `process()` completes all submitted work and must not return until every version is complete.
- `last_version()` returns the last submitted version number.

The benchmark writes no result file. It prints the loaded counts, elapsed seconds, and throughput to stdout. For the example above, a hypothetical three-thread run taking 0.002 seconds has this output shape:

```
# Loaded records:	2
# Loaded transactions:	3
# Time Used (S)
contreap	case.data	3	0.002
# Transaction throughput (KTPS)
contreap	case.data	3	1.5
```

The final field is `m / elapsed_seconds / 1000` KTPS. The evaluator requires normal exit and parses a finite, positive value from the final tab-separated `contreap`, workload path, thread count, and throughput line. Timed query values are not printed, so correctness is established on behaviorally equivalent hidden traces by the concurrent snapshot and range-sum checks described below; query values are not a separate score term. The evaluator, not the program, measures each benchmark process's peak resident memory: it reaps the process with `wait4` and reads the kernel's `ru_maxrss` for exactly that process.

The fixed historical range-query benchmark is a judge-owned program compiled from the same three editable headers and the fixed tree, construction and query code. It is invoked as:

```
BUILD_DIR/history-candidate DATA_DIR/case-I.data THREADS 1000 100000 QUERY_SEED
```

It loads the same trace, builds version 0, submits every transaction in order through `insert_elem`, `delete_elem`, and `nop` of a scheduler constructed with `THREADS` server threads and block size 1000, and calls `process()`. It then requires `last_version() == m` and, from one thread, answers 100,000 inclusive range-sum queries with `get_snapshot(version)` and the fixed `range_estimate`. Each query draws, from a private per-pair seed, a version uniformly from `0..m`, an inclusive width `w` uniformly from `0..floor(4n/10)`, and a left endpoint uniformly from `1..4n-w`, so a query covers up to 10% of the key domain. Only the query loop is timed with a steady clock; the reported value is the elapsed time divided by 100,000, in microseconds. The program also prints an order-dependent checksum of all answers. The evaluator runs the same program once with the fixed serial scaffold to obtain the expected checksum for the same trace and seed; every reference run must reproduce it as well.

## Validity
The submitted artifact must be a nonempty textual unified diff no larger than 262,144 bytes, applicable by `git apply`, and limited to at most the three listed files. A patch with no `+++ b/<path>` entry is rejected, including an empty patch. Absolute paths, parent-directory escapes, binary patches, new or deleted files, and renames are rejected. After applying it to a pristine tree, the evaluator concatenates the editable sources and rejects source containing any of these literal inspection facilities or markers: `/proc`, `/sys`, `FRONTIER_`, `getenv(`, `secure_getenv(`, `readlink(`, `getpid(`, `getppid(`, `clock_gettime(`, `std::filesystem`, `std::ifstream`, `std::ofstream`, `std::chrono`, `steady_clock`, `high_resolution_clock`, `rdtsc`, `rdtscp`, `program_invocation_name`, `__progname`, `getcwd(`, or `realpath(`.

The candidate benchmark, correctness, and historical-query executables are compiled with GCC as C++20 using `-O3 -DNDEBUG -march=native -msse4 -maes`, pthreads, and mimalloc. The correctness executable runs for every suite case and timed server-thread/query-client configuration on a behaviorally equivalent trace, always with block size 1000. After all operations are submitted but before `process()` joins the workers, the configured number of query-client threads concurrently obtain version 0, the final version, and checkpoints at every `max(1, floor(m/16))` versions. At every checked root they verify strict BST ordering, exact multiplicities and total multiplicity, `node.ver <= snapshot_version`, and an independently computed inclusive range sum. The harness then calls `process()` and requires normal completion.

A rejected patch, forbidden source, build failure, nonzero exit, crash, deadlock, malformed or nonpositive throughput, or any snapshot mismatch scores zero before performance is considered. Each individual compilation has a 300-second timeout. Each correctness or benchmark process attempt has a 30-second timeout; only a timeout is retried once, and a second timeout scores zero. The full evaluation has a 1,800-second limit. These rules apply to the candidate. If the private reference or the fixed serial scaffold fails to build, run, pass the same correctness harness, or answer every historical query correctly, the verification fails as a judging error without a score; such a failure is never attributed to the candidate and its measurements are never used as a comparison bar.

## Scoring
The secondary continuous objective is to maximize a continuous score derived independently from the fixed serial naive baseline. The private optimized reference is not used in this score.

For suite case `i`, configuration `q` among `(4,1)`, `(5,1)`, `(5,2)`, and `(6,2)`, and repetition `r in {1,2,3,4,5}`, let `C[i,q,r]`, `R[i,q,r]`, and `N[i,q,r]` be the positive candidate, private-reference, and naive throughputs in KTPS. All three implementations run once in each repetition using a rotating order. Define

```
G[i,q] = median(C[i,q,1]/N[i,q,1], ..., C[i,q,5]/N[i,q,5])
Q      = exp((1/24) * sum_i sum_q ln(G[i,q]))
```

The reported score is

```
score = 0                                      if Q <= 1
score = 100 * ln(Q) / (ln(10) + ln(Q))         if Q > 1.
```

Thus the naive baseline scores zero, every increase in `Q` above one increases the score, and 100 is approached only asymptotically. There is no upper clamp or separate read-heavy cap. The fixed naive baseline and all workload aggregation are independent of the optimized reference. The evaluator retains all five raw candidate, reference, and naive throughputs and the paired ratios for every one of the 24 groups for operator recomputation.

A separate binary verdict compares the candidate with the optimized reference independently in every case/configuration group on three metrics, each with its own strict rule and no tolerance:

- `case_throughput_ktps`: `median(C[i,q,*]) > median(R[i,q,*])` over the five throughput samples (paper Section 6.6 mixed read/write throughput).
- `peak_memory_mib`: the median of the candidate's five peak resident memory samples, in MiB, of the same timed benchmark processes must be strictly below the reference's median (paper Section 6.5, peak memory with every version retained).
- `historical_query_us`: every one of the candidate's five historical-query checksums must equal the fixed serial scaffold's checksum, and the median of the candidate's five average historical range-query times, in microseconds, must be strictly below the reference's median (paper Section 6.2, average time of 10^5 historical range statistics). A wrong historical answer fails this metric for that group.

All 24 groups must clear all three metrics; equality, or one group that is slower, uses more memory, or answers historical queries more slowly, fails the verdict. A faster throughput cannot compensate for a memory or query-time regression. Correctness and validity conditions must also pass. The continuous score is unchanged, uses only throughput, and is still returned when this separate verdict fails.

## Calibration
The shipped serial functional scaffold is the fixed valid naive baseline and therefore anchors the score at zero. The optimized upstream implementation is retained only for the separate strict 24-way verdict. Native timing still varies with host contention, so small score changes should be reproduced before being treated as an improvement.

## Feedback and grading
The judge returns a score after every submission. Development feedback also gives only the aggregate number and percentage of workload pairs that passed and failed each of the three stricter comparisons, as `beat_spec_progress` entries `case_throughput_ktps`, `peak_memory_mib`, and `historical_query_us`, each with `passed`, `failed`, `total`, `pass_percent`, and `fail_percent`. Individual candidate throughput, memory and query-time rows, workload identities, the reference, margins, and normalization measurements remain operator-only. The reported score remains a secondary optimization objective and is returned even when the separate criterion is not cleared.

The submission budget is in `/frontier/task/README.md`. **The graded suite is not the suite you receive feedback on**, and neither its membership seeds nor its transaction order are disclosed. Development and final grading each use six separately seeded cases drawn by the same profile and size procedure, so they carry the same distribution but share no item; development measurements do not enumerate the final workload.

## Resources
The evaluation container provides 8 vCPUs, 16,384 MiB of memory, and 8,192 MiB of storage. It runs Ubuntu 24.04 on x86-64 with GCC, C++20, pthreads, mimalloc, and AES-NI. The configured build timeout is 300 seconds, every correctness, throughput or historical-query benchmark attempt is limited to 30 seconds with one retry after a timeout, and the full evaluation timeout is 1,800 seconds.
