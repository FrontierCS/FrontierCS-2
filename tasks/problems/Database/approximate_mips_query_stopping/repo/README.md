# Approximate-MIPS Query Stopping

Design the per-query early-stopping rule of FARGO, an approximate maximum-inner-product search
system: decide after each probed bucket whether to stop searching the current partition. The
score rewards result quality together with lower query time and fewer verified candidates than
the shipped fixed-budget policy. **First priority: beat the reference policy on every workload and
every metric (see Scoring); the continuous score is secondary** and never compensates for a
failed comparison.

## Task

FARGO (Zheng et al., PVLDB 2023) ranks collection vectors by dot product with a query. It
partitions vectors by norm, indexes each partition in locality-sensitive hash tables, and probes
buckets in one global order across the tables. The authors' per-query termination condition and
function-table loader were removed; the shipped `should_stop` stops at a fixed candidate budget.
Supply the stopping decision.

## What you may change

- `src/alg.cpp` only, a single file: the policy `should_stop(const SearchState&)` and any
  supporting standard-library C++ declarations and definitions inside it. Its signature stays.
- Everything else is fixed: the `SearchState` layout, call site, query and result types, heap
  capacity, partition order, index, hash tables, bucket frontier, candidate verification, timers,
  metric printer, build files and harness.
- The whole of `src/alg.cpp`, lowercased, must not contain the string `fargo_metric`.
- `should_stop` must return normally on every call and must not terminate or block the process,
  cast away constness to change the search state, alter measurement, inspect the process command
  line, or identify the train/final role. It must not recognize particular queries, suites, the
  dataset or the hash seed (for example by matching vector values, scores or call counts) to
  change its behaviour.

## Interface

```cpp
bool should_stop(const SearchState& search_state);
```

For a partition on the indexed path, the query loop first consumes that partition's matching
bucket in each hash table, then calls `should_stop` once after each further globally ordered
bucket, when all new candidates in it have exact inner products. `true` ends probing of the
partition after that bucket; `false` requests the next bucket.

```text
top_scores             pointer to the live result-heap inner-product array (heap order)
top_score_count        number of float entries addressable through top_scores
kth_score              heap threshold tracked by the query loop (see below)
frontier_score         ordering score of the non-initial bucket just consumed
probes_performed       non-initial frontier buckets consumed in this partition
candidates_verified    distinct candidates verified so far in this partition
candidate_budget       total number of vectors in this partition
remaining_frontier     5 * 2^12 - probes_performed
approximation_ratio    current value in {0.6, 0.8, 0.9}
maximum_inner_product  query norm times the partition's maximum vector norm
hash_tables            5 for this workload
policy_resource        path to the judge-provided binary function table
```

`top_scores` is valid during the call. `kth_score` starts at `-1`; if the heap was full on entry
to the partition it starts at the heap minimum, otherwise it becomes the heap minimum when a later
candidate replaces that minimum, so it can still be `-1` when `top_score_count` has just reached
capacity. Example call: `top_scores={0.77, 0.91, 0.83}`, `top_score_count=3`, `kth_score=0.77`,
`frontier_score=0.42`, `probes_performed=18`, `candidates_verified=231`, `candidate_budget=6000`,
`remaining_frontier=20462`, `approximation_ratio=0.8`, `maximum_inner_product=1.14`,
`hash_tables=5`; returning `true` ends the partition after 18 probes and 231 candidates.

`policy_resource` (7,547,112 bytes, the upstream function table) is little-endian:
`float32 theta_min, theta_max, score_min, score_max; int32 bytes_per_value, row_count,
column_count; float32 values[column_count][row_count]` (columns consecutive). Its header is
`theta_min=0.0, theta_max=1.5707964, score_min=0.0, score_max=12.0, bytes_per_value=4,
row_count=1571, column_count=1201`, and `values[0][0..7]` starts `1.000000, 0.996187, 0.992387`.
Nothing is required to be read from it.

The judge builds with `make -j2` and runs `./fg <data-base-path> <ground-truth-path>
<policy-resource-path> <suite-id> 150` (the data path omits the `_new` suffix the loader appends).
For each `c` and capacity the program prints one record,
`FARGO_METRIC <c> <capacity> <mean_query_ms> <recall_at_50> <mean_result_ratio> <kth_ratio> <mean_candidates>`,
e.g. `FARGO_METRIC 0.8 200 1.750000 0.940000 0.990000 0.980000 3400.000000`. The data is the
Audio vector file (`uint32 bytes_per_value, vector_count, dimension; float32 vectors[][]`) and
its exact top-100 ground truth (`int32 query_count, neighbors_per_query; int32 ids[][]; float32
inner_products[][]`); both are judge-only.

## Correctness

The build (120 seconds) must succeed and each workload process must exit successfully within 180
seconds and print exactly one record per `c ∈ {0.6, 0.8, 0.9}` and capacity, ordered by `c`
then capacity. Every field must be finite, `mean_query_ms > 0`, `mean_candidates >= 0`,
`recall_at_50` in `[0,1]` and `kth_ratio` in `[0,1.01]`. A rejected patch, build failure or
timeout, crash, missing, extra, malformed, misordered, non-finite or out-of-range record scores
zero.

## Workload

The Audio dataset of the authors' artifact: 54,387 vectors of dimension 192 and exact top-100
neighbors for the first 300 vectors, which are the queries. They form two fixed disjoint suites of
150 queries: train runs public suite 0; final freshly re-runs suite 0 and adds hidden suite 1,
pooling all 300 queries per workload. A workload is one pair of approximation ratio
`c ∈ {0.6, 0.8, 0.9}` and result-heap capacity `100, 200, 400, 800` (12 workloads), with `k = 50`,
five hash tables and 12 hash bits each. Final also measures further capacities not shown in train,
scored and compared the same way. Each suite process builds an index with hash seed `314159`
and runs its queries once per workload; the candidate, the fixed-budget starter policy and the
reference policy run separately on the same suite.

## Scoring

```text
Q_j     = sqrt(clip(R_j, 0, 1) * clip(K_j, 0, 1))
t_j     = max(0.25, T_j^fixed / max(10^-9, T_j))
w_j     = max(0.25, C_j^fixed / max(1, C_j))
E_j     = t_j^0.25 * w_j^0.75
point_j = 100 * Q_j * E_j / (1 + E_j)
score   = clip(mean over workloads j of point_j, 0, 100)
```

For a query returning `n` results let `m = min(n, 50)`. `R_j` is the mean of
|first `m` returned ids ∩ exact first `m` ids| / 50; `K_j` the mean of (returned inner product at
rank `m`) / (exact inner product at rank `m`), 0 if `m = 0`; `T_j` the mean query milliseconds
and `C_j` the mean distinct verified candidates; `T_j^fixed`, `C_j^fixed` are the shipped
fixed-budget policy's values on the same queries. Matching the starter gives `point_j = 50·Q_j`;
the starter scores about 49.15 on the train suite, and repeated measurements vary by under 2
points.

**Beats the reference** when, on every workload including the final-only capacities, all four
hold against the reference policy on the same queries: `mean_query_time` strictly lower,
`recall_at_50` at least as high, `last_result_inner_product_ratio` at least as high, and
`verified_candidate_count` at most as high. Tolerance: none; no compensation across workloads or
metrics.

## Running locally

```sh
cd /workspace/repo && make -j2
```

## Environment

4 vCPUs, CPU-only Ubuntu 24.04 with the shipped C++11 build. The program needs AVX2 and
AVX-512F/DQ/BW.
