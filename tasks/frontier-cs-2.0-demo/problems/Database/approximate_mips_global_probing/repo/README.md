# FARGO paper-aligned query core

Implement an efficient approximate maximum inner product search (MIPS) query core in
`src/alg.cpp`: generate a global multi-probing bucket sequence and decide when to stop verifying.
Higher quality and fewer verified candidates and lower latency score higher.

## Task

The target is the query algorithm of FARGO: Fast Maximum Inner Product Search via Global
Multi-Probing (PVLDB 16(5) 2023, Sections 3.2.2–3.2.3 and 4.2, Equations 11–12 and 16,
Algorithm 1). The starter enumerates buckets in numeric order and never stops adaptively. The
harness supplies the index (random XBOX transformation, norm partitioning), input vectors, result
heap, exact inner products, duplicate suppression and measurements; you supply a query-local
probing frontier across all hash tables and the termination rule. The reference follows the
paper's squared quantization distance and adaptive termination formula (see Notes); the upstream FARGO
code uses different formulas, and the paper is the specification.

## What you may change

- Only `src/alg.cpp`; the patch may change exactly that one file.
- `src/alg.cpp` must not contain (case-insensitively, anywhere) `fargo_metric`, `preprocess.h`,
  `mf_alsh.h` or `performance.h`; such a patch is rejected.
- Do not inspect process roles, hidden files or workload identities, and do not generate metric
  output.

## Interface

Implement the factory in `src/query_core.h` (`std::unique_ptr<QueryCore> make_query_core()`); a
fresh `QueryCore` is created per query.

- `reset(L,K,p)`: the query's normalized Gaussian projections, table-major (`p[t*K+j]`),
  `1 <= K <= 20`.
- `next(out)`: emit a `(table,key,score)` bucket. The key's highest bit is projection zero; a
  positive projection has hash bit one, zero and negative have bit zero. `score` must equal the sum
  of **squared** projections at disagreeing bits, within `1e-9 * (1 + QD)`. Any order is allowed, but a fully consumed
  sequence must contain every one of the `L*2^K` buckets exactly once.
- `stop(state)`: called **before** consuming the next bucket, with the current kth inner product,
  result count, next bucket's QD, approximation ratio, partition maximum possible inner product,
  table count and calibration path (`SearchState`). Never stop with fewer than k results or a
  nonpositive kth result.

The harness replays the query-global sequence for each norm partition, verifies each candidate
once and keeps exactly k results; a deterministic norm bound can end the partition loop, and small
partitions are scanned. State is query-local except immutable calibration caching.

**Calibration table** (`calibration_path`, read-only, little-endian): 8 bytes `FARGOSQ1`; int32 K,
rows, columns; float64 theta step and score step; then `rows*columns` float64 CDF entries,
row-major. Theta starts at 0 and covers pi/2; QD starts at 0; entry (r,j) is at theta
`r*theta_step`, QD `j*score_step`, with nonzero squared contributions rounded up before the K-fold
convolution. Rounding theta up and QD down gives conservative lookups.

## Correctness

Before measurement `src/alg.cpp` alone is compiled with a conformance harness
(`g++ -std=c++11 -O2`, no OpenMP, so `src/alg.cpp` must not need symbols from the other sources)
and run (108 cases: K=1..9, L∈{1,2,5}, zero/one/random projections, reset reuse); it checks
ranges, completeness, no duplicates, scores and termination, must finish within 15 s, and its
stdout must be exactly the harness's PASS line, so `src/alg.cpp` must print nothing to stdout.
Its `stop()` calls use the calibration path `"unused"`. A run scores 0 if the conformance check, the build (120 s) or a
workload run (180 s) fails, or if the probe stream is invalid, incomplete, duplicated or reports a
false QD, or the metrics are malformed or non-finite.

## Workload

| dataset | vectors × dim | queries | train | final |
|---|---|---|---|---|
| Audio | 54,387 × 192 | 300 fixed in-dataset queries, split into disjoint halves with different hash seeds | train half | train half + held-out half |
| MNIST | 60,000 × 784 (raw grayscale pixels) | same | train half | train half + held-out half |

Every query runs at c ∈ {0.6, 0.8, 0.9} and k ∈ {1, 10, 50, 100}: 12 operating points per
dataset/split, 24 in train, 48 in final. Performance tests use K=12, L=5. Ground truth is the
exhaustive top-100, computed outside candidate execution. Measured per point: recall, overall
ratio (mean rankwise inner-product ratio), query time and verified candidates. Query time covers,
per query, the projection, `make_query_core()`, `reset`, all probing, `stop` calls and candidate
verification, averaged in ms; dataset loading, index construction and ground truth are not
timed.

## Scoring

```text
quality    = sqrt(clip01(recall) * clip01(overall_ratio))
efficiency = (t_naive / t)^0.25 * (cost_naive / cost)^0.75
point      = 100 * quality * efficiency / (1 + efficiency)
score      = mean of point over the 24 (train) or 48 (final) operating points
```

`t` is query time in ms and `cost` the mean verified candidates per query; `naive` is the starter
run in the same evaluation (`t` floored at 1e-9 ms and `cost` at 1). No speedup clamp. Anchors:
the unmodified starter scores about 43; a perfect-quality core as fast and as cheap as the
starter gets 50 per point; the reference scores about 50; the maximum is 100.

**Beats the reference** when, on every operating point: `recall` and `overall_ratio` are each at
least the reference's, `candidates_verified` (mean per query) is no greater, and `query_time_ms` is
strictly lower; all conjunctive, with no tolerance.

## Running locally

`make && ./fg` builds and runs a public API smoke example. `bash make_submission.sh` writes the
`src/alg.cpp` diff to `solution.patch` for a local check.

## Environment

The Verifier runs on 4 CPUs.

## Notes

**Adaptive rule (paper).** Let `I_k` be the current kth inner product, `c` the approximation ratio,
`U = M_i * ||q||` the partition bound, `d` the next bucket's squared QD, `L` the table count. If
`I_k >= c * U` the partition can stop. Otherwise `theta = acos(I_k / (c * U))`,
`alpha_2 = 1 - phi(d; theta)^L` with `phi(d; theta) = Pr[QD <= d]` from the calibration table, and
stop when `alpha_2 < p_tau`; the reference uses `p_tau = 0.1`. Global probing takes the lowest-QD
unvisited bucket across all tables; a bucket's QD is the sum of squared normalized projections whose
bits differ from the query's natural key. Other valid algorithms are allowed.
