# GPU Brute-Force k-NN Kernel Optimization

Make exact brute-force k-nearest-neighbour search on one H100 as fast as possible. The score
rewards speedup over the frozen naive search; every timed call must reach the ball-recall floor
and report distances consistent with its indices.

## Task

`knnlib` is a small Python/Triton package: squared-L2 brute-force search of query vectors against
a database, returning the `k` nearest rows. The naive starter computes distances with
`cdist`/`topk`. Rewrite its internals (fused distance and top-k kernels, better memory traffic).
The task follows the FlashLib classical-ML operator kernels.

## What you may change

- Python files below `knnlib/` (`knnlib/**/*.py`), including new Python/Triton modules and local
  regression tests such as `test_regress.py` in `knnlib/`; at most 80 changed files. The patch must not
  be empty.
- Files named `refknn.py` or `conftest.py` are reserved at every depth and cannot be changed or
  added. No file may be deleted or renamed. Everything outside `knnlib/` is fixed.
- Added lines must not import `flashlib`, `cuml`, `cudf`, `cupy`, `faiss`, `sklearn`, `scikit`,
  `cutlass` or `cuvs`. A violation is rejected as `content_policy`.
- Do not spawn processes, access the network or the judge, or tamper with timing or the evaluator.

## Interface

```python
distances, indices = knnlib.knn(queries, database, k)
```

- `queries` `(Q, D)` and `database` `(M, D)`: contiguous CUDA bfloat16 tensors.
- Return `(Q, k)` float32 squared-L2 distances and `(Q, k)` int64 database row indices on the same
  GPU. Valid distances are finite and nonnegative; indices are distinct valid rows in ascending
  distance order. Only when `k > M`, trailing padding `index = -1, distance = +inf` follows all
  `M` real neighbours (every graded workload has `k <= M`).
- Work on the current inputs; do not cache previous results.

## Correctness

- Output of the wrong type, shape, dtype or device, duplicate or out-of-range indices, non-finite
  or negative distances, unsorted distances, invalid padding, an import error, a crash, or a workload process running longer than
  3,600 s fails the evaluation (no score).
- The score is 0 unless every timed call reaches ball-recall at least `0.88` and every valid
  returned distance agrees with the fp64 squared-L2 recomputation from its index and the actual
  bf16 inputs: `abs(reported - oracle) <= 1e-6 * abs(oracle)` (a zero oracle needs exactly zero).
- Ball-recall is the share of returned neighbours whose fp32 distance from the actual bf16 inputs
  is at most the fp32 distance of the naive search's k-th neighbour plus `1e-6`.

## Workload

Real image descriptors from SIFT (D = 128) and GIST (D = 960). Every warmup and timed iteration
samples fresh query and database rows and converts them to bf16.

| ID | Dataset | Queries Q | Database M | Dimensions D | k |
| --- | --- | ---: | ---: | ---: | ---: |
| sift_k10 | SIFT | 1,024 | 500,000 | 128 | 10 |
| sift_k50 | SIFT | 1,024 | 500,000 | 128 | 50 |
| sift_k100 | SIFT | 1,024 | 500,000 | 128 | 100 |
| gist_k10 | GIST | 512 | 500,000 | 960 | 10 |

- Train runs these four. Final runs them with the same seeds plus two hidden workloads with
  independent seeds (these six make up `S`), and two further hidden shapes over the full
  1,000,000-row SIFT and GIST bases (one with k up to 256) whose speedups do not enter `S` but
  which count for beating the reference. The correctness gates apply to them too (a failure there
  makes the score 0). Hidden parameters and all graded seeds are withheld.
- Each workload has two warmups and seven timed iterations; each runs the frozen naive search and
  then the implementation under test on the same sampled values. The synchronized call is timed.
  The naive baseline is the original `cdist`/`topk`; the shipped starter additionally recomputes
  selected distances in fp32 and sorts them, inside its measured call.

## Scoring

```text
r_w   = median over the 7 timed iterations i of naive_time_i / candidate_time_i
S     = geometric mean of r_w over the scored workloads (4 in train, 6 in final)
score = 0                                   if a correctness gate fails or S <= 1
score = 100 * log(S) / (log(10) + log(S))   otherwise
```

Anchors: `S = 1` scores 0; `S = 10` scores 50; there is no upper clip. The score does not use the
optimized reference.

Exact recall (outside timing) is the share of returned neighbours whose fp64 squared distance is
at most the fp64 distance of the query's exact k-th nearest database row.

**Beats the reference** when all four hold on every selected workload (4 in train, 8 in final),
with no added tolerance:
- median latency is strictly lower than the optimized reference's;
- worst timed ball-recall is at least the reference's and every iteration reaches `0.88`;
- worst timed exact recall is at least the reference's;
- every returned distance passes the consistency check above.

## Running locally

On a GPU runner (see `/frontier/task/README.md`): `bash frontier-build.sh` checks imports;
`bash frontier-bench.sh` runs a small ungraded search.

## Environment

PyTorch 2.6.0 with CUDA 12.4 and Triton, on one NVIDIA H100.
