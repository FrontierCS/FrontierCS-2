# GPU K-Means Kernel Optimization

Make one exact Lloyd k-means step on one H100 as fast as possible. The score rewards speedup of a
two-step Lloyd loop over the naive starter; the returned labels and centroids must stay valid.


## Task

`kmeanslib` is a small Python/Triton k-means package whose entry point is one Lloyd iteration.
The shipped implementation is correct but deliberately unoptimized. Rewrite its internals (new
Triton kernels, a fused assignment and update pass, better memory traffic) while keeping the
public contract. The task follows Flash-KMeans (fast exact k-means on GPUs).

## What you may change

- Python files below `kmeanslib/` (`kmeanslib/**/*.py`), including new Python/Triton modules and
  local regression tests such as `test_regress.py` in `kmeanslib/`; at most 80 changed files.
- Files named `refkmeans.py` or `conftest.py` are reserved at every depth and cannot be changed or
  added. No file may be deleted or renamed. Everything outside `kmeanslib/` is fixed.
- Added lines must not import `flashlib`, `cuml`, `cudf`, `cupy`, `faiss`, `sklearn`, `scikit`,
  `cutlass` or `cuvs`. A violation is rejected as `content_policy`.
- Do not spawn processes, access the network or the judge, or alter timing.

## Interface

```python
labels, new_centroids = kmeanslib.step(x, centroids)
```

- `x`: `(N, D)` CUDA bfloat16 points; `centroids`: `(K, D)` CUDA bfloat16.
- Return `(N,)` int32 or int64 labels in `[0, K)`, the nearest centroid of every point, and the
  `(K, D)` finite means of the points assigned to each centroid. An empty cluster keeps its
  previous centroid. Results must be deterministic functions of the inputs.
- The judge owns the data, the initial centroids and the loop: it calls `step` exactly twice per
  run, feeding the returned centroids into the second call.

## Correctness

- Labels of the wrong shape, dtype or range, or centroids of the wrong shape or with non-finite
  values, an import error, a crash, or a workload process running longer than 900 s fail the
  evaluation (no score).
- Quality is not a score gate; it enters the reference comparison (see Scoring): labeled inertia
  of the final labels and centroids, and the label mismatch fraction, the share of final labels
  whose squared distance to the centroids supplied to the last call exceeds the exact minimum by
  more than `1e-6` relative plus `1e-6` absolute.

## Workload

Inputs are planted bfloat16 clusters (Gaussian noise around random centers); the initial
centroids are `K` distinct points. Fresh data is generated for every warmup and timed run.

| ID | N | D | K | Lloyd steps |
| --- | ---: | ---: | ---: | ---: |
| w0 | 200,000 | 64 | 512 | 2 |
| w1 | 300,000 | 128 | 1,024 | 2 |
| w2 | 500,000 | 128 | 1,024 | 2 |
| w3 | 150,000 | 512 | 512 | 2 |

- Train runs these four. Final runs them with the same seeds plus two hidden shapes with
  independent seeds; the score and the reference comparison both cover all six.
- Each implementation gets two warmup and three timed runs of the two-step loop per workload; the
  synchronized loop is timed. Input generation and quality checks are not.
- Implement a general step; do not recognize a workload or cache a prior result.

## Scoring

```text
r_w   = median naive loop latency / median candidate loop latency   (3 timed runs each)
S     = geometric mean of r_w over the suite (4 in train, 6 in final)
score = 0                                   if S <= 1
score = 100 * log(S) / (log(10) + log(S))   otherwise
```

Anchors: the unchanged starter scores about 0 (`S ≈ 1`); `S = 10` scores 50, `S = 100` about 66.7;
there is no upper clip. The score does not use the optimized reference.

**Beats the reference** when all three hold on every workload of the suite (per metric, the worst
of the three timed runs for the quality metrics):
- median loop latency is strictly lower than the optimized reference's;
- labeled inertia is at most `1.05 ×` the reference's;
- label mismatch fraction is at most the reference's `+ 0.002`.

Averages cannot compensate; a failed comparison does not change the score.

## Running locally

On a GPU runner (see `/frontier/task/README.md`): `bash frontier-build.sh` checks imports;
`bash frontier-bench.sh` times a small two-step loop (32,768 × 64 points, 256 centroids,
ungraded).

## Environment

PyTorch 2.6.0 with CUDA 12.4 and Triton, on one NVIDIA H100.
