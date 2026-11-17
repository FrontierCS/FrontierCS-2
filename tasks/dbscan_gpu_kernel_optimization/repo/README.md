# GPU DBSCAN Kernel Optimization

Make Euclidean DBSCAN clustering on one H100 as fast as possible while producing the same
clustering as the naive starter. The score rewards speedup over the frozen naive implementation;
a result counts only if every timed clustering agrees with the naive labels.

## Task

`dbscanlib` is a small Python/Triton DBSCAN package. The shipped implementation is correct but
naive: a chunked O(N²) neighbour scan plus label propagation over the core graph. Rewrite its
internals (for example grid or bucketed radius search, fused neighbour and connected-components
kernels) to make `dbscanlib.dbscan` fast. The task follows the FlashLib classical-ML operator
kernels.

## What you may change

- Python files below `dbscanlib/` (`dbscanlib/**/*.py`), including new modules such as a local
  regression test `test_regress.py` in `dbscanlib/`; at most 80 changed files.
- Files named `refdbscan.py` or `conftest.py` are reserved at every depth and cannot be changed or
  added. No file may be deleted or renamed. Everything outside `dbscanlib/` is fixed.
- Added lines must not import `flashlib`, `cuml`, `cudf`, `cupy`, `faiss`, `sklearn`, `scikit`,
  `cutlass` or `cuvs`. A violation is rejected as `content_policy`.
- Do not spawn processes, access the network, the environment or the judge, or alter timing.

## Interface

```python
labels = dbscanlib.dbscan(x, eps, min_samples, max_neighbors=32)
```

- `x`: `(N, D)` float32 CUDA tensor whose coordinates carry bfloat16 precision; `eps` float;
  `min_samples` int. The judge calls `dbscan(x, eps, min_samples)`.
- Return an `(N,)` int64 CUDA tensor on the input device: nonnegative cluster labels, `-1` for
  noise.
- Euclidean DBSCAN: the eps-neighbour count includes the point itself (distance `<= eps`); core
  points have at least `min_samples` neighbours; connected core points form clusters; a border
  point joins the smallest core-component label among its eps-neighbours; any other point is
  noise. Cluster numbers may be permuted; `-1` keeps its meaning.

Example: for the seven points `(0,0) (.05,0) (0,.05) (3,3) (3.05,3) (3,3.05) (10,10)`, `eps=0.2`,
`min_samples=3`, the first three share one label, the next three share another, and the last is
`-1` (the check in `frontier-bench.sh`).

## Correctness

- Every timed result must reach adjusted Rand index (ARI) at least `0.99` against the frozen naive
  labels on the same input (ARI treats noise as one ordinary group); otherwise the score is 0.
- A result that is not an `(N,)` int64 tensor on the input device, has a label below `-1`, or
  whose latency is not finite and positive, an import error, a crash, or a workload process
  running past the per-evaluation timeout fails the evaluation (no score).

## Workload

One workload is a DBSCAN call on `N` points in `D` dimensions: interleaving half-moon pairs
embedded in random 2-D subspaces with thin ambient noise, rounded to bfloat16 precision. Fresh
data is generated for every warmup and timed repetition.

| ID | N | D | eps | min_samples | n_centers |
| --- | ---: | ---: | ---: | ---: | ---: |
| w0 | 20000 | 8 | 0.90 | 5 | 4 |
| w1 | 20000 | 32 | 1.15 | 5 | 6 |

- Train runs `w0` and `w1`. Final runs `w0` and `w1` with the same public seeds, two hidden shapes
  with independent hidden seeds (these four make up `S`), and two further hidden shapes whose
  speedups do not enter `S` but which count for beating the reference: one with 200,000 points,
  and one with `min_samples=20` where noise and border points are present. Their ARI gate and
  validity checks still apply (an ARI below `0.99` there makes the score 0). Hidden shapes and seeds are withheld.
- Each implementation gets two warmups and three timed repetitions per workload. One synchronized
  `dbscan` call is timed; input generation and label checking are not.
- Implement the general operation; do not recognize inputs, cache prior outputs or inspect the
  judge.

## Scoring

```text
r_w   = median over the 3 timed repetitions i of naive_latency_i / candidate_latency_i
S     = geometric mean of r_w over the scored workloads (2 in train, 4 in final)
score = 0                                   if any timed ARI < 0.99 or S <= 1
score = 100 * log(S) / (log(10) + log(S))   otherwise
```

Anchors: the unchanged starter scores about 0 (`S ≈ 1`); `S = 10` scores 50; there is no upper clip.
The score does not use the optimized reference.

**Beats the reference** when all three hold on every selected workload (2 in train, 6 in final),
with no added tolerance:
- median call latency is strictly lower than the optimized reference's;
- minimum timed ARI is at least the reference's minimum (and every ARI is at least `0.99`);
- maximum timed noise-mask mismatch fraction (share of points whose `-1` status differs from the
  naive labels) is at most the reference's; if the reference has none, neither may the candidate.

A fast candidate can score while failing a comparison; the score never compensates.

## Running locally

On a GPU runner (see `/frontier/task/README.md`): `bash frontier-build.sh` compiles and imports
`dbscanlib`; `bash frontier-bench.sh` runs a small ungraded cluster-equivalence and noise check.

## Environment

PyTorch 2.6.0 with CUDA 12.4 and Triton, on one NVIDIA H100.
