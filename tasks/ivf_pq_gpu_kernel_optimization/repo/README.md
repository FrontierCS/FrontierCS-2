# GPU IVF-PQ Search Kernel Optimization

Make IVF-PQ approximate nearest-neighbour search on one H100 as fast as possible. The score
rewards speedup over the frozen naive search; every timed call must return valid, consistent
results that agree with the naive search's top-k.

## Task

`ivfpqlib` is a Python/Triton residual IVF-PQ package (FAISS-style IVFADC): a coarse quantizer
picks `nprobe` inverted lists per query and product-quantization codes give approximate distances.
Optimize the search; the index is built by the frozen naive builder. The task follows the
FlashLib classical-ML operator kernels.

## What you may change

- Python files below `ivfpqlib/`, including new Triton modules and local
  regression tests such as `test_regress.py` in `ivfpqlib/`; at most 80 changed files. An empty patch
  is valid and evaluates the unchanged starter.
- Files named `refivfpq.py` or `conftest.py` are reserved at every depth and cannot be changed or
  added. No file may be deleted or renamed. Everything outside `ivfpqlib/` (including
  `benchmark/`) is fixed.
- Added lines must not import `flashlib`, `cuml`, `cudf`, `cupy`, `faiss`, `sklearn`, `scikit`,
  `cutlass` or `cuvs`. A violation is rejected as `content_policy`.
- Do not spawn processes, access the network, the environment or the judge, or alter timing.

## Interface

```python
vals, ids = ivfpqlib.ivf_pq_search(index, Q, k, *, nprobe=None)
```

- `index` is the judge's frozen copy of `IvfPqIndex`, with the fields documented in
  `ivfpqlib/index.py`; it is not that class itself, so do not rely on `isinstance` or on methods
  added to `ivfpqlib/index.py`. `Q` is a float32 CUDA query tensor `(nq, D)`, `k` an int.
- Return float32 approximate squared-L2 distances and int64 original database IDs, both
  `(nq, k)`, on the input GPU. Valid IDs are in range and do not repeat within a query; an unused
  slot is `id = -1, distance = +inf` (padding never counts as a hit).
- Each call receives a fresh index object and freshly cloned tensors with identical contents. Do
  any layout conversion or preprocessing inside the call; do not amortize it into warmup or cache
  answers. Keep the signature and the deterministic input/output meaning.

## Correctness

Every timed call must satisfy all of these, or the score is 0:
- the output shape, dtype, device, ID range, no-duplicate and padding rules above;
- valid distances are finite and match the frozen float32 residual ADC calculation on the same
  inputs: `abs(returned - oracle) <= 1e-6 * abs(oracle)` (a zero oracle needs exactly zero);
- iso-result recall at least `0.99`: valid returned neighbours also in the full-nprobe naive ADC
  top-k, divided by `nq * k`.

An import error, a crash, or a workload process running longer than 3,600 s fails the evaluation
(no score).

## Workload

The data is the real SIFT-128 ANN-benchmarks base and query pool (`/data/sift.hdf5`). The frozen
naive builder constructs the index outside timing; centroids, PQ codebooks and queries are rounded
to bf16 precision and kept as float32. Each iteration draws a fresh deterministic query subset.

| Workload | nlist | m | k | queries | nprobe |
| --- | ---: | ---: | ---: | ---: | ---: |
| sift_q4k_np32 | 8192 | 16 | 100 | 4096 | 32 |
| sift_q4k_np64 | 8192 | 16 | 100 | 4096 | 64 |
| sift_q2k_np64 | 8192 | 16 | 100 | 2048 | 64 |
| sift_q2k_np128 | 8192 | 16 | 100 | 2048 | 128 |

- Train runs these four. Final runs them again with the same seeds, two hidden shapes with
  independent seeds (these six make up `S`), and two further hidden shapes whose speedups do not
  enter `S` but which count for beating the reference: up to the full 10,000-query test set, and
  k up to 1024. The correctness checks apply to them too (a failure there makes the score 0).
- Each workload gets two warmups and seven timed calls; each runs the naive search and then the
  implementation under test on the same index and queries. Only the search call is timed
  (synchronized GPU wall clock).
- Implement the general search; do not special-case workloads.

## Scoring

```text
r_w   = median over the 7 timed calls i of naive_latency_i / candidate_latency_i
S     = geometric mean of r_w over the scored workloads (4 in train, 6 in final)
score = 0                                   if any correctness check fails or S <= 1
score = 100 * log(S) / (log(10) + log(S))   otherwise
```

Anchors: the unchanged starter scores about 0 (`S ≈ 1`); `S = 10` scores 50; there is no upper clip.
The score does not use the optimized reference.

Nearest-neighbour recall@k (outside timing) is the fraction of queries whose exact nearest
neighbour in the uncompressed SIFT base (float32 brute force on the actual queries) is among the
`k` returned vectors; a vector at the exact nearest distance also counts.

**Beats the reference** when all three hold on every selected workload (4 in train, 8 in final):
- median search latency is strictly lower than the optimized reference's;
- worst timed iso-result recall is at least the reference's;
- worst timed nearest-neighbour recall@k is at least the reference's.

There is no added tolerance; equality passes for the two recalls. Failing a comparison does not
zero a valid score.

## Running locally

On a GPU runner (see `/frontier/task/README.md`): `bash frontier-build.sh` checks imports;
`bash frontier-bench.sh [workload_id]` runs one public workload (default: the first). It can be
slow: the naive search and the frozen index build are expensive.

## Environment

PyTorch 2.6.0 with CUDA 12.4 and Triton, h5py 3.14.0, on one NVIDIA H100.
