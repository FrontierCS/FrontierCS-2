# Pairwise Distances on a GPU

Make `pairwise_sq_dists` in `kernel/` as fast as you can on one H100 while keeping its results
within tolerance.

## Task
`kernel.pairwise_sq_dists(x, y)` returns the matrix of squared Euclidean distances between the
rows of `x` and the rows of `y`. The starter broadcasts every difference into an `N x M x D`
temporary: it is exact and slow.

## What you may change
- Only Python files below `kernel/`, at most 10 of them; new modules are allowed. The scripts
  `frontier-build.sh` and `frontier-bench.sh` are fixed.
- The function must compute its result on the GPU from its inputs: caching results across
  calls, reading files, starting processes and detecting the judge's inputs are forbidden.
  PyTorch (and the Triton it ships) is available; nothing else may be installed.

## Interface
`pairwise_sq_dists(x, y)` takes `x` of shape `(N, D)` and `y` of shape `(M, D)`, both
contiguous `float32` CUDA tensors, and returns an `(N, M)` `float32` CUDA tensor `out` with
`out[i, j] = sum_k (x[i, k] - y[j, k]) ** 2`. `import kernel` must expose it.

## Correctness
Each workload's output is checked on rows the judge picks at random: every checked entry must
satisfy `|out - exact| <= 1e-3 * (1 + |exact|)`, where `exact` is computed in double precision
from the same float32 inputs. A wrong shape, a non-finite or out-of-tolerance entry, an
exception, or a run that exceeds 10 minutes scores the whole submission 0.

## Workload
One workload is one shape `(N, M, D)` with inputs drawn from a standard normal distribution.
Train measures three public shapes: `(2048, 2048, 64)`, `(4096, 1024, 128)` and
`(1024, 4096, 32)`. Final measures those three again plus two held-out shapes of the same
kind, up to `4096` rows and `D = 256`. Each workload runs three times for your code and three
times for the judge's reference, interleaved, each in a fresh process on the same GPU: three
warm-up calls, then the median of ten CUDA-synchronized calls. A workload's latency is the
smallest of its three medians.

## Scoring

```text
score = clamp(50 + 25 * log2(S), 0, 100)
S     = geometric mean over workloads of (reference latency / your latency)
```

50 at the reference's speed, 75 at twice its speed, 100 at four times.

**Beats the reference**: on every workload, latency_ms (the smallest of three median latencies)
must be strictly less than the reference's (zero tolerance; equality fails), with every
checked entry within tolerance.

## Running locally
On a GPU runner (see the Agent README), from the copied tree:

```sh
/opt/frontier/task-runtime sh frontier-build.sh   # imports kernel, checks the interface
/opt/frontier/task-runtime sh frontier-bench.sh   # one small shape: latency and max error
```

## Environment
The judge runs on one H100 with the same PyTorch 2.6 / CUDA 12.4 image as your GPU runner.
