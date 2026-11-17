# KAMI FP64 Block GEMM — trusted batched harness adaptation

Implement the missing FP64 KAMI tensor-core block-GEMM device kernel for one exact H100 and make
it as fast as possible. The score rewards validated batched throughput; every output element of
every timed launch must be correct.

## Task

KAMI (communication-avoiding GEMM within a single GPU, SC 2025) computes small dense blocks with
tensor-core MMA and a rank decomposition. This tree keeps the upstream FP64 1-D method,
compile-time block shapes and solver-facing signature, but the device implementation of
`block_gemm_1d_double_mma` is excised. Evaluation uses a trusted batched harness, not the upstream
host benchmark: every scored CTA owns distinct A/B/C slices, `NUM_ITER` is exactly one, every
result element is checked, and the counted work is exactly the real independent batch.

## What you may change

- `src/block_gemm/KAMI/1d/fp64/block_gemm_1d_double_mma.cu`, **only inside the original device
  region**: every byte before `#define BLOCKS_PER_GRID 16384` plus its following blank line, and
  from the blank line before `int main(int argc, char *argv[])` on, stays as the starter has it.
  A change outside the region is rejected as `out_of_scope`.
- New or changed files below `src/block_gemm/KAMI/1d/fp64/solver_extensions/` (ordinary,
  non-executable text files, no links).
- At most 12 changed files; no deletions; textual patches only.

There is no ban on `return`, `blockIdx`, implementation strings, scalar or tensor-core code:
correctness is enforced by independent data and full-output validation.

## Interface

The source must define the original entry exactly once, as a definition:

```cpp
__global__ void block_gemm_1d_double_mma(
    const double *A, const double *B, double *C,
    const double alpha, const double beta);
```

The public adapter (`frontier_harness/adapter.py`) converts candidate and reference alike: it
locates that definition (ignoring comments and literals), turns it into an inline device helper
with the same arguments and body, overrides `NUM_ITER` to `1`, and adds the fixed entry
`frontier_batched_block_gemm_entry`. For CTA `b` the entry passes

```text
A + b * (M_BLOCK * K_BLOCK),  B + b * (N_BLOCK * K_BLOCK),  C + b * (M_BLOCK * N_BLOCK)
```

to your helper. Per workload the translation unit is compiled with `nvcc --cubin -arch=sm_90a
-std=c++17 -O3` and `-DM_BLOCK=M -DN_BLOCK=N -DK_BLOCK=K -DNUM_RANK_BLOCK=r
-DNUM_ALLOC_RANK_BLOCK=a`, with `(r, a)` = (1, 1) for M = 16, (2, 4) for M = 32, (4, 4) for
M = 64 and (8, 8) for M = 128 (train and hidden shapes alike); the CUDA host suffix is not
compiled or linked. The judge launches `batch` CTAs of `a × 32` threads with `(N / r) · K · 8`
bytes of dynamic shared memory. `frontier-build.sh` compiles only the (1, 1) case. A separately compiled trusted host loads only the fixed entry
through the CUDA Driver API, with candidate and reference in separate CUDA contexts, so a
submitted host initializer or timing function never runs.

Each CTA computes, with row-major FP64 `A` (`M_BLOCK × K_BLOCK`), `B` (`N_BLOCK × K_BLOCK`) and
`C` (`M_BLOCK × N_BLOCK`):

```text
C[b] = 1.0 * (A[b] * transpose(B[b])) + 0.5 * C_initial[b]
```

## Correctness

Warm-up and every timed output are copied back and checked element by element against a trusted
CPU oracle:

```text
abs(observed - expected) <= 1e-12 + 2e-13 * abs(expected)
```

An unwritten slice, a CTA-0 clone, exchanged slices, reuse of the warm-up result, a non-finite
output or one wrong element fails the workload. A build failure (build 300 s), a validation
failure or a workload running longer than 900 s scores 0. A candidate checksum is never trusted.

## Workload

Inputs are small signed dyadic values from a public counter-based SplitMix64 permutation, keyed
per phase, tensor, CTA and element; seeds are withheld.

| M×N×K | (r, a) | batch CTAs | real scored FLOP | device A+B+C | host A+B+C+golden C |
| --- | --- | ---: | ---: | ---: | ---: |
| 16×16×24 | (1, 1) | 65,536 | 805,306,368 | 512 MiB | 640 MiB |
| 32×64×48 | (2, 4) | 8,192 | 1,610,612,736 | 416 MiB | 544 MiB |
| 64×64×32 | (4, 4) | 8,192 | 2,147,483,648 | 512 MiB | 768 MiB |
| 128×64×64 | (8, 8) | 4,096 | 4,294,967,296 | 640 MiB | 896 MiB |

- Train runs these four points. Final reruns them with the same (withheld) train seeds plus four
  hidden points (shapes absent from train, independent seeds).
- Host and device data are each at most 1 GiB; every batch has at least 4,096 CTAs.
- Per workload the trusted host prepares one batch and its CPU golden output, then runs candidate
  and reference in six alternating repetitions (three candidate-first, three reference-first),
  each a new timed phase. The CUDA event encloses one launch of the fixed entry; input
  preparation and the oracle are outside it.

## Scoring

```text
operations = 2 * M_BLOCK * N_BLOCK * K_BLOCK * batch_count
tflops     = operations / (event_milliseconds * 1e9)
value_w    = sqrt(median(candidate-first samples) * median(reference-first samples))
G          = geometric mean of value_w over the suite (4 in train, 8 in final)
score      = max(0, min(100, 100 * G / (G + 20)))
```

Anchors: `G = 20` TFLOPS scores 50; `G = 60` scores 75.

**Beats the reference** when, on every workload of the suite, the candidate's `value_w` divided
by the source-locked reference's measured in the same run is strictly above a private strict
improvement ratio (derived from the exact-H100 reference self-noise envelope); equality fails.
`beats_reference` is the conjunction over the workloads.

## Running locally

On the H100 runner (see `/frontier/task/README.md`): `bash frontier-build.sh` compiles the
device-only cubin and the trusted host without running a workload; `bash frontier-bench.sh` runs
the public debug workload through the same batched adapter with a public seed. The repository's
`bash make_submission.sh` writes a local `solution.patch`; the submitted patch comes from
`/frontier/task/make_submission.sh`.

## Environment

CUDA 12.8 (`nvcc`, sm_90a) on one exact NVIDIA H100 80GB HBM3, Python 3 judge. Upstream plots
and scripts target GH200 and RTX 5090; only the H100 is used here.
