# KAMI FP64 Block GEMM — trusted batched harness adaptation

## Scope
This task asks for the missing FP64 KAMI block-GEMM device implementation. It
preserves the upstream KAMI tensor-core method, compile-time block shapes,
rank decomposition, and solver-facing CUDA function signature. Evaluation uses
a versioned **trusted batched harness adaptation**: it does not claim to execute
the upstream host benchmark unchanged.

The adaptation closes two invalid shortcuts in the old harness. That harness
gave 16,384 CTAs one shared A/B/C matrix, required a result scaled by 1,000,
and counted `16,384 * 1,000` matrix products without proving that distinct CTAs
or iterations performed them. Here every scored CTA owns distinct A/B/C slices,
`NUM_ITER` is exactly one, every result element is checked, and the counted work
is exactly the real independent batch.

## Editable interface
The submission remains a unified patch. It may modify only:

```text
src/block_gemm/KAMI/1d/fp64/block_gemm_1d_double_mma.cu
src/block_gemm/KAMI/1d/fp64/solver_extensions/**
```

The source must define the original entry signature:

```cpp
__global__ void block_gemm_1d_double_mma(
    const double *A, const double *B, double *C,
    const double alpha, const double beta);
```

Bytes outside the original device region remain locked. Solver extension
entries must be ordinary files or directories. There is no ban on `return`,
`blockIdx`, implementation strings, scalar code, tensor-core code, or other
computation choices: correctness is enforced behaviorally by independent data
and full-output validation.

## Trusted compilation boundary
The public adapter performs one deterministic conversion on candidate and
reference alike:

1. It lexically locates the single original kernel definition while ignoring
   comments and literals.
2. It changes that definition into an inline device helper with the same five
   arguments and unchanged body.
3. It overrides the source-locked upstream `NUM_ITER` macro to `1`.
4. It adds the fixed task-owned global entry
   `frontier_batched_block_gemm_entry`.

For CTA `b`, the fixed entry passes:

```text
A + b * (M_BLOCK * K_BLOCK)
B + b * (N_BLOCK * K_BLOCK)
C + b * (M_BLOCK * N_BLOCK)
```

to the submitted helper. The resulting translation unit is compiled with
`nvcc --cubin -arch=sm_90a`. The original CUDA host suffix is not compiled or
linked.

A separately compiled task-owned host loads only the fixed entry through the
CUDA Driver API. Candidate and reference modules use separate CUDA contexts.
Consequently, a submitted host static initializer or same-named CUDA timing
function cannot run in, replace, or link into the trusted host. Device code may
still choose any way to compute the required result.

The public `frontier-bench.sh`, public compile check, and private evaluator all
import the same `frontier_harness/adapter.py` and generic protocol. Final
independently reruns the public workload set and adds verifier-only shapes and
seeds without changing pointer semantics or the trusted timing boundary.

## Batched workloads and memory
Development contains four public KAMI policy points. Final independently
reruns those four public workloads with the same public seeds and adds the four
existing hidden points with shapes absent from train and independent private
seeds. The hidden workload identities and parameters remain verifier-only.

The public development batch table is:

| M×N×K | batch CTAs | real scored FLOP | device A+B+C | host A+B+C+golden C |
| --- | ---: | ---: | ---: | ---: |
| 16×16×24 | 65,536 | 805,306,368 | 512 MiB | 640 MiB |
| 32×64×48 | 8,192 | 1,610,612,736 | 416 MiB | 544 MiB |
| 64×64×32 | 8,192 | 2,147,483,648 | 512 MiB | 768 MiB |
| 128×64×64 | 4,096 | 4,294,967,296 | 640 MiB | 896 MiB |

Host and device data budgets are checked separately at 1 GiB. Every public and
hidden batch meets the 4,096-CTA scheduling floor. The H100 has 80 GiB, while
the verifier CPU allocation is 4 GiB; buffers are reused between warm-up and
timed phases rather than holding both phases at once.

The table is a GPU-calibration starting point, not a promise that these
launches are long enough for a stable strict comparison.

## Inputs and result
For every phase, tensor, CTA, and element, a public counter-based SplitMix64
permutation generates a small signed dyadic value. The full 64-bit counter
stream has no small modulo period. CTA-major slices are independently keyed.
Exact private seeds are withheld, but the generator, mapping, phase salts, and
all pointer semantics are public.

Each CTA computes:

```text
C[b] = 1.0 * (A[b] * transpose(B[b])) + 0.5 * C_initial[b]
```

`A` is `M_BLOCK × K_BLOCK`, `B` is `N_BLOCK × K_BLOCK`, and `C` is
`M_BLOCK × N_BLOCK`, all row-major FP64. The nonzero public `beta=0.5` makes the
independent C input part of correctness. Input numerators are bounded dyadic
integers, so the trusted CPU oracle has a tight finite mixed tolerance:

```text
abs(observed - expected) <= 1e-12 + 2e-13 * abs(expected)
```

Warm-up uses a phase distinct from every timed repetition. Warm-up output and
all timed outputs are copied back and checked element by element. An unwritten
slice, CTA-0 clone, exchanged slice, warm-result reuse, non-finite output, or
single corrupted element fails the whole workload. A checksum produced by the
candidate is never trusted.

## Timing, oracle, and score
For a workload, the trusted host prepares one batch and its CPU golden output,
then executes candidate and source-locked reference in isolated CUDA contexts.
Their order alternates across six repetitions, giving exactly three
candidate-first and three reference-first pairs; each repetition has a new
timed phase. Input preparation and CPU-oracle durations are reported privately
and are outside CUDA events. The event encloses exactly one launch of the fixed
entry.

The only scored operation count is:

```text
operations = 2 * M_BLOCK * N_BLOCK * K_BLOCK * batch_count
tflops = operations / (event_milliseconds * 1e9)
```

For each implementation, take the median of its three samples in
candidate-first pairs and the median of its three samples in reference-first
pairs. The workload value is the geometric mean of those two order-arm
medians. This balanced crossover statistic gives first and second launch
positions equal weight; a stable multiplicative position effect therefore
cancels in the candidate/reference ratio. Development uses four crossover-adjusted workload values; final uses all eight
public-plus-hidden values. The selected values form geometric mean `G`, then:

```text
score = max(0, min(100, 100 * G / (G + 20)))
```

The release snapshot uses a private strict improvement ratio derived by
upward-rounding the complete exact-H100 crossover self-noise envelope. A
workload wins only when its crossover-adjusted candidate/reference ratio is
strictly above that boundary; equality fails. `beats_reference` is the conjunction over every workload selected for that
verification. Final therefore requires all eight public-plus-hidden workload
comparisons to pass. The former `1.006` boundary, old 69.x
scores, and old stage/iteration witness belong to the invalid shared-input
benchmark and are not reused.

The exact H100 repeated the complete balanced reference-self crossover decision
statistic on all eight unique workloads before the original immutable release
snapshot. A previous balanced six-pair
snapshot still divided the two direct six-sample medians; two exact-H100
replications showed a repeatable first/second-position bias, so that snapshot
is calibration-invalid. The release threshold is strictly outside the fresh
crossover-ratio envelope. The earlier `1.467` mixed raw samples
into the wrong statistic, while `1.016` came from a single 2:1 order-imbalanced
three-pair run; neither is the release threshold. The remaining release gates
require unchanged reference to have zero wins on all eight workloads and a
distinct valid witness to clear the frozen rule on all eight.

## Public commands
On the configured H100 workspace, public debug uses the same batched adapter
with a public seed:

```bash
bash frontier-bench.sh
```

The Agent compile check performs device-only cubin and trusted-host compilation
without running a workload:

```bash
bash frontier-build.sh
```

The repository-level submission helper remains:

```bash
bash make_submission.sh
```

## Source locks and status
The adaptation preserves these release inputs:

```text
reference.patch SHA-256:
  e104c534ba929b73d0ca1a7104aebdfc6bdb2a0a5a646d7119b14e2859d2228a
public excised CUDA source SHA-256:
  54ca1476848d70bedc469a04e24c85224305bf4a5926bc798ec807b81b7ee08e
reference-restored upstream CUDA source SHA-256:
  2b822cd274b0ff600d574c06cb0e52da97216c8d01b15d0b5932d97edaafebf9
```

Compile-only validation is not a performance claim. Calibration GPU execution, batch
adequacy, exact-H100 timing noise, new reference scores, a degraded measured
baseline, and a distinct beating witness remain release gates.
