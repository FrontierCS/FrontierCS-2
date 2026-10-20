# GPU K-Means Kernel Optimization

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

Optimize one exact Lloyd K-Means step:

```python
kmeanslib.step(x, centroids) -> (labels, new_centroids)
```

`x` is an `(N, D)` CUDA bfloat16 tensor and `centroids` is `(K, D)` bfloat16.
Return the nearest-centroid assignment of every point as `(N,)` integer labels,
then the `(K, D)` means of those assigned points. Empty clusters retain their
previous centroid. Results must be deterministic functions of the inputs.

The judge owns the data, initial centroids and loop. It calls `step` exactly two
times per workload, feeding each returned centroid tensor into the next call.
You can rewrite the implementation and add Python/Triton modules below
`kmeanslib/`, including a fused assignment/update kernel, while preserving the
public contract.

## Workloads

Train runs these four public shapes on one exact H100:

| ID | N | D | K | Lloyd steps |
| --- | ---: | ---: | ---: | ---: |
| w0 | 200,000 | 64 | 512 | 2 |
| w1 | 300,000 | 128 | 1,024 | 2 |
| w2 | 500,000 | 128 | 1,024 | 2 |
| w3 | 150,000 | 512 | 512 | 2 |

Final independently measures all six shapes: the four public shapes with the
same train seeds, plus two hidden shapes with independent seeds. Hidden shapes
are withheld from the Agent images and feedback. Final does not reuse train
measurements.
Inputs are deterministic planted bfloat16 clusters. The generator and seeds are
Verifier-only. Implement a general step; do not recognize a workload or cache a
prior result.

## Metrics

Train scores the four public workloads. Final scores all six public and hidden
workloads together; its geometric mean has six speedups.
The independent raw metric is the geometric mean of that suite's speedups over the
naive starter, where each speedup is naive median loop latency divided by
candidate median loop latency. The reported continuous score maps that raw metric
to `[0, 100)` using `S = geomean_speedup`: score is 0 when `S <= 1`,
and otherwise `100 * log(S) / (log(10) + log(S))`. Thus `10x` scores 50,
`100x` scores about 66.67, and `1000x` scores 75. Above `1x`, the curve
is strictly increasing and approaches 100 without a finite full-score threshold
or upper clipping. Raw speedup remains available for precise comparisons;
neither the speedup nor its score uses `reference.patch`.

A separate binary verdict compares the candidate with the frozen optimized
`reference.patch` on every workload in the selected suite. Required metrics are
the synchronized latency of the two-step loop, labeled inertia of the final returned
labels/centroids, and the fraction of final labels that are not an exact nearest
centroid of the centroids supplied to the last step. Every required comparison
on every workload must clear for `beats_reference=1`; final requires 6/6 for
each of the three metrics. Averages cannot compensate.
For every workload, candidate latency must be strictly lower than reference
latency, candidate labeled inertia must be at most `1.05 * reference`, and
candidate label mismatch fraction must be at most `reference + 0.002`.

Train returns only independently sorted candidate measurements, the independent
speedup, and aggregate per-metric pass counts over the four train workloads.
Final measurements and the full reference comparison remain operator-only.

## Patch policy

Only `.py` files under `kmeanslib/**` may change. New Python and Triton modules in
that package are allowed. Deletions, binary patches, more than 80 changed files,
external optimized ML/kernel libraries, network/process access and measurement
tampering are rejected. The submitted patch must apply to the pristine starter.

## Local regression modules

Local regression-test modules such as `kmeanslib/test_regress.py` are allowed and
packaged without filtering. Files named `refkmeans.py` or `conftest.py` are reserved
at every depth and cannot be changed or added. Trusted files outside `kmeanslib/`
remain read-only.
