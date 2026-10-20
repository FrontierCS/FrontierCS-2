# GPU DBSCAN Kernel Optimization

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

Optimize the Python/Triton implementation under `dbscanlib/`. Keep this interface:

```python
labels = dbscanlib.dbscan(x, eps, min_samples, max_neighbors=32)
```

`x` is an `(N, D)` float32 CUDA tensor whose coordinates carry bfloat16
precision. Return an `(N,)` int64 CUDA tensor on the input device, with
nonnegative cluster labels and `-1` for noise. The result must represent Euclidean
DBSCAN: the eps-neighbor count includes the point itself; core points have at
least `min_samples` neighbors; connected core points form clusters; a border
point joins the smallest core-component label among its eps-neighbors. The naive
starter is the frozen baseline for the independent speed objective and clustering
agreement. Cluster numbers can be permuted; the noise marker keeps its meaning.

## Workloads

Train uses these two public shapes on one H100:

| ID | N | D | eps | min_samples | n_centers |
| --- | ---: | ---: | ---: | ---: | ---: |
| w0 | 20000 | 8 | 0.90 | 5 | 4 |
| w1 | 20000 | 32 | 1.15 | 5 | 6 |

Final independently runs both public shapes with the same public seed policy
and two additional hidden shapes with independent hidden seeds; these four
workloads contribute to its score and pass/fail. From beat spec v3, final
pass/fail (not the score) also covers two further hidden shapes: one with
200,000 points, and one with `min_samples=20` where noise and border points are
present. The generator, graded seeds,
hidden shapes and optimized reference are Verifier-only. Implement the general
clustering operation; do not recognize inputs, cache prior outputs or inspect
the judge. Fresh data is generated for every warmup and timed repetition.

## Measurement and score

Each implementation receives two warmups and three timed repetitions per
workload. A synchronized DBSCAN call is timed; input generation and label-quality
checking are outside timing. The judge computes the median of three paired
ratios `naive_latency_i / candidate_latency_i` for each workload, then the
geometric mean `S` over the selected suite's original workloads (the two train
workloads, or the four original final workloads). This preserves the original timing
aggregation; it is not a ratio of independently selected median latencies.

Every timed candidate result must attain Adjusted Rand Index (ARI) at least
`0.99` against the frozen naive labels on that same input. If any timed result
misses this original correctness gate, its continuous score is zero. Otherwise,
score is zero for `S <= 1`; for `S > 1` it is
`100 * log(S) / (log(10) + log(S))`. Thus 10x maps to 50 and there is no finite
upper clipping. `S` remains available unrounded. The score never uses the
optimized reference or its historical speed calibration.

## Independent reference verdict

All three comparisons must pass on every selected workload (two in train, six
in final):

- Candidate median call latency is strictly less than optimized reference latency.
- Candidate minimum timed ARI is at least reference minimum timed ARI, and every
  candidate timed ARI is at least `0.99`.
- Candidate maximum timed noise-mask mismatch fraction is at most the reference
  maximum. Each mismatch is measured against the frozen naive noise mask; when
  the reference has zero mismatches, the candidate must also have zero.

There is no added quality tolerance. The optimized reference itself must meet
the `0.99` ARI gate on every timed input; otherwise the evaluation is a verifier
error rather than a comparison. ARI ignores arbitrary cluster numbering,
including a permutation that swaps a cluster with the noise group, so the
separate noise-mask comparison preserves the special meaning of `-1`. A fast
candidate may earn an independent continuous score while failing a reference
comparison. `beats_reference=1` requires the full conjunction, not an average.
Train returns sorted candidate aggregates, `S`, and per-metric aggregate progress;
reference measurements and individual workload decisions stay private.

The returned `geomean_speedup` is `S`. Candidate metric arrays are
`dbscan_latency_ms`, `adjusted_rand_index` and `noise_mismatch_fraction`.
`beat_spec_progress` reports `passed`, `failed`, `total`, `pass_percent` and
`fail_percent` separately for each reference comparison.

## Development and submission

Run `bash frontier-build.sh` on your pushed GPU workspace for imports, then
`bash frontier-bench.sh` for a small ungraded correctness diagnostic. It checks
cluster equivalence and noise identity without using graded inputs or reference
code. Use `/frontier/task/bin/README.md` for Candidate-owned GPU helper v3, train,
store-final, explicit final, separate filesystems and the exact task budgets.

Only Python files below `dbscanlib/` may change. Do not import external optimized
libraries, spawn processes, access the network/environment/judge, or alter timing.
Use an early meaningful train submission, submit after substantive progress,
and use feedback to decide whether to update your stored final. There is no fixed
stop time or required submission cadence.

## Local regression modules

Local regression-test modules such as `dbscanlib/test_regress.py` are allowed and
packaged without filtering. Files named `refdbscan.py` or `conftest.py` are reserved
at every depth and cannot be changed or added. Trusted files outside `dbscanlib/`
remain read-only.
