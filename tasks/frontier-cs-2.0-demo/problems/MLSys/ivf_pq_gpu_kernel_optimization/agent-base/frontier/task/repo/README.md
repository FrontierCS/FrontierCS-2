# GPU IVF-PQ Search Kernel Optimization

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

## Objective and editable code

Optimize `ivfpqlib.ivf_pq_search(index, Q, k, *, nprobe=None) -> (vals, ids)`
on one H100. Edit only Python files under `ivfpqlib/`; new Triton modules there
are allowed. Preserve the signature and deterministic input/output meaning.
External optimized ANN/kernel libraries, process/network/environment access,
and measurement tampering are forbidden in submitted code.

The index is a fixed residual IVF-PQ index. `Q` contains float32 CUDA queries.
Return float32 approximate squared-L2 distances and int64 original database
IDs, both of shape `(Q.shape[0], k)`. Valid IDs must be in range and cannot repeat
within a query. An unused slot is `id=-1, distance=+inf`; padding is excluded
from recall hits. Distances for valid neighbors must be finite and numerically consistent with
the frozen float32 residual ADC calculation: each valid distance must satisfy
`abs(returned - oracle) <= 1e-6 * abs(oracle)` (relative tolerance1e-6, absolute
tolerance0). A zero oracle distance requires an exactly zero returned distance.
Legal `-1`/`+inf` padding is checked separately and excluded from this comparison.

## Experiment

The data is the real SIFT ANN-benchmarks database and query pool. The frozen
naive builder constructs the index outside timing. Each search call receives a
fresh index object and freshly cloned tensors with identical index contents.
Centroids, PQ codebooks and queries are rounded to bf16 precision, then kept as
float32 tensors. Each iteration draws a fresh deterministic subset of queries.
Do all index layout conversion or preprocessing required by search inside that
search call; do not amortize it into warmup or cache prior answers.

The public train workloads all use `nlist=8192`, `m=16`, `k=100`:

| Workload | Queries | Probed lists |
| --- | ---: | ---: |
| sift_q4k_np32 | 4096 | 32 |
| sift_q4k_np64 | 4096 | 64 |
| sift_q2k_np64 | 2048 | 64 |
| sift_q2k_np128 | 2048 | 128 |

Train evaluates these four workloads. Final freshly evaluates the same public
workloads together with two original hidden shapes using independent index
and query seeds; final score and PASS use these six workloads. From beat spec v3,
final PASS (not the score) also covers two further hidden shapes in regimes the
public workloads do not reach: up to the full 10,000-query SIFT test set, and
large-k selection with k up to 1024. Hidden shapes, seeds, optimized reference
measurements and per-workload verdicts are not available through train feedback.

## Measurement and validity

Only the search call is timed, using synchronized GPU wall-clock milliseconds.
Each workload has two warmup calls and seven timed repetitions. Each repetition
runs frozen naive search and then your search on the same index and queries.
The workload speedup is the median of these seven paired timing ratios.

Every timed call must pass output validation and achieve iso-result recall at
least 0.99 against the full-nprobe naive ADC ranking. Iso-result recall counts
valid returned neighbors also present in the naive top-k, divided by queries
 times k. It measures agreement with that fixed approximate-index search, not
recall against exact nearest neighbors of the original uncompressed database.
Duplicated valid IDs are invalid; repeated padding never adds hits.

Outside timing, each timed call also reports nearest-neighbor recall@k, the
IVFADC evaluation measure: the fraction of queries whose exact nearest neighbor
in the uncompressed SIFT base (float32 brute force on the actual bf16-rounded
queries) is among the k returned vectors. A returned vector at the exact nearest
distance, such as a duplicate database vector, also counts; padding never counts.
It has no absolute floor of its own.

## Independent continuous score

Let S be the geometric mean of the workload speedups over the selected suite's
original workloads (the four train workloads, or the six original final
workloads; the two v3 hidden shapes are excluded).
The score is 0 when S <= 1; otherwise:

```text
score = 100 * log(S) / (log(10) + log(S))
```

Tenfold speedup maps to 50. Finite improvements above S=1 remain ordered with no
upper clipping; raw speedup is reported separately. This score depends on the
naive starter and your measurements, independently of the optimized reference
used for PASS. Failing a reference comparison does not zero a valid score.

## Reference PASS

Every selected workload must satisfy all three comparisons against the frozen
optimized reference measured on the same workload and seed:

- Your median search latency is strictly lower; equality fails.
- Your worst iso-result recall over the seven timed calls is at least the
  reference's worst iso-result recall; equality is allowed, with no additional
  quality band.
- Your worst nearest-neighbor recall@k over the seven timed calls is at least
  the reference's; equality is allowed, with no additional quality band.

No aggregate improvement compensates for a failed workload or metric. The
absolute 0.99 validity gate still applies to every timed call.

## Development and submission

Read `/frontier/task/README.md` and `/frontier/task/bin/README.md` for the exact
patch archive, train, replaceable stored final, explicit final, GPU and quota
commands. GPU development runs the public experiment; final uses an independent
Verifier. `frontier-build.sh` checks imports, and `frontier-bench.sh` runs a
public workload on the development GPU. It defaults to the first public shape;
pass a public workload ID to select another. This can be slow because the naive
search includes many Python dispatches and the frozen index build is substantial.

## Local regression modules

Local regression-test modules such as `ivfpqlib/test_regress.py` are allowed and
packaged without filtering. Files named `refivfpq.py` or `conftest.py` are reserved
at every depth and cannot be changed or added. Trusted files outside `ivfpqlib/`
remain read-only.
