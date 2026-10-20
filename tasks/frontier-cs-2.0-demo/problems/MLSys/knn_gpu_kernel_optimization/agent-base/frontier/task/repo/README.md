# GPU Brute-Force k-NN Kernel Optimization

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

Implement `knnlib.knn(queries, database, k) -> (distances, indices)` on one H100.
The inputs are contiguous CUDA bfloat16 tensors with shapes `(Q, D)` and `(M, D)`.
They contain real image descriptors sampled from SIFT and GIST. Every warmup and
timed iteration samples fresh query rows and fresh database rows, then converts
them to bf16. Work on the current inputs; do not cache previous search results.

Return `(Q, k)` squared-L2 distances and `(Q, k)` int64 database row indices on
the same GPU. Distances are float32, finite and nonnegative for valid neighbors.
Return distinct valid indices in ascending distance order. When `k > M`, only
trailing padding with index `-1` and distance `+inf` is permitted after all `M`
real neighbors. The graded workloads all have `k <= M`. Duplicate valid indices
are rejected rather than counted repeatedly.

## Public workloads

Train runs these four public shapes on one exact H100:

| ID | Dataset | Queries Q | Database M | Dimensions D | k |
| --- | --- | ---: | ---: | ---: | ---: |
| sift_k10 | SIFT | 1,024 | 500,000 | 128 | 10 |
| sift_k50 | SIFT | 1,024 | 500,000 | 128 | 50 |
| sift_k100 | SIFT | 1,024 | 500,000 | 128 | 100 |
| gist_k10 | GIST | 512 | 500,000 | 960 | 10 |

Final independently measures all six original workloads: these four public
workloads with the same train seeds, plus two hidden workloads with independent
seeds. From beat spec v3, final PASS (not the score) also covers two further
hidden shapes over the full 1,000,000-row SIFT and GIST bases, one with k up to
256. Final measurements are fresh. Hidden workload parameters and all graded
seeds remain Verifier-only. Agent GPU checks are diagnostics; use the 25 train
calls for graded feedback.

## Quality and scoring

Each workload has two warmups and seven timed iterations. Each timed iteration
runs the frozen naive search followed by your implementation on the same sampled
values. The workload speedup is the median of the seven paired ratios
`naive_time / candidate_time`; it is not the ratio of two latency medians.
`S` is the geometric mean across four workloads for train and the six original
workloads for final (the two v3 hidden shapes are excluded).
The continuous score is zero when `S <= 1`, otherwise
`100 * log(S) / (log(10) + log(S))`. It is 50 at 10x and approaches 100 without
finite saturation. The optimized reference does not normalize this score.

Every timed output must achieve ball-recall at least `0.88`; failing any timed
iteration makes the score zero. The judge computes fp32 squared-L2 distances
from the actual bf16 inputs and your returned indices. The ball radius is the
fp32 distance to the frozen naive search's kth returned neighbor, with the
legacy `1e-6` absolute epsilon. This preserves the original tie-tolerant oracle;
it does not claim a separate exact fp32 top-k search over the entire database.

Outside timing, each timed call also reports exact recall: the share of your
returned neighbors whose fp64 squared distance from the actual bf16 inputs is at
most the fp64 distance of the query's exact k-th nearest database row. The judge
ranks the whole sampled database by fp32 brute force and recomputes the k
selected rows in fp64, so this radius is never below the true k-th distance.

The timing denominator is the frozen original naive cdist/topk implementation.
The shipped starter additionally recomputes selected distances in FP32 and sorts
them with their IDs; this work is included in its measured call.

A separate PASS requires every check on every final workload (8/8 each):

- Median candidate latency is strictly lower than the optimized reference.
- Worst timed candidate ball-recall is at least the optimized reference's worst
  timed ball-recall, with no additional tolerance, and every iteration meets 0.88.
- Worst timed candidate exact recall is at least the optimized reference's worst
  timed exact recall, with no additional tolerance.
- Every returned distance agrees with direct squared-L2 recomputation from its
  returned index and actual bf16 inputs. The judge uses fp64 differences and
  accumulation. Every valid returned distance must satisfy
  `abs(reported - oracle) <= 1e-6 * abs(oracle)` (`rtol=1e-6`, `atol=0`).
  A zero oracle requires exactly zero; legal padding is excluded.

Train returns independently sorted absolute candidate latency and worst-recall
samples, independent speedup and aggregate per-metric progress. Reference values,
per-workload decisions and the final PASS verdict remain operator-only.

## Patch and workflow

Read `/frontier/task/README.md` and `/frontier/task/bin/README.md` for train,
replaceable store-final, explicit final, GPU helper and the actual task budgets.
Edit only `.py` files under `knnlib/`; new Python/Triton modules are allowed.
Local regression-test modules such as `knnlib/test_regress.py` are allowed and
packaged without filtering. Files named `refknn.py` or `conftest.py` are reserved
at every depth and cannot be changed or added.
Deleting files, external optimized libraries, network/process access and timing
or evaluator tampering are forbidden. Submit a patch against the pristine
starter with `/frontier/task/bin/make_submission.sh`.
