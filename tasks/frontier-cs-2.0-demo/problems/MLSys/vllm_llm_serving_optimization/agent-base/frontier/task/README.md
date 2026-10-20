# vLLM serving and scheduling optimization

Optimize the Python serving/scheduling implementation of vLLM 0.11.0 for
Qwen/Qwen3-Coder-30B-A3B-Instruct on one H100. The benchmark runs SWE-bench
software-engineering conversations and BFCL memory/tool conversations against
your server. Improve end-to-end conversation latency while preserving correctness.
The optimized reference is the original job-aware scheduling patch; it is
independent of the unmodified vLLM baseline used for continuous scoring. Both
baselines are private immutable measurements from three authenticated calibration
captures per phase; TRAIN and FINAL use their own independently authenticated
three-capture sets, which need not have identical run identities. Ordinary
training and final evaluation run only your candidate.

Read [testbed operations](bin/README.md) for workspace preparation, the patch/tar
format, train, store-final, final and GPU operations. The packaged clean checkout is `/frontier/task/repo/vllm`; prepare
an editable copy at `/workspace/repo`. The source/assets in the task package
are the fixed baseline used when applying your submitted patch.

Submission: `bin/make_submission.sh` writes your diff to `payload/solution.patch`;
submit the archive with `frontier-submit train` (at most 25 train calls, each a
complete public-rate evaluation) and finally once with `frontier-submit final`.

Budgets: the formal answering window is 18000 seconds (five hours), including
train waits. After you exit, independent final evaluation runs with its own
108000-second budget; it does not consume the answering window.

## Allowed changes

The payload is a text `.patch` against the original vLLM source. It may change
at most 80 files and occupy at most 1,500,000 bytes. Empty means unmodified
vLLM. Only Python `.py`/`.pyi` changes in the original allowlist are accepted:

- `vllm/v1/core/**`, including scheduler and KV-cache code;
- `vllm/config/scheduler.py` and `vllm/config/cache.py`;
- `vllm/v1/worker/**`, `vllm/v1/engine/**`, `vllm/v1/executor/**`;
- `vllm/v1/request.py`, `outputs.py` and `serial_utils.py`;
- `vllm/entrypoints/openai/protocol.py`, `serving_engine.py`, `serving_chat.py`
  and `serving_completion.py`;
- `vllm/sampling_params.py`.

Do not change native kernels, model implementation/weights, build/dependency
files, tests, datasets or the evaluation workload. Source deletions and binary
patches are rejected. Preserve generation parameters and tool/model semantics;
benchmark-specific hardcoding and judge/credential inspection are forbidden by
the original added-code policy. Static patch validity alone earns no score.

## Workloads and timing

Train uses the original five public SWE instances at 0.5 jobs/s and eight public
BFCL instances at 1 job/s. Independent final includes the complete original
48 SWE and 155 BFCL instances, including the public instances, at the public
load and separate hidden load groups. Hidden arrival-rate values, seeds and
instance details are not exposed in training feedback.

Testbeds are prepared before the original arrival schedule begins. Their setup
cost is recorded separately, not counted as conversation latency. Each measured
conversation includes model requests, interactive tools and final patch
extraction. Timing runs on one host; model/tool work is not approximated by
server-only token timings or post-hoc network subtraction. SWE and BFCL are
separate sequential workloads, not a simultaneously mixed 80/20 request stream.
The three phase-specific calibration captures measured unmodified vLLM and optimized reference
with this procedure. Ordinary evaluation measures only the candidate against
those fixed baselines; it does not rerun baseline services.

Fixed vanilla timing uses the per-instance median latency across the three
captures. For both SWE and BFCL, an instance is vanilla-correct if at least two
of the three captures answered it correctly. This per-instance majority count
is used consistently for the continuous accuracy penalty.

## Identical group requirements

SWE and BFCL each use the same five fixed-reference comparisons described below.
There is no vanilla BFCL flip/regression gate, extra BFCL accuracy threshold,
flip-triggered whole-submission invalidity or score-zero override. Vanilla
correctness remains the baseline for the continuous accuracy multiplier only.
Candidate service Runtime Error still denies credit to its affected workload;
unrelated platform faults remain unscored infrastructure errors. Malformed
submissions still deterministically FAIL0 under the existing input contract.

## Continuous score

The continuous score uses the unmodified vLLM baseline, never the
optimized reference.
For every expected instance in a workload/load group, let
`r_i = vanilla_latency_i / candidate_latency_i`. A valid finite positive ratio
is used exactly, with neither lower nor upper clipping: a real ratio of `1/20`
remains `1/20`, and a speedup above `8` remains above `8`.

The expected instance denominator is fixed. A missing or unusable execution
result attributable to the candidate contributes `r_i = 1/8`; it is not omitted
and cannot claim a large speedup from failing quickly. This value is an explicit
penalty, not a measured latency. Baseline, provider authentication, testbed and
platform failures are separate execution errors rather than candidate scores.

For each load group, `S_group = exp(mean(log(r_i)))`. In final, combine the public
and hidden load groups within each workload with equal geometric weight:
`S_workload = sqrt(S_public * S_hidden)`. Train uses its public group only.
Map that workload speedup to a latency score:

```text
L(S) = 0                                      if S <= 1
L(S) = 100 * log(S) / (log(10) + log(S))         if S > 1
```

Thus `10x` maps to 50 and improving finite speedups remain comparable above old
caps. Keep full-precision measurements; do not round intermediate ratios.

Correctness is measured by real SWE test resolution and original BFCL answer/
tool correctness. Within each workload, pool correct counts from public and
hidden load runs and divide by their total instance evaluations. Compute the
same pooled fraction separately for vanilla and candidate. Train uses its
public group's fraction. If `A_v` and `A_c` are these fractions:

```text
absolute_drop = max(0, A_v - A_c)
relative_drop = absolute_drop / max(A_v, 1e-9)
quality_multiplier = 1                       if absolute_drop <= 0.05
                                               OR relative_drop <= 0.05
quality_multiplier = 0.05 / relative_drop      otherwise
workload_score = L(S_workload) * quality_multiplier
continuous_score = 0.8 * SWE_score + 0.2 * BFCL_score
```

Both workloads are required. A missing BFCL workload does not renormalize the
score to SWE-only. The final continuous score is on the 0-100 scale; p90/p95/p99
are reported and used for PASS, not added to this scalar formula.

## PASS and tails

PASS is evaluated against the optimized reference separately for every
SWE/BFCL x public/hidden load group in final. All groups must pass all checks:

- Candidate paired geometric-mean conversation timing must strictly improve over
  the fixed vector taking the fastest reference latency per instance across the
  three calibration captures.
- Candidate p90, p95 and p99 conversation latency must each be strictly below
  the smallest observed value of that group percentile across the three captures.
  These three targets can come from different captures; they are not the tails
  of the per-instance fastest vector.
- Candidate correct count C must satisfy C >= median(R1, R2, R3) -
  max(1, ceil(0.05 * N)), where each R is the full group correct count from
  one reference capture and N is the fixed full selector size. Equality passes.
  FINAL SWE groups have N=48 and allowance 3; FINAL BFCL groups have N=155
  and allowance 8. TRAIN SWE N=5 and BFCL N=8 both have allowance 1.
  This uses the median of group totals, not per-case majority voting.
- The group must have complete usable candidate execution results. A missing
  or failed execution makes this group's PASS false despite its continuous
  penalty contribution.

Timing equality fails. There is no added percentage margin.
Improvements in another group or metric cannot compensate for a failed check.
Pooled correctness is used only for the continuous penalty; PASS does not pool
public and hidden correctness to hide a regression.

For p90/p95/p99, sort the instance end-to-end latencies and select one-based
item `ceil(p*N)` (zero-based index `ceil(p*N)-1`). This is nearest rank, without
interpolation. For five public SWE instances all three percentiles are the
maximum. For 48 final SWE instances p90 is item 44, p95 item 46 and p99 item 48.

## Wrong answers and incomplete execution

A normally completed wrong answer is a correctness outcome. It retains its
actual latency; it is not automatically an execution failure. BFCL `force_quit`
and SWE `limit_with_patch` caused by the original step limit can still have
complete usable retained results; those retain actual timing with correctness
judged independently. A shell command timing out inside an ongoing conversation
is tool feedback, not by itself a failed conversation. Truly absent/unusable
results or errors attributable to the submitted server receive the prescribed
execution penalty. Infrastructure errors are diagnosed separately.

SWE correctness requires the original issue-specific tests and regression tests.
A patch existing or applying successfully does not establish a resolved issue.
Only a measured case whose tests pass counts as correct. Missing, partial, timed-out,
unusable or malformed per-case results count as incorrect; other cases continue.
An unavailable shared scheduler or parent sidecar for the entire group is an
infrastructure error; independent per-case failures do not establish such an outage.

The independent Verifier evaluates the submitted archive afresh. Training
feedback returns these public aggregates for each of `swe_public` and
`bfcl_public` in `metrics`:

- `geometric_mean_latency_seconds`: geometric mean of candidate per-case
  end-to-end latency on the complete fixed public case list.
- `speedup_over_vanilla`: paired geometric-mean vanilla/candidate latency ratio,
  using the per-case median of the three fixed vanilla captures.
- `p90_latency_seconds`, `p95_latency_seconds`, `p99_latency_seconds`: candidate
  end-to-end latency percentiles, using integer nearest rank.
- `correct`: candidate correct count (out of 5 public SWE or 8 public BFCL cases).

Metric names use the group prefix, for example
`swe_public_geometric_mean_latency_seconds`. Unavailable timing aggregates are
omitted rather than reported as zero. A failed candidate service has no timing
aggregate and has zero correct cases. The same candidate watchdog timing bounds
used for scoring apply to model-caused case failures.

`metrics.beat_spec_progress` contains `passed`, `failed`, `total`,
`pass_percent`, and `fail_percent` for each of `paired_timing`, `p90`, `p95`,
`p99`, and `correctness`, aggregated across the two public workload groups.
Each metric has total 2, so there are 10 checks. These counts describe the
existing PASS checks; they do not change the scoring or PASS rules. Incomplete
comparison evidence produces no progress object. No per-group PASS decisions,
reference measurements, thresholds derived from measurements, margins, hidden
measurements, case identities, or case-level decisions are returned.

Saving or training a patch does not substitute for its independent final
evaluation.
