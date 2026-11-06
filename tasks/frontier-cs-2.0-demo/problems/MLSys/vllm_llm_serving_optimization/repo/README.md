# vLLM serving and scheduling optimization

Make vLLM 0.11.0 serve multi-turn agent conversations faster. You patch vLLM's Python
serving and scheduling code; the judge serves Qwen/Qwen3-Coder-30B-A3B-Instruct on one H100
with your patched vLLM and replays SWE-bench and BFCL conversations against it. Lower
end-to-end conversation latency is better; a correctness drop beyond 5% reduces the score. A
valid submission is a text patch of allowlisted Python files that applies cleanly to the
packaged vLLM source.

## Task

`/frontier/task/repo` is the clean vLLM 0.11.0 checkout (upstream commit
`b8b302cde434df8c9289a2b465406b47ebab1c2d`); your patch is applied to this source.
Two fixed baselines:

- **Unmodified vLLM** ("vanilla") is the baseline of the continuous score.
- **The optimized reference** is the original task's job-aware scheduling patch. It defines
  "beats the reference" and is independent of the vanilla baseline.

Both are fixed private measurements, three captures each per phase (train and final have
their own). An evaluation runs only your candidate server and compares it with them; it never
reruns the baselines.

## What you may change

Only text changes to `.py`/`.pyi` files in this allowlist (new `.py`/`.pyi` files in it are
allowed):

- `vllm/v1/core/**` (including `vllm/v1/core/sched/**` and `vllm/v1/core/kv_cache_utils.py`)
- `vllm/config/scheduler.py`, `vllm/config/cache.py`
- `vllm/v1/worker/**`, `vllm/v1/engine/**`, `vllm/v1/executor/**`
- `vllm/v1/request.py`, `vllm/v1/outputs.py`, `vllm/v1/serial_utils.py`
- `vllm/entrypoints/openai/protocol.py`, `serving_engine.py`, `serving_chat.py`,
  `serving_completion.py`
- `vllm/sampling_params.py`

Everything else stays fixed: native kernels, model implementations and weights, build and
dependency files, tests, benchmarks, datasets and the evaluation workload. Deleting a file and
binary hunks are rejected. Preserve generation parameters and tool/model semantics.

Do not special-case the workload: no hardcoded benchmark knowledge, and no detection of the
judge, its instances, inputs, arrival schedule or phase. Added lines must not contain these
benchmark tokens (any case):

```text
swebench  swe-bench  swe_bench  sweb.eval  minisweagent  mini-swe  princeton-nlp  SWE-bench_Verified
```

nor these judge/credential tokens (exact case):

```text
MODAL_TOKEN  MODAL_TOKEN_ID  MODAL_TOKEN_SECRET  HF_TOKEN  HUGGING_FACE_HUB_TOKEN  HUGGINGFACEHUB
FRONTIER_  HARBOR_  JUDGE_URL  RUN_OUTPUT_DIR  scheduler_timestamps
```

## Interface

The submission is a unified diff relative to the checkout root (paths such as
`vllm/v1/core/sched/scheduler.py`), made with `git diff` of your working copy. An empty patch is valid and evaluates unmodified vLLM.

The judge builds vLLM from your patched Python sources on top of the precompiled CUDA kernels
and starts its OpenAI-compatible server with the fixed model and
serving parameters. Clients then run whole conversations against it:

- **SWE-bench:** a software-engineering agent alternates model requests with tool commands in
  the instance's test sandbox (a container holding the instance's repository) and ends by
  extracting its patch.
- **BFCL:** memory/tool-use conversations from BFCL v4.

Your server sees only the model requests; what it may change is how they are scheduled,
batched, cached and served.

## Correctness

- SWE-bench: correct only if the original issue-specific and regression tests pass on the
  extracted patch. BFCL: the original answer and tool-call checks. Missing, partial, timed-out,
  unusable or malformed case results are incorrect; the other cases continue.
- A normally completed wrong answer keeps its measured latency. BFCL `force_quit` and SWE
  `limit_with_patch` from the original step limit can still be complete results; a shell
  command timing out inside a conversation is tool feedback.
- A missing or unusable case result, or a single-case error of your server, gets `r_i = 1/8`;
  if your server fails a whole load group (fails to start, crashes), that benchmark (SWE or
  BFCL) scores 0. Either way the group fails beats-the-reference. The scoring watchdog time
  bounds apply to model-caused case failures.
- Baseline, provider-authentication, test-sandbox and platform failures (a whole-group
  scheduler or test-sandbox outage counts; independent per-case failures do not) are unscored
  infrastructure errors.
- A patch that breaks the file and token rules above or the size limits in
  `/frontier/task/README.md`, or does not apply, is rejected and scores 0.

## Workload

One workload is one benchmark (SWE or BFCL) at one load group. Conversations arrive on the
original fixed arrival schedule; SWE and BFCL run as separate sequential workloads, not as a
mixed request stream.

| Phase | SWE-bench | BFCL | Load groups |
|---|---|---|---|
| Train | 5 public instances at 0.5 jobs/s | 8 public instances at 1 job/s | public |
| Final | all 48 instances (the 5 public ones included) | all 155 instances (the 8 public ones included) | public, and a separate hidden load |

Hidden arrival rates, seeds and instance details are never exposed. Test sandboxes are
prepared before the arrival schedule starts; their setup time is recorded separately and not
counted. A conversation's latency covers its model requests, its interactive tool calls and the
final patch extraction, timed end to end on one host (not from server-side token timings).

Vanilla timing per instance is the median of its three captures; an instance is
vanilla-correct if at least two of its three captures answered it correctly.

## Scoring

For each expected instance i of a load group, `r_i = vanilla_latency_i / candidate_latency_i`,
used exactly (no clipping). A missing or unusable result attributable to the candidate gets
`r_i = 1/8`; the instance count never shrinks.

```text
S_group     = exp(mean(log r_i))                   over the group's expected instances
S_workload  = sqrt(S_public * S_hidden)            final;  train: S_public
L(S)        = 0                                    if S <= 1
            = 100 * log(S) / (log(10) + log(S))    if S > 1
A_v, A_c    = pooled correct fraction (public and hidden runs together) of vanilla, candidate
absolute_drop = max(0, A_v - A_c);   relative_drop = absolute_drop / max(A_v, 1e-9)
quality     = 1 if absolute_drop <= 0.05 or relative_drop <= 0.05, else 0.05 / relative_drop
workload_score = L(S_workload) * quality
score       = 0.8 * SWE_score + 0.2 * BFCL_score   (0-100)
```

Both workloads are required; a missing BFCL workload does not renormalize to SWE only. Anchors:
vanilla speed (S <= 1) scores 0, a 10x speedup 50 per workload; 100 is the limit as S grows. The tail latencies are not part of this
score.

**Beats the reference** requires, in every load group (train: SWE and BFCL public; final: SWE
and BFCL × public and hidden), complete usable candidate results for every instance and all
five comparisons against the optimized reference:

```text
ref_i       = fastest of the three reference captures of instance i
(1)         mean(log(ref_i) - log(cand_i)) > 0
(2-4)       candidate p90, p95, p99 < min over the three captures of that group percentile
            (targets may come from different captures; not the tails of ref_i)
(5)         C >= median(R1, R2, R3) - max(1, ceil(0.05 * N))
            C: candidate correct count; R: captures' group correct totals; N: instances in the group
percentile  nearest rank: sorted N latencies, item ceil(p * N) (one-based), no interpolation
```

Timing equality fails, correct-count equality passes; no added margin. A failed check cannot be
compensated by another group or metric; public and hidden correctness are not pooled here.

## Running locally

`/frontier/task/bin/prepare_workspace.sh` copies the checkout to `/workspace/repo`;
`/frontier/task/bin/make_submission.sh` writes `payload/solution.patch` and refuses a patch
outside the editable files. To try a server, push your copy to the H100 runner (see
`/frontier/task/README.md`) and run it there.
