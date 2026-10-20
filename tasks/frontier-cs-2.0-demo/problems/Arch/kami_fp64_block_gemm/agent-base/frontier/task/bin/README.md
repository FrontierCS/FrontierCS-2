# Testbed usage

Science and scoring are in [the task statement](../repo/README.md).
[The operational entry](../README.md) gives the exact tar command and layout.

`prepare_workspace.sh` copies the source to `/workspace/repo` and records its
Git baseline. Edit only the CUDA file and optional solver-extension tree named
in the statement. `make_submission.sh` writes a nonempty textual patch at
`/workspace/payload/solution.patch`; package it in a tar yourself. Train,
store-final, and final use identical `payload/solution.patch` tar entries and a
262,144-byte complete-archive limit. Archive framing means a maximum-size patch
may not fit. The payload is data, not an executable. All writes belong below
`/workspace`.

`frontier-submit train /workspace/answer.tar` accepts development verification
and returns immediately with its verification ID and the authoritative remaining
train-call count. Use `frontier-submit status <verification_id>` for one status
read and, when verification completes, the score and approved candidate
TFLOPS/progress. Accepted requests consume a train call even when
verification fails. Workload identities, reference values, ratios, margins, and
individual decisions remain operator-only. Train feedback never chooses final.

`frontier-submit store-final /workspace/answer.tar` saves one Driver-owned
snapshot and returns a receipt. Saving neither runs verification, spends a train
call, ends answering, nor locks final. Successful replacement supersedes the old
copy; failed replacement preserves it. Editing the local archive does not alter
saved bytes. After train feedback, decide whether to save the current selection.
You must save the final answer you select; you decide which answer to save
and whether to replace it. There is no save-count quota; the receipt's
`remaining_checkpoint_submissions` is `null`. Archive, retained-payload,
audit, and Agent-time limits still apply.

An accepted `frontier-submit final /workspace/answer.tar` is irreversible and
has priority over any saved copy. Submit it once when finished, then exit.
Without an accepted explicit final, normal exit, answering timeout, or a process
crash confirmed by trusted container state collects a successfully saved copy.
User cancellation and platform failure do not collect it. Driver Pod failure
recovery is not promised. Without either selection, no train is promoted.
Driver stops Agent/train resources and confirms cleanup before independent
final; cleanup failure remains a runtime failure.

Formal answering is 18,000 seconds, including startup and train waits.
Independent final is 1,800 seconds from the cleanup barrier, including GPU
Verifier provisioning. A separately published short POC uses 1,800 answering
seconds and the same independent final budget. Use the actual timing shown by the
Candidate for the selected TaskVersion; the formal source is not a countdown.

## Candidate-owned GPU v3

The CPU and GPU filesystems are independent. The composed Candidate supplies
`frontier-gpu`; this task supplies no provider client, bearer, private key, or
credential. Querying quota consumes no SSH session:

```sh
command -v frontier-gpu
frontier-gpu --help
frontier-gpu quota
```

Use one H100 runner for at most 900 seconds. Replace the example runner ID with
the ID returned by create:

```sh
frontier-gpu create --type H100 --count 1 --seconds 900
frontier-gpu push gpurunner_EXAMPLE /workspace/repo /workspace
frontier-gpu exec gpurunner_EXAMPLE -- \
  env PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  bash /workspace/repo/frontier-build.sh
frontier-gpu exec gpurunner_EXAMPLE -- \
  env PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  bash /workspace/repo/frontier-bench.sh
frontier-gpu pull gpurunner_EXAMPLE /workspace/repo /workspace/gpu-results
frontier-gpu delete gpurunner_EXAMPLE
frontier-gpu status gpurunner_EXAMPLE
```

Create/delete/quota/status do not consume SSH sessions. Each accepted
push/exec/pull/shell operation does, including an accepted command that fails.
The Agent pool is 1,024 sessions. Each session is bounded to 536,870,912 bytes
per direction; tar framing counts, and stdout plus stderr share the output
direction. Inspect quota before transfers, transfer only required files, and keep
debug output bounded. Splitting a transfer does not bypass the 32 GiB aggregate
audit limit. Confirm the runner is terminated after use.

## Resource and traffic limits

The Agent CPU role has 3 CPUs, 4 GiB memory, and a 4 GiB workspace. The Agent
GPU role has one exact NVIDIA H100 80GB HBM3, CUDA 12.8, at most one runner,
900 seconds per runner, and 900 GPU-seconds total. The trusted final/train
Verifier uses one exact H100 and up to 1,800 GPU-seconds per call. Its separately
funded SSH pools are 24 development and 2 final sessions; they cannot borrow
from the Agent pool. The aggregate SSH allocation is 1,050 sessions and the
deployed combined model-plus-SSH ceiling is 2,048.

The preserved task-specific development allowance is 12 calls, with 21,600
cumulative Verifier seconds, still bounded by the answering clock. The audit
allows 1,024 model sessions, 64 MiB per model request, 8 MiB per response, and
32 GiB total. The separate platform model-log allowance is 4 GiB within that
32 GiB total. Submission is 262,144 bytes and task evidence is 4 MiB. Submission,
workspace, GPU, model, SSH, evidence, and audit limits are independent.

Upstream plots/scripts include GH200 and RTX 5090 targets. This frozen task's
actual validation and runtime requirement is exact NVIDIA H100 80GB HBM3; do not
treat the upstream plotting targets as allowed TaskVersion GPU types.
