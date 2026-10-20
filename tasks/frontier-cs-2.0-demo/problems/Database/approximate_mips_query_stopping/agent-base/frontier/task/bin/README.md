# Testbed usage

Science and scoring are in [the task statement](../repo/README.md).
[The operational entry](../README.md) gives the exact tar command and layout.

`prepare_workspace.sh` copies the source to `/workspace/repo` and records its Git
baseline. Edit only `src/alg.cpp`. `make_submission.sh` writes a nonempty textual
patch at `/workspace/payload/solution.patch`; package it in a tar yourself.
Train, store-final and final use identical `payload/solution.patch` tar entries
and a 65,536-byte complete-archive limit. The patch limit is also 65,536 bytes;
archive framing means a maximum-size patch may not fit. The payload is data,
not an executable. All your writes belong below `/workspace`.

`frontier-submit train /workspace/answer.tar` accepts development verification
and returns immediately with its verification ID and remaining train-call count.
Use `frontier-submit status <verification_id>` for a single status read and,
when verification completes, the score and approved metrics. This native
source exposes `score_unbounded` only; mean measurements and private reference
values remain in operator evidence. Accepted requests consume a train call even
if verification fails. Train feedback never chooses the final automatically.

`frontier-submit store-final /workspace/answer.tar` saves one Driver-owned
snapshot and returns a receipt. Saving neither runs verification, spends a train
call, ends answering nor locks final. Successful replacement supersedes the old
copy; failed replacement preserves it. Editing the local archive does not alter
the saved bytes. After train feedback, decide whether to save your current
selection. Keep receipts and check whether each operation succeeded. The agreed
protocol has no save-count quota; archive, retained-payload and audit capacity
limits still apply. Saving has no save-count quota. The save receipt reports
`remaining_checkpoint_submissions` as `null`; check the accepted receipt.
Saving does not extend answering time.

An accepted `frontier-submit final /workspace/answer.tar` is irreversible and
has priority over any saved copy. Submit it once when finished, then exit.
Without an accepted explicit final, normal exit, answering timeout or a process
crash confirmed by trusted container state collects a successfully saved copy.
User cancellation and platform failure do not collect it. Driver Pod failure
recovery is not promised. Without either selection, no train is promoted to
final. Driver stops Agent/train resources and confirms cleanup before starting
an independent final; cleanup failure remains a runtime failure.

Formal answering: 18,000 seconds, including startup and train waits. Independent
final: 900 seconds from the cleanup barrier, including verifier provisioning.
A separately published short debugging version uses 1,800 answering seconds and
the same independent final budget. Use the actual Run timing supplied by the
Candidate; the formal source is not a remaining-time display.

The preserved C++ workload requires AVX2 and AVX-512F/DQ/BW CPU support.
CPU resources are 4 CPUs and 4 GiB memory for Agent and Verifier, with a 4 GiB
Agent workspace. There is no GPU or SSH access. Development permits 100 calls
and 90,000 cumulative verifier seconds (100 times the source 900-second timeout),
also bounded by the answering clock. The audit allows 1024 model sessions,
1 MiB per model request, 8 MiB per response and 8 GiB total audit bytes. The
schema's dormant SSH fields grant no SSH permission on this CPU-only task.
Verifier evidence is bounded to 4 MiB. The separate platform model-log allowance
requires deployed 4 GiB support; it is included within the total 8 GiB audit.
Keep outputs bounded. Submission, workspace, model and audit limits are distinct.
