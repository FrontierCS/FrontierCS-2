# Testbed guide

Saving has no save-count quota. Payload, retained-data, audit and Agent-time
limits still apply. The save receipt reports `remaining_checkpoint_submissions`
as `null` for this contract. Check the accepted receipt; saving does not
extend answering time or select an answer automatically.

You must save your first usable chosen answer with `frontier-submit store-final`
and confirm the accepted receipt while continuing to work. You choose which
answer to save; after train feedback, decide whether to replace that selection.
Train never automatically selects the best or latest submission as final.

The Candidate supplies `frontier-submit`; the task supplies workspace and patch
helpers. Neither helper uploads data. The [operational entry](../README.md)
defines the identical train/store-final/final tar format and examples.

The formal answering budget is 18,000 seconds (5 hours). An explicitly published
short debugging version may instead allow 1,800 seconds; use the actual Run
configuration and Candidate budget information. Independent final has its own
900-second budget, beginning after Agent/train cleanup at verifier provisioning.
Do not infer the final budget from remaining answering time.

There are 100 train calls, each with the original 240-second verifier ceiling,
and 24,000 aggregate train seconds. Every accepted train request consumes a call,
including a request whose verification fails. Completed train feedback includes the task score,
`score_unbounded`, the authoritative remaining call count, and the aggregate
pass/fail counts described in `/frontier/task/README.md`. This task exposes
no per-query identities, reference values, margins, or per-query comparison decisions.
Final metrics and evidence are operator-only; the CLI does not print final score.

After train feedback, consider whether the current answer should replace your
saved selection. `frontier-submit store-final /absolute/path/final.tar` snapshots
one archive; a successful replacement supersedes the previous save, while a
failed replacement preserves it. Saving does not run verification, consume a
train call, stop answering, or lock explicit final. Editing the local file does
not change the saved copy. Keep and check the returned receipt.

Rejected updates preserve the last accepted selection. Payload and audit limits still apply. An accepted explicit
`frontier-submit final` has priority. Without explicit final, a saved copy is
collected on normal exit, answering timeout, or process crash confirmed by trusted
container state. User cancellation and platform failure do not collect it;
Driver Pod failure recovery is not promised. With no selected archive, there is
no final submission. A train score or last train archive is never promoted.

The CPU Agent and Verifier each have 2 vCPUs and 8 GiB RAM; Agent workspace is
1 GiB. The reproduced executor is single-threaded. Python 3.11, Git and tar are
provided. The fixed data/executor and hidden evaluator are Verifier-only. There
is no GPU allocation, GPU helper requirement, or SSH transport for this task.
The 30-second plan-construction and 20-second conversion subprocess limits remain
unchanged. Each cumulative engine invocation has a 600-second limit.

Each submitted archive is limited to 65,536 bytes. Verifier evidence is limited
to 4 MiB per invocation. Total audit storage is 14 GiB. The model transport allows
1024 sessions, with at most 1 MiB per request and 8 MiB per response; model logs
also depend on the platform's deployed 4 GiB log allowance. These are separate
limits from workspace, train calls, and answering time. No external network
domains are granted by the task.
