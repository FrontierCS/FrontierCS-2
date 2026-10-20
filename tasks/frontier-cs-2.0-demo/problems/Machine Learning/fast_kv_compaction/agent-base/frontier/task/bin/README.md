# Testbed instructions

Saving has no save-count quota. Payload, retained-data, audit and Agent-time
limits still apply. The save receipt reports `remaining_checkpoint_submissions`
as `null` for this contract. Check the accepted receipt; saving does not
extend answering time or select an answer automatically.

You must save your first usable chosen answer with `frontier-submit store-final`
and confirm the accepted receipt while continuing to work. You choose which
answer to save; after train feedback, decide whether to replace that selection.
Train never automatically selects the best or latest submission as final.

Read `/frontier/task/repo/README.md` for the scientific task and
`/frontier/task/README.md` for the exact archive format and GPU command examples.

## Answering and submission

Formal answering lasts 5 hours (18000 seconds), including train waits.
Independent final has another 3600 seconds after answering ends. Short trials
use a separate version and stated answering budget. No fixed early-stop schedule
applies. Follow the actual Run instructions and authoritative remaining quotas.

Prepare the workspace with `/frontier/task/bin/prepare_workspace.sh`.
`/frontier/task/bin/make_submission.sh` creates `/workspace/payload/solution.patch`;
it stages existing edits and newly created files, checks that every change is
under `compaction/` and that at most 40 files changed, and does not package or submit
it. The patch size limit is 262144 bytes; the separately packaged complete tar
must also fit the original 262144-byte submission limit. Train, store-final and explicit final all use
uncompressed POSIX tar, at most 262144 bytes, with `payload/solution.patch` as
the entrypoint. Follow the outer README's packaging command.

Submit an early validated patch with `frontier-submit train /workspace/train.tar`
and use feedback after substantive progress. There are 8 accepted train calls;
failed evaluations still consume accepted calls. There is no mandatory call
count or automatic selection of the best train result.

Use `frontier-submit store-final /workspace/final.tar` to save your chosen answer.
Check its accepted receipt. You may continue working and replace the selection;
only the latest accepted stored archive is selected. After train feedback,
decide whether to update it. Editing the local archive without store-final does
not update the selection; a failed save does not replace a prior accepted save.

When finished, `frontier-submit final /workspace/final.tar` explicitly commits
your final. Use it once and exit after acceptance. It takes precedence over a
stored selection. Evaluation occurs independently after answering ends, and
the command does not return its score.

Without an accepted explicit final, normal exit, answering timeout or an Agent
crash permits collection of the last accepted stored archive. User cancellation
and platform faults do not. Neither train results nor an unsent local archive
substitute for an accepted selected answer.

## GPU and limits

Read `/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` and use the
Candidate-owned `/opt/frontier/candidate/bin/frontier-gpu` (GPU helper v3).
CPU and GPU workspaces are separate: push/pull transfers files. Prefix GPU task
commands with `/opt/frontier/task-runtime` for the offline model/toolchain
environment. Reuse a runner when useful, then delete it and confirm termination.
Do not call Modal or Driver endpoints yourself.

- Agent CPU: 4 vCPUs, 16 GiB memory, 10 GiB writable workspace.
- Agent GPU: one H100 at a time, 18000 seconds per allocation, 18000 GPU seconds
  total. Answering time remains a separate limit.
- Agent SSH: 1024 cumulative sessions. Each accepted push, pull, exec or ssh
  consumes one, even if it later fails. Reusing a runner does not reset it.
- Verifier SSH: 64 independent train sessions and 16 final sessions. Each
  invocation uses one upload and one execution, without automatic upload retries.
- SSH traffic: 512 MiB (536870912 bytes) per session per direction, including
  tar overhead. Stdout and stderr share the GPU-to-Agent direction. Transfer
  only needed files and handle rejected transfers as failed commands.
- Model requests: 1024; each request is limited to 64 MiB and each response to
  8 MiB. A local tool call is not itself a model API request.
- Whole-Run audit capture: 32 GiB, independent from the per-stream limits.

`frontier-gpu quota` consumes no SSH session and reports the Agent allowance;
private Verifier usage is not exposed. Exhausting debugging SSH does not itself
forbid CPU editing or submissions within their independent limits.
