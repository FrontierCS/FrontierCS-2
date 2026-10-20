# Testbed operations

Read `/frontier/task/README.md` for the research task and scoring rules. Use the installed Candidate `frontier-submit` and
`frontier-gpu` skills; work under `/workspace`.

## Workspace and archive preparation


Prepare the packaged clean vLLM checkout, retaining its Git history:

```sh
/frontier/task/bin/prepare_workspace.sh
```

The writable checkout is `/workspace/repo`. Its original base is recorded when
prepared, so local commits made while optimizing are included in your patch.
Generate a payload after editing:

```sh
/frontier/task/bin/make_submission.sh
```

This returns `/workspace/payload/solution.patch`. It only creates the patch;
you create the submission tar separately. The empty patch is a valid request
to evaluate unmodified vLLM. The patch may be at most 1,500,000 bytes and change
at most 80 files. It must apply cleanly to the packaged original source.

Only text changes to allowlisted `.py`/`.pyi` source are permitted: vLLM v1 core,
worker, engine and executor code; scheduler/cache configuration; v1 request,
outputs and serialization; sampling parameters; and the explicitly permitted
OpenAI serving/protocol modules. Binary patches, source-file deletions, changes
to model implementation/weights, native kernels, build files, tests and benchmark
code are rejected. The independent checker applies the original task's exact
path and added-code policy.

## Train and final archives

All three submission operations use the same uncompressed POSIX tar entry:
`payload/solution.patch`, a regular file. Build it explicitly:

```sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/train.tar payload/solution.patch
frontier-submit train /workspace/train.tar
```

## Train and selected final

Generate `payload/solution.patch` with `make_submission.sh`, then build the
uncompressed tar as described above. Train requests use:

```sh
frontier-submit train /workspace/train.tar
```

An accepted request consumes one train call even if verification fails. Inspect
feedback and the authoritative remaining-call count. Training does not choose
the final answer and does not produce its independent score.

Build `/workspace/final.tar` with the same `payload/solution.patch` entry and save
the answer you currently choose:

```sh
frontier-submit store-final /workspace/final.tar
```

Wait for a successful `stored` receipt. Driver keeps only one selection; a
successful update replaces it, while a failed update preserves the previous
selection. Saving does not consume train calls or start evaluation. Archive and
audit limits still apply. After train feedback, decide whether to rebuild and
save an improved selected answer. Editing a local file alone does not update
Driver's saved archive, and the last train is never selected automatically.

On normal exit, answer timeout or Agent process crash, Driver uses the saved
selection if no explicit final was accepted. User cancellation and platform
failures are excluded. Without a saved or accepted explicit final, exiting or
timing out does not produce a final score.

## Finish explicitly

```sh
frontier-submit final /workspace/final.tar
```

An accepted explicit final locks that archive and takes priority over a saved
selection. It cannot be replaced. After acceptance, stop platform calls and
exit. Driver stops Agent/train resources before independent final evaluation;
its result arrives after evaluation, not in the submission receipt.

## Time, quotas and GPU use

The formal answering budget is 18,000 seconds (5 hours), with 25 accepted train
calls. A short trial uses its actual shorter answering window. Build time,
local work and train waits consume answering time; independent final has a
separate configured budget. Do not infer the current remaining time from the
formal default. Follow the Run's authoritative time and quota feedback.

The protocol is Candidate-owned `frontier.gpu-helper/v3`. Read
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` and query
`frontier-gpu quota` before creating a GPU machine. Request one H100 within the
remaining lifetime allowance and reuse its returned runner ID. CPU/GPU
filesystems are separate; push and pull files explicitly.

This task has 512 cumulative accepted Agent SSH sessions under its reviewed
vLLM-specific allocation. Train and final use separate Verifier pools of 825
and 336 sessions in the published task configuration; they do not consume the
Agent pool. The same configuration allocates at most 256 model sessions, so its
1,673 SSH sessions plus 256 model sessions remain within the platform's 2,048
combined-session ceiling. Accepted push, pull, exec and interactive SSH
operations consume Agent sessions even if a command fails. Quota, status,
create and delete are not SSH sessions. Reusing SSH does not create another
Modal container. A tool may issue several SSH operations; local tool-call count
is not an SSH or model-request count. Idle GPU lifetime still consumes its
allowance; delete unused runners.

Use actual quota feedback for GPU time, train/final reserves, model calls and
byte limits. Counts alone do not guarantee available audit space. Provider and
cloud credentials belong to Driver. Local development is diagnostic; final
always evaluates its selected archive through the independent Verifier.
