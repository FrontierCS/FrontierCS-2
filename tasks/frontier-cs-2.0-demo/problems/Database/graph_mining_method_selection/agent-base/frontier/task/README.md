# Graph Mining Method Selection

Saving has no save-count quota. Payload, retained-data, audit and Agent-time
limits still apply. The save receipt reports `remaining_checkpoint_submissions`
as `null` for this contract. Check the accepted receipt; saving does not
extend answering time or select an answer automatically.

## Answering time

Formal Agent answering is **18000 seconds (five hours)**, including bootstrap
and train waits. Final judging starts after Agent and train cleanup and has its
own **900-second** budget; it does not consume the answering window.
Train-call, cumulative train-wall and GPU quotas are separate upper bounds,
not extra answering time.

The read-only source tree is at `/frontier/task/repo`. Copy it into your
workspace and make it a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `performance_model/src/main.py` there, then:

```sh
/frontier/task/make_submission.sh            # -> /workspace/payload/solution.patch
```

Nothing outside `/workspace` is writable.

The task statement is at `/frontier/task/repo/README.md`. Read it first: it
defines the problem, the workload, and how you are scored. This file covers only
where things are and how to submit.

## Train and final tar format

Train and final use the same uncompressed POSIX tar format. The archive must
contain one top-level `payload/` directory holding exactly `payload/solution.patch`. That
file is the entrypoint and must be non-empty. The complete archive must be at
most 65,536 bytes. Member paths must be relative and must not contain
`..`; links, devices and FIFO members are rejected.

```sh
cd /workspace/repo && /frontier/task/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/train.tar payload
frontier-submit train /workspace/train.tar
```

`frontier-submit train` returns immediately after acceptance with a verification
ID and the authoritative count of remaining train calls. It does not wait for
verification. Use `frontier-submit status <verification_id>` for one status read;
once verification completes, it includes your score and task-approved development
`metrics`. **100 train calls are available for the
whole run**, and every accepted request consumes one even when verification fails.
`metrics` includes the approved candidate measurements and
`beat_spec_progress`: per-metric workload pass/fail counts and percentages. It
does not return the evaluator's message, exact workload pass/fail identities, or
private reference measurements.

Save your first usable chosen answer while continuing to work:

```sh
frontier-submit store-final /workspace/final.tar
```

Build that tar using the final submission layout described above. After train
feedback, decide whether to replace your saved selection. Saving does not train,
end answering, or lock final. A successful replacement supersedes the previous
snapshot; a rejected replacement preserves it. Local edits do not change saved bytes.
Rejected saves do not replace the last successful selection.

An accepted explicit final takes priority over the saved selection. Without one,
normal exit, answering timeout, or a trusted process crash can collect the saved
answer. User cancellation and platform failure do not. After Agent/train cleanup,
the selected archive is judged independently with the final budget stated above.
Train submissions are never automatically selected as final.

When you are ready, build `/workspace/final.tar` with the same layout, run

```sh
frontier-submit final /workspace/final.tar
```

exactly once, and exit. The final score is produced after you exit; the command
does not print it.
