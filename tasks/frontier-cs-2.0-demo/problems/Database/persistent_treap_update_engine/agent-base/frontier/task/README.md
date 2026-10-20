# Persistent Treap Update Engine

## Optimization priority

First satisfy every required PASS condition on every workload and metric in the
scientific task statement. Optimize continuous score second; do not sacrifice
PASS for a higher score. Successful execution or a positive score is not PASS.
Use train pass/fail progress to guide experiments; final evaluates independently.

The read-only source tree is at `/frontier/task/repo`. Copy it into your
workspace and make it a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `lib/treap/scheduler.hpp`, `lib/treap/search_insert.hpp`, `lib/treap/search_delete.hpp` there, then:

```sh
/frontier/task/make_submission.sh            # -> /workspace/payload/solution.patch
```

Nothing outside `/workspace` is writable.

The task statement is at `/frontier/task/repo/README.md`. Read it first: it
defines the problem, the workload, and how you are scored. This file covers only
where things are and how to submit. Operational details, including saved-final
selection, are in `/frontier/task/bin/README.md`.

## Train and final tar format

Train and final use the same uncompressed POSIX tar format. The archive must
contain one top-level `payload/` directory holding exactly `payload/solution.patch`. That
file is the entrypoint and must be non-empty. The complete archive must be at
most 262,144 bytes. Member paths must be relative and must not contain
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
For this task, `metrics` includes only `beat_spec_progress`. Its
`case_throughput_ktps`, `peak_memory_mib`, and `historical_query_us` entries each
contain `passed`, `failed`, and `total` workload-pair counts, and `pass_percent`
and `fail_percent` percentages. These are aggregate comparison results, not
throughput, memory, or query-time measurements. Feedback does not return the
evaluator's message, exact workload pass/fail identities, or private reference
measurements.

You may preserve a chosen archive while continuing to work:

```sh
frontier-submit store-final /workspace/train.tar
```

A successful replacement supersedes the prior saved archive; a rejected save
preserves it. Saving neither trains nor ends answering. Local edits do not alter
the Driver-owned saved bytes. The protocol has no save-count quota.

When you are ready, build `/workspace/final.tar` with the same layout, run

```sh
frontier-submit final /workspace/final.tar
```

exactly once, and exit. The final score is produced after you exit; the command
does not print it. Explicit final has priority over a saved archive. Without an
explicit final, normal exit, answering timeout, or a confirmed process crash may
collect the saved archive. User cancellation and platform failure do not. A
train result is never selected automatically.
