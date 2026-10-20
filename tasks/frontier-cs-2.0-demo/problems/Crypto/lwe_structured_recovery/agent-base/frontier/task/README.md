# Structured-LWE Public Witness Recovery

## Optimization priority

First satisfy every required PASS condition on every workload and metric in the
scientific task statement. Optimize continuous score second; do not sacrifice
PASS for a higher score. Successful execution or a positive score is not PASS.
Use train pass/fail progress to guide experiments; final evaluates independently.

Read `/frontier/task/bin/README.md` for train feedback, replaceable stored final,
explicit final, time budgets and resource limits. The archive layout below applies
to train, store-final and explicit final alike.

The read-only source tree is at `/frontier/task/repo`. Copy it into your
workspace and initialize the cumulative JSON ledger:

```sh
/frontier/task/bin/prepare_workspace.sh      # -> /workspace/repo
```

Write your solver under `/workspace/repo` and accumulate valid witnesses in
`/workspace/payload/solution.json`, then:

```sh
/frontier/task/bin/make_submission.sh        # -> /workspace/payload/solution.patch
```

Nothing outside `/workspace` is writable.

The task statement is at `/frontier/task/repo/README.md`. Read it first: it
defines the problem, the workload, and how you are scored. This file covers only
where things are and how to submit.

Like every task, this one is submitted as a patch. The answer is the ledger
`payload/solution.json`; the read-only starter ledger is
`/frontier/task/starter/payload/solution.json`. `make_submission.sh` writes
`/workspace/payload/solution.patch`, a unified diff from the starter to your
current ledger, and returns its path. The Verifier applies that patch to its own
copy of the same starter and scores the resulting ledger.

## Train and final tar format

Train and final use the same uncompressed POSIX tar format. The archive contains
exactly `payload/solution.patch`; an explicit top-level `payload/` directory member
is also accepted. The complete archive must be at most 4,194,304 bytes. Links,
devices, FIFO members, duplicate members and extra files are rejected.

The patch must be a text unified diff that changes only `payload/solution.json`
(file headers `a/payload/solution.json` and `b/payload/solution.json`), with
context that matches the starter exactly. Binary, rename, mode-change and
deletion records are rejected. An empty patch submits the starter ledger
unchanged. The ledger the patch produces must be non-empty and at most 2,000,000
bytes.

```sh
/frontier/task/bin/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/train.tar payload/solution.patch
frontier-submit train /workspace/train.tar
```

Package only the patch file, not the whole payload directory: the directory also
holds your ledger and possibly a sibling `solution.json.lock`.

`frontier-submit train` returns immediately after acceptance with a verification
ID and the authoritative count of remaining train calls. It does not wait for
verification. Use `frontier-submit status <verification_id>` for one status read;
once verification completes, it includes your score and task-approved development
`metrics`. **100 train calls are available
for the whole run**, and every accepted request consumes one even when verification
fails. `metrics` includes approved counts and `beat_spec_progress`: per-metric
workload pass/fail counts and percentages. It does not return exact workload
pass/fail identities or private reference measurements. Use the public checker
for local diagnosis, and batch new solutions as useful; there is no mandatory
submission after each discovery.

Build `/workspace/final.tar` with the same layout and save your current selection:

```sh
/frontier/task/bin/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/final.tar payload/solution.patch
frontier-submit store-final /workspace/final.tar
```

A successful `stored` receipt replaces the one saved ledger and lets you continue
working. After train feedback, decide whether to rebuild and update it; retain
all solved instances. Local file edits alone do not update the saved selection.
The Driver collects it on normal exit, answer timeout or Agent process crash;
cancellation and platform failures are excluded. The last train is not selected
automatically.

When done, `frontier-submit final /workspace/final.tar` still explicitly locks
that archive and overrides the saved selection. After acceptance, stop platform
calls and exit. Independent final evaluation has its own 120-second budget; the
submission command does not return the eventual final score.
