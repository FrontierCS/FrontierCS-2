# ParButterfly

`frontier-submit train` returns immediately after acceptance with a verification
ID and the authoritative count of remaining train calls. It does not wait for
verification. Use `frontier-submit status <verification_id>` for one status read;
once verification completes, it includes your score and task-approved development
`metrics`.

The immutable source tree is at `/frontier/task/repo`. Read `/frontier/task/repo/README.md`, then prepare a writable Git copy:

```sh
/frontier/task/prepare_workspace.sh
```

Edit only the seven butterfly headers listed in the scientific statement and files below `apps/hyper/solver_extensions/`. Package the patch with:

```sh
/frontier/task/make_submission.sh
```

This writes `/workspace/payload/solution.patch`. Train and final use an uncompressed POSIX tar containing exactly that nonempty regular file under the top-level `payload/` directory. The source patch is limited to 524288 bytes; the complete tar archive is limited to 1048576 bytes.

```sh
cd /workspace/repo
/frontier/task/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 -cf /workspace/train.tar payload
frontier-submit train /workspace/train.tar
```

There are 100 train calls available for accepted development submissions. Completed train feedback includes the independent continuous score, candidate-only timing summaries `dev_counting_seconds_median` and `dev_counting_seconds_p90`, and `beat_spec_progress` aggregates for each metric with fields `passed`, `failed`, `total`, `pass_percent`, and `fail_percent`. It withholds workload identities, reference measurements, margins, samples, hashes, and all final metrics.

Save a chosen archive without ending work using `frontier-submit store-final /workspace/train.tar`. A successful replacement supersedes the saved bytes; a failed replacement preserves them. Saving does not evaluate, consume train, or end answering, and the protocol has no save-count limit.

For explicit final, create the same archive as `/workspace/final.tar`, run `frontier-submit final /workspace/final.tar` exactly once, and exit. Final independently evaluates all public workloads together with the hidden subset after Agent cleanup; it never reuses train measurements. See `/frontier/task/bin/README.md` for the complete lifecycle and budgets.

You must save the final answer you select with `frontier-submit store-final /workspace/train.tar`.
You decide which answer to save and whether to replace that selection.
Confirm its accepted receipt; local edits and train feedback do not update saved bytes.
Saving does not evaluate, end answering, or lock final. There is no save-count quota;
`remaining_checkpoint_submissions` is `null`. Payload, audit and Agent-time limits still apply.
