# E-Graph Invariant Restoration

`frontier-submit train` returns immediately after acceptance with a verification
ID and the authoritative count of remaining train calls. It does not wait for
verification. Use `frontier-submit status <verification_id>` for one status read;
once verification completes, it includes your score and task-approved development
`metrics`.

The immutable source tree is at `/frontier/task/repo`. Read `/frontier/task/repo/README.md`, then prepare a writable Git copy:

```sh
/frontier/task/prepare_workspace.sh
```

Edit only the source paths listed in the scientific statement; `src/lib.rs` is source-locked and must remain byte-identical. Package the patch with:

```sh
/frontier/task/make_submission.sh
```

This writes `/workspace/payload/solution.patch`. Train and final use an uncompressed POSIX tar containing exactly that nonempty regular file under the top-level `payload/` directory. The complete archive is limited to 1048576 bytes; the patch inside remains limited to 262144 bytes.

```sh
cd /workspace/repo
/frontier/task/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 -cf /workspace/train.tar payload
frontier-submit train /workspace/train.tar
```

There are 100 train calls available (accepted calls consume this quota). Completed train feedback includes the independent continuous score, `candidate_dev_wall_clock_seconds_median`, `candidate_dev_congruence_seconds_median`, and aggregate per-metric workload pass/fail progress. The `beat_spec_progress` object reports anonymous counts and percentages for each of the `wall_clock_seconds` and `congruence_seconds` criteria: `passed`, `failed`, `total`, `pass_percent`, and `fail_percent`. It withholds workload identities, reference measurements, margins, samples, hashes, and all final metrics.

Save a chosen archive without ending work using `frontier-submit store-final /workspace/train.tar`. A successful replacement supersedes the saved bytes; a failed replacement preserves them. Saving does not evaluate, consume train, or end answering, and the protocol has no save-count limit.

For explicit final, create the same archive as `/workspace/final.tar`, run `frontier-submit final /workspace/final.tar` exactly once, and exit. Final independently re-evaluates the three public workloads together with the three hidden graded workloads after Agent cleanup; it never reuses train measurements. See `/frontier/task/bin/README.md` for the complete lifecycle and budgets.
