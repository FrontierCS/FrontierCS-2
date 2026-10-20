# Hypergraph Core Numbers

`frontier-submit train` returns immediately after acceptance with a verification
ID and the authoritative count of remaining train calls. It does not wait for
verification. Use `frontier-submit status <verification_id>` for one status read;
once verification completes, it includes your score and task-approved development
`metrics`.

## Optimization priority

First satisfy every required PASS condition on every workload and metric in the
scientific task statement. Optimize continuous score second; do not sacrifice
PASS for a higher score. Successful execution or a positive score is not PASS.
Use train pass/fail progress to guide experiments; final evaluates independently.

The immutable source tree is at `/frontier/task/repo`. Read `/frontier/task/repo/README.md`, then prepare a writable Git copy:

```sh
/frontier/task/prepare_workspace.sh
```

Edit only `apps/hyper/HyperKCore-Efficient.C` and files below `apps/hyper/frontier_solution/`. Package the patch with:

```sh
/frontier/task/make_submission.sh
```

This writes `/workspace/payload/solution.patch`. Train and final use an uncompressed POSIX tar containing exactly that nonempty regular file under the top-level `payload/` directory. The complete archive is limited to 262144 bytes.

```sh
cd /workspace/repo
/frontier/task/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 -cf /workspace/train.tar payload
frontier-submit train /workspace/train.tar
```

There are 100 train calls; each accepted request consumes one call. Completed train feedback includes the independent continuous score, `candidate_runtime_seconds_median`, `candidate_sequential_runtime_seconds_median`, and `beat_spec_progress`. For each of the `runtime_seconds` and `sequential_runtime_seconds` metrics, progress reports `passed`, `failed`, `total`, `pass_percent`, and `fail_percent` across the evaluated workloads. It withholds workload identities, reference measurements, margins, samples, hashes, and all final metrics.

Save a chosen archive without ending work using `frontier-submit store-final /workspace/train.tar`. A successful replacement supersedes the saved bytes; a failed replacement preserves them. Saving does not evaluate, consume train, or end answering, and the protocol has no save-count limit.

For explicit final, create the same archive as `/workspace/final.tar`, run `frontier-submit final /workspace/final.tar` exactly once, and exit. Final independently re-evaluates both feedback workloads together with the two hidden graded workloads after Agent cleanup; it never reuses train measurements. See `/frontier/task/bin/README.md` for the complete lifecycle and budgets.
