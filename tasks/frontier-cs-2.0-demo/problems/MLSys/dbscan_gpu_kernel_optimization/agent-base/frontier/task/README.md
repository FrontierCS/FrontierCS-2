# GPU DBSCAN Kernel Optimization

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

Read `/frontier/task/bin/README.md` for train feedback, replaceable stored final,
explicit final, time budgets and resource limits. The archive layout below applies
to train, store-final and explicit final alike.

The read-only starter is at `/frontier/task/repo`. Prepare a writable Git copy:

```sh
/frontier/task/bin/prepare_workspace.sh       # -> /workspace/repo
```

Read `/workspace/repo/README.md`, then edit only Python files below
`/workspace/repo/dbscanlib/`.

New local regression-test modules such as `dbscanlib/test_regress.py` are allowed
and included in the patch. Files named `refdbscan.py` or `conftest.py` are reserved
at every depth; trusted files outside `dbscanlib/` remain read-only.

Generate the payload with:

```sh
/frontier/task/bin/make_submission.sh         # -> /workspace/payload/solution.patch
```

The formal answering window is 18,000 seconds (5 hours). Independent final
evaluation has a separate 2,400-second (40-minute) budget.

## Train and final

Train evaluates two public shapes. Final independently measures six shapes:
the same two public shapes with the same train seeds, two original hidden shapes
and two beat-spec-v3 hidden shapes (200,000 points; `min_samples=20` with noise)
with independent seeds. Final score covers the original four; PASS covers all six. See the problem
README for the train shapes and unchanged clustering/quality contract.

Train and final use the same uncompressed POSIX tar. It contains exactly the
regular file `payload/solution.patch` and may also contain its top-level
`payload/` directory entry. The patch must be non-empty and at most 2,000,000
bytes; the complete archive must be at most 2,097,152 bytes. Links, devices,
FIFOs, absolute paths, parent traversal, duplicate members and extra files are
rejected.

```sh
/frontier/task/bin/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/train.tar payload/solution.patch
frontier-submit train /workspace/train.tar
```

The Run permits 25 train calls. Each accepted call consumes
one, including a failed verification. Completed train feedback includes the candidate's absolute
measurements, independent geometric-mean speedup over the naive starter, safe
aggregate reference progress, and the authoritative remaining-call count. It
withholds reference measurements and workload-level decisions.

Build and save your chosen final patch while continuing to iterate:

```sh
/frontier/task/bin/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/final.tar payload/solution.patch
frontier-submit store-final /workspace/final.tar
```

A successful `stored` receipt replaces the saved selection without ending your
work. After train feedback, decide whether to rebuild and update it. Local file
edits alone do not update it. The Driver collects it on normal exit, answer
timeout or Agent process crash; cancellation and platform failures are excluded.
The last train payload is never selected automatically.

When done, `frontier-submit final /workspace/final.tar` still explicitly locks
that archive and overrides the saved selection. After acceptance, stop platform
calls and exit. Independent final evaluation has its own 2,400-second budget.

## GPU development

This task uses Candidate-owned `frontier-gpu` v3 and a reusable Driver-managed
H100. See `/frontier/task/bin/README.md` for commands, separate CPU/GPU
workspaces and the Agent/train/final quotas. GPU development is diagnostic;
final scoring always uses the independent Verifier.
