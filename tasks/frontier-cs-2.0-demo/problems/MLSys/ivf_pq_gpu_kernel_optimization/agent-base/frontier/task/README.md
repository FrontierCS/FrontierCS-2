# GPU IVF-PQ Kernel Optimization

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
`/workspace/repo/ivfpqlib/`.

New local regression-test modules such as `ivfpqlib/test_regress.py` are allowed
and included in the patch. Files named `refivfpq.py` or `conftest.py` are reserved
at every depth; trusted files outside `ivfpqlib/` remain read-only.

Generate the payload with:

```sh
/frontier/task/bin/make_submission.sh         # -> /workspace/payload/solution.patch
```

## Train and final

The formal answering window is 18,000 seconds (5 hours). Train waits and GPU
development consume that window; independent final evaluation is timed separately.

Train evaluates four public shapes. Final independently measures eight shapes:
the same four public shapes with the same train seeds, two original hidden shapes
with independent seeds, and two beat-spec-v3 hidden shapes (up to the full
10,000-query test set; k up to 1024). Final score covers the original six; PASS
covers all eight. See the problem README for the train shapes and the
search/recall contract.

Train and final use the same uncompressed POSIX tar. It contains exactly the
regular file `payload/solution.patch` and may also contain its top-level
`payload/` directory entry. An empty patch evaluates the unchanged naive starter. The patch may contain
at most 2,000,000 bytes; the complete archive must be at most 2,097,152 bytes. Links, devices,
FIFOs, absolute paths, parent traversal, duplicate members and extra files are
rejected.
Git rename/copy patch operations are unsupported; ordinary Python additions,
including empty `__init__.py` files, remain allowed.

```sh
/frontier/task/bin/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/train.tar payload/solution.patch
frontier-submit train /workspace/train.tar
```

There are 25 train calls for the Run. Each accepted call consumes
one, including a failed verification. Completed train feedback includes the candidate's absolute
measurements, independent geometric-mean speedup over the naive starter, safe
aggregate reference progress, and the authoritative remaining-call count. It
withholds reference measurements and workload-level decisions.

The public metrics are `geomean_speedup` (geometric mean of per-workload
median paired speedups over the naive starter), `search_latency_ms`
(independently sorted candidate median latencies), `iso_recall`
(independently sorted candidate worst timed iso-result recalls), and `nn_recall`
(independently sorted candidate worst timed nearest-neighbor recall@k). The
sorted lists have no workload labels and do not establish paired workload results.
For each comparison metric, `beat_spec_progress.<metric>` reports `passed`,
`failed`, `total`, `pass_percent`, and `fail_percent` over the evaluated
workloads. These are aggregate pass/fail counts and percentages, not reference
measurements or per-workload verdicts. Train progress covers the four public
workloads; final independently evaluates all eight.

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
calls and exit. Independent final evaluation has its own 14,400-second (4-hour) budget.

## GPU development

This task uses Candidate-owned `frontier-gpu` v3 and a reusable Driver-managed
H100. See `/frontier/task/bin/README.md` for commands, separate CPU/GPU
workspaces and the Agent/train/final quotas. GPU development is diagnostic;
final scoring always uses the independent Verifier.
