# Access-Pattern Rewriting

The read-only source tree is at `/frontier/task/repo`. Copy it into your
workspace and make it a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Implement `code/src/language/solver_extensions/access_analysis.rs` there. Generic rewrite optimizations in `code/src/language/rewrites.rs` and Rust helpers under `code/src/language/solver_extensions/` remain legal; fixed mapping functions and all other files are immutable. Then:

```sh
/frontier/task/make_submission.sh            # -> /workspace/payload/solution.patch
```

Nothing outside `/workspace` is writable.

The task statement is at `/frontier/task/repo/README.md`. Read it first: it
defines the problem, the workload, and how you are scored. This file covers only
where things are and how to submit. This distinct draft successor removes the compositional access-analysis core (`make` and `merge`) while supplying the four generic rewrite constructors. It keeps all 14 records per split, the 100-call budget, scoring comparisons, hardware and the 300-second evaluator/final limit unchanged. The five-hour successor uses 18000 seconds answering and a shared 12000-second train pool; the 100-call ceiling is subject to both clocks. The missing analysis is an explicit panic stub, not an executable fallback: it fails evaluation with score zero and empty task metrics, rather than reporting fourteen measured misses. The 132 upstream analysis-result tests are judge-only; the worked numerical example and public example program remain usable interface documentation. See the statement for the supported input domain and the limited rejection contract. Both ordinary and blocked systolic-array analysis forms are legal; complete output correctness remains required. Final data and the authentic restoration stay private.

## Train and final tar format

Train and final use the same uncompressed POSIX tar format. The archive must
contain one top-level `payload/` directory holding exactly `payload/solution.patch`. That
file is the entrypoint and must be non-empty. The complete archive must be at
most 1,048,576 bytes. Member paths must be relative and must not contain
`..`; links, devices and FIFO members are rejected.

```sh
cd /workspace/repo && /frontier/task/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/train.tar payload
/opt/frontier/candidate/bin/frontier-submit train /workspace/train.tar
```

`/opt/frontier/candidate/bin/frontier-submit train` returns immediately with the
`verification_id`, `payload_id`, state (`queued` or `running`), and the
authoritative count of remaining train calls. Read the completed score and
task-approved development `metrics` with `frontier-submit status <verification_id>`.
Continue working while verification runs. **100 train calls are available for the
whole run**, and every accepted request consumes one even when verification fails.
For a beat-spec task, `metrics` includes the approved candidate measurements and
`beat_spec_progress`, keyed by metric name. Each metric contains `passed` (the
number of workloads meeting its reference-beating criterion), `failed` (the
number not meeting it), and `total` (`passed + failed`). A pass is a reference
comparison, not a correctness verdict. `pass_percent` is `100 * passed / total`;
`fail_percent` is `100 * failed / total`. Both percentages range from 0 to 100
and are rounded to two decimal places. This feedback is train-only; final/eval
metrics remain private. It does not return the evaluator's message, exact
workload pass/fail identities, or private reference measurements.

This task requires a Driver and Candidate supporting `store-final`. Save your
first usable answer with `/opt/frontier/candidate/bin/frontier-submit store-final /workspace/final.tar`,
then save again when you select a better revision. This saves one
replaceable snapshot; it does not evaluate it, consume a train call, or end your
work. A successful save replaces it; a rejected replacement preserves the last
accepted snapshot. Editing your local file alone does not update it. When the
tool reports `remaining_runtime_s` and `remaining_checkpoint_submissions`, honor
the remaining answering time. If `remaining_checkpoint_submissions` is an
integer, honor that reported allowance; `null` means there is no save-count quota.
Saving does not extend your answering time. A save receipt is not a final score. An
accepted explicit final always wins. Otherwise the Driver can collect the saved
answer on normal exit, answering timeout, or a confirmed Agent process crash,
but not on cancellation or platform failure. Train submissions are never chosen
automatically. Check each save receipt; a local archive or a train result is not
a stored selection. If this required command is unavailable, report the runtime
incompatibility; do not assume a saved selection exists.

When you are ready, build `/workspace/final.tar` with the same layout, run

```sh
/opt/frontier/candidate/bin/frontier-submit final /workspace/final.tar
```

exactly once, and exit. The final score is produced after you exit; the command
does not print it.
