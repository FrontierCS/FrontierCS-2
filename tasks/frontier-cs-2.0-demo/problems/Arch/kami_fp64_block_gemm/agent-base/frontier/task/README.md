# KAMI FP64 Block GEMM — trusted batched harness adaptation

Read the [research statement](repo/README.md) for the scientific problem,
editable scope, workloads, correctness contract, timing statistic, and scoring.
The [testbed guide](bin/README.md) covers submission, saved-final selection,
GPU transport, independent clocks, and resource limits.

The source is read-only at `/frontier/task/repo`. Prepare a writable Git copy:

```sh
/frontier/task/prepare_workspace.sh
# Edit only the permitted CUDA device region or solver_extensions/.
/frontier/task/make_submission.sh
```

The helper writes `/workspace/payload/solution.patch`; it does not submit or
create a tar. Train, store-final, and explicit final use the same archive: an
uncompressed POSIX tar containing one top-level `payload/` directory and
exactly one nonempty regular file, `payload/solution.patch`. No executable bit
is required. The complete tar must be at most 262,144 bytes, including headers
and padding. Paths are relative; links, devices, FIFOs, duplicate entries, extra
members, and parent traversal are rejected.

```sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/answer.tar payload
wc -c /workspace/answer.tar
frontier-submit train /workspace/answer.tar
frontier-submit store-final /workspace/answer.tar
# Continue improving and replace the saved selection after useful progress.
# When finished:
frontier-submit final /workspace/answer.tar
```

Use train for an early valid implementation and again after substantive
progress. After feedback, decide whether to replace your saved final. There is
no fixed submission schedule or minimum call count. A successful explicit final
locks the answer and has priority over a saved selection; exit afterward. Its
score is computed independently after Agent cleanup and is not printed by the
submission command. No train submission is implicitly promoted to final.

Formal answering lasts 18,000 seconds (5 hours), including startup and train
waits. Independent final has a separate 1,800 seconds beginning after Agent and
train cleanup. There are 12 train calls: this task-specific retained allowance
comes from the frozen KAMI task rather than the general GPU default. A separately
packaged short POC uses 1,800 answering seconds while retaining the independent
final budget, and its Agent-visible guide must say so. Read the testbed guide and
the actual Candidate-provided Run timing for the version you are using.

The Agent CPU container has no GPU. Use the Candidate-owned
`frontier-gpu` v3 tool for the one-H100, 900-second Agent GPU allowance; do not
look for a task-owned provider client or credentials. Start with
`frontier-gpu quota` and read the testbed guide before creating a runner.

## Train feedback

The approved candidate-only `tflops` measurement is an anonymous numeric array
from the selected train suite. `beat_spec_progress` reports aggregate reference
comparison counts per metric: `passed`, `failed`, `total` (`passed + failed`),
`pass_percent` (`100 * passed / total`), and `fail_percent`
(`100 * failed / total`), rounded to two decimal places. Final metrics remain
private. Reference values, ratios, thresholds, seeds, margins and workload
identities are not returned. This describes the existing shipped feedback
helper; it adds no measurements or submission allowance.
