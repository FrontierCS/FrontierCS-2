# Testbed operations

Read `/frontier/task/README.md` for this task's workspace and exact payload/tar
format, and `/frontier/task/repo/README.md` for the problem and scoring. Use the
installed `frontier-submit` and, for GPU tasks, `frontier-gpu` skills. Work under
`/workspace`; task files under `/frontier/task` are read-only.

## Train feedback

Build the task-defined uncompressed tar and run:

```sh
frontier-submit train /workspace/train.tar
```

An accepted train request consumes one call, including a failed verification.
Read its score, approved metrics and `remaining_train_calls`. Train provides
feedback; it does not select the final answer. Local checks and GPU development
do not produce the independent final score.

## Keep a selected final while iterating

Build `/workspace/final.tar` using the exact same task-defined tar layout, then:

```sh
frontier-submit store-final /workspace/final.tar
```

A successful `stored` receipt means the Driver saved that exact archive. Only
one selection is kept. You may replace it and continue working; a failed update
preserves the previous selection. There is no save-count quota, and saving does
not consume train calls or start evaluation. Archive size and audit capacity
limits still apply.

After train feedback, decide whether your selected answer should change. If it
should, rebuild the final archive and save it. Changing the local file alone
does not change the saved selection. Save the answer you choose, rather than
assuming the last train or highest train score will be selected automatically.

On normal exit, answer timeout, or an Agent process crash, the Driver collects
the saved selection if no explicit final has been accepted. User cancellation
and platform failures do not collect it. Without a saved or accepted explicit
final, exiting or timing out does not produce a final score.

## Finish explicitly

When done, you can still use the existing final command:

```sh
frontier-submit final /workspace/final.tar
```

An accepted explicit final locks that archive and takes priority over the saved
selection. It cannot be replaced. After acceptance, make no further platform
calls and exit. The Driver stops Agent/train resources before independently
evaluating the selected archive. Final receives a separate evaluation budget;
its score appears to the user after evaluation, not in the submission receipt.

## Time and quotas

The values below describe the task's normal published configuration. A short
trial can set a shorter answering window in its Run configuration; it does not
change submission semantics. Build time, local work and train waits consume the
answering window. Final evaluation has its own budget after collection/cleanup.
A train-call count is not a model-request or SSH-session count. Local tool calls
do not themselves consume model/SSH sessions; a tool can issue multiple SSH
operations or lead to another model request. Per-request byte bounds and the
total audit budget also apply, so counts alone do not guarantee remaining space.

| Resource | Normal task limit |
| --- | --- |
| Answering window | 18,000 seconds (5 hours) |
| Independent final evaluation | 120 seconds |
| Train calls | 100 accepted calls |
| Total train evaluation time | 12,000 seconds |
| Agent CPU / memory / workspace | 8 cores / 32 GiB / 32 GiB |
| Model requests | 992 sessions |
| Model request / response size | 8 MiB / 8 MiB |
| Total audit bundle | 8 GiB |
| GPU development | Not available; this is a CPU-only task |

The TaskConfig's legacy SSH allowance does not grant a GPU or an SSH development
machine. Do not call `frontier-gpu` for this task. Keep all solved instances in
your selected cumulative JSON ledger.
