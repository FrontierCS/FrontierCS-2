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
feedback over the four public train shapes; it does not select the final answer. Local checks and GPU development
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
Final evaluates all six shapes together: the four public shapes with the same
train seeds, plus two hidden shapes with independent seeds. Final score uses all
six speedups; PASS requires all three comparisons on all six workloads.
Train feedback contains no hidden workload measurements or reference values.

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
| Independent final evaluation | 1,800 seconds (30 minutes) |
| Train calls | 25 accepted calls |
| Total train evaluation time | 16,200 seconds |
| Agent CPU / memory / workspace | 4 cores / 8 GiB / 16 GiB |
| Agent GPU | One H100 at a time; 10,800 total GPU seconds; at most 10,800 seconds per instance |
| Verifier GPU | One H100 at a time; 18,000 total GPU seconds |
| Agent SSH | 1,024 accepted sessions |
| Verifier train / final SSH | 50 / 2 independent sessions |
| Model requests | 256 sessions |
| Model request / response size | 1 MiB / 8 MiB |
| SSH captured bytes | 64 MiB per direction per session |
| Total audit bundle | 8 GiB |

## GPU development

The Agent CPU container is the editor. Use Candidate-owned `frontier-gpu` v3 to
create and reuse a Driver-managed Modal GPU machine. Read
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` first.

```sh
frontier-gpu quota
frontier-gpu create --type H100 --count 1 --seconds 10800
frontier-gpu push gpurunner_... /workspace/repo /workspace
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-build.sh
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh
frontier-gpu delete gpurunner_...
frontier-gpu status gpurunner_...
```

Use the returned runner ID. CPU and GPU filesystems are separate: push/pull
explicitly. Repeated SSH operations reuse that runner; they do not create a new
Modal container. Accepted push, pull, exec and interactive SSH/shell operations
consume Agent SSH sessions, including commands that fail after acceptance.
Quota/status/create/delete do not consume SSH sessions. Query `frontier-gpu quota`
for authoritative remaining SSH counts; train and final have separate reserves.
Delete a runner when finished; GPU lifetime consumes its time budget while idle
as well as while computing. Cloud credentials belong to the Driver, not the Agent.
