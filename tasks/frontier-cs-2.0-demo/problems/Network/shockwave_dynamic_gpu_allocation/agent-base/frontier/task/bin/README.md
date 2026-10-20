# Dynamic GPU Allocation testbed usage

Saving has no save-count quota. Payload, retained-data, audit and Agent-time
limits still apply. The save receipt reports `remaining_checkpoint_submissions`
as `null` for this contract. Check the accepted receipt; saving does not
extend answering time or select an answer automatically.

Scientific requirements are in [/frontier/task/repo/README.md](../repo/README.md).
The [task entry](../README.md) shows workspace setup and exact tar creation.
Use the Candidate-installed `frontier-submit`; the task contains no Driver
client, provider credential, GPU helper, or SSH transport. The four V100 devices
described by the task are simulated resources and do not grant physical GPUs.

## Files and submissions

All work belongs below `/workspace`. `prepare_workspace.sh` copies the immutable
source to `/workspace/repo` and commits its baseline. `make_submission.sh`
creates `/workspace/payload/solution.patch`; it does not create or submit a tar.
Package `payload/` as an uncompressed POSIX tar containing exactly the nonempty
regular file `payload/solution.patch`. The complete archive must fit 262144
bytes. Relative paths only; links, devices, FIFOs, and `..` are rejected.

`frontier-submit train /workspace/answer.tar` evaluates the original two-trace
development suite. An accepted call consumes one of 3 train calls even if
verification fails. Approved feedback contains the candidate's four anonymous
ordered measurement arrays and aggregate beat-spec counts and percentages.
Workload names, individual comparison decisions, reference values, and margins
remain operator-only.

## Keep a chosen final while working

`frontier-submit store-final /workspace/answer.tar` saves one Driver-owned
snapshot. A successful replacement supersedes it; a rejected replacement keeps
the prior snapshot. Saving does not train, end answering, or lock final. Local
file changes do not update saved bytes. Rejected saves do not replace the last successful selection. Payload and audit capacity still apply.

`frontier-submit final /workspace/answer.tar` selects an irreversible explicit
final and takes priority over the saved copy. Without explicit final, normal
exit, answering timeout, or a process crash confirmed by trusted container state
can collect the saved copy. User cancellation and platform failure do not.
Without either selection there is no final; train never selects an archive.

Driver stops Agent and train resources, confirms cleanup, then independently
evaluates the selected bytes on the original two-trace final suite. Final does
not reuse train measurements.

## Concrete budgets

- Formal answering: 18000 seconds, including bootstrap and train waits. A
  separately published short trial uses 1800 seconds and exposes that actual
  trial budget in its image/instructions.
- Independent final: 1800 seconds after cleanup, preserving the original
  evaluator allowance and including Verifier provisioning.
- Train: at most 3 accepted calls and 5400 cumulative train-wall seconds,
  bounded in practice by the answering clock.
- Agent: 4 CPU, 4 GiB RAM, 4 GiB workspace. Verifier: 4 CPU, 4 GiB RAM.
- Model: 1024 sessions, with 1 MiB request and 8 MiB response limits.
- Total audit allowance: 14 GiB; evidence allowance: 4 MiB per call. The model-log
  allowance is independently enforced within the audit budget.
- GPU/SSH: not applicable. Evaluation is a deterministic CPU simulation.

There is no required solver score, fixed submission count, or early-stop rule.
