# Residual State Advancement testbed usage

Scientific requirements are in [/frontier/task/repo/README.md](../repo/README.md).
The [task entry](../README.md) shows workspace setup and exact tar creation.
Use the Candidate-installed `frontier-submit`; the task contains no Driver
client, provider credential, GPU helper, or SSH transport.

## Files and submissions

All work belongs below `/workspace`. `prepare_workspace.sh` copies the immutable
source to `/workspace/repo` and commits its baseline. `make_submission.sh`
creates `/workspace/payload/solution.patch`; it does not create or submit a tar.
Package `payload/` as an uncompressed POSIX tar containing exactly the nonempty
regular file `payload/solution.patch`. The complete archive must fit 131072
bytes. Relative paths only; links, devices, FIFOs, and `..` are rejected.

`frontier-submit train /workspace/answer.tar` evaluates the original development
suite. An accepted call consumes one of 100 train calls even if verification
fails. Feedback includes the bounded score and unbounded score only. Reference
measurements and the final aggregate reference verdict remain operator-only.

## Keep a chosen final while working

`frontier-submit store-final /workspace/answer.tar` saves one Driver-owned
snapshot. A successful replacement supersedes it; a rejected replacement keeps
the prior snapshot. Saving does not train, end answering, or lock final. Local
file changes do not update saved bytes. Saving has no save-count quota.
Payload, retained-data, audit and Agent-time limits still apply. The save receipt
reports `remaining_checkpoint_submissions` as `null`; check the accepted receipt.

`frontier-submit final /workspace/answer.tar` selects an irreversible explicit
final and takes priority over the saved copy. Without explicit final, normal
exit, answering timeout, or a process crash confirmed by trusted container state
can collect the saved copy. User cancellation and platform failure do not.
Without either selection there is no final; train never selects an archive.

Driver stops Agent and train resources, confirms cleanup, then independently
evaluates the selected bytes. Final does not reuse train measurements.

## Concrete budgets

- Formal answering: 18000 seconds, including bootstrap and train waits. A
  separately published short trial uses 1800 seconds and must expose that actual
  trial budget in its Agent image and instructions.
- Independent final: 7200 seconds after cleanup, preserving the original
  evaluator allowance and including Verifier provisioning.
- Train: at most 100 accepted calls and 720000 cumulative train-wall seconds,
  bounded in practice by the answering clock.
- Agent: 2 CPU, 16 GiB RAM, 4 GiB workspace. Verifier: 2 CPU, 16 GiB RAM.
- Model: 1024 sessions, with 1 MiB request and 8 MiB response limits.
- Total audit allowance: 14.75 GiB; evidence allowance: 4 MiB per call. The deployed
  4 GiB model-log allowance is inside, not in addition to, that audit budget.
- CPU-only: there is no GPU, SSH session pool, SSH byte budget, or Modal access.

There is no required solver score, fixed submission schedule, or early-stop rule.
