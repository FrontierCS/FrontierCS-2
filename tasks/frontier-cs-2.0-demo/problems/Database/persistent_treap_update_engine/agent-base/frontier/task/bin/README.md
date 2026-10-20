# Persistent Treap testbed usage

Scientific requirements are in [/frontier/task/repo/README.md](../repo/README.md).
The [task entry](../README.md) shows workspace setup and exact tar creation.
Use the Candidate-installed `frontier-submit`; the task contains no Driver
client, provider credential, GPU helper, or SSH transport.

## Files and submissions

All work belongs below `/workspace`. `prepare_workspace.sh` copies the immutable
source to `/workspace/repo` and commits its baseline. `make_submission.sh`
creates `/workspace/payload/solution.patch`; it does not create or submit a tar.
Package `payload/` as an uncompressed POSIX tar containing exactly the nonempty
regular file `payload/solution.patch`. The complete archive must fit 262144
bytes. Relative paths only; links, devices, FIFOs, and `..` are rejected.

`frontier-submit train /workspace/answer.tar` evaluates the original development
suite. An accepted call consumes one of 100 train calls even if verification
fails. Approved feedback contains aggregate beat-spec progress only. Candidate per-case
throughput, private reference values, and workload identities remain operator-only.

## Keep a chosen final while working

`frontier-submit store-final /workspace/answer.tar` saves one Driver-owned
snapshot. A successful replacement supersedes it; a rejected replacement keeps
the prior snapshot. Saving does not train, end answering, or lock final. Local
file changes do not update saved bytes. The protocol has no save-count limit;
payload and audit capacity still apply.

`frontier-submit final /workspace/answer.tar` selects an irreversible explicit
final and takes priority over the saved copy. Without explicit final, normal
exit, answering timeout, or a process crash confirmed by trusted container state
can collect the saved copy. User cancellation and platform failure do not.
Without either selection there is no final; train never selects an archive.

Driver stops Agent and train resources, confirms cleanup, then independently
evaluates the selected bytes. Final does not reuse train measurements.

## Concrete budgets

- Formal answering: 18000 seconds, including bootstrap and train waits. A
  separately published short trial uses 1800 seconds and exposes that actual
  trial budget in its image/instructions.
- Independent final: 1800 seconds after cleanup, preserving the original
  evaluator allowance and including Verifier provisioning.
- Train: at most 100 accepted calls and 180000 cumulative train-wall seconds,
  bounded in practice by the answering clock.
- Agent: 8 CPU, 16 GiB RAM, 8 GiB workspace. Verifier: 8 CPU, 16 GiB RAM.
- Model: 1024 sessions, with 64 MiB request and 8 MiB response limits.
- Total audit allowance: 32 GiB; evidence allowance: 4 MiB per call. The model-log
  allowance is independently enforced within the audit budget.

There is no required solver score, fixed submission count, or early-stop rule.
