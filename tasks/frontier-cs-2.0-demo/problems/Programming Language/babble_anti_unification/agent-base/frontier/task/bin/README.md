# BABBLE testbed usage

Saving has no save-count quota. Payload, retained-data, audit and Agent-time
limits still apply. The save receipt reports `remaining_checkpoint_submissions`
as `null` for this contract. Check the accepted receipt; saving does not
extend answering time or select an answer automatically.

Scientific requirements are in [/frontier/task/repo/README.md](../repo/README.md).
The [task entry](../README.md) shows workspace setup and exact tar creation.
Use the Candidate-installed `frontier-submit`; no task-specific Driver client
or confidential provider credential is required.

The current Agent and Verifier both use Rust 1.64.0. This supersedes the
historical Rust 1.85 solver-tooling sentence in the preserved research statement;
the authoritative Rust 1.64 build and scientific requirements are unchanged.

## Files and submissions

All work belongs below `/workspace`. `prepare_workspace.sh` copies the immutable
source into `/workspace/repo` and commits its baseline. `make_submission.sh`
creates `/workspace/payload/solution.patch`, including new allowed extension
files. It does not create or submit the archive. Package `payload/` separately
into an uncompressed POSIX tar with exactly `payload/solution.patch`, a nonempty
regular text patch. No executable bit is required. The whole tar (headers and
padding included) must fit 262144 bytes; the patch itself also has a 262144-byte
bound. Relative paths only; no `..`, symlinks, hardlinks, devices or FIFOs.

`frontier-submit train /workspace/answer.tar` verifies the five development
corpora. Its approved feedback includes the continuous compression score,
absolute candidate compression/runtime arrays and per-metric aggregate pass/fail
counts and percentages, without private reference values or workload identities.
The ten distinct final corpora use the same original scientific evaluator.

## Keep a chosen final while working

`frontier-submit store-final /workspace/answer.tar` saves one Driver-owned snapshot.
A successful replacement supersedes it; a rejected replacement preserves it.
Inspect the save receipt before relying on the selection. Saving does not train,
end answering, or lock final. After training, decide whether the measured change
belongs in your saved answer. Editing the local tar alone does not change the
saved bytes. Rejected saves do not replace the last successful selection. Payload and audit capacity still apply.

`frontier-submit final /workspace/answer.tar` is an irreversible explicit final
once accepted and has priority over the saved copy. Without explicit final,
normal exit, answering timeout, or a process crash confirmed by trusted container
state can collect the saved copy. User cancellation and platform failure do not;
Driver Pod failure recovery is not promised. Without either selection, there is
no final; a previous train submission is never selected automatically.

Driver stops Agent and train resources, confirms cleanup, and independently
evaluates the selected archive. Cleanup failure remains a failure. Final has
its own time budget and does not reuse a train score or measurements. The final
command does not print the final score.

## Concrete budgets

- Formal answering: 18000 seconds, including bootstrap and train waits.
  A separately published short trial uses 1800 seconds, with the actual trial
  budget supplied in its image/instructions. Follow the Run's actual timing
  information; there is no task-specific remaining-time command.
- Independent final: 600 seconds after the cleanup barrier, including verifier
  provisioning. Build timeout 300 seconds; each corpus repetition 30 seconds;
  three repetitions per corpus. Actual completion must fit the outer budget.
- Train: at most 100 accepted calls and 1800 cumulative train wall seconds;
  these are separate limits. Failed verification still consumes an accepted call.
- Agent: 2 CPU, 8 GiB RAM, 4 GiB workspace; Rust 1.64 tooling, matching the authoritative Verifier.
- Verifier: 2 CPU, 2 GiB RAM, 4 GiB storage; authoritative Rust 1.64 build.
  The task is CPU-only, with no GPU or SSH capability/pool.
- Model: 1024 sessions; each request at most 1 MiB and response at most 8 MiB.
  Local shell/tool commands are not themselves model sessions.
- Total audit allowance: 14 GiB; verifier evidence allowance: 4 MiB per call.
  Submission/archive limits are separate. Keep debug output bounded. The
  platform's model-log allowance is independently enforced, not added to audit.

There is no required solver score, fixed number of submissions or fixed early
stopping schedule. Use the available answering window to improve your solution.
