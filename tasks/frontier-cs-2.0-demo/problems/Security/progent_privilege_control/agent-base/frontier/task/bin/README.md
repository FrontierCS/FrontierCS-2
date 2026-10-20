# Testbed usage

Read the task statement at `/frontier/task/repo/README.md` for the scientific
problem and scoring. In the source checkout it is
[agent-base/README.md](../../../README.md).
The [operational entry](../README.md) retains the task-specific preparation,
archive layout, and submission commands.

## Prepare and package

Use the preparation and submission helpers described in the operational
entry. Preparing the workspace and producing a patch are separate from
packaging that patch into the submission tar. Repackage after changing it;
an older tar does not include later workspace edits.

The policy entrypoint is `payload/solution.patch`. The policy archive limit is
8,388,608 bytes. Preserve the task's exact archive member layout.

## Train, checkpoint, and final

`frontier-submit train /workspace/payload.tar` requests development feedback;
it does not select a final answer. Use the feedback to improve the artifact
within the existing task limits.

`frontier-submit store-final /workspace/payload.tar` stores a replaceable snapshot
and returns a receipt. It does not run verification or end the answer phase.
A successful later store replaces the earlier snapshot; a failed store does
not replace it. Keep the receipt. Editing local files does not update stored bytes.

`frontier-submit final /workspace/payload.tar` explicitly selects the final
artifact. An accepted explicit final takes priority over a stored snapshot;
submit it when finished. Training results are never automatically promoted
to a best or last final answer.

The platform's stored-final recovery applies on normal completion, answer
timeout, or a supported trusted-process crash. User cancellation and platform
failure do not imply final collection. A receipt is not a promise of recovery
from arbitrary infrastructure failure.

## Existing policy limits

These are the unchanged task-policy values, not new allowances. Scientific
and submitted-program limits in the statement still apply. Time values are
seconds; size values are bytes; CPU values are millicores. The policy's total
wall-time cap must not be interpreted as an additional answer-time allowance.

### agent

```json
{
  "cpu_millis": 4000,
  "memory_bytes": 8589934592,
  "workspace_bytes": 4294967296
}
```

### quota

```json
{
  "eval_timeout_seconds": 10800,
  "max_train_calls": 5,
  "max_train_wall_seconds": 18000,
  "wall_time_seconds": 18000
}
```

### submission

```json
{
  "entrypoint": "payload/solution.patch",
  "max_bytes": 8388608
}
```

### verifier

```json
{
  "command": [
    "/opt/frontier/verifier"
  ],
  "cpu_millis": 8000,
  "max_evidence_bytes": 4194304,
  "memory_bytes": 17179869184
}
```

### network

```json
{
  "allowed_domains": [],
  "audit": {
    "max_bundle_bytes": 34359738368,
    "max_model_request_bytes": 67108864,
    "max_model_response_bytes": 8388608,
    "max_model_sessions": 1024,
    "max_ssh_bytes_per_direction": 8388608,
    "max_ssh_sessions": 32
  }
}
```

You must save the final answer you select with `frontier-submit store-final /workspace/final.tar`.
You decide which answer to save and whether to replace that selection.
Confirm the accepted receipt; local edits and train results do not update
the saved selection or automatically select a final. Saving has no save-count
quota; `remaining_checkpoint_submissions` is `null`. Payload, audit and
Agent-time limits still apply. Saving does not extend answering time.
