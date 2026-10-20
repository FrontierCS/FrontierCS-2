# Testbed usage

Read the task statement at `/frontier/task/repo/README.md` for the scientific
problem and scoring. In the source checkout it is
[task statement](../repo/README.md).
The [operational entry](../README.md) retains the task-specific preparation,
archive layout, and submission commands.

## Prepare and package

Use the preparation and submission helpers described in the operational
entry. Preparing the workspace and producing a patch are separate from
packaging that patch into the submission tar. Repackage after changing it;
an older tar does not include later workspace edits.

The policy entrypoint is `payload/solution.patch`. The policy archive limit is
262,144 bytes. Preserve the task's exact archive member layout.

## Train, checkpoint, and final

`frontier-submit train /workspace/payload.tar` requests development feedback;
it does not select a final answer. Use the feedback to improve the artifact
within the existing task limits.

`frontier-submit store-final /workspace/payload.tar` stores a replaceable snapshot
and returns a receipt. It does not run verification or end the answer phase.
You must save the answer you select; you decide which answer to save and
whether to replace it. A successful later store replaces the earlier snapshot;
a failed store preserves it. There is no save-count quota, and the receipt's
`remaining_checkpoint_submissions` is `null`. Payload, audit and Agent-time
limits still apply. Keep the receipt. Editing local files does not update stored bytes.

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
  "cpu_millis": 3000,
  "memory_bytes": 12884901888,
  "workspace_bytes": 10737418240
}
```

### quota

```json
{
  "eval_timeout_seconds": 14400,
  "max_train_calls": 3,
  "max_train_wall_seconds": 16200,
  "wall_time_seconds": 18000
}
```

### submission

```json
{
  "entrypoint": "payload/solution.patch",
  "max_bytes": 262144
}
```

### verifier

```json
{
  "command": [
    "/opt/frontier/verifier"
  ],
  "cpu_millis": 3000,
  "max_evidence_bytes": 4194304,
  "memory_bytes": 12884901888
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
    "max_ssh_sessions": 48,
    "ssh_session_budgets": {
      "agent": 32,
      "verifier_eval": 4,
      "verifier_train": 12
    }
  }
}
```

### gpu

```json
{
  "agent": {
    "allowed_types": [
      "H100"
    ],
    "max_gpu_seconds": 14400,
    "max_instance_seconds": 14400,
    "max_instances": 1
  },
  "verifier": {
    "max_gpu_instances": 1,
    "max_gpu_seconds": 30600
  }
}
```

## GPU workflow

Use the installed Candidate-owned GPU helper and its task-specific operational
instructions. Transfer inputs and changed code explicitly with push, and
retrieve outputs explicitly with pull; the CPU and GPU filesystems are not
implicitly synchronized. Reuse a running instance within its existing limits.
Inspect remaining quota through the helper's non-consuming quota queries.
Agent, train-verifier, and final-verifier SSH budgets remain separate where
the policy lists them. No credentials belong in submitted artifacts.
