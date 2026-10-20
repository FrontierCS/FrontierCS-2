# Testbed usage

Read [the task statement](../repo/README.md) for the scientific problem and
scoring, and [the operational entry](../README.md) for preparation and packaging.

## Prepare and package

Use the preparation and submission helpers in the operational entry. After
changing your patch, package it again; an older archive does not contain later
workspace edits. Preserve the exact `payload/solution.patch` archive member.

## Train, selected snapshot, and final

`frontier-submit train /workspace/train.tar` returns after acceptance with a
verification ID and remaining train calls. Judging runs asynchronously; continue
working while it runs. `frontier-submit status <verification_id>` reads its status
once. A train result does not select an answer for final judging.

You must keep an Agent-selected answer saved with
`frontier-submit store-final /workspace/final.tar`. Save the first answer you
select and replace it when you choose a better revision. The command saves a
replaceable snapshot; it neither judges it, consumes a train call, nor ends your
work. Saves are not count-limited: `remaining_checkpoint_submissions: null`
indicates unlimited saves. Payload, audit and remaining runtime limits still
apply. Check every receipt; local edits do not update saved bytes, and a rejected
replacement preserves the last accepted snapshot.

When finished, `frontier-submit final /workspace/final.tar` explicitly selects
the final artifact, then exit. An accepted explicit final takes priority over a
saved snapshot. Train submissions are never automatically promoted to final.

The Driver can collect the selected saved answer on normal exit, answering
timeout or a supported confirmed Agent process crash. Cancellation and platform
failure do not imply final collection. A save receipt is not a final score or a
promise of recovery from arbitrary infrastructure failure.

## Runtime limits

The Agent answering window is **5 hours (18,000 seconds)**. Independent final
judging starts afterward and has its own **3720-second** limit. Scientific
workload limits in the task statement still apply. Saving does not extend either
clock. CPU values below are millicores; memory, archive and audit sizes are bytes;
time values are seconds.

### agent

```json
{
  "cpu_millis": 2000,
  "memory_bytes": 4294967296,
  "workspace_bytes": 4294967296
}
```

### quota

```json
{
  "eval_timeout_seconds": 3720,
  "max_train_calls": 100,
  "max_train_wall_seconds": 372000,
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

### verifier resources

```json
{
  "cpu_millis": 2000,
  "memory_bytes": 4294967296,
  "max_evidence_bytes": 4194304
}
```
