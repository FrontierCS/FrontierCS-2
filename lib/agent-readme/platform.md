## Objective

{pass}
2. **Then raise the score.** Among passing submissions, a higher continuous score is better.
   Never replace a passing submission with a failing one to gain score. Until something passes,
   keep the best valid submission saved with `/.pumpkin/bin/sandboxctl call final`.

## Submitting

Every platform call is a `sandboxctl` command. Run it by the absolute path shown here: it is
not on the image's `PATH`. Each command prints one line of JSON whose `kind` names the reply,
and exits with 0, with 2 for the status of a job that gave no result, or with 1 when the call is
refused (`{"kind":"refused","message":...}`) or `sandboxctl` itself fails.
`/.pumpkin/bin/sandboxctl ops` lists your limits: the time you have left to work
(`remaining_seconds`), your CPU and memory (`resources`), and the calls you may make, with
their limits. This task's facts:

- Make the patch with `/frontier/task/make_submission.sh` (writes
  `/workspace/payload/solution.patch`). It applies the same scope check as the judge and
  refuses a patch that leaves this task's editable files, before you spend a train call.
{scope}
- Train and final use one uncompressed POSIX tar holding exactly `payload/solution.patch`, at
  most the `payload_max_bytes` that `sandboxctl ops` lists for the function you call with it;
  links, devices and FIFO members are rejected:

```sh
tar --create --file=/workspace/train.tar --directory=/workspace --format=ustar \
  --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 payload/solution.patch
/.pumpkin/bin/sandboxctl call train /workspace/train.tar
```

`sandboxctl call train` returns as soon as the call is accepted, with the call's `job` number
and what the `train` function has left, this call counted, in `remaining` (`calls` and
`seconds`). Read the call's `state`, then its score and metrics, each a field of `returned`
(see Feedback), later with `/.pumpkin/bin/sandboxctl status <job>`.

Save your first usable answer with `/.pumpkin/bin/sandboxctl call final /workspace/final.tar`
(same layout), then save again when you select a better revision. This saves one replaceable
snapshot; it does not evaluate it, consume a train call, or end your work. Its reply's
`returned` holds the `state` `stored`. A successful save replaces it; a refused replacement
preserves the last accepted snapshot. Editing your local file alone does not update it. The
reply's `remaining_seconds` is the time you have left to work; saves are not count-limited, and
do not extend your time. A save receipt is not a final score. An accepted explicit final always
wins. Otherwise the saved answer is judged when you exit, run out of time or crash, but not
when the run is cancelled or the platform fails. Save before your time runs out: a save without
a `stored` reply is not a saved answer. A train submission is never chosen automatically as the
final answer.

When you are ready, build `/workspace/final.tar` with the same layout, run

```sh
/.pumpkin/bin/sandboxctl call final /workspace/final.tar --arg finalize=true
```

exactly once, and exit within its reply's `remaining_seconds`: every later call is refused.

## Limits

- Your CPU and memory are the `resources` that `sandboxctl ops` lists. `nproc` counts the
  host's CPUs, not your CPU limit.
{judge}
- The `train` function's `calls`, which `sandboxctl ops` lists; a call is consumed when it is
  accepted and is never refunded, even when verification or the infrastructure fails.
- The `train` function's cumulative train time (`cumulative_seconds`): the time your calls
  take, summed over them. One train call may run for the function's `call_timeout_seconds`, but
  no longer than the remaining train time or the time you have left to work. The train time
  can run out before the calls do.
- Only one train runs at a time: a train call made while another is queued or running is
  refused, and consumes nothing.
- These limits are ceilings, not a promised number of completed evaluations.
- Your working time is the `remaining_seconds` that `sandboxctl ops` lists; agent work and
  train share it.
- Final evaluation has a separate allowance; it does not consume train calls or the train
  time. It runs after you exit, and `--arg finalize=true` does not print its score.

{feedback}
