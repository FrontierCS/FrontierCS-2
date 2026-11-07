## Using a GPU

How the GPU runner works (create, push, exec, pull, delete, quota) is in the
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` skill; the helper is
`/opt/frontier/candidate/bin/frontier-gpu`. This task's facts:

- Runners: {max_instances} at a time, type {gpu_type}, at most {instance_seconds} s each and
  {gpu_seconds} s in total for the whole run.
- **{agent_ssh} SSH sessions for Agent GPU debugging**; every accepted `exec`, `ssh`, `push` or
  `pull` consumes one, including a command that later fails. The Verifier has its own pools
  ({train_ssh} sessions for train, {final_ssh} for final), separate from yours.
- At most {ssh_bytes} bytes per SSH session in each direction, including tar overhead.
- On the runner, prefix task commands with `/opt/frontier/task-runtime` (SSH is not a login
  shell); `/opt/frontier/runtime-python` is the runner image's Python.
- Wait until a runner is `ready` before using SSH. Sessions are not refunded, so batch work into
  one bounded command rather than many small ones. GPU work counts against the answering window.
