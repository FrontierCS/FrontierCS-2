# FrontierCS tasks for Pumpkin

This branch holds tasks of [FrontierCS](https://github.com/FrontierCS/FrontierCS-2), each as a
workflow that [Pumpkin](https://github.com/FrontierCS/Pumpkin/tree/96942725f939ac1a5de80fbe68ff77ec2738ee96)
runs. In a run, an agent works on the task's statement and starter, may submit to the task's
verifier for scored train calls, and submits once for the final call, whose score is the run's.
Each role, the agent and the two judges that run the verifier, states what its sandbox needs. The
agent is a harness, one of the coding agents that FrontierCS evaluates, such as Codex or Claude
Code, on a model of your choice, or `reference`, which submits the task's reference solution.

## What you need

- Linux on x86_64 with Git, Git LFS and Python 3. On first use, `./pumpkin` downloads the
  release pinned in `.pumpkinversion`, verifies the archive’s SHA-256 and installs it under
  `${XDG_CACHE_HOME:-$HOME/.cache}/pumpkin`. The launcher needs TLS certificates, `curl` or
  `wget`, `sha256sum`, `tar`, `xz` and `flock`. Run records stay in this repository’s `.pumpkin/`.
- Docker, which builds the tasks' images and runs their sandboxes.
- For an agent that works on a model, a key for a service that serves the model.

## Check and run a task

Clone with `git lfs` installed, so that the clone fetches its Git LFS files, and write
`~/.config/pumpkin/pumpkin.toml` so that Pumpkin runs every role on this machine's Docker engine:

```toml
[docker]
context = "default"

[providers.local.docker]
```

The reference agent submits the task's reference solution and needs no model, so it is the cheap
way to see a task end to end; a task whose judges call a model needs an upstream (below) that
serves it, with its prices, even for the reference agent. `<name>` is a directory of `tasks/`:

```console
$ git clone --single-branch --branch pumpkin https://github.com/FrontierCS/FrontierCS-2.git fcs-tasks
$ cd fcs-tasks
$ ./ci.py                    # installs Pumpkin on first use, then checks every task
...
==> pumpkin check, <n> tasks, --arg=agent=codex --arg=model=gpt-6.1-terra
==> pumpkin check, <n> tasks, --arg=agent=reference
==> pumpkin check, 2 smoke workflows
ci.py: all checks passed
$ ./pumpkin check tasks/<name>/task.star --arg agent=reference   # silent if sound
$ ./pumpkin run tasks/<name>/task.star --arg agent=reference     # prints the run's id
...
20261117-040053-978f3ae9
$ ./pumpkin wait 978f        # follows the run, named by a prefix of its last part, to its end
...
pumpkin: run 20261117-040053-978f3ae9 finished with a result (score=<score> status="passed" beats_reference=<n> summary="<summary>" ...), after the work phase ended when the agent exited with status 0; its record is /home/alice/fcs-tasks/.pumpkin/runs/20261117-040053-978f3ae9
...
$ ./pumpkin wait --outcome 978f > outcome.json
$ ./pumpkin ls               # the runs of the last day
20261117-040053-978f3ae9 finished      1m56s  -               -  frontier-cs/<name>: score=<score> status="passed" beats_reference=<n> ...
...
```

`run` builds the task's images, starts the run and returns; the run goes on without it. A run's
record is `.pumpkin/runs/<id>/`, and `wait --outcome` prints its `outcome.json` once the run has
ended: `state` is `finished` for a run that ended with a verdict, and `verdict.result.values` lists
the final call's values, each a `name` and a `value`: `score`, `beats_reference`, then the
verifier's `summary` and metrics; a verification that failed adds `status`, `failed`.

Each task has one agent and one work deadline. Its train calls use fresh judges. Saving a
submission replaces the agent's previous archive. Explicit finalization locks that archive;
otherwise the last saved archive is judged when the agent exits or its time ends. Leaving
without saving an archive finishes without a result and `pumpkin wait` exits with status 3.

A harness works on a model that an upstream serves: an `[upstreams.<name>]` table of the same
`pumpkin.toml` names the upstream's URL, the file that holds its key and the models it serves, by
the `name` that `--arg model` gives; `./pumpkin help config` says every key, with an example
([models](https://github.com/FrontierCS/Pumpkin/blob/96942725f939ac1a5de80fbe68ff77ec2738ee96/docs/models.md)):

```console
$ ./pumpkin run tasks/<name>/task.star --arg agent=codex --arg model=gpt-6.1-terra
```

`--arg agent` names one of `HARNESSES` in `lib/fcs.star`; `check` says what a harness lacks:
under Codex's default, `--arg tools=functions`, it refuses a model outside the catalog that
Pumpkin's Codex carries and names the models it does carry (`TOOLS` in `lib/fcs.star` says why
`functions` is the default). The `arg()` calls at the top of `lib/fcs.star` declare what every
task takes, such as `deadline_seconds`, the work phase, five hours by default.
[Running](https://github.com/FrontierCS/Pumpkin/blob/96942725f939ac1a5de80fbe68ff77ec2738ee96/docs/running.md)
goes through each command and `pumpkin view`, the runs' page;
[records](https://github.com/FrontierCS/Pumpkin/blob/96942725f939ac1a5de80fbe68ff77ec2738ee96/docs/record.md)
says what a record holds.

## A task that needs a GPU or another provider

The local Docker engine offers this machine's CPUs, memory and CPU flags, but no GPU and no CPU
model, so the tasks that state either (`grep --files-with-matches --extended-regexp
'"(gpu|cpu_model)"' tasks/*/task.star`) run on EC2 or Modal, each of which needs the AWS envelope
that Pumpkin's `deploy/aws` makes. `run` refuses a role that no class of the configured providers
offers, saying what each class lacks, before anything is reserved.
[Providers](https://github.com/FrontierCS/Pumpkin/blob/96942725f939ac1a5de80fbe68ff77ec2738ee96/docs/providers.md)
says what EC2, Modal and a controller, which builds the images and drives the runs there, each
need, and where each is set up.

## What is in the repository

- `tasks/<id>/`: a task each. `task.star`; the published task's `judge/`, `datasets/` and `repo/`;
  `generated/`, the configuration that FrontierCS's build derives from the task; the recipes of its
  images (`agent-base/`, `verifier/`, and `runtime/` where they share one); and this branch's own
  judge files: `eval/`, the evaluator of a task whose port has its own, and `train/`, what
  the train judge's image holds or runs in place of what it leaves out. A task using the shared
  verifier base keeps these overlays beside its verifier Dockerfile when that directory is its build context.
- `lib/`: what every task shares: `fcs.star`, FrontierCS's conventions and `fcs_task`, which each
  `task.star` calls; the platform's part of each agent's README; `frontier-gpu`, the agent's client
  for its GPU runners.
  Each task's `visible_metrics` maps public train metric names to their descriptions in
  display order. The verifier applies the benchmark's native feedback policy before the
  train grant chooses which returned values to show. To add a public metric, have the judge
  compute it, include it in the native solver-visible list, and add the same name and its
  description to `visible_metrics`. Check a train result and the agent's Feedback section.
  Keep held-out workload identities and reference results in the final judge.
  Set `agent_recipe = package_agent(ubuntu_agent_base)` to supply a package's `ARG AGENT_BASE`,
  or `verifier_image = ubuntu_verifier_image` for its `ARG VERIFIER_BASE`. These named image
  contexts are built from `lib/agent-base/` and `lib/verifier-base/`. Both
  recipes pin the Ubuntu image by digest and install packages from a dated archive snapshot.
  The shared tools image copies the verifier modules selected by
  `lib/task-tools/.dockerignore`. To add a shared verifier helper, allow its path there and
  include it in the verifier recipe that imports it. Keep task-specific adapters in the
  task directory so they travel with the task when it is published.
- `shared/`: the components that FrontierCS shares among its tasks, which the recipes copy, and
  `verifier-lease/`, a judge's GPU adapter with FrontierCS's interface, over runners that Pumpkin
  leases to the judge.
- `fetch/`: the helpers that recipes run to fetch what a digest pins.
- `smoke/`: two workflows that check a deployment, not a task: `lease/` leases GPU and CPU
  runners, and `routes/` sends model requests from every role; no task loads them.
- `PUMPKIN.star`, `pumpkin`, `.pumpkinversion` and `ci.py`: the root of the `//` labels, the
  launcher that installs and runs the pinned Pumpkin, the pin, and the checks.

## How the ports differ

A task here is a port of the published one: its `judge/`, `datasets/` and `repo/` are the files of
FrontierCS at commit [`2fc5ff7c`](https://github.com/FrontierCS/FrontierCS-2/tree/2fc5ff7c6e43e11c64d7d644b9808b010b3814b5), unchanged; its `task.star` and recipes restate the published
`task.toml`, `README.md` and `docker/` recipes for Pumpkin, and its agent submits with Pumpkin's
`sandboxctl` in place of `frontier-submit`. A port may depart from how FrontierCS runs the task in
three ways: the judges' images hold the port's `eval/evaluator.py` in place of
`judge/evaluator.py`, so that the score no longer rests on what the submission's code prints or
leaves behind; the train judge's image leaves out what only the final evaluation reads, so that a
train call cannot reach it; and a value or a line of the judge differs, where Pumpkin's sandboxes or
the port's evaluator need another. Where a port departs from the published task,
`tasks/<name>/PORT.md` says how.
