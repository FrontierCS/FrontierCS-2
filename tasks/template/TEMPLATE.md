# Task template

`Theory/template_task/` is a complete, working FrontierCS2 task in the `task.toml` format: a toy
C problem (make `count_primes` fast) whose every file shows what a real task's file must
contain. Start a new task by copying it:

```sh
cp -r tasks/template/Theory/template_task tasks/problems/<Category>/<task_id>
# then: rename id in task.toml, replace every file's content
make validate
scripts/task_manifest.py show tasks/problems/<Category>/<task_id>
```

The template is not in the catalog and is never published. `tests/test_template_task.py` keeps
it valid and runs its judge, so a format change that breaks the template fails CI.

## Layout

```
<task_id>/
  task.toml                 the definition (the only one)
  README.md                 Agent-facing: where things are and how to submit
  repo/                     Agent-facing starter tree, copied to /frontier/task/repo
    README.md               the task statement
    ...                     the starter code
  judge/                    Verifier-only: /opt/frontier/task in the Verifier image
    evaluator.py            evaluate(tree) -> (score, unbounded, message, metrics); its
                            docstring is the scoring note
    reference.patch         the reference solution, as a diff of repo/ (patch tasks)
    ...                     private suites, helpers, fixtures
  datasets/
    train.json              selector object for train (hash pinned in task.toml)
    eval.json               selector object for final
  docker/
    agent.Dockerfile        FROM the shared Agent base + this task's packages and public files
    verifier.Dockerfile     FROM the shared Verifier base + this task's packages and judge
```

What a task does **not** contain any more, because it is shared or generated:

| was per task | now |
|---|---|
| `policy.json` | `task.toml` |
| `agent-base/frontier/task/bin/*` (prepare_workspace.sh, make_submission.sh, submission tool) | `tasks/shared/agent/bin/`, in the Agent base image |
| `verifier/*.py` (fcs2_verifier.py, verify.py, beat_spec_feedback.py) | `tasks/shared/verifier/`, in the Verifier base image |
| `judge/submission_patch.py` | none: the shared verifier applies the patch with `tasks/shared/lib/edit_scope.py` |
| `judge/beat_spec.json` | the scoring note in `judge/evaluator.py`'s docstring; what train returns is `task.toml` `[feedback]`; old specs are archived in `docs/task-authoring/<task_id>/provenance/` |
| `judge/config.yaml` | generated from `task.toml` (`.frontier-generated/judge/config.yaml`) |
| `bin/submission-policy.json` | generated from `task.toml` |
| `upstream.lock.json`, provenance notes | `docs/task-authoring/<task_id>/` |
| `judge/harbor/app` (a second copy of the starter) | the Verifier Dockerfile copies `repo/` where the judge reads it (`/opt/frontier/task/pristine` here; migrated judges still read `harbor/app`) |
| a task's own submission tool, patch checker or verifier adapter | none: every task uses the shared ones (see "No forks" below) |

## No forks of shared code

The submission tools, the edit-scope check, the verifier adapter and the train feedback are
shared interfaces: there is one copy, in `tasks/shared/`, and every task uses it. A task
differs from another only through

- `task.toml` values (resources, quota, `[submission]` scope, overrides of
  `tasks/shared/task-defaults.toml`), and
- the judge hooks the shared verifier calls: `judge/evaluator.py`, and optionally
  `judge/candidate-diagnostic.json` + `judge/candidate_failure.py`.

A task never ships `bin/`, `verifier/` or its own `submission_patch.py`. When a task needs
something the shared code cannot express, extend the shared code (with a test) instead of
copying it. The answer always lives in `repo/` and is submitted as `git diff` of `repo/`,
even when it is a data file (lwe_structured_recovery: `repo/solution.json`).

## Files

### `task.toml`

Only what differs between tasks and cannot be derived: identity (`id`, `display_name`,
`version`), `[agent]` and `[verifier]` resources, `[quota]`, and `[submission]`
`max_bytes` / `allowed_paths`. `scripts/task_manifest.py` refuses unknown keys and missing
required ones. The rest is not written here:

- **Defaults**, written out once in `tasks/shared/task-defaults.toml`: the entrypoint
  (`payload/solution.patch`), `max_changed_files`, the verifier command, evidence cap and metrics cap
  (`[verifier] max_metrics_bytes`, the largest metrics dict the evaluator may return), no
  Internet, the traffic-audit limits, the `[pumpkin]` semantics. They are generous on
  purpose. A task overrides a key by setting it in its own table (a GPU task sets its SSH
  session budgets in `[network.audit]`).
- **Derived**: the category is the task's folder (`problems/<Category>/`, one of FrontierCS-2
  main's folders; the definition's `family` is the same value); the datasets are `datasets/train.json` and
  `datasets/eval.json` (or `.tar`), hash and size computed; the submission is always a patch.

`scripts/task_manifest.py show <task_dir>` prints the complete effective definition with
each value's origin (`task.toml`, `default` or `derived`), like `helm template`; the build
writes the same file to `.frontier-generated/effective-task.toml` and the Verifier image
holds it at `/opt/frontier/effective-task.toml`.

Rules that the schema cannot express:

- `id` equals the directory name.
- `version` changes whenever judging changes (the source hash changes on any edit anyway).
- Recipes name the shared images only by build argument: `ARG AGENT_BASE` / `ARG VERIFIER_BASE`
  / `ARG GPU_TOOLS` (global, no default) and `FROM ${AGENT_BASE}` (Agent), `FROM ${VERIFIER_BASE}`
  (Verifier); the build injects `tasks/shared/bases.json`, so a base release moves every task with
  one bases.json change (checked; a literal shared reference is refused). Besides a shared base
  stage (`FROM ${GPU_TOOLS} AS frontier_gpu_tools`, and in a Verifier recipe
  `FROM ${VERIFIER_BASE} AS frontier_verifier`), only a Verifier recipe may `COPY --from=` another image,
  and only one named by full repository and digest
  (`COPY --from=docker.io/<ns>/<repo>@sha256:<64 hex> ...`, never a tag), for task-private
  judge data or toolchains too large for the task tree. An Agent recipe never may.
  There is one current set of shared images for every task: a base change moves all tasks at
  once (`scripts/build_base_images.py`), so tasks never run on mixed bases.
- `quota.max_train_calls` is stated in `README.md` as "**N train calls**" (checked). Keep the
  other quota numbers in `README.md` equal to `task.toml` by hand.
- When `allowed_paths` names a directory, set `submission.max_changed_files` to the limit the
  statement gives the solver; the default only fits a list of single files.
- Train and eval datasets differ.

### `README.md` (Agent-facing)

Only what is specific to this task: a pointer to the statement, `prepare_workspace.sh`, which
file(s) to edit, and any task tool the Agent should know about. Write nothing about submitting,
train feedback, quotas or finals: the build appends the shared platform text
(`tasks/shared/agent/README.platform.md`) with this task's numbers from `task.toml`, and the
Agent image holds the result at `/frontier/task/README.md`
(`.frontier-generated/agent/README.md`; `scripts/task_manifest.py show` prints the definition,
`task_manifest.agent_readme()` the README).

### `repo/`

The starter tree, read-only at `/frontier/task/repo`; the solver copies it with
`prepare_workspace.sh` and submits `git diff` via `make_submission.sh`. Every
`submission.allowed_paths` entry exists here. Keep it buildable as shipped and give it the
same build command the judge uses (here the `Makefile`).

### `repo/README.md` (the statement)

Task facts only, in a fixed shape (checked): exactly one `# ` title with a short opening, then
the `## ` sections `Task`, `What you may change`, `Interface`, `Correctness`, `Workload`,
`Scoring` in that order, then optionally `Running locally`, `Environment`, `Notes`, and nothing
else; at most 1,600 words (a warning above 1,200). The Scoring section must agree with the
scoring note in `judge/evaluator.py` and contain a paragraph starting `**Beats the reference**`
(per-workload comparisons and tolerances), which the generated README's Objective points to. Platform facts (submitting, quotas, time budgets, feedback
format, rejection categories, the objective/priority rule) are in the generated README and are
never repeated here or in `README.md`, which has no scoring, objective or priority section.

### `judge/evaluator.py`

`evaluate(tree)` returning `(score, score_unbounded, message, metrics)`; the comment under the
template evaluator's docstring is the contract. The module docstring is the task's one
scoring note, at most about 25 lines: (a) how `raw_score` and `score_unbounded` are computed,
(b) each beat-spec metric in one line (what is measured, better direction, when a workload
passes), (c) the `beats_reference` rule in a sentence or two. The validator requires it. The
points tasks get wrong:

- The suite comes from `FRONTIER_SUBMISSION_ROLE == "final"`, never from the dataset file.
- A bad submission is a scored zero (`return 0.0, 0.0, reason, metrics`); raise only for the
  judge's own faults (the reference fails, a fixture is missing).
- `metrics` always carries `beats_reference` and `beat_spec_workload_passes` (one entry per
  workload of the evaluated suite, the same beat-spec metrics in each: the rows define the
  metric names, and `beats_reference` is the authority), and the finite numbers
  `reference_score` and `margin`, **also on a scored zero**: the shared verifier rejects a
  final result without them.
- Measure candidate and reference in the same run, interleaved, with the same build.
- Everything else in `metrics` is private evidence unless `task.toml` `[feedback]` names it.
- `tree` is already the pristine starter with only the allowed files overlaid; the evaluator
  never applies a patch.
- No network, uid 65532, read-only root: build and write only under `TMPDIR`.
- Run submitted code (an interpreter that imports it, a binary built from it) with the shared
  launcher, `import isolate; isolate.run(argv, timeout_seconds=..., tree=...,
  inputs=..., read_paths=..., outputs=...)` (tasks/shared/lib/isolate.py, in the Verifier
  base; its docstring is the contract), not with plain `subprocess` or in the judge's own
  interpreter: the judge runs as the same uid, so file modes do not hide its data. Each run is a
  fresh instance that reads only the system, its inputs and the named paths, writes only its
  own TMPDIR/HOME, and leaves nothing running. Never copy or fork the launcher into a task.
  `isolate.start` runs a long-lived worker over pipes to the judge. When a trusted program that
  is not the judge's Python spawns the submitted program itself and measures it by its child's
  exit status and wait4 resource usage (a C harness), it spawns a small Python program that
  calls `isolate.exec_on_stdio(argv, failure_record=PATH, ...)` instead: that process becomes
  the instance's launcher on the harness's own pipes (fds 0 and 1), ends with the command's exit
  code or signal, whatever it is, and its resource usage includes the command's plus a fixed
  launch cost. A failure of the launcher itself is never an exit status: it writes PATH (a new
  file the judge's side owns) and the judge treats a run with that record as its own failure.

### `task.toml` `[feedback]` (what train returns)

A train result returns `raw_score`, `metrics.score_unbounded`, `metrics.beat_spec_progress`
(computed by the shared verifier from `beat_spec_workload_passes`; the evaluator never builds
it), `metrics.rejected` on a rejection, and the evaluator metrics `[feedback]` names, each with
the one sentence the generated README shows:

```toml
[feedback]
solved_count = "Distinct instances whose submitted answer passes every check."
```

Each sentence describes the metric neutrally ("Candidate latency per train workload, in seconds;
sorted."), never addressed to the Agent ("Your latency ..."). Names are lower_snake_case and never private (anything with `reference` or `margin`, starting
`beats`, or a shared field name is refused); each must be a literal in the evaluator. Default:
none. There is no `judge/beat_spec.json`: the validator refuses one.

### `judge/reference.patch`

The reference the candidate is compared against, as a diff of `repo/`. It is private: it is
in the Verifier image only.

### `datasets/train.json`, `datasets/eval.json`

Selector objects: `schema_version`, `problem_id`, `mode`, `selects`, `suite_location`. The
suites themselves are in `judge/`. A task whose data is large uses a tar dataset
(`media_type = "application/x-tar"`) instead.

### `docker/agent.Dockerfile`

`FROM` the base digest, install packages, `COPY README.md`, `repo` and
`.frontier-generated/agent/submission-policy.json`, set read-only modes, end as
`USER 10001:10001`. Nothing from `judge/` (checked). No `ENTRYPOINT`/`CMD`.

### `docker/verifier.Dockerfile`

`FROM` the base digest, install the judge's packages (pinned), `COPY judge` to
`/opt/frontier/task`, `repo` to `/opt/frontier/task/pristine`, the generated `config.yaml`
and `verifier.json`, install the shared `submission_patch.py`, end as `USER 65532:65532`.

### Optional files

- `docker/agent-gpu.Dockerfile`, `docker/verifier-gpu.Dockerfile`: with `[gpu.*]` and
  `[accelerator]` in `task.toml`. Each GPU task keeps its own CUDA/PyTorch image and copies
  the shared GPU tools (SSH bootstrap, GPU adapter) from `gpu-tools` in `bases.json`:
  `ARG GPU_TOOLS`, `FROM ${GPU_TOOLS} AS frontier_gpu_tools`, then `COPY --from=frontier_gpu_tools`
  (a GPU Verifier takes the shared verifier modules the same way, from
  `FROM ${VERIFIER_BASE} AS frontier_verifier`).
  The control `docker/verifier.Dockerfile` copies the adapter
  (`/opt/frontier/shared-gpu/fcs2_verifier_gpu.py` to `/opt/frontier/fcs2_verifier_gpu.py`)
  and the generated `verifier.json`, and sets `FRONTIER_VERIFIER_GPU_TYPE` and
  `FRONTIER_VERIFIER_GPU_COUNT` (never `FRONTIER_VERIFIER_GPU_SECONDS`, which is retired).
  `[gpu.verifier]` holds the Driver quota (`max_gpu_instances`, `max_gpu_seconds`) and the
  shared GPU adapter's settings, which go to `verifier.json` only (the module docstring of
  `tasks/shared/verifier-gpu/fcs2_verifier_gpu.py` is the contract):
  - Without a command test sandbox (below) the placement is `"runner"` (derived, never
    written): the whole verification runs on one runner with `FRONTIER_VERIFIER_GPU_COUNT`
    GPUs; the control image holds no judge.
  - `lease_seconds = { train = N, eval = M }`: the runner lease per mode (default: both
    `quota.eval_timeout_seconds`).
  - Every runner command gets `FRONTIER_VERIFIER_LOCAL_DEADLINE`, a `time.monotonic()` value on
    the runner: the end of the granted lease for verify.py, the end of its limit for a command
    sandbox step. A judge with long runs bounds itself by it, leaving a few minutes to write and
    return its result; a candidate timeout the old judge scored 0 must still be a scored 0
    before that deadline.
  - Submitted code on a runner also goes through the shared launcher: the Verifier GPU image
    takes it from gpu-tools (`COPY --from=frontier_gpu_tools
    /opt/frontier/shared-gpu/isolate.py /opt/frontier/`, then `install_frontier_pth.py` so the
    image's Python imports it by name), and on the runner (gVisor: no Landlock) `run`/`start`
    build the `userns` backend. `devices=["nvidia"]` gives an instance the GPU,
    `network="loopback"` a private `lo` (vLLM), `start` a long-lived worker over pipes.

- Test sandboxes, `[test_sandboxes.<name>]`: machines or containers beside the judge (the FCS
  paper's term). Every table has `dockerfile`, `workdir` and `trust` (`"untrusted"`, the default,
  or `"trusted"`). Two kinds, by `access`:

  `access = "command"` declares a machine of its own the judge runs commands
  on: a GPU runner or a CPU machine, the same protocol, API and journal; only `resources`
  differs. The judge then runs on the CPU control image (for a GPU task derived
  `placement = "control"`, `[gpu.verifier]` keeping only the quota); the control image holds
  `judge/` and `pristine/` like a CPU task and takes the shared adapter from a `${GPU_TOOLS}`
  stage (checked), with `openssh-client` and `libnss-wrapper`. Keys: `trust`, `dockerfile`,
  `workdir` (below `/workspace/`: the upload lands and steps run there), `resources`,
  `max_concurrent` and `lease_seconds = { train, eval }` (at least the steps plus upload,
  collect and per-step grace; at most `quota.eval_timeout_seconds`).
  - A GPU runner: `resources = { gpus, gpu_type }`; `dockerfile` is `docker/agent-gpu.Dockerfile`
    (the judge-free Agent GPU image) or, trusted only, `docker/verifier-gpu.Dockerfile`. The sum
    of `max_concurrent x gpus` is at most `[gpu.verifier] max_gpu_instances`.
  - A CPU machine: `resources = { cpu_millis, memory_bytes }`, exactly that machine (whole
    cores, whole MiB; reserved = capped, so every machine of one sandbox is the same machine
    type and a candidate and its reference are timed alike); `dockerfile` is
    `docker/agent.Dockerfile` (the Agent image: the candidate's own environment, no judge) or,
    trusted only, `docker/verifier.Dockerfile` (the Verifier image). The build adds the shared
    runner layer (`tasks/shared/images/cpu-runner.Dockerfile`: sshd, the SSH bootstrap,
    `/opt/frontier/task-runtime`); nothing else is needed. The task's CPU machine quota is
    derived (`max_concurrent` machines at once, each for a whole verification); a task needs
    no `[gpu]` table for it. A step's argv is `["/opt/frontier/task-runtime", program, ...]`
    to run with the image's environment (sshd clears it). A step that needs a terminal opens
    its own pty (`script -qec CMD /dev/null`, Python `pty`); steps have no network.
  A sandbox task's judge holds the Driver bearer in its process (fcs2_verifier `_sealed`, for
  run_sandboxes()): it never imports or runs candidate code in-process on the control image;
  candidate code runs only on the sandbox machines (or, on the control image, through
  `isolate.run`).
  `evaluate` calls `fcs2_verifier_gpu.run_sandboxes([SandboxRun(name, files, steps, outputs)],
  into=dir)`: one runner per run, exactly `files` uploaded, each `SandboxStep(argv, seconds,
  env)` killed at its `seconds` (a reported `timed_out`, not an error), declared outputs back;
  the last step may set `seconds=None` to take the rest of the runner's grant. The two classes:
  - class 1, `trust = "untrusted"` (ragen_multi_turn_rl, nano_world_model,
    cute_crest): the candidate's own machine, on the Agent (GPU) image. For output-scored
    work with a deadline. Nothing on it is trusted, including any judge script uploaded beside
    the candidate: the judge validates the returned files as untrusted input and recomputes the
    score on the control image from data that never leaves it, never from a number the
    candidate's code reports. A timeout, a non-zero exit, a lost session or malformed output is
    the candidate's scored zero; the same on the reference's run is the judge's fault (raise).
  - class 2, `trust = "trusted"` (fast_kv_compaction, polygon): a machine of
    the Verifier (GPU) image carrying the judge's toolchain and the shared isolate launcher. The
    judge may upload its own harness; the harness runs candidate code through `isolate.run`
    there (timing-scored work belongs here, measured from outside the candidate's process) and
    its output is trusted. Its failure is the judge's fault, never a score; a long harness takes
    `seconds=None` and budgets itself by `FRONTIER_VERIFIER_LOCAL_DEADLINE`.
  A task usually needs one class.

  `access = "service"` (the default; vllm_llm_serving_optimization, no template) declares a
  container the platform starts beside a GPU runner of the Verifier or the Agent (a "sidecar";
  `dockerfile` is `docker/test-sandbox[-<x>].Dockerfile`; only a trusted one may hold judge
  secrets such as `attestation_key`). It is either
  - one sandbox: `args`, `phases` (`agent`/`train`/`eval`), optional `attestation_key`; sidecar
    and image `<name>` (e.g. a judge's scheduler service), or
  - a template: `instances = { dataset_key = "...", manifest = "judge/....json" }` and
    `[test_sandboxes.<name>.roles.<role>] args = [...]` with `{instance}`. One sandbox per
    instance id that `datasets/{train,eval}.json` select in their list `dataset_key`, running
    in the modes whose dataset selects it; the manifest's
    `"test_sandboxes": {"<name>": {"<id>": {"index", "base", "build_args"?}}}` gives each its
    image slot, its digest-pinned base image and extra `TEST_SANDBOX_*` build args. The image
    is built once per instance with `TEST_SANDBOX_BASE` (used in a `FROM ${TEST_SANDBOX_BASE}`
    stage), `TEST_SANDBOX_INSTANCE` and those build args (a per-instance variant selects a stage,
    e.g. `FROM source-${TEST_SANDBOX_SOURCE}` with a global `ARG TEST_SANDBOX_SOURCE=declared`);
    sidecars `<name>-<index>-<role>` on image `<name>-<index>`.

  Images are published as `fcs2-task-<id>-sidecar-<image>`. `load_policy` returns the expanded
  `sidecars` mapping; `task_manifest.py show` prints only their count.

  `access = "command"` (the default is `"service"`, all of the above) declares instead a machine
  of its own the judge runs commands on. The
  judge then runs on the CPU control image (derived `placement = "control"`; `[gpu.verifier]`
  keeps only the quota, and the control image holds `judge/` and `pristine/` like a CPU task).
  Keys: `trust`, `dockerfile` (`docker/agent-gpu.Dockerfile`, the judge-free Agent GPU image,
  or trusted only `docker/verifier-gpu.Dockerfile`, an image carrying the judge's harness),
  `workdir` (below `/workspace/`: the upload lands and steps run there),
  `resources` (`{ gpus, gpu_type }`: a GPU runner; `{ cpu_millis, memory_bytes }` without
  `gpus`: a CPU machine, declared in the schema but refused until its provisioning path exists),
  `max_concurrent` and `lease_seconds = { train, eval }`.
  `evaluate` calls `fcs2_verifier_gpu.run_sandboxes([SandboxRun(name, files, steps, outputs)],
  into=dir)`: one runner per run, exactly `files` uploaded, each `SandboxStep(argv, seconds,
  env)` killed at its `seconds` (a reported `timed_out`, not an error), declared outputs back;
  the last step may set `seconds=None` to take the rest of the runner's grant (a trusted
  harness that budgets itself by `FRONTIER_VERIFIER_LOCAL_DEADLINE`).
  Untrusted (candidate-only) sandboxes are for output-scored tasks with a deadline; a
  timing-scored task keeps its trusted harness on the runner (a trusted sandbox, whose judge
  may put its own bundle in `files`; fast_kv_compaction).
- `verifier/verify.py` with a non-default `[verifier] command`: a standalone Verifier, only for
  a judge the shared verifier cannot express (vllm_llm_serving_optimization). Everything else
  in this list still applies.
- `judge/candidate-diagnostic.json` + `judge/candidate_failure.py`: a pinned private
  validator that turns static rejections into one fixed public category (see edlib).

## Checks before review

```sh
make validate                          # layout and contract checks
make test                              # includes tests/test_template_task.py
scripts/task_build_context.py tasks/.../<task_id> /tmp/ctx
docker build -f /tmp/ctx/docker/agent.Dockerfile    -t local/<task_id>-agent /tmp/ctx
docker build -f /tmp/ctx/docker/verifier.Dockerfile -t local/<task_id>-verifier /tmp/ctx
scripts/run_task_verifier.sh local/<task_id>-verifier payload.tar tasks/.../datasets/train.json train /tmp/out-train
```

Make the payloads inside the Agent image with the solver's own tools
(`prepare_workspace.sh`, edit, `make_submission.sh`, `tar` as `README.md` shows), and run the
Verifier on the starter, the reference and a wrong answer in both modes. Expect: the
reference scores about 50 (the score formula should give it a known value), a wrong answer is
a scored zero in both modes, and an eval result carries `private_metrics`.
