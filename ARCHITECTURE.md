# Architecture

A map of FrontierCS 2's code: where each thing lives, and what stays true across it. The rules
for changing the code are in the guides in [`docs/dev/`](docs/dev/). Names in backticks are for
symbol search.

## Bird's-eye view

FrontierCS 2 is a benchmark of research tasks and the tooling that installs an evaluation site
for them. A task is a directory: a real repository the Agent starts from, a `task.toml` that is
its only definition, Dockerfiles for its Agent and Verifier images, workload datasets, and a
judge whose `evaluate(tree)` scores a submitted patch. Tooling reads the task format, builds each
task into images tagged by the task's source identity (a hash of its git-tracked files), and
publishes the tasks to a site's Management service, together with the Candidates (a model plus
its harness) that run on them.

The source identity is the boundary that matters most: the files in a task directory are exactly
what runs, and changing any of them, a comment included, makes a new task version whose images
must be built and pinned again.

## Entry points

`make` is the front door (`Makefile`): `make ci` runs every check, `make test` the unit tests and
`make validate` the offline validation of tasks, catalog and site configuration. `./python` and
`./x` run repository code on the pinned interpreter and tools (`scripts/bootstrap-uv.sh`,
`scripts/repo.py`). The operator's command line is `scripts/frontier_deploy.py`.

## Code map

Each module depends only on modules above it. **Architecture Invariant** callouts often name
absences.

**`tasks/shared`.** What every task builds on: the base images (`images/`), the shared Verifier
(`verifier/`, `fcs2_verifier`) and GPU adapter (`verifier-gpu/`), the task-side libraries
(`lib/`: `isolate`, `edit_scope`, `task_runtime`), the defaults for every value a `task.toml`
leaves out (`task-defaults.toml`) and the current base image pins (`bases.json`).
- **Architecture Invariant:** it knows no task and no site.

**Task format.** In `scripts/`: `task_manifest` parses `task.toml` and renders the files
generated from it, `source_tree` and `render_task_definition` compute a task's source identity,
`verifier_layout` is the one definition of the Verifier container's layout, and
`validate_task_layout`, `task_build_context`, `import_task` and `task_selection` check, stage,
admit and select tasks. `task_layers` says where tasks and their catalog rows live: the shared
tree's (`tasks/problems/`, `config/`), plus any layer a top-level directory declares in its
`layer.json`. Like a schema crate, it is the one place a format is parsed.
- **Architecture Invariant:** it reaches no network, no Docker engine and no cluster.
- **Architecture Invariant:** the identity algorithm (`preview-source-identity/v1`) is frozen:
  a change re-identifies every task.

**Image build.** In `scripts/`: `build_base_images`, `build_task_images`,
`resolve_catalog_digests` and `native_datasets` turn a task into images pinned by digest.
- **Architecture Invariant:** it knows no site: an image is the same wherever it runs.
- **Architecture Invariant:** a published tag is never overwritten with different content.

**`tasks/problems/<Category>/<task>`.** One directory per task; `tasks/template/` is a complete
example task that covers every `task.toml` feature.
- **Architecture Invariant:** a task is self-contained: it depends only on `tasks/shared`,
  through `bases.json`, and no task imports another.
- **Architecture Invariant:** a site never changes a task's files: it selects its subset of
  tasks (`task_selection`, [`docs/task-selection.md`](docs/task-selection.md)).
- **Architecture Invariant:** a task's tests are its own. Tests of one task now in `tests/` are
  moving into that task.

**Candidates.** The harnesses an Agent runs. Today a harness is an image pin in
`config/*-candidate-image.json`, with its build overlay in `overlays/`; they are converging on
`candidates/<harness>/`, one directory per harness.
- **Architecture Invariant:** a harness knows no task.

**Deploy.** `scripts/frontier_deploy.py` and the `scripts/preview/` package (`config`, `infra`,
`images`, `publication`, `acceptance`, `cli`; to be renamed `deploy/`), `terraform/`,
`runtime/` (the storage controller and the LiteLLM hooks) and `smoke/`. It installs and upgrades
a site and publishes tasks through the Management API. The package's own layering is in
[`docs/deploy.md`](docs/deploy.md).
- **Architecture Invariant:** it is the only module that touches a cluster or an AWS account.

**`config/`.** The task catalog (`task-catalog.json`: membership, image digests, references and
baselines), the task definitions rendered from the task trees (`task-definitions.json`, written
by `render_task_definition --all`), the site schema and its example (`site.example.json`), and
the tool pins.
- **Architecture Invariant:** `task-definitions.json` is generated; only the catalog is edited
  by hand, and only through the importer or a release.

**`tests/`.** The unit tests `make test` runs, with golden fixtures under `tests/fixtures/golden/`
that lock rendered manifests and CLI help byte for byte. `tests/deployment_test_guard.py` keeps
every test away from real state, the network and cluster tools.

## Cross-cutting concerns

Every check runs offline and from a clean checkout: `make test` and `make validate` need no
credentials, registry or cluster. Writing rules for code, tests and prose are in
[`docs/dev/style.md`](docs/dev/style.md), [`docs/dev/testing.md`](docs/dev/testing.md) and
[`docs/dev/writing.md`](docs/dev/writing.md).
