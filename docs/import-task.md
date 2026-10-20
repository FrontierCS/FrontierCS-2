# Add a native task

Import a task directory (the layout described in the README's
[How a task works](../README.md#how-a-task-works)) into a clean worktree of this
repository. The directory name must equal `policy.json.problem_id` (lowercase
letters, digits and underscores, starting with a letter).

```bash
python3 scripts/import_task.py /path/to/problem_id --plan
make import-task EXPORT=/path/to/problem_id
# Build, verify datasets, then publish using existing Docker authentication:
make import-task EXPORT=/path/to/problem_id PUBLISH=1 IMAGE_NAMESPACE=<your-namespace>
```

Use one of the last two commands: changing from a local build to publication
requires `--replace`, because publication is a different import option. The
importer never logs in, deploys, starts evaluations or writes site/state files.
Builds require Docker with BuildKit and network access; publication requires
existing Docker authentication with push permission in your namespace. Published-image adoption
only needs anonymous registry access. `--plan` is offline and writes nothing,
including no copy, receipt or registry lookup. It prints copy, definition,
catalog, image, baseline and validation actions.

When task source and generated definitions were merged before image publication,
`--replace` can reconcile that task's stale catalog row. It first tries normal
catalog validation; if reconciliation is necessary, it validates the old target
row structurally and against the TaskConfig schema, while validating every
unaffected row against its current definition and source tree. The replacement
still passes full staged source, catalog, schema and image checks. Other stale
rows remain errors; `--check` does not reconcile anything.

The CLI accepts:

```text
import_task.py EXPORT_DIR
  [--published-images manifest.json] [--images auto|published|build]
  [--publish] [--image-namespace NS] [--plan | --check] [--replace]
  [--no-reference | --reference-payload FILE]
  [--baseline-comparable "<operator note>"]
```

`auto` uses published images when a manifest is supplied and otherwise builds.
An invalid manifest is an error. `published` requires a manifest; `build` cannot
also receive one. `--publish` applies only to builds. The default namespace is
`joyemang33`; `EXPORT`, `PUBLISH` and `IMAGE_NAMESPACE` configure the Make target.

## Source checks and prerequisites

The importer, Make targets and test guard disable Python bytecode writes in both
the current interpreter and inherited child processes. Dynamic test imports must
not add `__pycache__` to locked publication or acceptance trees. Source inventory
checks remain strict; generated cache files are never ignored or excluded from
source-lock identity.

Every test imports `deployment_test_guard` before other runtime imports. Keep
that import exempt from automatic sorting so formatting preserves the guard's
required initialization order.

The renderer's frozen `preview-source-identity/v1` hashes the complete task tree
using normalized GNU tar. Hidden files, executable bits and `upstream.lock.json`
participate. Symlinks, special files, nonzero tar exits and any set `TAR_OPTIONS`
are errors. All native tasks use this algorithm; keep generated contexts and
receipts outside task trees. Resources, quotas and audit budgets come from the
exported policy without a Preview cap.

## Export contract

The input contains only regular files and directories, including these members:

```text
problem_id/
  policy.json
  upstream.lock.json
  agent-base/Dockerfile
  verifier/Dockerfile
  verifier/verify.py
  datasets/train.json
  datasets/eval.json
  judge/evaluator.py       # needed for a bound baseline
  judge/reference.patch   # optional reference candidate payload
  judge/result.json       # optional authoring scores
```

Policy schema is `frontier.frontiersmith-task-policy/v1`. Policy quota, agent,
verifier, submission, metadata, family, display_name and version feed the
existing renderer. Dataset body hashes must equal
`policy.metadata.datasets.{train,eval}.version`, using `sha256:<64 lowercase hex>`.
GPU roles are enabled by `gpu.agent.max_instances > 0` and/or
`gpu.verifier.max_gpu_instances > 0`; each enabled role needs its corresponding
`agent-gpu/Dockerfile` or `verifier-gpu/Dockerfile`.

Optional `policy.sidecars` declares same-host containers that the Driver starts
beside the GPU runner of each listed phase (Core `TaskConfig.images.gpu_sidecars`).
Each entry `{"args": [...], "workdir": "/abs", "phases": ["train", "eval"],
"attestation_key": false, "image": "<role>"}` runs the image role `image`
(default: its own name) built from `sidecar-<role>/Dockerfile`; the importer
builds and publishes `fcs2-task-NAME-sidecar-<role>` under the source tag and
records it in the catalog's `sidecar_images`, so several sidecars can share one
published image. Names and roles are lowercase DNS labels (`main` is reserved),
at most 250 sidecars per task (Modal's concurrent-sidecar cap per Sandbox),
each phase needs its GPU role enabled, and more than four sidecars on one
parent need `policy.gpu_parent_resources.{agent,verifier}` (`cpu_millis`,
whole-MiB `memory_bytes`) of at least 32 mCPU and 32 MiB per sidecar, which is
the provider's admission floor. Sidecar images are public
Modal-side images like the GPU images and are not mirrored into the cluster.

The importer refuses symlinks, special files, traversal in member paths,
unhydrated Git LFS pointers, files over 512 MiB unless Git attributes mark them
`filter=lfs`, secret-bearing policy fields, mismatched dataset hashes and
unreviewed/private-registry Dockerfile bases. It reuses the builder's approved
base list and permits public Docker Hub digest references. Public availability
is established by anonymous registry reads when adopting images.

The shared base parser accepts case-insensitive, indented `FROM`, optional
`--platform=...` and build-stage aliases. It joins instruction continuations
after removing full-line comments, so a comment's trailing backslash cannot
hide the next instruction. RUN/COPY/ADD heredoc bodies are skipped through
their terminators, including quoted delimiters and `<<-` tab stripping.
It accepts approved public tags and
Docker Hub digest references, including shorthand such as
`python:3.11-slim-bookworm@sha256:...` and `ubuntu@sha256:...`. Other mutable tags
and non-Docker-Hub bases are refused. An enabled agent or verifier GPU Dockerfile
that references `${FRONTIER_GPU_SSH_BASE_IMAGE}` triggers bootstrap resolution.

The input must already be hydrated; the importer does not download LFS objects. It does not
execute judge/reference.py or rewrite any exported bytes.

Published task images normally use `source-<full 64-character source hash>`.
A corrected build of the same source may instead use exactly
`source-<full source hash>-image-<first 32 lowercase hex characters of the actual image digest>`.
The full manifest digest, original source hash, role repository and anonymous
registry readback remain required. Arbitrary suffixes, a different source hash,
or a suffix that does not match that image digest are rejected. Publish the
corrected artifact under its new immutable tag; never overwrite the original
tag. This artifact tag does not change the rendered task source or image tag.

Place a published-images manifest **outside** the export tree. An extra file
inside the tree changes `preview-source-identity/v1`. Copying uses `copytree`
under `.generated/import/NAME/staged-*/`, alongside the proposed definitions and
catalog. Source identity, the renderer `--all` comparison, task-config JSON Schema
and catalog validation all run against those staged inputs before any build or
push and again before installation. Unchanged task trees are read-only inputs to
this validation. Existing definition rows
retain their exact JSON bytes; all catalog tasks are rendered again with the
same renderer used by `--all` and compared to the updated definitions.

## Publication and immutable tags

`config/task-catalog.json` is the only publication manifest (schema
`frontier.deployment-task-catalog/v2`); definitions remain generated from native
trees. Each task has agent, verifier, dataset and enabled GPU image descriptors,
plus `reference` and `baseline`. The generic `candidate` remains unchanged.

For namespace `NS` and task `NAME`, the importer uses:

| Role | Repository after `docker.io/NS/` | Tag |
| --- | --- | --- |
| Agent | `fcs2-task-NAME-agent` | `source-<64 hex export identity>` |
| Verifier | `fcs2-task-verifier-NAME` | same |
| Datasets | `fcs2-task-NAME-datasets` | legacy `v1`, or the exact task `source-<identity>` tag for a new dataset revision |
| Agent GPU | `fcs2-task-NAME-agent-gpu` | source tag |
| Verifier GPU | `fcs2-task-verifier-NAME-gpu` | source tag |
| Reference | `fcs2-task-NAME-reference` | source tag |

Local builds use the existing builder with `linux/amd64` and
`--provenance=false`. Dataset images have exactly one layer containing the
exact content-addressed train/eval selector bodies. Dataset verification must
finish before any push. Build contexts and reference Dockerfiles live outside
the source tree. Without `--publish`, catalog digests remain all-zero SHA-256
placeholders; the local build receipt is `.generated/import/NAME.json`.
Placeholders pass `make validate` but cannot be deployed or pass `--check`.

Publication checks every remote tag through the anonymous resolver before any
push and refuses differing digests. If Docker has no manifest RepoDigest for an
existing tag, it refuses rather than treating the Docker config ID as a manifest
digest. Use `--published-images` to adopt already-published images in that case.
An absent tag is checked again immediately before push. Docker Hub has no
conditional tag creation: serialize publishers using the same namespace/tags.
This is particularly relevant to the legacy dataset `v1` tag during replacement.
If selector bodies change, preserve the existing `v1` image and publish the new
single-layer dataset image under the task's exact `source-<identity>` tag. Supply
that tag in the published-images manifest. Import and catalog validation accept
only `v1` or the exact reviewed task tag; digest, platform and exact selector-body
checks remain mandatory. Arbitrary version labels and another source's tag are
rejected. An incremental source-only release re-tags an unchanged versioned
dataset digest for the new source after collision checks; it never overwrites a
different digest. The ordinary image builder still defaults to legacy `v1` and
must not be used to overwrite an existing dataset version.
After publishing, only the new task's descriptors are resolved; candidate and
other task pins are unchanged.

A `judge/reference.patch` produces a reference overlay unless `--no-reference`
is used. A task with only `judge/reference.py` has `reference: null`; prepare its
output separately and pass `--reference-payload FILE` to include it. An explicit
file must agree with reference.patch if both are present. When creating a reference,
the policy submission entrypoint must be under `payload/`; with `--no-reference`,
other safe relative entrypoints are allowed. The file is mapped to
that entrypoint, never copied into the native
tree. The frozen launcher adds that prefix during packaging, so the image puts
its member directly under `/opt/frontier/reference-payload/`. Both launcher and
payload are copied with mode 0555. Parity retains the existing launcher execution
(user `10001:10001`, `/workspace`, shared Driver readiness guard and task-authored Run deadline).
The guard polls the Driver's `/readyz` endpoint and starts the launcher only
after exact HTTP **204**. HTTP **503** and transient connection failures retry;
other HTTP statuses fail immediately. TLS validation and malformed protocol
responses are fatal. The task-authored Run budget includes this wait; the guard
adds no separate total deadline. Its first 503 and successful 204 emit these
fixed diagnostic lines, respectively:

```text
[frontier-reference] Driver not ready (503); waiting
[frontier-reference] Driver ready (204); starting launcher
```

## Adopt published images

```bash
python3 scripts/import_task.py /path/to/problem_id \
  --published-images /path/to/manifest.json
```

The manifest shape is:

```json
{
  "schema_version": "frontier.task-release-images/v1",
  "problem_id": "problem_id",
  "source_sha256": "sha256:<64 lowercase hex preview-source-identity/v1>",
  "images": {
    "agent": {"repository": "docker.io/NS/fcs2-task-problem_id-agent", "tag": "source-<identity hex>", "digest": "sha256:<manifest hex>"},
    "verifier": {"repository": "docker.io/NS/fcs2-task-verifier-problem_id", "tag": "source-<identity hex>", "digest": "sha256:<manifest hex>"},
    "datasets": {"repository": "docker.io/NS/fcs2-task-problem_id-datasets", "tag": "v1", "digest": "sha256:<manifest hex>"}
  }
}
```

Add `reference` when importing a reference payload and `agent_gpu` /
`verifier_gpu` for each enabled GPU role, each with the same three descriptor
fields. Missing and extra roles are errors. Descriptors may use another public
Docker Hub namespace, but must retain repository role names and tags consistent
with the definition. Digests must be the actual Linux amd64 leaf; a tag may point
to an index whose unique amd64 leaf matches that digest. The resolver verifies
manifest hashes, config blob hash/size/platform and the dataset single-layer
constraint. It also verifies dataset selector bodies and reference launcher/payload
bytes from the registry layers. Only those two files and their parent directories
are allowed; file modes must be 0555, and every parent directory must be present
and readable/traversable by all users (`a+rx`). Conflicting later entries, extra
files/directories and links are refused. Both the digest and tag must resolve to the same leaf. No Docker
commands execute in this mode.

## Baseline sources and classifications

`judge/result.json` must use `frontiersmith/result@1`, identify the task, and
contain `final.reference.ok=true`, finite `scores`, matching `samples` and
`spread.n`, and consistent `spread.mean` / `spread.spread`.
`final.reference.spread.rel` is also mandatory and finite: it must equal
`spread.spread / max(abs(spread.mean), 1e-12)` within `math.isclose`'s configured
`rel_tol=1e-12` and `abs_tol=1e-12`. The importer retains
those numeric values and binds the current source, datasets, evaluator, verifier
adapter and payload hashes. It uses `verifier/fcs2_verifier.py` when present,
otherwise `verifier/verify.py`. `cross-site-agreement/v1` is Preview's configured
rule (absolute 0.01, relative 0.05, one replay), not an inferred result field.

New imports default to `historical-authoring` and `comparable: false` because
result.json does not attest the measured export identity. If the operator has
confirmed that it measured this exact exported revision, use:

```bash
python3 scripts/import_task.py /path/to/problem_id \
  --baseline-comparable "Confirmed upstream result.json measured this exported revision"
```

This explicit operator assertion sets `comparable: true`, retains
`provenance: "historical-authoring"`, and stores the supplied nonempty text as
`comparability_note`. It changes no scores or binding hashes and requires a valid
result with complete baseline bindings. The note is part of import intent;
changing it on an existing import requires `--replace`. Never promote a Preview
replay into its own baseline. If no result exists, the exact default is:

```json
{"provenance": "unverified", "comparable": false, "comparability_note": "no judge/result.json in the export"}
```

A result lacking evaluator/payload bindings is also unverified; its valid
scores and available hashes are retained, and the note lists missing binding files. Invalid results
are refused. Reference payload hashes remain separately bound in the descriptor.

Tests preserve existing baseline numeric/input fields. Operator classifications
and measurement provenance cannot be reconstructed from `result.json` alone;
publication does not silently regenerate or reclassify existing baselines.

## Independent measurements outside the export

```bash
python3 scripts/import_task.py /physical/path/to/problem_id \
  --published-images /evidence/task-images.json \
  --measured-baseline /evidence/measurement.json --plan
# when replacing an existing catalog entry). Verify with the same inputs:
python3 scripts/import_task.py /physical/path/to/problem_id \
  --published-images /evidence/task-images.json \
  --measured-baseline /evidence/measurement.json --check
```

For a JSON reference, add `--reference-payload /physical/path/to/problem_id/judge/reference.json`.
This option requires exact published images and the native prepared-reference CPU
profile. It is mutually exclusive with `--baseline-comparable`. Legacy result.json
and unverified imports keep their existing behavior. It creates no measurement,
does not execute a candidate, and does not change the frozen identity algorithms,
scientific verdict or absolute 0.01 plus relative 0.05 parity rule.

The packet is an operator-maintained JSON integrity manifest with exactly three
fields: `schema_version: "frontier.independent-reference-measurement/v1"`,
`documents`, and `artifacts`. Every role below maps to exactly
`{"path":"relative/file", "sha256":"sha256:<hex>", "size_bytes":123}`.
Copy retained receipt/artifact bytes verbatim; do not synthesize past receipts.

| Document role | Retained native input |
| --- | --- |
| `task_create` | prepared release `create-task-version.json` including full author receipt and TaskConfig |
| `task_catalog` | actual TaskVersion catalog GET body, or bounded successful GET receipt with that body |
| `task_publication` | prepared release `publication.json` |
| `candidate_create` | adopted reference `create-candidate-version.json` |
| `candidate_binding` | original `reference-binding.json` |
| `candidate_adoption` | `adoption-receipt.json` including actual catalog readback |
| `evaluation_create` | actual independent authoring `create-evaluation.json` |
| `evaluation` | completed Evaluation GET body |
| `run` | completed, successful, scored Run GET body |
| `events` | complete retained Run event list with successful train and accepted final result |

| Artifact role | Retained actual bytes |
| --- | --- |
| `platform_evidence` | Driver final platform evidence archive |
| `train_evidence` | original verifier train artifact referenced by the train-completed event |
| `final_evidence` | original verifier final artifact referenced by platform `manifest.verifier_evidence` |

The original final artifact is needed even when the platform archive contains its
decompressed tar: the original descriptor hashes compressed bytes. Qualification
checks both the original hash/size and equality of its decoded tar with the nested
platform member. Recompression is not a substitute for retaining the original.
Raw tar and ZSTD tar are supported. Packet/document files are bounded to 4 MiB,
compressed artifacts to 64 MiB, expanded archives to 128 MiB, individual members
to 32 MiB, and member count to 256. Absolute/traversing packet paths, symlinks in
any path component, duplicate JSON fields, nonfinite JSON numbers, duplicate tar
members, links, special files, sparse members and extended tar metadata are errors.

The catalog stores `provenance: measured`, exact measurement IDs, packet hash,
original PASS/FAIL verdict, train score and final sample. Spread zero describes a
single sample, not repeatability. `--check` repeats qualification, compares the
entire derived baseline including classification, and binds every catalog image
to the measured manifest before normal registry verification. It fails if the
packet is missing or changed; there is no fallback to trusting a provenance tag.

## Re-run, check, and deploy

Identical inputs and options return `already imported, no changes`, without
rebuilding or changing any files, only when the gitignored receipt
`.generated/import/<name>.json` is present and matches. It records intent,
descriptor hashes and build information. On a fresh clone without that receipt,
even an identical export requires `--replace`. For re-imports, prefer
`--replace --published-images /path/to/manifest.json` to retain verified published
digests. With neither `--published-images` nor `--publish`, `--replace` builds locally
and leaves placeholder digests in the catalog; it does not preserve the old pins.
Different trees with the same name list changed
members and require `--replace`; publication-option changes also require it.
`--replace` updates the same catalog position. After staged validation succeeds,
the importer moves the old destination to `.generated/import/NAME/previous/`,
moves the staged tree into place, and atomically writes definitions, catalog and
receipt. Original bytes for both configuration files and the receipt are saved
before the swap. After a write or post-import check failure, rollback restores
the tree and older backup first, then attempts definitions, catalog and receipt
in that order. Every restoration is attempted even if another fails; the error
reports the original failure and all rollback failures together. A successful replacement retains the old
tree in `previous/`; a failed first import removes the newly installed tree.
Importing does not create a commit.

```bash
python3 scripts/import_task.py /path/to/problem_id --check
# If imported with --no-reference, repeat it even when judge/reference.patch exists.
make validate CONFIG=config/site.example.json
make validate CONFIG=config/site.json
make test
git diff --check
```

`--check` is read-only: it compares the imported tree and generated definition,
checks catalog bindings, rejects placeholders and verifies remote digests.
For the legacy result.json path, it compares every derivable baseline field to a fresh derivation, permitting only
`provenance`, `comparable` and `comparability_note` to differ. Duplicate JSON keys
in `judge/result.json` are refused.
Successful imports run `make validate CONFIG=config/site.example.json` and
`make test` in subprocesses and print the changed files. The importer never
creates or changes `config/site.json`.

The import integration test exercises two real imports, including both Makefile
gates and a repeat-import filesystem comparison. Its disposable repository uses
a focused unittest recipe for `make test`; the outer repository suite runs once,
without recursively repeating itself inside each import. The real importer and
release Makefile retain their full validation and test commands.

## Domain-grouped source directories

Existing tasks can live at `tasks/frontier-cs-2.0-demo/problems/<domain>/<task>/`.
Imports locate an existing task by its unique task name and replace it in the
same domain. New imports initially use the flat path; an operator may move the
complete task root into its paper domain after successful import. Do not edit
source bytes or regenerate identities merely to relocate a directory. Both
layouts work, but duplicate task names are rejected. See
the migration and import check.
