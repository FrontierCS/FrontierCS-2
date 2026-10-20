# Deployment operations

The current example-site Management/Driver pair includes
native cleanup after unconfirmed Driver creation
and the current native recovery runtime.
Publication alone does not clean historical attempts or authorize their retry.

## Clusterwise selected-final recovery and resource outcomes

An authenticated Agent OOM with no accepted final can receive score 0 / FAIL
through the existing no-final adjudication route. The console displays OOM while
preserving the original execution failure. Neither recovery is applied merely
by upgrading the runtime.

The Clusterwise verifier records private native failure observations: the trusted
candidate/reference phase, actual child return code, binary digest, fixed limits,
and cgroup memory counters before and after execution. Missing counters are
explicitly unobserved. These diagnostics do not change scoring, resource budgets,
or the measured timing interval; a return code alone does not establish OOM.

Clusterwise's shared prefix scans partition by the actual OpenMP team. A fixed
64-partition loop is invalid under the authored 60-thread ceiling and leaves
large-workload prefix entries unwritten. The solver and trusted reference share
the same corrected utility; all workload, timing and resource contracts remain
unchanged. Native reference failures do not establish a candidate FAIL.

## Verifier worker placement

For hardware-bound private datasets, configure an explicit
TaskVersion verifier worker allowlist.
The published platform pair enforces `verifier_placement_required_task_ids`
before registration, evaluation creation, replacement, retry and fresh dispatch.
Configure each exact qualified TaskVersion before admitting it. Existing frozen
attempts retain their placement; ordinary tasks can continue while a required
policy is missing.

## Change evaluation concurrency

The native `POST /api/v1/evaluations/{id}/parallelism` endpoint changes a running
evaluation's concurrency without adding tasks or changing its CandidateVersion,
formal marker, existing Runs or results. Supply `Idempotency-Key` and a closed
body such as `{"expected_max_parallel_runs":2,"max_parallel_runs":4}`. A stale
expected limit is rejected. Worker capacity remains an independent operator
setting; this action does not resize machines or interrupt active attempts.
See the pinned platform source's `docs/evaluation-task-extension.md`.

## Infrastructure observation on current runtime upgrades

Kubernetes installs and `upgrade-runtime` configure the Infrastructure page's
cluster and region from current validated site/Terraform outputs, even when
manual worker capacity is disabled. Management receives only node `get/list`
through its existing role. The upgrade preserves unrelated rules, uses both
role UID and resourceVersion guards, performs server-side dry-run before any
patch, and verifies role identity/rules after rollout. Simulator installation
omits this observation wiring.

## 1. Preflight, render, and plan

Load a short-lived AWS session through the standard credential chain, then:

```bash
python3 scripts/frontier_deploy.py preflight --config config/site.json
python3 scripts/frontier_deploy.py render --config config/site.json
python3 scripts/frontier_deploy.py plan --config config/site.json
```

Preflight prints the Terraform/AWS/kubectl/Docker executable paths, the STS
command invocation, and the returned account ID and principal ARN. It also
reports the registry copy tool as optional (or missing) for runtime upgrades;
absence does not fail preflight. Generated non-secret
files live under `.generated/<site>/`; local plans, kubeconfig, image records,
and smoke receipt live under `.state/<site>/`. `render` also writes
`task-selection.json` under `.generated/<site>/`; it is separate from the
checksummed deployment bundle.

The state bucket must exist before Terraform can produce a remote-state
platform plan. On the first pass, `plan` plans bootstrap and validates the
platform root without a backend.

## 2. Full apply

```bash
python3 scripts/frontier_deploy.py apply \
  --config config/site.json \
  --confirm-account 123456789012 \
  --confirm-site preview
```

The command performs account fencing, state/bootstrap apply, platform apply,
registry rollout, four deployment-repository builds, four exact runtime image
mirrors, digest readback, PostgreSQL/LiteLLM rollout, the speed-add dataset seed,
the Preview admission bootstrap, the final Management rollout, and a scripted
smoke Evaluation. If `config/task-catalog.json` is present, the command then
selects tasks using `features.modal_gpu`, mirrors selected CPU/dataset/overlay
images and reference candidates, retains selected public GPU descriptors,
sequentially recreates one digest-checking dataset Job per selected task, and
creates or reuses one Codex CandidateVersion plus selected TaskVersions before
final readiness checks. Any nonzero exit means the claimed
gate did not pass.

Task publication includes an author release receipt. The local-k8s Management
verifier independently checks the seeded EFS objects' sizes and SHA-256 values
through its read-only mount. `local-sha256:<hex>` in the receipt is an explicit
local content version, not an S3 VersionId; production S3 validation is unchanged.
Publication generation 2 creates new immutable task identities for this contract.

Release preparation may contain zero digests until Docker Hub publication and
anonymous readback complete. Placeholder refusal is enforced before AWS or
cluster mutation. Structural `validate` is not a publication gate.

The example selects CPU tasks and needs no Modal credentials. To enable GPU
tasks, set `features.modal_gpu=true` and load Modal credentials into the operator environment
before `preflight`/`apply`/`deploy`. The named platform Secret is delivered only
to trusted per-Run Drivers, including replacement cleanup Drivers. Existing
credentials are compared, not silently overwritten. Terraform must expose
`nat_gateway_public_ips`; an older platform needs its outputs refreshed before
the GPU runtime can render. This does not require GPU EKS nodes.

The runtime renderer also installs `frontier-management-modal-observer`, a
namespaced Role and RoleBinding allowing only the Management ServiceAccount to
read the exact `modal.credentials_secret` in `frontier-system`. This supports
the native Modal capacity/usage observer without giving Secret access to Agents
or exposing credentials through the API. The observer grant has only `get`, no
list/watch or mutation permission. A custom Secret name is taken from the site
configuration; do not copy a role from another installation or grant cluster-wide
Secret access. CPU-only sites render no observer role and need no Modal Secret.
New GPU installations receive this RBAC through the normal rendered runtime
manifest. Existing sites apply the updated runtime during a
[live site change window](#live-site-changes); no manual RBAC patch is required.
The Management image must implement the native observer API before its endpoint
is available; rendering this grant alone does not upgrade that image.

### Build the bundled task images locally

Installation pulls the catalog's public digest-pinned task images. Local task
builds require hydrated Git LFS objects (`git lfs pull`) and Docker with BuildKit.
The builder is useful for checking existing exports; adding or replacing a task
always uses [the importer](import-task.md).

```bash
make build-task-images
TASK_PROBLEM=sparse_video_attention_policy make build-task-images
TASK_PROBLEM=cpu make build-task-images
TASK_PROBLEM=graph_mining_method_selection TASK_IMAGE_PREFIX=joyemang33 make build-task-images
```

These commands build local `linux/amd64` images. The script
`python3 scripts/build_task_images.py --push` opts into registry publication
using existing Docker authentication; it never performs login. Build manifests
stay outside task trees, defaulting to `.generated/native-build.json`.
Building or pushing images does not update the catalog. Use `import-task` for
task release changes and never overwrite a published tag with different bytes.

For one release, `make build-demo-images`, `DEMO_PROBLEM` and `DEMO_IMAGE_PREFIX`
remain deprecated aliases. Either the old target or variables print one warning
line. `TASK_PROBLEM` and `TASK_IMAGE_PREFIX` take precedence when both names are
set. The script also accepts the old image-prefix environment variable for that
release, with a warning; explicit `--image-prefix` still takes precedence.

The builder uses each task's own Dockerfiles and root as its build context.
It freezes mutable Ubuntu, buildpack-deps and Python bases through BuildKit
image contexts and retains already pinned CUDA bases. The required public GPU
SSH bootstrap is digest-pinned. Agent/Verifier source tags follow the renderer;
dataset images contain exactly one layer with both content-addressed selector
bodies. A Docker-save check verifies those bytes before pushing. Source files
are never rewritten. GPU model images need substantial local storage; building
them does not allocate GPUs.

The compatibility flags `--incremental-from <hex-commit>` and `--crane` retain
the optional source-layer path. It verifies the prior Git/LFS source and all
selected changes before registry mutation, supports only reviewed COPY inputs,
and refuses dataset, Dockerfile, solver, reference payload, removal or executable
mode changes. Without `--push` it writes plans only. It does not install or
replace tasks; new exports still go through the importer.

Interruption is recoverable by rerunning the same command with the same site
identity and state key. The image publication tag is derived from source bytes,
and each push is followed by authoritative digest readback.

For an already-created platform:

```bash
python3 scripts/frontier_deploy.py deploy --config config/site.json
```

`build-images`, `smoke`, `seed-tasks`, and `verify` expose narrower recovery
steps. `smoke`
uses the persisted exact runtime image record; it does not rebuild or resolve
tags. The registry ingress policy explicitly admits task seed pods in
`frontier-system` with the deployment's task-seed name, owner, site, and part-of
labels on TCP 5000. This peer is separate from node image pulls and Driver
namespace access; temporary diagnostic policies are not publication receipts.
An explicit task-seed egress policy admits only the registry pods and the exact
registry Service IPv4 `/32` on TCP 5000, including AWS VPC CNI strict mode.
`seed-tasks` requires an already-running registry, seed PVC, and Management
service:

```bash
python3 scripts/frontier_deploy.py seed-tasks \
  --config config/site.json \
  --catalog config/task-catalog.json
```

To check existing identities before migration or re-seeding:

```bash
python3 scripts/frontier_deploy.py seed-tasks --check \
  --config config/site.json --catalog config/task-catalog.json
```

This command reuses `.state/<site>/kubeconfig` and `task-images.json`. It verifies
saved mirrors against the selected catalog, computes the Codex and selected
TaskVersion documents and GETs each ID from Management. Every ID reports
`MATCH`, `MISSING` or `DIFFERS`; any missing/different definition digest makes
it exit nonzero. It performs no POST, dataset Job, mirroring or `.state` write.
It needs the existing kubeconfig and normal access to Management, but no Modal
credentials. It does not prove dataset health or a successful Evaluation.

Successful seeding writes `frontier.deployment-task-seed-receipt/v2` to
`.state/<site>/task-seed-receipt.json`. It records the site, tracked and effective
catalog hashes, selection digest, ordered selected names and IDs, exclusions
with reasons, normalized model domains, Codex ID, and each task's ID and
`definition_digest`. Readers accept v1 by binding its `catalog_sha256` to the
current effective catalog; v2 binds both hashes and the selection digest.
See [selection contract](task-selection.md) and the
[live site changes](#live-site-changes) before upgrading an
existing site. Changing selection does not delete historical Management versions.

The following `gpu-smoke` command **needs updating before it can provide
acceptance for the current Sparse task**. Its implementation still invokes the
removed `/frontier/task/frontier-gpu` and selects Codex V2; Sparse requires the
Candidate-owned V3 helper at `/opt/frontier/candidate/bin/frontier-gpu`.
The intended procedure after that update uses a reviewed archived submission
patch (not private judge/reference code):

```bash
python3 scripts/frontier_deploy.py gpu-smoke --config config/site.json \
  --patch /absolute/path/to/solution.patch --patch-sha256 sha256:<exact-body-hash> \
  --attempt release-r001
```

If an exact clean FrontierCS2 checkout is already present, avoid a network
clone without putting its path in the site file:

```bash
FRONTIERCS2_SOURCE_DIR=/absolute/path/to/frontiercs2 \
  python3 scripts/frontier_deploy.py build-images --config config/site.json
```

The checkout HEAD must equal `application.commit`, have no tracked or untracked
changes, and contain both release Dockerfiles plus the complete
`skills/add-frontier-task` protocol and publication helper. Otherwise the build
fails before publication. These checks and Git read authorization apply only
when `application.require_source=true`; setting it to `false` skips the source
gate and disables only this task-authoring handoff.

When `application.require_source=true`, `apply` or `build-images` prints the
exact task-authoring skill path. With the default clone it is:

```text
.state/<site>/application-source/skills/add-frontier-task/SKILL.md
```

Use that frozen skill and its adjacent references/scripts for the platform
task-authoring handoff; it is from the same application commit as Management
and Driver. Importing its native export into Preview still uses `import-task`.

## Live site changes

Changes to an existing site (`make apply`, `seed-tasks`, `upgrade-runtime`, or
`upgrade-local-network`) require an operator window with submissions stopped
and no active Runs. Coordinate the window before changing shared resources.

1. Record the deployed commit and privately back up the complete
   `.state/<site>/` directory, including receipts, mirrors, Terraform data and
   recovery files. Keep credentials out of Git. This directory backup does not
   replace consistent [database backups](#5-backup-and-deletion).
2. Run `make test` and `make validate` with both the example and the private site
   config on the proposed release. Review site selection and authored budgets.
3. Run the GET-only comparison before mutation:

   ```bash
   python3 scripts/frontier_deploy.py seed-tasks --check \
     --config config/site.json --catalog config/task-catalog.json
   ```

   Review each MATCH/MISSING/DIFFERS result. New releases can intentionally
   produce MISSING/DIFFERS or fail saved-mirror validation before HTTP; establish
   the expected release difference before proceeding. A collection 404 is an
   error, not an absent version. Reads do not update the receipt or mirrors.
4. Confirm the site is idle, review `make plan`, then run the required change.
   Full apply uses the normal exact account/site confirmations. Require smoke
   and readiness success. After seeding, inspect the v2 receipt's tracked and
   effective catalog hashes, selection digest, domains, IDs and definition
   digests; require MATCH from a fresh `seed-tasks --check`.
5. For rollback, preserve failed migration state privately and restore the
   previous checkout and complete saved state together. Restoring local files
   neither deletes immutable Management versions nor reverses cloud changes;
   reconcile changed runtime resources in the same operator window.

Readers retain v1 receipt support. The [migration fixtures](golden-fixtures.md#live-state-migration-fixtures)
prove historical identity reconstruction and rehearse exact-byte receipt rollback
in temporary directories. They do not establish live health. Reference parity
and model-enabled scoring remain separate acceptance gates.

### Upgrade the runtime or local network controller

The reviewed public-reference route repair is a release-bound compatibility path.
When both live digests equal the configured pins, the tool preserves the complete
Docker Hub tuple for Core `78905901bccc1d51301961cafc4fb638ec6d6c8b`:
Management and the LiteLLM key worker use
`docker.io/joyemang33/fcs2-management@sha256:433ae9f0ce2f04240554d27f86973bfc5af4aa8d3753572bcc5c7ea10d04bf95`,
and Driver uses
`docker.io/joyemang33/fcs2-driver@sha256:8613f2976870dabf40eb9b344e35a4b691cd14816b3b5f808ce3b77e86ab2442`.
All configured repositories, digests and `sha-<commit>` tags must agree. Unknown
or mixed same-digest references and direct image environment overrides stop the
upgrade before registry writes. This path skips mirroring and records the public
references in both the accepted receipt and validated `runtime-images.json`.
Ordinary mirrored releases and upgrades to different digests retain their existing
path.

In this path, the only effective Management ConfigMap/Deployment changes allowed
are the discovered API CIDRs, their derived configuration hash, and the existing
`gateway-artifact-retention` initializer image. That initializer may transition
from exactly
`172.20.0.50:5000/frontier/management@sha256:b8e439ba2e1427776b0cd4b03529c40b07cf6dfadf73fae1003b5c9f218486c9`
to the public Management reference above. Every non-image field must match;
only Kubernetes' default termination-message path and `File` policy are normalized.
Missing initializers, unknown images, or additional effective configuration,
resource or mount changes stop the repair. The operator must still reconstruct
and compare the actual site configuration and all generated resources before the
idle upgrade: this exception does not authorize other RBAC, gateway, Secret or
network changes. Keep the backup, idle guards, server dry-runs, readback and normal
smoke/readiness/verification requirements below. Merge the reviewed tool change
before using it on a site.

For a Driver-only upgrade, an unchanged Management digest already deployed under
`<site-registry>:5000/frontier/runtime/management` can retain that exact reference.
The requested public source repository, commit tag and digest must match the
persisted native record, and Management, its key worker and retention initializer
must agree without direct image environment overrides. The tool verifies the
native cached manifest and mirrors only Driver. Additional effective ConfigMap,
Deployment or infrastructure-role changes reject this path, apart from the derived
checksum and the two exact initializer defaults described above. On Modal sites,
the existing observer Role/RoleBinding must match the desired permissions and
metadata; they remain untouched and their snapshots are rechecked before writes.
This compatibility path does not establish equality of other RBAC or network
resources: the complete operator plan and all normal deployment gates still apply.

Prerequisites: establish the [idle operator window and state backup](#live-site-changes)
and install `crane`, or set `FRONTIER_REGISTRY_COPY_TOOL` to a compatible
executable supporting `copy --platform linux/amd64 --jobs 2 SOURCE DESTINATION`.
`upgrade-runtime` requires this tool even when images are cached;
`upgrade-local-network` requires it on cache misses.

```bash
python3 scripts/frontier_deploy.py upgrade-runtime --config config/site.json
```

This checks active/queued Run states, mirrors and verifies the pinned Management
and Driver images, then uses resourceVersion-guarded patches for the Driver pin,
Management image and configuration checksum. This is a runtime-only operation:
it does not install, reconcile or republish the LiteLLM gateway or its Secrets.
Before mutation it requires the existing serving receipt, PostgreSQL/PVC readiness,
and Ready endpoints owned by the Deployment selected by the live Service, running
its pinned image. Blue-green selectors, provider credentials and audit/config files
are preserved, with stable identity/spec/data snapshots checked again before
Management mutation and after rollout. Private gateway snapshots (which contain
Secret data) are written with mode 600 inside the mode-700 upgrade receipt directory.
Volatile observation metadata and status are excluded from equality checks.
Missing/unready gateways fail closed; provision them separately with `deploy`, or
use the explicit gateway blue-green procedure for a gateway-only update.
Other Management configuration is preserved and checked after rollout. GPU-enabled
upgrades also install the exact Modal observer Role/RoleBinding, after
server-side dry-run and a fresh idle check, before restarting Management. The
upgrade rejects a live namespace, credential Secret name, or ServiceAccount
that differs from the site configuration. All upgrades also apply the diagnostic
ClusterRole/ClusterRoleBinding described below; the complete runtime manifest is
not reapplied. CPU-only upgrades add no Modal permissions. The rendered RBAC manifests are retained with the
private runtime-upgrade receipt. The upgrade does not cancel Runs, repeat
bootstrap, seed tasks or run model evaluations. Run
`make verify` and the model/audit/key-cleanup smoke after the
upgrade.

Publish the storage controller from its clean Preview source commit before
upgrading it. The image receipt must bind the checked-out
`runtime/storage-controller` directory hash and that directory's bytes at the
receipt's Git commit, with the matching `sha-<commit>` tag. Then use that receipt:

```bash
python3 scripts/frontier_deploy.py upgrade-local-network --config config/site.json \
  --image-receipt config/storage-controller-image.json
```

The operator must establish the idle window: `upgrade-local-network` does not
query Runs or check that the site is idle. It replaces the controller image and registry Service
environment value through guarded patches and full Deployment readback. The
controller permits the configured registry Service IPv4 `/32` on TCP 5000 for
the exact Run/attempt Driver before exposing its object-store PVC. It leaves
the upstream Run policies and Agent egress policy intact.

## Reference parity acceptance gate

After the operator runs `make apply`, use the Management endpoint and the
local seed/mirror receipts from that same site:

```bash
python3 scripts/reference_parity.py --all --config config/site.json \
  --management-url http://127.0.0.1:8080 \
  --seed-receipt .state/preview/task-seed-receipt.json \
  --receipt .state/preview/reference-parity.json
```

`--task <name>` can be repeated instead of `--all`. Other overrides are
`--catalog`, `--mirrors`, `--attempt` (default
`1`) and `--poll-seconds` (default 15, at most 60). The default mirrors file is
`task-images.json` beside the seed receipt. Reference images use the same
in-cluster mirroring helper as Codex, but are stored separately from the generic Codex
`candidate` within each catalog task.

Evaluations are created with the idempotency key
`parity-<task>-<candidate_version>-<attempt>`. Re-running a task whose
TaskVersion and reference candidate did not change (for example after a
re-seed that only touched other tasks, or after a CPU pool change) returns the
previous Evaluation and its old score unless `--attempt` is raised. Bump
`--attempt` and use a new `--receipt` whenever a genuinely new replay is
wanted; receipts with the same attempt are resumed, not repeated.

The harness verifies baseline/source/dataset/launcher/payload bindings, the
catalog hash in the seed receipt, task mirror source digests, the seeded
TaskVersion ID and Management definition digest before evaluation. Execution
and identity metadata are hashed before CandidateVersion creation; identical
409 conflicts are idempotent. Evaluation keys are
`parity-<task>-<candver>-<attempt>`.

The full tracked catalog is validated before site selection. V1 seed receipts
bind the current effective catalog hash; v2 receipts bind the full and effective
catalog hashes, selection digest and per-task definition digests. A mismatch
reports the receipt/catalog paths and conflicting binding. Use the same
`--config` (default `config/site.json`), catalog and receipt for seed and replay.
`--all` replays selected reference tasks; `--task` for an excluded task fails
before HTTP. Do not prune tracked catalogs for a CPU site.

A pass requires Evaluation `COMPLETED`, Run `COMPLETED` / `SUCCEEDED`, a finite
numeric score, and a Driver `VERIFIER_TRAIN_COMPLETED` event with status
`success`. It requires `abs(score - historical_mean) <= 0.01 +
0.05 * abs(historical_mean)`. This rule compares the replay with its bound baseline; it does not estimate
within-sample timing variance.

The `quota.wall_time_seconds + quota.verifier.eval_timeout_seconds + 900`
deadline covers answering, independent final and cleanup, and is checkpointed before
the first Evaluation POST. A saved deadline without a submission marker or
Evaluation ID is unsubmitted work and is rejected after expiry. Immediately
before POST, `submission_state=SUBMITTING` marks an uncertain outcome; resume
reissues the identical request and idempotency key to recover its Evaluation,
even after expiry. This marker also covers a crash immediately before the
request leaves the process. It never resets the deadline. Receipts created by an
older release retain their original deadline when resumed.

Formal catalog tasks use a five-hour answering window; see
answering budgets and runtime upgrade.

On timeout, cancellation of that Evaluation proceeds independently of event
reads. There are at most three cancellation attempts per invocation, using the
same idempotency key, with attempt/error counts saved in the receipt. Accepted
cancellation records `TIMEOUT`. Failed cancellation leaves `CANCEL_PENDING`,
allows subsequent tasks to proceed and is retried when this receipt is resumed.
Cancellation request acceptance is not proof that cleanup has finished.

Rerunning with the same receipt resumes in-flight tasks and skips terminal
PASS/FAIL/TIMEOUT entries. Changed bindings are rejected. To intentionally
replay again, choose both a new receipt and a new `--attempt` label. Retain the
real receipt and review every selected task verdict. Do not treat mocked test receipts
as live acceptance.

## V3 runtime notes

### Capacity and preflight

The scheduler counts **STARTING + RUNNING + FINALIZING Run rows**, plus
**PENDING + APPLYING + RUNNING + BLOCKED `driver_attempt_bindings` rows**,
excluding bindings with a valid historical-GC completion. These are independent
row counts, not distinct Run identities. QUEUED Runs are reported but do not
occupy a slot. `available_slots = max(configured_cap - occupied_slots, 0)`.

`GET /api/v1/infrastructure/scheduler-capacity` is unauthenticated and read-only;
operator credentials are still needed for Kubernetes port-forward access. Its
JSON response is an unwrapped object. The deployment preflight runs immediately
after the Kubernetes-mode rollout, before smoke submission, and again in
`verify`. It first checks `/api/v1/overview` response headers for
`x-management-driver-mode: kubernetes` and
`x-management-accepting-evaluations: true`, then prints:

```text
capacity: cap=<n> occupied=<n> available=<n> queued=<n> active=<n> cleanups pending/applying/running=<n> blocked=<n> released=<n>
```

`active` sums the three active Run states; `pending/applying/running` sums
progressing cleanups. `blocked` comes from `counts.cleanup_counts.BLOCKED`,
not the blocker list's length. `released` is `released_historical_cleanups`.
The list contains all four cleanup states, with BLOCKED first, capped at 100;
retained-capacity failure diagnostics show at most ten entries, tolerate null error
messages, and report `blockers_truncated` when the server truncated its list.

If no slot is available but BLOCKED reservations are below the cap, active
STARTING/RUNNING/FINALIZING Runs or PENDING/APPLYING/RUNNING cleanups can still
free capacity. The preflight prints a warning and continues:

```text
capacity temporarily exhausted: N active Runs, M progressing cleanups, B retained; the smoke will queue
```

This also applies to mixed temporary and retained occupancy below the retained
cap threshold. The smoke keeps its own timeout while waiting. The preflight
is an observation, not a slot reservation; other submissions can consume the
reported headroom. See [capacity configuration](configuration.md).

`seed-tasks --check` prints these diagnostics informationally, including when
the gate is closed. Capacity or header failures never replace seed comparison
results or suppress a missing/different-version failure. For a capacity 404/503
it prints `capacity endpoint unavailable (pre-V3 runtime or non-kubernetes mode)`
and continues. Truncated HTTP responses, connection failures, timeouts and
invalid JSON are also informational. It remains GET-only and does not write `.state`.

### Preview's admission command channel

The simulator bootstrap opens only a fresh SQLite singleton **FENCED/0 →
OPEN/1**, then deployment restarts Management in Kubernetes mode. It does not
reopen a later deliberate FENCE. Kubernetes mode GETs the command map and
JSON-patches `/data/receipt.json` in the receipt map; Management creates neither
map and does not GET the receipt. The receipt map and data key must pre-exist.

Only the Kubernetes-mode SSA manifest includes the operator ServiceAccount
(`automountServiceAccountToken: false`), its Role/RoleBinding, and Management's
namespace-local inventory Role/RoleBinding. Operator permissions are command
GET/PATCH and receipt GET. Inventory grants LIST on ConfigMaps, Jobs, Roles and
RoleBindings in `frontier-system`, as required by SNAPSHOT observation.
The simulator bootstrap prints `admission control maps: skipped for simulator
bootstrap` and does not call the map-creation helper. The deployment entrypoint
always follows that bootstrap with the Kubernetes phase; it has no separate
simulator-only deploy target. Rendering a simulator manifest also creates no maps.

After applying that manifest and before the rollout restart, deployment GETs
each map and creates it only if absent, using `kubectl create -f -` with a
temporary JSON file supplied on stdin. An AlreadyExists race is harmless;
other errors fail. Existing operator commands and Management receipts are
left untouched. Neither map participates in SSA. Never client-side apply the
command map: the added last-applied annotation fails Rust validation.

The command data is exactly this single line, **with no trailing newline**:

```json
{"attempt_ids":[],"command_nonce":"bootstrap-fenced","expected_epoch":0,"issued_at_unix_millis":1,"operation":"FENCE","schema_version":"frontier.management-production-command/v3"}
```

The command map has only `command.json` and exactly four labels:
`app.kubernetes.io/name=frontier-management`,
`app.kubernetes.io/component=production-admission-command`,
`app.kubernetes.io/part-of=frontiercs2`, and
`frontier.dev/owner=production-operator`. It has no annotations, owner references,
finalizers or binaryData and is not immutable. Management binds its UID on the
first accepted observation; resourceVersion can change. Deleting/recreating
the maps is not a recovery procedure for already bound identities.

The receipt starts with the pinned `PENDING/controller_not_observed` literal.
`verify` reads it as the operator and prints operation, status, error code and
revision. On a simulator-opened OPEN/1 site, bootstrap produces
`FENCE/ERROR/fence_epoch_or_state_conflict`: this is the expected steady state,
leaves admission unchanged, and does not itself cause a reconciliation error
log. Identical commands replay silently after receipt publication. On FENCED/0,
bootstrap instead produces APPLIED. Receipt status alone does not reject
verification; the live overview headers determine whether admission is open.
A missing map prints `admission receipt: absent`; kubectl failures or malformed
receipt JSON print one `admission receipt: unreadable (<reason>)` diagnostic
without the error payload. Header/capacity preflight still runs afterwards.

**Preview has no ValidatingAdmissionPolicies for these objects.** The platform's
writer-identity, deletion and shape enforcement is therefore absent. Rust label
and payload checks do not authenticate the writer. This channel does not give
Preview the complete production admission security boundary.

Preview remains `local-k8s`. Production IAM worker inputs,
`MANAGEMENT_LITELLM_SERVICE_CIDR`, serving-ready and key-worker objects are not
required here. `MANAGEMENT_DORMANT_ATTEMPT_REQUESTS_JSON` and
`MANAGEMENT_HISTORICAL_GC_REQUESTS_JSON` are optional and default to `[]`.

### Smoke operation identity and replay detection

Smoke TaskVersion/CandidateVersion documents remain deterministic. Mutable
operation identity lives in `.state/<site>/smoke-operation.json`, mode 0600:
`schema_version: frontier.deployment-smoke-operation/v1`, `nonce`,
`idempotency_key`, `started_at` (UTC), `candidate_version_id`, `task_version_id`,
`runtime: {management_digest, driver_digest}`, and `retry`. After POST, it also
records `evaluation_id` and `evaluation_created_at` before polling.

The key is `deployment-smoke-<site>-<cand8>-<task8>-<nonce>`, with the existing
bounded `-retry-N` suffix for immediately FAILED/CANCELLED Runs. The Evaluation
body includes `labels.smoke_nonce`; no version document changes. Each key is
saved before POST, so a lost POST response can be retried with the same body.
A nonterminal Evaluation resumes with the original start time, nonce and
retry when the target Management/Driver digests and version IDs still match.
A terminal Evaluation, changed target, or HTTP 404 for an archived/deleted
Evaluation rotates to a new nonce/key. Other lookup errors fail and preserve
the existing operation record.

After POST, `created_at` must be at least `started_at - 60 seconds`; an older
Evaluation fails as an idempotent replay and asks the operator to rotate the
smoke operation. This checks the persisted operation start, not the current
process start. No intermediate state must be observed: fast completion is
valid, while successful acceptance still requires the existing terminal
Evaluation/Run and numeric-score checks. The successful `.state/<site>/smoke.json`
receipt includes the nonce, Evaluation ID/creation time, and runtime digests.
These are target runtime bindings, not an attestation of a particular serving
Pod. Development tests use temporary directories; they do not change live state.

## 3. Access and inspect

```bash
KUBECONFIG=.state/preview/kubeconfig \
  kubectl -n frontier-system port-forward service/frontier-management 8080:8080
```

Useful read-only checks:

```bash
KUBECONFIG=.state/preview/kubeconfig kubectl -n frontier-system get deploy,pod,pvc,job
KUBECONFIG=.state/preview/kubeconfig kubectl get namespace -l frontier.dev/owner=management
```

The final ConfigMap must show `MANAGEMENT_DRIVER_MODE=kubernetes` and
`MANAGEMENT_DRIVER_BACKENDS=local-k8s`; the simulator value is only a transient
bootstrap manifest under `.generated/`.
`MANAGEMENT_ARTIFACT_PRESIGN_ENABLED=true` uses the dedicated Management IRSA;
it does not grant S3 or KMS authority to Agent or Verifier Pods.

To administer LiteLLM:

```bash
KUBECONFIG=.state/preview/kubeconfig \
  kubectl -n frontier-litellm port-forward service/frontier-litellm 4000:4000
```

Open `http://127.0.0.1:4000/ui/`. Load the local login values into your shell
without printing them:

```bash
set -a
. .state/preview/litellm-admin.env
set +a
```

In the UI, add a provider credential under **Models → LLM Credentials**, then
add the model using that existing credential. Provider credentials are
operator-owned external authorization: never place them in this repository,
`site.json`, Terraform variables, or deployment logs. PostgreSQL persistence
protects the record across Pod restarts, but it is not a backup.

Management reserves 4 GiB of memory and 34 GiB of ephemeral storage, with the
same limits and a 34 GiB disk-backed `/tmp` volume. Fresh installations and
`upgrade-runtime` apply this allocation; upgrades reject missing, memory-backed
or mismatched `/tmp` volumes before mutation. These allocations support the paired
runtime's large-result reader: projection streams the immutable source archive
and bounds retained result spools to 32 GiB; seek-based reads and projections
share a lifetime lease. Before rollout, check actual system-node allocatable
memory/storage, other Pod requests and filesystem usage. Free space alone is
not scheduling capacity, and these settings do not expand any node group.
Deploy this allocation with the compatible large-result runtime and its forward
database migration; changing a rendered manifest does not prove live readiness.

Fresh gateway installations use LiteLLM's native global retry policy for upstream authentication,
rate-limit/cooldown, internal-server and timeout errors: up to 16 retries with a
five-second retry delay. This also covers model aliases registered later in the
database. Other error classes retain the zero-retry default, and no fallback
model is selected. Client authentication is checked before routing; rejected
client credentials are not retried. These retries do not restart the Agent or
extend its run deadline. A persistently unavailable provider can still fail a Run.
Keep all failed and successful callback records when verifying this policy;
successful recovery alone does not prove complete usage and audit capture.
`upgrade-runtime` preserves the existing gateway and does not enable this policy
on an existing site. Before separately enabling it, verify its deployed Driver
accepts multiple immutable retry callbacks for one logical model call and sums
their reported usage. A legacy single-key collector that rejects repeated call
identities is incompatible; upgrade the collector before enabling gateway
retries. Testing a multi-key collector does not establish compatibility for a
Run whose frozen dispatch uses the single-key format.

The UI/gateway can be healthy while model-enabled Frontier Runs remain
unavailable. The automated smoke is intentionally scripted; this `local-k8s`
profile does not issue LiteLLM virtual keys to Drivers.

## 5. Backup and deletion

SQLite online backup/restore and PostgreSQL backup/restore are not automated
here. Before upgrades or deletion, fence the sole SQLite writer and use an
application-supported online snapshot; never copy active WAL/SHM files. Take
and test a consistent PostgreSQL backup before changing LiteLLM/database
versions or deleting the site.

There is no destroy command. EKS deletion protection plus Terraform
`prevent_destroy` protects state, KMS, artifact S3, and EFS. Deletion requires
a reviewed change that proves backup integrity, zero active Run namespaces,
exact retained-data disposition, and explicit site/account identity.

## Local gateway terminal artifact retention

Executable Preview renders a gateway artifact source at
`/var/lib/frontier-local/artifacts` and a separate retained root at
`/var/lib/frontier/gateway-terminal-retained`. The retained root lives on the
existing Management data PVC backed by EBS, outside the Driver/Agent EFS mounts.
An init container using the same pinned Management image creates this directory
as UID65532/mode0700. It reuses an existing private directory and rejects symlinks,
wrong ownership or shared permissions without modifying them. Simulator bootstrap
does not enable this receiver configuration.

This configuration does not by itself upgrade the native runtime, consume a terminal handoff,
repair model-key cleanup or release scheduler capacity. Runtime publication still
requires a clean official source revision, immutable image pins and the documented
idle/backup/seed-check operator window. Before a schema-changing runtime upgrade,
retain a supported consistent database backup separately from the deployment-state
backup. Verify the actual EBS/EFS mount identities and private directory owner/mode
after rollout; do not place retained snapshots inside the shared object-store mount.

The supported upgrade-runtime path also installs the paired roots and private
retention init container on an existing site. It checks the existing distinct
data/object mounts and UID/GID65532, rejects conflicting roots or an unexpected
same-name init container, preserves other init containers, and updates the init
image with Management's immutable pin. ResourceVersion-guarded patches and full
ConfigMap/Deployment readback remain required. Fresh rendering and upgrades use
the same init recipe; upgrades do not repeat bootstrap, reseed or reset cleanup.

## Run governance runtime prerequisites

The governance support introduced in FrontierCS2 `8264993e08dc40f6deb1c3ac8abe149fbe6bdd21`
needs more than an image change. Fresh rendering and `upgrade-runtime` grant
trusted Management `get` on `pods/log` and `list` on Events through the separate
`frontier-management-diagnostics` ClusterRole and binding. The existing Management
role already grants Jobs LIST. The upgrade dry-runs the additive RBAC and applies
it after a fresh idle check, preserving other site RBAC. Fresh renders use the
upstream 15000 ms controller poll; upgrades preserve the live site's configured
poll interval and other configuration.

Before activating migration27, obtain a consistent SQLite snapshot using a fixed
read transaction/SQLite backup API, verify integrity and foreign keys, and retain
the old images, configuration and accepted-result baseline. The `.state` backup
alone is insufficient. Migration27's reservation protocol is unsupported by old
binaries: rollback requires the pre-migration database plus the matching old
application/configuration and reconciliation of external changes since the snapshot.
An image-only rollback is unsupported. After rollout verify diagnostic reads,
retained writes and expiry deletion, Jobs TTL behavior, cleanup/slot state, unchanged
historical results and the existing model smoke. Local manifest tests do not prove
these deployed checks or mark a release accepted.

The original-ten 1024-session release and its platform prerequisite are tracked
in the release record. The example site
pins the matching 32 GiB Driver and Management release. Existing private site
configs must update both image pins and the application commit together before
seeding these task revisions; copying task definitions alone is insufficient.

### Kubernetes API destination discovery and rotation

Preview resolves Kubernetes API egress from the cluster's `default/kubernetes`
Service, Endpoints, and EndpointSlices. All must agree on ready IPv4 TCP 443
backends in two stable reads. The resulting allowlist contains only the Service
IP and exact backend `/32` destinations. DNS on the deployment machine is not
an authority for in-cluster Service DNAT targets: an external machine can resolve
public EKS addresses while the cluster uses private control-plane addresses.
Missing, mismatched or transitioning discovery fails closed; no VPC-wide CIDR is
added as a fallback.

Fresh deployment saves `api-routes.json` beside its rendered runtime manifest.
An idle `upgrade-runtime` refreshes `MANAGEMENT_KUBERNETES_API_CIDRS` and only the
API rule in the existing Management and storage-controller NetworkPolicies,
using resource-version tests. It preserves all other network rules and task,
Candidate, quota and scientific settings. The upgrade directory retains the
observed object identities/resource versions, collector hash, exact routes,
policy before/patch files and an accepted receipt bound to the route receipt.
Discovery is repeated before applying changes and policies are read back after
upgrade. A failure after a partial apply is not acceptance; inspect the retained
receipts, keep the site idle, and retry the guarded upgrade.

Run `verify` before a new evaluation campaign. The CLI `verify`, `smoke`, and
`gpu-smoke` commands reject a route mismatch and direct the operator to the
existing idle/backup/seed-check `upgrade-runtime` window. EKS can rotate its
private control-plane endpoints. This installer refreshes routes during deploy
and upgrade; it is not a running endpoint watcher and does not rewrite immutable
active Run dispatches or namespace policies. Direct UI/API evaluation creation
bypasses these CLI checks. A native Management pre-dispatch route check is needed
to enforce this boundary for every caller; the installer does not claim to
provide that platform guarantee. Do not extend readiness deadlines or allow
`10.0.0.0/8` to compensate for an incomplete API destination allowlist.

### Native API route-reader grant

The native Management pre-dispatch route gate needs a dedicated Role and
RoleBinding named `frontier-management-api-routes` in `default`. Fresh production
rendering and the guarded idle runtime upgrade install exactly two permissions:
`get` on only the `kubernetes` Service and Endpoints, and `list` on EndpointSlices
in `default`. The sole binding subject is
`frontier-system/frontier-management`. There are no write, Secret, wildcard or
Agent grants. Kubernetes RBAC cannot label-filter LIST; the native reader sends
`kubernetes.io/service-name=kubernetes` and validates the returned identities.

Runtime upgrade server-dry-runs the grant, saves the manifest with its other
receipts, applies it before changing runtime configuration/images, and checks
its exact rules, role reference and subjects before proceeding. `verify` and
smoke preflights also check that readback. Simulator bootstrap does not need the
grant. The CPU/Modal production runtime golden manifests intentionally gain
these two RBAC resources; other rendered resources and task identities are
unchanged. The grant alone does not activate a native gate in older Management
images: release the paired, merged platform code and pinned images through the
normal source-bound upgrade process. This source change does not authorize an
active-run rollout or policy mutation.

### Storage reconciliation during namespace termination

The storage controller skips namespaces with a deletion timestamp and isolates
reconciliation errors per namespace. A terminating or rejected namespace cannot
prevent later active Runs from receiving their normal network policy and object
store PVC. Terminating namespaces remain in the observed namespace set for PV
retention; orphan PV cleanup still requires confirmed namespace absence and the
existing ownership and Retain-policy checks.

### Localize frozen native CPU images before registration

Native task documents can retain private registry references even after their
S3 datasets have been seeded locally. The local-k8s Driver composes CPU images
through an anonymous HTTP registry client; these images must be available at the
existing site registry. The `native-localize-images` command copies the **same
frozen leaf-manifest digests** and derives new immutable documents. It never
changes the global registry map, runtime, catalog, source/build facts, datasets,
quotas, network policy, Candidate execution, or GPU image references.

Run this before `native-publish prepare` using the original frozen document and,
when needed, the frozen Candidate document. Existing local Candidate references
and IDs are retained.

```sh
./python scripts/frontier_deploy.py native-localize-images --config config/site.json \
  --document frozen-task.json --candidate frozen-candidate.json \
  --plan --output image-plan.json
./python scripts/frontier_deploy.py native-localize-images --config config/site.json \
  --document frozen-task.json --candidate frozen-candidate.json \
  --check --output image-check.json
FRONTIER_REGISTRY_COPY_TOOL=crane ./python scripts/frontier_deploy.py native-localize-images \
  --config config/site.json --document frozen-task.json --candidate frozen-candidate.json \
  --apply --output image-localization.json
```

`--plan` is offline. `--check` only reads registry manifests and blob metadata;
it does not copy images. `--apply` uses the existing registry port-forward and
approved copy tool/credential chain, then verifies manifests and blob metadata.
An index-to-leaf digest substitution is rejected: obtain an author-reviewed
leaf-pinned release instead. Missing or changed manifests fail closed. No image
is rebuilt, no version is registered, and no definition is uploaded by this step.

The image-localization output alone is not a published TaskVersion or CandidateVersion.
Use the following stages after reviewing the plan and copying the exact CPU images.
Every stage recomputes its localization from the original frozen input documents;
a manually edited envelope cannot redirect uploads or registration.

1. Rebind the **verified original dataset transport bundle** to the localized
   task document. This operation writes a new bundle directory and small custody
   JSON files. Verified regular content-addressed OCI blobs are hardlinked on the
   same filesystem, with no dataset payload copy, rebuild, or symlinks. Original
   files remain intact. The transport manifest digest does not change, so the
   existing registry transport and seeded PVC can be reused. A cross-filesystem
   output is rejected; choose a new directory beside the original bundle.

```sh
./python scripts/frontier_deploy.py native-local-artifacts --config config/site.json rebind \
  --document frozen-task.json --candidate frozen-candidate.json \
  --bundle original-bundle --rebound-bundle localized-bundle --output localization.json
```

2. Check and publish only the newly derived immutable definitions. The official
   command preserves the original S3 bucket and key parent, requires bucket
   versioning, verifies every existing version, rejects deletion markers and
   truncated histories, conditionally creates missing objects, and reads the
   precise VersionId back. It uses the existing AWS CLI credential chain and the
   configured account/region; it does not create buckets or change permissions.
   Each successful artifact receipt is persisted before processing the next.
   A failed conditional creation is never retried as an overwrite. Retry only
   after inspecting the recorded state with `--check`.

```sh
./python scripts/frontier_deploy.py native-local-artifacts --config config/site.json definitions \
  --document frozen-task.json --candidate frozen-candidate.json --check --output definitions-check.json
./python scripts/frontier_deploy.py native-local-artifacts --config config/site.json definitions \
  --document frozen-task.json --candidate frozen-candidate.json --apply --output definitions-published.json
```

3. Extract `document` from `localization.json` without editing it, and run the
   ordinary native registration check/apply with **localized-bundle**, using the
   actual seeded PVC inspector or read-only mount. This independently verifies
   the same dataset bytes and normalizes only their receipt versions to local
   content versions. No dataset seeding is repeated when the bytes are already
   present. The published TaskConfig remains identical before/after normalization.

```sh
./python -c 'import json; d=json.load(open("localization.json")); print(json.dumps(d["document"]))' > localized-task.json
./python scripts/frontier_deploy.py native-publish --config config/site.json register \
  --document localized-task.json --bundle localized-bundle --pvc-pod EXISTING_TRUSTED_INSPECTOR \
  --check --output task-check.json
./python scripts/frontier_deploy.py native-publish --config config/site.json register \
  --document localized-task.json --bundle localized-bundle --pvc-pod EXISTING_TRUSTED_INSPECTOR \
  --apply --output task-registration.json
```

4. Check/register the Candidate using the same original frozen documents.
   Already-local Candidates are checked and retained; they are never re-created.
   Changed Candidates require the exact derived S3 artifact to exist, preserve
   all execution/model API settings, and receive a new immutable identity.

```sh
./python scripts/frontier_deploy.py native-local-artifacts --config config/site.json candidate \
  --document frozen-task.json --candidate frozen-candidate.json --check --output candidate-check.json
./python scripts/frontier_deploy.py native-local-artifacts --config config/site.json candidate \
  --document frozen-task.json --candidate frozen-candidate.json --apply --output candidate-registration.json
```

Preserve the original frozen documents, original bundle, localized envelope,
artifact version receipts, and registration readbacks together. These steps do
not submit evaluations or change existing versions, runtime, configuration,
network rules, IAM, Secrets, or catalog membership. GPU images remain unchanged.

### Trusted verifier workspace

`execution.task_verifier_workspace_bytes` is an optional map from selected problem
names to trusted verifier workspace limits in bytes (1 through 1 TiB). For example,
`{"residual_state_advancement": 17179869184}` allocates 16 GiB to both Train
and final verification. Agent workspace limits, task source, datasets and image
pins remain unchanged. An omitted or empty map preserves existing identities.
A nonempty map changes selection and TaskVersion configuration identities; use
the official seed/check flow to register the resulting immutable version. Existing
Runs retain their frozen configuration. Authored `verifier.workspace_bytes` is
also preserved by the importer; site settings take precedence.
