# Site configuration

`config/site.json` is the only operator-authored input and must contain no
secret. Copy `config/site.example.json`; the real file is ignored by Git.

## Identity and state

- `site.expected_account_id` must equal `aws sts get-caller-identity` before
  any plan/apply.
- `site.aws_region` is explicit; no old profile default is reused.
- `terraform.state_bucket` must be globally unique.
- `terraform.operator_principals` is optional and defaults to `[]`. Each unique
  IAM user or role ARN gets `AmazonEKSClusterAdminPolicy` at cluster scope through
  the EKS module's `access_entries`. Omitting or leaving the list empty retains
  existing access only for sites that have never managed optional operators
  through this variable. Removing an ARN from a list Terraform already manages,
  including by omitting or emptying the list, revokes that principal's managed
  access at the next apply. The cluster-creator entry is managed separately.
  Use IAM ARNs, including any user/role path; STS session ARNs, root ARNs,
  wildcard ARNs and duplicates are
  rejected. Supported partitions are `aws`, `aws-cn` and `aws-us-gov`.
- `application.repository` is the fixed FrontierCS2 source repository and
  `application.commit` is the exact clean main commit for the release and task
  authoring handoff. `application.require_source` is required explicitly: when
  `true`, the installer clones the source under
  `.state/<site>/application-source`, or uses `FRONTIERCS2_SOURCE_DIR` after
  verifying exact HEAD and a clean worktree; when `false`, it skips the source
  checkout and only the task-authoring handoff is unavailable.
- `images.management` and `images.driver` contain public Docker Hub
  `repository`, `tag`, and lowercase `sha256` `digest` records. Each tag must equal
  `sha-<application.commit>`.
- Registry, LiteLLM, and PostgreSQL source images use the same three fields.
  Every built or mirrored image is pushed into the site registry, read back,
  and Kubernetes receives `repository@digest`.

For example, extend the existing `terraform` block with:

```json
"operator_principals": [
  "arn:aws:iam::123456789012:user/operator",
  "arn:aws:iam::123456789012:role/team/preview-admin"
]
```

Use the site's partition and the actual principal account ID. Entries are keyed
by full ARN, so reordering the list does not recreate access. With
`enable_cluster_creator_admin_permissions=true`, EKS module v21 derives its
`cluster_creator` entry from the current Terraform caller. Changing the caller
destroys and recreates that entry. Keep the Terraform caller stable and never
list its cluster-creator ARN again in `operator_principals`. Import manually
created entries before adding them to the managed list.
Network access still requires an allowed `network.operator_cidrs` address.

## Networks

- `vpc_cidr` is a private `10.0.0.0/8` `/16`.
- `service_ipv4_cidr` is a disjoint `/16` fixed at cluster creation.
- Management, registry, and LiteLLM ClusterIPs are distinct stable addresses
  inside that Service CIDR and cannot use the Kubernetes API `.1` or CoreDNS
  `.10` addresses.
- `operator_cidrs` accepts only exact IPv4 `/32` routes.

Changing a Service CIDR or fixed ClusterIP after cluster creation is a site
replacement, not an in-place config edit. Node bootstrap freezes the registry
IP into `containerd` configuration.

## Compute and storage

Both `system` and `cpu` node groups must start with at least one node so the
automatic smoke Run can execute. `single_nat_gateway=true` is the Preview cost
default and is not multi-AZ egress HA.

The `system` pool is pinned to the first private subnet because Management
and the LiteLLM Postgres volume are zonal EBS disks; a multi-zone pool would
strand them on the next node rollout.

Size the `cpu` pool for the largest task's Agent and Verifier running at the
same time: train verifications run while the Agent is still alive, so the
pool must hold `agent.cpu_millis + verifier.cpu_millis` (16 vCPU for
`bounded_evm_path_prediction` and `graph_mining_method_selection`, whose
authored budgets are used verbatim) plus the Driver Pod. The default is two
`c6i.4xlarge` nodes; one node leaves the Verifier unschedulable for those
tasks. The instance family matters as much as the count: several evaluators
are single-threaded and carry authored engine timeouts calibrated on modern
x86 cores (`join_tree_rewriting` allows 120 s per Stats-CEB engine pass), and
an older, slower core family such as `m5a` times out where `c6i` finishes.

`execution.max_active_drivers` is required, with a **1..1000** bound matching
Management's `MANAGEMENT_MAX_ACTIVE_DRIVERS` validation; the example
supplies `1`. Omitting it fails validation.
On V3 it caps active Run rows **plus cleanup reservations**, including BLOCKED
cleanup bindings without a valid historical-GC completion. Retained reservations
need headroom: set the cap above their count for a new Run to start. Review the
live scheduler-capacity response before changing it; active Runs or progressing
cleanups may only need time to finish. The capacity preflight fails only when
`available_slots < 1` and `counts.cleanup_counts.BLOCKED >= configured_cap`.
When slots are temporarily exhausted and BLOCKED reservations remain below the
cap, it warns and lets the smoke queue with its own timeout. This includes mixed
active Runs, progressing cleanups and retained reservations. The equation and
diagnostics are in
[V3 runtime notes](operations.md#v3-runtime-notes).

Size the CPU pool for the actual concurrent Runs permitted by the remaining
slots. Each heavy Agent and Verifier requests 8 vCPU; two such Pods cannot share
a 16-vCPU node after system reservations. Two concurrent heavy Runs therefore
need four such nodes, with Driver CPU and memory headroom checked as well.
On the default two nodes, one Agent on each node leaves both train Verifiers
waiting for CPU. A cap increase compensating for retained reservations
does not immediately grant that many concurrent Runs, but releasing those
reservations later increases available concurrency; review the cap again then.

Check your account's regional On-Demand vCPU quota before installation or
upgrades. The default pools (two `c6i.4xlarge` plus one `m6i.xlarge`) use 36 vCPUs.
The single system worker has 16 GiB of memory. Management reserves 1 GiB and
has a 4 GiB limit, alongside LiteLLM, Postgres and Kubernetes services. These
infrastructure budgets do not alter task Agent, Driver or verifier budgets.

Existing installations must review the Terraform plan before changing the
system instance type. Management's SQLite database and LiteLLM Postgres use
zonal, retained EBS volumes; the system node group remains pinned to the same
first private subnet. The replacement also selects the module's current default
AMI release; review that version change separately from the memory increase.
During an idle maintenance window, verify an online
database backup, stop database writers and owned audit readers, and migrate
the system node in that volume's availability zone. Wait for EBS detach and
reattach and verify database integrity, API readiness and the original image
digests before resuming evaluations. Do not delete PVCs or apply a plan that
changes scientific worker pools or unrelated infrastructure.
Replacing the CPU pool briefly needs both old and new groups; for example,
a 64-vCPU quota cannot hold both two-node pools and the system pool. Either
request a quota increase first or replace in
two applies: set `cpu.min_size` and `desired_size` to 1, apply, then restore
2 and apply again. A node group that failed to create keeps any instance it
managed to launch and still counts against the quota; delete it with
`aws eks delete-nodegroup` and wait for it to disappear before retrying.

In-place node group upgrades hit the same limit: when the EKS release version
drifts (a newer AMI is published), `apply` updates both node groups in place and
EKS launches the replacement nodes for a group before draining the old ones, so
the CPU pool needs headroom for two extra `c6i.4xlarge`. With that 64-vCPU
quota the update fails with `NodeCreationFailure` (`VcpuLimitExceeded` in the
Auto Scaling activity log) and rolls back. Upgrade in two steps without editing
the site configuration: `aws eks update-nodegroup-config` with
`minSize=1,desiredSize=1`, `aws eks update-nodegroup-version --release-version
<version>` and wait for the update to report `Successful`, restore
`minSize=2,desiredSize=2`, then run `make apply` again so Terraform reconciles
an already-upgraded group.

`management.storage_gib` sizes the encrypted EBS claim.
`litellm.postgres_storage_gib` sizes a separate encrypted EBS claim whose
StorageClass and PVC both retain data. `litellm.exposure=port-forward`,
`ui_access=admin-only`, and `store_model_in_db=true` are fixed Preview safety
properties, not optional conveniences.
`execution.object_store_gib` and `registry_storage_gib` are Kubernetes capacity
claims over protected EFS access points; they do not impose an EFS billing
quota. Raising `max_active_drivers` does not provision workers or increase AWS
or Modal quotas. For example, 30 active Runs plus 5 retained cleanup reservations
need a cap of at least 35, as well as sufficient independently verified worker
capacity. The example remains at 1.

`execution.agent_external_egress` is a required boolean. When `true`, the
storage controller adds `frontier-agent-external-egress` to every verified Run
namespace. The additive policy selects only Agent Pods and permits CoreDNS on
UDP/TCP 53 plus public TCP 443, excluding private, loopback, carrier-grade NAT,
and link-local IPv4 ranges. Labs using this internal-beta posture configure
the model endpoint and key
through private Candidate execution settings; credentials never belong in
images, Git or `site.json`. When `false`, the controller does not
create or update this policy. It does not delete a previously created policy;
Run namespace deletion removes it.

## Feature contract

Execution and LiteLLM are required. The example selects CPU tasks by default:

```json
{
  "run_execution": true,
  "litellm": true,
  "modal_gpu": false,
  "task_model_aliases": ["progent-agent-gpt-4o-mini"]
}
```

`features.task_model_aliases` (optional) lists the LiteLLM model aliases this
site serves for tasks that declare `additional_keys`. A task whose aliases are
not all listed is excluded from selection. Each run then receives LiteLLM
virtual keys limited to the task's `budget_usd`, so the alias needs a nonzero
price in the gateway. In this release only `progent_privilege_control` declares
keys (alias `progent-agent-gpt-4o-mini`, for example routed to
`openrouter/openai/gpt-4o-mini`).

`features.modal_gpu` selects tasks from the full tracked catalog: `false`
excludes tasks whose agent or verifier needs Modal GPUs; `true` selects all
compatible tasks. `validate` lists selected/excluded tasks and the effective
catalog hash. The tracked catalog and definitions remain complete. An empty
selection or an unsupported declared GPU helper interface is an error.
`render` saves `.generated/<site>/task-selection.json` outside `bundle.sha256`.
See [task selection](task-selection.md) for hash and compatibility rules.

To enable GPU tasks, set `features.modal_gpu=true` and export `MODAL_TOKEN_ID`
and `MODAL_TOKEN_SECRET`. The example retains the non-secret `modal` block;
it is allowed while the feature is disabled, but its presence does not enable
GPU tasks or cause Modal credential reads. The current Sparse task requires
the separately registered Kimi helper V3 overlay (or another reviewed V3
Candidate); the default seeded Codex overlay remains V2.

The installer creates the LiteLLM service, admin UI, database, master key, and
salt, but deliberately creates no model-provider credential. The operator adds
provider credentials and model definitions through the admin UI; those records
are held by LiteLLM/PostgreSQL and must not be copied into `site.json`.

For private ECR GPU images, optional `modal.ecr_pull_secret_ref` names an existing
Modal Secret in `modal.environment`. Its `AWS_ROLE_ARN` and `AWS_REGION` fields
select the existing AWS OIDC pull role and ECR region; keep these fields and all
credentials out of `site.json`. Confirm that the role trusts the exact Modal
workspace, environment and app, with read access to the required repositories.
This option only renders `MANAGEMENT_MODAL_ECR_PULL_SECRET_REF`; it does not
create or rotate the Secret or grant IAM permissions. Omit it or use an empty
string to preserve the existing public-image behavior. A non-empty value uses
the existing Management configuration hash and triggers a Management rollout;
apply it in an operator window after active Runs finish, with a current site
backup. Restore the previous reference and apply in another such window to
roll back. Management and Driver image pins are unaffected by this option.

## Runtime images

The installer pulls Management and Driver from public Docker Hub by the
configured digests, builds four deployment-repository contexts, and mirrors the exact
Management/Driver/LiteLLM/PostgreSQL images after the registry is ready. Supplied
FrontierCS2 public release images use `docker.io/joyemang33/fcs2-*`; imported task
images may use other public Docker Hub namespaces. The four installer-built
images are published only to the site registry. Installation requires no
`docker login` or registry credential. All eight
publication tags, source identities, digests, and digest-pinned references are
stored with mode `0600` in
`.state/<site>/runtime-images.json`. `render` writes configuration and Terraform
documents without consuming this record. `apply` and `deploy` rebuild runtime
images and replace it; narrower operations such as `smoke` load its saved exact
image references without substituting mutable tags. GPU provisioning also
resolves the frozen public tag over HTTPS and checks it still equals its pinned
digest before importing `tag@digest` into Modal; a moved tag fails closed.

The smoke CandidateVersion and TaskVersion ULIDs are derived from the frozen
runtime/dataset identities. A changed build creates new immutable catalog
versions and a new idempotency key instead of attempting to overwrite an old
version.

## Task publication catalog v2

The catalog task trees under `tasks/frontier-cs-2.0-demo/problems/` are the
released task directories. `scripts/render_task_definition.py --all` is the
only producer of `config/task-definitions.json`: policy fields and authored
budgets are copied, JSON selector bodies are hashed, and source
identities follow `preview-source-identity/v1`. Sparse uses that same native
identity algorithm. Add or replace tasks through `scripts/import_task.py`.

The optional `scripts/build_task_images.py` builder produces `linux/amd64`
Agent, Verifier and dataset images plus enabled GPU roles. `--problem cpu`
selects tasks with no Agent or Verifier GPU requirements; `--manifest <path>`
records frozen base images without editing exported
sources. Dataset images contain both content-addressed JSON selectors in one
layer. Builds do not update publication pins; pushes require `--push`.
The reference overlays are recorded in
`config/task-catalog.json` task `reference` sections; only references for selected tasks are mirrored
alongside the Codex overlay.
They are not additional catalog candidates.

`execution.model_domains` is optional and defaults to `["openrouter.ai"]`.
Supply a nonempty array of lowercase DNS names, for example
`"model_domains": ["api.example.com", "openrouter.ai"]`, alongside the existing
execution fields. URLs, ports, wildcard names, trailing dots and uppercase
names are rejected. Selection sorts and deduplicates the list; publication
unions it with each task's authored network allowlist. Changing this list can
create new immutable TaskVersions and changes the selection digest. It does
not configure a model endpoint, credential or LiteLLM route. The seed receipt
records the normalized list. Authored audit values are preserved. The five
database tasks reserve an 8 GiB bundle for long model runs; see
model audit capacity for reservations and limits.
See the generated definitions for exact budgets and
[operations](operations.md#reference-parity-acceptance-gate) for acceptance.

`config/task-catalog.json` is a tracked, non-secret immutable-input catalog for
one generic Codex Candidate overlay and the exact names declared in the tracked
`config/task-definitions.json`. There is no fixed numeric limit. The full catalog
membership is:
`batch_local_counting`,
`bounded_evm_path_prediction`, `evm_state_checkpoints`,
`graph_mining_method_selection`, `join_tree_rewriting`, and
`sparse_video_attention_policy`. The example's initial selection contains the
five CPU tasks and excludes Sparse because `features.modal_gpu=false`.
Unknown, duplicate, and missing tasks are
rejected. Each record contains
Agent, Verifier, and public dataset image `{repository,tag,digest}` fields plus
the exact train/eval body SHA-256 values. The definitions file freezes resources,
quotas, submission/verifier protocol, provenance, dataset extension and task
image source tag. The five CPU tasks use their exported native policy verbatim. Sparse video
uses native JSON datasets, one H100 Agent runner, a four-GPU Verifier quota and
a 9000-second eval timeout with separate lightweight control/GPU images. The
published Verifier requests all four H100 GPUs for each train/final evaluation;
the Agent build's single GPU is a separate, sequential stage.

The generic Codex overlay uses its independent `sha-<commit>` tag and
`/opt/frontier/candidate/launch` entrypoint, including the native `frontier-submit`
skill/tool. The gateway route selects an immutable Candidate `model_alias`;
explicit HTTPS direct routing remains available for existing Candidates. The launcher
delegates the hard deadline to Driver. `frontier-submit train` returns immediately
after acceptance; `frontier-submit status <verification_id>` performs one status
read. The client has no train polling budget or deadline. Each task freezes
its own `source-<64 hex source hash>` tag in the definitions;
all of its Agent/Verifier variants must match. Dataset tags are `v1` and bodies
are digest-addressed. Platform images still must match `application.commit`.
All formal tasks use 18000 answering seconds with independent final timing;
see answering budgets for the required runtime pins and
existing-site upgrade procedure.
`validate` reports unresolved digest counts; `seed-tasks`, `deploy`, and `apply`
refuse placeholders before cloud mutation. Only authoritative public registry
readback can replace release placeholders.

Task seeding pulls selected sources as `repository@digest`, mirrors them into
the in-cluster registry, and uses the registry readback digest for Kubernetes
and TaskConfig. Public GPU images are the exception: Modal imports them directly
from Docker Hub, without mirroring their approximately 80 GB of offline model
assets into EFS and without ECR credentials. Seeding also creates or reuses a
Codex CandidateVersion using its digest-pinned in-cluster overlay mirror. Dataset Jobs run sequentially on the system pool,
use the trusted storage-controller runtime to extract and verify both immutable
dataset bodies from the in-cluster registry, and
copy them through the `frontier-object-seed` PVC to these root-relative EFS
paths:

```text
datasets/frontier-cs-2.0-demo/<problem>/train/sha256/<sha256>.json
datasets/frontier-cs-2.0-demo/<problem>/eval/sha256/<sha256>.json
```

All six tasks use JSON selector bodies at these paths.

The corresponding frozen TaskConfig URIs use the
`s3://frontier-local-task-data/` prefix and the same root-relative paths.
TaskVersion IDs and definition digests include policy, mirrored image digests,
and dataset hashes, so changed immutable input produces a new version;
unchanged reruns reuse the existing version after digest comparison.
Each rerun replaces only the exact content-derived disposable seed Job before
applying it again, so a stale Completed condition cannot hide a deleted or
changed durable PVC object.

The application release pinned by this Preview derives the local dataset and
registry maps for each dispatch. Every demo Driver receives only its own two
dataset entries and its own Agent/overlay/Verifier references plus the composed
image repository; the static speed-add entries remain only as explicit smoke
overrides. The application unit suite covers those exact map shapes and rejects
unsafe paths, missing composed-image mappings, and cross-dispatch restamping.
This removes the earlier code-level blocker that sent the speed-add map to
every demo. It is not a fresh-site live acceptance claim: a real Preview
Evaluation for each task remains the deployment gate. The CPU scripted smoke
also does not prove sparse-video H100 execution or a model-enabled Run.

Terraform creates a separate Management artifact-reader role. Its web-identity
trust is fixed to `system:serviceaccount:frontier-system:frontier-management`;
its object authority is `s3:GetObject` on the site artifact bucket's `runs/*`
prefix plus `kms:Decrypt` through that bucket's regional S3 service. It is not
the Driver role and is never attached to Agent or Verifier Pods.

The catalog schema is `frontier.deployment-task-catalog/v2`. Each task retains
its image/dataset fields and requires `reference` (descriptor with repository,
tag, digest, launcher_sha256 and payload_sha256, or null) and `baseline`
(with provenance and boolean comparable). Missing historical baseline data is
`{"provenance": "unverified", "comparable": false}`. Reference tags match the
definition source tag; launcher, payload and baseline input hashes are checked
against the native release. Public task repositories may use any bounded
Docker Hub namespace while retaining the role-specific repository names.
The generic Codex candidate is unchanged. V1 catalogs are refused. The one-time
v1-to-v2 upgrade folded the separate reference and baseline authority
files into the catalog while preserving task order and old record contents.

Seed receipt hashes retain the v1 deployment projection: the v1 schema marker,
candidate, task images and datasets. Reference/baseline fields are validated
separately and do not change existing task or candidate documents or receipts.
