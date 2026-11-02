# Deployment architecture and ownership

## Deployment package

`scripts/frontier_deploy.py` is a compatibility shim. It explicitly re-exports
existing functions and constants and calls `preview.cli.main()`; the Makefile
and operator command lines keep the same entrypoint and argument order.

| Module | Responsibility |
| --- | --- |
| `preview.config` | Site validation, generated bundles, Terraform variables/backend documents; exposes generated/state directory helpers |
| `preview.infra` | Shared errors, frozen hash/ULID algorithms, process/filesystem helpers, image-field validation, Terraform, Kubernetes/port-forward, runtime/RBAC/admission/LiteLLM manifests |
| `preview.images` | Source checks, builds, registry readback, mirroring, shared linux/amd64 child selection; separate readback and cache trust checks |
| `preview.publication` | Definition/catalog/reference/baseline loading and validation, authoring baseline extraction, site selection, immutable version documents, author/seed receipts, seeding and GET-only checks |
| `preview.acceptance` | CPU/GPU smoke, reference readiness and resumable parity replay, persisted smoke operations, runtime/LiteLLM verification, admission receipt printing; exposes runtime/capacity preflight |
| `preview.cli` | Argument parser and every command dispatcher, ordered deployment/upgrade orchestration and Modal credential delivery |

Directory calculation and runtime/capacity preflight implementations are shared
primitives in `infra`, explicitly re-exported by `config` and `acceptance`.
Publication needs these checks and acceptance needs publication documents; placing
the shared checks in `infra` keeps the **entire import graph acyclic**, including
function-local imports, without adding a seventh implementation module.
Dependency order is `infra → images → publication → config → acceptance → cli`
(each module imports only earlier dependencies; unrelated layers can be skipped).

The standalone `reference_parity.py`, `reference_candidates.py` and
`task_baselines.py` scripts are explicit compatibility wrappers. Catalog readers,
reference/baseline validators and `baseline_from_result_json` live in
`publication`: catalog validation and the importer need them before acceptance
runs. `REFERENCE_EXECUTION`, its readiness guard, `candidate_document`,
`successful_train`, `verdict`, receipt/error/deadline helpers, `management_json`,
`cancel_expired`, `replay_task` and `reference_parity_main` live in `acceptance`.
Putting catalog validation there would make publication depend back on
acceptance. Internal callers and test patches use the owning package modules;
the scanner rejects patches on all four compatibility wrappers.

`build_task_images.py` is the local builder used by the importer. Its optional
source-layer compatibility path lives in `preview.images`, so the former
standalone implementation is retired without changing its flags or operations.
Make aliases and their one-release deprecation are documented in
[operations](operations.md#build-the-bundled-task-images-locally).

Importing the package or shim reads no repository data. Task definitions load
through `publication.load_task_definitions()` when requested; the historical
`frontier_deploy.TASK_DEFINITIONS` read access remains available as a lazy module
attribute. The importer passes its candidate definitions explicitly to document
and dataset-URI construction, so nested validation sees the same task set without
changing a module global.

Tests patch the namespace that actually looks up a dependency. A patch on a shim
re-export does not affect the implementation. The package regression scanner
resolves module identities, including nested attribute aliases
and importlib-loaded shim copies. Fresh-import probes run from an empty directory.

Every test module imports `tests/deployment_test_guard.py` before ordinary imports;
a scanner enforces this order. The prerequisites entrypoint installs it too, and
`tests/__init__.py` covers package-style unittest invocation. The guard installs
`sys.addaudithook` with these checks:

- Write-capable opens and filesystem mutations (rename/replace, remove/unlink,
  mkdir/makedirs, rmdir, chmod, utime, hard links, symlink destinations, truncate,
  and the underlying operations used by shutil/tempfile) cannot target the real
  checkout `.state` tree or `/root/FrontierCS-2.0-Preview/.state`. Paths follow
  symlinks; relative paths use the current directory or `/proc/self/fd/<dir_fd>`.
  Rename/replace, remove/unlink, symlink destinations, hard links and rmdir also
  check the entry itself (resolved parent directory plus the final component), so
  an outward-pointing symlink inside protected state cannot be replaced or removed.
  An `os.open` wrapper supplies the directory descriptor omitted by CPython's
  built-in `open` audit event. Reads are allowed. Tests must patch directory
  helpers to temporary directories when they need to write state. Hard-link
  sources are also checked, since a new alias could otherwise allow later writes.
- Process audit events inspect command positions: the executable and `argv[0]`,
  the command after an `env` wrapper, and the first word of each pipeline or
  `&&`/`||`/`;`-separated command in shell `-c` strings and `os.system`.
  They reject kubectl, terraform, docker, aws, helm, ssh, scp, curl, wget, crane,
  git-push/git-receive-pack, and `git push`/`git fetch`/`git clone`, including
  absolute executable paths. Arguments such as `/bin/echo aws` are allowed;
  `bash -c "echo hi | kubectl apply"` is blocked. A temporary directory of failing
  stubs precedes `PATH` (exit 99); its git stub rejects those same network verbs
  and forwards other commands, including `git --version` and `git status`.
  `PYTHONPATH` supplies a `sitecustomize.py` so child Python interpreters install
  the same guard.
  Inherited state roots remain protected when a child enters a copied repository;
  that repository adds its own `.state` root.
- Socket connect/sendto/sendmsg and getaddrinfo/gethostbyname/gethostbyaddr events
  permit only loopback IP addresses (`127.0.0.0/8`, `::1`) and `localhost`. External
  connections and lookups raise `RuntimeError("test isolation: external network")`
  before the operation; state writes raise
  `RuntimeError("test isolation: write into checkout state")`. AF_UNIX sockets
  with local path addresses are allowed. A sendmsg without a destination uses
  the connection already checked at connect time.

The patch-target scanner is the **primary control** for stale patches. This guard
is a safety net against accidental leaks from missing or stale patches, not a
defence against a hostile test author and **not an OS sandbox**. It does not cover:

- Writes through pre-opened descriptors (for example `os.write`) or saved
  `os.open` aliases that bypass the wrapper's directory-descriptor check.
- Pre-existing hardlinks outside protected state, or concurrent symlink races.
- Children started with `python -S`, `python -I` or `python -E`, which bypass the
  injected startup hook; `env -i` with a replaced PATH can also remove both the
  Python hook and the PATH stubs.
- Shells that assemble absolute tool paths. Command checks recognize simple
  literal commands; they do not evaluate shell expansions or Git aliases.
  Non-Python children can also perform their own filesystem/network operations.
- Raw syscalls and native extensions that bypass Python audit events.

## Ordered control flow

```text
operator AWS session
  -> Terraform bootstrap: versioned encrypted state bucket
  -> Terraform platform: VPC / EKS / KMS / S3 / ECR / EBS CSI / EFS CSI
  -> Kubernetes phase 1: EFS-backed local registry
  -> Docker publication + registry digest readback
     -> mirror by exact public Docker Hub digest: Management / Driver
     -> build: storage-controller / Agent base / Candidate overlay / Verifier
     -> mirror: exact LiteLLM / PostgreSQL upstream digests
  -> Kubernetes phase 2a: PostgreSQL + LiteLLM gateway/admin UI
  -> Kubernetes phase 2b: simulator migration/admission bootstrap on SQLite PVC
  -> Kubernetes phase 2c: Management restarts as kubernetes/local-k8s
  -> Management API publishes scripted speed-add versions + Evaluation
  -> Management creates one frontier-run-* namespace and Driver Job
  -> storage-controller creates the exact run-local PV/PVC
  -> Driver creates Agent and train/eval Verifier Jobs
  -> Driver returns score/evidence; Management removes the Run namespace
```

AWS and Kubernetes are separate apply stages because EKS must exist before
fresh API endpoint `/32` facts and cluster objects can be rendered. Registry
and runtime are also separate phases because Kubernetes must be able to pull
the exact digests produced by the site-local publication.

The application release is not assembled from a dirty source tree. The site
file freezes one FrontierCS2 Git commit, requires each Management/Driver Docker Hub
tag to equal `sha-<commit>`, and pins both images by digest. The exact clean
checkout remains required for the task-authoring handoff; the prebuilt images
are mirrored into the site-owned registry and deployed by its digest readback.

## LiteLLM gateway and UI

LiteLLM is a site-owned service in `frontier-litellm`, not a Run-owned Pod. It
uses one PostgreSQL StatefulSet backed by an encrypted retained EBS volume and
stores UI-created model and credential records in that database. The installer
generates the stable master key, salt, and database password under
`.state/<site>/` with mode `0600`, then writes only the corresponding Kubernetes
Secret. It never accepts, searches for, logs, or creates a model-provider key.

The UI and API have a fixed ClusterIP and no public LoadBalancer or Ingress.
An operator reaches `/ui/` only through `kubectl port-forward`; UI access is
admin-only. NetworkPolicy permits DNS, PostgreSQL, and public HTTPS provider
egress while excluding private and link-local ranges. Private model endpoints
therefore require a reviewed network-policy change.

Gateway/UI readiness and model-Run integration are separate gates. The current
Management `local-k8s` backend does not enable
`MANAGEMENT_LITELLM_SERVICE_CIDR`, issue per-Run virtual keys, or route Agent
traffic through the gateway. Those controls remain part of the production
backend rather than this Preview model path. GPU execution is separate: both
profiles now instantiate the same `ModalGpuBackend` and Driver quota/SSH/
verification/cleanup implementation. Only storage and registry adapters remain
profile-specific. Docker Hub GPU tag readback uses an independent HTTPS-only
anonymous client, never the cluster HTTP client or ECR credentials.

## Storage and registry

Management remains one SQLite writer on an encrypted EBS RWO PVC. A KMS-
encrypted EFS filesystem has two access points:

- registry: persistent OCI content, UID/GID 65532;
- object store: shared Preview datasets and Run-fenced artifact directories.

The storage controller only reconciles Management-owned Run namespaces. It
creates the exact PV/PVC objects and, when
`execution.agent_external_egress=true`, one additional additive NetworkPolicy
named `frontier-agent-external-egress`. That policy selects
`frontier.dev/network-role=agent` and permits only CoreDNS plus public TCP 443;
it does not alter or imitate Management's frozen six-policy Run contract. This
is an internal-beta escape hatch for lab-owned model endpoints and credentials,
not the production isolation boundary. With the flag `false` the controller
deletes the policy from every Run namespace.

With GPU disabled, the controller cannot read Secrets. With GPU enabled, it
mounts the named site Modal Secret read-only, can get that exact Secret name,
list Jobs, and create immutable per-Run copies. It requires a matching
Management-owned Run/attempt and digest-pinned Driver/cleanup Job with the exact
secret references. It never gives credentials to Agent/Verifier Pods and cannot
create Pods/Jobs, mutate Management state, or launch a Driver. Namespace cleanup
removes the per-Run Secret; the site Secret remains. Its own
`frontier-storage-controller-boundary` policy is
unchanged: the new NetworkPolicy writes use the same Kubernetes API server and
fit its existing API-CIDR TCP 443 egress. Multiple static PV identities point
to the same Preview-only EFS access point so every Driver sees the two frozen
datasets. The trusted Driver still enforces the exact Run artifact prefix;
this shared-filesystem profile is not the production cross-Run isolation
boundary.

GPU-enabled sites also render the namespaced
`frontier-management-modal-observer` Role and RoleBinding for the native
Management Modal observer. The binding grants only the `frontier-management`
ServiceAccount `get` on `modal.credentials_secret` in the site platform namespace
(`frontier-system`, matching `MANAGEMENT_NAMESPACE`). It grants no Secret
list/watch, write, or cluster-wide access and does not bind Agent or Verifier
accounts. CPU-only manifests omit this observer grant. This permission is
separate from the storage controller's per-Run credential delivery permissions.

After Management deletes a terminal Run namespace, the controller deletes
only the matching `Released`/`Failed`, `Retain`, exact-volume-handle PV object.
The EFS bytes are not deleted by that reconciliation.

## Bundled task image source

The catalog's native task trees and their Dockerfiles are bundled under
`tasks/frontier-cs-2.0-demo/`. Each task uses its own root as the build context;
sparse additionally supplies Agent/Verifier GPU images. Judge inputs are
absent from Agent layers. Dataset selectors are JSON bodies stored at
content-addressed paths in a single-layer image. The pure renderer derives
policy, budgets and source identities; the network-only resolver pins published
images separately. Deployment consumes public digest-pinned catalog images.

Reference overlays use the same in-cluster mirroring path as the catalog
Codex overlay, while remaining outside the catalog's single-candidate contract.
Their launcher submits the reference patch for train and final verification.
The [parity harness](operations.md#reference-parity-acceptance-gate) requires
terminal Management success, a successful train event and agreement with a
comparable baseline bound to the exported revision. CPU smoke
and registration alone do not establish reference parity.

AL2023 node bootstrap installs one `containerd` `hosts.toml` entry for the
fixed HTTP registry ClusterIP. The registry is reachable only inside the VPC
and Run NetworkPolicy path. Locally built images use source-tree hash tags;
mirrored Management/Driver images retain the existing site-local Git tag
scheme. The registry-authoritative digest is read back for every workload.

## Admission bootstrap

The currently pinned Management release opens non-production admission only
when started in simulator mode. A fresh site therefore starts that image once
against the final SQLite PVC to run migrations and record nonproduction OPEN,
then immediately rolls the same Deployment to `kubernetes/local-k8s`. No
Evaluation is submitted during the simulator phase. The final process creates
and observes the real Kubernetes Driver.

This is an explicit Preview compatibility bridge. It is not a production
admission receipt, production matrix, or substitute for the strict production
rollout protocol.

## Ownership

| Owner | Resources | Lifecycle |
| --- | --- | --- |
| portable release | pinned source commit; public Docker Hub Management/Driver tag + digest | application-owned immutable input mirrored into the site registry |
| Terraform bootstrap | state S3 bucket and local bootstrap state | protected, long-lived |
| Terraform platform | VPC, EKS, node groups, KMS, artifact S3, ECR, EFS, CSI add-ons | site-owned; never Run cleanup |
| Management evidence identity | exact Management KSA web identity; artifact `runs/*` read; site KMS decrypt via S3 | site-owned; never exposed to Agent/Verifier |
| Kubernetes platform | Management/registry/controller Deployments, SQLite PVC, seed PV/PVC | site-owned |
| LiteLLM platform | gateway Deployment, PostgreSQL StatefulSet/PVC, generated platform Secret | site-owned; never Run cleanup |
| Modal platform identity | mounted named site Secret; fixed environment/app; exact NAT `/32` allowlist | operator-owned; copied only to trusted Run Drivers |
| RunAttempt | `frontier-run-*`, Driver/Agent/Verifier, run PV/PVC identity | Management/controller created; namespace cleaned at terminal |

## Production boundary

Executable Preview's automatic gate proves the CPU orchestration path with a
scripted Candidate. The pinned application now derives dispatch-scoped local
dataset and registry maps for the selected catalog tasks. Each deployment still
requires its own live workload Evaluations before those tasks can be called accepted.
Production still requires strict VPC CNI bootstrap, the closed
ValidatingAdmissionPolicy set, attempt-scoped AWS IAM/S3/ECR, LiteLLM
serving-ready acceptance and per-Run virtual-key lifecycle for model tasks,
fresh-site Modal GPU identity/budget and live acceptance,
authenticated browser ingress, backup/restore, and fresh positive/negative
acceptance.

## Renovation record

The five database tasks receive an imported native policy revision reserving
8 GiB of model audit capacity. Runtime image contents and scientific semantics
are unchanged; new source tags and TaskVersions bind the policy and source-lock
changes. See model audit capacity for the observed
44 MiB capture failure, rollout procedure, and remaining session limit.
