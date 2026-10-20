# Site task selection

The full tracked catalog is validated against the complete definitions before
site selection. `scripts/task_selection.py` derives Modal requirements from
the agent and verifier GPU instance limits. Malformed GPU policies fail closed.
With `features.modal_gpu=false`, GPU tasks are excluded with the reason
`requires features.modal_gpu`; an empty selection is an error.

Selected tasks must use a helper contract attested by a pinned Candidate overlay.
Codex remains V2. Kimi Code's separately registered V3 overlay is accepted only after
its checked-in image digest, source tag/commit, helper digest binding and local
file hashes are verified from `config/kimi-candidate-image.json`. Its
`gpu_helper.files` receipt must include every source of the recipe in
`candidates/kimi-code`: `Dockerfile`, `launcher.py`, `submit/frontier_submit.py`,
`submit/SKILL.md`, `gpu-helper/frontier_gpu.py`,
`gpu-helper/frontier_gpu_connect_proxy.py` and `gpu-helper/SKILL.md`. Every listed hash must be a lowercase
`sha256:<64 hex digits>` value matching the local file. Missing or invalid hashes
reject V3 attestation and selection with an error identifying the source. This does not
make Codex V3-compatible or register Kimi automatically. CPU-only selection does
not require a GPU Candidate. Unknown selected helper contracts fail closed.
To include compatible GPU tasks, enable `features.modal_gpu=true`, supply the documented Modal
configuration/credentials, and use a V3 Candidate for Sparse.

The effective catalog preserves the raw catalog's keys, task order, candidate
and selected task objects. Receipt hashes use the deployment projection before
validation adds image references: retain the v1 schema marker and omit the v2
reference/baseline sections. This preserves existing receipt bindings; publication
metadata is validated separately. The selection digest binds the full catalog hash,
effective catalog hash, ordered selection, exclusions and sorted unique model
domains. Definitions remain unchanged; Candidate timeout calculations always
use the complete definitions, including excluded tasks.

`execution.model_domains` optionally supplies a list of lowercase DNS names
(up to 32). Omitting the field defaults to `["openrouter.ai"]`; explicitly setting
`[]` adds no model domains and preserves the authored task allowlist, including an
empty allowlist. Nonempty lists are unioned with the authored allowlist; `[]` does
not remove domains required by the task. This applies to both `seed-tasks --check`
and `seed-tasks`. Explicit changes intentionally create new task identities and
selection receipts; they do not rewrite existing versions. Selection itself never
edits the tracked catalog or task trees.

`validate` prints selection and its effective catalog hash. `render` writes
`.generated/<site>/task-selection.json` beside the deployment bundle; this
artifact does not participate in `bundle.sha256`. Deployment mirrors only
selected task images, Codex, and references for selected tasks.

The `frontier.deployment-task-seed-receipt/v2` receipt binds `site`, the full
`catalog_sha256`, `effective_catalog_sha256`, `selection_digest`, `selected`,
`excluded`, `model_domains`, `candidate_version_id`, ordered `task_version_ids`,
and `tasks` indexed by name with each TaskVersion ID and definition digest.
Readers retain v1 support: its catalog hash must equal the current effective
catalog hash. V2 binds both catalog hashes and the selection digest. Parity
uses the site configuration (`--config`) and rejects excluded tasks before HTTP.

`seed-tasks --check` reads saved `task-images.json` and the existing kubeconfig,
computes selected task and Codex documents, and GETs their immutable IDs from
Management. It prints MATCH, MISSING or DIFFERS for each ID and exits nonzero
if any record is absent or its definition digest differs. It does not mirror,
run dataset Jobs, POST versions, render files, refresh kubeconfig or write
`.state`. Missing or stale saved mirrors require operator reconciliation.

See [Candidates](candidates.md) for the current Codex V2 and Kimi V3 contracts.
