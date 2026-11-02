# Golden behaviour fixtures

The Agent egress closure updates ten CPU/Modal fixtures. The example
site sets `execution.agent_external_egress` to `false`, so each bundle
`config.json`, its `release.json` checksum and `bundle.sha256` change, and the
four runtime manifests render `FRONTIER_AGENT_EXTERNAL_EGRESS=false`. The
storage-controller ClusterRole gains `delete` on `networkpolicies`, so the
controller can remove `frontier-agent-external-egress` while the flag is false.
No other fixture bytes change.

The official Codex 0.162.1 overlay update changes the catalog Candidate image
pin. Golden Candidate documents use a synthetic digest and remain unchanged;
live registration derives a new identity from the actual mirrored digest.
TaskVersion documents and CLI help retain their existing bytes. The Candidate bundles the
full official native package from merged platform source
`cc87b067cb5b103de6bfc05e40fe0da8527dba5b`; publication verification is described
in [Candidates](candidates.md#codex-with-stored-final-support). Earlier release
notes below describe their historical fixture changes.

The workspace-limit release bound Management and Driver to the same published
`78905901bccc1d51301961cafc4fb638ec6d6c8b` source, documented in the
workspace-limit publication record.
CPU and Modal bundles update only source/image pins, release documents and
checksums. Existing Management resource allocations and gateway policy remain
unchanged. Runtime upgrades preserve the existing gateway configuration.
Task and Candidate identities and CLI help are unchanged. This update does not
deploy a live site; active evaluations retain their frozen dispatches.

The workspace-limit release refreshes six CPU/Modal bundle fixtures: only the
platform source, Management/Driver pins and derived configuration/release
checksums change. All other fixture bytes remain unchanged.

`tests/test_golden.py` compares every output byte against
`tests/fixtures/golden/`. Initial fixtures were captured before Phase 0 changes
from `c64eeeff6d604050eb8cf364d5ee75553bea42e9` (`codex-overlay-v2`), then
compared byte for byte with the integrated test's output. These are synthetic,
offline fixtures, separate from the live-state migration fixtures.

The current 46 files cover:

- `modal/` and `cpu/`: bootstrap, runtime with `simulator_bootstrap` false and
  true, LiteLLM namespace and manifests, and every `render_bundle` file plus
  its returned config. Inputs use `config/site.example.json` with Modal
  explicitly enabled, and explicitly disabled with its `modal`
  block removed, respectively. The example itself defaults to disabled.
- `documents/`: TaskVersion documents for every catalog task, the Codex
  CandidateVersion document, and both smoke documents. Documents use the
  renderer's `json_text` serialization; rendered bytes are kept as returned.
- `cli/`: root help and every subcommand's help, each from a fresh Python
  subprocess with an explicit minimal environment: `PATH`, temporary `HOME`,
  `LC_ALL=C`, `LANG=C`, `PYTHON_COLORS=0`, `NO_COLOR=1`, `TERM=dumb`,
  `COLUMNS=80`, `PYTHONHASHSEED=0`, and `PYTHONIOENCODING=utf-8`, plus the required
  inherited `PYTHONPATH`, `TMPDIR` and `_PREVIEW_TEST_STATE_PATHS`.
  `PYTHONPATH` propagates the isolation guard to child processes. `LANGUAGE`
  and other inherited variables are absent. Python's `-B` flag prevents bytecode writes.

The capture reuses `Fixture.outputs` / `Fixture.images` and
`resolved_catalog()` / `mirrored_images()` from the existing tests. Bundle
output goes into a temporary directory; its state path is the synthetic
`/frontier/golden-state/preview`, rendered as text without accessing `.state`.
No cloud, Docker, network, real site config, or credentials are needed.

The Modal runtime fixtures include the namespaced
`frontier-management-modal-observer` Role/RoleBinding, granting Management only
`get` on the configured Modal Secret. Both executable and simulator renderings
include it when GPU is enabled; CPU fixtures omit it. Renderer tests also check a
custom Secret name, exact namespace/subject/verbs, and CPU-only omission.

The governance update deliberately changes all runtime fixtures to the upstream
15000 ms controller poll. Executable CPU and Modal fixtures additionally include
Management-only diagnostic read RBAC and an EFS `artifacts` subPath write mount
for diagnostic expiry; their dataset/object parent mounts remain read-only.
Simulator fixtures omit the diagnostic grant and mount. This changes the rendered
configuration checksum, not task or Candidate identities.

The PVC alias correction changes only executable CPU/Modal fixtures: both EFS
mounts reference one `objects` volume. Its PVC source permits writes; its parent
mount stays read-only and only `artifacts` is mounted writable. Separate aliases
for the same PVC blocked kubelet startup in the observed Preview rollout.

Run the comparison with `python3 tests/test_golden.py` (also in `make test`).
To regenerate deliberately, run `python3 tests/test_golden.py --update`.
Review every golden diff as a behaviour change, including fixture membership,
IDs, digests, CLI text, and bundle checksums; do not update merely to silence
a regression. Obsolete fixtures require explicit removal and review. The
example site defaults to CPU selection. Both capture variants set the flag
explicitly. Phase 1 adds only `--check` to the existing seed CLI contract.

For integration onto `c33db66` on September 11, all fixtures were regenerated
with the new platform pins and six exported definitions. An isolated temporary
worktree of `origin/main` rendered the same artifacts with main's own functions
and fixture helpers, using only locally verified LFS objects. Comparison found
45 byte-identical files and one additive-flag-only CLI file: stripping exactly
the new `--check` usage line and help stanza reproduces main's help bytes.
There were no other differences. The two additional CLI files cover main's
`upgrade-runtime` and `upgrade-local-network` subcommands.

## Live-state migration fixtures

Before copying, both JSON documents passed duplicate-key and token-pattern
checks and strict field/value allowlists: the known schema and site, typed
TaskVersion/CandidateVersion IDs, SHA-256 digests, Docker Hub image references,
and in-cluster `172.20.0.50:5000/frontier/...` image tags/references. There are
21 mirror records (Codex, CPU task images, and reference candidates), with no
credentials or unexpected fields. A preliminary substring detector incorrectly
matched `sk-` inside the schema's `task-seed`; this was checked to be the sole
match and corrected to recognize a token prefix at a word boundary before the
strict allowlist check and copy.

| File | SHA-256 of copied bytes |
| --- | --- |
| `task-seed-receipt.json` | `fa68403ee6c3f083a62b0a2fb207f686172badf794f092351da16f5829c5135d` |
| `task-images.json` | `9596d707921a8a19426887377ee99fe41b3f0a9729fb2e87363eedb5faad6706` |
| `task-catalog.json` (`c64eeef`) | `584e26a921ff8b88c9ee9c04a156b9e2d5b4cbb1a37e75ef9ee2db0b0ed14fd8` |
| `task-definitions.json` (`c64eeef`) | `c26bf5888d9eceaf85be500f521aca1dd06a9f785878f3ff0152c7129447f002` |

`tests/test_live_state_migration.py` reproduces the historical site-local recipe
with the site selector: `features.modal_gpu=false` removes only
`sparse_video_attention_policy` from the effective catalog, preserving task
order and the Codex candidate. The archived `task-catalog.json` and `task-definitions.json` in this fixture
directory are exact `c64eeef` Git blobs. Tests upgrade the archived v1 catalog
in memory with unverified baselines and no reference descriptors, then validate
and select against the archived definitions. Current tracked files have the six
re-exported beat-spec tasks and are deliberately not used for this proof.
The raw, unnormalized pruned catalog hashes to
`sha256:01782eaaba38091dea2c8502dd4cbc0372de69741e65882903cf772c26933e8b`,
exactly the receipt's `catalog_sha256`; the full catalog does not match.
Validation adds image references. For catalog v2, call
`task_selection.deployment_catalog` before hashing: it preserves the v1 schema
marker and omits reference/baseline sections. This recreates the exact original
deployment input while keeping publication metadata in the tracked catalog.

Task documents merge the default `openrouter.ai` model domain with the authored
allowlist, as documented in the September 8 release. They use the saved
in-cluster Agent/Verifier mirrors and catalog dataset selectors. Omitting that
domain changes each TaskVersion ID. All ordered receipt IDs reproduce:

| Task | TaskVersion ID |
| --- | --- |
| `batch_local_counting` | `taskver_78DEK7R6H5KCZ5G9X7SGN0F2HR` |
| `bounded_evm_path_prediction` | `taskver_5QSVB08MFVKDG895H20QSZ1XMG` |
| `evm_state_checkpoints` | `taskver_2T6D59T3627WW9J74235N8P1H7` |
| `graph_mining_method_selection` | `taskver_5EZY5P5H1RNQHVTWBA7R20JWQT` |
| `join_tree_rewriting` | `taskver_1T09NQZP5ZKMPJ47E671PQGJCJ` |

The Codex candidate uses the saved mirrored overlay and existing default
execution/model configuration; its train poll timeout uses the complete
definitions' maximum wall-time budget, `378000` seconds (also the CPU maximum
in this release). This reproduces
`candver_0ASDBET59WHMHVB37TX6P5R02C`. No historical identity algorithm override
or additional catalog edits are needed.

The receipt binds that exact pruned catalog and the resulting version IDs.
It does not bind the full committed catalog, the whole mirror-file byte hash,
site infrastructure health, or any Evaluation result. These offline checks
prove migration identity compatibility, not current deployment or workload
acceptance; reference mirrors are captured but are not IDs in the seed receipt.

The fixture tests also rehearse backup, v2 receipt replacement and exact-byte
rollback in temporary directories. They use the captured mirrors and IDs;
archived v1 catalogs have no reference publication descriptors. Current catalog
seeding and reference mirroring are covered separately. The current catalog
need not reproduce historical IDs. Follow [Live site changes](operations.md#live-site-changes)
for the actual operator procedure.

The RAGEN issue370 paired runtime changes only the application commit,
Management/Driver public pins and their derived release/config checksums in
the six CPU/Modal bundle fixtures. Task/Candidate documents and CLI help remain
unchanged. See the paired publication.

The current-run restoration release advances only Management's application
source and public image pin. Its six CPU/Modal bundle fixtures regenerate the
derived configuration, release metadata and checksums. Driver, task/candidate
documents and CLI help remain unchanged. See
the Management publication.
