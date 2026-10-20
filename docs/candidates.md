# Candidates

A Candidate is the agent harness that works on a task. It ships as a
filesystem overlay image that the Driver mounts over the task's Agent image at
dispatch. The task provides its README, source and verification contract; the
Candidate provides the harness, its launcher (`/opt/frontier/candidate/launch`),
the `frontier-submit` and `frontier-gpu` tools and their skills.

Every harness used in the FrontierCS 2 formal evaluation has a recipe under
[`candidates/`](../candidates). [`config/candidates.json`](../config/candidates.json)
lists each evaluated model with its harness, harness version, model id,
reasoning effort and the published overlay image (repository, tag and digest):

| Model | Harness | Version | Effort | Recipe |
| --- | --- | --- | --- | --- |
| Claude Opus 5.5 | Claude Code | 2.1.296 | max | [`candidates/claude-code`](../candidates/claude-code) |
| GPT-6 Astra, GPT-6.1 Sol | Codex CLI | 0.162.1 | xhigh | [`candidates/codex`](../candidates/codex) |
| Gemini 4 Argon | Pi | 1.2.0 | high | [`candidates/pi-gemini`](../candidates/pi-gemini) |
| GLM 5.3, GLM 5.3 Flash | ZCode | 0.16.9 | max | [`candidates/zcode`](../candidates/zcode) |
| Muse Wonder 1.0 (max), Muse Spark 1.3 (high) | Muse Code | 1.4.2 | see model | [`candidates/muse-code`](../candidates/muse-code) |
| DeepSeek V4.1 Flash | DeepSeek Harness | 0.2.0-rc.2 | max | [`candidates/deepseek-harness`](../candidates/deepseek-harness) |
| Kimi K3 (high), Kimi K2.7 (medium) | Kimi Code | 0.31.0 | see model | [`candidates/kimi-code`](../candidates/kimi-code) |
| Qwen 3.8 Max | Qwen Code | 0.25.0 | high | [`candidates/qwen-code`](../candidates/qwen-code) |
| Step 5 Preview | Step Code | 0.1.0 | high | [`candidates/stepcode-step5`](../candidates/stepcode-step5) |

Publishing or registering a Candidate does not establish that a model works
with it. Each new model/harness pair still needs its own smoke run covering
model calls, train feedback, a stored or formal final, usage accounting and
cleanup.

## Codex (built in)

Codex is the built-in Candidate. The catalog's `candidate.overlay_image` in
[`config/task-catalog.json`](../config/task-catalog.json) pins its overlay at
`docker.io/joyemang33/fcs2-codex-overlay`; `make apply` (`seed-tasks`) mirrors
it and registers the Codex CandidateVersion. Nothing else is needed to run it.
The recipe is in [`candidates/codex`](../candidates/codex), which also holds
the canonical sources of the shared submission and GPU clients.

Use a private Candidate execution configuration for the direct model route:

| Variable | Value |
| --- | --- |
| `FRONTIER_MODEL_DIRECT_BASE_URL` | Explicit HTTPS OpenAI-compatible provider endpoint |
| `FRONTIER_CODEX_MODEL` | Exact model alias accepted by the provider |
| `FRONTIER_CODEX_REASONING_EFFORT` | Optional effort, e.g. `xhigh` |
| `CODEX_API_KEY` | Operator-supplied provider key |

Keep keys out of Git, images, shell arguments and public receipts. The Driver
owns the hard Run deadline.

## Building another harness

Each `candidates/<harness>/` directory is a self-contained Docker build
context. Build a `linux/amd64` image from it:

```bash
docker build --platform linux/amd64 -t REGISTRY/frontier-candidate-HARNESS:TAG candidates/HARNESS
docker push REGISTRY/frontier-candidate-HARNESS:TAG
```

The Dockerfiles download only public, version-pinned upstream artifacts (npm
packages with lockfiles or checksums, a pinned GitHub commit, or the vendor's
download URL with a SHA-256 check). No vendor binaries are stored in this
repository. Each directory's `README.md` documents its launcher variables and
how to run its unit tests; `make test` runs the offline ones.

`source.json` in each directory records the platform source revision and the
SHA-256 of the official Dockerfile, of the local Dockerfile and of every file
the Dockerfile copies. The local Dockerfile differs from the official one only
in COPY source paths: the shared clients live under `submit/` and `gpu-helper/`
beside each harness instead of in a shared directory. To refresh a recipe from a
clone of the platform source repository:

```bash
python3 scripts/sync_candidate_overlay.py --source /path/to/source --harness HARNESS
python3 scripts/sync_candidate_overlay.py --source /path/to/source --harness HARNESS --check
```

The commit defaults to `harnesses.<name>.recipe.source_commit` in
`config/candidates.json`; pass `--commit` to select another one.

## Registering another harness

1. Push the image and read its digest back from the registry. Never relabel an
   existing published tag with different bytes. Use the published digest from
   `config/candidates.json` to reproduce an evaluated configuration.
2. Before registration, verify the deployment mirror's manifest, configuration
   blob and every layer against the published digest. Pass the explicit site
   registry authority as `deployment_registry` to
   `preview.publication.candidate_version_document`, along with the verified
   `mirrored_overlay`. The helper rejects a missing mirror or a reference outside
   that registry.
3. Register a CandidateVersion with entrypoint `/opt/frontier/candidate/launch`,
   working directory `/workspace`, user `1000:1000` and the harness's execution
   environment from its README (model alias and effort).

All harnesses except Codex direct mode and Claude Code call the model through
the Driver's Run-local model proxy: the launcher receives only the scoped
`FRONTIER_MODEL_PROXY_TOKEN` and the model alias, and the provider key lives in
the LiteLLM gateway ([`runtime/litellm`](../runtime/litellm)). Configure one
LiteLLM route per alias; `candidates/muse-code/gateway-models.yaml` and
`candidates/stepcode-step5/gateway-models.json` are examples. Claude Code
calls the Anthropic API directly with an API key supplied to the Run as a
secret (`ANTHROPIC_API_KEY`).

Changing the registered reference creates a new CandidateVersion even when the
image bytes are identical. Preserve the earlier Run and CandidateVersion and
require the replacement's own smoke before using it in a campaign.

## Submission and GPU tools

Every overlay puts `/opt/frontier/candidate/bin` first on PATH and installs the
same clients:

- `frontier-submit train ARCHIVE` returns immediately with a verification id;
  `frontier-submit status VERIFICATION_ID` reads one result. There is no wait
  flag or client polling timeout.
- `frontier-submit store-final /absolute/path/final.tar` saves the Agent-chosen
  fallback without running verification or ending the Agent. The receipt reports
  remaining answering seconds and a null save allowance (no save-count quota).
  If no formal final exists, normal exit, crash or answering timeout selects the
  latest accepted save; cancellation does not.
- `frontier-submit final` is irreversible and takes priority over saves.
- `frontier-gpu` reports `frontier.gpu-helper/v3` and provides `create`, `list`,
  `status`, `push`, `exec`, `ssh`, `pull`, `delete` and `quota`. The Driver
  proxies and audits SSH and meters runner time against the task's GPU budget.
  With `features.modal_gpu=false`, GPU tasks are excluded and runner creation is
  unavailable.

## GPU helper V3 attestation (Kimi Code)

Site selection accepts `frontier.gpu-helper/v3` tasks only for a Candidate whose
image and sources are attested in the repository. The attested V3 Candidate is
Kimi Code: [`config/kimi-candidate-image.json`](../config/kimi-candidate-image.json)
binds the published image digest, its source tag/commit, the linux/amd64
manifest selection, and the SHA-256 and installed mode of each file:

- `candidates/kimi-code/Dockerfile`
- `candidates/kimi-code/launcher.py`
- `candidates/kimi-code/submit/frontier_submit.py`
- `candidates/kimi-code/submit/SKILL.md`
- `candidates/kimi-code/gpu-helper/frontier_gpu.py`
- `candidates/kimi-code/gpu-helper/frontier_gpu_connect_proxy.py`
- `candidates/kimi-code/gpu-helper/SKILL.md`

`scripts/task_selection.py::attested_agent_gpu_interfaces` verifies these
before selection. Every listed source must remain inside `candidates/kimi-code`
and match its receipt; the source provenance in `candidates/kimi-code/source.json`
must bind the same commit and official Dockerfile, and the registry readback
must bind all six installed files and modes to the same digest. Missing or
invalid bindings reject V3 selection. This offline check does not register Kimi
automatically and does not upgrade the seeded Codex overlay.

Changed local source bytes cannot be attested by the previous image's receipt.
To refresh the attestation, rebuild from `candidates/kimi-code`, publish a
public `linux/amd64` manifest, read back the six installed files and their
modes from the registry image, then update the digest and hashes in
`config/kimi-candidate-image.json`, run `make test` and both config validations.
