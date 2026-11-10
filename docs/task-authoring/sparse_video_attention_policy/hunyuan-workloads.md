# HunyuanVideo workloads for Sparse Video Attention Policy

Status: source and generated targets are merged. All six HunyuanVideo
workloads have retained dense/reference videos and ready manifest records. Image
publication and native qualification are separate release gates; this document's
generation commands describe how the retained targets were produced.

## Design rules

- **Option B.** The HunyuanVideo EAR processor is a second editable, stubbed
  surface exactly like Wan's. A candidate must implement both models' policies.
  Existing candidate patches are not rejudged; the 11 candidates are rerun as
  agents later in separate, non-formal evaluations.
- **Workloads.** Final = the existing 8 Wan clips + 4 HunyuanVideo clips (12).
  Development = 2 Wan clips (the existing `feedback_0/1`, prompts 1/2) + 2
  HunyuanVideo clips (prompts 3/4, new seeds). Each clip is its own workload with
  the unchanged strict per-workload, per-metric gate (latency <= 0.96 x reference,
  PSNR >= ref - 0.25 dB, SSIM >= ref - 0.01, LPIPS <= ref + 0.005).
- **Final timeout 12,600 s** everywhere it is defined, with the Verifier GPU
  budget scaled by the existing formula.
- **81 frames** for HunyuanVideo (one explicit constant, `HUNYUAN_NUM_FRAMES`).
- **License.** Tencent Hunyuan Community License; text stored from the pinned
  revision.

## Workloads

| id | model | prompt | seed |
| --- | --- | --- | --- |
| `wan_t2v_720p_feedback_0` | Wan2.1 | 1 | 104729 |
| `wan_t2v_720p_feedback_1` | Wan2.1 | 2 | 104759 |
| `hunyuan_t2v_720p_feedback_0` | HunyuanVideo | 3 | 104779 |
| `hunyuan_t2v_720p_feedback_1` | HunyuanVideo | 4 | 104789 |
| `wan_t2v_720p_graded_0..7` | Wan2.1 | 1,2,3,4,1,2,3,4 | 130363, 130367, 130399, 130409, 130411, 130423, 130439, 130447 (unchanged) |
| `hunyuan_t2v_720p_graded_0..3` | HunyuanVideo | 1,2,3,4 | 130457, 130469, 130477, 130483 |

All 16 seeds are distinct and no (model, prompt, seed) item is shared between
suites (`task_tests/.../test_hunyuan_workloads.py`). The former Wan
`feedback_2/3` (prompts 3/4) leave the development suite; their four cached MP4s
were removed from `judge/dense_targets/`. Wan graded ids, seeds, targets and
latencies are unchanged.

Waves (four H100s, one process per GPU, as before):

- development: one wave `[Wan fb0, Wan fb1, Hunyuan fb0, Hunyuan fb1]`;
- final: three waves `[Wan g0-3]`, `[Wan g4-7]`, `[Hunyuan g0-3]`.

Graded waves are model-homogeneous, so every HunyuanVideo graded latency is
measured under the same four-HunyuanVideo concurrency its targets are generated
with. The mixed development wave is reproduced exactly by the target generator
(Wan slots run as fillers).

The development progress resolution stays 25 points per workload.

## Editable surfaces: stub and reference split

| file | public tree | `judge/reference.patch` |
| --- | --- | --- |
| `svg/models/wan/attention.py` | `WanAttn_EARAttn_Processor.__init__` raises (unchanged) | unchanged hunk |
| `svg/models/hyvideo/attention.py` (new) | upstream bytes except `Hunyuan_EARAttn_Processor`, whose body is replaced by `"""EAR processor hook."""` and an `__init__` that raises `NotImplementedError` | new hunk restoring the upstream class (`semantic_aware_permutation`, `attention_core_logic`) |
| `svg/kmeans_utils.py` | three helpers stubbed (unchanged, shared by both models) | unchanged hunks |

Applying `reference.patch` to the public tree reproduces all three upstream files
byte-for-byte at `f89aedaf169ac2ae5b186bda674e53c3dc08c476` (checked locally with
`git apply`, and by the task test through `apply_submission_patch`).
`baseline_naive.patch` gains the matching `pass` body for the HunyuanVideo class.

Allowed paths are now `svg/models/wan/attention.py`,
`svg/models/hyvideo/attention.py`, `svg/kmeans_utils.py`,
`svg/solver_extensions/` in the evaluator `CONFIG`, `judge/config.yaml`,
`policy.json`, `bin/make_submission.sh`, `artifact_manifest.json` and both
statements. The 12-file and 262,144-byte limits are unchanged.
`HunyuanVideoPipeline` joins the forbidden added-source tokens, and the
structural check also requires `Hunyuan_EARAttn_Processor` to be defined in
`svg/models/hyvideo/attention.py`.

## Vendored upstream files

From `svg-project/Sparse-VideoGen` at `f89aedaf169ac2ae5b186bda674e53c3dc08c476`
(fetched with `git fetch --depth 1 origin <commit>`, no account). Hashes are in
`hunyuan/provenance.json`; the task test checks them.

- byte-identical: `hyvideo_t2v_inference.py`,
  `scripts/hyvideo/hyvideo_t2v_720p_ear.sh` (mode 0755 like the Wan script),
  `examples/7/prompt.txt` (only because the byte-identical shell script reads it;
  the evaluator replaces the prompt), `svg/models/hyvideo/{__init__,custom_models,inference,placement,utils}.py`;
- stubbed: `svg/models/hyvideo/attention.py`.

Verified Phase 1 facts: the 720p EAR script uses 50 steps, TFP 0.1, LFP 0.03,
QC 400, KC 1000, top-p 0.9, min-KC 0.10, k-means init 50 / step 2 and
`--zero_step_kmeans_init`; the inference program sets `guidance_scale=6.0`
(embedded guidance; true CFG stays off in Diffusers 0.34) and
`FlowMatchEulerDiscreteScheduler(shift=7.0)`; it loads with
`revision='refs/pr/18'`, which Diffusers 0.34 ignores for a local directory
(`pipeline_utils.py` and `hub_utils._get_model_file` take the local-path branch).
So, as for Wan, only `--model_id` is redirected (task-owned `.frontier/python`
for the Agent; the evaluator's private shim for scoring).

## 81-frame deviation

Upstream and the paper run HunyuanVideo at 129 frames (33 latent frames,
119,056 joint tokens). The task fixes 81 frames (21 latent frames of 45 x 80,
75,600 video tokens + 256 text slots = 75,856 joint tokens), the same video
scale as the Wan workloads. Rationale: comparable per-clip cost and timeout
across models, a third wave that fits the 12,600 s bound, and the same
21-latent-frame geometry for both policies. The retained integration derives
`num_frame = 1 + num_frames // 4 = 21` and `frame_size = 3600` from the argument,
so no upstream code changes. The constant is `HUNYUAN_NUM_FRAMES` in the
evaluator (shim argument `--num_frames 81`) and the `.frontier/python` override
for the Agent's bench. The upstream default of 129 is not used anywhere.

The HunyuanVideo prompt lengths after the pipeline template are 144, 139, 225 and
117 tokens for prompts 1-4 (tokenizer from the pinned revision), all below the
256-slot context, so the retained post-processing always has a non-empty padding
segment.

## Attention backend

The retained HunyuanVideo module imports
`flash_attn.flash_attn_interface.flash_attn_varlen_func` at module level; the
pinned images do not ship `flash-attn`. Usage at the pinned commit:

- EAR (scored path, candidate and reference): full-attention steps call
  `flashinfer_varlen_func` (FlashInfer `VariableBlockSparseAttentionWrapper`),
  sparse steps the FlashInfer block-sparse kernels; `flash_attn` is never called.
- dense (`--pattern dense`, the fixed targets): `HunyuanVideoAttnProcessor2_0_FlashAttention`
  calls `flash_attn_varlen_func` with packed `(L, B*H, D)` tensors and two
  segments (valid tokens, text padding).
- the old SVG processor (not used).

Decision: no flash-attn wheel. A task-owned `flash_attn/flash_attn_interface.py`
implements exactly that call (non-causal, dropout-free, segment-isolated) with
PyTorch `scaled_dot_product_attention`, the dense backend the retained Wan
processor already uses; unsupported arguments raise. On H100/bf16 SDPA dispatches
to its FlashAttention-2 kernel, so the dense targets are exact attention up to
bf16 accumulation order. It keeps `svg/models/hyvideo/attention.py` byte-identical
to upstream outside the stub. A CPU float64 test checks it against an explicit
per-segment softmax. It is outside the editable surface.

## Warm-up state isolation

The Wan shim already rebuilds processors after the two-step warm-up. The
retained HunyuanVideo SAP/EAR processors keep their k-means caches in
class-level dicts (`centroids_init`, `q_centroids`, `k_centroids`, keyed by
layer), so a new instance alone would let the warm-up's centroids skip the
scored call's 50-iteration initialization. The shim therefore rebuilds the
processors on `transformer_blocks[*].attn` and `single_transformer_blocks[*].attn`
and clears those three dicts on every processor class in the MRO. Wan's shim
behaviour is unchanged (no class caches listed). The statement documents both.

## Timeouts and budgets

| place | before | after |
| --- | --- | --- |
| evaluator `OUTER_TIMEOUT_SECONDS` | 9000 | 12600 |
| `judge/config.yaml` `runtime.timeout_seconds` | 9000 | 12600 |
| `judge/config.yaml` verifier `max_instance_seconds` | 9000 | 12600 |
| `policy.json` `quota.eval_timeout_seconds` | 9000 | 12600 |
| Verifier images `FRONTIER_VERIFIER_GPU_SECONDS` | 9000 | 12600 |
| Verifier `max_gpu_seconds` (policy and config) | 936000 | 1310400 |

The GPU budget formula was verified against the old values:
`(25 train calls + 1 final) x 4 GPUs x timeout` = 26 x 4 x 9000 = 936,000, so
26 x 4 x 12,600 = 1,310,400. Generation (3,600 s), worker (3,900 s) and build
(600 s) limits are unchanged; the static final bound is 600 + 3 x 3,900 =
12,300 s <= 12,600 s. `max_train_wall_seconds` (225,000 = 25 x 9,000) and the
Agent quotas are unchanged (see open items). The verifier transport now requires
12 final workloads (`FINAL_WORKLOADS`), and operator pair/generation records allow
up to 12 rows.

## Dense-target manifest (schema 2)

`judge/dense_targets/manifest.json` now has `schema_version: 2`,
`parallel_gpu_workers: 4`, a `models` map (entrypoint and frame count per model),
and per-workload records with `model`, `prompt_id`, `seed` and `status`. Wan
records are `ready` with their existing dense/reference files and latencies.
Before generation, HunyuanVideo records were explicit placeholders
`{"model": "hunyuan", "prompt_id": p, "seed": s, "status": "PENDING"}`.
The merged records are now `ready` and bind the generated files and latencies.

`evaluate()` checks every record of the selected suite before patch application
or any GPU work. A missing, mismatched or `PENDING` record raises
`SuiteTargetsUnavailable` (deliberately not a `RuntimeError`), which the verifier
reports as `evaluator_error`, not as a scored 0 or a final verdict. This holds for
both development and final when target readiness is missing.

## Generating the HunyuanVideo targets

Script: `scripts/maintenance/sparse_video_attention_policy/generate_hunyuan_targets.py`
(not run yet). It imports the task evaluator and calls its own
`_run_generation` (same entrypoint, shim, warm-up, rebuild, synchronized timing
and decode check), so the protocol is the candidate's protocol. It applies
`reference.patch` once, builds the CUDA extension once, and runs every wave that
contains a HunyuanVideo workload as one four-GPU wave instance per kind: dense,
reference, and `R` reference replays. Each instance runs exactly four
concurrent processes on one four-H100 node (Wan slots of the mixed development
wave run the same kind as fillers; their latencies become a drift check against
the existing Wan manifest). High parallelism comes from more four-H100 nodes
(shards), never from more processes per node. With `R = 3` there are 10 wave
instances: 40 H100s finish in one round.

Where: the Wan targets were prepared on the Driver's Modal H100 pool (four-H100
Modal sandboxes, profile `frontier-cs-2`). The launcher
`modal_generate_hunyuan_targets.py` follows the repository's Modal precedent
(`docs/evidence/beat-spec-v3-gpu-refresh-20261011/modal_common.py`): it starts from
the task's published verifier-gpu image
`docker.io/joyemang33/fcs2-task-verifier-sparse_video_attention_policy-gpu@sha256:81a52c7461a57e8d0edc075332fe9e2be408e9e3bf94bf73ad564983d3688dcf`
(same Python/PyTorch/Diffusers/FlashInfer stack and Wan checkpoint), adds the
same pinned HunyuanVideo snapshot layer as the Dockerfiles, and lays this
branch's `judge/` and public repo out as the verifier image does. Once a
verifier-gpu image is built from this branch, set `SVG_TARGET_IMAGE=<that image
by digest>` and `SVG_TARGET_IMAGE_HAS_HUNYUAN=1`.

From the repository root, on a machine with Modal access:

```sh
# 0. Plan (CPU only): shards, wave instances, rough GPU-seconds
python3 scripts/maintenance/sparse_video_attention_policy/generate_hunyuan_targets.py plan --gpus 40 --replays 3

# 1. Generate on 40 H100s = 10 Modal containers of H100:4, then calibrate on 1 H100
MODAL_PROFILE=frontier-cs-2 modal run \
  scripts/maintenance/sparse_video_attention_policy/modal_generate_hunyuan_targets.py \
  --gpus 40 --replays 3 --run svg-hunyuan-targets-$(date -u +%Y%m%d)

# 2. Fetch shard outputs (videos, per-process records, logs, calibration.json)
modal volume get svg-hunyuan-targets svg-hunyuan-targets-<date> \
  <reports>/svg-hunyuan-targets-<date>

# 3. Review calibration.json, then merge (dry run first, then --write)
R=<reports>/svg-hunyuan-targets-<date>/svg-hunyuan-targets-<date>
python3 scripts/maintenance/sparse_video_attention_policy/generate_hunyuan_targets.py merge --inputs $R/shard-*
python3 scripts/maintenance/sparse_video_attention_policy/generate_hunyuan_targets.py merge --inputs $R/shard-* \
  --write --report docs/task-authoring/sparse_video_attention_policy/hunyuan/targets-<date>.json
```

The release run (release rule 3 below) uses the minimal pack instead:

```sh
MODAL_PROFILE=frontier-cs-2 modal run --detach \
  scripts/maintenance/sparse_video_attention_policy/modal_generate_hunyuan_targets.py \
  --gpus 12 --replays 0 --pack hunyuan-only --run svg-hunyuan-targets-20261015
```

`--gpus N` must be a multiple of 4 (`N / 4` shards). Any `N >= 4` works; fewer
nodes run their instances sequentially. Without Modal, run the same `run`
subcommand inside the verifier-gpu image on each four-H100 machine:
`python3 -B generate_hunyuan_targets.py run --task-root /opt/frontier/task --output <dir> --shard i --num-shards M --replays 3`,
then `calibrate --judge /opt/frontier/task --inputs <dirs> --output calibration.json`
on one GPU.

`merge` refuses records produced by a different `evaluator.py` or
`reference.patch`, shards that did not run on exactly four GPUs, failed slots,
hash mismatches, or anything other than exactly one dense and one reference
record per HunyuanVideo workload. With `--write` it copies
`hunyuan-{feedback,graded}-<i>-{dense,reference}.mp4` into
`judge/dense_targets/` and flips the six records to `ready`.

`calibrate` decodes every reference and replay video, computes fidelity against
its dense video with the evaluator's own `_fidelity`, and applies
`_beat_decisions` with each replay as the candidate. Release requires, per the
existing calibration rule: the largest apparent replay speedup plus one
percentage point stays within the 4% latency margin, no replay passes the
latency gate, and every replay is within the PSNR/SSIM/LPIPS bands. If the
margin does not hold, the release stops; the threshold is not changed to fit.

After merging: re-run the task tests, regenerate `upstream.lock.json` and the
task definition, build and publish the four images, and update the catalog
entry (see open items).

## Cost estimate

Planning numbers (to be replaced by measurements): Wan dense ~2,300 s and EAR
~1,500 s per process including load and warm-up (from the existing manifest);
HunyuanVideo at 81 frames is estimated at ~1,500 s dense and ~1,000 s EAR (one
transformer pass per step; ~0.5x the 129-frame attention cost).

- Target generation, `R = 3`: 10 wave instances x 4 GPUs, about 55,000
  H100-seconds (~15 H100-hours) plus image pulls; wall time about 40 minutes on
  40 H100s. Calibration: under 1 H100-hour.
- Per final evaluation: one extra four-GPU wave, about +4 x 1,300 s (~1.5
  H100-hours) and +20-25 minutes wall time.
- Per development evaluation: unchanged wave count; two Wan clips are replaced by
  shorter HunyuanVideo clips.
- Images: verifier-gpu and agent-gpu each grow by 41,903,727,509 bytes
  (HunyuanVideo Diffusers subset).

## License and distribution

`tencent/HunyuanVideo` is under the Tencent Hunyuan Community License Agreement
(Hub `license_name: tencent-hunyuan-community`). `hunyuan/LICENSE` and
`hunyuan/Notice` are the files from revision
`2a15b5574ee77888e51ae6f593b2ceed8ce813e5` (git blob ids match the Hub tree).
Key terms: the license does not apply in, and use/distribution is not licensed
for, the European Union, United Kingdom and South Korea; redistribution must
include the Agreement and the "Notice" file; separate terms above 100 million
MAU. The image snapshot layers include `LICENSE` and `Notice` in
`/opt/frontier/models/HunyuanVideo/`, and `model-assets.json` lists every file
with its size, git blob id and (for LFS files) sha256.

## Image changes

- `verifier-gpu/Dockerfile` and `agent-gpu/base.Dockerfile`: two pinned
  `snapshot_download` layers (all non-transformer files; transformer shards)
  after the Wan layers, read-only like Wan.
- `agent-gpu/Dockerfile` builds on the pinned published Agent GPU image, which
  predates HunyuanVideo; it adds the same two layers on top (reopening only
  `/opt/frontier/models` for the write).
- No new Python packages.

## Verification performed (CPU only)

- `task_tests/sparse_video_attention_policy`: new `test_hunyuan_workloads.py`
  (workloads, seeds, waves, shims, timeouts, manifest fail-closed, allowed paths,
  reference/naive patch application to both stubs, helper, adapters, assets,
  generator plan) plus the final-transport test updated to 12 workloads.
- `import_task.validate_export` and `validate_task_layout.check_inventory` pass
  for the task; `config/task-definitions.json` re-rendered for this task only.
- Expected failures until images exist: every test that loads
  `config/task-catalog.json` (the catalog still pins the old images and the
  image-tag/source interlock rejects them).

## Generated HunyuanVideo targets (2026-10-15)

Modal app `ap-MgoW6gmwTzShZibwworqsl` (profile `frontier-cs-2`), image = the
published verifier-gpu image `@sha256:81a52c74…` + pinned HunyuanVideo layer +
this branch's judge/repo; three `H100:4` containers, one process per GPU, four
concurrent per container, `--pack hunyuan-only --replays 0`; app wall time
1,012 s, all 12 processes succeeded. Merge report:
`hunyuan/targets-20261015.json`. Two earlier launches
(`ap-4TzPhptj6sSs34kXNWXe9U`, `ap-9CqWw1AN3tSohwWiPoqPd7`) failed at module
import (repository-relative paths inside the container) before any generation
and were stopped; both scripts were fixed.

| workload | prompt/seed | dense s | reference s | ref/dense | dense speedup |
| --- | --- | --- | --- | --- | --- |
| hunyuan feedback_0 | 3 / 104779 | 797.58 | 445.95 | 0.559 | 1.79x |
| hunyuan feedback_1 | 4 / 104789 | 803.25 | 444.25 | 0.553 | 1.81x |
| hunyuan graded_0 | 1 / 130457 | 800.79 | 443.34 | 0.554 | 1.81x |
| hunyuan graded_1 | 2 / 130469 | 801.88 | 438.32 | 0.547 | 1.83x |
| hunyuan graded_2 | 3 / 130477 | 811.28 | 446.09 | 0.550 | 1.82x |
| hunyuan graded_3 | 4 / 130483 | 795.57 | 438.88 | 0.552 | 1.81x |

Timed 50-step calls are about 800 s dense and 440 s EAR, faster than the
planning estimates above (whole processes about 950 s and 670 s).

## Release rules (2026-10-15)

1. Agent answering time (18,000 s), `max_train_wall_seconds` (225,000) and the
   Agent/Verifier GPU quotas other than the scaled final budget stay
   **unchanged**, although candidates now edit a second attention file.
2. The Tencent Hunyuan Community License territory exclusion (EU, UK, South
   Korea) is **accepted**. Images are published publicly on Docker Hub with
   `LICENSE` and `Notice` included in `/opt/frontier/models/HunyuanVideo/`.
3. Targets are generated **once per video, one GPU per video**, packed four
   processes per four-H100 container to match evaluation concurrency
   (`generate_hunyuan_targets.py --pack hunyuan-only --replays 0`: graded dense,
   graded reference, feedback dense+reference), with **no replay calibration**
   and no Wan companion slots — the same single-generation standard as the Wan
   targets. The 4% latency margin is the existing Wan-derived value.

## Open items

- Build/publish images; update the catalog entry to the new source tag and
  digests; regenerate the golden publication fixture.
- The task `version` string in `policy.json` is unchanged.
