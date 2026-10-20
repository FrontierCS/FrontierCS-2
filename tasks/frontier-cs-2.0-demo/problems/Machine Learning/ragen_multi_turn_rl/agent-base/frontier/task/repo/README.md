# Multi-turn Trajectory Batches

## Problem
RAGEN runs language-model agents inside interactive environments and feeds completed interactions to PPO training. It reads interleaved model responses and genuine SimpleSokoban observations, actions, and rewards from groups of trajectories that begin at the same procedural puzzle state. It produces a veRL `DataProto` whose token sequences, shifted responses, masks, group identities, and token-level rewards must describe those interactions exactly. Every completed trajectory must remain non-empty, every executed action must be valid, left padding and attention must agree, rewards must occupy the correct terminal token, and all tensor values must be finite. The objective rewards implementations whose resulting policy rollouts solve more puzzles and collect more undiscounted environment reward. The trajectory construction and model/environment exchange in `ragen/llm_agent/ctx_manager.py` have been removed and must be supplied.

## Design space
The design area includes truthful context selection, token packing, padding, action-token masks, reward placement and normalization, rollout-group bookkeeping, and `DataProto` assembly. Utility modules may be added below `ragen/llm_agent/solver_extensions/`. Public methods may delegate to those modules; internal data structures are unrestricted. Generation and completed training batches must satisfy the provenance contract below and remain consumable by unchanged veRL callers.

### Model and trajectory provenance

Each executed action must come from the actual model-generated text for that same turn. The fixed answer format, configured action separator, vocabulary and per-turn action limit apply. `llm_raw_response` must be the actual decoded generation with its fixed `<answer>` or `<think>` prefix; `llm_response` and `actions` must match the published normalization and parsing rules. Malformed output remains its original text with no parsed actions; unrecognized action names retain the existing environment format penalty. Environment termination and the remaining action budget may shorten the executed prefix.

Generation prompts contain the fixed system/environment instructions, the current real observation and remaining action count, and genuine prior normalized model responses and observed rewards. You may keep all prior turns or a contiguous suffix of completed turns, retaining original turn numbers and the current observation. Instructions, observations, responses and rewards may not be invented, rewritten or supplemented with candidate-computed answers. This applies to the actual model input, including optional `raw_prompt_ids`. This is a text-only workload.

The read-only `ragen.llm_agent.rollout_contract` module specifies the format. Construct `RolloutContract(config, tokenizer, REGISTERED_ENV_CONFIGS, mode)` with the registry from `ragen.env`. Its `render(env_outputs, training=False, history_windows=None)` returns `(texts, conversations)`; an optional mapping from `env_id` to number of retained prior completed turns selects a suffix. `training=True` renders the complete genuine normalized conversation without its final unresponded observation. `parse_response(raw, enable_think=..., action_sep=..., max_actions=...)` specifies canonical text and action parsing. You may use these helpers or implement equivalent behavior.

Training conversation tokens, including every assistant response, must match that complete actual history; packing and left padding may differ. Masking, reward alignment, normalization and assembly remain candidate responsibilities under the structural checks. Trusted code outside the editable surface captures real model generation and validates these boundaries. Candidate metadata does not establish provenance. An independently detected action, prompt or assistant-token mismatch is a scored structural rejection, with a fixed development diagnostic; final feedback remains private.

## Workload
Each run executes the retained `train.py` path for one PPO training step on an NVIDIA A100, including vLLM generation, genuine SimpleSokoban transitions, veRL log-probability and critic computation, advantage construction, and actor and critic updates. A run contains six rollout groups of four trajectories; all four trajectories in a group share one procedurally generated 6-by-6, one-box initial state. Each trajectory permits at most ten environment actions. The evaluator uses three model turns and 96 generated tokens per turn; these two reduced evaluation settings are challenge constants (not from materials). The offline model is Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775` installed in the runtime. Workloads are representative generated puzzles, not a fixed input table to recognise; graded execution may vary ordering, repetition, and scale.

## Interface
`get_masks_and_scores(input_ids, tokenizer, all_scores=None, use_turn_scores=False)` returns `(loss_mask, score_tensor, response_mask)`, aligned with next-token responses. For example, if a two-turn row has environment rewards `[0.0, 10.8]` and terminal-only scoring is configured, the returned score row sums to `10.8` and has that reward at its final response-token position.

`ContextManager._parse_response(response)` returns `(canonical_response, actions)`. For example, with separator `||` and a two-action limit, `<think>move the box</think><answer>Right || Up || Left</answer>` yields two actions, `['Right', 'Up']`, and canonical text containing only those accepted actions.

`ContextManager._normalize_score_tensor(score_tensor, env_outputs)` applies the configured grouping, penalty, and normalization contract while preserving tensor shape.

`ContextManager.get_lm_inputs(env_outputs, prepare_for_update)` returns a batch for generation when the flag is false and a complete PPO training batch when it is true. In the latter case a concrete batch with four histories and padded token length 265 has `input_ids`, `attention_mask`, and `position_ids` shaped `(4, 265)`, with `responses`, `loss_mask`, `rm_scores`, and `original_rm_scores` shaped `(4, 264)`.

`ContextManager.get_env_inputs(lm_outputs)` decodes one response per `env_id` and returns dictionaries containing `env_id`, `llm_raw_response`, `llm_response`, and `actions`.

`ContextManager.formulate_rollouts(env_outputs)` returns the final training `DataProto` and preserves every input row's `env_id` and `group_id`.

After resolving the baked model directory below `$FRONTIER_MODEL_ROOT`, the judge runs the retained entrypoint with this complete argument set:

```bash
python /workspace/repo/train.py \
  --config-name _2_sokoban \
  trainer.total_training_steps=1 \
  trainer.validation_steps=1 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  'trainer.logger=[console]' \
  trainer.generations_to_log_to_wandb.train=0 \
  trainer.generations_to_log_to_wandb.val=0 \
  es_manager.train.env_groups=6 \
  es_manager.train.group_size=4 \
  'es_manager.train.env_configs.n_groups=[6]' \
  actor_rollout_ref.rollout.response_length=96 \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.actor.use_ref=false \
  actor_rollout_ref.actor.use_torch_compile=false \
  ppo_mini_batch_size=4 \
  micro_batch_size_per_gpu=4 \
  agent_proxy.max_turn=3 \
  model_path="$FRONTIER_MODEL_ROOT/qwen2.5-0.5b-instruct"
```

## Validity
The patch must apply cleanly and remain within the editable surface. Missing signatures, remaining public stubs and independently established source-preflight failures score zero before rollout quality is considered. Preflight checks Python syntax in submitted editable Python files and may prove an unresolved eager annotation in a restricted declaration-only prefix. It never executes submitted code. Programs outside that proof subset continue to ordinary execution. The diagnostic preflight path also requires the fixed build and dependency checks to succeed. Existing signature, stub and context-syntax checks retain their earlier source-validation behavior. Except for independently detected provenance violations, unclassified dynamic Python/subprocess failures, fixed-runtime or reference failures, cancellation, explicit exits, signals, resource exhaustion and execution deadlines remain verification/control failures without a score. Trusted DataProto checks report a bounded contract reason for invalid candidate tensors, masks, rewards or identities; these candidate rejections score zero in Train and Final after a successful exact-reference control. Unclassified exceptions do not count as such a rejection. The generated training batch must be non-empty and accepted by downstream veRL PPO. Token shifts and tensor shapes must align exactly; masks must be finite and consistent with left padding; every row needs trainable tokens; training rewards must match genuine environment rewards plus the configured format penalty and appear at the terminal token; `original_rm_scores` may preserve raw rewards or the upstream penalty-adjusted representation; group identities and same-initial-state multiplicity must be preserved. Binary patches, more than 20 changed files, or patches larger than 262,144 bytes are rejected.

## Scoring
The scored rollouts occur before the optimizer calls. The single-step run checks
trajectory validity and downstream optimizer execution; it does not measure a
trained policy or require nonzero parameter changes. The fixed scheduler can
use a zero learning rate at this first optimizer step. Independent final grading
starts a fresh run on its own puzzle suite; it is not post-update evaluation.

For each evaluated group `w`, let `s_w` be its success fraction and `r_w` its average undiscounted episodic reward. Define `q_w = clip((r_w + 1) / 12, 0, 1)`. The score is

```
score = clip(10 + 60 * mean_w(s_w) + 30 * mean_w(q_w), 0, 100).
```

The additive 10 and the 60/30 weights and reward scaling are challenge constants (not from materials). Maximise the score; a scored validity rejection yields zero.

Every selected rollout group has equal weight, regardless of the number of actions produced inside its trajectories. `s_w` is computed only from genuine terminal SimpleSokoban success states, while `r_w` is computed from the original unnormalized rewards returned by environment steps. The clipping in `q_w` bounds unusually low or high episodic rewards before groups are averaged. The ten-point structural component is awarded only after all token, mask, reward, identity, and downstream-training checks pass; it is not partial credit for malformed batches. The returned bounded and unbounded scores are identical for this task.

## Calibration
The score formula assigns 12.5 to structurally valid zero-success, zero-environment-reward trajectories. This is an arithmetic anchor, not a measured reference or solver result. Genuine model rollout and PPO execution are required even at that anchor. Earlier versions have measured GPU reference and solver results; those results do not establish acceptance or difficulty under this version's stronger model-provenance contract. Version-specific reference remeasurement is required. Rollout quality and execution checks do not establish repeated-run variance, broad model difficulty, parameter learning, or the paper's self-evolution result. Version-bound results and limitations are recorded in the task's external authoring dossier. A source or scored-data validity rejection scores zero; a typed rejection by a trusted DataProto validity check scores zero after the exact reference execution succeeds. Unclassified dynamic structural or PPO failures remain unscored. Neither establishes task difficulty.

The upstream implementation adds format penalties in place, so its `original_rm_scores` field shares penalty-adjusted values with `rm_scores`. The judge accepts that exact upstream behavior or a separately retained raw-reward tensor; scored episodic reward always comes from a snapshot of the genuine environment history taken before trajectory assembly.

## Feedback and grading
The judge returns a score for completed evaluations, including zero for independently established source-preflight failures. Infrastructure/control failures do not produce a score. The submission budget is in `/frontier/task/README.md`. **The graded suite is not the suite you receive feedback on**, and neither its membership nor its seeds are disclosed. Development feedback reports only aggregate candidate-only mean `success_rate` and `average_episodic_reward` across development groups; aggregate pass/fail counts and percentages show progress on the stricter per-workload criterion without identifying which groups passed. The two suites are stratified so they carry the same distribution, and they share no item.

A source-preflight zero may include development feedback in `metrics.candidate_diagnostic`: category `syntax_error` or `unbound_eager_annotation`, the public submitted source path, a 1-based line number and fixed corrective guidance. The annotation category asks you to check names and imports in an eager annotation. It is a source analysis result, not an observed runtime exception. The analysis requires the unchanged public bootstrap and a supported sequence of inert declarations and fixed imports, including inspected solver helper imports. Dynamic statements, decorators, nonliteral defaults, ambiguous imports and deferred annotations cause the annotation analysis to abstain. The full function header must be supported before a missing-name diagnosis is emitted. Implicit or dunder annotation names and unmodeled type subscriptions cause abstention; only explicitly declared or imported helper attributes count as exports, not builtin lookup names. No exception text, traceback, private values or per-workload result is exposed. The zero does not assert that a corrected implementation will pass. Remaining development calls come from the submission tool. Final feedback never exposes these diagnostics or workload metrics.

A separate stricter per-workload criterion is evaluated in addition to the continuous score. On every graded fixed-initial-state SimpleSokoban rollout group, your GPU-executed implementation must achieve both a strictly higher trajectory success rate and a strictly higher average undiscounted episodic reward than the withheld implementation on that same group. The criterion passes only if both comparisons hold for every group; the withheld values and pairwise differences are not disclosed. This criterion does not replace the continuous objective, and a submission that does not clear every comparison is still scored normally.

## Resources
The control pod provides 3 vCPUs, 12,288 MiB memory, and 10,240 MiB workspace storage. Agent debugging allows one A100-80GB runner, at most 1,800 seconds per runner and 1,800 GPU-seconds in total. The Verifier uses one A100-80GB at a time, with a 7,200 GPU-second pool across development and final verification. Each verification has a 1,800-second evaluator timeout. The build timeout is 600 seconds.
