# Multi-turn Trajectory Batches

Supply the trajectory construction of RAGEN's multi-turn PPO path so that the language-model
agent's real interactions with SimpleSokoban become a correct veRL training batch, and make the
resulting rollouts solve more puzzles and collect more reward. A valid submission keeps every
model and environment fact genuine and passes every structural check below. A beats-reference
criterion is evaluated alongside the score (see Scoring).

## Task

RAGEN ("RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement
Learning") runs language-model agents in interactive environments and feeds completed
interactions to PPO. It reads interleaved model responses and genuine SimpleSokoban observations,
actions and rewards from groups of trajectories that start at the same procedural puzzle state,
and produces a veRL `DataProto` whose token sequences, shifted responses, masks, group identities
and token-level rewards must describe those interactions exactly. The trajectory construction and
model/environment exchange in `ragen/llm_agent/ctx_manager.py` have been removed.

## What you may change

- `ragen/llm_agent/ctx_manager.py`
- new or changed modules below `ragen/llm_agent/solver_extensions/`, imported from it.

At most 20 files may change; everything else is fixed. The required public functions and methods
below must keep their signatures in `ctx_manager.py`; removing or renaming one is rejected as
`out_of_scope`. Internal data structures are unrestricted and public methods may delegate to the
helper modules.

## Interface

In `ctx_manager.py`:
- `get_masks_and_scores(input_ids, tokenizer, all_scores=None, use_turn_scores=False)` returns
  `(loss_mask, score_tensor, response_mask)` aligned with next-token responses. With terminal-only
  scoring, a two-turn row with rewards `[0.0, 10.8]` gets a score row summing to `10.8`, at its
  final response-token position.
- `ContextManager._parse_response(response)` returns `(canonical_response, actions)`. With
  separator `||` and a two-action limit, `<think>move the box</think><answer>Right || Up || Left</answer>`
  yields `['Right', 'Up']` and canonical text with only those actions.
- `ContextManager._normalize_score_tensor(score_tensor, env_outputs)` applies the configured
  grouping, penalty and normalization, preserving shape.
- `ContextManager.get_lm_inputs(env_outputs, prepare_for_update)` returns a generation batch
  (flag false) or a complete PPO training batch (flag true): for four histories padded to 265
  tokens, `input_ids`, `attention_mask`, `position_ids` are `(4, 265)` and `responses`,
  `loss_mask`, `rm_scores`, `original_rm_scores` are `(4, 264)`.
- `ContextManager.get_env_inputs(lm_outputs)` decodes one response per `env_id` into dicts with
  `env_id`, `llm_raw_response`, `llm_response` and `actions`.
- `ContextManager.formulate_rollouts(env_outputs)` returns the training `DataProto`, preserving
  every row's `env_id` and `group_id`.

The read-only `ragen.llm_agent.rollout_contract` (`RolloutContract(config, tokenizer,
REGISTERED_ENV_CONFIGS, mode)`, its `render` and `parse_response`) specifies the format; use it or
equivalent code.

**Provenance.** Each executed action comes from the actual model text of its turn (fixed answer
format, separator, vocabulary, per-turn limit); `llm_raw_response` is the actual generation with
its `<answer>`/`<think>` prefix, and `llm_response` and `actions` follow the published parsing;
malformed output keeps its text with no actions; unknown actions keep the format penalty;
termination and the action budget may shorten the executed prefix. Training tokens match the full actual history; packing and left padding may differ.

## Correctness

Score 0:
- `ctx_manager.py` does not compile, or a required public method is still the original stub.
- A source preflight (never executing submitted code) proves a syntax error or an unresolved
  eager annotation in an editable file.
- A provenance mismatch of actions, prompts or assistant tokens.
- A DataProto check fails (see Notes).
- Your PPO process crashes or does not finish within 750 s (shared with the dependency probe;
  preflight and build share a separate 60 s).
- Initial states differ from the reference's, or a rollout metric is non-finite or out of range.

The compile/stub, preflight and provenance zeros come before the reference run; the DataProto,
crash/750 s, initial-state and metric zeros count once the exact reference run succeeded.
Reference, build or infrastructure failures and the 60 s deadline are verification failures
without a score.

## Workload

One run executes the retained `train.py` path for one PPO step on an NVIDIA A100 (vLLM generation,
genuine SimpleSokoban transitions, veRL log-probability and critic computation, advantages, actor
and critic updates) with the command below. A run has six rollout groups of four trajectories; the
four share one procedurally generated 6x6, one-box initial state; each trajectory allows at most
ten environment actions, three model turns and 96 generated tokens per turn. The model is the
offline Qwen2.5-0.5B-Instruct (revision `7ae557604adf67be50417f59c2c2f167def9a775`).

| suite | graded rollout groups |
|---|---|
| train | 3 (of a fresh 6-group run) |
| final | 3 (of its own fresh 6-group run) |

The suites share no puzzle, are stratified to the same distribution, and final seeds are
withheld; order and repetition may vary. The scored rollouts happen before the optimizer step,
so the score measures trajectory quality, not a trained policy.

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

## Scoring

With `s_w` the success fraction (genuine terminal success states) and `r_w` the average
undiscounted episodic reward (original unnormalized environment rewards) of graded group `w`:

```text
q_w   = clip((r_w + 1) / 12, 0, 1)
score = clip(10 + 60 * mean_w(s_w) + 30 * mean_w(q_w), 0, 100)
```

Every group has equal weight. Anchors: a valid batch with zero success and zero reward scores
12.5; full success with reward >= 11 scores 100; any validity zero scores 0 (the 10 points are
not partial credit).

**Beats the reference** when, on every graded group, both the success rate and the average
episodic reward are strictly higher than the withheld implementation's.

## Running locally

On a GPU runner: `bash frontier-build.sh` builds the tree, `bash frontier-bench.sh` runs the
one-step PPO command above (see `/frontier/task/README.md`).

## Environment

`hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.2` (PyTorch 2.6, vLLM 0.8.2), Python 3.10; one NVIDIA
A100 80GB; the model offline below `$FRONTIER_MODEL_ROOT`.

## Notes

DataProto checks: non-empty batch accepted by veRL PPO; `responses == input_ids[:, 1:]` with
`input_ids`, `attention_mask`, `position_ids` `(N, L)` and the rest `(N, L-1)`; finite values;
binary, left-padded masks; `position_ids` the running sum of `attention_mask` (1-based, or 0-based
clamped at 0); `input_ids` = `pad_token_id` where attention is 0; `loss_mask` 0 where
`attention_mask[:, :-1]` is 0 and set in every row; `rm_scores` and `original_rm_scores` zero
except the last column, `rm_scores` summing to genuine reward + format penalty (atol 1e-6),
`original_rm_scores` raw or penalty-adjusted; valid executed actions; `non_tensor_batch`
`env_ids` and `group_ids` in input order; 4 rows per group from one initial state. Provenance of prompts: they hold the fixed instructions, the real
observation, remaining actions and genuine prior responses and rewards (all prior turns or a
contiguous suffix); nothing invented or rewritten, `raw_prompt_ids` included. Workloads are
generated puzzles, not inputs to recognise.

