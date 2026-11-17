"""Private reference (judge only): the paper's AM-HighestAttnKeys, on-policy.

Implements the task's solver interface (setup/compact) on top of the unchanged
upstream pipeline, with exactly the configuration and call the v1 calibration
used: RMS key scores, NNLS bias (2 iterations, bounds e^-3..e^3), least-squares
values, ss-plus-repeat self-study queries, optimized_agnostic Qwen3-4B head
budgets, max ratio per head 0.75. The worker's timed span is unchanged: prefill
(by the worker) + this compact() + handoff.
"""
import math

TARGET_SIZE = 0.1


def setup(model, tokenizer, engine):
    from compaction.compaction_methods.registry import get_compaction_method
    from evaluation.configs.utils import load_query_config

    method = get_compaction_method("highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy", {
        "algorithm": "highest_attention_keys",
        "score_method": "rms",
        "nnls_iters": 2,
        "nnls_lower_bound": math.exp(-3),
        "nnls_upper_bound": math.exp(3),
        "c2_method": "lsq",
        "on_policy": True,
        "precomputed_budget_path": "head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json",
        "max_ratio_per_head": 0.75,
    })
    return method, load_query_config("ss-plus-repeat")


def compact(state, request):
    method, query_config = state
    compacted, _stats = method.compact_kv_cache(
        past_key_values=request["past_key_values"], target_size=request["target_tokens"],
        indices=range(*request["article_span"]), query_config=query_config, model=request["model"],
        tokenizer=request["tokenizer"], formatted_context=request["text"], compute_stats=False,
        verbose_logging=False, vllm_model=request["engine"], sliding_layer_indices=set())
    return [{"keys": keys, "attention_bias": bias, "values": values} for keys, bias, values in compacted]
