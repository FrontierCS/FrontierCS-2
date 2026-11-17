"""Public development check for compaction/solver_entry.py (one timed run per article).

Runs the same steps as the evaluator's worker (prefill, then your compact()),
rebuilds the cache, and answers the public article's questions with the fixed
decoding code. It handles plain bf16/fp16/fp32 tensors; quantized returns are
checked only by the evaluator. The evaluator also runs a warmup, takes the
median of three fresh processes and seeds each run, so expect more noise here.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from evaluation.datasets import load_quality_data  # noqa: E402
from evaluation.utils import (  # noqa: E402
    extract_full_kv_cache, format_question, initialize_vllm, load_model_and_tokenizer, parse_model_choice)
from models.generate import generate_with_compacted_cache_batch  # noqa: E402
import compaction.solver_entry as entry  # noqa: E402

MODEL = os.path.join(os.environ.get("FRONTIER_MODEL_ROOT", "/opt/frontier/models"), "Qwen3-4B")


def target_tokens(start, stop, total):
    article = stop - start
    return min(article, max(1, int(article * entry.TARGET_SIZE))) + (total - article)


def main():
    wanted = set(sys.argv[1:])
    model, tokenizer = load_model_and_tokenizer(MODEL, device="cuda")
    engine = initialize_vllm(MODEL)
    state = entry.setup(model, tokenizer, engine)
    for article in load_quality_data("data/QuALITY.v1.0.1.htmlstripped.dev"):
        identifier = article["article_id"].removeprefix("quality_")
        if wanted and identifier not in wanted:
            continue
        torch.manual_seed(0)
        seq_len, cache, span, text, _ = extract_full_kv_cache(model, tokenizer, article["article"], "cuda",
                                                              model_name=MODEL)
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        request = {"model": model, "tokenizer": tokenizer, "engine": engine, "input_ids": input_ids,
                   "text": text, "article_span": (span.start, span.stop),
                   "cache": [(cache[i][0], cache[i][1]) for i in range(len(cache))], "past_key_values": cache,
                   "target_tokens": target_tokens(span.start, span.stop, seq_len), "seed": 0}
        torch.cuda.synchronize()
        started = time.perf_counter()
        layers = entry.compact(state, request)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        compressed, stored = [], 0
        for layer in layers:
            keys, values = layer["keys"].to(torch.bfloat16), layer["values"].to(torch.bfloat16)
            bias = layer.get("attention_bias")
            stored += layer["keys"].numel() * layer["keys"].element_size() * 2
            if bias is None:
                bias = torch.zeros(keys.shape[:3], dtype=torch.float32, device=keys.device)
            else:
                stored += bias.numel() * 4
            compressed.append((keys, bias, values))
        prompts = [format_question(tokenizer, q["question"], q["options"], MODEL) for q in article["questions"]]
        answers = generate_with_compacted_cache_batch(model=model, tokenizer=tokenizer, prompts=prompts,
                                                      compacted_cache=tuple(compressed), max_new_tokens=2048,
                                                      original_seq_len=seq_len)
        correct = sum(parse_model_choice(a) == q["gold_label"] for a, q in zip(answers, article["questions"]))
        print(f"{identifier}: correct {correct}/{len(prompts)} compact {seconds:.2f}s bytes {stored}", flush=True)


if __name__ == "__main__":
    main()
