#!/usr/bin/env python3
"""Untrusted side: runs the submitted compaction method in its own process.

Started by the harness through /opt/frontier/fastkv-worker as the unprivileged
fastkv-worker user, with a scrubbed environment, a private empty HOME, and a
copy of the patched public tree without its data directory. It cannot read the
judge, the hidden articles, the targets or the harness's memory, and it does not
hold the model that answers the questions: it loads its own Qwen3-4B (and vLLM
for query generation) and returns only tensors.

Protocol (wire.py frames on the inherited stdin/stdout pipes):
  -> {"kind": "ready"} once the model is loaded and solver_entry.setup() returned
  <- {"kind": "compact", "article": text, "target_size": int, "seed": int}
  -> {"kind": "cache", "layers": L, "has_bias": [bool] * L, "seq_len": S, "tensors": [...]}
     + raw bytes (per layer: keys, values, then attention_bias if present), or
     {"kind": "error", "message": ...}
  <- {"kind": "exit"}
Each worker process serves one warmup and then each article at most once, so a
result can never be replayed from an earlier identical request.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wire  # noqa: E402

MODEL = "/opt/frontier/models/Qwen3-4B"


def _channel():
    # Keep the protocol pipe private; everything the method prints goes to stderr.
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return sys.stdin.buffer, out


def _seed(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tensor_bytes(tensor) -> tuple[dict, bytes]:
    import torch
    if type(tensor) is not torch.Tensor:
        raise TypeError("returned tensors must be plain torch tensors")
    name = str(tensor.dtype).removeprefix("torch.")
    if name not in wire.DTYPES:
        raise TypeError(f"returned tensor dtype {name} is not allowed")
    flat = tensor.detach().contiguous().cpu()
    return {"dtype": name, "shape": list(flat.shape)}, flat.view(torch.uint8).numpy().tobytes()


def _layer_tensors(layer):
    """(keys, values) or (keys, values, attention_bias) of one returned layer."""
    if not isinstance(layer, dict) or not {"keys", "values"} <= set(layer) <= {"keys", "values", "attention_bias"}:
        raise TypeError("each layer must be a dict with keys, values and optional attention_bias")
    if layer.get("attention_bias") is None:
        return layer["keys"], layer["values"]
    return layer["keys"], layer["values"], layer["attention_bias"]


def main() -> int:
    tree = Path(sys.argv[1]).resolve()
    inbox, outbox = _channel()
    os.chdir(tree)
    sys.path.insert(0, str(tree))
    import torch
    from evaluation.utils import extract_full_kv_cache, initialize_vllm, load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL, device="cuda")
    vllm_model = initialize_vllm(MODEL)
    import compaction.solver_entry as entry
    state = entry.setup(model, tokenizer, vllm_model)
    wire.write_frame(outbox, {"kind": "ready"})
    while True:
        request, _ = wire.read_frame(inbox)
        if request["kind"] == "exit":
            return 0
        try:
            _seed(int(request["seed"]))
            seq_len, cache, indices, context, _ = extract_full_kv_cache(
                model, tokenizer, request["article"], "cuda", model_name=MODEL)
            input_ids = tokenizer(context, return_tensors="pt", add_special_tokens=False)["input_ids"]
            layers = entry.compact(state, {
                "model": model, "tokenizer": tokenizer, "engine": vllm_model,
                "input_ids": input_ids, "text": context, "article_span": (indices.start, indices.stop),
                "cache": [(cache[index][0], cache[index][1]) for index in range(len(cache))],
                "past_key_values": cache, "target_tokens": int(request["target_size"]),
                "seed": int(request["seed"])})
            torch.cuda.synchronize()
            specs, payloads, has_bias = [], [], []
            for layer in layers:
                tensors = _layer_tensors(layer)
                has_bias.append(len(tensors) == 3)
                for tensor in tensors:
                    spec, payload = _tensor_bytes(tensor)
                    specs.append(spec)
                    payloads.append(payload)
            wire.write_frame(outbox, {"kind": "cache", "layers": len(layers), "has_bias": has_bias,
                                      "seq_len": int(seq_len), "tensors": specs}, payloads)
        except Exception as exc:  # the submitted method failed on this article
            traceback.print_exc()
            wire.write_frame(outbox, {"kind": "error", "message": f"{type(exc).__name__}: {exc}"[:300]})


if __name__ == "__main__":
    raise SystemExit(main())
