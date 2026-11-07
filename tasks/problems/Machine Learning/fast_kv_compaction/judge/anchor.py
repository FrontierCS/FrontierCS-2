#!/usr/bin/env python3
"""Trusted machine-speed anchor for the vLLM engine path (the self-study path).

Run by the harness as a short-lived process of its own (so its engine memory is
released before any worker starts): it initializes vLLM through the pristine
tree's initialize_vllm, wakes it as self-study does, and prints the median
seconds of ENGINE_REPEATS fixed greedy generations of ENGINE_TOKENS tokens.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time

ENGINE_REPEATS = 3
ENGINE_TOKENS = 256
PROMPT = ("Summarize, in plain words, how a water mill turns the force of a river into "
          "the motion of a grinding stone, step by step.")


def main() -> int:
    pristine = Path(sys.argv[1])
    os.chdir(pristine)
    sys.path.insert(0, str(pristine))
    from vllm import SamplingParams
    from evaluation.utils import initialize_vllm
    model = str(Path(os.environ.get("FRONTIER_MODEL_ROOT", "/opt/frontier/models")) / "Qwen3-4B")
    engine = initialize_vllm(model)
    engine.wake_up()
    params = SamplingParams(max_tokens=ENGINE_TOKENS, min_tokens=ENGINE_TOKENS, temperature=0.0, seed=0)
    engine.generate([PROMPT], params, use_tqdm=False)  # warmup
    times = []
    for _ in range(ENGINE_REPEATS):
        engine.reset_prefix_cache()
        started = time.perf_counter()
        engine.generate([PROMPT], params, use_tqdm=False)
        times.append(time.perf_counter() - started)
    print(json.dumps({"engine_anchor_seconds": statistics.median(times)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
