# KV-cache compaction workspace

- `compaction/solver_entry.py`: your method (`setup` and `compact`). Its
  docstring documents the request and the returned layers. Everything under
  `compaction/` is yours.
- `models/`: fixed Qwen3 model and decoding code. It answers questions from a
  compressed cache of per-layer (keys, attention bias, values) tensors.
- `evaluation/utils.py`: fixed helpers. These are model loading
  (`load_model_and_tokenizer`), the vLLM engine (`initialize_vllm`), prompt
  formatting (`format_context`, `format_question`), the prefill used by the
  evaluator (`extract_full_kv_cache`) and answer parsing (`parse_model_choice`).
- `evaluation/datasets.py`: the QuALITY loader. `data/` holds the 12 public
  development articles.
- `frontier-bench.sh [article ids...]`: a public development check on a GPU
  runner.

See `FRONTIER_CHANGES.md` for what differs from the original sources. The
license is in `LICENSE`.
