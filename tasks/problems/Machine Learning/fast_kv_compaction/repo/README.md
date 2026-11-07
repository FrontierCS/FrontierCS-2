# KV Cache Compaction

Compress a long document's KV cache for Qwen3-4B so that its multiple-choice questions are
answered from the compressed cache about as well as from a private reference compaction, while
compacting faster or into fewer bytes. A valid submission returns a well-formed,
finite cache for every article within the time budget.

## Task

A language model reading a long document keeps a KV cache for every token of it. Replace each
layer's keys and values for the document with a much smaller set, so that questions asked
afterwards can be answered from the compressed cache alone. The workload is QuALITY on
Qwen/Qwen3-4B in bf16 on one H100: each article is compressed once, and every question is answered from the compressed cache (thinking enabled, at
most 2048 new tokens, the official answer parser). The starting point keeps the chat template and
a uniformly random subset of article tokens per head, with the original keys and values and no
bias.

## What you may change

- everything under `compaction/` (new files may be added and files deleted); the method is
  `compaction/solver_entry.py` (`setup`, `compact`, `TARGET_SIZE`).

Everything outside `compaction/` is fixed: `models/` (the Qwen3 model and decoding code),
`evaluation/` (`utils.py`: model loading, the vLLM engine, prompt formatting, the evaluator's
prefill `extract_full_kv_cache`, answer parsing; `datasets.py`: the QuALITY loader) and `data/`.
Your method runs as an unprivileged user with no access to the evaluator, the hidden articles or
the questions; it receives the article and the budget and returns tensors only. Do not
special-case articles (by text, length, id or position), the seed, or train vs final; the method
must be general.

## Interface

`compaction/solver_entry.py` runs in its own worker process with Qwen3-4B (Hugging Face, bf16, on
the GPU), its tokenizer and a vLLM engine for the same weights; its docstring lists every field.

```python
def setup(model, tokenizer, engine) -> state      # once, before any article, not timed;
                                                  # engine is asleep: wake_up() / sleep()
def compact(state, request) -> list               # once per article, timed
```

`request` holds `model`, `tokenizer`, `engine`; `input_ids` `(1, S)` and `text` (chat template
plus article; questions come later); `article_span` `(start, stop)`; `cache`, 36 layers of
`(keys, values)` each `(1, 8, S, 128)` bf16 on the GPU, keys already rotary-embedded;
`past_key_values` (the same as a Hugging Face cache); `target_tokens` (tokens per head: every
template token plus `TARGET_SIZE` of the article tokens; advisory, the judge checks only `t ≤ S`); `seed` (already applied).

Return one dict per layer: `keys` and `values` `(1, 8, t, 128)` and an optional `attention_bias`
`(1, 8, t)`, an additive term on the attention logits of each kept key (absent: zeros; `-inf`
marks an unused slot). Heads of a layer share `t`; layers may differ. Tensors must be
plain `torch.Tensor` (batch 1; keys and values of a layer the same shape and dtype: bf16, fp16 or
fp32), and a layer dict may hold only `keys`, `values` and `attention_bias`. Questions are appended after position `S`
and answered from the rebuilt cache alone. Bytes count as returned: `t = 100` per head in bf16 is
2 × 8 × 100 × 128 × 2 = 409,600 bytes; an fp32 bias adds 3,200.

`TARGET_SIZE` must be a literal number in `(0, 0.1]`: `0.1` is **time mode** (compete on time at
the reference's budget), anything smaller **bytes mode** (compete on bytes).

## Correctness

Each runner handles 3 articles in a lease of 3,600 s (train) or 4,500 s (final), counted from
the runner request. After start-up, a 425-s margin for returning the result, trusted setup and
150 s of QA per article and repetition, the rest is the method's: every worker start (with `setup`) and every `compact` call,
warmups included, at most 1,800 s each. The score is 0 when a limit is exceeded, `TARGET_SIZE` is
missing, not a literal or outside `(0, 0.1]`, or, on any article, `compact` raises, the worker
exceeds 48 GiB of host memory or 8 GiB of disk (its home, temporary and working directories) or
writes more than 256 MiB to stderr, or the result has the wrong number of layers or head layout (8
heads, head size 128), keeps more tokens than the prompt has or has non-finite keys or values.
A runner failure, a lease too short for setup and QA, a GPU-model mismatch or missing targets is
an unscored failed verification.

## Workload

One workload is one QuALITY article (a long Project Gutenberg story with 13–20 multiple-choice
questions). `data/` holds the 12 public articles.

| suite | articles | repetitions | single-H100 runners |
|---|---|---|---|
| train | 12 public | 1 timed (after an untimed warmup), QA once | 4 |
| final | 12 public + 12 hidden (disjoint, same QuALITY set) | 3, QA on every repetition | 8 |

Articles are spread over runners that run at the same time (3 per runner), as during
calibration. Per article: **qa_correct**, the questions answered correctly (final: the lowest of
the three repetitions); **compaction_seconds**, the wall time from handing the article to your
process to receiving its result, including that process's prefill and the transfer (final: the
median of 3 fresh processes, each compressing a fixed warmup text first); **compact_cache_bytes**,
the bytes you return over all layers (final: the largest over the 3 repetitions). Time targets follow the evaluation machine's measured speed
relative to calibration.

## Scoring

```text
score = 100 * mean over articles of (correct / questions) * (0.75 + 0.25 * e)
e     = clip(0.5 + log10(r) / 6, 0, 1)
r     = (reference / yours on the mode's axis) * min(1, reference / yours on the other axis)
```

The mode's axis is seconds in time mode and bytes in bytes mode; reference values are the medians
of several seeded reference runs on the same hardware. Anchors: `e = 0.5` at parity, so matching
the reference scores 87.5 × its accuracy; `e = 1` at 1000× better; the maximum is 100 (full
accuracy, `e = 1`).

**Beats the reference** requires, on every graded article: `qa_correct` at least the reference's
median correct count minus max(1, ceil(5% of its questions)); and in time mode `compaction_seconds`
strictly below the reference's median with `compact_cache_bytes` no larger, or in bytes mode
`compact_cache_bytes` strictly below the reference's with compaction no slower than its median
time. There is no further margin; matching the reference does not pass. It does not change the
score.

## Running locally

On a GPU runner (prefix with `/opt/frontier/task-runtime`, see `/frontier/task/README.md`):
`bash /workspace/repo/frontier-bench.sh [article ids ...]` compacts and answers public articles
with your `solver_entry.py` and prints correct answers, compaction seconds and cache bytes per
article (one timed run, noisier than final's median of three).

## Environment

Python 3.12 venv from a hash-locked closure (torch 2.8.0, vLLM 0.11.0, transformers 4.57.1,
accelerate 1.12.0, datasets 4.4.1) on CUDA 12.8.1; Qwen/Qwen3-4B pinned by revision, offline.
`FRONTIER_CHANGES.md` lists what differs from the original sources; the license is in `LICENSE`.
