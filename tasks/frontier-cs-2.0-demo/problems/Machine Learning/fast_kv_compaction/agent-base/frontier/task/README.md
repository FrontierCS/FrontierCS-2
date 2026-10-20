# KV Cache Compaction

Saving has no save-count quota. Payload, retained-data, audit and Agent-time
limits still apply. The save receipt reports `remaining_checkpoint_submissions`
as `null` for this contract. Check the accepted receipt; saving does not
extend answering time or select an answer automatically.

## The problem

A language model that reads a long document keeps a KV cache for every token
of it. Your task is to compress that cache: replace each layer's keys and
values for the document with a much smaller set, so that questions asked
afterwards can be answered from the compressed cache alone, as well as from a
private reference compaction method, but faster or smaller.

The workload is QuALITY (long Project Gutenberg stories with 13 to 20
multiple-choice questions each) on Qwen/Qwen3-4B in bf16, one H100. Each
article is compressed once, and every question is answered from the compressed
cache (thinking enabled, at most 2048 new tokens, the official answer parser).
The tree at `/frontier/task/repo` holds the fixed model, decoding and prompt
code and, in `data/`, the 12 public development articles; final adds 12 hidden
articles.

## What you write

Everything under `compaction/` is yours. The evaluator runs
`compaction/solver_entry.py` in its own worker process, which owns a copy of
Qwen3-4B, its tokenizer and a vLLM engine for the same weights:

- `setup(model, tokenizer, engine)` is called once before any article (not
  timed) and returns any state you like.
- `compact(state, request)` is called once per article. The worker has already
  run the prefill, so `request` holds the prompt prefix (chat template +
  article) as `input_ids` and `text`, the article's token range
  `article_span`, the full prefilled cache `cache` (for each of the 36 layers,
  keys and values of shape (1, 8 KV heads, S tokens, 128), bf16, keys already
  rotary-embedded), the same cache as a Hugging Face cache object, the
  per-head token budget `target_tokens` and the run's `seed`.
- It returns, for each layer, a dict with `keys` and `values` of shape
  (1, 8, t, 128) and an optional `attention_bias` of shape (1, 8, t): an
  optional additive term on the attention logits of each kept key (absent
  means zeros; -inf marks an unused slot). All heads of a layer share t, and t
  may differ between layers. Questions are then appended after position S, and
  each one is answered from your returned cache alone.

`TARGET_SIZE` in that file is the fraction of article tokens to keep; it must
be a literal number in (0, 0.1]. `target_tokens` is every template token plus
that fraction of the article's tokens. The starting point keeps the template
and a uniformly random subset of article tokens per head, with the original
keys and values and no bias. The docstring of `solver_entry.py` lists every
field.

**Formats.** `keys`, `values` and `attention_bias` may be bf16, fp16 or fp32
tensors (keys and values of a layer in the same dtype). The evaluator checks
the exact layout, rebuilds the tensors itself, and decodes with the unchanged
model code.

**Bytes.** The bytes you return are counted exactly as stored, over all layers;
an absent `attention_bias` costs nothing. For example, a layer that keeps
t = 100 tokens in each of 8 heads holds 2 × 8 × 100 × 128 × 2 = 409,600 bytes
of bf16 keys and values. An fp32 `attention_bias` adds 8 × 100 × 4 = 3,200
bytes.

Everything outside `compaction/` (`evaluation/`, `models/`, the data) is fixed.
Your method runs as an unprivileged user without access to the evaluator, the
hidden articles or the questions. It receives the article and the budget, and
returns tensors only; a separate trusted process rebuilds the cache from them,
times the call and answers the questions.

## How you are measured

Per article w, three measurements of your method:

- **qa_correct**: questions answered correctly from your compressed cache. In
  final evaluation every repetition's cache is answered, and the lowest of the
  three counts.
- **compaction_seconds**: median over 3 repetitions of the wall time from
  handing the article to your method's process to receiving its result. Each
  repetition is a fresh process that first compresses a fixed warmup text and
  then sees every article once; the time includes that process's prefill of the
  article and the transfer of your result. Time targets follow the evaluation
  machine's measured speed relative to calibration. Articles are spread over
  several single-GPU machines that run at the same time, as during calibration.
- **compact_cache_bytes**: the bytes you return, as above, over all layers.

## Scoring and passing

Besides the continuous score, a separate beats_reference criterion compares your
method with a private reference compaction method on every graded article.
Every article must keep QA accuracy within max(1, ceil(5% of its questions)) of
the reference's median correct count, and must improve one axis: at
TARGET_SIZE = 0.1 (time mode) its compaction must be strictly faster than the
reference's median time with no more cache bytes; at TARGET_SIZE below 0.1
(bytes mode) its compacted prefix must use strictly fewer bytes with compaction
no slower than the reference's median time. Reference medians come from several
seeded reference runs on the same hardware; there is no further margin. A
method that matches the reference does not pass.

The continuous score is the mean over articles of 100·ln(S)/(ln 10 + ln S),
where S is your improvement ratio on the mode's axis (reference median seconds
over yours, or reference bytes over yours), and 0 for an article whose accuracy
or other-axis condition fails or where S <= 1. It is 50 at a 10x improvement and
stays below 100.

A development submission grades the 12 public articles with **one** timed
repetition per article (after one untimed warmup, one QA pass), so its timings
are a single, noisier sample than final's median of three. It returns
`dev_qa_accuracy` (your pooled fraction of correct answers),
`dev_compaction_seconds` and `dev_compact_cache_bytes` (your per-article values,
each list sorted independently, without article labels), and
`beat_spec_progress`: for each metric, the `passed`, `failed` and `total`
article counts with `pass_percent` and `fail_percent`. No reference values are
returned. Final grades all 24 articles and returns nothing to you.

## Optimization priority

First satisfy every required PASS condition on every article and metric. Optimize
continuous score second; do not sacrifice PASS for a higher score. Successful
execution or a positive score is not PASS. Use train pass/fail progress to guide
experiments; final evaluates independently.

## Workspace and submission

```sh
/frontier/task/bin/prepare_workspace.sh          # -> /workspace/repo
# edit /workspace/repo/compaction/...
/frontier/task/bin/make_submission.sh            # -> /workspace/payload/solution.patch
```

The helper stages tracked edits and new files, checks that every change is under
`compaction/` and at most 40 files changed, and writes the patch. Nothing outside
`/workspace` is writable. Read [the testbed instructions](bin/README.md) for
saving a selected final and resource limits.

The formal Agent answering budget is **5 hours (18000 seconds)**. Independent
final evaluation has a separate **5400-second** budget after answering ends.
Train waits and GPU development count within answering time.

Train and final use the same uncompressed POSIX tar: one top-level `payload/`
directory holding exactly `payload/solution.patch` (non-empty), at most 262,144
bytes in total, relative member paths without `..`, no links, devices or FIFOs.

```sh
cd /workspace/repo && /frontier/task/bin/make_submission.sh
tar -C /workspace --format=ustar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 -cf /workspace/train.tar payload
frontier-submit train /workspace/train.tar
```

`frontier-submit train` returns immediately after acceptance with a verification
ID and the authoritative count of remaining train calls. Use
`frontier-submit status <verification_id>` for one status read; once
verification completes it includes your score and the development `metrics`
above. **8 train calls are available for the whole run**, and every accepted
request consumes one even when verification fails. A development evaluation
has a 3600-second timeout. The cumulative train wall-time allowance is 28800 seconds.

Save your first usable chosen answer with `frontier-submit store-final
/workspace/final.tar` and confirm the accepted receipt while continuing to work;
a later accepted `store-final` replaces it. Train never selects a final for you.
When you are ready to finish, run

```sh
frontier-submit final /workspace/final.tar
```

exactly once, and exit. The final score is produced after you exit; the command
does not print it. If answering ends normally, times out, or the Agent crashes
without an accepted explicit final, the Driver collects the last accepted stored
archive. User cancellation and platform faults do not trigger this fallback.

## Using a GPU

Your container is a CPU editor/control environment. Use the Candidate-owned
`frontier-gpu` helper V3 (`frontier.gpu-helper/v3`); read its skill at
`/opt/frontier/candidate/skills/frontier-gpu/SKILL.md` first. Its executable is
`/opt/frontier/candidate/bin/frontier-gpu`. The CPU workspace and GPU runner do
not share files.

```bash
frontier-gpu create --type H100 --count 1 --seconds 18000
frontier-gpu status gpurunner_...
frontier-gpu push gpurunner_... /workspace/repo /workspace
frontier-gpu exec gpurunner_... -- /opt/frontier/task-runtime bash /workspace/repo/frontier-bench.sh 22073
frontier-gpu delete gpurunner_...
```

`frontier-bench.sh [article ids...]` compacts and answers public articles with
your `solver_entry.py` and prints correct answers, compaction seconds and cache
bytes per article (one timed run, so expect more noise than the evaluator's
median of three). Prefix runtime commands with `/opt/frontier/task-runtime` to
restore the offline model environment; `/opt/frontier/runtime-python` is the
image's Python.

You have **1024 SSH sessions for Agent GPU debugging**; every accepted `exec`,
`ssh`, `push` or `pull` consumes one. The Verifier has independent pools: **16
sessions across the 8 train calls** and **16 for final evaluation**. Each SSH
session allows at most 512 MiB (536870912 bytes) per direction. GPU budget: one
H100 runner at a time, 18,000 seconds in total for the run. Delete a runner when
done and confirm termination. The Driver owns the GPU credentials; do not
invoke Modal or call Driver endpoints yourself.
