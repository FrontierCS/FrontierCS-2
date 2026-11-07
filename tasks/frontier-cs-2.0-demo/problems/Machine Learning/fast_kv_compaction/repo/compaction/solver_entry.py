"""The method this task evaluates. Everything under compaction/ is yours to edit.

The evaluator runs this module in its own worker process, which owns a copy of
Qwen3-4B (Hugging Face model, bf16, on the GPU), its tokenizer and a vLLM
engine for the same weights. Two functions are called:

  setup(model, tokenizer, engine) -> state
      Once, before any article is seen (not timed). Build whatever you need.
      `engine` is a vllm.LLM kept asleep between uses: call engine.wake_up()
      before generating and engine.sleep() afterwards to give its memory back.

  compact(state, request) -> list of layers
      Once per article (timed, together with the worker's prefill of the
      article and the transfer of your result). `request` is a dict:
        model, tokenizer, engine   as above
        input_ids        (1, S) token ids of the whole prompt prefix: chat
                         template + article (the questions come later)
        text             the same prefix as a string
        article_span     (start, stop): the article's token positions in it;
                         the positions outside are the chat template
        cache            list over the 36 layers of (keys, values), each
                         (1, 8, S, 128) bf16 on the GPU; keys carry their
                         rotary position embedding already
        past_key_values  the same cache as the Hugging Face cache object
        target_tokens    tokens per head you may keep: every template token
                         plus TARGET_SIZE of the article tokens
        seed             the run's seed (already applied to torch/numpy/random)

      Return one entry per layer, a dict:
        keys            (1, 8, t, 128)
        values          (1, 8, t, 128)
        attention_bias  optional (1, 8, t) float32: an additive term on the
                        attention logits of each kept key (absent = zeros;
                        -inf marks an unused slot)
      All heads of a layer share t; t may differ between layers. Tensors may be
      bf16, fp16 or fp32 (keys and values of a layer in the same dtype). The
      evaluator rebuilds the tensors from what you return and answers every
      question from that cache alone, appending each question after position
      S. Bytes are counted exactly as returned; an absent attention_bias costs
      nothing.

TARGET_SIZE is the fraction of article tokens to keep. It must be a literal
number in (0, 0.1]: 0.1 compares compaction time with the reference at its own
budget; anything smaller compares cache bytes instead.

This starting point keeps every template token and a uniformly random subset
of the article's tokens, separately for each head, with the original keys and
values and no bias.
"""
import torch

TARGET_SIZE = 0.1


def setup(model, tokenizer, engine):
    return None


def compact(state, request):
    start, stop = request["article_span"]
    total = request["input_ids"].shape[1]
    article = stop - start
    keep_article = max(0, min(article, request["target_tokens"] - (total - article)))
    outside = torch.cat([torch.arange(0, start), torch.arange(stop, total)])
    generator = torch.Generator().manual_seed(int(request["seed"]))
    layers = []
    for keys, values in request["cache"]:
        heads = keys.shape[1]
        chosen = []
        for _ in range(heads):
            picked = start + torch.randperm(article, generator=generator)[:keep_article]
            chosen.append(torch.sort(torch.cat([outside, picked])).values)
        index = torch.stack(chosen).to(keys.device)                    # (heads, t)
        gather = index[None, :, :, None].expand(1, heads, index.shape[1], keys.shape[3])
        layers.append({"keys": torch.gather(keys, 2, gather), "values": torch.gather(values, 2, gather)})
    return layers
