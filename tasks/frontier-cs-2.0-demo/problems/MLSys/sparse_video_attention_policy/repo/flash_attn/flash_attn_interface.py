"""Task-owned ``flash_attn_varlen_func`` for the retained HunyuanVideo integration.

Upstream ``svg/models/hyvideo/attention.py`` imports
``flash_attn.flash_attn_interface.flash_attn_varlen_func`` and uses it for the
dense HunyuanVideo processor (``--pattern dense``, which prepares the fixed dense
targets) and for the older SVG processor. The selected EAR path uses FlashInfer
for its full-attention steps and never calls this function, but the module-level
import must succeed.

The ``flash-attn`` wheel is not part of the pinned image. This adapter computes
the same non-causal attention, isolated per packed segment, with PyTorch
``scaled_dot_product_attention`` -- the dense backend the retained Wan processor
already uses. The input layout is FlashAttention's packed
``(total_tokens, heads, head_dim)``; ``cu_seqlens_*`` are cumulative segment
offsets. Only the call shape used by the retained code is supported; anything
else raises instead of silently computing something different.

This file is task-owned and outside the editable surface.
"""

import torch
import torch.nn.functional as F


def _offsets(cu_seqlens, total):
    values = [int(value) for value in cu_seqlens.tolist()]
    if not values or values[0] != 0 or values[-1] != total or any(b < a for a, b in zip(values, values[1:])):
        raise ValueError("cu_seqlens must be non-decreasing offsets from 0 to the packed length")
    return values


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **unsupported,
):
    if unsupported or dropout_p or causal:
        raise NotImplementedError("only non-causal, dropout-free varlen attention is provided")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3 or k.shape != v.shape or q.shape[1:] != k.shape[1:]:
        raise ValueError("expected packed (tokens, heads, head_dim) tensors with equal head counts")
    q_offsets = _offsets(cu_seqlens_q, q.shape[0])
    k_offsets = _offsets(cu_seqlens_k, k.shape[0])
    if len(q_offsets) != len(k_offsets):
        raise ValueError("query and key segment counts differ")
    output = torch.empty_like(q)
    for (q_start, q_end), (k_start, k_end) in zip(
        zip(q_offsets, q_offsets[1:]), zip(k_offsets, k_offsets[1:])
    ):
        if q_end == q_start:
            continue
        if k_end == k_start:
            raise ValueError("a non-empty query segment has no keys")
        segment = F.scaled_dot_product_attention(
            q[q_start:q_end].transpose(0, 1).unsqueeze(0),
            k[k_start:k_end].transpose(0, 1).unsqueeze(0),
            v[k_start:k_end].transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
            is_causal=False,
            scale=softmax_scale,
        )
        output[q_start:q_end] = segment.squeeze(0).transpose(0, 1)
    return output
