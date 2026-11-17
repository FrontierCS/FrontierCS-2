import contextvars
import inspect
import os
import re
import textwrap
from contextlib import contextmanager


def _try_import_triton():
    try:
        import triton
        import triton.language as tl
    except Exception:
        return None, None

    @triton.jit
    def _svg_kvidx_kernel(
        base_ptr,
        base_off_ptr,
        lengths_ptr,
        kv_indices_ptr,
        MAX_BLOCK_SIZE2: tl.constexpr,
    ):
        pid = tl.program_id(0)

        base = tl.load(base_ptr + pid).to(tl.int64)
        base_off = tl.load(base_off_ptr + pid).to(tl.int64)
        length = tl.load(lengths_ptr + pid).to(tl.int32)

        offset = tl.arange(0, MAX_BLOCK_SIZE2)
        off_mask = offset < length

        kv_indices_ptr += base_off
        kv_idx = base + offset

        tl.store(kv_indices_ptr + offset, kv_idx, mask=off_mask)

    try:
        # Triton JITFunction usually stores the original python function in .fn
        fn = getattr(_svg_kvidx_kernel, "fn", None)
        g = fn.__globals__ if fn is not None else _svg_kvidx_kernel.__globals__
        g["tl"] = tl
        g["triton"] = triton
    except Exception:
        pass
    
    return triton, _svg_kvidx_kernel


_PATCH_ENABLED = contextvars.ContextVar("svg_flashinfer_patch_enabled", default=False)
_LOG_FAILURES = os.getenv("SVG_FLASHINFER_PATCH_DEBUG", "0") == "1"


@contextmanager
def flashinfer_patch_enabled():
    token = _PATCH_ENABLED.set(True)
    try:
        yield
    finally:
        _PATCH_ENABLED.reset(token)


def _make_expand_kv_indices(triton_mod, kvidx_kernel):
    import torch

    def _svg_expand_kv_indices(lengths, base, device, dtype_i, cum=None):
        if cum is None:
            cum = torch.cumsum(lengths, 0)
        if cum.numel() == 0:
            return torch.empty((0,), dtype=dtype_i, device=device)

        total_len = int(cum[-1].item())
        if total_len == 0:
            return torch.empty((0,), dtype=dtype_i, device=device)

        if (
            kvidx_kernel is not None
            and triton_mod is not None
            and lengths.is_cuda
            and base.is_cuda
        ):
            num_blocks = int(lengths.numel())
            base_i64 = base.to(torch.int64)
            base_off = torch.cat(
                [torch.zeros(1, dtype=dtype_i, device=device), cum[:-1]]
            )
            kv_indices = torch.empty((total_len,), dtype=dtype_i, device=device)

            max_block = int(
                triton_mod.next_power_of_2(int(lengths.max().item()))
            )
            kvidx_kernel[(num_blocks,)](
                base_i64,
                base_off,
                lengths,
                kv_indices,
                MAX_BLOCK_SIZE2=max_block,
            )
            return kv_indices

        starts = torch.repeat_interleave(cum - lengths, lengths)
        offsets_within = torch.arange(
            total_len, device=device, dtype=dtype_i
        ) - starts
        return (torch.repeat_interleave(base, lengths) + offsets_within).to(
            dtype_i
        )

    return _svg_expand_kv_indices


def _build_patched_plan(orig_plan, logger=None):
    if getattr(orig_plan, "__svg_flashinfer_patch_applied__", False):
        return orig_plan

    try:
        src = inspect.getsource(orig_plan)
    except Exception:
        if _LOG_FAILURES and logger is not None:
            try:
                logger.warning("FlashInfer patch: getsource failed.")
            except Exception:
                pass
        return None

    if "_svg_expand_kv_indices" in src:
        return orig_plan

    dedented = textwrap.dedent(src)

    pattern = re.compile(
        r"(?P<indent>\s*)cum\s*=\s*torch\.cumsum\(\s*lengths\s*,\s*(?:0|dim\s*=\s*0)\s*\)\s*\n"
        r"(?P=indent)starts\s*=.*\n"
        r"(?P=indent)offsets_within\s*=.*\n"
        r"(?P=indent)kv_indices\s*=\s*torch\.repeat_interleave\(\s*base\s*,\s*lengths\s*\)\s*\+\s*offsets_within",
        re.MULTILINE,
    )

    match = pattern.search(dedented)
    if not match:
        if _LOG_FAILURES and logger is not None:
            try:
                logger.warning("FlashInfer patch: pattern miss.")
            except Exception:
                pass
        return None

    if re.search(r"\boffsets_within\b", dedented[match.end() :]) or re.search(
        r"\bstarts\b", dedented[match.end() :]
    ):
        if _LOG_FAILURES and logger is not None:
            try:
                logger.warning("FlashInfer patch: extra refs after match.")
            except Exception:
                pass
        return None

    indent = match.group("indent")
    replacement = (
        f"{indent}cum = torch.cumsum(lengths, 0)\n"
        f"{indent}kv_indices = _svg_expand_kv_indices(lengths, base, device, dtype_i, cum=cum)"
    )

    patched = dedented[: match.start()] + replacement + dedented[match.end() :]

    assign_pat = re.compile(
        r"^(?P<indent>\s*)kv_indices_host\s*=\s*kv_indices\.to\(\s*\"cpu\"\s*,\s*non_blocking\s*=\s*non_blocking\s*\)\s*$",
        re.MULTILINE,
    )
    if assign_pat.search(patched):
        patched = assign_pat.sub(r"\g<indent># svg: skip kv_indices_host copy", patched, count=1)

    patched = re.sub(r"\bkv_indices_host\b", "kv_indices", patched)

    triton_mod, kvidx_kernel = _try_import_triton()
    if triton_mod is None and kvidx_kernel is None:
        kvidx_kernel = None

    expand_fn = _make_expand_kv_indices(triton_mod, kvidx_kernel)

    globals_dict = dict(orig_plan.__globals__)
    globals_dict["_svg_expand_kv_indices"] = expand_fn
    if triton_mod is not None:
        globals_dict["triton"] = triton_mod
    if kvidx_kernel is not None:
        globals_dict["_svg_kvidx_kernel"] = kvidx_kernel

    locals_dict = {}
    try:
        exec(patched, globals_dict, locals_dict)
    except Exception:
        if _LOG_FAILURES and logger is not None:
            try:
                logger.exception("FlashInfer patch: exec failed.")
            except Exception:
                pass
        return None

    new_plan = locals_dict.get("plan")
    if new_plan is None:
        if _LOG_FAILURES and logger is not None:
            try:
                logger.warning("FlashInfer patch: plan missing after exec.")
            except Exception:
                pass
        return None

    setattr(new_plan, "__svg_flashinfer_patch_applied__", True)
    return new_plan


def apply_flashinfer_patch(logger=None) -> bool:
    if os.getenv("SVG_DISABLE_FLASHINFER_PATCH", "0") == "1":
        return False

    try:
        import flashinfer
    except Exception:
        return False

    sparse_mod = getattr(flashinfer, "sparse", None)
    if sparse_mod is None:
        return False

    cls = getattr(sparse_mod, "VariableBlockSparseAttentionWrapper", None)
    if cls is None:
        return False
    current_plan = cls.plan
    if getattr(current_plan, "__svg_flashinfer_wrapper__", False):
        return True

    orig_plan = getattr(cls, "__svg_flashinfer_plan_orig__", current_plan)
    patched_plan = _build_patched_plan(orig_plan, logger=logger)
    if patched_plan is None:
        return False

    def plan_wrapper(self, *args, **kwargs):
        if _PATCH_ENABLED.get():
            return patched_plan(self, *args, **kwargs)
        return orig_plan(self, *args, **kwargs)

    setattr(plan_wrapper, "__svg_flashinfer_wrapper__", True)
    setattr(cls, "__svg_flashinfer_plan_orig__", orig_plan)
    setattr(cls, "__svg_flashinfer_plan_patched__", patched_plan)
    setattr(cls, "plan", plan_wrapper)

    if logger is not None:
        try:
            logger.info("Installed SVG FlashInfer gated monkey patch.")
        except Exception:
            pass
    return True
