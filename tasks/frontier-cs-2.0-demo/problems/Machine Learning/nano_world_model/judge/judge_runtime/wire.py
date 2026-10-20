"""Bounded primitive tensor messages; never deserialize Python objects."""

from __future__ import annotations

import base64
import math
import struct


MAX_MESSAGE_BYTES = 1024 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024
SAMPLE_SHAPE = (1, 4, 4, 32, 32)
CONTEXT_SHAPE = (1, 1, 4, 32, 32)
ACTION_SHAPE = (1, 4, 10)


class InvalidResponse(ValueError):
    pass


def encode_tensor(data: bytes, shape) -> dict:
    shape = tuple(shape)
    if shape not in (SAMPLE_SHAPE, CONTEXT_SHAPE, ACTION_SHAPE):
        raise ValueError("unsupported task tensor shape")
    if len(data) != math.prod(shape) * 4:
        raise ValueError("tensor bytes do not match shape")
    if any(not math.isfinite(value[0]) for value in struct.iter_unpack("<f", data)):
        raise ValueError("non-finite tensor")
    return {"dtype": "float32-le", "shape": list(shape),
            "data": base64.b64encode(data).decode("ascii")}


def decode_tensor(value, shape) -> bytes:
    if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data"}:
        raise InvalidResponse("invalid tensor fields")
    if (value["dtype"] != "float32-le" or value["shape"] != list(shape)
            or any(type(dim) is not int for dim in value["shape"])):
        raise InvalidResponse("invalid tensor type or shape")
    encoded = value["data"]
    size = math.prod(shape) * 4
    if not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3):
        raise InvalidResponse("invalid tensor length")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, UnicodeError) as exc:
        raise InvalidResponse("invalid tensor encoding") from exc
    if len(data) != size:
        raise InvalidResponse("invalid decoded tensor length")
    if any(not math.isfinite(item[0]) for item in struct.iter_unpack("<f", data)):
        raise InvalidResponse("non-finite candidate tensor")
    return data


def memory_sources(root) -> dict[str, dict[str, object]]:
    """Read only the fixed public diffusion package, including allowed helpers."""
    from pathlib import Path

    root = Path(root).resolve()
    sources = {}
    total = 0
    for path in sorted((root / "src/diffusion").rglob("*.py")):
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if root in p.parents):
            raise ValueError("candidate module path contains a symlink")
        relative = path.relative_to(root / "src").with_suffix("")
        parts = relative.parts
        package = parts[-1] == "__init__"
        module = ".".join(parts[:-1] if package else parts)
        raw = path.read_bytes()
        total += len(raw)
        if total > MAX_SOURCE_BYTES:
            raise ValueError("candidate module closure exceeds bound")
        source = raw.decode("utf-8")
        compile(source, "<candidate:" + module + ">", "exec")
        sources[module] = {"package": package, "source": source}
    if "diffusion" not in sources or "diffusion.df_sample" not in sources:
        raise ValueError("candidate diffusion package is incomplete")
    for module in list(sources):
        parts = module.split(".")
        for index in range(1, len(parts)):
            sources.setdefault(".".join(parts[:index]), {"package": True, "source": ""})
    return sources
