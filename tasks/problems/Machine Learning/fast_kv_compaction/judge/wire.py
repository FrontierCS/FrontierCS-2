"""Framed messages between the trusted harness and the compaction worker.

A frame is: 8-byte big-endian header length, UTF-8 JSON header, then the raw
little-endian bytes of each tensor in header order. Requests and status replies
carry no tensors. The parser accepts only plain JSON objects, the dtypes and
ranks below, and exact byte counts; nothing is unpickled, so the worker cannot
hand the harness an object, only bytes that the harness turns into new tensors.
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any, BinaryIO

MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 8 << 30
MAX_TENSORS = 4096
# dtype name -> bytes per element
DTYPES = {"bfloat16": 2, "float16": 2, "float32": 4}


class ProtocolError(ValueError):
    """A malformed frame: the worker broke the protocol."""


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    chunks, remaining = [], count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ProtocolError("worker closed the channel mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def write_frame(stream: BinaryIO, header: dict, payloads: list[bytes] = ()) -> None:
    body = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    stream.write(struct.pack(">Q", len(body)))
    stream.write(body)
    for payload in payloads:
        stream.write(payload)
    stream.flush()


def tensor_specs(header: dict) -> list[dict]:
    """Validate a tensor frame header; return [{dtype, shape, nbytes}]."""
    specs = header.get("tensors")
    if not isinstance(specs, list) or len(specs) > MAX_TENSORS:
        raise ProtocolError("tensor list is invalid")
    total = 0
    clean = []
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != {"dtype", "shape"}:
            raise ProtocolError("tensor spec is invalid")
        dtype, shape = spec["dtype"], spec["shape"]
        if dtype not in DTYPES:
            raise ProtocolError("tensor dtype is not allowed")
        if (not isinstance(shape, list) or not 1 <= len(shape) <= 4
                or any(type(size) is not int or size < 0 for size in shape)):
            raise ProtocolError("tensor shape is invalid")
        nbytes = DTYPES[dtype] * math.prod(shape)
        total += nbytes
        if total > MAX_PAYLOAD_BYTES:
            raise ProtocolError("tensor payload is too large")
        clean.append({"dtype": dtype, "shape": list(shape), "nbytes": nbytes})
    return clean


def read_frame(stream: BinaryIO) -> tuple[dict, list[bytes]]:
    (length,) = struct.unpack(">Q", _read_exact(stream, 8))
    if not 2 <= length <= MAX_HEADER_BYTES:
        raise ProtocolError("frame header length is invalid")
    try:
        header = json.loads(_read_exact(stream, length).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("frame header is not JSON") from exc
    if type(header) is not dict or not isinstance(header.get("kind"), str):
        raise ProtocolError("frame header is invalid")
    payloads = []
    if "tensors" in header:
        for spec in tensor_specs(header):
            payloads.append(_read_exact(stream, spec["nbytes"]))
    return header, payloads


def layer_layout(specs: list[dict], layers: int, has_bias: list) -> list[tuple[dict, dict, dict | None]]:
    """A returned cache is `layers` entries of keys, values and, where has_bias
    says so, attention_bias, in that order.

    keys and values are (1, KV, t, D) in the same dtype; attention_bias is
    (1, KV, t). Returns (keys, values, attention_bias or None) per layer, or
    raises ProtocolError.
    """
    if (type(layers) is not int or layers <= 0 or not isinstance(has_bias, list) or len(has_bias) != layers
            or any(type(flag) is not bool for flag in has_bias)
            or len(specs) != 2 * layers + sum(has_bias)):
        raise ProtocolError("returned cache layout does not match its layer list")
    grouped, position = [], 0
    for flag in has_bias:
        keys, values = specs[position], specs[position + 1]
        bias = specs[position + 2] if flag else None
        position += 3 if flag else 2
        if len(keys["shape"]) != 4 or keys["shape"] != values["shape"] or keys["dtype"] != values["dtype"]:
            raise ProtocolError("returned keys and values disagree")
        if bias is not None and bias["shape"] != keys["shape"][:3]:
            raise ProtocolError("returned attention_bias shape is invalid")
        if keys["shape"][0] != 1:
            raise ProtocolError("returned cache batch must be 1")
        grouped.append((keys, values, bias))
    return grouped


def cache_bytes(specs: list[dict]) -> int:
    """Stored bytes of the returned cache: every element of every returned tensor, once
    (an absent attention_bias costs nothing)."""
    return sum(spec["nbytes"] for spec in specs)
