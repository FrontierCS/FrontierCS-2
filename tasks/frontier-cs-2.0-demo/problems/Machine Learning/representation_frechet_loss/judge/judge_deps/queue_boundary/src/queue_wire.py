"""Bounded tensor-only FD messages; no pickle, arbitrary metadata or paths."""

import math
import os
import select
import struct
import time

MAGIC = b"FQ01"
HEADER = struct.Struct("<4sIHH")
DESCRIPTOR = struct.Struct("<BBHII")
MAX_BYTES = 1024 * 1024 * 1024
MAX_TENSORS = 8
READY, CREATE, IMPORT, EXPORT, INIT, ACCUMULATE, FINALIZE = range(1, 8)
STATS, SNAPSHOT, STATS_NOGRAD, SNAPSHOT_NOGRAD, VJP, ENQUEUE = range(8, 14)
DEVICE = 14
OPS = set(range(1, 15))
DTYPES = {1: ("float32", 4), 2: ("float64", 8), 3: ("int64", 8),
          4: ("bfloat16", 2), 5: ("float16", 2)}


def exact(fd, size, deadline):
    if not 0 <= size <= MAX_BYTES:
        raise ValueError("Frame size exceeds protocol bounds")
    data = bytearray()
    while len(data) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            raise TimeoutError("Queue response deadline")
        part = os.read(fd, min(65536, size - len(data)))
        if not part:
            raise EOFError("Queue worker closed its output")
        data.extend(part)
    return data


def write(fd, data, deadline):
    data = memoryview(data)
    while data:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
            raise TimeoutError("Queue input deadline")
        n = os.write(fd, data[:4096])
        data = data[n:]


def extent(dtype, rank, reserved, row, col):
    if dtype not in DTYPES or reserved or rank > 2:
        raise ValueError("Invalid tensor descriptor")
    if (rank == 0 and (row, col) != (1, 1)) or (rank == 1 and col != 1):
        raise ValueError("Noncanonical tensor dimensions")
    if row > 50000 or col > 4096:
        raise ValueError("Tensor dimensions exceed workload bounds")
    shape = () if rank == 0 else (row,) if rank == 1 else (row, col)
    size = math.prod(shape) * DTYPES[dtype][1]
    if size > MAX_BYTES:
        raise ValueError("Tensor bytes exceed protocol bounds")
    return shape, size


def send(fd, seq, op, tensors, deadline):
    import torch
    if type(seq) is not int or not 0 <= seq < 2**32 or op not in OPS or len(tensors) > MAX_TENSORS:
        raise ValueError("Invalid queue message")
    descriptors, total = [], 0
    for tensor in tensors:
        if type(tensor) is not torch.Tensor or tensor.layout != torch.strided:
            raise ValueError("Expected a plain dense tensor")
        code = next((k for k, (name, _) in DTYPES.items() if getattr(torch, name) == tensor.dtype), None)
        shape = tensor.shape
        if len(shape) > 2:
            raise ValueError("Too many tensor dimensions")
        descriptor = (code, len(shape), 0, shape[0] if shape else 1, shape[1] if len(shape) == 2 else 1)
        _, size = extent(*descriptor)
        total += size
        if total > MAX_BYTES or not bool(torch.isfinite(tensor).all()):
            raise ValueError("Nonfinite or oversized tensor message")
        descriptors.append(descriptor)
    write(fd, HEADER.pack(MAGIC, seq, op, len(tensors)), deadline)
    for tensor, descriptor in zip(tensors, descriptors):
        write(fd, DESCRIPTOR.pack(*descriptor), deadline)
        body = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy()
        write(fd, memoryview(body), deadline)


def receive(fd, deadline):
    import torch
    magic, seq, op, count = HEADER.unpack(exact(fd, HEADER.size, deadline))
    if magic != MAGIC or op not in OPS or count > MAX_TENSORS:
        raise ValueError("Invalid queue frame header")
    result, total = [], 0
    for _ in range(count):
        descriptor = DESCRIPTOR.unpack(exact(fd, DESCRIPTOR.size, deadline))
        shape, size = extent(*descriptor)
        total += size
        if total > MAX_BYTES:
            raise ValueError("Message exceeds byte bound")
        data = exact(fd, size, deadline)
        dtype = getattr(torch, DTYPES[descriptor[0]][0])
        # The tensor owns a reference to this private bytearray. No shared
        # memory or pickle is involved, so another full payload copy is needless.
        tensor = (torch.frombuffer(data, dtype=dtype).reshape(shape)
                  if size else torch.empty(shape, dtype=dtype))
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Worker supplied a nonfinite tensor")
        result.append(tensor)
    return seq, op, result


def state_layout(size, dim, online, beta):
    if type(size) is not int or not 0 <= size <= 50000 or type(dim) is not int or not 1 <= dim <= 4096:
        raise ValueError("Queue size/feature dimension outside contract")
    if type(online) is not bool or type(beta) is not float or not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("Invalid estimator options")
    if beta > 0:
        return {"mu_ema": ((dim,), "float64"), "m2_ema": ((dim, dim), "float64"),
                "_ema_count": ((1,), "int64")}
    result = {"feats": ((size, dim), "float32"), "ptr": ((1,), "int64")}
    if online and size:
        result.update(feat_sum_old=((dim,), "float64"), feat_outer_old=((dim, dim), "float64"))
    return result


def validate_state(tensors, layout):
    import torch
    if len(tensors) != len(layout):
        raise ValueError("Wrong number of queue state tensors")
    for tensor, (shape, dtype) in zip(tensors, layout.values()):
        if type(tensor) is not torch.Tensor or tuple(tensor.shape) != shape or tensor.dtype != getattr(torch, dtype):
            raise ValueError("Queue state descriptor mismatch")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Nonfinite queue state")
    state = dict(zip(layout, tensors))
    if "ptr" in state and not 0 <= state["ptr"].item() < max(1, state["feats"].shape[0]):
        raise ValueError("Queue pointer outside population")
    # A solver may count post-warm-start observations too. The scalar does not
    # size a tensor; its int64 descriptor already bounds storage and transport.
    if "_ema_count" in state and state["_ema_count"].item() < 0:
        raise ValueError("Negative EMA population count")
