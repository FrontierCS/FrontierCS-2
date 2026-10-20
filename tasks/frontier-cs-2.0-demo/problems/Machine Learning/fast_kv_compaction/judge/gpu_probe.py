#!/usr/bin/env python3
"""GPU residue probe, run inside the worker namespaces by the launcher's gpu-write/gpu-read modes.

`write` fills most free GPU memory with a marker and exits; `read`, in a fresh
process after the harness's scrub, allocates the same amount uninitialized and
counts marker words left over. The harness requires zero.

The read side counts in fixed-size chunks so the only temporaries are a small
per-chunk mask, never a tensor the size of the whole block. A probe that fails
(CUDA error, out of memory) prints FASTKV_GPU_PROBE_ERROR instead of a count,
which the harness reports as a probe failure, not as residue.
"""
import sys

import torch

MARKER = 0x5AC35AC35AC35AC3
FRACTION = 0.85
CHUNK_WORDS = (1 << 30) // 8  # 1 GiB of int64 words per counting view


def write(words: int) -> None:
    block = torch.empty((words,), dtype=torch.int64, device="cuda")
    block.fill_(MARKER)
    torch.cuda.synchronize()
    print(f"FASTKV_GPU_WRITTEN {block.numel()}")


def read(words: int) -> None:
    block = torch.empty((words,), dtype=torch.int64, device="cuda")
    hits = 0
    for start in range(0, words, CHUNK_WORDS):
        view = block[start:start + CHUNK_WORDS]
        hits += int(torch.count_nonzero(view == MARKER))
    print(f"FASTKV_GPU_RESIDUE {hits}")


def main() -> int:
    mode = sys.argv[1]
    try:
        free, _total = torch.cuda.mem_get_info()
        words = int(free * FRACTION) // 8
        {"write": write, "read": read}[mode](words)
    except Exception as error:  # reported as a probe failure, never as a residue count
        message = " ".join(f"{type(error).__name__}: {error}".split())
        print(f"FASTKV_GPU_PROBE_ERROR {mode} {message}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
