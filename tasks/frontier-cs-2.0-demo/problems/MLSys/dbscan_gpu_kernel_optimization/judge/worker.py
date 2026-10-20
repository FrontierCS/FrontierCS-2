"""One implementation/workload process inside the Driver-owned Verifier GPU."""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path
from time import perf_counter


def generate(w, seed):
    import torch
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(int(seed))
    # Embed interleaving half-moon pairs in random 2D subspaces with ambient noise.
    nc, D, N = int(w["n_centers"]), int(w["D"]), int(w["N"])
    pairs = max(1, nc // 2)
    basis, _ = torch.linalg.qr(torch.randn(D, D, generator=g, device=dev))  # orthonormal
    per = (N + pairs - 1) // pairs
    parts = []
    for p in range(pairs):
        nout = per // 2; nin = per - nout
        to = torch.rand(nout, generator=g, device=dev) * torch.pi
        ti = torch.rand(nin, generator=g, device=dev) * torch.pi
        mx = torch.cat([torch.cos(to), 1.0 - torch.cos(ti)])
        my = torch.cat([torch.sin(to), 0.5 - torch.sin(ti)])
        m = (torch.stack([mx, my], 1) + torch.randn(per, 2, generator=g, device=dev) * 0.05) * 6.0
        b0 = basis[:, (2 * p) % D]; b1 = basis[:, (2 * p + 1) % D]
        pts = m[:, 0:1] * b0[None, :] + m[:, 1:2] * b1[None, :]
        pts = pts + torch.randn(per, D, generator=g, device=dev) * 0.1          # thin ambient noise
        pts = pts + basis[:, (2 * p + 4) % D][None, :] * (p * 30.0)             # separate the pairs
        parts.append(pts)
    x = torch.cat(parts, 0)
    x = x[torch.randperm(x.shape[0], generator=g, device=dev)][:N]
    # Preserve the legacy bf16 precision lock while exposing float32 inputs.
    return {"x": x.to(torch.bfloat16).to(torch.float32), "eps": float(w["eps"]),
            "min_samples": int(w["min_samples"])}


def measure(source: Path, workload: dict, warmups: int, iterations: int) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("DBSCAN grading requires a CUDA GPU")
    synchronize = torch.cuda.synchronize
    sys.path.insert(0, str(source))
    module = importlib.import_module("dbscanlib")
    base_seed = workload["base_seed"] + 1000 * (int(workload["id"][1:]) + 1)
    samples = []
    for measured, count, offset in ((False, warmups, 100), (True, iterations, 10000)):
        for index in range(count):
            data = generate(workload, base_seed + offset + index)
            synchronize()
            started = perf_counter()
            labels = module.dbscan(data["x"], data["eps"], data["min_samples"])
            synchronize()
            milliseconds = (perf_counter() - started) * 1000.0
            if (not isinstance(labels, torch.Tensor) or labels.ndim != 1
                    or labels.shape[0] != workload["N"]
                    or labels.dtype != torch.int64 or labels.device != data["x"].device):
                raise ValueError("DBSCAN must return int64 labels (N,) on the input device")
            if bool((labels < -1).any()):
                raise ValueError("DBSCAN labels must be nonnegative or noise -1")
            if not math.isfinite(milliseconds) or milliseconds <= 0:
                raise ValueError("DBSCAN measurement must be finite and positive")
            if measured:
                samples.append({"latency_ms": milliseconds, "labels": labels.cpu().tolist()})
            del labels, data
            torch.cuda.empty_cache()
    return {"id": workload["id"], "samples": samples}


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        raise ValueError("worker requires source, workload, warmups, iterations, result")
    source, encoded, warmups, iterations, result = argv[1:]
    row = measure(Path(source), json.loads(encoded), int(warmups), int(iterations))
    Path(result).write_text(json.dumps(row, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
