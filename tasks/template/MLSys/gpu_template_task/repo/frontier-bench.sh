#!/bin/sh
# An ungraded timing and accuracy check on one small public shape (run it on a GPU runner).
set -eu
python3 - <<'PY'
import time
import torch
import kernel
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
g = torch.Generator(device="cuda").manual_seed(20261107)
x = torch.randn((2048, 64), generator=g, device="cuda")
y = torch.randn((2048, 64), generator=g, device="cuda")
for _ in range(3):
    out = kernel.pairwise_sq_dists(x, y)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    out = kernel.pairwise_sq_dists(x, y)
torch.cuda.synchronize()
exact = ((x.double()[:, None, :] - y.double()[None, :, :]) ** 2).sum(-1)
error = ((out.double() - exact).abs() / (1 + exact.abs())).max().item()
print({"latency_ms": round((time.perf_counter() - start) * 100, 3), "max_error": error})
PY
