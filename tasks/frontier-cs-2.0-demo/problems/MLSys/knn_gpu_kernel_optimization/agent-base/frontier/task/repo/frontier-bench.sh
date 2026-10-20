#!/bin/sh
set -eu
python3 - <<'PY'
import time
import torch
import knnlib
if not torch.cuda.is_available():
    raise SystemExit('CUDA is required')
generator = torch.Generator(device='cuda').manual_seed(20260912)
queries = torch.randn((128, 128), generator=generator, device='cuda', dtype=torch.bfloat16)
database = torch.randn((8192, 128), generator=generator, device='cuda', dtype=torch.bfloat16)
knnlib.knn(queries, database, 10)
torch.cuda.synchronize()
started = time.perf_counter()
distances, indices = knnlib.knn(queries, database, 10)
torch.cuda.synchronize()
assert distances.shape == indices.shape == (128, 10)
assert torch.isfinite(distances).all()
print({'diagnostic_only': True, 'latency_ms': (time.perf_counter() - started) * 1000})
PY
