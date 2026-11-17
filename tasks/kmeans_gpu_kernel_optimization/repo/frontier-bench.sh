#!/bin/sh
set -eu
python3 - <<'PY'
import time
import torch
import kmeanslib
if not torch.cuda.is_available():
    raise SystemExit('CUDA is required')
g=torch.Generator(device='cuda').manual_seed(20260912)
x=torch.randn((32768,64),generator=g,device='cuda',dtype=torch.bfloat16)
c=x[:256].clone()
for _ in range(2):
    labels,c=kmeanslib.step(x,c)
torch.cuda.synchronize(); start=time.perf_counter()
for _ in range(2):
    labels,c=kmeanslib.step(x,c)
torch.cuda.synchronize()
assert labels.shape==(32768,) and c.shape==(256,64)
assert torch.isfinite(c.float()).all()
print({'two_step_latency_ms':round((time.perf_counter()-start)*1000,3),'labels':labels.numel()})
PY
