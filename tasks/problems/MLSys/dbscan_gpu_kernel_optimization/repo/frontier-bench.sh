#!/bin/sh
set -eu
python3 - <<'PY'
import torch
from dbscanlib import dbscan
x = torch.tensor([[0,0], [.05,0], [0,.05], [3,3], [3.05,3], [3,3.05], [10,10]], device='cuda', dtype=torch.float32)
labels = dbscan(x, .2, 3)
assert labels.shape == (7,) and labels.dtype == torch.int64 and labels.device == x.device
values = labels.cpu().tolist()
assert values[6] == -1 and all(value >= 0 for value in values[:6])
assert len(set(values[:3])) == len(set(values[3:6])) == 1 and values[0] != values[3]
print('Ungraded cluster-equivalence and noise diagnostic passed.')
PY
