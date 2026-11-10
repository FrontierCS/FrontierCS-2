"""One measurement of one implementation on one workload, inside an isolate instance on the GPU.

judge/evaluator.py passes this file as an input of isolate.run(devices=["nvidia"]): the
instance's work dir is the implementation's tree (holding kernel/), and it reads nothing of the
judge but its inputs. It loads the workload's float32 inputs, imports `kernel` from the tree,
runs the warm-up calls, times the CUDA-synchronized calls and writes the median latency and the
output rows the judge asked for. The judge checks those rows itself; this file decides nothing.

    worker.py WORK X_BIN Y_BIN N M D ROWS_JSON WARMUPS TIMED RESULT_JSON
"""

import importlib
import json
import statistics
import sys
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    import torch

    work, x_bin, y_bin = argv[0], Path(argv[1]), Path(argv[2])
    n, m, d = int(argv[3]), int(argv[4]), int(argv[5])
    rows, warmups, timed, result = json.loads(argv[6]), int(argv[7]), int(argv[8]), Path(argv[9])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    x = torch.frombuffer(bytearray(x_bin.read_bytes()), dtype=torch.float32).view(n, d).cuda()
    y = torch.frombuffer(bytearray(y_bin.read_bytes()), dtype=torch.float32).view(m, d).cuda()
    sys.path.insert(0, work)
    kernel = importlib.import_module("kernel")
    for _ in range(warmups):
        kernel.pairwise_sq_dists(x, y)
    torch.cuda.synchronize()
    latencies = []
    for _ in range(timed):
        start = time.perf_counter()
        out = kernel.pairwise_sq_dists(x, y)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
    if not isinstance(out, torch.Tensor) or tuple(out.shape) != (n, m):
        raise ValueError("pairwise_sq_dists returned the wrong shape")
    picked = out[torch.tensor(rows, device=out.device)].double().cpu().tolist()
    result.write_text(json.dumps({"latency_ms": statistics.median(latencies), "rows": picked}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
