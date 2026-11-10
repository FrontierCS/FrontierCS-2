"""The trusted timing harness on the class-2 machine (task.toml [test_sandboxes.timing]).

    timing_harness.py ROOT WORKLOADS OUT_JSON

judge/evaluator.py uploads it to ROOT (the sandbox's workdir) with predict_worker.py, the trees
(trees/<impl>/), their fitted parameters (params/<impl>/<id>.json) and the queries
(queries/<id>.json). It runs on the Verifier GPU image, where the shared isolate launcher is on
the interpreter's import path, and runs every predict in a fresh isolate instance with the GPU,
REPLICATES times per workload, candidate and reference interleaved. OUT_JSON:
{impl: {workload: [seconds or null]}}: the instance's run time (process start to exit), null
when it failed or hit its limit. The harness itself fails (non-zero exit) only for its own
faults, e.g. when FRONTIER_VERIFIER_LOCAL_DEADLINE (the end of its step's limit) leaves no time
for the next run.
"""

import json
import os
import sys
import time
from pathlib import Path

import isolate  # the shared launcher, from gpu-tools (frontier.pth in the Verifier GPU image)

IMPLEMENTATIONS = ("candidate", "reference")
REPLICATES = 3
RUN_TIMEOUT = 60
RESERVE_SECONDS = 30   # to write OUT_JSON and exit before the deadline


def main(argv: list[str]) -> int:
    root, workloads, out = Path(argv[0]), argv[1].split(","), Path(argv[2])
    deadline = float(os.environ["FRONTIER_VERIFIER_LOCAL_DEADLINE"])
    python = os.path.realpath(sys.executable)
    times = {impl: {wid: [] for wid in workloads} for impl in IMPLEMENTATIONS}
    for _ in range(REPLICATES):
        for wid in workloads:
            for impl in IMPLEMENTATIONS:
                remaining = deadline - time.monotonic() - RESERVE_SECONDS
                if remaining < RUN_TIMEOUT:
                    raise SystemExit("the step's deadline leaves no time for the next run")
                result = isolate.run(
                    [python, "{in}/predict_worker.py", "{work}", "{in}/params.json",
                     "{in}/queries.json"],
                    tree=root / "trees" / impl,
                    inputs={"predict_worker.py": root / "predict_worker.py",
                            "params.json": root / "params" / impl / f"{wid}.json",
                            "queries.json": root / "queries" / f"{wid}.json"},
                    devices=["nvidia"], timeout_seconds=RUN_TIMEOUT, path=os.environ["PATH"],
                    env={"PYTHONDONTWRITEBYTECODE": "1"})
                times[impl][wid].append(result.command_seconds if result.ok else None)
    out.write_text(json.dumps(times))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
