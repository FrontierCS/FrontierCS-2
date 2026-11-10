"""Judge of gpu_template_task for the shared verifier: evaluate(tree), on one H100 runner.

The whole verification runs on a runner of docker/verifier-gpu.Dockerfile (placement "runner"):
the shared verifier applies the patch there and calls evaluate(tree). Every implementation runs
through the shared isolate launcher (userns backend on the runner) with the GPU.

Score: raw_score = clamp(score_unbounded, 0, 100), score_unbounded = 50 + 25 * log2(S), where S
is the geometric mean over workloads of reference latency / candidate latency (50 at the
reference's speed; each doubling adds 25). A run that fails, exceeds its limit, returns the wrong
shape or a checked entry outside |out - exact| <= 1e-3 * (1 + |exact|) scores 0; the judge
recomputes the checked rows in double precision itself. Train measures the three public
workloads, final those plus two held-out ones.

Beat-spec metrics (beat_spec_workload_passes, one row per workload):
- latency_ms, lower is better: the smallest of three medians of ten CUDA-synchronized calls,
  each median from a fresh instance, candidate and reference interleaved; passes when
  0 < candidate < reference.
beats_reference = 1 iff every workload of the evaluated suite passes every metric; otherwise 0.
"""

# The contract with the shared verifier is the CPU template's (tasks/template/Theory/
# template_task/judge/evaluator.py). What a GPU judge adds:
#   - it runs on the runner as the Verifier GPU image's user, with the image's CUDA toolchain;
#   - isolate.run(..., devices=["nvidia"]) gives an instance the GPU; the worker interpreter is
#     the judge's own (sys.executable), which every instance may read;
#   - FRONTIER_VERIFIER_LOCAL_DEADLINE (a time.monotonic() value on the runner) is the end of
#     the runner's lease: a judge whose runs are long bounds itself by it;
#   - a run the runner loses is never a score: the shared GPU adapter reports it as an unscored
#     verifier_gpu_error.

from __future__ import annotations

import array
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import isolate  # the shared launcher, from gpu-tools (frontier.pth in the Verifier GPU image)

ROOT = Path(__file__).resolve().parent
PRISTINE = ROOT / "pristine"
WORKER = ROOT / "worker.py"
METRIC = "latency_ms"
REFERENCE_SCORE = 50.0
REPLICATES = 3
WARMUPS, TIMED = 3, 10
RUN_TIMEOUT = 600
SAMPLE_ROWS = 4
TOLERANCE = 1e-3

# Private suites: workload -> (N, M, D, seed). Train measures PUBLIC; final PUBLIC + HELD_OUT.
PUBLIC = {"a": (2048, 2048, 64, 11), "b": (4096, 1024, 128, 12), "c": (1024, 4096, 32, 13)}
HELD_OUT = {"d": (3072, 3072, 64, 21), "e": (2048, 2048, 256, 22)}


class CandidateFailure(Exception):
    """The candidate's run failed: a scored zero."""


def _inputs(scratch: Path, wid: str, n: int, m: int, d: int, seed: int) -> tuple[dict, list, list]:
    """The workload's float32 inputs as files (the instance's inputs) and as the float32-rounded
    values the judge recomputes from."""
    rng = random.Random(seed)
    x = array.array("f", (rng.gauss(0.0, 1.0) for _ in range(n * d)))
    y = array.array("f", (rng.gauss(0.0, 1.0) for _ in range(m * d)))
    files = {}
    for name, values in (("x.bin", x), ("y.bin", y)):
        path = scratch / f"{wid}-{name}"
        path.write_bytes(values.tobytes())
        files[name] = path
    return files, x, y


def _exact_rows(x, y, rows: list[int], m: int, d: int) -> list[list[float]]:
    exact = []
    for i in rows:
        xi = x[i * d:(i + 1) * d]
        exact.append([math.fsum((a - b) ** 2 for a, b in zip(xi, y[j * d:(j + 1) * d]))
                      for j in range(m)])
    return exact


def _measure(tree: Path, files: dict, shape: tuple[int, int, int], rows: list[int]) -> dict:
    """One fresh instance on the GPU: the worker times `tree`'s kernel and returns the rows."""
    n, m, d = shape
    result = isolate.run(
        [os.path.realpath(sys.executable), "{in}/worker.py", "{work}", "{in}/x.bin",
         "{in}/y.bin", str(n), str(m), str(d), json.dumps(rows), str(WARMUPS), str(TIMED),
         "{out}/result.json"],
        tree=tree, inputs={"worker.py": WORKER, **files}, outputs=["result.json"],
        devices=["nvidia"], timeout_seconds=RUN_TIMEOUT, path=os.environ["PATH"],
        env={"PYTHONDONTWRITEBYTECODE": "1", "TRITON_CACHE_DIR": "{tmp}/triton"})
    if result.timed_out or result.limit is not None:
        raise CandidateFailure(f"run exceeded its {result.limit or 'time'} limit")
    if not result.ok or not result.outputs.get("result.json"):
        raise CandidateFailure("run failed: " + result.stderr[-300:].decode(errors="replace"))
    return json.loads(result.outputs["result.json"])


def _check(row: dict, exact: list[list[float]], m: int) -> float:
    """The latency of a run whose checked rows are within tolerance."""
    latency, rows = row.get("latency_ms"), row.get("rows")
    if (isinstance(latency, bool) or not isinstance(latency, (int, float))
            or not math.isfinite(latency) or latency <= 0):
        raise CandidateFailure("invalid latency")
    if not isinstance(rows, list) or len(rows) != len(exact):
        raise CandidateFailure("wrong output rows")
    for got, want in zip(rows, exact):
        if not isinstance(got, list) or len(got) != m:
            raise CandidateFailure("wrong output shape")
        for value, truth in zip(got, want):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or abs(value - truth) > TOLERANCE * (1.0 + abs(truth))):
                raise CandidateFailure("an output entry is outside the tolerance")
    return float(latency)


def _reference_tree(scratch: Path) -> Path:
    tree = scratch / "reference"
    shutil.copytree(PRISTINE, tree)
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(scratch.resolve())}
    subprocess.run(["git", "apply", "--whitespace=nowarn", str(ROOT / "reference.patch")],
                   cwd=tree, check=True, capture_output=True, timeout=60, env=env)
    return tree


def evaluate(tree: str):
    tree = Path(tree)
    final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
    suite = {**PUBLIC, **HELD_OUT} if final else dict(PUBLIC)
    rejected = {"beats_reference": 0, "beat_spec_workload_passes": [
        {"workload": wid, "metrics": {METRIC: False}} for wid in suite],
        "reference_score": REFERENCE_SCORE, "margin": -REFERENCE_SCORE}
    picker = random.Random()  # the checked rows: unknown to the candidate's code in advance
    scratch = Path(tempfile.mkdtemp(prefix="gpu-template-judge-"))
    try:
        reference = _reference_tree(scratch)
        prepared = {}
        for wid, (n, m, d, seed) in suite.items():
            files, x, y = _inputs(scratch, wid, n, m, d, seed)
            rows = sorted(picker.sample(range(n), SAMPLE_ROWS))
            prepared[wid] = (files, (n, m, d), rows, _exact_rows(x, y, rows, m, d))
        cand, ref = {}, {}
        for _ in range(REPLICATES):          # interleave lanes so drift hits both alike
            for wid, (files, shape, rows, exact) in prepared.items():
                try:
                    c = _check(_measure(tree, files, shape, rows), exact, shape[1])
                except CandidateFailure as failure:
                    where = "a held-out" if wid in HELD_OUT else "a public"
                    return 0.0, 0.0, f"{failure} on {where} workload", rejected
                try:
                    r = _check(_measure(reference, files, shape, rows), exact, shape[1])
                except CandidateFailure as failure:
                    raise RuntimeError(f"the reference failed: {failure}") from failure
                cand.setdefault(wid, []).append(c)
                ref.setdefault(wid, []).append(r)
        c_min = {wid: min(v) for wid, v in cand.items()}
        r_min = {wid: min(v) for wid, v in ref.items()}
        decisions = [{"workload": wid, "metrics": {METRIC: 0 < c_min[wid] < r_min[wid]}}
                     for wid in suite]
        speedup = math.exp(sum(math.log(r_min[w] / c_min[w]) for w in suite) / len(suite))
        unbounded = 50.0 + 25.0 * math.log2(speedup)
        score = min(100.0, max(0.0, unbounded))
        metrics = {
            "beats_reference": int(all(d["metrics"][METRIC] for d in decisions)),
            "beat_spec_workload_passes": decisions,
            "reference_score": REFERENCE_SCORE,
            "margin": score - REFERENCE_SCORE,
            "latency_ms_candidate_by_workload": c_min,
            "latency_ms_reference_by_workload": r_min,
            # Candidate-only and sorted, without workload labels: task.toml [feedback].
            "latency_ms": sorted(c_min.values()),
        }
        return score, unbounded, f"geometric-mean speedup {speedup:.3f}", metrics
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
