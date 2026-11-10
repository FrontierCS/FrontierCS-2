"""Judge of sandbox_template_task for the shared verifier: evaluate(tree), on the control
image, with two command test sandboxes (task.toml [test_sandboxes.*], access = "command").

1. Class 1, untrusted (`candidate`): two machines of the Agent GPU image at once, one with the
   submitted tree and one with the reference tree, each with the suite's public data (training
   points and query x) and judge/run_model.py, which calls the tree's model.fit and
   model.predict per workload within one 300-s step. The predictions and fitted parameters
   come back; nothing on those machines is trusted. The judge scores the predictions here,
   against the true curve values that never leave this image.
2. Class 2, trusted (`timing`): one machine of the Verifier GPU image gets
   judge/timing_harness.py, judge/predict_worker.py, both trees, their fitted parameters and the
   queries. The harness runs each implementation's predict in a fresh isolate instance with the
   GPU, three times per workload, interleaved, and returns the run times.

Score: raw_score = clamp(score_unbounded, 0, 100), score_unbounded = 50 + 25 * log2(G), where G
is the geometric mean over workloads of reference MSE / candidate MSE (50 at the reference's
error; halving the error adds 25). A candidate whose fit or predict fails, exceeds its limit,
loses its session or returns malformed output scores 0. Train measures three public curves,
final those plus two held-out ones.

Beat-spec metrics (beat_spec_workload_passes, one row per workload):
- mse, lower is better: mean squared error of the 500 predictions against the true curve;
  passes when candidate < reference.
- predict_seconds, lower is better: the smallest of three run times (process start to exit) of
  a fresh predict process; passes when candidate <= reference + 1.0.
beats_reference = 1 iff every workload of the evaluated suite passes every metric; otherwise 0.
"""

# The contract with the shared verifier is the CPU template's (tasks/template/Theory/
# template_task/judge/evaluator.py). What command sandboxes add (the module docstring of
# tasks/shared/verifier-gpu/fcs2_verifier_gpu.py is the contract):
#   - run_sandboxes([SandboxRun(sandbox, files, steps, outputs)], into=dir) leases one machine per
#     run, uploads exactly `files` as the sandbox's workdir, runs the steps there and returns the
#     declared outputs; a timeout, a non-zero exit or a lost session is a reported fact
#     (StepResult), never an exception. The judge classifies it: on the candidate's machine a
#     scored zero, on a trusted machine (or the reference's) the judge's own fault.
#   - outputs of an untrusted machine are untrusted input: validate their shape and score from
#     what this judge recomputes, never from a number the candidate's code reports.
#   - VerifierGpuError (a lost runner, an upload failure) is not a RuntimeError: let it
#     propagate; the shared verifier reports it as an unscored verifier_gpu_error.

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from fcs2_verifier_gpu import SandboxRun, SandboxStep, run_sandboxes, sandbox_workdir

ROOT = Path(__file__).resolve().parent
PRISTINE = ROOT / "pristine"
PYTHON = "/opt/frontier/runtime-python"   # both GPU images' interpreter
REFERENCE_SCORE = 50.0
FIT_SECONDS = 300
TIME_BAND_SECONDS = 1.0
TRAIN_POINTS, QUERY_POINTS, NOISE = 2000, 500, 0.1
MAX_PARAMS_BYTES = 1 << 20

# Private suites: workload -> (amplitude, frequency, phase, quadratic, seed) of
# y = amplitude * sin(frequency * x + phase) + quadratic * x^2 (+ noise in the training points).
PUBLIC = {"w1": (0.8, 3.0, 0.4, 0.5, 101), "w2": (1.2, 5.0, -0.7, -0.3, 102),
          "w3": (0.5, 8.0, 1.1, 0.8, 103)}
HELD_OUT = {"h1": (1.0, 6.0, 0.2, 0.4, 201), "h2": (0.7, 4.0, -1.3, -0.6, 202)}


class CandidateFailure(Exception):
    """The candidate's run failed or returned malformed output: a scored zero."""


def _curve(amplitude, frequency, phase, quadratic):
    return lambda x: amplitude * math.sin(frequency * x + phase) + quadratic * x * x


def _data(suite: dict) -> tuple[dict, dict]:
    """The public data of each workload (training points, query x) and its private targets."""
    public, targets = {}, {}
    for wid, (amplitude, frequency, phase, quadratic, seed) in suite.items():
        curve, rng = _curve(amplitude, frequency, phase, quadratic), random.Random(seed)
        xs = [rng.uniform(-1.0, 1.0) for _ in range(TRAIN_POINTS)]
        ys = [curve(x) + rng.gauss(0.0, NOISE) for x in xs]
        queries = [rng.uniform(-1.0, 1.0) for _ in range(QUERY_POINTS)]
        public[wid] = {"train": {"x": xs, "y": ys}, "queries": {"x": queries}}
        targets[wid] = [curve(x) for x in queries]
    return public, targets


def _writable(directory: Path) -> None:
    for path in (directory, *directory.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)


def _fit_run(tree: Path, public: dict, files: Path) -> SandboxRun:
    """Class 1: the tree, the public data and the driver go up; out/ comes back."""
    root = PurePosixPath(sandbox_workdir("candidate"))
    shutil.copytree(tree, files / "tree", symlinks=True)
    shutil.copy2(ROOT / "run_model.py", files / "run_model.py")
    # By index, not by workload id: nothing on the candidate's machine names a held-out curve.
    for index, data in enumerate(public.values()):
        (files / "data" / str(index)).mkdir(parents=True)
        for name in ("train", "queries"):
            (files / "data" / str(index) / f"{name}.json").write_text(json.dumps(data[name]))
    (files / "out").mkdir()
    _writable(files)
    step = SandboxStep([PYTHON, str(root / "run_model.py"), str(root / "tree"),
                        str(root / "data"), str(root / "out")], FIT_SECONDS,
                       {"PYTHONDONTWRITEBYTECODE": "1"})
    return SandboxRun("candidate", files, [step], ["out"])


def _fitted(result, public: dict) -> dict:
    """Per workload: the returned parameters (bytes, for the timing machine) and predictions,
    or CandidateFailure."""
    step = result.steps[0]
    if step.lost_session:
        raise CandidateFailure("the fit lost its session")
    if step.timed_out:
        raise CandidateFailure(f"the fit exceeded its {FIT_SECONDS}-second limit")
    if step.exit_code != 0:
        raise CandidateFailure(f"the fit exited with status {step.exit_code}")
    if result.outputs is None:
        raise CandidateFailure("the fit returned more than the output cap")
    fitted = {}
    for index, (wid, data) in enumerate(public.items()):
        folder = result.outputs / "out" / str(index)
        params, predictions = folder / "params.json", folder / "predictions.json"
        if (not params.is_file() or params.stat().st_size > MAX_PARAMS_BYTES
                or not predictions.is_file()):
            raise CandidateFailure("the fit did not return parameters and predictions for "
                                   "every workload")
        try:
            values = json.loads(predictions.read_text())
            json.loads(params.read_text())
        except ValueError as error:
            raise CandidateFailure("the fit returned malformed JSON") from error
        if (not isinstance(values, list) or len(values) != len(data["queries"]["x"])
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           and math.isfinite(v) for v in values)):
            raise CandidateFailure("the predictions are not one finite number per query")
        fitted[wid] = (params.read_bytes(), values)
    return fitted


def _mse(predictions: list[float], targets: list[float]) -> float:
    return math.fsum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(targets)


def _timing_run(trees: dict, fitted: dict, public: dict, files: Path) -> SandboxRun:
    """Class 2: the judge's own harness and worker, both trees, their parameters, the queries."""
    root = PurePosixPath(sandbox_workdir("timing"))
    for name in ("timing_harness.py", "predict_worker.py"):
        shutil.copy2(ROOT / name, files / name)
    for impl, tree in trees.items():
        shutil.copytree(tree, files / "trees" / impl, symlinks=True)
        for wid, (params, _predictions) in fitted[impl].items():
            (files / "params" / impl).mkdir(parents=True, exist_ok=True)
            (files / "params" / impl / f"{wid}.json").write_bytes(params)
    (files / "queries").mkdir()
    for wid, data in public.items():
        (files / "queries" / f"{wid}.json").write_text(json.dumps(data["queries"]))
    (files / "out").mkdir()
    _writable(files)
    step = SandboxStep([PYTHON, str(root / "timing_harness.py"), str(root),
                        ",".join(public), str(root / "out" / "timing.json")], None)
    return SandboxRun("timing", files, [step], ["out/timing.json"])


def _timings(result) -> dict:
    """The trusted harness's run times: impl -> workload -> [seconds | None]."""
    step = result.steps[0]
    if step.exit_code != 0 or result.outputs is None:
        raise RuntimeError(f"the trusted timing run failed: {step}: "
                           + (result.logs[0] or b"").decode(errors="replace")[-500:])
    return json.loads((result.outputs / "out" / "timing.json").read_text())


def _reference_tree(scratch: Path) -> Path:
    tree = scratch / "reference-tree"
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
        {"workload": wid, "metrics": {"mse": False, "predict_seconds": False}} for wid in suite],
        "reference_score": REFERENCE_SCORE, "margin": -REFERENCE_SCORE}
    public, targets = _data(suite)
    scratch = Path(tempfile.mkdtemp(prefix="sandbox-template-judge-"))
    try:
        trees = {"candidate": tree, "reference": _reference_tree(scratch)}

        # Class 1: both fits at once, each on its own machine.
        runs = []
        for impl, source in trees.items():
            (scratch / "fit" / impl).mkdir(parents=True)
            runs.append(_fit_run(source, public, scratch / "fit" / impl))
        (scratch / "fit-out").mkdir()
        candidate_fit, reference_fit = run_sandboxes(runs, into=scratch / "fit-out")
        try:
            reference_fitted = _fitted(reference_fit, public)
        except CandidateFailure as failure:
            raise RuntimeError(f"the reference fit failed: {failure}") from failure
        try:
            candidate_fitted = _fitted(candidate_fit, public)
        except CandidateFailure as failure:
            return 0.0, 0.0, str(failure), rejected
        fitted = {"candidate": candidate_fitted, "reference": reference_fitted}
        mse = {impl: {wid: _mse(fitted[impl][wid][1], targets[wid]) for wid in suite}
               for impl in trees}

        # Class 2: the trusted harness times both predicts on one machine.
        (scratch / "timing").mkdir()
        (scratch / "timing-out").mkdir()
        (timing,) = run_sandboxes([_timing_run(trees, fitted, public, scratch / "timing")],
                                  into=scratch / "timing-out")
        times = _timings(timing)
        if any(None in times["reference"][wid] for wid in suite):
            raise RuntimeError("the reference predict failed on the timing machine")
        if any(None in times["candidate"][wid] for wid in suite):
            reason = "the predict failed or exceeded its limit on the timing machine"
            return 0.0, 0.0, reason, rejected
        seconds = {impl: {wid: min(times[impl][wid]) for wid in suite} for impl in trees}

        decisions = [{"workload": wid, "metrics": {
            "mse": mse["candidate"][wid] < mse["reference"][wid],
            "predict_seconds": seconds["candidate"][wid]
            <= seconds["reference"][wid] + TIME_BAND_SECONDS}} for wid in suite]
        gain = math.exp(sum(math.log(mse["reference"][w] / mse["candidate"][w]) for w in suite)
                        / len(suite))
        unbounded = 50.0 + 25.0 * math.log2(gain)
        score = min(100.0, max(0.0, unbounded))
        metrics = {
            "beats_reference": int(all(all(d["metrics"].values()) for d in decisions)),
            "beat_spec_workload_passes": decisions,
            "reference_score": REFERENCE_SCORE,
            "margin": score - REFERENCE_SCORE,
            "mse_by_workload": mse,
            "predict_seconds_by_workload": seconds,
            # Candidate-only and sorted, without workload labels: task.toml [feedback].
            "mse": sorted(mse["candidate"].values()),
        }
        return score, unbounded, f"geometric-mean error ratio {gain:.3f}", metrics
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
