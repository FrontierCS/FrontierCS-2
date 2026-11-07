"""Shockwave dynamic GPU allocation judge: the trace-driven Gavel simulator on the candidate.

The shared verifier applies the patch (task.toml [submission]) and calls evaluate(tree). Each
trace runs in a fresh shared isolate instance. A build, run, timeout or output-validity
failure is a bare scored zero, as before.

Score: per trace w (two per suite), with makespan M, average JCT J, worst finish-time-fairness
rho R and unfair-job percentage U (rho > 1.05):
score_w = 40 * 12000 / (12000 + M) + 40 * 7000 / (7000 + J) + 15 * 0.55 / (R + 0.05)
          + 5 * (1 - U / 100)
raw_score = score_unbounded = mean over the traces (constants: task.toml [judge.config]).

Metrics, one row per trace in beat_spec_workload_passes, each lower is better and passes when
the candidate is strictly below the reference (pristine + reference.patch) on that trace:
makespan_seconds, average_jct_seconds, worst_finish_time_fairness_rho,
unfair_job_fraction_percent. beats_reference = 1 iff every trace passes all four. Train also
returns the candidate's four values per trace (anonymous ordered arrays).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

import isolate  # the shared launcher, beside edit_scope in the Verifier base

TASK_DIR = Path(__file__).resolve().parent
# config.yaml is generated from task.toml; the simulation and score constants are its
# [judge.config.evaluation] keys.
CONFIG = yaml.safe_load((TASK_DIR / "config.yaml").read_text(encoding="utf-8"))
EVAL = CONFIG["evaluation"]
PRISTINE = TASK_DIR / "pristine"  # the starter repo; the shared verifier overlays the submission on a copy
REFERENCE_PATCH = TASK_DIR / "reference.patch"
ENTRYPOINT = "scripts/drivers/simulate_scheduler_with_trace.py"
SPEC_METRICS = (
    "makespan_seconds",
    "average_jct_seconds",
    "worst_finish_time_fairness_rho",
    "unfair_job_fraction_percent",
)
DEV_SUITE = ("dev_a", "dev_b")
FINAL_SUITE = ("final_a", "final_b")
class EvaluationError(RuntimeError):
    pass


def _build(repo: Path) -> None:
    scheduler = repo / "scheduler"
    protos = [
        "runtime/protobuf/iterator_to_scheduler.proto",
        "runtime/protobuf/scheduler_to_worker.proto",
        "runtime/protobuf/worker_to_scheduler.proto",
        "runtime/protobuf/common.proto",
        "runtime/protobuf/enums.proto",
    ]
    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        "-Iruntime/protobuf",
        "--python_out=runtime/rpc_stubs",
        "--grpc_python_out=runtime/rpc_stubs",
        *protos,
    ]
    try:
        subprocess.run(
            command,
            cwd=scheduler,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except Exception as exc:
        raise EvaluationError("the patched tree did not build") from exc


def _workload_layout(workload: str) -> tuple[dict[int, int], int]:
    trace_path = TASK_DIR / "judge_data" / f"{workload}.trace"
    try:
        records = [line.split("\t") for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
        demands = {job_id: int(fields[6]) for job_id, fields in enumerate(records)}
        worker_count = int(str(EVAL["cluster_spec"]).split(":", 1)[0])
    except (OSError, ValueError, IndexError, KeyError) as exc:
        raise EvaluationError("private workload metadata is malformed") from exc
    if not demands or worker_count < 1 or any(demand < 1 or demand > worker_count for demand in demands.values()):
        raise EvaluationError("private workload metadata is invalid")
    return demands, worker_count


def _result_metrics(result: Any, workload: str) -> dict[str, float]:
    demands, worker_count = _workload_layout(workload)
    try:
        makespan = float(result["makespan"])
        average_jct = float(result["avg_jct"])
        rhos = [float(x) for x in result["finish_time_fairness_list"]]
        jcts = [float(x) for x in result["jct_list"]]
        schedules = result["per_round_schedule"]
    except Exception as exc:
        raise EvaluationError("scheduler produced malformed statistics") from exc
    if len(rhos) != len(demands) or len(jcts) != len(demands) or not schedules:
        raise EvaluationError("scheduler did not complete every job")
    for schedule in schedules:
        if not isinstance(schedule, dict) or not schedule:
            raise EvaluationError("scheduler emitted an empty allocation round")
        workers: list[int] = []
        for job_id, assigned in schedule.items():
            if type(job_id) is not int or job_id not in demands:
                raise EvaluationError("scheduler emitted an invalid job identity")
            if not isinstance(assigned, (tuple, list)) or len(assigned) != demands[job_id]:
                raise EvaluationError("scheduler emitted an impossible worker allocation")
            if any(type(worker) is not int or worker < 0 or worker >= worker_count for worker in assigned):
                raise EvaluationError("scheduler emitted an invalid worker identity")
            workers.extend(assigned)
        if len(workers) > worker_count or len(workers) != len(set(workers)):
            raise EvaluationError("scheduler emitted an invalid GPU allocation")
    values = [makespan, average_jct, *rhos, *jcts]
    if not all(math.isfinite(x) and x >= 0.0 for x in values):
        raise EvaluationError("scheduler produced invalid numeric statistics")
    unfair_threshold = float(EVAL["unfair_threshold_rho"])
    unfair = round(
        100.0 * sum(rho > unfair_threshold for rho in rhos) / len(rhos), 2)

    return {
        "makespan_seconds": makespan,
        "average_jct_seconds": average_jct,
        "worst_finish_time_fairness_rho": max(rhos),
        "unfair_job_fraction_percent": unfair,
    }


# Runs inside its isolate instance: the trace driver (submitted scheduler code) as a child, then its
# result pickle turned into plain JSON there, so the judge never unpickles submitted objects.
# Types the judge checks (job and worker identities) are kept as ints or sent as null.
RUNNER = r'''
import json, pickle, shutil, subprocess, sys
pickle_path, inputs, scratch, command = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
# The driver writes beside its trace, so it reads writable copies of the read-only inputs.
shutil.copytree(inputs, scratch, dirs_exist_ok=True)
def report(**value):
    sys.stdout.write(json.dumps(value) + "\n")
    raise SystemExit(0)
code = subprocess.run([sys.executable, *command], stdin=subprocess.DEVNULL,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
if code != 0:
    report(status="exit")
try:
    data = open(pickle_path, "rb").read()
except OSError:
    report(status="no_result")
try:
    sys.path.insert(0, ".")
    result = pickle.loads(data)
    def identity(value):
        return value if type(value) is int else None
    def schedule(round_):
        if not isinstance(round_, dict):
            return None
        return [[identity(job), [identity(worker) for worker in assigned]
                 if isinstance(assigned, (tuple, list)) else None]
                for job, assigned in round_.items()]
    report(status="ok", result={
        "makespan": float(result["makespan"]), "avg_jct": float(result["avg_jct"]),
        "finish_time_fairness_list": [float(x) for x in result["finish_time_fairness_list"]],
        "jct_list": [float(x) for x in result["jct_list"]],
        "per_round_schedule": [schedule(round_) for round_ in result["per_round_schedule"]]})
except Exception:
    report(status="malformed")
'''


def _run_workload(repo: Path, workload: str) -> dict[str, float]:
    """One trace through the original driver, in a fresh isolate instance (the shared
    launcher: no judge files, no FRONTIER_* environment, no network, nothing kept)."""
    result_name = f"shockwave_{workload}_simulation.pickle"
    command = [
        ENTRYPOINT,
        "--trace_file", f"{{tmp}}/{workload}.trace",
        "--policy", "shockwave",
        "--throughputs_file", "{tmp}/tacc_throughputs.json",
        "--cluster_spec", str(EVAL["cluster_spec"]),
        "--num_gpus_per_server", str(EVAL["num_gpus_per_server"]),
        "--seed", str(EVAL["seed"]),
        "--solver", "GUROBI",
        "--time_per_iteration", str(EVAL["time_per_iteration"]),
        "--config", "{tmp}/shockwave_config.json",
        "--pickle_output_dir", "{out}",
    ]
    data = TASK_DIR / "judge_data"
    run = isolate.run(
        [sys.executable, "-c", RUNNER, "{out}/" + result_name, "{in}", "{tmp}", *command],
        timeout_seconds=int(EVAL["workload_timeout_seconds"]), tree=repo,
        cwd="{work}/scheduler",
        inputs={f"{workload}.trace": data / f"{workload}.trace",
                "tacc_throughputs.json": data / "tacc_throughputs.json",
                "shockwave_config.json": data / "shockwave_config.json"},
        env={"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
             "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    if run.timed_out:
        raise EvaluationError("scheduler timed out")
    try:
        report = json.loads(run.stdout.decode().strip().splitlines()[-1])
        status = report["status"]
    except (ValueError, IndexError, KeyError, TypeError, AttributeError):
        status = "exit"
    if run.returncode != 0 or run.limit or status == "exit":
        raise EvaluationError("scheduler exited unsuccessfully")
    if status == "no_result":
        raise EvaluationError("scheduler produced no result")
    if status != "ok":
        raise EvaluationError("scheduler produced malformed statistics")
    result = report["result"]
    try:
        result["per_round_schedule"] = [
            round_ if round_ is None else {job: tuple(assigned) if assigned is not None else None
                                           for job, assigned in round_}
            for round_ in result["per_round_schedule"]]
    except (TypeError, ValueError) as exc:
        raise EvaluationError("scheduler produced malformed statistics") from exc
    return _result_metrics(result, workload)


def _run_suite(repo: Path, suite: tuple[str, ...]) -> dict[str, dict[str, float]]:
    _build(repo)
    return {workload: _run_workload(repo, workload) for workload in suite}


def _score(rows: dict[str, dict[str, float]]) -> float:
    scale_m = float(EVAL["makespan_scale_seconds"])
    scale_j = float(EVAL["average_jct_scale_seconds"])
    rho_numerator = float(EVAL["fairness_numerator_rho"])
    rho_offset = float(EVAL["fairness_offset_rho"])
    workload_scores = []
    for row in rows.values():
        workload_scores.append(
            float(EVAL["makespan_weight"]) * scale_m
            / (scale_m + row["makespan_seconds"])
            + float(EVAL["average_jct_weight"]) * scale_j
            / (scale_j + row["average_jct_seconds"])
            + float(EVAL["worst_fairness_weight"]) * rho_numerator
            / (row["worst_finish_time_fairness_rho"] + rho_offset)
            + float(EVAL["unfair_fraction_weight"])
            * (1.0 - row["unfair_job_fraction_percent"] / 100.0)
        )
    return sum(workload_scores) / len(workload_scores)


def _metric_columns(rows: dict[str, dict[str, float]]) -> dict[str, list[float]]:
    return {
        metric: [row[metric] for row in rows.values()]
        for metric in SPEC_METRICS
    }


def _all_comparisons_beat(passes: list[dict[str, Any]]) -> int:
    """Require every declared metric to beat reference on every workload."""
    if len(passes) != len(DEV_SUITE):
        return 0
    if any(set(row.get("metrics", {})) != set(SPEC_METRICS) for row in passes):
        return 0
    decisions = [row["metrics"] for row in passes]
    return int(all(value for row in decisions for value in row.values()))


def _reference_tree(scratch: Path) -> Path:
    """The pristine starter with reference.patch applied (trusted, judge-owned)."""
    tree = scratch / "reference"
    shutil.copytree(PRISTINE, tree, symlinks=True)
    for path in (tree, *tree.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(scratch.resolve())}
    subprocess.run(["git", "apply", "--whitespace=nowarn", str(REFERENCE_PATCH)], cwd=tree,
                   check=True, capture_output=True, timeout=60, env=env)
    return tree


def evaluate(tree: str) -> tuple[float, float, str, dict[str, Any]]:
    """Score the judged tree: the starter with the submission's allowed files overlaid (the
    shared verifier has applied the patch). A build, run or output failure is a scored zero
    with no metrics, as before."""
    scratch = Path(tempfile.mkdtemp(prefix="shockwave-reference-"))
    try:
        final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
        suite = FINAL_SUITE if final else DEV_SUITE
        try:
            reference = _reference_tree(scratch)
        except (OSError, subprocess.SubprocessError):
            return 0.0, 0.0, "withheld reference could not be prepared", {}
        candidate_rows = _run_suite(Path(tree), suite)
        reference_rows = _run_suite(reference, suite)
        score = _score(candidate_rows)
        reference_score = _score(reference_rows)
        passes = []
        for workload in suite:
            decisions = {
                metric: bool(candidate_rows[workload][metric] < reference_rows[workload][metric])
                for metric in SPEC_METRICS
            }
            passes.append({"workload": workload, "metrics": decisions})
        beats = _all_comparisons_beat(passes)
        metrics: dict[str, Any] = {
            **_metric_columns(candidate_rows),
            # Verifier-only audit fields required by beat_spec.json. The judge's
            # solver_visible_metrics allowlist removes all four before replying
            # to an agent and derives only anonymous aggregate progress counts.
            "beat_spec_workload_passes": passes,
            "beats_reference": beats,
            "reference_score": reference_score,
            "margin": score - reference_score,
        }
        return score, score, "scheduler completed the authentic trace suite", metrics
    except EvaluationError as exc:
        return 0.0, 0.0, str(exc), {}
    except isolate.IsolateError:
        raise  # a judging fault, not a candidate verdict
    except Exception as exc:
        return 0.0, 0.0, f"evaluation failed: {type(exc).__name__}", {}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def prepare() -> dict[str, Any]:
    return {"execution": "original trace driver", "development_workloads": len(DEV_SUITE), "final_workloads": len(FINAL_SUITE)}
