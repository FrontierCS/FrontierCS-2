from __future__ import annotations

import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from submission_patch import apply_submission_patch


TASK_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((TASK_DIR / "config.yaml").read_text(encoding="utf-8"))
EVAL = CONFIG["evaluation"]
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


def _run_workload(repo: Path, workload: str) -> dict[str, float]:
    scheduler = repo / "scheduler"
    with tempfile.TemporaryDirectory(prefix="shockwave-workload-") as td:
        run_dir = Path(td)
        trace = run_dir / f"{workload}.trace"
        throughput = run_dir / "tacc_throughputs.json"
        config = run_dir / "shockwave_config.json"
        shutil.copyfile(TASK_DIR / "judge_data" / f"{workload}.trace", trace)
        shutil.copyfile(TASK_DIR / "judge_data" / "tacc_throughputs.json", throughput)
        shutil.copyfile(TASK_DIR / "judge_data" / "shockwave_config.json", config)
        output_dir = run_dir / "output"
        command = [
            sys.executable,
            ENTRYPOINT,
            "--trace_file", str(trace),
            "--policy", "shockwave",
            "--throughputs_file", str(throughput),
            "--cluster_spec", str(EVAL["cluster_spec"]),
            "--num_gpus_per_server", str(EVAL["num_gpus_per_server"]),
            "--seed", str(EVAL["seed"]),
            "--solver", "GUROBI",
            "--time_per_iteration", str(EVAL["time_per_iteration"]),
            "--config", str(config),
            "--pickle_output_dir", str(output_dir),
        ]
        env = os.environ.copy()
        env.update({
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        })
        try:
            proc = subprocess.run(
                command,
                cwd=scheduler,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=int(EVAL["workload_timeout_seconds"]),
            )
        except subprocess.TimeoutExpired as exc:
            raise EvaluationError("scheduler timed out") from exc
        if proc.returncode != 0:
            raise EvaluationError("scheduler exited unsuccessfully")
        result_path = output_dir / f"shockwave_{workload}_simulation.pickle"
        if not result_path.is_file():
            raise EvaluationError("scheduler produced no result")
        try:
            sys.path.insert(0, str(scheduler))
            result = pickle.loads(result_path.read_bytes())
        except Exception as exc:
            raise EvaluationError("scheduler produced malformed statistics") from exc
        finally:
            if sys.path and sys.path[0] == str(scheduler):
                sys.path.pop(0)
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


def _cleanup_patch_result(result: Any) -> None:
    workdir = getattr(result, "workdir", None)
    if workdir is not None:
        shutil.rmtree(Path(workdir).parent, ignore_errors=True)


def evaluate(solution_path: str) -> tuple[float, float, str, dict[str, Any]]:
    candidate = None
    reference = None
    try:
        final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
        suite = FINAL_SUITE if final else DEV_SUITE
        candidate = apply_submission_patch(solution_path, TASK_DIR, CONFIG)
        if not candidate.ok or candidate.workdir is None:
            return 0.0, 0.0, candidate.message, {}
        reference = apply_submission_patch(str(REFERENCE_PATCH), TASK_DIR, CONFIG)
        if not reference.ok or reference.workdir is None:
            return 0.0, 0.0, "withheld reference could not be prepared", {}
        candidate_rows = _run_suite(candidate.workdir, suite)
        reference_rows = _run_suite(reference.workdir, suite)
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
    except Exception as exc:
        return 0.0, 0.0, f"evaluation failed: {type(exc).__name__}", {}
    finally:
        if candidate is not None:
            _cleanup_patch_result(candidate)
        if reference is not None:
            _cleanup_patch_result(reference)


def prepare() -> dict[str, Any]:
    return {"execution": "original trace driver", "development_workloads": len(DEV_SUITE), "final_workloads": len(FINAL_SUITE)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"score": 0.0, "message": "usage: evaluator.py PATCH"}))
        raise SystemExit(2)
    score, unbounded, message, metrics = evaluate(sys.argv[1])
    print(json.dumps({"score": score, "score_unbounded": unbounded, "message": message, "metrics": metrics}, sort_keys=True))
