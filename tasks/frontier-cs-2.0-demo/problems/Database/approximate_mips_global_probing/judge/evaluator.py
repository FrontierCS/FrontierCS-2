"""Evaluator for implementing the paper-aligned FARGO query core to the authors' FARGO source."""
from __future__ import annotations

import math
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from submission_patch import apply_submission_patch

CONFIG = {
    "submission": {
        "allow_paths": ["src/alg.cpp"],
        "max_patch_bytes": 65536,
    }
}
DATASETS = ("audio", "mnist")
DEV_SUITE = (0,)
FINAL_SUITE = (0, 1)
APPROXIMATION_RATIOS = (0.6, 0.8, 0.9)
CAPACITIES = (1, 10, 50, 100)
RUN_TIMEOUT = 180
BUILD_TIMEOUT = 120
HASH_SEED = 314159
FORBIDDEN_POLICY_TOKENS = (
    "audio", ".ben", "fargo_metric", "/proc/", "cmdline",
    "judge_data", "frontier_submission", "submission_role", "preprocess.h", "mf_alsh.h", "performance.h",
)


DRIVER = r'''#include "Preprocess.h"
#include "mf_alsh.h"
#include "performance.h"
#include "basis.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 6) return 2;
    std::string data = argv[1], ben = argv[2], table = argv[3];
    int split = atoi(argv[4]), qnum = atoi(argv[5]);
    Preprocess prep(data, ben);
    Partition parti(0.8f, prep);
    Parameter param(prep, 5, 12, 1);
    mf_alsh::Hash index(prep, param, "unused.index", parti, table);
    const float ratios[] = {0.6f, 0.8f, 0.9f};
    const int capacities[] = {1, 10, 50, 100};
    for (float c : ratios) {
        for (int b : capacities) {
            Performance<mf_alsh::Query> perf;
            for (int q = 0; q < qnum; ++q) {
                int query_id = 2 * q + split;
                mf_alsh::Query query(query_id, c, b, index, prep, b);
                perf.update(query, prep);
            }
            std::cout << "FARGO_METRIC " << c << " " << b << " " << std::setprecision(9)
                      << (1000.0 * perf.time_total / perf.num) << " "
                      << ((double)perf.NN_num / (perf.num * b)) << " "
                      << ((double)perf.ratio / perf.res_num) << " "
                      << ((double)perf.kRatio / perf.num) << " "
                      << ((double)perf.cost / perf.num) << "\n";
        }
    }
    return 0;
}
'''


class ReferenceFailure(RuntimeError):
    """The paired reference did not build or run.

    An evaluator error, never a candidate result: a failed reference cannot set
    a bar and the candidate is not scored against it.
    """


def _invalid(reason: str):
    points = len(DATASETS) * len(APPROXIMATION_RATIOS) * len(CAPACITIES)
    if os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final":
        points *= 2
    return 0.0, 0.0, reason, {
        "beats_reference": 0, "reference_score": 0.0, "margin": 0.0,
        "mean_recall": 0.0, "mean_ratio": 0.0, "mean_kratio": 0.0,
        "mean_time_ms": 0.0, "mean_candidates": 0.0,
        "beat_spec_workload_passes": [
            {"workload": str(i), "metrics": {"recall": False, "overall_ratio": False,
                                             "query_time_ms": False,
                                             "candidates_verified": False}}
            for i in range(points)],
    }


def _query_count() -> int:
    return 12 if os.environ.get("FRONTIER_TEST_FAST") == "1" else 150


def _instrument_source(app: Path, split: int) -> None:
    seed = HASH_SEED if split == 0 else 271828
    source = app / "src" / "mf_alsh.cpp"
    text = source.read_text(encoding="utf-8-sig")
    text = text.replace("std::mt19937 rng(int(std::time(0)));", f"std::mt19937 rng({seed});")
    if text.count(f"std::mt19937 rng({seed});") != 2:
        raise RuntimeError("deterministic seed instrumentation failed")
    source.write_text(text, encoding="utf-8")
    (app / "src" / "main.cpp").write_text(DRIVER, encoding="utf-8")


def _build_and_run(app: Path, case_root: Path, split: int, dataset: str = "audio") -> list[dict]:
    _instrument_source(app, split)
    case_root.mkdir(parents=True, exist_ok=True)
    table = case_root / "calibration.bin"
    shutil.copy2(TASK_DIR / "judge_data" / "squared_qd_cdf.bin", table)
    if dataset not in DATASETS:
        raise ValueError("unknown dataset")
    source = TASK_DIR / "judge_data" / (dataset + ".data_new")
    target = case_root / (dataset + ".data_new")
    if source.is_file():
        shutil.copy2(source, target)
    else:
        with gzip.open(str(source) + ".gz", "rb") as inp, target.open("wb") as out:
            shutil.copyfileobj(inp, out)
    shutil.copy2(TASK_DIR / "judge_data" / (dataset + ".ben"), case_root / (dataset + ".ben"))
    data, ben, qnum = case_root / (dataset + ".data"), case_root / (dataset + ".ben"), _query_count()
    built = subprocess.run(
        ["make", "-j2"], cwd=app, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=BUILD_TIMEOUT,
    )
    if built.returncode != 0:
        raise RuntimeError("submission does not build")
    proc = subprocess.run(
        [str(app / "fg"), str(data), str(ben), str(table), str(split), str(qnum)],
        cwd=app, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        timeout=RUN_TIMEOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError("submission failed while running the workload")
    rows = []
    for line in proc.stdout.splitlines():
        if line.startswith("FARGO_METRIC "):
            parts = line.split()
            if len(parts) != 8:
                raise RuntimeError("workload produced malformed metrics")
            rows.append({
                "c": float(parts[1]), "capacity": int(parts[2]),
                "time_ms": float(parts[3]), "recall": float(parts[4]),
                "ratio": float(parts[5]), "kratio": float(parts[6]),
                "cost": float(parts[7]),
            })
    expected_pairs = [(c, capacity) for c in APPROXIMATION_RATIOS for capacity in CAPACITIES]
    if len(rows) != len(expected_pairs) or any(
        abs(row["c"] - expected_c) > 1e-6 or row["capacity"] != expected_capacity
        for row, (expected_c, expected_capacity) in zip(rows, expected_pairs)
    ):
        raise RuntimeError("workload did not produce every required metric record")
    if any(not math.isfinite(v) for r in rows for v in r.values()):
        raise RuntimeError("workload did not produce complete finite metrics")
    if any(
        r["time_ms"] <= 0.0 or r["cost"] < 0.0
        or not 0.0 <= r["recall"] <= 1.0
        or not 0.0 <= r["kratio"] <= 1.01
        or not 0.0 <= r["ratio"] <= 1.01
        for r in rows
    ):
        raise RuntimeError("workload produced out-of-range metrics")
    return rows


def _apply(patch: Path):
    result = apply_submission_patch(str(patch), TASK_DIR, CONFIG)
    if not result.ok or not result.workdir:
        raise ValueError(result.message)
    app = Path(result.workdir)
    source = (app / "src" / "alg.cpp").read_text(encoding="utf-8", errors="replace").lower()
    if any(token in source for token in FORBIDDEN_POLICY_TOKENS):
        shutil.rmtree(app, ignore_errors=True)
        raise ValueError("policy contains a forbidden benchmark or process-inspection token")
    try:
        binary = app / "conformance"
        subprocess.run(["g++", "-std=c++11", "-O2", "-I", str(app / "src"),
                        str(TASK_DIR / "conformance.cpp"), str(app / "src/alg.cpp"),
                        "-o", str(binary)], check=True, capture_output=True, timeout=BUILD_TIMEOUT)
        checked = subprocess.run([str(binary)], check=True, capture_output=True,
                                 text=True, timeout=15)
        if checked.stdout.strip() != "conformance cases=108 ordered=0 PASS":
            raise ValueError("query-core conformance missing")
    except Exception as exc:
        shutil.rmtree(app, ignore_errors=True)
        raise ValueError("query-core conformance failed") from exc
    return app


def _raw_score(rows: list[dict], naive: list[dict]) -> float:
    point_scores = []
    for row, base in zip(rows, naive):
        quality = math.sqrt(max(0.0, min(1.0, row["recall"])) * max(0.0, min(1.0, row["ratio"])))
        time_gain = base["time_ms"] / max(1e-9, row["time_ms"])
        work_gain = base["cost"] / max(1.0, row["cost"])
        efficiency = (time_gain ** 0.25) * (work_gain ** 0.75)
        point_scores.append(100.0 * quality * efficiency / (1.0 + efficiency))
    return sum(point_scores) / len(point_scores)


def _paired_scores(patch: Path, splits: tuple[int, ...]) -> tuple[float, float, dict]:
    candidate_scores, reference_scores, diagnostics = [], [], []
    comparisons = []
    same_as_reference = patch.read_bytes() == (TASK_DIR / "reference.patch").read_bytes()
    for dataset, split in ((dataset, split) for dataset in DATASETS for split in splits):
        trial = Path(tempfile.mkdtemp(prefix="fargo_pair_"))
        apps = []
        try:
            candidate_app = _apply(patch)
            apps.append(candidate_app)
            naive_app = trial / "naive"
            shutil.copytree(TASK_DIR / "harbor" / "app", naive_app, ignore=shutil.ignore_patterns(".git"))
            naive = _build_and_run(naive_app, trial / "naive_case", split, dataset)
            candidate = _build_and_run(candidate_app, trial / "candidate_case", split, dataset)
            if same_as_reference:
                reference = candidate
            else:
                try:
                    reference_app = _apply(TASK_DIR / "reference.patch")
                    apps.append(reference_app)
                    reference = _build_and_run(reference_app, trial / "reference_case", split, dataset)
                except Exception as error:
                    raise ReferenceFailure(f"reference run failed: {type(error).__name__}") from error
            comparisons.extend({
                "recall": row["recall"] >= ref["recall"],
                "overall_ratio": row["ratio"] >= ref["ratio"],
                "query_time_ms": row["time_ms"] < ref["time_ms"],
                # Beat spec v3: harness-counted verified candidates (paper
                # Theorem 1 query cost; Section 6.2.2-6.2.3 compare at equal
                # candidate counts). Recall cannot be bought with extra work.
                "candidates_verified": row["cost"] <= ref["cost"],
            } for row, ref in zip(candidate, reference))
            candidate_scores.append(_raw_score(candidate, naive))
            reference_scores.append(_raw_score(reference, naive))
            diagnostics.append({
                "mean_recall": sum(r["recall"] for r in candidate) / len(candidate),
                "mean_ratio": sum(r["ratio"] for r in candidate) / len(candidate),
                "mean_kratio": sum(r["kratio"] for r in candidate) / len(candidate),
                "mean_time_ms": sum(r["time_ms"] for r in candidate) / len(candidate),
                "mean_candidates": sum(r["cost"] for r in candidate) / len(candidate),
            })
        finally:
            for app in apps:
                shutil.rmtree(app, ignore_errors=True)
            shutil.rmtree(trial, ignore_errors=True)
    return (
        sum(candidate_scores) / len(candidate_scores),
        sum(reference_scores) / len(reference_scores),
        {**{key: sum(d[key] for d in diagnostics) / len(diagnostics)
              for key in diagnostics[0]},
         "_comparisons": comparisons},
    )


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    try:
        patch = Path(solution_path)
        if not patch.is_file():
            return _invalid("submission patch does not exist")
        final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
        suite = FINAL_SUITE if final else DEV_SUITE
        candidate_score, reference_score, public = _paired_scores(patch, suite)
        candidate_score = max(0.0, min(100.0, candidate_score))
        reference_score = max(0.0, min(100.0, reference_score))
        tol = 1e-9 * max(1.0, abs(reference_score))
        comparisons = public.pop("_comparisons")
        metrics = {
            "beats_reference": int(all(all(row.values()) for row in comparisons)),
            "beat_spec_workload_passes": [{"workload": str(i), "metrics": row}
                                           for i, row in enumerate(comparisons)],
            "reference_score": reference_score,
            "margin": candidate_score - reference_score,
            **public,
        }
        message = (
            f"score={candidate_score:.4f}; recall={public['mean_recall']:.4f}; "
            f"k-ratio={public['mean_kratio']:.4f}; query_ms={public['mean_time_ms']:.4f}; "
            f"candidates={public['mean_candidates']:.1f}"
        )
        return candidate_score, candidate_score, message, metrics
    except ReferenceFailure:
        raise
    except subprocess.TimeoutExpired:
        return _invalid("submission exceeded the evaluation timeout")
    except (ValueError, RuntimeError) as exc:
        return _invalid(str(exc))
    except Exception:
        return _invalid("submission could not be evaluated")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(TASK_DIR / "reference.patch")
    print(evaluate(target))
