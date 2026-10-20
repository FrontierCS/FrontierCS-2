"""Evaluator for restoring per-query stopping to the authors' FARGO source."""
from __future__ import annotations

import math
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
DEV_SUITE = (0,)
FINAL_SUITE = (0, 1)
SUITE_NAMES = {0: "public", 1: "hidden"}
APPROXIMATION_RATIOS = (0.6, 0.8, 0.9)
CAPACITIES = (100, 200, 400, 800)
# Final additionally measures these heap capacities. They are neither in the
# statement nor in any development feedback, and their workload names are
# opaque, so the values exist only in this judge-side file. They sit between
# and slightly beyond the disclosed capacities (geometric midpoints).
FINAL_ONLY_CAPACITIES = (140, 280, 560, 1120)
# On those capacities PASS uses the deterministic comparisons only. Query time
# is still measured and recorded there, but its strict comparison is applied
# on the disclosed grid alone.
FINAL_ONLY_METRICS = (
    "recall_at_50", "last_result_inner_product_ratio", "verified_candidate_count")
RUN_TIMEOUT = 180
BUILD_TIMEOUT = 120
HASH_SEED = 314159
FORBIDDEN_POLICY_TOKENS = (
    "audio", ".ben", "fargo_metric", "/proc/", "cmdline",
    "judge_data", "frontier_submission", "submission_role",
)


class WorkloadTimeout(RuntimeError):
    """A bounded diagnostic that does not reveal suite membership or data."""

    def __init__(self, process_role: str, suite_index: int):
        super().__init__(
            f"workload process timed out (process_role={process_role}; "
            f"suite_index={suite_index})"
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
    const int capacities[] = {__CAPACITIES__};
    for (float c : ratios) {
        for (int b : capacities) {
            Performance<mf_alsh::Query> perf;
            for (int q = 0; q < qnum; ++q) {
                int query_id = 2 * q + split;
                mf_alsh::Query query(query_id, c, 50, index, prep, b);
                perf.update(query, prep);
            }
            std::cout << "FARGO_METRIC " << c << " " << b << " " << std::setprecision(9)
                      << (1000.0 * perf.time_total / perf.num) << " "
                      << ((double)perf.NN_num / (perf.num * 50)) << " "
                      << ((double)perf.ratio / perf.res_num) << " "
                      << ((double)perf.kRatio / perf.num) << " "
                      << ((double)perf.cost / perf.num) << "\n";
        }
    }
    return 0;
}
'''


def _is_final() -> bool:
    return os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"


def _expected_pairs(capacities: tuple[int, ...] = CAPACITIES) -> list[tuple[float, int]]:
    return [(c, capacity) for c in APPROXIMATION_RATIOS for capacity in capacities]


def _driver(capacities: tuple[int, ...] = CAPACITIES) -> str:
    return DRIVER.replace("__CAPACITIES__", ", ".join(str(int(b)) for b in capacities))


def _invalid(reason: str):
    workload_passes = [
        {
            "workload": _workload_name({"c": c, "capacity": capacity}),
            "metrics": {
                name: False
                for name in (
                    "mean_query_time", "recall_at_50",
                    "last_result_inner_product_ratio", "verified_candidate_count")
                if capacity in CAPACITIES or name in FINAL_ONLY_METRICS
            },
        }
        for c, capacity in _expected_pairs() + (
            _expected_pairs(FINAL_ONLY_CAPACITIES) if _is_final() else [])
    ]
    return 0.0, 0.0, reason, {
        "beats_reference": 0,
        "reference_score": 0.0,
        "margin": 0.0,
        "beat_spec_workload_passes": workload_passes,
    }


def _query_count() -> int:
    return 12 if os.environ.get("FRONTIER_TEST_FAST") == "1" else 150


def _instrument_source(app: Path, capacities: tuple[int, ...] = CAPACITIES) -> None:
    source = app / "src" / "mf_alsh.cpp"
    text = source.read_text(encoding="utf-8-sig")
    text = text.replace("std::mt19937 rng(int(std::time(0)));", f"std::mt19937 rng({HASH_SEED});")
    if text.count(f"std::mt19937 rng({HASH_SEED});") != 2:
        raise RuntimeError("deterministic seed instrumentation failed")
    source.write_text(text, encoding="utf-8")
    (app / "src" / "main.cpp").write_text(_driver(capacities), encoding="utf-8")


def _build_and_run(
        app: Path, case_root: Path, split: int, *, process_role: str,
        suite_index: int, capacities: tuple[int, ...] = CAPACITIES) -> list[dict]:
    if capacities == CAPACITIES:
        _instrument_source(app)
    else:
        # A second process of an already measured tree: only the driver's
        # capacity list differs, so make rebuilds the driver alone.
        _instrument_source(app, capacities)
    case_root.mkdir(parents=True, exist_ok=True)
    table = case_root / "calibration.bin"
    shutil.copy2(TASK_DIR / "judge_data" / "MyfunctionXTheta.data", table)
    shutil.copy2(TASK_DIR / "judge_data" / "audio.data_new", case_root / "audio.data_new")
    shutil.copy2(TASK_DIR / "judge_data" / "audio.ben", case_root / "audio.ben")
    data, ben, qnum = case_root / "audio.data", case_root / "audio.ben", _query_count()
    built = subprocess.run(
        ["make", "-j2"], cwd=app, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=BUILD_TIMEOUT,
    )
    if built.returncode != 0:
        raise RuntimeError("submission does not build")
    try:
        proc = subprocess.run(
            [str(app / "fg"), str(data), str(ben), str(table), str(split), str(qnum)],
            cwd=app, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired as error:
        raise WorkloadTimeout(process_role, suite_index) from error
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
                "cost": float(parts[7]), "query_count": qnum,
            })
    expected_pairs = _expected_pairs(capacities)
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
    return app


def _raw_score(rows: list[dict], naive: list[dict]) -> float:
    point_scores = []
    for row, base in zip(rows, naive):
        quality = math.sqrt(max(0.0, min(1.0, row["recall"])) * max(0.0, min(1.0, row["kratio"])))
        time_gain = max(0.25, base["time_ms"] / max(1e-9, row["time_ms"]))
        work_gain = max(0.25, base["cost"] / max(1.0, row["cost"]))
        efficiency = (time_gain ** 0.25) * (work_gain ** 0.75)
        point_scores.append(100.0 * quality * efficiency / (1.0 + efficiency))
    return sum(point_scores) / len(point_scores)


def _workload_name(row: dict) -> str:
    if row["capacity"] in FINAL_ONLY_CAPACITIES:
        # A final-only capacity is named by position, never by value.
        position = FINAL_ONLY_CAPACITIES.index(row["capacity"]) + 1
        return f"c={row['c']:.1f},capacity=final-only-{position}"
    return f"c={row['c']:.1f},capacity={row['capacity']}"


def _combine_suite_rows(
        suite_rows: list[list[dict]], capacities: tuple[int, ...] = CAPACITIES) -> list[dict]:
    if not suite_rows:
        raise RuntimeError("at least one query suite is required")
    expected_pairs = _expected_pairs(capacities)
    if any(len(rows) != len(expected_pairs) for rows in suite_rows):
        raise RuntimeError(
            f"every query suite must contain all {len(expected_pairs)} workloads")

    combined = []
    for index, (expected_c, expected_capacity) in enumerate(expected_pairs):
        rows = [suite[index] for suite in suite_rows]
        if any(
            abs(row["c"] - expected_c) > 1e-6 or row["capacity"] != expected_capacity
            for row in rows
        ):
            raise RuntimeError("query suite workloads do not align")
        counts = [row.get("query_count") for row in rows]
        if any(type(count) is not int or count <= 0 for count in counts):
            raise RuntimeError("query suite workload has an invalid observation count")
        total = sum(counts)
        combined.append({
            "c": expected_c,
            "capacity": expected_capacity,
            **{
                field: sum(row[field] * count for row, count in zip(rows, counts)) / total
                for field in ("time_ms", "recall", "ratio", "kratio", "cost")
            },
            "query_count": total,
        })
    return combined


def _reference_verdict(candidate: list[dict], reference: list[dict]) -> tuple[int, list[dict]]:
    disclosed = len(_expected_pairs())
    allowed = (disclosed, disclosed + len(_expected_pairs(FINAL_ONLY_CAPACITIES)))
    if len(candidate) != len(reference) or len(candidate) not in allowed:
        raise RuntimeError("candidate and reference must each contain every workload")
    rows = []
    for candidate_row, reference_row in zip(candidate, reference):
        if (candidate_row["c"], candidate_row["capacity"]) != (
                reference_row["c"], reference_row["capacity"]):
            raise RuntimeError("candidate and reference workloads do not align")
        decisions = {
            "mean_query_time": candidate_row["time_ms"] < reference_row["time_ms"],
            "recall_at_50": candidate_row["recall"] >= reference_row["recall"],
            "last_result_inner_product_ratio": candidate_row["kratio"] >= reference_row["kratio"],
            "verified_candidate_count": candidate_row["cost"] <= reference_row["cost"],
        }
        if candidate_row["capacity"] in FINAL_ONLY_CAPACITIES:
            decisions = {name: decisions[name] for name in FINAL_ONLY_METRICS}
        rows.append({
            "workload": _workload_name(candidate_row),
            "metrics": decisions,
            "candidate": {"mean_query_time_ms": candidate_row["time_ms"], "recall_at_50": candidate_row["recall"], "last_result_inner_product_ratio": candidate_row["kratio"], "mean_verified_candidate_count": candidate_row["cost"]},
            "reference": {"mean_query_time_ms": reference_row["time_ms"], "recall_at_50": reference_row["recall"], "last_result_inner_product_ratio": reference_row["kratio"], "mean_verified_candidate_count": reference_row["cost"]},
        })
    return int(all(decision for row in rows for decision in row["metrics"].values())), rows


def _paired_scores(patch: Path, splits: tuple[int, ...]) -> tuple[float, float, dict]:
    candidate_suites, reference_suites, naive_suites = [], [], []
    extra_candidate_suites, extra_reference_suites = [], []
    same_as_reference = patch.read_bytes() == (TASK_DIR / "reference.patch").read_bytes()
    # Train measures the disclosed grid only. Final also measures the
    # undisclosed capacities, in a separate workload process per policy so the
    # disclosed measurement and its process timeout are exactly as in train.
    # The fixed baseline is not run there: it feeds only the continuous score,
    # which stays defined over the disclosed grid.
    final_only = FINAL_ONLY_CAPACITIES if tuple(splits) == FINAL_SUITE else ()
    for suite_index, split in enumerate(splits, start=1):
        trial = Path(tempfile.mkdtemp(prefix="fargo_pair_"))
        apps = []
        try:
            candidate_app = _apply(patch)
            apps.append(candidate_app)
            naive_app = trial / "naive"
            shutil.copytree(TASK_DIR / "repo", naive_app, ignore=shutil.ignore_patterns(".git"))
            naive = _build_and_run(
                naive_app, trial / "naive_case", split,
                process_role="fixed_baseline", suite_index=suite_index)
            candidate = _build_and_run(
                candidate_app, trial / "candidate_case", split,
                process_role="candidate", suite_index=suite_index)
            if same_as_reference:
                reference = candidate
            else:
                reference_app = _apply(TASK_DIR / "reference.patch")
                apps.append(reference_app)
                reference = _build_and_run(
                    reference_app, trial / "reference_case", split,
                    process_role="reference", suite_index=suite_index)
            if final_only:
                extra_candidate = _build_and_run(
                    candidate_app, trial / "candidate_final_only_case", split,
                    process_role="candidate", suite_index=suite_index,
                    capacities=final_only)
                extra_candidate_suites.append(extra_candidate)
                extra_reference_suites.append(
                    extra_candidate if same_as_reference else _build_and_run(
                        reference_app, trial / "reference_final_only_case", split,
                        process_role="reference", suite_index=suite_index,
                        capacities=final_only))
            candidate_suites.append(candidate)
            reference_suites.append(reference)
            naive_suites.append(naive)
        finally:
            for app in apps:
                shutil.rmtree(app, ignore_errors=True)
            shutil.rmtree(trial, ignore_errors=True)
    candidate = _combine_suite_rows(candidate_suites)
    reference = _combine_suite_rows(reference_suites)
    naive = _combine_suite_rows(naive_suites)
    # PASS covers every measured workload (four comparisons on the disclosed
    # grid, the three deterministic ones on the final-only grid); score and
    # aggregate diagnostics below keep their definition over the disclosed grid.
    if final_only:
        verdict, workload_passes = _reference_verdict(
            candidate + _combine_suite_rows(extra_candidate_suites, final_only),
            reference + _combine_suite_rows(extra_reference_suites, final_only))
    else:
        verdict, workload_passes = _reference_verdict(candidate, reference)
    diagnostics = {
        "mean_recall": sum(r["recall"] for r in candidate) / len(candidate),
        "mean_kratio": sum(r["kratio"] for r in candidate) / len(candidate),
        "mean_time_ms": sum(r["time_ms"] for r in candidate) / len(candidate),
        "mean_candidates": sum(r["cost"] for r in candidate) / len(candidate),
        "beats_reference": verdict,
        "beat_spec_workload_passes": workload_passes,
        "aggregation": (
            "public_plus_hidden_pooled_by_workload"
            if tuple(splits) == FINAL_SUITE else "public_only_by_workload"
        ),
        "evaluated_suites": [SUITE_NAMES[split] for split in splits],
        "queries_per_workload": candidate[0]["query_count"],
    }
    return _raw_score(candidate, naive), _raw_score(reference, naive), diagnostics


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    try:
        patch = Path(solution_path)
        if not patch.is_file():
            return _invalid("submission patch does not exist")
        suite = FINAL_SUITE if _is_final() else DEV_SUITE
        candidate_score, reference_score, public = _paired_scores(patch, suite)
        candidate_score = max(0.0, min(100.0, candidate_score))
        reference_score = max(0.0, min(100.0, reference_score))
        metrics = {
            "beats_reference": public["beats_reference"],
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
    except WorkloadTimeout as exc:
        return _invalid(str(exc))
    except subprocess.TimeoutExpired:
        return _invalid("submission exceeded the build timeout")
    except (ValueError, RuntimeError) as exc:
        return _invalid(str(exc))
    except Exception:
        return _invalid("submission could not be evaluated")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(TASK_DIR / "reference.patch")
    print(evaluate(target))
