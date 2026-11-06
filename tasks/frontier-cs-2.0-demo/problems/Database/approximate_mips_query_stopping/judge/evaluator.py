"""Evaluator for restoring per-query stopping to the authors' FARGO source.

Judge for the shared verifier (#757): evaluate(tree). `tree` is the starter repo (`pristine/`)
with only `src/alg.cpp` overlaid; the shared edit scope has already applied the patch and the
source-token denylist (task.toml `forbidden_content`). The fixed-budget baseline is `pristine/`
and the reference is `pristine/` + reference.patch, run on the same suites.

Score: per workload j (approximation ratio x heap capacity; 12 train, more in final)
Q_j = sqrt(clip(recall, 0, 1) * clip(kratio, 0, 1)), t_j = max(0.25, T_fixed / max(1e-9, T)),
w_j = max(0.25, C_fixed / max(1, C)), E_j = t_j^0.25 * w_j^0.75, point_j = 100 Q_j E_j / (1+E_j);
raw_score = score_unbounded = clip(mean point_j, 0, 100). Build, run, timeout or record failures
are scored zeros (every workload failed).

Beat-spec metrics (beat_spec_workload_passes, one row per workload):
- mean_query_time, lower is better: passes when strictly below the reference's.
- recall_at_50, higher is better: passes when at least the reference's.
- last_result_inner_product_ratio, higher is better: passes when at least the reference's.
- verified_candidate_count, lower is better: passes when at most the reference's.
beats_reference = 1 iff all four pass on every workload; no tolerance. reference_score is the
reference's own score; margin = raw_score - reference_score.

Process split (#757 isolate): every build and workload run (starter, candidate, reference)
happens in a fresh shared isolate instance that sees only the tree, the vector file under a
neutral name, the calibration table and a one-row placeholder ground truth; the judge's data,
environment and other runs are out of reach. The instance prints, per workload, the harness's
query time and verified-candidate count and each query's top-50 (id, inner product); recall_at_50
and the ratios are computed here against judge_data/audio.ben with the harness's float32
arithmetic (performance.h), so the numbers are those the in-process harness printed before.

The full prior specification is archived at
docs/task-authoring/approximate_mips_query_stopping/provenance/beat_spec.json.
"""
from __future__ import annotations

import array
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
import isolate  # the shared launcher, beside edit_scope in the Verifier base

PRISTINE = TASK_DIR / "pristine"
DEV_SUITE = (0,)
FINAL_SUITE = (0, 1)
SUITE_NAMES = {0: "public", 1: "hidden"}
APPROXIMATION_RATIOS = (0.6, 0.8, 0.9)
CAPACITIES = (100, 200, 400, 800)
# Final additionally measures these heap capacities. They are neither in the
# statement nor in any development feedback, and their workload names are
# opaque, so the values exist only in this judge-side file. They sit between
# and slightly beyond the disclosed capacities (geometric midpoints). Every
# metric is measured, compared and scored there exactly as on the disclosed grid.
FINAL_ONLY_CAPACITIES = (140, 280, 560, 1120)
RUN_TIMEOUT = 180
BUILD_TIMEOUT = 120
HASH_SEED = 314159
K = 50  # the driver's k: recall_at_50 and the last-result ratio use each query's top 50
# A valid one-row benchmark (N=1, num=1) so the unchanged Preprocess loads no ground truth.
PLACEHOLDER_BEN = struct.pack("<iiif", 1, 1, 0, 1.0)
RUN_OUTPUT_BYTES = 64 << 20


class WorkloadTimeout(RuntimeError):
    """A bounded diagnostic that does not reveal suite membership or data."""

    def __init__(self, process_role: str, suite_index: int):
        super().__init__(
            f"workload process timed out (process_role={process_role}; "
            f"suite_index={suite_index})"
        )


DRIVER = r'''#include "Preprocess.h"
#include "mf_alsh.h"
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
    std::cout << std::setprecision(9);
    for (float c : ratios) {
        for (int b : capacities) {
            // Time and cost accumulate exactly as performance.h does; the judge scores the
            // printed top-k against the ground truth this process never sees.
            int num = 0, cost = 0;
            float time_total = 0;
            for (int q = 0; q < qnum; ++q) {
                int query_id = 2 * q + split;
                mf_alsh::Query query(query_id, c, 50, index, prep, b);
                num++;
                cost += query.cost;
                time_total += query.time_total;
                int n = query.res.size();
                if (n > query.k) n = query.k;
                std::cout << "FARGO_METRIC_QUERY " << query_id << " " << n;
                for (int j = 0; j < n; ++j)
                    std::cout << " " << query.res[j].id << " " << query.res[j].inp;
                std::cout << "\n";
            }
            std::cout << "FARGO_METRIC " << c << " " << b << " "
                      << (1000.0 * time_total / num) << " "
                      << ((double)cost / num) << "\n";
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
            "metrics": dict.fromkeys((
                "mean_query_time", "recall_at_50",
                "last_result_inner_product_ratio", "verified_candidate_count"), False),
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


def _f32(value: float) -> float:
    """`value` rounded to float32 (overflow to infinity), as a C++ float holds it."""
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _fdiv(a: float, b: float) -> float:
    """float32 a / b with IEEE division by zero (a double quotient rounds exactly)."""
    if b == 0.0:
        if a == 0.0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return _f32(a / b)


def _printed(value: float) -> float:
    """What the old harness line carried: the value at std::setprecision(9), parsed back."""
    return float("%.9g" % value)


def _ground_truth() -> tuple[int, int, array.array, array.array]:
    raw = (TASK_DIR / "judge_data" / "audio.ben").read_bytes()
    rows, width = struct.unpack_from("<ii", raw)
    ids, products = array.array("i"), array.array("f")
    ids.frombytes(raw[8:8 + 4 * rows * width])
    products.frombytes(raw[8 + 4 * rows * width:8 + 8 * rows * width])
    if len(ids) != rows * width or len(products) != rows * width:
        raise OSError("audio.ben is truncated")
    return rows, width, ids, products


def _point_quality(queries: list[tuple[int, list[tuple[int, float]]]],
                   truth) -> tuple[float, float, float]:
    """recall_at_50, mean result ratio and last-result ratio of one workload, as
    Performance::update accumulates them (float32 sums, int counts) and the old driver
    printed them."""
    rows, width, ids, products = truth
    nn, res_num, ratio, kratio = 0, 0, 0.0, 0.0
    for query_id, results in queries:
        if not 0 <= query_id < rows or len(results) > width:
            raise RuntimeError("workload produced malformed metrics")
        res_num += len(results)
        found, exact = set(), set()
        for j, (identifier, product) in enumerate(results):
            rate = _fdiv(product, products[query_id * width + j])
            ratio = _f32(ratio + rate)
            if j == len(results) - 1:
                kratio = _f32(kratio + rate)
            found.add(identifier)
            exact.add(ids[query_id * width + j])
        nn += len(found & exact)
    count = len(queries)
    if res_num:
        overall = ratio / res_num
    else:
        overall = math.nan if ratio == 0.0 or math.isnan(ratio) else math.copysign(math.inf, ratio)
    return (_printed(nn / (count * K)), _printed(overall), _printed(kratio / count))


def _build_and_run(
        app: Path, case_root: Path, split: int, *, process_role: str,
        suite_index: int, capacities: tuple[int, ...] = CAPACITIES) -> list[dict]:
    if capacities == CAPACITIES:
        _instrument_source(app)
    else:
        # A second process of an already measured tree: only the driver's
        # capacity list differs.
        _instrument_source(app, capacities)
    case_root.mkdir(parents=True, exist_ok=True)
    qnum = _query_count()
    # Build in its own instance; the binary comes back as a declared output and runs in a
    # fresh instance of the same tree (no build state, no judge data, private dirs).
    built = isolate.run(["sh", "-c", "make -j2 >/dev/null 2>&1 && cp fg {out}/fg"],
                             tree=str(app), timeout_seconds=BUILD_TIMEOUT, outputs=["fg"],
                             output_bytes=RUN_OUTPUT_BYTES)
    if built.timed_out:
        raise subprocess.TimeoutExpired("make", BUILD_TIMEOUT)
    if not built.ok or not built.outputs.get("fg"):
        raise RuntimeError("submission does not build")
    binary = app / "fg"
    binary.write_bytes(built.outputs["fg"])
    binary.chmod(0o755)
    # Neutral names: nothing the instance sees names the dataset or holds its ground truth.
    inputs = {"vectors.data_new": str(TASK_DIR / "judge_data" / "audio.data_new"),
              "placeholder.ben": PLACEHOLDER_BEN,
              "calibration.bin": str(TASK_DIR / "judge_data" / "MyfunctionXTheta.data")}
    proc = isolate.run(["./fg", "{in}/vectors.data", "{in}/placeholder.ben",
                             "{in}/calibration.bin", str(split), str(qnum)],
                            tree=str(app), timeout_seconds=RUN_TIMEOUT, inputs=inputs,
                            output_bytes=RUN_OUTPUT_BYTES)
    if proc.timed_out:
        raise WorkloadTimeout(process_role, suite_index)
    if not proc.ok:
        raise RuntimeError("submission failed while running the workload")
    truth = _ground_truth()
    rows, queries = [], []
    for line in proc.stdout.decode().splitlines():
        if line.startswith("FARGO_METRIC_QUERY "):
            parts = line.split()
            try:
                count = int(parts[2])
                results = [(int(parts[3 + 2 * j]), _f32(float(parts[4 + 2 * j])))
                           for j in range(count)]
                if len(parts) != 3 + 2 * count:
                    raise ValueError
                queries.append((int(parts[1]), results))
            except (IndexError, ValueError) as error:
                raise RuntimeError("workload produced malformed metrics") from error
        elif line.startswith("FARGO_METRIC "):
            parts = line.split()
            if len(parts) != 5:
                raise RuntimeError("workload produced malformed metrics")
            if [query_id for query_id, _ in queries] != [2 * q + split for q in range(qnum)]:
                raise RuntimeError("workload did not produce every required metric record")
            recall, ratio, kratio = _point_quality(queries, truth)
            queries = []
            rows.append({
                "c": float(parts[1]), "capacity": int(parts[2]),
                "time_ms": float(parts[3]), "recall": recall,
                "ratio": ratio, "kratio": kratio,
                "cost": float(parts[4]), "query_count": qnum,
            })
    if queries:
        raise RuntimeError("workload did not produce every required metric record")
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


def _copy_app(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git"))
    for path in (destination, *destination.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    return destination


def _reference_app(destination: Path) -> Path:
    """The pristine starter with the judge's reference.patch applied (trusted)."""
    app = _copy_app(PRISTINE, destination)
    applied = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(TASK_DIR / "reference.patch")],
        cwd=app, capture_output=True, timeout=60,
        env={**os.environ, "GIT_CEILING_DIRECTORIES": str(destination.parent.resolve())})
    if applied.returncode != 0:
        raise OSError("reference patch does not apply")
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
        rows.append({
            "workload": _workload_name(candidate_row),
            "metrics": decisions,
            "candidate": {"mean_query_time_ms": candidate_row["time_ms"], "recall_at_50": candidate_row["recall"], "last_result_inner_product_ratio": candidate_row["kratio"], "mean_verified_candidate_count": candidate_row["cost"]},
            "reference": {"mean_query_time_ms": reference_row["time_ms"], "recall_at_50": reference_row["recall"], "last_result_inner_product_ratio": reference_row["kratio"], "mean_verified_candidate_count": reference_row["cost"]},
        })
    return int(all(decision for row in rows for decision in row["metrics"].values())), rows


def _paired_scores(tree: Path, splits: tuple[int, ...]) -> tuple[float, float, dict]:
    candidate_suites, reference_suites, naive_suites = [], [], []
    extra_candidate_suites, extra_reference_suites, extra_naive_suites = [], [], []
    with tempfile.TemporaryDirectory(prefix="fargo_reference_") as scratch:
        reference_source = (_reference_app(Path(scratch) / "app") / "src" / "alg.cpp").read_bytes()
    same_as_reference = (tree / "src" / "alg.cpp").read_bytes() == reference_source
    # Train measures the disclosed grid only. Final also measures the
    # undisclosed capacities, in a separate workload process per policy so the
    # disclosed measurement and its process timeout are exactly as in train.
    final_only = FINAL_ONLY_CAPACITIES if tuple(splits) == FINAL_SUITE else ()
    for suite_index, split in enumerate(splits, start=1):
        trial = Path(tempfile.mkdtemp(prefix="fargo_pair_"))
        try:
            candidate_app = _copy_app(tree, trial / "candidate")
            naive_app = _copy_app(PRISTINE, trial / "naive")
            naive = _build_and_run(
                naive_app, trial / "naive_case", split,
                process_role="fixed_baseline", suite_index=suite_index)
            candidate = _build_and_run(
                candidate_app, trial / "candidate_case", split,
                process_role="candidate", suite_index=suite_index)
            if same_as_reference:
                reference = candidate
            else:
                reference_app = _reference_app(trial / "reference")
                reference = _build_and_run(
                    reference_app, trial / "reference_case", split,
                    process_role="reference", suite_index=suite_index)
            if final_only:
                extra_naive_suites.append(_build_and_run(
                    naive_app, trial / "naive_final_only_case", split,
                    process_role="fixed_baseline", suite_index=suite_index,
                    capacities=final_only))
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
            shutil.rmtree(trial, ignore_errors=True)
    candidate = _combine_suite_rows(candidate_suites)
    reference = _combine_suite_rows(reference_suites)
    naive = _combine_suite_rows(naive_suites)
    # PASS, score and aggregate diagnostics cover every measured workload with
    # the same four metrics: the disclosed grid, then in final the final-only grid.
    if final_only:
        candidate += _combine_suite_rows(extra_candidate_suites, final_only)
        reference += _combine_suite_rows(extra_reference_suites, final_only)
        naive += _combine_suite_rows(extra_naive_suites, final_only)
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


def evaluate(tree: str) -> tuple[float, float, str, dict]:
    try:
        suite = FINAL_SUITE if _is_final() else DEV_SUITE
        candidate_score, reference_score, public = _paired_scores(Path(tree), suite)
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
    except isolate.IsolateError:
        # Failed verification, not a candidate score.
        raise
    except WorkloadTimeout as exc:
        return _invalid(str(exc))
    except subprocess.TimeoutExpired:
        return _invalid("submission exceeded the build timeout")
    except (ValueError, RuntimeError) as exc:
        return _invalid(str(exc))
    except Exception:
        return _invalid("submission could not be evaluated")

