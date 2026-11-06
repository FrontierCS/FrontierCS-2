"""Evaluator for implementing the paper-aligned FARGO query core to the authors' FARGO source.

The shared verifier (#757) applies the patch (only src/alg.cpp may change) and rejects a patch
whose alg.cpp contains a forbidden benchmark or process-inspection token (task.toml
forbidden_content); evaluate(tree) gets the pristine starter with that file overlaid.

Scoring (_raw_score). Per operating point (dataset x split x c in {0.6,0.8,0.9} x k in
{1,10,50,100}): quality = sqrt(clip01(recall) * clip01(overall_ratio)), efficiency =
(t_naive / max(1e-9, t))**0.25 * (cost_naive / max(1, cost))**0.75 against the starter run in
the same evaluation, point = 100 * quality * efficiency / (1 + efficiency); raw_score =
score_unbounded = mean over the 24 train or 48 final points, clipped to [0, 100].
Metrics per point: recall and overall_ratio (higher better; pass when >= the reference's),
query_time_ms (lower; passes when strictly lower), candidates_verified (lower; passes when <=).
beats_reference = every metric passes on every point. A failed conformance check, build or run
is a scored zero (_invalid).

Process split (#757 isolate): conformance, build and every workload run (starter, candidate,
reference) happen in fresh shared isolate instances that see only the tree, the vector file
under a neutral name, the calibration table and a one-row placeholder ground truth; the judge's
data, environment and other runs are out of reach. The instance prints, per operating point, the
harness's query time and verified-candidate count and each query's top-k (id, inner product);
recall, overall ratio and k-ratio are computed here against judge_data/<dataset>.ben with the
harness's float32 arithmetic (performance.h), so the numbers are those the in-process harness
printed before the split."""
from __future__ import annotations

import array
import math
import gzip
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

# The starter repo/, copied by the Verifier Dockerfile.
PRISTINE = TASK_DIR / "pristine"
DATASETS = ("audio", "mnist")
DEV_SUITE = (0,)
FINAL_SUITE = (0, 1)
APPROXIMATION_RATIOS = (0.6, 0.8, 0.9)
CAPACITIES = (1, 10, 50, 100)
RUN_TIMEOUT = 180
BUILD_TIMEOUT = 120
HASH_SEED = 314159
# A valid one-row benchmark (N=1, num=1) so the unchanged Preprocess loads no ground truth.
PLACEHOLDER_BEN = struct.pack("<iiif", 1, 1, 0, 1.0)
RUN_OUTPUT_BYTES = 64 << 20


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
    const int capacities[] = {1, 10, 50, 100};
    std::cout << std::setprecision(9);
    for (float c : ratios) {
        for (int b : capacities) {
            // Time and cost accumulate exactly as performance.h does; the judge scores the
            // printed top-k against the ground truth this process never sees.
            int num = 0, cost = 0;
            float time_total = 0;
            for (int q = 0; q < qnum; ++q) {
                int query_id = 2 * q + split;
                mf_alsh::Query query(query_id, c, b, index, prep, b);
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


def _ground_truth(dataset: str) -> tuple[int, int, array.array, array.array]:
    raw = (TASK_DIR / "judge_data" / (dataset + ".ben")).read_bytes()
    rows, width = struct.unpack_from("<ii", raw)
    ids, products = array.array("i"), array.array("f")
    ids.frombytes(raw[8:8 + 4 * rows * width])
    products.frombytes(raw[8 + 4 * rows * width:8 + 8 * rows * width])
    if len(ids) != rows * width or len(products) != rows * width:
        raise OSError(f"{dataset}.ben is truncated")
    return rows, width, ids, products


def _point_quality(queries: list[tuple[int, list[tuple[int, float]]]], capacity: int,
                   truth) -> tuple[float, float, float]:
    """recall, overall ratio and k-ratio of one operating point, as Performance::update
    accumulates them (float32 sums, int counts) and the old driver printed them."""
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
            if j == capacity - 1:
                kratio = _f32(kratio + rate)
            found.add(identifier)
            exact.add(ids[query_id * width + j])
        nn += len(found & exact)
    count = len(queries)
    if res_num:
        overall = ratio / res_num
    else:
        overall = math.nan if ratio == 0.0 or math.isnan(ratio) else math.copysign(math.inf, ratio)
    return (_printed(nn / (count * capacity)), _printed(overall), _printed(kratio / count))


def _build_and_run(app: Path, case_root: Path, split: int, dataset: str = "audio") -> list[dict]:
    _instrument_source(app, split)
    case_root.mkdir(parents=True, exist_ok=True)
    if dataset not in DATASETS:
        raise ValueError("unknown dataset")
    source = TASK_DIR / "judge_data" / (dataset + ".data_new")
    # A neutral name: nothing the instance sees says which dataset it measures.
    vectors = case_root / "vectors.data_new"
    if source.is_file():
        shutil.copy2(source, vectors)
    else:
        with gzip.open(str(source) + ".gz", "rb") as inp, vectors.open("wb") as out:
            shutil.copyfileobj(inp, out)
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
    inputs = {"vectors.data_new": str(vectors), "placeholder.ben": PLACEHOLDER_BEN,
              "calibration.bin": str(TASK_DIR / "judge_data" / "squared_qd_cdf.bin")}
    proc = isolate.run(["./fg", "{in}/vectors.data", "{in}/placeholder.ben",
                             "{in}/calibration.bin", str(split), str(qnum)],
                            tree=str(app), timeout_seconds=RUN_TIMEOUT, inputs=inputs,
                            output_bytes=RUN_OUTPUT_BYTES)
    if proc.timed_out:
        raise subprocess.TimeoutExpired("fg", RUN_TIMEOUT)
    if not proc.ok:
        raise RuntimeError("submission failed while running the workload")
    truth = _ground_truth(dataset)
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
            capacity = int(parts[2])
            if [query_id for query_id, _ in queries] != [2 * q + split for q in range(qnum)]:
                raise RuntimeError("workload did not produce every required metric record")
            recall, ratio, kratio = _point_quality(queries, capacity, truth)
            queries = []
            rows.append({
                "c": float(parts[1]), "capacity": capacity,
                "time_ms": float(parts[3]), "recall": recall,
                "ratio": ratio, "kratio": kratio,
                "cost": float(parts[4]),
            })
    if queries:
        raise RuntimeError("workload did not produce every required metric record")
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


def _copy(tree: Path) -> Path:
    app = Path(tempfile.mkdtemp(prefix="fargo_app_")) / "app"
    shutil.copytree(tree, app, ignore=shutil.ignore_patterns(".git"))
    for path in (app, *app.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    return app


def _reference_tree() -> Path:
    """The pristine starter with judge/reference.patch applied (trusted, judge-owned)."""
    app = _copy(PRISTINE)
    subprocess.run(["git", "apply", "--whitespace=nowarn", str(TASK_DIR / "reference.patch")],
                   cwd=app, check=True, capture_output=True, timeout=60,
                   env={**os.environ, "GIT_CEILING_DIRECTORIES": str(app.parent)})
    return app


def _apply(tree: Path, reference: bool = False):
    """A fresh build copy of a judged tree, after the query-core conformance check."""
    app = _reference_tree() if reference else _copy(tree)
    try:
        # Compiled and run in an instance of its own (the check's source is its only input).
        checked = isolate.run(
            ["sh", "-c", "g++ -std=c++11 -O2 -I src {in}/conformance.cpp src/alg.cpp "
                         "-o {tmp}/conformance >/dev/null && exec {tmp}/conformance"],
            tree=str(app), inputs={"conformance.cpp": str(TASK_DIR / "conformance.cpp")},
            timeout_seconds=BUILD_TIMEOUT + 15)
        if not checked.ok or checked.stdout.decode().strip() != \
                "conformance cases=108 ordered=0 PASS":
            raise ValueError("query-core conformance missing")
    except isolate.IsolateError:
        shutil.rmtree(app.parent, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(app.parent, ignore_errors=True)
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


def _paired_scores(tree: Path, splits: tuple[int, ...]) -> tuple[float, float, dict]:
    candidate_scores, reference_scores, diagnostics = [], [], []
    comparisons = []
    restored = _reference_tree()
    try:
        same_as_reference = ((tree / "src/alg.cpp").read_bytes()
                             == (restored / "src/alg.cpp").read_bytes())
    finally:
        shutil.rmtree(restored.parent, ignore_errors=True)
    for dataset, split in ((dataset, split) for dataset in DATASETS for split in splits):
        trial = Path(tempfile.mkdtemp(prefix="fargo_pair_"))
        apps = []
        try:
            candidate_app = _apply(tree)
            apps.append(candidate_app)
            naive_app = trial / "naive"
            shutil.copytree(PRISTINE, naive_app, ignore=shutil.ignore_patterns(".git"))
            naive = _build_and_run(naive_app, trial / "naive_case", split, dataset)
            candidate = _build_and_run(candidate_app, trial / "candidate_case", split, dataset)
            if same_as_reference:
                reference = candidate
            else:
                try:
                    reference_app = _apply(PRISTINE, reference=True)
                    apps.append(reference_app)
                    reference = _build_and_run(reference_app, trial / "reference_case", split, dataset)
                except isolate.IsolateError:
                    raise
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
                shutil.rmtree(app.parent, ignore_errors=True)
            shutil.rmtree(trial, ignore_errors=True)
    return (
        sum(candidate_scores) / len(candidate_scores),
        sum(reference_scores) / len(reference_scores),
        {**{key: sum(d[key] for d in diagnostics) / len(diagnostics)
              for key in diagnostics[0]},
         "_comparisons": comparisons},
    )


def evaluate(tree: str) -> tuple[float, float, str, dict]:
    """`tree` is the pristine starter with the submission's src/alg.cpp overlaid."""
    try:
        tree = Path(tree)
        if not (tree / "src/alg.cpp").is_file():
            return _invalid("submission tree has no src/alg.cpp")
        final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
        suite = FINAL_SUITE if final else DEV_SUITE
        candidate_score, reference_score, public = _paired_scores(tree, suite)
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
    except (ReferenceFailure, isolate.IsolateError):
        # Failed verification, not a candidate score.
        raise
    except subprocess.TimeoutExpired:
        return _invalid("submission exceeded the evaluation timeout")
    except (ValueError, RuntimeError) as exc:
        return _invalid(str(exc))
    except Exception:
        return _invalid("submission could not be evaluated")


if __name__ == "__main__":
    print(evaluate(sys.argv[1] if len(sys.argv) > 1 else str(PRISTINE)))
