#!/usr/bin/env python3
"""Evaluator for factorized between-round updates in JoinBoost."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

try:
    from submission_patch import apply_submission_patch
except ImportError:  # Local task-author smoke tests; the judge supplies the canonical helper.
    from _local_submission_patch import apply_submission_patch


TASK_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((TASK_DIR / "config.yaml").read_text(encoding="utf-8"))
REFERENCE_PATCH = TASK_DIR / "reference.patch"
# The obvious correct implementation: one UPDATE per leaf instead of the single
# batched CASE the withheld version builds. Same SQL, same semantics, N queries
# rather than 1 — so it isolates exactly the batching the paper is about, and it
# is the zero of the scale.
BASELINE_PATCH = TASK_DIR / "baseline_naive.patch"
WORKER = TASK_DIR / "benchmark.py"
RUN_TIMEOUT_SECONDS = 600
RMSE_TOLERANCE = 1.0e-3
# The sorted-prediction L1 bound is per row, so that it does not tighten as the
# fact table grows: sum |candidate - oracle| <= rows * 1e-6. The reference's own
# floating-point noise on the real-row workloads is up to 9.1e-9 per row (the
# previous absolute 1e-3 at 70,000 rows was 1.4e-8 per row, too close to that),
# while a different tree moves predictions by many orders of magnitude more.
PREDICTION_L1_TOLERANCE_PER_ROW = 1.0e-6
# The trusted executor is byte-locked: the patch cannot touch it, and the
# patched tree must still carry the pristine bytes.
LOCKED_FILES = ("src/joinboost/executor.py",)
POLICY_EXIT_CODE = 3
# Closed reason strings for trusted-executor violations (score 0, invalid).
POLICY_REASONS = {
    "data_cap": "submission returned more data to Python than the trusted executor allows",
    "settings": "submission issued a statement that the trusted executor does not allow; database settings are fixed",
    "executor": "submission replaced or bypassed the trusted executor",
    "materialised_join": "submission materialised the join into a fact-scale relation",
}
STATIC_BYPASS_REASON = "patch reaches the database outside the trusted executor"
# Beat spec v2: each metric is decided per workload and conjoined.
METRICS = ("elapsed_seconds", "residual_update_seconds", "rmse_error")
# Interleaved paired measurement for the reference comparison. Each paired
# repetition runs one fresh reference and one fresh candidate process for every
# workload, back to back; this pre-recorded order says which of the two runs
# first in each repetition. Strict alternation over an even number of
# repetitions puts each side first equally often and, under a monotone host
# drift, makes the two middle repetitions (which decide a median of four) one
# of each order, so neither the mean nor the median favors either side. The
# repetition loop is outside the workload loop, spreading each workload's
# samples over the whole evaluation. Timing decisions compare medians.
PAIR_ORDER = ("reference", "candidate", "reference", "candidate")
FAST_PAIR_ORDER = ("reference", "candidate")


class PolicyRejection(RuntimeError):
    """The candidate broke a trusted-executor rule at run time."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code if code in POLICY_REASONS else "executor"


class ReferenceVerificationError(RuntimeError):
    """The optimized reference or the bundled baseline failed or was invalid.

    A failed verification, never a candidate score: it propagates to the
    verifier adapter, which reports evaluator_error.
    """
# Fact tables are slices of real Corporacion Favorita sales rows (see the
# statement and docs/tasks/residual_state_advancement/paper-setting.md). The
# development slices cover 2017-02-02..2017-04-30 and ship in the agent-visible
# tree; the graded slices cover 2017-05-01..2017-07-30 and exist only under
# judge_data. Within a date range, workload i takes the rows whose source id
# falls in fixed residue classes modulo 18, so no row is used twice.
# Eight workloads per suite: each of the four fact slices is measured under two
# profiles. The first four are the original profiles; the second four spread
# tree depth (1 to 5), round count (3 to 24) and topology, within what the
# supplied implementation fits in memory without spilling (peak 7.7 GB).
DEV_SUITE = (
    {"fact": "sales-dev-0.parquet", "rows": 2_054_144, "depth": 2, "estimators": 8, "learning_rate": 0.5, "topology": "core_four_table"},
    {"fact": "sales-dev-1.parquet", "rows": 1_540_608, "depth": 3, "estimators": 7, "learning_rate": 0.1, "topology": "five_table_oil"},
    {"fact": "sales-dev-2.parquet", "rows": 1_027_072, "depth": 4, "estimators": 6, "learning_rate": 1.0, "topology": "six_table"},
    {"fact": "sales-dev-3.parquet", "rows": 2_054_144, "depth": 2, "estimators": 12, "learning_rate": 0.25, "topology": "six_table"},
    {"fact": "sales-dev-0.parquet", "rows": 2_054_144, "depth": 1, "estimators": 24, "learning_rate": 0.5, "topology": "core_four_table"},
    {"fact": "sales-dev-1.parquet", "rows": 1_540_608, "depth": 2, "estimators": 20, "learning_rate": 0.1, "topology": "five_table_oil"},
    {"fact": "sales-dev-2.parquet", "rows": 1_027_072, "depth": 5, "estimators": 3, "learning_rate": 0.5, "topology": "six_table"},
    {"fact": "sales-dev-3.parquet", "rows": 2_054_144, "depth": 3, "estimators": 4, "learning_rate": 0.5, "topology": "six_table"},
)
FINAL_SUITE = (
    {"fact": "sales-final-0.parquet", "rows": 2_142_914, "depth": 2, "estimators": 8, "learning_rate": 0.5, "topology": "core_four_table"},
    {"fact": "sales-final-1.parquet", "rows": 1_607_195, "depth": 3, "estimators": 7, "learning_rate": 0.1, "topology": "five_table_oil"},
    {"fact": "sales-final-2.parquet", "rows": 1_071_466, "depth": 4, "estimators": 6, "learning_rate": 1.0, "topology": "six_table"},
    {"fact": "sales-final-3.parquet", "rows": 2_142_929, "depth": 2, "estimators": 12, "learning_rate": 0.25, "topology": "six_table"},
    {"fact": "sales-final-0.parquet", "rows": 2_142_914, "depth": 1, "estimators": 24, "learning_rate": 0.5, "topology": "core_four_table"},
    {"fact": "sales-final-1.parquet", "rows": 1_607_195, "depth": 2, "estimators": 20, "learning_rate": 0.1, "topology": "five_table_oil"},
    {"fact": "sales-final-2.parquet", "rows": 1_071_466, "depth": 5, "estimators": 3, "learning_rate": 0.5, "topology": "six_table"},
    {"fact": "sales-final-3.parquet", "rows": 2_142_929, "depth": 3, "estimators": 4, "learning_rate": 0.5, "topology": "six_table"},
)
FAST_SUITE = (
    {"fact": "sales-dev-2.parquet", "limit": 20_000, "rows": 20_000, "depth": 2, "estimators": 2, "learning_rate": 0.5, "topology": "core_four_table"},
)

FORBIDDEN_ADDITIONS = (
    "/proc", "FRONTIER_", "benchmark.py", "sys.argv", "os.environ",
    "getenv(", "perf_counter", "monotonic(", "atexit", "sys.modules",
    "judge_data", "import benchmark", "__main__", "builtins", "inspect",
    "importlib", "ctypes", "os.listdir", "glob(", "rglob(", "walk(",
    ".fit =", ".fit=", "setattr(",
    "_getframe", "currentframe", "f_back", "f_globals", "f_locals",
    "traceback", "gc.get_objects",
)

# Added lines may not reach the DuckDB connection or its result-export APIs, and
# may not use the reflection routes that would uncover the guarded connection.
# This is a static screen of the patch text; the row cap, the statement filter
# and the locked settings are enforced at run time by the executor and DuckDB.
STATIC_BYPASS = re.compile(
    r"\bduckdb\b|\.conn\b|[\"']conn[\"']|fetch(all|one|many|numpy|df|_|\b)|\.df\(|\.arrow\(|\.pl\("
    r"|\.to_df\(|\.torch\(|\.tf\(|record_batch|__closure__|cell_contents|__globals__|__code__"
    r"|__dict__|\bvars\(|\beval\(|\bexec\(|\bcompile\(|__import__|__builtins__|\bgc\."
    r"|__subclasses__|__reduce|_guard_connection|GuardedConnection|MAX_ROWS_PER|_check_statements"
)

PROTECTED_METHODS = {
    "DummyModel": {"fit", "_fit", "predict"},
    "DecisionTree": {"fit", "_fit", "train_one", "_build_model", "compute_rmse", "predict", "_build_tree", "_get_best_split"},
    "GradientBoosting": {"fit", "_fit", "train_one", "_build_model", "compute_rmse", "predict", "_build_tree"},
    "RandomForest": {"fit", "_fit", "train_one", "_build_model", "compute_rmse", "predict", "_build_tree"},
}


def _empty_metrics(reference_score: float = 0.0, candidate_score: float = 0.0) -> dict[str, Any]:
    suite = _suite()
    return {
        "beats_reference": 0,
        "reference_score": reference_score,
        "margin": candidate_score - reference_score,
        "beat_spec_workload_passes": [
            {
                "workload": f"workload_{index}",
                "metrics": {name: False for name in METRICS},
            }
            for index, _case in enumerate(suite)
        ],
    }


def _suite() -> tuple[dict[str, Any], ...]:
    if os.environ.get("FRONTIER_EVALUATOR_FAST") == "1":
        return FAST_SUITE
    return FINAL_SUITE if os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final" else DEV_SUITE


def _patch_policy(path: Path) -> tuple[bool, str]:
    try:
        added = [line[1:] for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("+") and not line.startswith("+++")]
    except Exception:
        return False, "patch is not readable UTF-8"
    joined = "\n".join(added)
    for fragment in FORBIDDEN_ADDITIONS:
        if fragment.lower() in joined.lower():
            return False, "patch contains prohibited evaluator-introspection code"
    if STATIC_BYPASS.search(joined):
        return False, STATIC_BYPASS_REASON
    return True, "ok"


def _method_fingerprints(path: Path) -> dict[tuple[str, str], str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[tuple[str, str], str] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in PROTECTED_METHODS:
            found[(node.name, "__bases__")] = ast.dump(ast.Tuple(elts=node.bases), include_attributes=False)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in PROTECTED_METHODS[node.name]:
                    found[(node.name, child.name)] = ast.dump(child, include_attributes=False)
    return found


def _structural_policy(tree: Path) -> tuple[bool, str]:
    expected = _method_fingerprints(TASK_DIR / "harbor" / "app" / "src" / "joinboost" / "app.py")
    actual = _method_fingerprints(tree / "src" / "joinboost" / "app.py")
    if actual != expected:
        return False, "public fit and generic tree-growth methods must remain unchanged"
    for name in LOCKED_FILES:
        pristine = TASK_DIR / "harbor" / "app" / name
        if (tree / name).read_bytes() != pristine.read_bytes():
            return False, "the trusted executor must remain unchanged"
    return True, "ok"


def _archive_patch(solution_path: str) -> bool:
    try:
        data = Path(solution_path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()[:16]
        configured = os.environ.get("FRONTIER_PATCH_ARCHIVE_DIR")
        roots = [Path(configured)] if configured else [Path("/work/frontier-patch-archive"), TASK_DIR / ".patch-archive"]
        for archive in roots:
            try:
                archive.mkdir(parents=True, exist_ok=True)
                name = f"{secrets.token_hex(8)}-{digest}.patch"
                (archive / name).write_bytes(data)
                return True
            except Exception:
                continue
    except Exception:
        return False
    return False


_ORACLE_DIR: Path | None = None


def _oracle_cache(case: dict[str, Any]) -> Path:
    """Per-evaluation cache of the materialized-join oracle of one case."""
    global _ORACLE_DIR
    if _ORACLE_DIR is None:
        _ORACLE_DIR = Path(tempfile.mkdtemp(prefix="joinboost-oracle-"))
    return _ORACLE_DIR / f"oracle-{case['fact']}-{case['rows']}-{case['depth']}-{case['estimators']}.npz"


_WORKER_ENV = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _worker_command(tree: Path, case: dict[str, Any]) -> list[str]:
    command = [
        sys.executable, str(WORKER), "--tree", str(tree),
        "--fact", str(case["fact"]), "--rows", str(case["rows"]),
        "--depth", str(case["depth"]), "--estimators", str(case["estimators"]),
        "--learning-rate", str(case["learning_rate"]),
        "--topology", str(case["topology"]),
        "--oracle-cache", str(_oracle_cache(case)),
    ]
    if case.get("limit") is not None:
        command += ["--limit", str(case["limit"])]
    return command


def _prepare_oracle(tree: Path, case: dict[str, Any]) -> None:
    """Fit the materialized-join oracle of one case in its own untimed process."""
    proc = subprocess.run(
        _worker_command(tree, case) + ["--oracle-only"], cwd=TASK_DIR,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=RUN_TIMEOUT_SECONDS, env={**os.environ, **_WORKER_ENV},
    )
    if proc.returncode != 0 or not _oracle_cache(case).is_file():
        raise RuntimeError("oracle preparation failed")


def _run_once(tree: Path, case: dict[str, Any]) -> dict[str, Any]:
    proc = subprocess.run(
        _worker_command(tree, case),
        cwd=TASK_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=RUN_TIMEOUT_SECONDS,
        env={**os.environ, **_WORKER_ENV},
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    if proc.returncode == POLICY_EXIT_CODE and lines:
        raise PolicyRejection(str(json.loads(lines[-1]).get("policy_violation")))
    if proc.returncode != 0 or not lines:
        raise RuntimeError("training failed")
    result = json.loads(lines[-1])
    if "error" in result:
        raise RuntimeError("training failed")
    required = {"elapsed_seconds", "rows", "estimators", "rmse_error", "prediction_l1"}
    if not required.issubset(result):
        raise RuntimeError("benchmark result is incomplete")
    if not all(math.isfinite(float(result[key])) for key in required):
        raise RuntimeError("benchmark returned a non-finite value")
    update = result.get("residual_update_seconds")
    if update is not None and (
        isinstance(update, bool) or not isinstance(update, (int, float))
        or not math.isfinite(float(update)) or float(update) <= 0
    ):
        # Untrusted or degenerate update clocks are unmeasured, not invalid fits.
        result["residual_update_seconds"] = None
    return result


def _summarize(samples_by_case: list[list[dict[str, Any]]], score_repetitions: int = 2) -> dict[str, Any]:
    """One implementation's run in the legacy shape, from its per-case samples.

    Comparison fields are medians over every sample. The continuous score keeps
    its definition: the lower of the first ``score_repetitions`` fresh
    repetitions of each case, the same count the sequential baseline uses (two,
    or one on the local fast suite). Correctness is the worst value over every
    sample.
    """
    score_elapsed = 0.0
    work = 0
    worst_rmse = 0.0
    worst_prediction = 0.0
    worst_per_row = 0.0
    case_runs: list[dict[str, Any]] = []
    for index, samples in enumerate(samples_by_case):
        elapsed = [float(item["elapsed_seconds"]) for item in samples]
        updates = [item.get("residual_update_seconds") for item in samples]
        rmse_error = max(float(item["rmse_error"]) for item in samples)
        prediction_l1 = max(float(item["prediction_l1"]) for item in samples)
        per_row = max(float(item["prediction_l1"]) / max(int(item["rows"]), 1) for item in samples)
        worst_per_row = max(worst_per_row, per_row)
        case_runs.append({
            "workload": f"workload_{index}",
            "elapsed_seconds": statistics.median(elapsed),
            "score_elapsed_seconds": min(elapsed[:score_repetitions]),
            # Unmeasured if any repetition lacks a trusted update clock.
            "residual_update_seconds": (
                None if None in updates else statistics.median(float(v) for v in updates)),
            "rmse_error": rmse_error,
            "prediction_l1": prediction_l1,
            "prediction_l1_per_row": per_row,
            "elapsed_samples": elapsed,
            "residual_update_samples": updates,
        })
        score_elapsed += min(elapsed[:score_repetitions])
        work += int(samples[0]["rows"]) * int(samples[0]["estimators"])
        worst_rmse = max(worst_rmse, rmse_error)
        worst_prediction = max(worst_prediction, prediction_l1)
    return {
        "elapsed_seconds": score_elapsed,
        "work": work,
        "rmse_error": worst_rmse,
        "prediction_l1": worst_prediction,
        "prediction_l1_per_row": worst_per_row,
        "case_count": len(samples_by_case),
        "workloads": case_runs,
    }


def _run_paired(candidate_tree: Path | None, reference_tree: Path,
                suite: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], dict[str, Any], list[list[str]]]:
    """Interleave fresh candidate and reference processes in PAIR_ORDER.

    ``candidate_tree=None`` reuses the reference samples for a byte-identical
    submission. A reference failure is a failed verification; a candidate
    failure propagates as the candidate's own scored failure.
    """
    order = FAST_PAIR_ORDER if suite is FAST_SUITE else PAIR_ORDER
    candidate_samples: list[list[dict[str, Any]]] = [[] for _ in suite]
    reference_samples: list[list[dict[str, Any]]] = [[] for _ in suite]
    schedule: list[list[str]] = []
    for first in order:
        roles = ("reference",) if candidate_tree is None else (
            (first, "candidate" if first == "reference" else "reference"))
        schedule.append(list(roles))
        for index, case in enumerate(suite):
            for role in roles:
                if role == "reference":
                    try:
                        reference_samples[index].append(_run_once(reference_tree, case))
                    except Exception as error:  # includes PolicyRejection
                        raise ReferenceVerificationError("reference benchmark failed") from error
                else:
                    candidate_samples[index].append(_run_once(candidate_tree, case))
    score_repetitions = 1 if suite is FAST_SUITE else 2
    reference_run = _summarize(reference_samples, score_repetitions)
    candidate_run = (reference_run if candidate_tree is None
                     else _summarize(candidate_samples, score_repetitions))
    return candidate_run, reference_run, schedule


def _run_suite(tree: Path, suite: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Sequential best-of-two suite, kept unchanged for the per-leaf baseline."""
    repetitions = 1 if suite is FAST_SUITE else 2
    elapsed = 0.0
    work = 0
    worst_rmse = 0.0
    worst_prediction = 0.0
    worst_per_row = 0.0
    case_runs: list[dict[str, Any]] = []
    for index, case in enumerate(suite):
        samples = [_run_once(tree, case) for _ in range(repetitions)]
        representative = min(samples, key=lambda item: float(item["elapsed_seconds"]))
        rmse_error = max(float(item["rmse_error"]) for item in samples)
        prediction_l1 = max(float(item["prediction_l1"]) for item in samples)
        per_row = max(float(item["prediction_l1"]) / max(int(item["rows"]), 1) for item in samples)
        worst_per_row = max(worst_per_row, per_row)
        updates = [item.get("residual_update_seconds") for item in samples]
        case_runs.append({
            "workload": f"workload_{index}",
            "elapsed_seconds": float(representative["elapsed_seconds"]),
            # Best of the same fresh processes, independently of elapsed, and
            # unmeasured if any repetition lacks a trusted update clock.
            "residual_update_seconds": None if None in updates else min(float(v) for v in updates),
            "rmse_error": rmse_error,
            "prediction_l1": prediction_l1,
            "prediction_l1_per_row": per_row,
        })
        elapsed += float(representative["elapsed_seconds"])
        work += int(case["rows"]) * int(case["estimators"])
        worst_rmse = max(worst_rmse, rmse_error)
        worst_prediction = max(worst_prediction, prediction_l1)
    return {
        "elapsed_seconds": elapsed,
        "work": work,
        "rmse_error": worst_rmse,
        "prediction_l1": worst_prediction,
        "prediction_l1_per_row": worst_per_row,
        "case_count": len(suite),
        "workloads": case_runs,
    }


def _workload_decisions(
    candidate_run: dict[str, Any], reference_run: dict[str, Any]
) -> list[dict[str, Any]]:
    candidate_workloads = candidate_run.get("workloads")
    reference_workloads = reference_run.get("workloads")
    if (
        not isinstance(candidate_workloads, list)
        or not isinstance(reference_workloads, list)
        or len(candidate_workloads) != len(reference_workloads)
        or len(candidate_workloads) != candidate_run.get("case_count")
        or len(reference_workloads) != reference_run.get("case_count")
    ):
        raise RuntimeError("benchmark workload results are incomplete")
    rows = []
    for candidate, reference in zip(candidate_workloads, reference_workloads, strict=True):
        if candidate.get("workload") != reference.get("workload"):
            raise RuntimeError("candidate and reference workloads do not align")
        valid = (
            float(candidate["rmse_error"]) <= RMSE_TOLERANCE
            and float(candidate["prediction_l1_per_row"]) <= PREDICTION_L1_TOLERANCE_PER_ROW
        )
        candidate_update = candidate.get("residual_update_seconds")
        reference_update = reference.get("residual_update_seconds")
        rows.append({
            "workload": candidate["workload"],
            "metrics": {
                "elapsed_seconds": bool(
                    valid
                    and float(candidate["elapsed_seconds"]) < float(reference["elapsed_seconds"])
                ),
                "residual_update_seconds": bool(
                    valid
                    and candidate_update is not None
                    and reference_update is not None
                    and float(candidate_update) < float(reference_update)
                ),
                "rmse_error": bool(float(candidate["rmse_error"]) <= RMSE_TOLERANCE),
            },
        })
    return rows


def _require_valid_reference(reference_run: dict[str, Any]) -> None:
    """An incorrect or unmeasured reference never sets the bar."""
    for case in reference_run.get("workloads", []):
        if not (
            float(case["rmse_error"]) <= RMSE_TOLERANCE
            and float(case["prediction_l1_per_row"]) <= PREDICTION_L1_TOLERANCE_PER_ROW
            and case.get("residual_update_seconds") is not None
        ):
            raise ReferenceVerificationError("reference failed the materialized-join oracle")


# SCORE_DECADES is the improvement at which the asymptotic score reaches half
# scale. The fixed per-leaf baseline remains independent of the optimized
# reference used only by the binary verdict.
SCORE_DECADES = 2.0


def _score_elapsed(elapsed_seconds: float, baseline_seconds: float) -> float:
    """Map improvement decades over the fixed baseline asymptotically to [0, 100)."""
    if (
        not math.isfinite(elapsed_seconds)
        or not math.isfinite(baseline_seconds)
        or elapsed_seconds <= 0
        or baseline_seconds <= 0
    ):
        return 0.0
    improvement_decades = max(
        0.0, math.log10(baseline_seconds) - math.log10(elapsed_seconds)
    )
    return 100.0 * improvement_decades / (SCORE_DECADES + improvement_decades)


def _score_run(run: dict[str, Any], baseline_seconds: float | None = None) -> tuple[float, float]:
    """Return the naturally bounded score through the legacy two-value interface."""
    if run["rmse_error"] > RMSE_TOLERANCE or run["prediction_l1_per_row"] > PREDICTION_L1_TOLERANCE_PER_ROW:
        return 0.0, 0.0
    baseline = float(run["elapsed_seconds"]) if baseline_seconds is None else baseline_seconds
    score = _score_elapsed(run["elapsed_seconds"], baseline)
    return score, score


def _cleanup(result: Any) -> None:
    workdir = getattr(result, "workdir", None)
    if workdir:
        shutil.rmtree(Path(workdir), ignore_errors=True)


_BASELINE: dict[str, float] = {}


def _baseline_seconds(suite) -> float:
    """Time the obvious per-leaf implementation, once per process, on this suite.

    This is the zero of the scale. It used to be the withheld implementation
    itself, which put the reference at 0 and left every correct-but-slower
    submission there beside it with nothing to order them by — and needed an
    exemption from the reference-floor check plus a byte-identical short circuit
    to survive preflight at all. Three special cases to hold up one anchor.

    Measured on the eight-workload real-row development suite (2026-10-11, two
    pinned cores), the per-leaf version takes about 87s against the batched
    version's 82s: the batching alone is worth about 6% of runtime. That is a fact about the
    workload rather than about the scale, and it is what this number reports.
    """
    key = "final" if os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final" else "dev"
    if key in _BASELINE:
        return _BASELINE[key]
    applied = apply_submission_patch(str(BASELINE_PATCH), TASK_DIR, CONFIG)
    if not applied.ok or applied.workdir is None:
        raise RuntimeError("bundled baseline patch could not be prepared")
    try:
        _BASELINE[key] = float(_run_suite(Path(applied.workdir), suite)["elapsed_seconds"])
    finally:
        _cleanup(applied)
    return _BASELINE[key]


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    candidate_result = None
    reference_result = None
    reference_score = 0.0
    try:
        if not _archive_patch(solution_path):
            return 0.0, 0.0, "submission could not be archived", _empty_metrics()
        suite = _suite()
        reference_result = apply_submission_patch(str(REFERENCE_PATCH), TASK_DIR, CONFIG)
        if not reference_result.ok:
            raise ReferenceVerificationError("reference patch could not be prepared")
        ok, reason = _structural_policy(Path(reference_result.workdir))
        if not ok:
            raise ReferenceVerificationError(reason)

        candidate_result = apply_submission_patch(solution_path, TASK_DIR, CONFIG)
        if not candidate_result.ok:
            return 0.0, 0.0, str(candidate_result.message), _empty_metrics(reference_score, 0.0)
        ok, reason = _patch_policy(Path(solution_path))
        if not ok:
            return 0.0, 0.0, reason, _empty_metrics(reference_score, 0.0)
        ok, reason = _structural_policy(Path(candidate_result.workdir))
        if not ok:
            return 0.0, 0.0, reason, _empty_metrics(reference_score, 0.0)

        # Untimed warm-up: one discarded reference process on the first workload
        # absorbs cold-start effects (page cache, lazy library loading) before
        # any timed baseline, reference or candidate sample, as the published
        # schedule's leading reference block did.
        # The materialized-join oracle of each case is fitted once, in its own
        # process, before any timed run; timed processes only read it back.
        try:
            for case in suite:
                _prepare_oracle(Path(reference_result.workdir), case)
            _run_once(Path(reference_result.workdir), suite[0])
        except Exception as error:
            raise ReferenceVerificationError("reference warm-up failed") from error
        try:
            baseline_seconds = _baseline_seconds(suite)
        except Exception as error:
            raise ReferenceVerificationError("bundled baseline failed") from error
        # A byte-identical reference submission is the same implementation, not a
        # second noisy timing sample. Reuse its paired run so it scores exactly
        # what the reference scores against the baseline, with no spread.
        identical = Path(solution_path).read_bytes() == REFERENCE_PATCH.read_bytes()
        candidate_run, reference_run, schedule = _run_paired(
            None if identical else Path(candidate_result.workdir),
            Path(reference_result.workdir), suite)
        _require_valid_reference(reference_run)
        reference_score, _ = _score_run(reference_run, baseline_seconds)
        score, score_aux = _score_run(candidate_run, baseline_seconds)
        decisions = _workload_decisions(candidate_run, reference_run)
        metrics = _empty_metrics(reference_score, score)
        metrics.update({
            "beats_reference": int(all(
                value for row in decisions for value in row["metrics"].values()
            )),
            "beat_spec_workload_passes": decisions,
            "paired_schedule": schedule,
            # Matching the reference is now 0, so a score of 0 no longer implies a
            # correctness failure — ask the tolerances that decided it instead.
            "correct": 1 if (candidate_run["rmse_error"] <= RMSE_TOLERANCE
                             and candidate_run["prediction_l1_per_row"] <= PREDICTION_L1_TOLERANCE_PER_ROW) else 0,
            "speedup_over_reference": round(
                reference_run["elapsed_seconds"] / max(candidate_run["elapsed_seconds"], 1e-9), 4),
            "baseline_seconds": round(baseline_seconds, 6),
            "speedup_over_baseline": round(
                baseline_seconds / max(candidate_run["elapsed_seconds"], 1e-9), 4),
            "elapsed_seconds": round(candidate_run["elapsed_seconds"], 6),
            "case_count": candidate_run["case_count"],
            "rmse_error": candidate_run["rmse_error"],
            "prediction_l1": candidate_run["prediction_l1"],
            "prediction_l1_per_row": candidate_run["prediction_l1_per_row"],
            "workload_measurements": [
                {
                    "workload": candidate["workload"],
                    "candidate_elapsed_seconds": candidate["elapsed_seconds"],
                    "reference_elapsed_seconds": reference["elapsed_seconds"],
                    "candidate_residual_update_seconds": candidate["residual_update_seconds"],
                    "reference_residual_update_seconds": reference["residual_update_seconds"],
                    "candidate_rmse_error": candidate["rmse_error"],
                    "candidate_prediction_l1": candidate["prediction_l1"],
                    "candidate_elapsed_samples": candidate["elapsed_samples"],
                    "reference_elapsed_samples": reference["elapsed_samples"],
                    "candidate_residual_update_samples": candidate["residual_update_samples"],
                    "reference_residual_update_samples": reference["residual_update_samples"],
                }
                for candidate, reference in zip(
                    candidate_run["workloads"], reference_run["workloads"], strict=True
                )
            ],
        })
        if candidate_run["rmse_error"] > RMSE_TOLERANCE or candidate_run["prediction_l1_per_row"] > PREDICTION_L1_TOLERANCE_PER_ROW:
            return 0.0, 0.0, "predictions do not match the materialized-join oracle", metrics
        return score, score_aux, "correct ensemble; score reflects paired fixed-workload training time", metrics
    except ReferenceVerificationError:
        # Failed verification, not a candidate score.
        raise
    except PolicyRejection as rejection:
        return 0.0, 0.0, POLICY_REASONS[rejection.code], _empty_metrics(reference_score, 0.0)
    except subprocess.TimeoutExpired:
        return 0.0, 0.0, "training timed out", _empty_metrics(reference_score, 0.0)
    except Exception:
        return 0.0, 0.0, "evaluation failed", _empty_metrics(reference_score, 0.0)
    finally:
        if candidate_result is not None:
            _cleanup(candidate_result)
        if reference_result is not None:
            _cleanup(reference_result)
        global _ORACLE_DIR
        if _ORACLE_DIR is not None:
            shutil.rmtree(_ORACLE_DIR, ignore_errors=True)
            _ORACLE_DIR = None


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: evaluator.py SOLUTION_PATCH", file=sys.stderr)
        return 2
    score, unbounded, message, metrics = evaluate(sys.argv[1])
    print(json.dumps({"score": score, "score_unbounded": unbounded, "message": message, "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
