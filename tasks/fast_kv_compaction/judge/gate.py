"""Fixed-target gate arithmetic for fast_kv_compaction (pure Python, no torch).

Targets come from a frozen manifest of at least three seeded reference captures
plus uncompressed-cache runs (see calibrate.py). An ordinary evaluation runs the
candidate only and compares each article with those targets.

Per article w, with N(w) questions and medians taken over the captures:

  qa_correct           C(w) >= median R_correct(w) - max(1, ceil(ACCURACY_TOLERANCE * N(w)))
  compaction_seconds   time mode:  T_C(w) <  median T_R(w)
                       bytes mode: T_C(w) <= median T_R(w)
  compact_cache_bytes  time mode:  B_C(w) <= B_R(w)
                       bytes mode: B_C(w) <  B_R(w)

The mode is fixed for the whole submission: bytes mode iff the candidate's
declared TARGET_SIZE is below the reference's. There is no other margin: the
median of the reference's own repeated measurements is the bar, and the cache
footprint is deterministic.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from statistics import median
from typing import Any

MANIFEST_SCHEMA = "frontier.fast-kv-compaction-targets/v1"
READY = "READY"
MIN_CAPTURES = 3
# Machine-speed anchor: trusted HF prefill (summed over the suite) and a trusted
# vLLM generation benchmark; their geometric-mean ratio to calibration rescales
# the time targets, and a deviation beyond this band is an infrastructure fault.
MAX_ANCHOR_DEVIATION = 0.10

ACCURACY_TOLERANCE = 0.05
REFERENCE_TARGET_SIZE = 0.1
METRICS = ("qa_correct", "compaction_seconds", "compact_cache_bytes")
TIME_MODE = "time"
BYTES_MODE = "bytes"


SOLVER_ENTRY = "compaction/solver_entry.py"


class TargetsUnavailable(RuntimeError):
    """The frozen targets are missing, on HOLD or malformed: no verdict exists."""


class MachineMismatch(RuntimeError):
    """The evaluation hardware does not match calibration: an infrastructure failure."""


class SubmissionDeclarationError(RuntimeError):
    """solver_entry.py does not declare a usable literal TARGET_SIZE."""


def declared_target_size(root: Path) -> float:
    """TARGET_SIZE must be a literal number in solver_entry.py (read, never imported)."""
    path = Path(root) / SOLVER_ENTRY
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise SubmissionDeclarationError("solver_entry.py is unreadable") from exc
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "TARGET_SIZE" for target in targets):
                value = node.value
                if isinstance(value, ast.Constant) and type(value.value) in (int, float):
                    return float(value.value)
                raise SubmissionDeclarationError("TARGET_SIZE must be a literal number")
    raise SubmissionDeclarationError("solver_entry.py does not define TARGET_SIZE")


def accuracy_allowance(questions: int) -> int:
    return max(1, math.ceil(ACCURACY_TOLERANCE * questions))


def mode_for(target_size: float) -> str:
    if not (isinstance(target_size, (int, float)) and math.isfinite(target_size)
            and 0 < target_size <= REFERENCE_TARGET_SIZE):
        raise ValueError(f"TARGET_SIZE must be in (0, {REFERENCE_TARGET_SIZE}]")
    return BYTES_MODE if target_size < REFERENCE_TARGET_SIZE else TIME_MODE


def _positive(values: Any, kind: type, name: str) -> list:
    if not isinstance(values, list) or not values:
        raise TargetsUnavailable(f"{name} must be a non-empty list")
    for value in values:
        if type(value) is bool or not isinstance(value, kind):
            raise TargetsUnavailable(f"{name} has a non-{kind.__name__} value")
        if isinstance(value, float) and not math.isfinite(value):
            raise TargetsUnavailable(f"{name} has a non-finite value")
    return values


def validate_manifest(manifest: Any, workload_ids: tuple[str, ...]) -> dict[str, dict]:
    """Return per-workload targets {n, correct, seconds, bytes}; fail closed."""
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise TargetsUnavailable("target manifest schema mismatch")
    if manifest.get("status") != READY:
        raise TargetsUnavailable("target manifest is not READY (calibration pending)")
    if manifest.get("reference_target_size") != REFERENCE_TARGET_SIZE:
        raise TargetsUnavailable("target manifest reference budget mismatch")
    captures = manifest.get("captures")
    if not isinstance(captures, list) or len(captures) < MIN_CAPTURES:
        raise TargetsUnavailable(f"target manifest needs at least {MIN_CAPTURES} captures")
    seeds = [capture.get("seed") if isinstance(capture, dict) else None for capture in captures]
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise TargetsUnavailable("reference captures need distinct integer seeds")
    rows = manifest.get("workloads")
    if not isinstance(rows, dict):
        raise TargetsUnavailable("target manifest workloads must be an object")
    targets = {}
    for workload in workload_ids:
        row = rows.get(workload)
        if not isinstance(row, dict):
            raise TargetsUnavailable(f"no target for workload {workload}")
        n = row.get("questions")
        if type(n) is not int or n <= 0:
            raise TargetsUnavailable(f"invalid question count for {workload}")
        correct = _positive(row.get("reference_correct"), int, f"{workload} reference_correct")
        seconds = _positive(row.get("reference_seconds"), float, f"{workload} reference_seconds")
        footprint = _positive(row.get("reference_bytes"), int, f"{workload} reference_bytes")
        uncompressed = (_positive(row["uncompressed_correct"], int, f"{workload} uncompressed_correct")
                        if row.get("uncompressed_correct", []) != [] else [])
        anchors = _positive(row.get("anchor_seconds"), float, f"{workload} anchor_seconds")
        if any(value <= 0 for value in anchors):
            raise TargetsUnavailable(f"{workload} anchor times must be positive")
        if not len(correct) == len(seconds) == len(footprint) == len(captures):
            raise TargetsUnavailable(f"{workload} needs one value per capture")
        if any(not 0 <= value <= n for value in correct + uncompressed):
            raise TargetsUnavailable(f"{workload} correct counts out of range")
        if any(value <= 0 for value in seconds) or any(value <= 0 for value in footprint):
            raise TargetsUnavailable(f"{workload} timings and bytes must be positive")
        if len(set(footprint)) != 1:
            raise TargetsUnavailable(f"{workload} reference cache bytes are not deterministic")
        targets[workload] = {
            "questions": n,
            "correct": float(median(correct)),
            "seconds": float(median(seconds)),
            "bytes": footprint[0],
            "uncompressed_correct": float(median(uncompressed)) if uncompressed else None,
            "anchor_seconds": float(median(anchors)),
        }
    return targets


def manifest_device(manifest: dict) -> str:
    device = manifest.get("device")
    if not isinstance(device, str) or not device:
        raise TargetsUnavailable("target manifest does not record its GPU")
    return device


def manifest_engine_anchor(manifest: dict) -> float:
    values = _positive(manifest.get("engine_anchor_seconds"), float, "engine_anchor_seconds")
    if any(value <= 0 for value in values):
        raise TargetsUnavailable("engine anchor times must be positive")
    return float(median(values))


def _instance(row: dict) -> str:
    return str(row.get("instance", 0))


def check_machine(device: str, expected_device: str, rows: list[dict], targets: dict,
                  instance_engines: dict, expected_engine_anchor: float) -> dict[str, float]:
    """Per-instance speed ratios to calibration; returns {workload: its instance's ratio}.

    Each instance's ratio is the geometric mean of its trusted HF-prefill anchors
    (summed over its articles, against their calibration anchors) and its vLLM
    engine anchor. A different GPU model is an infrastructure fault;
    anchor deviations are diagnostic only and never reject a measured instance.
    """
    if device != expected_device:
        raise MachineMismatch(f"evaluation GPU {device!r} differs from calibration {expected_device!r}")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(_instance(row), []).append(row)
    ratios = {}
    for instance, members in groups.items():
        observed = sum(float(row["anchor_seconds"]) for row in members)
        expected = sum(targets[row["workload"]]["anchor_seconds"] for row in members)
        engine = instance_engines.get(instance) if isinstance(instance_engines, dict) else None
        if (engine is None or not math.isfinite(observed) or observed <= 0
                or not math.isfinite(float(engine)) or float(engine) <= 0):
            raise MachineMismatch(f"machine-speed anchor of instance {instance} was not measured")
        ratio = math.sqrt((observed / expected) * (float(engine) / expected_engine_anchor))
        for row in members:
            ratios[row["workload"]] = ratio
    return ratios


def scale_targets(targets: dict[str, dict], ratios: dict[str, float]) -> dict[str, dict]:
    """Time targets measured at calibration, rescaled to each article's own instance."""
    return {workload: dict(target, seconds=target["seconds"] * ratios[workload]) if workload in ratios
            else dict(target) for workload, target in targets.items()}


def decide(mode: str, candidate: dict, target: dict) -> dict[str, bool]:
    """One article's three comparisons."""
    if mode not in (TIME_MODE, BYTES_MODE):
        raise ValueError("unknown comparison mode")
    if candidate["questions"] != target["questions"]:
        raise ValueError("question coverage differs from the frozen target")
    qa = candidate["correct"] >= target["correct"] - accuracy_allowance(target["questions"])
    seconds = candidate["seconds"]
    footprint = candidate["bytes"]
    if mode == TIME_MODE:
        time_ok = seconds < target["seconds"]
        bytes_ok = footprint <= target["bytes"]
    else:
        time_ok = seconds <= target["seconds"]
        bytes_ok = footprint < target["bytes"]
    return {"qa_correct": bool(qa), "compaction_seconds": bool(time_ok),
            "compact_cache_bytes": bool(bytes_ok)}


def score_from_ratio(ratio: float) -> float:
    """Bounded improvement credit: 0 at parity or worse, 50 at 10x, below 100."""
    if not math.isfinite(ratio) or ratio <= 1.0:
        return 0.0
    return 100.0 * math.log(ratio) / (math.log(10.0) + math.log(ratio))


def article_score(mode: str, candidate: dict, target: dict) -> float:
    """Continuous credit for one article: zero unless accuracy holds."""
    if not decide(mode, candidate, target)["qa_correct"]:
        return 0.0
    if mode == TIME_MODE:
        if candidate["bytes"] > target["bytes"]:
            return 0.0
        return score_from_ratio(target["seconds"] / candidate["seconds"])
    if candidate["seconds"] > target["seconds"]:
        return 0.0
    return score_from_ratio(target["bytes"] / candidate["bytes"])


def candidate_row(record: Any) -> dict:
    """Validate one harness record into {questions, correct, seconds, bytes}.

    correct is the minimum over the repetitions' caches (each one is answered).
    """
    if not isinstance(record, dict):
        raise ValueError("harness record must be an object")
    n, correct = record.get("questions"), record.get("correct")
    seconds, footprint = record.get("seconds"), record.get("bytes")
    if type(n) is not int or n <= 0 or type(correct) is not int or not 0 <= correct <= n:
        raise ValueError("harness record has invalid question counts")
    per_repetition = record.get("correct_per_repetition")
    if (not isinstance(per_repetition, list) or not per_repetition
            or any(type(value) is not int for value in per_repetition) or min(per_repetition) != correct):
        raise ValueError("harness record must score every repetition and gate on the minimum")
    if not isinstance(seconds, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("harness record has invalid compaction time")
    if type(footprint) is not int or footprint <= 0:
        raise ValueError("harness record has invalid cache bytes")
    return {"questions": n, "correct": correct, "seconds": seconds, "bytes": footprint}
