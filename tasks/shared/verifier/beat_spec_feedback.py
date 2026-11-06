"""Train feedback and operator results from an evaluator's beat-spec decisions.

An evaluator returns, in its metrics, the exact per-workload decisions
``beat_spec_workload_passes`` (one row per workload: ``{"workload": id, "metrics":
{name: bool}}``, the same metric names in every row) and its binary reward
``beats_reference``. Those stay in private evidence. A train result carries only the
aggregate counts and percentages per metric (``beat_spec_progress``) and the evaluator
metrics the task's ``[feedback]`` table names (config.yaml
``evaluation.solver_visible_metrics``); neither workload identities nor reference
measurements cross the judge boundary. There is no per-task spec file: the evaluator's
rows define the metric names and its ``beats_reference`` is the authority.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

PRIVATE_WORKLOAD_PASSES = "beat_spec_workload_passes"
PUBLIC_PROGRESS = "beat_spec_progress"


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _safe_metric(key: str) -> bool:
    lowered = key.lower()
    return not (
        key in {PRIVATE_WORKLOAD_PASSES, PUBLIC_PROGRESS}
        or "reference" in lowered
        or "margin" in lowered
        or lowered.startswith("beats")
    )


def _contains_private_workload_identity(value: Any, workload_ids: set[str]) -> bool:
    """Whether a nested string or map key equals an exact private identity.

    Identity equality is intentional: arbitrary absolute feedback can naturally
    contain an identifier as a substring, while returning the exact identifier
    as a string or structured key lets a solver join feedback to a private row.
    """
    if isinstance(value, str):
        return value in workload_ids
    if isinstance(value, dict):
        return any(
            _contains_private_workload_identity(key, workload_ids)
            or _contains_private_workload_identity(item, workload_ids)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(
            _contains_private_workload_identity(item, workload_ids) for item in value
        )
    return False


def _beats_reference(metrics: dict[str, Any]) -> int:
    actual = metrics.get("beats_reference")
    if isinstance(actual, bool):
        actual = int(actual)
    if not isinstance(actual, int) or actual not in (0, 1):
        raise ValueError("metrics.beats_reference must be 0 or 1")
    return actual


def progress(metrics: dict[str, Any]) -> dict[str, Any]:
    """Aggregate the evaluator's workload decisions per metric."""
    rows = metrics.get(PRIVATE_WORKLOAD_PASSES)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"metrics.{PRIVATE_WORKLOAD_PASSES} must be a non-empty list")

    metric_names: list[str] = []
    seen_workloads: set[str] = set()
    passed: dict[str, int] = {}
    for index, row in enumerate(rows):
        where = f"metrics.{PRIVATE_WORKLOAD_PASSES}[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{where} must be an object")
        workload = row.get("workload")
        if not isinstance(workload, str) or not workload.strip():
            raise ValueError(f"{where}.workload must be a non-empty string")
        if workload in seen_workloads:
            raise ValueError(
                f"metrics.{PRIVATE_WORKLOAD_PASSES} contains duplicate workload {workload!r}")
        seen_workloads.add(workload)
        decisions = row.get("metrics")
        if not isinstance(decisions, dict) or not decisions:
            raise ValueError(f"{where}.metrics must be a non-empty object")
        if not metric_names:
            metric_names = list(decisions)
            if not all(isinstance(name, str) and name.strip() for name in metric_names):
                raise ValueError(f"{where}.metrics names must be non-empty strings")
            passed = {name: 0 for name in metric_names}
        elif set(decisions) != set(metric_names):
            raise ValueError(
                f"{where}.metrics must contain exactly {sorted(metric_names)!r}, "
                "the metrics of every other workload")
        for name in metric_names:
            if not isinstance(decisions[name], bool):
                raise ValueError(f"{where}.metrics.{name} must be boolean")
            passed[name] += int(decisions[name])
    _beats_reference(metrics)

    total = len(rows)
    return {
        name: {
            "passed": passed[name],
            "failed": total - passed[name],
            "total": total,
            "pass_percent": round(100.0 * passed[name] / total, 2),
            "fail_percent": round(100.0 * (total - passed[name]) / total, 2),
        }
        for name in metric_names
    }


def public_metrics(metrics: dict[str, Any], *, visible: Iterable[str]) -> dict[str, Any]:
    """The train result's metrics: the allowlisted evaluator values (task.toml [feedback],
    passed in from config.yaml) and the aggregate beat-spec progress."""
    allowed = {str(name) for name in visible}
    result = {
        key: value
        for key, value in metrics.items()
        if key in allowed and _safe_metric(key)
    }
    result[PUBLIC_PROGRESS] = progress(metrics)
    workload_ids = {row["workload"] for row in metrics[PRIVATE_WORKLOAD_PASSES]}
    for key, value in result.items():
        if (key != PUBLIC_PROGRESS
                and _contains_private_workload_identity(value, workload_ids)):
            raise ValueError(
                f"solver-visible metric {key!r} equals a private workload identifier "
                "in a nested string or map key"
            )
    return result


def private_metrics(metrics: dict[str, Any]) -> dict[str, Any] | None:
    """Return validated operator-only metrics for a scored final evaluation.

    The caller controls mode, diagnostic exclusions, transport size, and error
    redaction. Neither these reference measurements nor this envelope belong in
    solver-visible feedback.
    """
    if "beats_reference" not in metrics:
        return None
    result = {
        PUBLIC_PROGRESS: progress(metrics),
        "beats_reference": _beats_reference(metrics),
    }
    for name in ("reference_score", "margin"):
        if not _finite_number(metrics.get(name)):
            raise ValueError(f"metrics.{name} must be a finite number")
        result[name] = float(metrics[name])
    return result
