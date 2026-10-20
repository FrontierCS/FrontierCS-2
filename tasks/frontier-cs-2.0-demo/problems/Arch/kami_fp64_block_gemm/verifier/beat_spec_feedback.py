"""Turn private per-workload beat-spec decisions into safe solver feedback.

Evaluators keep the exact workload decisions in their metrics so the benchmark
can audit ``beats_reference``.  A development submission receives only aggregate
counts and percentages: neither workload identities nor reference measurements
cross the judge boundary.

This module is copied into both runtime adapters (Harbor and FrontierCS2).  Keep
the policy here rather than reimplementing it in every generated task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

PRIVATE_WORKLOAD_PASSES = "beat_spec_workload_passes"
PUBLIC_PROGRESS = "beat_spec_progress"


def _load_spec(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("beat_spec.json is not readable JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("beat_spec.json must contain a JSON object")
    return payload


def _names(entries: object, *, field: str) -> list[str]:
    if not isinstance(entries, list):
        raise ValueError(f"beat_spec.json {field} must be a list")
    names: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError(f"beat_spec.json {field} entries must have string names")
        name = entry["name"].strip()
        if not name or name in names:
            raise ValueError(f"beat_spec.json {field} contains an empty or duplicate name")
        names.append(name)
    return names


def _safe_metric(key: str) -> bool:
    lowered = key.lower()
    return not (
        key == PRIVATE_WORKLOAD_PASSES
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

def _progress(metrics: dict[str, Any], metric_names: list[str]) -> dict[str, Any]:
    rows = metrics.get(PRIVATE_WORKLOAD_PASSES)
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"metrics.{PRIVATE_WORKLOAD_PASSES} must be a non-empty list"
        )

    expected_metrics = set(metric_names)
    seen_workloads: set[str] = set()
    passed = {name: 0 for name in metric_names}
    all_pass = True

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"metrics.{PRIVATE_WORKLOAD_PASSES}[{index}] must be an object"
            )
        workload = row.get("workload")
        if not isinstance(workload, str) or not workload.strip():
            raise ValueError(
                f"metrics.{PRIVATE_WORKLOAD_PASSES}[{index}].workload must be a non-empty string"
            )
        if workload in seen_workloads:
            raise ValueError(
                f"metrics.{PRIVATE_WORKLOAD_PASSES} contains duplicate workload {workload!r}"
            )
        seen_workloads.add(workload)

        decisions = row.get("metrics")
        if not isinstance(decisions, dict) or set(decisions) != expected_metrics:
            raise ValueError(
                f"metrics.{PRIVATE_WORKLOAD_PASSES}[{index}].metrics must contain exactly "
                f"{sorted(expected_metrics)!r}"
            )
        for name in metric_names:
            decision = decisions[name]
            if not isinstance(decision, bool):
                raise ValueError(
                    f"metrics.{PRIVATE_WORKLOAD_PASSES}[{index}].metrics.{name} must be boolean"
                )
            passed[name] += int(decision)
            all_pass = all_pass and decision

    actual = metrics.get("beats_reference")
    if isinstance(actual, bool):
        actual = int(actual)
    if not isinstance(actual, int) or actual not in (0, 1):
        raise ValueError("metrics.beats_reference must be 0 or 1")
    if actual != int(all_pass):
        raise ValueError(
            "metrics.beats_reference does not equal the conjunction of "
            f"metrics.{PRIVATE_WORKLOAD_PASSES}"
        )

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


def public_metrics(
    metrics: dict[str, Any],
    *,
    beat_spec_path: Path,
    configured_visible: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return solver-safe metrics, adding aggregate beat-spec progress when present.

    ``configured_visible`` is Harbor's config allowlist.  FrontierCS2 does not
    parse task YAML in its generic adapter, so it uses the equivalent allowlist
    stored in ``beat_spec.json``.  A mismatch fails closed instead of exposing a
    field that the design pass did not approve.
    """
    spec = _load_spec(beat_spec_path)
    declared = None if configured_visible is None else [str(name) for name in configured_visible]

    if spec is None:
        allowed = set(declared) if declared is not None else None
        return {
            key: value
            for key, value in metrics.items()
            if (allowed is None or key in allowed) and _safe_metric(key)
        }

    metric_names = _names(spec.get("metrics"), field="metrics")
    if not metric_names:
        raise ValueError("beat_spec.json must name at least one metric")
    spec_visible = _names(spec.get("solver_feedback", []), field="solver_feedback")
    if declared is not None and set(declared) != set(spec_visible):
        raise ValueError(
            "evaluation.solver_visible_metrics does not match beat_spec.json solver_feedback"
        )

    allowed = set(spec_visible)
    result = {
        key: value
        for key, value in metrics.items()
        if key in allowed and _safe_metric(key)
    }
    result[PUBLIC_PROGRESS] = _progress(metrics, metric_names)
    workload_ids = {row["workload"] for row in metrics[PRIVATE_WORKLOAD_PASSES]}
    for key, value in result.items():
        if key != PUBLIC_PROGRESS and _contains_private_workload_identity(value, workload_ids):
            raise ValueError("solver-visible metric contains a private workload identity")
    return result
