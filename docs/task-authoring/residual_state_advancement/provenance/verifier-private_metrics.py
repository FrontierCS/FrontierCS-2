"""Transport Residual State's per-workload conjunction without Agent exposure."""
from __future__ import annotations

import json
import math
from pathlib import Path

from beat_spec_feedback import public_metrics


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def final_metrics(metrics: dict, spec_path: Path) -> dict | None:
    """Return the bounded eval-only envelope for an actually measured candidate."""
    elapsed = metrics.get("elapsed_seconds")
    if not _finite_number(elapsed) or float(elapsed) <= 0.0:
        return None

    verdict = metrics.get("beats_reference")
    if type(verdict) is not int or verdict not in (0, 1):
        raise ValueError("invalid final reference metrics")
    reference_score = metrics.get("reference_score")
    margin = metrics.get("margin")
    if not _finite_number(reference_score) or not _finite_number(margin):
        raise ValueError("invalid final reference metrics")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    expected_workloads = spec.get("workload_count")
    if type(expected_workloads) is not int or expected_workloads <= 0:
        raise ValueError("invalid final reference metrics")

    visible = public_metrics(metrics, beat_spec_path=spec_path)
    progress = visible.get("beat_spec_progress")
    if not isinstance(progress, dict) or not progress:
        raise ValueError("invalid final reference metrics")
    if any(
        not isinstance(summary, dict) or summary.get("total") != expected_workloads
        for summary in progress.values()
    ):
        raise ValueError("invalid final reference metrics")

    aggregation = spec.get("beats_reference_aggregation")
    if aggregation != {"kind": "conjunction"}:
        raise ValueError("invalid final reference metrics")

    result = {
        "beats_reference": verdict,
        # Retain the original continuous-score diagnostics for the platform
        # envelope; neither field determines the conjunction.
        "reference_score": float(reference_score),
        "margin": float(margin),
        "aggregation": aggregation,
        "beat_spec_progress": progress,
    }
    encoded = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > 16384:
        raise ValueError("invalid final reference metrics")
    return result
