"""Transport Shockwave's all-workload/all-metric reference verdict privately."""
from __future__ import annotations

import json
import math
from pathlib import Path

from beat_spec_feedback import public_metrics


def final_metrics(metrics: dict, spec_path: Path) -> dict | None:
    # Invalid/rejected submissions have no measured rows or reference verdict.
    # Their scored-zero result and full operator evidence remain valid.
    if not metrics.get("makespan_seconds"):
        return None

    visible = public_metrics(metrics, beat_spec_path=spec_path)
    verdict = metrics.get("beats_reference")
    if type(verdict) is not int or verdict not in (0, 1):
        raise ValueError("invalid final reference metrics")
    for key in ("reference_score", "margin"):
        value = metrics.get(key)
        if (type(value) not in (float, int) or isinstance(value, bool)
                or not math.isfinite(value)):
            raise ValueError("invalid final reference metrics")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    aggregation = spec.get("beats_reference_aggregation", {"kind": "conjunction"})
    if not isinstance(aggregation, dict) or not aggregation:
        raise ValueError("invalid final reference metrics")

    result = {
        "beats_reference": verdict,
        "reference_score": metrics["reference_score"],
        "margin": metrics["margin"],
        "beat_spec_progress": visible["beat_spec_progress"],
        "aggregation": aggregation,
    }
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > 16384:
        raise ValueError("invalid final reference metrics")
    return result
