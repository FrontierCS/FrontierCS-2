"""Transport BABBLE's existing verdict; never derive margin from score difference."""
from __future__ import annotations

import json
import math
from pathlib import Path

from beat_spec_feedback import public_metrics


def final_metrics(metrics: dict, spec_path: Path) -> dict | None:
    # BABBLE emits empty observation arrays for a rejected/failed submission.
    # Keep its scored-zero outcome and full evidence, without a measured verdict.
    if not metrics.get("compression_ratio") and not metrics.get("runtime_seconds"):
        return None
    visible = public_metrics(metrics, beat_spec_path=spec_path)
    verdict = metrics.get("beats_reference")
    if type(verdict) is not int or verdict not in (0, 1):
        raise ValueError("invalid final reference metrics")
    for key in ("reference_score", "margin"):
        value = metrics.get(key)
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("invalid final reference metrics")
    spec = json.loads(spec_path.read_text())
    result = {
        "beats_reference": verdict,
        "reference_score": metrics["reference_score"],
        # Source meaning: qualified relative-runtime margin, or -1 sentinel.
        "margin": metrics["margin"],
        "beat_spec_progress": visible["beat_spec_progress"],
        "aggregation": spec["beats_reference_aggregation"],
    }
    # This task has exactly two metrics; its envelope is comfortably below 16KiB.
    # Reject unexpected growth rather than silently changing the source feedback.
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > 16384:
        raise ValueError("invalid final reference metrics")
    return result
