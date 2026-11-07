"""Approved instance end-to-end nearest-rank percentiles, independent of scoring."""
from __future__ import annotations

import math
from collections.abc import Iterable


def tail_latencies(samples: Iterable[float]) -> dict[str, float]:
    """Return exact observations at ceil(p*N); do not omit invalid samples.

    Callers must resolve full-instance coverage before invoking this helper.
    Rejecting unusable timing data here does not choose its score or penalty.
    """
    values = []
    for sample in samples:
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            raise ValueError("latencies must be finite positive numbers")
        value = float(sample)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("latencies must be finite positive numbers")
        values.append(value)
    if not values:
        raise ValueError("at least one measured latency is required")
    values.sort()
    n = len(values)
    # Integer arithmetic preserves exact nearest-rank boundaries.
    return {f"p{p}": values[(p * n + 99) // 100 - 1] for p in (90, 95, 99)}
