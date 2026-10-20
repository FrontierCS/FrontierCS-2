"""Confirmed full-denominator timing primitives; continuous aggregation is defined separately."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

FAILURE_SPEEDUP = 1 / 8


@dataclass(frozen=True)
class PairedTiming:
    ratios: dict[str, float]
    unusable: dict[str, str]

    @property
    def complete(self) -> bool:
        """False requires group PASS to fail; True does not establish PASS."""
        return not self.unusable


def paired_speedups(
    expected_ids: Iterable[str],
    vanilla_latency: Mapping[str, float],
    candidate_latency: Mapping[str, float],
    *,
    candidate_api_error_ids: Iterable[str] = (),
) -> PairedTiming:
    # Only errors independently attributed to the submitted vLLM service belong
    # here. Provider auth, lifecycle and platform failures invalidate execution.
    expected = tuple(expected_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected instances must be nonempty and unique")
    expected_set = set(expected)
    if set(vanilla_latency) != expected_set:
        raise ValueError("vanilla timing must cover the exact complete instance set")
    if not set(candidate_latency).issubset(expected_set):
        raise ValueError("candidate timing contains an unexpected instance")
    failed = set(candidate_api_error_ids)
    if not failed.issubset(expected_set):
        raise ValueError("API error identity is outside the expected instance set")
    ratios = {}
    unusable = {}
    for ident in expected:
        baseline = vanilla_latency[ident]
        if isinstance(baseline, bool) or not isinstance(baseline, (int, float)) or not math.isfinite(baseline) or baseline <= 0:
            raise ValueError("vanilla timing must be finite and positive; repair baseline execution")
        candidate = candidate_latency.get(ident)
        reason = None
        if ident in failed:
            reason = 'api_error'
        elif ident not in candidate_latency:
            reason = 'missing'
        elif isinstance(candidate, bool) or not isinstance(candidate, (int, float)) or not math.isfinite(candidate) or candidate <= 0:
            reason = 'unusable_latency'
        if reason is not None:
            ratios[ident] = FAILURE_SPEEDUP
            unusable[ident] = reason
        else:
            ratio = baseline / candidate
            if not math.isfinite(ratio):
                raise ValueError("measured timing ratio is not finite")
            # Valid results retain their real ratio, with neither lower nor upper clipping.
            ratios[ident] = ratio
    return PairedTiming(ratios, unusable)


def score_from_speedup(speedup: float) -> float:
    """Approved scalar mapping; workload/rate composition lives in continuous.py."""
    if isinstance(speedup, bool) or not isinstance(speedup, (int, float)) or not math.isfinite(speedup) or speedup <= 0:
        raise ValueError("speedup must be finite and positive")
    if speedup <= 1:
        return 0.0
    value = math.log(speedup)
    return 100 * value / (math.log(10) + value)


def combine_rate_speedups(public_speedup: float, hidden_speedup: float) -> float:
    """Approved equal-rate geomean within one workload; quality aggregation is separate."""
    values = (public_speedup, hidden_speedup)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("rate speedups must be finite and positive")
    return math.exp((math.log(public_speedup) + math.log(hidden_speedup)) / 2)
