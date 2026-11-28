"""Confirmed original continuous accuracy penalty, relative to vanilla only."""
from __future__ import annotations

import math


def accuracy_multiplier(vanilla_accuracy: float, candidate_accuracy: float) -> float:
    for value in (vanilla_accuracy, candidate_accuracy):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("accuracy must be a finite fraction in [0, 1]")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("accuracy must be a finite fraction in [0, 1]")
    abs_drop = max(0.0, vanilla_accuracy - candidate_accuracy)
    base = max(vanilla_accuracy, 1e-9)
    rel_drop = max(0.0, abs_drop / base)
    if abs_drop <= 0.05 or rel_drop <= 0.05:
        return 1.0
    return max(0.0, min(1.0, 0.05 / rel_drop))
