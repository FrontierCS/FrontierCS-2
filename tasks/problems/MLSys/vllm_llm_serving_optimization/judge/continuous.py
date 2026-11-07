"""Confirmed continuous score; optimized reference and tail PASS stay separate."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from paired_timing import paired_speedups, score_from_speedup
from quality_penalty import accuracy_multiplier


@dataclass(frozen=True)
class RateMeasurement:
    expected_ids: tuple[str, ...]
    vanilla_latency: Mapping[str, float]
    candidate_latency: Mapping[str, float]
    vanilla_correct: int
    candidate_correct: int
    candidate_api_error_ids: tuple[str, ...] = ()


def workload_continuous(groups: Mapping[str, RateMeasurement], *, mode: str) -> dict:
    expected_groups = {'public'} if mode == 'train' else {'public', 'hidden'}
    if mode not in {'train', 'eval'} or set(groups) != expected_groups:
        raise ValueError('continuous requires the complete phase-specific arrival-rate groups')
    log_speedups = []
    total = vanilla_correct = candidate_correct = 0
    per_group = {}
    for name in sorted(expected_groups):
        group = groups[name]
        n = len(group.expected_ids)
        for correct in (group.vanilla_correct, group.candidate_correct):
            if isinstance(correct, bool) or not isinstance(correct, int) or not 0 <= correct <= n:
                raise ValueError('correct counts must be integers within the fixed denominator')
        paired = paired_speedups(group.expected_ids, group.vanilla_latency, group.candidate_latency, candidate_api_error_ids=group.candidate_api_error_ids)
        log_s = math.fsum(math.log(x) for x in paired.ratios.values()) / n
        log_speedups.append(log_s)
        total += n
        vanilla_correct += group.vanilla_correct
        candidate_correct += group.candidate_correct
        per_group[name] = {'speedup': math.exp(log_s), 'instances': n, 'unusable': paired.unusable, 'timing_complete': paired.complete}
    speedup = math.exp(math.fsum(log_speedups) / len(log_speedups))
    vanilla_accuracy = vanilla_correct / total
    candidate_accuracy = candidate_correct / total
    quality = accuracy_multiplier(vanilla_accuracy, candidate_accuracy)
    latency_score = score_from_speedup(speedup)
    return {'speedup': speedup, 'latency_score': latency_score, 'vanilla_accuracy': vanilla_accuracy, 'candidate_accuracy': candidate_accuracy, 'accuracy_multiplier': quality, 'score': latency_score * quality, 'instance_evaluations': total, 'groups': per_group}


def continuous_score(workloads: Mapping[str, Mapping[str, RateMeasurement]], *, mode: str) -> dict:
    if set(workloads) != {'swe', 'bfcl'}:
        raise ValueError('both SWE and BFCL are required; weights are never renormalized')
    swe = workload_continuous(workloads['swe'], mode=mode)
    bfcl = workload_continuous(workloads['bfcl'], mode=mode)
    return {'score': 0.8 * swe['score'] + 0.2 * bfcl['score'], 'swe': swe, 'bfcl': bfcl}
