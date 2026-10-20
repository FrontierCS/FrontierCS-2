"""Operator-only phase-specific reference PASS from validated measurements.

This module does not run workloads or expose an Agent-facing result envelope.
The runtime adapter must establish measurement identities and error provenance.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from continuous import RateMeasurement, continuous_score
from instance_manifest import expected_groups
from latency import tail_latencies


class MeasurementError(ValueError):
    """Invalid/incomplete trusted execution; never substitute a scientific zero."""


@dataclass(frozen=True)
class InstanceObservation:
    instance_id: str
    latency_seconds: float | None
    correct: bool | None
    execution: str = 'complete'
    source_status: str | None = None
    timing_bound: object | None = None


VALID_CANDIDATE_EXECUTION = {'complete', 'candidate_error', 'model_case_failure'}


def _positive_finite(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _normalize(records: Sequence[InstanceObservation], expected: tuple[str, ...], *, role: str, group=None) -> dict[str, InstanceObservation]:
    output = {}
    expected_set = set(expected)
    for record in records:
        if record.instance_id not in expected_set or record.instance_id in output:
            raise MeasurementError('unexpected or duplicate measurement identity')
        if record.execution not in VALID_CANDIDATE_EXECUTION:
            raise MeasurementError('platform or unclassified execution failure cannot become a candidate score')
        if record.correct is not None and not isinstance(record.correct, bool):
            raise MeasurementError('correctness must come from a boolean test/answer result')
        if record.execution == 'candidate_error' and record.correct is True:
            raise MeasurementError('candidate execution failure cannot claim a successful correctness result')
        if record.execution == 'complete' and record.correct is None:
            raise MeasurementError('complete execution lacks its correctness result')
        if record.execution == 'model_case_failure':
            from model_case_timing import CensoredTiming
            if (not isinstance(record.timing_bound, CensoredTiming)
                    or record.latency_seconds is not None or record.correct is not False
                    or record.source_status != 'cpu_observed_model_case_failure'
                    or group is None or not group.startswith('swe/')):
                raise MeasurementError('only CPU-owned SWE model cases accept censored timing')
            record.timing_bound.validate(instance_id=record.instance_id, variant=role, group=group)
        elif record.timing_bound is not None:
            raise MeasurementError('measured or C1 observations cannot carry censored model timing')
        if role != 'candidate' and (record.execution not in {'complete', 'model_case_failure'}
                or not _positive_finite(_timing(record))):
            raise MeasurementError('baseline/reference must have complete usable measurements')
        output[record.instance_id] = record
    if role != 'candidate' and set(output) != expected_set:
        raise MeasurementError('baseline/reference is missing an original instance')
    return output



def _timing(record):
    return record.timing_bound.bound_seconds if record.execution == 'model_case_failure' else record.latency_seconds


def _bfcl_regression_gate(vanilla, candidate, ids) -> dict:
    baseline_correct = sum(vanilla[i].correct for i in ids)
    regressions = sum(vanilla[i].correct and not (i in candidate and bool(candidate[i].correct)) for i in ids)
    allowed = max(1, math.ceil(0.05 * len(ids)), math.ceil(0.05 * baseline_correct))
    return {'pass': regressions <= allowed, 'regressions': regressions, 'allowed_regressions': allowed, 'expected_instances': len(ids), 'vanilla_correct': baseline_correct}


def bfcl_regression_gate(vanilla, candidate, *, mode: str, rate: str) -> dict:
    """Original per-instance validity gate, also available to the train adapter."""
    groups = expected_groups(mode)
    key = 'bfcl/' + rate
    if key not in groups:
        raise MeasurementError('unexpected BFCL load group for this phase')
    ids = groups[key]
    v = _normalize(vanilla, ids, role='vanilla', group=key)
    c = _normalize(candidate, ids, role='candidate', group=key)
    return _bfcl_regression_gate(v, c, ids)


def _evaluate_phase(
    vanilla: Mapping[str, Sequence[InstanceObservation]],
    candidate: Mapping[str, Sequence[InstanceObservation]],
    reference: Mapping[str, Sequence[InstanceObservation]],
    *, mode: str,
) -> dict:
    expected = expected_groups(mode)
    if set(vanilla) != set(expected) or set(reference) != set(expected):
        raise MeasurementError('vanilla/reference require all four fixed final groups')
    if not set(candidate).issubset(expected):
        raise MeasurementError('candidate contains an unexpected workload group')
    workloads = {'swe': {}, 'bfcl': {}}
    verdicts = {}
    validity_groups = {}
    for key, ids in expected.items():
        v = _normalize(vanilla[key], ids, role='vanilla', group=key)
        r = _normalize(reference[key], ids, role='reference', group=key)
        c = _normalize(candidate.get(key, ()), ids, role='candidate', group=key)
        candidate_errors = tuple(i for i, row in c.items() if row.execution == 'candidate_error')
        candidate_latencies = {i: _timing(row) for i, row in c.items()}
        workload, rate = key.split('/')
        if workload == 'bfcl':
            validity_groups[key] = _bfcl_regression_gate(v, c, ids)
        workloads[workload][rate] = RateMeasurement(ids, {i: _timing(v[i]) for i in ids}, candidate_latencies, sum(v[i].correct for i in ids), sum(bool(row.correct) for row in c.values()), candidate_errors)
        unusable = [i for i in ids if i not in c or c[i].execution not in {'complete', 'model_case_failure'} or not _positive_finite(_timing(c[i]))]
        reference_tails = tail_latencies(_timing(r[i]) for i in ids)
        quality_ok = sum(bool(row.correct) for row in c.values()) >= sum(r[i].correct for i in ids)
        comparisons = {'paired_timing': False, 'p90': False, 'p95': False, 'p99': False, 'correctness': quality_ok}
        paired_reference_speedup = None
        candidate_tails = None
        if not unusable:
            # Identical case lists make paired geomean > 1 equivalent to strictly
            # lower geometric-mean latency, without any per-instance compensation gate.
            log_ratio = math.fsum(math.log(_timing(r[i])) - math.log(_timing(c[i])) for i in ids) / len(ids)
            paired_reference_speedup = math.exp(log_ratio)
            comparisons['paired_timing'] = log_ratio > 0
            candidate_tails = tail_latencies(_timing(c[i]) for i in ids)
            comparisons.update({p: candidate_tails[p] < reference_tails[p] for p in ('p90', 'p95', 'p99')})
        verdicts[key] = {'pass': not unusable and all(comparisons.values()), 'complete': not unusable, 'expected_instances': len(ids), 'observed_instances': len(c), 'unusable_instances': unusable, 'comparisons': comparisons, 'paired_speedup_over_reference': paired_reference_speedup, 'candidate_tails': candidate_tails, 'reference_tails': reference_tails, 'candidate_correct': sum(bool(row.correct) for row in c.values()), 'reference_correct': sum(r[i].correct for i in ids)}
        censored = {role: {ident: row.timing_bound.provenance() for ident, row in rows.items()
                          if row.execution == 'model_case_failure'}
                    for role, rows in (('vanilla', v), ('reference', r), ('candidate', c))}
        if any(censored.values()):
            verdicts[key]['censored_timing'] = censored
    continuous = continuous_score(workloads, mode=mode)
    valid = all(row['pass'] for row in validity_groups.values())
    reference_pass = all(row['pass'] for row in verdicts.values())
    # Keep pre-validity arithmetic for operator audit; an original BFCL gate
    # failure invalidates the whole submission, not only its BFCL contribution.
    return {'valid': valid, 'pass': valid and reference_pass, 'score': continuous['score'] if valid else 0.0, 'continuous': continuous, 'continuous_is_before_validity': True, 'validity': {'pass': valid, 'gate': None if valid else 'bfcl_correctness', 'groups': validity_groups}, 'reference_verdict': {'pass': reference_pass, 'groups': verdicts, 'expected_groups': len(expected), 'passed_groups': sum(row['pass'] for row in verdicts.values())}}


def evaluate_final(vanilla, candidate, reference) -> dict:
    return _evaluate_phase(vanilla, candidate, reference, mode='eval')


def evaluate_train(vanilla, candidate, reference) -> dict:
    """Original public 5/8 groups, same arithmetic; output is operator-only.

    The Agent feedback adapter must apply its separately reviewed whitelist;
    this does not expose reference values or a private verdict envelope to train.
    """
    return _evaluate_phase(vanilla, candidate, reference, mode='train')
