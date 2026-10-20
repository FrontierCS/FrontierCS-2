"""CPU-owned runtime groups; ordinary measurements retain the original arithmetic.

A candidate service fault denies credit to its affected workload. Trusted
measurement failures invalidate scoring; the 0.8/0.2 weights stay fixed.
"""
import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from reference_verdict import (InstanceObservation, MeasurementError, _normalize,
    _timing, _positive_finite, _bfcl_regression_gate)
from continuous import RateMeasurement, continuous_score, workload_continuous
from instance_manifest import expected_groups
from latency import tail_latencies


class TrustedMeasurementError(MeasurementError):
    """CPU-owned missing trusted measurement, never a scientific candidate zero."""
    def __init__(self, role, group, reason):
        self.role, self.group, self.reason = role, group, reason
        super().__init__('trusted measurement unavailable')

    def private_classification(self):
        return {'origin': 'trusted_measurement_unavailable', 'attribution': 'infrastructure',
                'score_ready': False, 'variant': self.role, 'group': self.group,
                'reason': self.reason}


@dataclass(frozen=True)
class RuntimeGroupFailure:
    variant: str
    group: str
    instance_ids: tuple[str, ...]

    def __len__(self):
        return len(self.instance_ids)

    def __iter__(self):
        return iter(InstanceObservation(ident, None, False, execution='candidate_error',
            source_status='cpu_observed_runtime_error') for ident in self.instance_ids)

    def validate(self, *, group, ids):
        if (self.variant not in {'vanilla', 'reference', 'candidate'}
                or self.group != group or self.instance_ids != tuple(ids)):
            raise MeasurementError('CPU runtime failure differs from its original group')


def _evaluate_phase(
    vanilla: Mapping[str, Sequence[InstanceObservation]],
    candidate: Mapping[str, Sequence[InstanceObservation]],
    reference: Mapping[str, Sequence[InstanceObservation]],
    *, mode: str, bfcl_flip_gate: bool = True,
) -> dict:
    expected = expected_groups(mode)
    runtime_errors = {}
    if (not isinstance(vanilla, Mapping) or not isinstance(reference, Mapping)
            or set(vanilla) != set(expected) or set(reference) != set(expected)):
        raise TrustedMeasurementError(None, None, 'group_set_mismatch')
    if not set(candidate).issubset(expected):
        raise MeasurementError('candidate contains an unexpected workload group')
    workloads = {'swe': {}, 'bfcl': {}}
    verdicts = {}
    validity_groups = {}
    for key, ids in expected.items():
        # Trusted measurements are prerequisites, even when the candidate failed.
        # Never turn an unavailable baseline/reference into a scientific zero.
        trusted = {}
        for role, rows in (('vanilla', vanilla[key]), ('reference', reference[key])):
            if isinstance(rows, RuntimeGroupFailure):
                raise TrustedMeasurementError(role, key, 'runtime_failure')
            try:
                trusted[role] = _normalize(rows, ids, role=role, group=key)
            except MeasurementError as error:
                raise TrustedMeasurementError(role, key, 'invalid_or_incomplete_measurement') from error
            except Exception as error:
                # Broken trusted records must not escape as candidate dispositions.
                raise TrustedMeasurementError(role, key, 'malformed_measurement') from error
        v, r = trusted['vanilla'], trusted['reference']
        failed = [candidate[key]] if isinstance(candidate.get(key), RuntimeGroupFailure) else []
        if failed:
            for row in failed:
                row.validate(group=key, ids=ids)
                if row.variant != 'candidate':
                    raise MeasurementError('candidate runtime marker has the wrong variant')
            runtime_errors[key] = [row.variant for row in failed]
            verdicts[key] = {'pass': False, 'complete': False, 'expected_instances': len(ids),
                'observed_instances': 0, 'unusable_instances': list(ids),
                'comparisons': {name: False for name in ('paired_timing', 'p90', 'p95', 'p99', 'correctness')},
                'paired_speedup_over_reference': None, 'candidate_tails': None,
                'reference_tails': None, 'candidate_correct': 0, 'reference_correct': None,
                'runtime_error_variants': runtime_errors[key]}
            if bfcl_flip_gate and key.startswith('bfcl/'):
                validity_groups[key] = {'pass': False, 'regressions': None,
                    'allowed_regressions': None, 'expected_instances': len(ids),
                    'vanilla_correct': None, 'runtime_error_variants': runtime_errors[key]}
            continue
        c = _normalize(candidate.get(key, ()), ids, role='candidate', group=key)
        candidate_errors = tuple(i for i, row in c.items() if row.execution == 'candidate_error')
        candidate_latencies = {i: _timing(row) for i, row in c.items()}
        workload, rate = key.split('/')
        if bfcl_flip_gate and workload == 'bfcl':
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
        if mode == 'train' and not bfcl_flip_gate and not unusable:
            verdicts[key]['candidate_geometric_mean_latency_seconds'] = math.exp(
                math.fsum(math.log(_timing(c[i])) for i in ids) / len(ids))
        censored = {role: {ident: row.timing_bound.provenance() for ident, row in rows.items()
                          if row.execution == 'model_case_failure'}
                    for role, rows in (('vanilla', v), ('reference', r), ('candidate', c))}
        if any(censored.values()):
            verdicts[key]['censored_timing'] = censored
    if runtime_errors:
        parts = {}
        for workload in ('swe', 'bfcl'):
            errors = {key: value for key, value in runtime_errors.items() if key.startswith(workload + '/')}
            parts[workload] = ({'score': 0.0, 'runtime_error_groups': errors,
                'policy': 'affected_workload_no_credit', 'speedup': None,
                'latency_score': None, 'accuracy_multiplier': None}
                if errors else workload_continuous(workloads[workload], mode=mode))
        continuous = {'score': 0.8 * parts['swe']['score'] + 0.2 * parts['bfcl']['score'], **parts}
    else:
        continuous = continuous_score(workloads, mode=mode)
    valid = all(row['pass'] for row in validity_groups.values())
    observed_bfcl_failure = any(not row['pass'] and not row.get('runtime_error_variants')
                                for row in validity_groups.values())
    reference_pass = all(row['pass'] for row in verdicts.values())
    if not bfcl_flip_gate:
        return {'runtime_errors': runtime_errors,
            'execution': 'runtime_error' if runtime_errors else 'complete',
            'valid': True, 'pass': not runtime_errors and reference_pass,
            'score': continuous['score'], 'continuous': continuous,
            'continuous_is_before_validity': False, 'bfcl_flip_gate_enabled': False,
            'validity': {'pass': True, 'gate': None, 'groups': {}},
            'reference_verdict': {'pass': reference_pass, 'groups': verdicts,
                'expected_groups': len(expected),
                'passed_groups': sum(row['pass'] for row in verdicts.values())}}
    # Keep pre-validity arithmetic for operator audit; an original BFCL gate
    # failure invalidates the whole submission, not only its BFCL contribution.
    return {'runtime_errors': runtime_errors, 'execution': 'runtime_error' if runtime_errors else 'complete', 'valid': valid, 'pass': not runtime_errors and valid and reference_pass, 'score': continuous['score'] if not observed_bfcl_failure and (valid or runtime_errors) else 0.0, 'continuous': continuous, 'continuous_is_before_validity': True, 'validity': {'pass': valid, 'gate': None if valid else 'bfcl_correctness', 'groups': validity_groups}, 'reference_verdict': {'pass': reference_pass, 'groups': verdicts, 'expected_groups': len(expected), 'passed_groups': sum(row['pass'] for row in verdicts.values())}}


def evaluate_final(vanilla, candidate, reference, *, bfcl_flip_gate=True) -> dict:
    return _evaluate_phase(vanilla, candidate, reference, mode='eval', bfcl_flip_gate=bfcl_flip_gate)


def evaluate_train(vanilla, candidate, reference, *, bfcl_flip_gate=True) -> dict:
    """Original public 5/8 groups, same arithmetic; output is operator-only.

    The Agent feedback adapter must apply its separately reviewed whitelist;
    this does not expose reference values or a private verdict envelope to train.
    """
    return _evaluate_phase(vanilla, candidate, reference, mode='train', bfcl_flip_gate=bfcl_flip_gate)
