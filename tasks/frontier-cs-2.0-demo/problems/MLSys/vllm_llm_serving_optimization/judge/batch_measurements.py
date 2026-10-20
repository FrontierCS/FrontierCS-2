"""Project complete, trusted local batches into the approved score input type.

The caller must obtain records from its owned evaluator. This module checks
bindings, not transport authenticity. Unclassified or missing execution remains
unscored here until the runtime's candidate/platform attribution is integrated;
the existing scorer's approved candidate-failure penalties are unchanged.
"""
from __future__ import annotations

import hashlib
import math

from batch_coordinator import FINAL_HIDDEN_JPS, PreparedGroup, _digest
from instance_manifest import expected_groups
from reference_verdict import InstanceObservation, MeasurementError
from serving_eval import bfcl
from serving_eval.accuracy import build_correctness_request, parse_test_receipt, load_swe_cases
from sidecar_catalog import check_name, instance_index


COMPLETE_STATUSES = {'swe': {'submitted', 'limit_with_patch', 'incomplete'},
                     'bfcl': {'ok', 'force_quit'}}


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def is_candidate_model_error(row, ident, workload):
    """Recognize only a candidate server HTTP error with complete local evidence."""
    episode = row.get('episode')
    trace = row.get('trace')
    errors = ([event for event in trace if isinstance(event, dict) and 'error_origin' in event]
              if isinstance(trace, list) else [])
    if (row.get('requires_error_attribution') is not True or 'execution_error' in row
            or not isinstance(episode, dict) or episode.get('instance_id') != ident
            or len(errors) != 1 or errors[0].get('kind') != 'model'
            or errors[0].get('error_origin') != 'model_api_response'
            or not isinstance(errors[0].get('error_type'), str) or not errors[0]['error_type']
            or any(not isinstance(event, dict) for event in trace)
            or episode.get('error') != errors[0]['error_type']):
        return False
    start, end, latency = (episode.get(name) for name in
                           ('started_monotonic', 'ended_monotonic', 'latency_seconds'))
    if (not all(_number(value) for value in (start, end, latency)) or latency <= 0
            or end <= start or end - start != latency):
        return False
    if workload == 'swe':
        patch = episode.get('patch')
        return (episode.get('exit_status') in {'api_error', 'limit_with_patch'}
                and isinstance(patch, str)
                and row.get('patch_sha256') == 'sha256:' + hashlib.sha256(patch.encode()).hexdigest())
    return (workload == 'bfcl' and episode.get('exit_status') == 'api_error'
            and isinstance(episode.get('output'), str) and 'correct' not in episode)


def expected_input_digest(group):
    if group.workload == 'swe':
        instances = [{'instance_id': row['instance_id'], 'problem_statement': row['problem_statement']}
                     for row in load_swe_cases(group.mode)]
    else:
        instances = bfcl.load_memory_instances(expected_groups(group.mode)['bfcl/' + group.rate])
    return _digest(instances)


def validate_batch_plan(batch, group: PreparedGroup, *, calibration=False):
    key = group.workload + '/' + group.rate
    ids = expected_groups(group.mode).get(key)
    if (not isinstance(batch, dict) or ids is None
            or batch.get('schema') != 'frontier.vllm-local-batch/v1'
            or batch.get('identity') != group.identity()
            or batch.get('mode') != group.mode or batch.get('group') != key
            or batch.get('model_source_digest') != group.model_source_digest
            or batch.get('expected_instances') != len(ids)):
        raise MeasurementError('batch does not match its prepared evaluator group')
    expected_jps = ({'swe': 0.5, 'bfcl': 1.0} if group.rate == 'public'
                    else FINAL_HIDDEN_JPS)[group.workload]
    if (type(calibration) is not bool or batch.get('calibration_only') is not calibration
            or not _number(batch.get('jps')) or batch['jps'] != expected_jps
            or type(batch.get('arrival_seed')) is not int
            or batch['arrival_seed'] != {'swe': 20260604, 'bfcl': 20260605}[group.workload]
            or batch.get('input_digest') != expected_input_digest(group)):
        raise MeasurementError('batch rate, seed, calibration or original input differs from its plan')
    rows = batch.get('episodes')
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows) or tuple(row.get('instance_id') for row in rows) != ids:
        raise MeasurementError('complete projection requires every original instance in order')
    return ids, rows


def project_complete_batch(batch, group: PreparedGroup, *, swe_receipts=None,
                           calibration=False, variant=None, candidate_failures=frozenset(),
                           model_case_failures=frozenset(), model_case_bounds=None):
    ids, rows = validate_batch_plan(batch, group, calibration=calibration)
    # This set is supplied only by the surviving CPU controller, never a row's
    # remote attribution field. Baseline/reference failure arithmetic is absent.
    if (not isinstance(candidate_failures, frozenset) or not candidate_failures.issubset(ids)
            or (candidate_failures and variant != 'candidate')):
        raise MeasurementError('explicit CPU-owned candidate failure set required')
    if (not isinstance(model_case_failures, frozenset) or not model_case_failures.issubset(ids)
            or (model_case_failures and (group.workload != 'swe'
                or variant not in {'vanilla', 'reference', 'candidate'}))
            or model_case_failures & candidate_failures):
        raise MeasurementError('explicit CPU-owned SWE model case failure set required')
    model_case_bounds = {} if model_case_bounds is None else model_case_bounds
    if not isinstance(model_case_bounds, dict) or not set(model_case_bounds).issubset(model_case_failures):
        raise MeasurementError('CPU timing bounds must belong to the failed original model cases')
    if group.workload == 'swe':
        if (not isinstance(swe_receipts, dict) or
                set(swe_receipts) != set(ids) - candidate_failures - model_case_failures):
            raise MeasurementError('SWE requires separate correctness receipts for all original instances')
    elif swe_receipts is not None:
        raise MeasurementError('BFCL does not accept SWE correctness receipts')
    bfcl_answers = bfcl.load_memory_answers(ids) if group.workload == 'bfcl' else None
    output = []
    for ident, row in zip(ids, rows):
        if ident in candidate_failures:
            output.append(InstanceObservation(ident, None, False,
                execution='candidate_error', source_status='cpu_observed_case_failure'))
            continue
        episode = row.get('episode')
        if ident in model_case_failures:
            if row.get('supervised_terminal') is not None:
                from model_case_timing import owned_case_failure_seconds
                bound = model_case_bounds.get(ident)
                if bound is None:
                    raise MeasurementError('supervised timing lacks CPU-owned slot binding')
                elapsed = owned_case_failure_seconds(row, bound=bound, group=group, variant=variant)
                output.append(InstanceObservation(ident, elapsed, False,
                    source_status='cpu_observed_model_case_failure'))
                continue
            start, end, latency = (episode.get(name) if isinstance(episode, dict) else None
                for name in ('started_monotonic', 'ended_monotonic', 'latency_seconds'))
            if (ident not in model_case_bounds
                    and all(_number(value) for value in (start, end, latency)) and latency > 0
                    and end > start and end - start == latency):
                output.append(InstanceObservation(ident, latency, False,
                    source_status='cpu_observed_model_case_failure'))
            else:
                from model_case_timing import CensoredTiming
                bound = model_case_bounds.get(ident)
                if (not isinstance(bound, CensoredTiming)
                        or bound.slot.operation_id != group.operation_id
                        or bound.slot.phase != group.mode
                        or bound.slot.source_digest != group.model_source_digest):
                    raise MeasurementError('model case failure lacks its original CPU timing bound')
                bound.validate(instance_id=ident, variant=variant,
                               group=group.workload + '/' + group.rate)
                output.append(InstanceObservation(ident, None, False,
                    execution='model_case_failure', source_status='cpu_observed_model_case_failure',
                    timing_bound=bound))
            continue
        trace = row.get('trace')
        candidate_error = variant == 'candidate' and is_candidate_model_error(
            row, ident, group.workload)
        normal_step_limit = (isinstance(episode, dict) and group.workload == 'bfcl'
                             and episode.get('exit_status') == 'force_quit'
                             and episode.get('error') == 'max_steps')
        if (not candidate_error and (row.get('requires_error_attribution') is not False
                or 'execution_error' in row or not isinstance(trace, list)
                or any(not isinstance(event, dict) or 'error_origin' in event for event in trace)
                or not isinstance(episode, dict) or episode.get('instance_id') != ident
                or episode.get('exit_status') not in COMPLETE_STATUSES[group.workload]
                or (episode.get('error') and not normal_step_limit))):
            raise MeasurementError('missing or unclassified execution needs runtime attribution before scoring')
        start, end, latency = (episode.get(name) for name in ('started_monotonic', 'ended_monotonic', 'latency_seconds'))
        if (not all(_number(value) for value in (start, end, latency)) or latency <= 0
                or end <= start or end - start != latency):
            raise MeasurementError('episode lacks consistent positive original elapsed time')
        if group.workload == 'swe':
            patch = episode.get('patch')
            if not isinstance(patch, str) or row.get('patch_sha256') != 'sha256:' + hashlib.sha256(patch.encode()).hexdigest():
                raise MeasurementError('SWE patch differs from the timed episode artifact')
            identity = {**group.identity(), 'instance_id': ident,
                        'child_id': check_name(instance_index(ident)), 'phase': group.mode}
            request = build_correctness_request(ident, patch, identity=identity, variant=variant)
            parsed_correct = parse_test_receipt(request, swe_receipts[ident])['correct']
            correct = False if candidate_error else parsed_correct
        else:
            if 'correct' in episode or not isinstance(episode.get('output'), str):
                raise MeasurementError('BFCL GPU episode must contain answer-free output only')
            correct = (False if candidate_error else
                       bfcl.memory_correct(episode['output'], bfcl_answers[ident]))
        output.append(InstanceObservation(
            ident, latency, correct,
            execution='candidate_error' if candidate_error else 'complete',
            source_status=episode['exit_status']))
    return output
