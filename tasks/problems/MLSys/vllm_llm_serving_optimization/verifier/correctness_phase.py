"""Join an owned, completed SWE batch to the declared correctness workers.

The caller owns the parent lifetime and its timed-batch transport. This stage
runs only after every timed batch of the parent, through the trusted scheduler:
the CPU side builds the canonical requests, the scheduler dispatches them to
the per-instance check workers (eight at a time, as the original harness) and
signs a summary with each receipt's digest, and the CPU side fetches every raw
receipt, checks it against that summary and parses it with the original harness.
It neither provisions a GPU nor exposes its private receipts as Agent feedback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from batch_measurements import is_candidate_model_error, project_complete_batch, validate_batch_plan
from cpu_batch_measurements import (project_complete_batch as trusted_project_complete_batch,
                                    validate_trusted_timed_batch)
from driver_client import SidecarExecutionError
from cpu_reference_verdict import TrustedMeasurementError
from cpu_swe_accuracy import (TrustedCorrectnessError, build_correctness_request,
                                  correctness_envelope, parse_test_receipt)
from sidecar_catalog import VARIANTS, check_name, instance_index


SWE_CORRECTNESS_WORKERS = 8  # Preserved original EvalSettings.workers default.


def correctness_requests(batch, group, *, variant, calibration, request_bytes,
                         model_case_failures=frozenset(), construction_failures=None):
    """Canonical per-instance requests, bound to the timed patches of the batch."""
    ids, rows = validate_batch_plan(batch, group, calibration=calibration)
    if (group.workload != 'swe' or group.mode not in {'train', 'eval'} or variant not in VARIANTS
            or tuple(bed.instance_id for bed in group.testbeds) != ids):
        raise SidecarExecutionError('correctness must bind the complete owned SWE group')
    requests = {}
    if (not isinstance(model_case_failures, frozenset) or not model_case_failures.issubset(ids)
            or variant not in VARIANTS):
        raise ValueError('CPU-owned timed case failure set required')
    if construction_failures is not None and type(construction_failures) is not set:
        raise ValueError('CPU request construction failure sink must be a set')
    for ident, row in zip(ids, rows):
        if ident in model_case_failures:
            continue
        try:
            requests[ident] = _case_request(row, ident, group, variant, request_bytes)
        except Exception:
            if construction_failures is None:
                raise
            construction_failures.add(ident)
    return ids, requests


def _case_request(row, ident, group, variant, request_bytes):
    episode = row.get('episode')
    patch = episode.get('patch') if isinstance(episode, dict) else None
    trace = row.get('trace')
    candidate_error = variant == 'candidate' and is_candidate_model_error(row, ident, 'swe')
    if (not candidate_error and (not isinstance(patch, str) or episode.get('instance_id') != ident
            or episode.get('exit_status') not in {'submitted', 'limit_with_patch', 'incomplete'}
            or episode.get('error')
            or row.get('requires_error_attribution') is not False
            or 'execution_error' in row or not isinstance(trace, list)
            or any(not isinstance(event, dict) or 'error_origin' in event for event in trace)
            or row.get('patch_sha256') != 'sha256:' + hashlib.sha256(patch.encode()).hexdigest())):
        raise SidecarExecutionError('timed SWE patch or execution needs attribution before correctness')
    identity = {**group.identity(), 'instance_id': ident,
                'child_id': check_name(instance_index(ident)), 'phase': group.mode}
    request = build_correctness_request(ident, patch, identity=identity, variant=variant)
    raw = correctness_envelope(request, identity, variant)
    if len(raw) > request_bytes:
        raise SidecarExecutionError('original correctness request exceeds its declared bound')
    return request


def _run_swe_correctness(batch, group, *, relay, variant, evidence, calibration=False,
                        request_bytes, timeout, model_case_failures=frozenset(), model_case_bounds=None):
    validate_trusted_timed_batch(batch, group, variant=variant, calibration=calibration,
        model_case_failures=model_case_failures, model_case_bounds=model_case_bounds)
    if group.identity() != {key: relay.session.runner_id for key in ('operation_id', 'group_id', 'placement_id')}:
        raise SidecarExecutionError('correctness group belongs to another owned parent')
    construction_failures = set()
    ids, requests = correctness_requests(batch, group, variant=variant, calibration=calibration,
        request_bytes=request_bytes, model_case_failures=model_case_failures,
        construction_failures=construction_failures)
    model_case_failures = model_case_failures | frozenset(construction_failures)
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=False)
    receipts = {}
    try:
        if getattr(relay, 'stream', False):
            captured = relay.correctness(variant, requests, ids=ids, timeout=timeout,
                name=f'{variant}-correctness', parse=parse_test_receipt,
                prefailed_ids=model_case_failures)
            receipts, summaries = captured['receipts'], captured['summaries']
            for index, ident in enumerate(ids):
                if ident not in captured['parsed']:
                    continue
                (evidence / f'parsed-{index:03d}.json').write_text(
                    json.dumps(captured['parsed'][ident], indent=2) + '\n')
            project = trusted_project_complete_batch if variant in {'vanilla', 'reference'} else project_complete_batch
            observations = project(batch, group, swe_receipts=receipts,
                calibration=calibration, variant=variant,
                model_case_failures=captured.get('model_case_failures', frozenset()),
                model_case_bounds=model_case_bounds)
        else:
            if model_case_failures:
                raise ValueError('C2 projection requires live CPU correctness ownership')
            observations, summaries = _legacy_collect(batch, group, relay=relay, variant=variant,
                ids=ids, requests=requests, timeout=timeout, evidence=evidence, receipts=receipts,
                calibration=calibration)
    except Exception as error:
        (evidence / 'error.json').write_text(json.dumps({'origin': 'correctness_or_scheduler_unclassified',
            'error_type': type(error).__name__, 'score_ready': False,
            'completed_instance_ids': list(getattr(error, 'completed_instance_ids', receipts))}, indent=2) + '\n')
        raise
    (evidence / 'summary.json').write_text(json.dumps({'schema': 'frontier.vllm-reset-correctness/v3',
        'identity': group.identity(), 'mode': group.mode, 'group': group.workload + '/' + group.rate,
        'variant': variant, 'calibration_only': calibration, 'instance_ids': list(ids),
        'resets': {ident: summaries[ident].get('reset') for ident in ids if ident in summaries},
        'correct': sum(row.correct for row in observations), 'score_ready': False,
        'model_failed_instance_ids': [row.instance_id for row in observations
                                     if row.source_status == 'cpu_observed_model_case_failure'],
        'scope': 'reset-verified correctness joined to an externally owned timed batch; not full phase acceptance'},
        indent=2) + '\n')
    return observations


def _legacy_collect(batch, group, *, relay, variant, ids, requests, timeout, evidence,
                    receipts, calibration):
    """Existing whole-job join retained for offline adapter compatibility."""
    payload = relay.job('correctness', variant, {'requests': requests}, timeout=timeout,
                        name=f'{variant}-correctness')
    summaries = payload.get('receipts')
    if (payload.get('instance_ids') != list(ids) or payload.get('workers') != SWE_CORRECTNESS_WORKERS
            or not isinstance(summaries, dict) or set(summaries) != set(ids)):
        raise SidecarExecutionError('signed correctness summary does not cover the owned group')
    run_index = VARIANTS.index(variant)
    for index, ident in enumerate(ids):
        summary = summaries[ident]
        if (not isinstance(summary, dict)
                or summary.get('request_sha256') != requests[ident]['request_sha256']
                or summary.get('run_index') != run_index
                or summary.get('worker') != check_name(instance_index(ident))
                or (run_index > 0) != isinstance(summary.get('reset'), dict)):
            raise SidecarExecutionError('correctness summary entry is not bound to its request, worker or order')
        receipt = relay.receipt(variant, instance_index(ident), expected_sha256=summary.get('receipt_sha256'),
                                expected_bytes=summary.get('receipt_bytes'),
                                name=f'{variant}-receipt-{index:03d}')
        parsed = parse_test_receipt(requests[ident], receipt)
        (evidence / f'parsed-{index:03d}.json').write_text(json.dumps(parsed, indent=2) + '\n')
        receipts[ident] = receipt
    project = trusted_project_complete_batch if variant in {'vanilla', 'reference'} else project_complete_batch
    observations = project(batch, group, swe_receipts=receipts,
                                          calibration=calibration, variant=variant)
    return observations, summaries


def run_swe_correctness(batch, group, *, relay, variant, evidence, calibration=False,
                        request_bytes, timeout, model_case_failures=frozenset(), model_case_bounds=None):
    """Keep missing trusted correctness evidence outside candidate dispositions."""
    try:
        return _run_swe_correctness(batch, group, relay=relay, variant=variant,
            evidence=evidence, calibration=calibration, request_bytes=request_bytes,
            timeout=timeout, model_case_failures=model_case_failures,
            model_case_bounds=model_case_bounds)
    except TrustedMeasurementError:
        raise
    except Exception as error:
        if variant in {'vanilla', 'reference'}:
            raise TrustedMeasurementError(variant, group.workload + '/' + group.rate,
                                          'invalid_or_incomplete_correctness') from error
        raise
