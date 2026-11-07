"""Original receipt SSH allocations capture completed siblings before aggregate loss."""
import hashlib
import json
from pathlib import Path
import secrets
import time

import attestation
from containment_policy import Boundary
from driver_client import RemoteExecutionError, SidecarExecutionError
from scheduler_relay import CORRECTNESS_VIEW_SCHEMA
from scheduler_stream import decode
from cpu_swe_accuracy import TrustedCorrectnessError
from cpu_reference_verdict import TrustedMeasurementError
from sidecar_catalog import SCHEDULER, VARIANTS, check_name, instance_index
from terminal_contract import terminal


class _CorrectnessBindingError(ValueError):
    """Remote metadata failed an explicit immutable binding check."""


def collect(relay, variant, requests, *, ids, timeout, name, parse, prefailed_ids=frozenset()):
    from relay_client import JOB_SCHEMA, RELAY, RELAY_ATTEMPTS, _excerpt, canonical_bytes
    if not relay.stream or relay.controller_factory is None:
        raise ValueError('incremental correctness requires surviving CPU ownership')
    if (not isinstance(prefailed_ids, frozenset) or not prefailed_ids.issubset(ids)
            or variant not in VARIANTS or
            set(requests) != set(ids) - prefailed_ids or len(set(ids)) != len(ids)):
        raise ValueError('correctness capture must freeze the original complete instance set')
    sequence = relay.sequence
    started = time.monotonic()
    owner_seconds = min(timeout, relay.session.client.remaining())
    deadline = started + owner_seconds
    document = {'schema': JOB_SCHEMA, 'kind': 'correctness', 'sequence': sequence,
        'nonce': secrets.token_hex(16), 'variant': variant, 'requests': requests,
        'containment': {'verification_id': relay.session.client.verification_id,
            'owner_seconds': owner_seconds, 'cleanup_seconds': relay.cleanup_seconds,
            'frame_bytes': relay.limits['response_bytes'], 'stderr_bytes': relay.max_stderr_bytes}}
    raw = canonical_bytes(document)
    if len(raw) > relay.limits['job_bytes']:
        raise ValueError('trusted original correctness requests exceed the existing job allocation')
    digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
    output = relay.evidence / name
    output.mkdir(parents=True, exist_ok=False)
    (output / 'request.json').write_bytes(raw)
    controller = relay.controller_factory('correctness', variant,
        {'requests': requests, 'prefailed_instance_ids': prefailed_ids}, digest,
                                         output, started=started, deadline=deadline)
    by_id = {slot.instance_id: slot for slot in controller.slots}
    if set(by_id) != set(ids):
        raise ValueError('CPU controller allocation differs from frozen correctness cases')
    receipts, summaries, parsed_cases, failures = {}, {}, {}, {}
    def captured():
        failed = set()
        for slot in controller.slots:
            record = controller.ledger.read(slot)
            if record is None:
                raise ValueError('CPU correctness projection has an unresolved original slot')
            if record['status'] != 'complete':
                context = record.get('failure_context', {})
                if (context.get('boundary', {}).get('kind') != 'swe_case' or
                        context.get('decision', {}).get('disposition') != 'case_fail'):
                    raise ValueError('CPU correctness projection has a foreign failure scope')
                failed.add(slot.instance_id)
        if set(receipts) != set(ids) - failed:
            raise ValueError('CPU correctness ledger and parsed original receipts disagree')
        relay.sequence += 1
        return {'receipts': receipts, 'summaries': summaries, 'parsed': parsed_cases,
                'model_case_failures': frozenset(failed)}
    # Legacy allowed three submissions + one fetch per original receipt. Any
    # transport replay consumes one of exactly those same allocations.
    operations, maximum_operations = 0, len(ids) + RELAY_ATTEMPTS
    received_cursor = False
    last_transport_error = None
    try:
        while True:
            operations += 1
            try:
                relay._relay('job-submit', raw, timeout=timeout,
                    max_output_bytes=relay.limits['response_bytes'],
                    output=output / f'submit-{operations}', operation_deadline=deadline)
                break
            except RemoteExecutionError:
                if operations >= RELAY_ATTEMPTS:
                    raise
        for cursor in range(len(ids)):
            selector = canonical_bytes({'sequence': sequence, 'request_sha256': digest,
                'variant': variant, 'cursor': cursor, 'expected_count': len(ids)})
            path = None
            while True:
                operations += 1
                receipt_output = output / f'cursor-{cursor:03d}-operation-{operations:03d}'
                # Before all workers finish, the old job wait remains available;
                # after it, only the original per-receipt transport reserve does.
                fetch_deadline = min(relay.session.client.deadline,
                    max(deadline, time.monotonic()) + relay.limits['receipt_timeout_seconds'])
                wait_seconds = fetch_deadline - time.monotonic()
                argv = [*RELAY, 'correctness-next', '--max-input-bytes', str(len(selector)),
                    '--max-output-bytes', str(relay.limits['receipt_bytes']),
                    '--max-metadata-bytes', str(relay.max_stderr_bytes),
                    '--timeout-seconds', str(wait_seconds)]
                try:
                    path = relay.session.run(argv, stdin=selector, output=receipt_output,
                        max_stdout_bytes=relay.limits['receipt_bytes'],
                        max_stderr_bytes=relay.max_stderr_bytes, operation_deadline=fetch_deadline)
                    body, metadata = path.read_bytes(), (receipt_output / 'stderr').read_bytes()
                    received_cursor = True
                    break
                except RemoteExecutionError as error:
                    if error.cause in {'protocol_error', 'output_limit'}:
                        received_cursor = True  # A per-case unusable reply, not an outage.
                    else:
                        last_transport_error = error
                    if operations + len(ids) - cursor - 1 >= maximum_operations:
                        break
            if path is None:
                continue
            # Parsing/attestation consumes only remote bytes. CPU parser filesystem,
            # ledger, signing and persistence defects remain outside this catch.
            try:
                view = decode(metadata)
            except (ValueError, KeyError, TypeError, RecursionError, UnicodeError) as error:
                # No trustworthy identity: leave its owned slot unresolved and
                # continue consuming siblings. Never accept foreign bytes as pass.
                continue
            try:
                if (not isinstance(view, dict) or set(view) !=
                        {'schema', 'sequence', 'request_sha256', 'state', 'cursor', 'event'}
                        or view['schema'] != CORRECTNESS_VIEW_SCHEMA
                        or view['sequence'] != sequence or view['request_sha256'] != digest
                        or view['cursor'] != cursor):
                    raise _CorrectnessBindingError('correctness receipt view has a foreign immutable cursor')
                envelope = attestation.verify(view['event'], relay.public_key)
                payload = attestation.bind(envelope, operation_id=relay.session.runner_id,
                    phase=relay.session.mode, name=SCHEDULER, kind='correctness',
                    sequence=sequence, request_sha256=digest)
                record = payload.get('record')
                if not isinstance(record, dict) or not isinstance(record.get('instance_id'), str):
                    raise _CorrectnessBindingError('correctness terminal has a malformed instance record')
                ident, summary = record['instance_id'], record.get('episode')
                if (set(payload) != {'variant', 'event_index', 'event', 'record'}
                        or payload['variant'] != variant or payload['event'] != 'terminal'
                        or type(payload['event_index']) is not int or payload['event_index'] < 0
                        or record.get('correctness_cursor') != cursor or ident not in by_id
                        or ident in summaries or ident in failures or record.get('trace') != []):
                    raise _CorrectnessBindingError('correctness terminal changed the original case or cursor')
                if summary is not None and (not isinstance(summary, dict)
                        or summary.get('request_sha256') != requests[ident]['request_sha256']
                        or summary.get('run_index') != VARIANTS.index(variant)
                        or summary.get('worker') != check_name(instance_index(ident))
                        or (VARIANTS.index(variant) > 0) != isinstance(summary.get('reset'), dict)
                        or summary.get('receipt_bytes') != len(body)
                        or summary.get('receipt_sha256') != 'sha256:' + hashlib.sha256(body).hexdigest()):
                    raise _CorrectnessBindingError('correctness receipt differs from its signed request-bound summary')
                if summary is None and (body or 'execution_error' not in record):
                    raise _CorrectnessBindingError('failed correctness cursor cannot carry a successful body')
            except (attestation.AttestationError, _CorrectnessBindingError, KeyError, TypeError, ValueError) as error:
                continue
            slot = by_id[ident]
            if ident not in prefailed_ids:
                controller.mark_dispatched(slot)
            parsed, receipt = None, None
            if summary is not None:
                try:
                    receipt = decode(body)
                except (ValueError, KeyError, TypeError, RecursionError, UnicodeError):
                    failures[ident] = 'original_receipt_decode_failed'
                else:
                    try:
                        parsed = parse(requests[ident], receipt)
                    except Exception:
                        failures[ident] = 'original_receipt_parse_failed'
                # Only model receipt parsing is contained here. Authentication,
                # controller signing and evidence writes remain outside it.
            else:
                failures[ident] = 'authenticated_correctness_boundary_failed'
            if parsed is None:
                failures.setdefault(ident, 'original_receipt_parse_failed')
            elif not isinstance(parsed, dict) or type(parsed.get('correct')) is not bool:
                parsed = None
                failures[ident] = 'original_receipt_invalid_outcome'
            else:
                parsed_cases[ident] = parsed
                receipts[ident], summaries[ident] = receipt, summary
            def observed(_):
                success = parsed is not None
                return terminal(slot, status='complete' if success else 'runtime_error',
                    cause='normal_completion' if success else 'protocol_error',
                    elapsed=time.monotonic() - started, cap=owner_seconds,
                    result={'instance_id': ident, 'parsed': parsed, 'receipt_summary': summary}
                        if success else None,
                    result_limit_bytes=controller.maximum_terminal_bytes if success else None,
                    result_limit_depth=128 if success else None)
            controller.observe(slot, metadata, verify=observed,
                boundary=Boundary('swe_case', 'correctness'),
                prefix_bytes=min(2048, controller.maximum_terminal_bytes))
            # Preserve the existing excerpt/digest allocation after the original
            # CPU parser has independently consumed the complete private body.
            (receipt_output / 'receipt-excerpt.json').write_text(json.dumps({
                'instance_id': ident, 'cursor': cursor, 'bytes': len(body),
                'sha256': 'sha256:' + hashlib.sha256(body).hexdigest(), **_excerpt(body)},
                indent=2, allow_nan=False) + '\n')
            path.unlink()
        if not received_cursor and last_transport_error is not None:
            raise last_transport_error
        controller.fail_unresolved(boundary=Boundary('swe_case', 'correctness'))
        return captured()
    except RemoteExecutionError as error:
        # This is an actual exhausted scheduler transport operation, rather
        # than a per-case negative result or malformed receipt.
        group = relay.controller_factory.groups[variant]
        raise TrustedMeasurementError(variant, group.workload + '/' + group.rate,
                                      'aggregate_correctness_unavailable') from error
