"""Forward one CPU-Verifier job to the trusted scheduler sidecar and return its answer.

This runs inside the GPU parent over the Verifier's SSH session with the same
uid as the candidate server, so it is not trusted: it carries bytes between
stdin/stdout and the scheduler on the bridge. Every answer the Verifier scores
is signed by the scheduler and verified on the CPU side; this process cannot
alter it, only delay or lose it, and a lost job is re-requested by its exact
sequence and request digest without being re-executed. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request

from scheduler_stream import SCHEMA as STREAM_SCHEMA, frame

SCHEDULER = 'http://scheduler:8002'
POLL_SECONDS = 2.0
RELAY_REASONS = frozenset({
    'asynchronous_submission_is_only_for_correctness',
    'correctness_cursor_belongs_to_a_different_job',
    'correctness_cursor_lookup_failed',
    'correctness_cursor_reached_owner_deadline',
    'correctness_ended_without_its_expected_cursor',
    'correctness_metadata_exceeds_existing_stderr_allocation',
    'correctness_needs_the_original_explicit_stderr_allocation',
    'correctness_summary_has_no_original_worker_selector',
    'explicit_positive_relay_bounds_required',
    'incremental_stream_exceeds_publisher_direction_allocation',
    'invalid_immutable_correctness_cursor',
    'invalid_receipt_selector',
    'job_failed',
    'job_lookup_failed',
    'job_rejected',
    'job_sequence_must_be_a_non_negative_integer',
    'receipt_lookup_failed',
    'relay_deadline_reached',
    'relay_request_is_empty_or_exceeds_its_stream_bound',
    'scheduler_did_not_become_ready_before_the_deadline',
    'scheduler_event_does_not_bind_streamed_job',
    'scheduler_event_lookup_failed',
    'scheduler_job_did_not_complete_before_the_deadline',
    'scheduler_job_sequence_belongs_to_another_request',
    'scheduler_rejected_asynchronous_correctness',
    'scheduler_rejected_streamed_job',
    'scheduler_response_exceeds_its_stream_bound',
    'scheduler_stream_reached_owner_deadline',
    'scheduler_streamed_job_failed',
    'unknown_relay_error',
})
CORRECTNESS_VIEW_SCHEMA = 'frontier.vllm-correctness-receipt-view-draft/v1'


class RelayError(RuntimeError):
    """Bounded diagnostics only; relay bytes never authorize infra attribution."""
    def __init__(self, reason, *, http_status=None, state=None, failure=None):
        self.reason = reason if isinstance(reason, str) and reason in RELAY_REASONS else 'unknown_relay_error'
        self.http_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        self.state = state if isinstance(state, str) and state in {'running', 'complete', 'failed', 'cancelled', 'pending'} else 'unknown'
        self.failure = None
        if (isinstance(failure, dict) and set(failure) == {'stage', 'sidecar_id', 'candidate_input_forwarded'}
                and failure['stage'] == 'preinput_sidecar_ready'
                and failure['candidate_input_forwarded'] is False
                and isinstance(failure['sidecar_id'], str)
                and re.fullmatch(r'[a-z][a-z0-9_-]{0,95}', failure['sidecar_id'])):
            self.failure = dict(failure)
        super().__init__(self.reason)


def error_diagnostic(error):
    diagnostic = {'schema': 'frontier.scheduler-relay-error/v1', 'reason': 'unknown_exception'}
    if isinstance(error, RelayError):
        diagnostic.update(reason=error.reason, http_status=error.http_status, state=error.state)
        if error.failure is not None:
            diagnostic['failure'] = error.failure
    elif isinstance(error, (OSError, urllib.error.URLError)):
        diagnostic['reason'] = 'transport_error'
    elif isinstance(error, (ValueError, KeyError, TypeError)):
        diagnostic['reason'] = 'protocol_error'
    # Closed strings and bounded integers only: no exception/body/payload text.
    return json.dumps(diagnostic, sort_keys=True, separators=(',', ':'))



def _read_stdin(maximum):
    raw = sys.stdin.buffer.read(maximum + 1)
    if len(raw) > maximum or not raw:
        raise RelayError('relay_request_is_empty_or_exceeds_its_stream_bound')
    return raw


def _open(request, deadline, maximum):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RelayError('relay_deadline_reached')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=min(remaining, 300)) as response:
        data = response.read(maximum + 1)
        if len(data) > maximum:
            raise RelayError('scheduler_response_exceeds_its_stream_bound')
        return response.status, data


def _request(method, path, body, deadline, maximum):
    headers = {'Accept': 'application/json'}
    if body is not None:
        headers['Content-Type'] = 'application/json'
    request = urllib.request.Request(SCHEDULER + path, data=body, headers=headers, method=method)
    try:
        return _open(request, deadline, maximum)
    except urllib.error.HTTPError as error:
        data = error.read(maximum + 1)
        return error.code, data[:maximum]


def wait_ready(deadline, maximum):
    while True:
        try:
            status, data = _request('GET', '/ready', None, deadline, maximum)
            if status == 200 and json.loads(data).get('ready') is True:
                return
        except (OSError, urllib.error.URLError, ValueError):
            pass
        if time.monotonic() + POLL_SECONDS >= deadline:
            raise RelayError('scheduler_did_not_become_ready_before_the_deadline')
        time.sleep(POLL_SECONDS)


def job(raw, deadline, maximum):
    """Submit once (idempotent on the scheduler by sequence and digest), then poll."""
    document = json.loads(raw)
    sequence = document['sequence']
    if type(sequence) is not int or sequence < 0:
        raise RelayError('job_sequence_must_be_a_non_negative_integer')
    digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
    wait_ready(deadline, maximum)
    status, data = _request('POST', '/jobs', raw, deadline, maximum)
    if status not in (200, 202):
        raise RelayError('job_rejected', http_status=status)
    while True:
        status, data = _request('GET', '/jobs/' + str(sequence), None, deadline, maximum)
        if status != 200:
            raise RelayError('job_lookup_failed', http_status=status)
        view = json.loads(data)
        if view.get('request_sha256') != digest:
            raise RelayError('scheduler_job_sequence_belongs_to_another_request')
        if view.get('state') == 'complete' or (str(view.get('state')).startswith('failed:') and isinstance(view.get('signed'), dict)):
            return data
        if view.get('state') != 'running':
            raise RelayError('job_failed', state=view.get('state'), failure=view.get('failure'))
        if time.monotonic() + POLL_SECONDS >= deadline:
            raise RelayError('scheduler_job_did_not_complete_before_the_deadline')
        time.sleep(POLL_SECONDS)


def receipt(raw, deadline, maximum):
    selector = json.loads(raw)
    variant, index = selector.get('variant'), selector.get('index')
    if variant not in {'vanilla', 'reference', 'candidate'} or type(index) is not int or index < 0:
        raise RelayError('invalid_receipt_selector')
    status, data = _request('GET', f'/receipts/{variant}/{index}', None, deadline, maximum)
    if status != 200:
        raise RelayError('receipt_lookup_failed', http_status=status)
    return data


def submit(raw, deadline, maximum):
    """Use the original job allocation to start asynchronous correctness once."""
    document = json.loads(raw)
    if document.get('kind') != 'correctness':
        raise RelayError('asynchronous_submission_is_only_for_correctness')
    wait_ready(deadline, maximum)
    status, data = _request('POST', '/jobs', raw, deadline, maximum)
    if status not in (200, 202):
        raise RelayError('scheduler_rejected_asynchronous_correctness', http_status=status)
    return data


def correctness_next(raw, deadline, maximum, metadata_maximum):
    """One existing receipt SSH allocation; metadata and body use separate caps."""
    selector = json.loads(raw)
    if (set(selector) != {'sequence', 'cursor', 'request_sha256', 'variant', 'expected_count'}
            or any(type(selector[key]) is not int or selector[key] < 0 for key in ('sequence', 'cursor'))
            or type(selector['expected_count']) is not int or selector['expected_count'] <= selector['cursor']
            or selector['variant'] not in {'vanilla', 'reference', 'candidate'}):
        raise RelayError('invalid_immutable_correctness_cursor')
    sequence, cursor = selector['sequence'], selector['cursor']
    while True:
        status, data = _request('GET', f'/jobs/{sequence}/correctness/{cursor}',
                               None, deadline, metadata_maximum)
        if status != 200:
            raise RelayError('correctness_cursor_lookup_failed', http_status=status)
        view = json.loads(data)
        if (view.get('sequence') != sequence or view.get('cursor') != cursor
                or view.get('request_sha256') != selector['request_sha256']):
            raise RelayError('correctness_cursor_belongs_to_a_different_job')
        if (view.get('event') is not None and
                (cursor + 1 < selector['expected_count'] or view.get('state') != 'running')):
            payload = view['event']['envelope']['payload']
            record = payload['record']
            body = b''
            if record.get('episode') is not None:
                worker = re.fullmatch(r'swe-([0-9]{3})-check', record['episode']['worker'])
                if worker is None:
                    raise RelayError('correctness_summary_has_no_original_worker_selector')
                body = receipt(json.dumps({'variant': selector['variant'],
                    'index': int(worker.group(1))}).encode(), deadline, maximum)
            metadata = json.dumps({'schema': CORRECTNESS_VIEW_SCHEMA, **view},
                sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            if len(metadata) > metadata_maximum:
                raise RelayError('correctness_metadata_exceeds_existing_stderr_allocation')
            return body, metadata
        if view.get('state') != 'running':
            raise RelayError('correctness_ended_without_its_expected_cursor')
        if time.monotonic() + POLL_SECONDS >= deadline:
            raise RelayError('correctness_cursor_reached_owner_deadline')
        time.sleep(POLL_SECONDS)


def job_stream(raw, deadline, maximum, *, emit):
    """One SSH stream, incremental signed evidence, no CPU arrival acknowledgments."""
    document = json.loads(raw)
    sequence = document['sequence']
    if type(sequence) is not int or sequence < 0:
        raise RelayError('job_sequence_must_be_a_non_negative_integer')
    digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
    wait_ready(deadline, maximum)
    status, _ = _request('POST', '/jobs', raw, deadline, maximum)
    if status not in (200, 202):
        raise RelayError('scheduler_rejected_streamed_job', http_status=status)
    index, emitted = 0, 0
    def send(kind, value):
        nonlocal emitted
        data = frame({'schema': STREAM_SCHEMA, 'event_kind': kind, 'event_index': index,
                      'value': value}, maximum)
        emitted += len(data)
        if emitted > maximum:
            raise RelayError('incremental_stream_exceeds_publisher_direction_allocation')
        emit(data)
    while True:
        status, data = _request('GET', f'/jobs/{sequence}/events/{index}', None, deadline, maximum)
        if status != 200:
            raise RelayError('scheduler_event_lookup_failed', http_status=status)
        view = json.loads(data)
        if (view.get('sequence') != sequence or view.get('request_sha256') != digest
                or type(view.get('event_count')) is not int or view['event_count'] < index):
            raise RelayError('scheduler_event_does_not_bind_streamed_job')
        if view.get('event') is not None:
            send('progress', view['event'])
            index += 1
            continue
        if view.get('state') == 'complete' or (str(view.get('state')).startswith('failed:') and isinstance(view.get('signed'), dict)):
            send('complete', {key: view[key] for key in
                 ('sequence', 'request_sha256', 'state', 'signed')})
            return
        if view.get('state') != 'running':
            raise RelayError('scheduler_streamed_job_failed', state=view.get('state'), failure=view.get('failure'))
        if time.monotonic() + POLL_SECONDS >= deadline:
            raise RelayError('scheduler_stream_reached_owner_deadline')
        time.sleep(POLL_SECONDS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['job', 'job-stream', 'job-submit', 'correctness-next', 'receipt'])
    parser.add_argument('--max-input-bytes', required=True, type=int)
    parser.add_argument('--max-output-bytes', required=True, type=int)
    parser.add_argument('--timeout-seconds', required=True, type=float)
    parser.add_argument('--max-metadata-bytes', type=int)
    args = parser.parse_args(argv)
    if min(args.max_input_bytes, args.max_output_bytes, args.timeout_seconds) <= 0 or not math.isfinite(args.timeout_seconds):
        raise RelayError('explicit_positive_relay_bounds_required')
    deadline = time.monotonic() + args.timeout_seconds
    raw = _read_stdin(args.max_input_bytes)
    if args.action == 'correctness-next':
        if args.max_metadata_bytes is None or args.max_metadata_bytes <= 0:
            raise RelayError('correctness_needs_the_original_explicit_stderr_allocation')
        data, metadata = correctness_next(raw, deadline, args.max_output_bytes, args.max_metadata_bytes)
        sys.stderr.buffer.write(metadata)
        sys.stderr.buffer.flush()
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        return
    if args.action == 'job-stream':
        def emit(data):
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        job_stream(raw, deadline, args.max_output_bytes, emit=emit)
        return
    function = {'job': job, 'job-submit': submit, 'receipt': receipt}[args.action]
    data = function(raw, deadline, args.max_output_bytes)
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # No scheduler payload, environment or raw transport detail is echoed.
        print(error_diagnostic(error), file=sys.stderr)
        raise SystemExit(1)
