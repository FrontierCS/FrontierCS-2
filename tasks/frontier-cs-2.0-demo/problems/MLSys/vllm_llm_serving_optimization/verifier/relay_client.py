"""CPU-side client of the trusted scheduler through the untrusted parent.

Each job is one bounded SSH operation into the GPU parent that runs
``scheduler_relay.py``; the relay submits the job to the scheduler sidecar on
the bridge and returns the signed envelope. The signature is checked against
the attestation public key the Driver reported on the runner view, and the
envelope is bound to this parent, phase, job sequence and exact request bytes.
A lost answer is re-requested; the scheduler never re-executes a bound job.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import secrets
import time

import attestation
from driver_client import RemoteExecutionError, SidecarExecutionError, SchedulerPreinputInfrastructureError
import scheduler_failure_contract
from cpu_reference_verdict import TrustedMeasurementError
from sidecar_catalog import SCHEDULER
from scheduler_stream import Decoder, FrameError, SCHEMA as STREAM_SCHEMA, decode
from containment_policy import Boundary
from terminal_contract import terminal

JOB_SCHEMA = 'frontier.vllm-scheduler-job/v1'
RELAY = ['/opt/frontier/task-runtime', '/opt/frontier/runtime-python', '/opt/frontier/judge/scheduler_relay.py']
RELAY_ATTEMPTS = 3
# Retained evidence is bounded by the platform's 64 MiB evidence archive: a
# fetched receipt keeps its digest, size and a head/tail excerpt once the
# original parser has consumed it; the signed job answer is kept once, raw.
RECEIPT_EXCERPT_BYTES = 32 * 1024


class TrustedRelayEvidenceError(RuntimeError):
    """CPU crypto, journal, or fixed-input evidence failed outside model execution."""


def _verify_attestation(*args):
    try:
        return attestation.verify(*args)
    except (ValueError, TypeError, KeyError, RecursionError, UnicodeError):
        raise
    except Exception as error:
        raise TrustedRelayEvidenceError('CPU attestation implementation failed') from error


def _excerpt(data):
    if len(data) <= 2 * RECEIPT_EXCERPT_BYTES:
        return {'content': data.decode('utf-8', errors='replace')}
    return {'head': data[:RECEIPT_EXCERPT_BYTES].decode('utf-8', errors='replace'),
            'tail': data[-RECEIPT_EXCERPT_BYTES:].decode('utf-8', errors='replace'), 'truncated': True}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def remote_terminal_classification(kind, group, remote):
    """Use the CPU-owned workload to contain every SWE episode failure."""
    context = remote.get('failure_context')
    declared = context.get('boundary') if isinstance(context, dict) else None
    declared_case = (isinstance(declared, dict) and declared.get('kind') == 'swe_case'
        and declared.get('stage') in {'setup', 'tool', 'patch', 'correctness', 'execution'})
    trace = remote.get('trace', [])
    errors = [event for event in trace if isinstance(event, dict) and 'error_origin' in event]
    swe = group.startswith('swe/')
    is_case = (kind == 'correctness' or swe or declared_case or
               any(event.get('kind') == 'testbed' for event in errors))
    episode = remote.get('episode')
    failed = episode is None or 'execution_error' in remote
    # A model error can retain a partial patch. That does not make its episode
    # complete, and reference/vanilla failures have the same per-case boundary.
    if swe:
        failed = (failed or remote.get('requires_error_attribution') is True or bool(errors)
                  or (isinstance(episode, dict) and bool(episode.get('error'))))
        if (isinstance(episode, dict) and episode.get('exit_status') == 'api_error'
                and errors and all(event.get('kind') == 'model'
                    and event.get('error_origin') == 'model_api_response' for event in errors)):
            # A serving endpoint failure is C1; generated commands/patches and
            # unknown SWE subprocess failures remain isolated C2 case failures.
            is_case = False
    if group.startswith('bfcl/') and not isinstance(episode, dict):
        failed = True
    if group.startswith('bfcl/') and isinstance(episode, dict):
        normal_limit = episode.get('exit_status') == 'force_quit' and episode.get('error') == 'max_steps'
        failed = (failed or remote.get('requires_error_attribution') is not False or bool(errors)
                  or (bool(episode.get('error')) and not normal_limit)
                  or episode.get('exit_status') not in {'ok', 'force_quit'}
                  or not isinstance(episode.get('output'), str) or 'correct' in episode
                  or episode.get('instance_id') != remote.get('instance_id')
                  or not isinstance(trace, list) or any(not isinstance(event, dict) for event in trace))
        import math
        times = [episode.get(key) for key in ('started_monotonic', 'ended_monotonic', 'latency_seconds')]
        valid_times = (all(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) for value in times)
                       and times[2] > 0 and times[1] > times[0] and times[1] - times[0] == times[2])
        failed = failed or not valid_times
        if failed:
            import math
            start, end, latency = (episode.get(key) for key in ('started_monotonic', 'ended_monotonic', 'latency_seconds'))
            measured = all(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) for value in (start, end, latency))
            # Legacy 'swe_case' is the shared controller's per-instance domain.
            # Without original measured time this remains a service/group RE.
            is_case = (measured and latency > 0 and end > start and end - start == latency
                       and 'execution_error' not in remote and episode.get('exit_status') != 'api_error'
                       and bool(errors) and all(event.get('kind') == 'model' for event in errors))
    return Boundary('swe_case', 'execution') if is_case else Boundary('candidate_service', 'response'), failed


class SchedulerRelay:
    def __init__(self, session, *, public_key, relay_limits, max_stderr_bytes, evidence,
                 stream=False, on_progress=None, controller_factory=None, cleanup_seconds=None,
                 payload_validator=None):
        if not isinstance(public_key, str) or not public_key.startswith('ssh-ed25519 '):
            raise SidecarExecutionError('relay requires the Driver-reported scheduler public key')
        self.session = session
        self.terminal_failure = None
        self.public_key = public_key
        self.limits = relay_limits
        self.max_stderr_bytes = max_stderr_bytes
        self.evidence = Path(evidence)
        self.evidence.mkdir(parents=True, exist_ok=False)
        self.sequence = 0
        self.stream, self.on_progress = stream, on_progress
        self.controller_factory = controller_factory
        self.cleanup_seconds = cleanup_seconds
        if payload_validator is not None and not callable(payload_validator):
            raise ValueError('trusted scheduler payload validator must be callable')
        self.payload_validator = payload_validator

    def _relay(self, action, raw, *, timeout, max_output_bytes, output, operation_deadline=None):
        argv = [*RELAY, action, '--max-input-bytes', str(len(raw)),
                '--max-output-bytes', str(max_output_bytes), '--timeout-seconds', str(timeout)]
        owned = {} if operation_deadline is None else {'operation_deadline': operation_deadline}
        path = self.session.run(argv, stdin=raw, output=output, max_stdout_bytes=max_output_bytes,
                                max_stderr_bytes=self.max_stderr_bytes, **owned)
        return path.read_bytes()

    def _stream_job(self, raw, *, timeout, deadline, output, kind, variant, sequence, request_sha256,
                    progress, controller=None):
        complete, records = [], {}
        def accept(data):
            # Only parsing/authentication of remote bytes is normalized here.
            # The CPU journal callback executes outside this catch.
            try:
                value = decode(data)
                if (not isinstance(value, dict) or set(value) !=
                        {'schema', 'event_kind', 'event_index', 'value'}
                        or value['schema'] != STREAM_SCHEMA or type(value['event_index']) is not int
                        or value['event_index'] < 0 or complete):
                    raise ValueError('invalid scheduler stream frame')
                if value['event_kind'] == 'complete':
                    if value['event_index'] != len(records):
                        raise ValueError('scheduler stream completion omitted progress')
                    complete.append(value['value'])
                    return
                if value['event_kind'] != 'progress' or value['event_index'] != len(records):
                    raise ValueError('scheduler event is outside ordered stream')
                envelope = _verify_attestation(value['value'], self.public_key)
                payload = attestation.bind(envelope, operation_id=self.session.runner_id,
                    phase=self.session.mode, name=SCHEDULER, kind=kind, sequence=sequence,
                    request_sha256=request_sha256)
                if (set(payload) != {'variant', 'event_index', 'event', 'record'}
                        or payload['variant'] != variant or payload['event_index'] != value['event_index']
                        or payload['event'] not in {'dispatch', 'terminal'}
                        or not isinstance(payload['record'], dict)
                        or not isinstance(payload['record'].get('instance_id'), str)):
                    raise ValueError('signed progress has invalid binding')
                if (payload['event'] == 'terminal' and
                        not isinstance(payload['record'].get('trace'), list)):
                    raise ValueError('signed terminal has no bounded trace list')
                index = payload['event_index']
                if index in progress and progress[index] != payload:
                    raise ValueError('scheduler replay changed committed event')
            except (ValueError, TypeError, KeyError, RecursionError, UnicodeError) as error:
                raise RemoteExecutionError('scheduler stream record failed validation', cause='protocol_error') from error
            records[index] = payload
            if index not in progress:
                if controller is not None:
                    slot = next((slot for slot in controller.slots
                                 if slot.instance_id == payload['record']['instance_id']), None)
                    if slot is None:
                        raise RemoteExecutionError('remote progress names an unexpected original instance', cause='protocol_error')
                    if payload['event'] == 'dispatch':
                        controller.mark_dispatched(slot)
                    else:
                        remote = payload['record']
                        try:
                            boundary, failed = remote_terminal_classification(kind, slot.group, remote)
                        except Exception:
                            # Only remote-record classification is in this catch.
                            # Durable controller observation remains outside.
                            boundary = (Boundary('swe_case', 'execution') if slot.group.startswith('swe/')
                                        else Boundary('candidate_service', 'response'))
                            failed = True
                        def verified_record(_):
                            return terminal(slot, status='runtime_error' if failed else 'complete',
                                cause='unknown_exception' if failed else 'normal_completion',
                                elapsed=time.monotonic() - controller.started,
                                cap=controller.deadline - controller.started,
                                result=None if failed else remote,
                                result_limit_bytes=None if failed else controller.maximum_terminal_bytes,
                                result_limit_depth=None if failed else 128)
                        controller.observe(slot, data, verify=verified_record, boundary=boundary,
                                           prefix_bytes=min(2048, controller.maximum_terminal_bytes))
                if self.on_progress is not None:
                    self.on_progress(payload, data)
                progress[index] = payload
        decoder = Decoder(maximum_frame_bytes=self.limits['response_bytes'],
            maximum_total_bytes=self.limits['response_bytes'], accept=accept)
        argv = [*RELAY, 'job-stream', '--max-input-bytes', str(len(raw)),
            '--max-output-bytes', str(self.limits['response_bytes']), '--timeout-seconds', str(timeout)]
        try:
            self.session.run(argv, stdin=raw, output=output,
                max_stdout_bytes=self.limits['response_bytes'], max_stderr_bytes=self.max_stderr_bytes,
                operation_deadline=deadline, on_stdout=decoder.feed)
            decoder.finish()
        except FrameError as error:
            raise RemoteExecutionError('scheduler stream framing was incomplete or invalid', cause='protocol_error') from error
        if len(complete) != 1:
            raise RemoteExecutionError('scheduler stream did not commit a final view', cause='protocol_error')
        return canonical_bytes(complete[0])

    def job(self, kind, variant, fields, *, timeout, name):
        """Run one scheduler job and return its verified, bound payload."""
        if self.terminal_failure is not None:
            raise self.terminal_failure
        sequence = self.sequence
        document = {'schema': JOB_SCHEMA, 'kind': kind, 'sequence': sequence,
                    'nonce': secrets.token_hex(16), 'variant': variant, **fields}
        started = time.monotonic()
        owner_seconds = min(timeout, self.session.client.remaining())
        deadline = started + owner_seconds
        if self.stream:
            if self.cleanup_seconds is None or self.cleanup_seconds <= 0:
                raise SidecarExecutionError('streaming scheduler requires existing owner cleanup reserve')
            document['containment'] = {'verification_id': self.session.client.verification_id,
                'owner_seconds': owner_seconds,
                'cleanup_seconds': self.cleanup_seconds,
                'frame_bytes': self.limits['response_bytes'], 'stderr_bytes': self.max_stderr_bytes}
        raw = canonical_bytes(document)
        if len(raw) > self.limits['job_bytes']:
            raise SidecarExecutionError('scheduler job exceeds its publisher relay bound')
        request_sha256 = 'sha256:' + hashlib.sha256(raw).hexdigest()
        output = self.evidence / name
        output.mkdir(parents=True, exist_ok=False)
        (output / 'request.json').write_bytes(raw)
        controller = (self.controller_factory(kind, variant, fields, request_sha256, output,
                                             started=started, deadline=deadline)
                      if self.controller_factory is not None else None)
        # Group protocol completion is independent of already committed cases.
        control = (self.controller_factory('group_control', variant, fields, request_sha256,
                    output / 'group-control', started=started, deadline=deadline)
                   if kind == 'batch' and self.controller_factory is not None else None)
        def remote_failed():
            for owner in (controller, control):
                if owner is not None:
                    owner.fail_unresolved(boundary=Boundary('candidate_service', 'response'))
        failures = []
        data = None
        progress = {}
        for attempt in range(RELAY_ATTEMPTS):
            try:
                if self.stream:
                    data = self._stream_job(raw, timeout=timeout, deadline=deadline, output=output / f'relay-{attempt}',
                        kind=kind, variant=variant, sequence=sequence,
                        request_sha256=request_sha256, progress=progress, controller=controller)
                else:
                    data = self._relay('job', raw, timeout=timeout, max_output_bytes=self.limits['response_bytes'],
                                       output=output / f'relay-{attempt}')
                break
            except RemoteExecutionError as error:
                failures.append(type(error).__name__)
                if (self.session.operations >= self.session.limits['ssh_sessions']
                        or attempt + 1 == RELAY_ATTEMPTS):
                    (output / 'error.json').write_text(json.dumps({'origin': 'scheduler_relay',
                        'attempts': failures, 'score_ready': False}) + '\n')
                    remote_failed()
                    raise
                self.session.client.remaining()
        try:
            view, envelope, payload = self._validated_view(data, kind=kind, variant=variant,
                sequence=sequence, request_sha256=request_sha256, progress=progress)
        except (TrustedMeasurementError, SchedulerPreinputInfrastructureError) as error:
            self.terminal_failure = error
            raise
        except TrustedRelayEvidenceError:
            raise
        except Exception as error:
            remote_failed()
            raise RemoteExecutionError('scheduler final view failed validation: ' + str(error),
                                       cause='protocol_error') from error
        try:
            if self.payload_validator is not None:
                self.payload_validator(kind, variant, payload)
        except TrustedRelayEvidenceError:
            raise
        except Exception as error:
            remote_failed()
            raise RemoteExecutionError('remote batch payload failed validation', cause='protocol_error') from error
        if control is not None:
            for slot in control.slots:
                def verified(_):
                    return terminal(slot, status='complete', cause='normal_completion',
                        elapsed=time.monotonic() - started, cap=deadline - started,
                        result={'lifecycle': 'group_control', 'state': 'complete'},
                        result_limit_bytes=control.maximum_terminal_bytes, result_limit_depth=8)
                control.observe(slot, b'', verify=verified,
                    boundary=Boundary('candidate_service', 'response'),
                    prefix_bytes=min(2048, control.maximum_terminal_bytes))
        # The raw relay stdout already retains the complete signed answer.
        (output / 'envelope-summary.json').write_text(json.dumps({
            'kind': kind, 'sequence': sequence, 'variant': variant, 'request_sha256': request_sha256,
            'identity': envelope['identity'], 'key_id': view['signed']['key_id'],
            'signature': view['signed']['signature'], 'payload_keys': sorted(payload),
            'response_bytes': len(data)}, indent=2) + '\n')
        self.sequence += 1
        return payload

    def _validated_view(self, data, *, kind, variant, sequence, request_sha256, progress):
        try:
            view = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SidecarExecutionError('relay returned no complete JSON job view') from None
        if (not isinstance(view, dict) or
                (view.get('state') != 'complete' and not str(view.get('state')).startswith('failed:'))
                or view.get('sequence') != sequence or view.get('request_sha256') != request_sha256):
            raise SidecarExecutionError('scheduler job view does not bind this job')
        try:
            envelope = _verify_attestation(view.get('signed'), self.public_key)
            payload = attestation.bind(envelope, operation_id=self.session.runner_id, phase=self.session.mode,
                                       name=SCHEDULER, kind=kind, sequence=sequence,
                                       request_sha256=request_sha256)
        except (ValueError, TypeError, KeyError, RecursionError, UnicodeError) as error:
            raise SidecarExecutionError('scheduler evidence failed attestation: ' + str(error)) from None
        if payload.get('variant') != variant:
            raise SidecarExecutionError('scheduler payload belongs to another variant')
        if str(view.get('state')).startswith('failed:') and payload.get('schema') != scheduler_failure_contract.SCHEMA:
            raise SidecarExecutionError('failed scheduler view lacks its signed terminal failure')
        if payload.get('schema') == scheduler_failure_contract.SCHEMA:
            failure = scheduler_failure_contract.validate(payload)
            environment = getattr(self.controller_factory, 'environment', {})
            if (not environment or envelope['identity']['run_id'] != environment.get('FRONTIER_RUN_ID')
                    or envelope['identity']['attempt_id'] != environment.get('FRONTIER_ATTEMPT_ID')):
                raise SidecarExecutionError('scheduler failure differs from CPU run/attempt')
            if (kind == 'batch' and failure['stage'] == 'preinput_sidecar_ready'
                    and failure['cause'] in scheduler_failure_contract.READINESS_CAUSES):
                raise SchedulerPreinputInfrastructureError({
                    'schema': 'frontier.vllm-authenticated-scheduler-readiness/v1',
                    'cause': failure['cause'], 'sidecar_id': failure['sidecar_id'],
                    'operation_id': self.session.runner_id, 'phase': self.session.mode,
                    'run_id': envelope['identity']['run_id'], 'attempt_id': envelope['identity']['attempt_id'],
                    'request_sha256': request_sha256, 'sequence': sequence,
                    'variant': variant, 'cleanup_required': False, 'candidate_input_forwarded': False,
                    'proof_kind': 'cpu_verified_scheduler_terminal_signature'})
            # Missing trusted controller output is not candidate Runtime Error.
            # Candidate service/case faults are contained before this boundary.
            raise TrustedMeasurementError(variant, None, 'invalid_or_incomplete_measurement')
        if self.stream and 'batch' in payload:
            expected = payload.get('episode_sha256')
            order = payload.get('episode_instance_ids')
            rows = {event['record']['instance_id']: event['record'] for event in progress.values()
                    if event['event'] == 'terminal'}
            if (not isinstance(expected, dict) or not isinstance(order, list)
                    or len(order) != len(rows) or set(order) != set(rows)
                    or sum(event['event'] == 'terminal' for event in progress.values()) != len(rows)
                    or set(rows) != set(expected)
                    or any('sha256:' + hashlib.sha256(canonical_bytes(row)).hexdigest() != expected[ident]
                           for ident, row in rows.items())):
                raise SidecarExecutionError('signed batch differs from retained terminal progress')
            payload = dict(payload)
            payload['batch'] = dict(payload['batch'], episodes=[rows[ident] for ident in order])
            del payload['episode_sha256']
            del payload['episode_instance_ids']
        return view, envelope, payload

    def receipt(self, variant, index, *, expected_sha256, expected_bytes, name):
        """Fetch one raw correctness receipt bound by the signed correctness summary."""
        if (type(expected_bytes) is not int or not 0 < expected_bytes <= self.limits['receipt_bytes']
                or not isinstance(expected_sha256, str) or len(expected_sha256) != 71):
            raise SidecarExecutionError('signed receipt summary is out of the relay bound')
        raw = json.dumps({'variant': variant, 'index': index}).encode()
        output = self.evidence / name
        data = self._relay('receipt', raw, timeout=self.limits['receipt_timeout_seconds'],
                           max_output_bytes=self.limits['receipt_bytes'], output=output)
        digest = 'sha256:' + hashlib.sha256(data).hexdigest()
        # The full body is bound by the signed summary and consumed by the
        # original parser in memory; evidence keeps a bounded excerpt of it.
        (output / 'stdout').unlink()
        (output / 'receipt-excerpt.json').write_text(json.dumps({
            'variant': variant, 'index': index, 'bytes': len(data), 'sha256': digest,
            'expected_sha256': expected_sha256, 'expected_bytes': expected_bytes, **_excerpt(data)},
            indent=2, allow_nan=False) + '\n')
        if len(data) != expected_bytes or digest != expected_sha256:
            raise SidecarExecutionError('fetched correctness receipt differs from its signed summary')
        try:
            receipt = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SidecarExecutionError('correctness receipt is not JSON') from None
        if not isinstance(receipt, dict):
            raise SidecarExecutionError('correctness receipt is not an object')
        return receipt

    def correctness(self, variant, requests, *, ids, timeout, name, parse, prefailed_ids=frozenset()):
        from correctness_stream import collect
        return collect(self, variant, requests, ids=ids, timeout=timeout, name=name, parse=parse,
                       prefailed_ids=prefailed_ids)
