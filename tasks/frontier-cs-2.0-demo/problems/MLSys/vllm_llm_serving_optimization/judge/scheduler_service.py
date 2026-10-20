"""Trusted scheduler sidecar: timed batches, testbed shutdown, correctness, signed evidence.

The Driver starts this container beside the GPU parent from its declared
profile and gives it the operation's attestation private key. Jobs arrive
from the CPU Verifier through the untrusted parent (``scheduler_relay.py``) in
one fixed sequence per parent and are executed exactly once; a repeated job is
recognised by its sequence and canonical digest and never re-run. Every answer
is signed, so the parent can delay or lose evidence but not alter or replay it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import attestation
from batch_coordinator import BackendError, TestbedClient, TestbedTransportError, run_group
import scheduler_failure_contract
from batch_run import prepared_group
from instance_manifest import expected_groups
from serving_eval import sidecar_env
from serving_eval.swe_patch_disposition import (correctness_envelope, validate_disposition,
                                              OVERSIZE_REQUEST_SCHEMA)
from serving_eval.tool_auth import ToolAuthSigner
from sidecar_catalog import CHECK_PORT, SCHEDULER, TESTBED_PORT, VARIANTS, check_name, instance_index, testbed_name

JOB_SCHEMA = 'frontier.vllm-scheduler-job/v1'
MODEL_ENDPOINT = 'http://main:8000/v1'
SWE_CORRECTNESS_WORKERS = 8  # Preserved original EvalSettings.workers default.
CORRECTNESS_TRANSPORT_GRACE_SECONDS = 300
NONCE = re.compile(r'[0-9a-f]{16,64}')
JOB_PATH = re.compile(r'/jobs/(0|[1-9][0-9]*)')
RECEIPT_PATH = re.compile(r'/receipts/(vanilla|reference|candidate)/(0|[1-9][0-9]*)')
EVENT_PATH = re.compile(r'/jobs/(0|[1-9][0-9]*)/events/(0|[1-9][0-9]*)')
CORRECTNESS_NEXT_PATH = re.compile(r'/jobs/(0|[1-9][0-9]*)/correctness/(0|[1-9][0-9]*)')


class JobError(ValueError):
    pass


class SidecarReadinessError(BackendError):
    """Scheduler-owned barrier failed before any input for this batch."""
    def __init__(self, sidecar_id, cause='unclassified'):
        super().__init__('declared testbed readiness barrier failed')
        self.sidecar_id = sidecar_id
        self.cause = cause


def canonical_sha256(raw):
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


class Job:
    def __init__(self, sequence, document, raw):
        self.sequence = sequence
        self.document = document
        self.request_sha256 = canonical_sha256(raw)
        self.state = 'running'
        self.signed = None
        self.failure = None
        self.events = []
        self.event_bytes = 0
        self.stream_signed = None
        self.created = time.monotonic()
        self.correctness_events = []

    def view(self):
        value = {'sequence': self.sequence, 'request_sha256': self.request_sha256, 'state': self.state}
        if self.failure is not None:
            value['failure'] = self.failure
        if self.signed is not None:
            value['signed'] = self.signed
        return value


class CheckClient:
    """Signed requests to one correctness worker; identity echoes are checked."""

    def __init__(self, group, instance_id, *, signer):
        self.name = check_name(instance_index(instance_id))
        self.endpoint = f'http://{self.name}:{CHECK_PORT}'
        self.identity = {**group.identity(), 'instance_id': instance_id,
                         'child_id': self.name, 'phase': group.mode}
        self.signer = signer
        self.transport = None

    def correctness(self, variant, request):
        body = correctness_envelope(request, self.identity, variant)
        headers = self.signer.headers(self.name, 'POST', '/correctness', body)
        headers['Content-Type'] = 'application/json'
        timeout = request['timeout_seconds'] + CORRECTNESS_TRANSPORT_GRACE_SECONDS
        try:
            if self.transport is not None:
                from bounded_http import request_signed
                transport = dict(self.transport,
                    deadline=min(self.transport['deadline'], time.monotonic() + timeout))
                result = request_signed(self.endpoint + '/correctness', 'POST', headers,
                                        body.decode(), **transport)
            else:
                with build_opener(ProxyHandler({})).open(Request(self.endpoint + '/correctness', data=body,
                                                              headers=headers, method='POST'),
                                                      timeout=timeout) as response:
                    result = json.loads(response.read())
        except HTTPError as error:
            raise BackendError(f'correctness worker refused the request: HTTP {error.code}') from None
        except Exception as error:
            raise BackendError('correctness worker transport failed: ' + type(error).__name__) from error
        if (not isinstance(result, dict) or result.get('identity') != self.identity
                or result.get('variant') != variant or not isinstance(result.get('receipt'), dict)):
            raise BackendError('correctness worker response identity or shape mismatch')
        return result


class SchedulerService:
    def __init__(self, identity, *, signer, tool_signer, max_response_bytes, run=None, testbed_client=None,
                 check_client=None, journal_directory=None, variants=VARIANTS):
        from sidecar_catalog import validate_variants
        self.variants = validate_variants(variants)
        self.identity = attestation.validate_identity(identity)
        if self.identity['name'] != SCHEDULER or self.identity['phase'] not in {'train', 'eval'}:
            raise ValueError('scheduler service requires the declared scheduler identity of a Verifier phase')
        self.signer = signer
        self.tool_signer = tool_signer
        self.max_response_bytes = max_response_bytes
        self.run_group = run_group if run is None else run
        self.testbed_client = TestbedClient if testbed_client is None else testbed_client
        self.check_client = CheckClient if check_client is None else check_client
        self.jobs = []
        self.workload = None
        self.batches = {}
        self.batch_jobs = {}
        self.receipts = {}
        self.lock = threading.Lock()
        self.journal = None if journal_directory is None else Path(journal_directory)
        if self.journal is not None:
            self.journal.mkdir(parents=True, exist_ok=False)

    # -- job admission -------------------------------------------------------
    def expected_steps(self):
        steps = [('batch', variant) for variant in self.variants]
        if self.workload in (None, 'swe'):
            steps.extend(('correctness', variant) for variant in self.variants)
        return steps

    def submit(self, raw):
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return 400, {'error': 'job is not JSON'}
        if (not isinstance(document, dict) or document.get('schema') != JOB_SCHEMA
                or type(document.get('sequence')) is not int or document['sequence'] < 0
                or document.get('kind') not in attestation.KINDS
                or document.get('variant') not in self.variants
                or not isinstance(document.get('nonce'), str) or not NONCE.fullmatch(document['nonce'])):
            return 400, {'error': 'invalid closed scheduler job'}
        digest = canonical_sha256(raw)
        with self.lock:
            sequence = document['sequence']
            if sequence < len(self.jobs):
                job = self.jobs[sequence]
                if job.request_sha256 != digest:
                    return 409, {'error': 'sequence already bound to another request'}
                return 202, job.view()
            if sequence != len(self.jobs):
                return 409, {'error': 'job sequence is not the next expected one'}
            steps = self.expected_steps()
            if sequence >= len(steps) or (document['kind'], document['variant']) != steps[sequence]:
                return 409, {'error': 'job differs from the fixed phase order'}
            if self.jobs and self.jobs[-1].state != 'complete':
                return 409, {'error': 'previous job has not completed'}
            try:
                prepared = self.validate(document)
            except (JobError, BackendError, KeyError, TypeError, ValueError) as error:
                return 400, {'error': 'job rejected', 'error_type': type(error).__name__}
            job = Job(sequence, document, raw)
            self.jobs.append(job)
        thread = threading.Thread(target=self.execute, args=(job, prepared), daemon=True)
        thread.start()
        return 202, job.view()

    def validate(self, document):
        kind, variant = document['kind'], document['variant']
        common = {'schema', 'kind', 'sequence', 'nonce', 'variant'}
        if 'containment' in document:
            common.add('containment')
            limits = document['containment']
            if (self.journal is None or not isinstance(limits, dict) or set(limits) !=
                    {'verification_id', 'owner_seconds', 'cleanup_seconds', 'frame_bytes', 'stderr_bytes'}
                    or not isinstance(limits['verification_id'], str)
                    or not 0 < len(limits['verification_id']) <= 256
                    or any(isinstance(limits[key], bool) or not isinstance(limits[key], (int, float))
                           or not math.isfinite(limits[key]) or limits[key] <= 0
                           for key in ('owner_seconds', 'cleanup_seconds'))
                    or any(type(limits[key]) is not int or not 0 < limits[key] <= self.max_response_bytes
                           for key in ('frame_bytes', 'stderr_bytes'))):
                raise JobError('invalid bounded original-owner containment request')
        if kind == 'batch':
            if set(document) != common | {'prepared', 'all_prepared', 'jps', 'calibration'}:
                raise JobError('invalid batch job fields')
            if type(document['calibration']) is not bool or not isinstance(document['all_prepared'], list):
                raise JobError('invalid batch job values')
            if len(document['all_prepared']) != len(self.variants):
                raise JobError('batch job must carry the complete variant barrier')
            group = prepared_group(document['prepared'])
            groups = [prepared_group(value) for value in document['all_prepared']]
            if groups[self.variants.index(variant)] != group:
                raise JobError('selected group is not the declared variant of the barrier')
            operation = self.identity['operation_id']
            for item in groups:
                if (item.identity() != {'operation_id': operation, 'group_id': operation, 'placement_id': operation}
                        or item.mode != self.identity['phase'] or item.model_endpoint != MODEL_ENDPOINT
                        or (item.workload, item.rate) != (group.workload, group.rate)):
                    raise JobError('barrier group belongs to another parent, phase or workload')
            if self.workload not in (None, group.workload):
                raise JobError('parent workload changed between jobs')
            for index, item in enumerate(groups):
                if item.workload == 'bfcl':
                    if item.testbeds:
                        raise JobError('BFCL groups carry no testbeds')
                    continue
                for bed in item.testbeds:
                    name = testbed_name(instance_index(bed.instance_id), self.variants[index])
                    if (bed.child_id != name or bed.endpoint != f'http://{name}:{TESTBED_PORT}'
                            or bed.phase != self.identity['phase']):
                        raise JobError('testbed descriptor differs from the declared sidecar')
            return group, groups
        if set(document) != common | {'requests'}:
            raise JobError('invalid correctness job fields')
        if self.workload != 'swe' or variant not in self.batches:
            raise JobError('correctness requires this parent\'s completed SWE batch of the variant')
        group, batch = self.batches[variant]
        ids = expected_groups(group.mode)['swe/public']
        requests = document['requests']
        # The retained selector owns every case. Missing or invalid per-case
        # requests are incorrect; they cannot invalidate healthy sibling work.
        requests = requests if isinstance(requests, dict) else {}
        accepted = {}
        episodes = {row['instance_id']: row.get('episode') for row in batch['episodes']}
        for ident in ids:
            if ident not in requests:
                continue
            request, episode = requests[ident], episodes.get(ident)
            try:
                self.validate_correctness_case(group, variant, ident, request, episode)
            except Exception:
                continue
            accepted[ident] = request
        return group, accepted

    def validate_correctness_case(self, group, variant, ident, request, episode):
        if isinstance(request, dict) and request.get('schema') == OVERSIZE_REQUEST_SCHEMA:
            identity = {**group.identity(), 'instance_id': ident,
                        'child_id': check_name(instance_index(ident)), 'phase': group.mode}
            if (not isinstance(episode, dict) or not isinstance(episode.get('patch'), str)
                    or request.get('instance_id') != ident
                    or request.get('correctness_identity') != identity
                    or request.get('variant') != variant):
                raise JobError('oversize disposition differs from this parent timed patch')
            validate_disposition(request, patch=episode['patch'])
            return
        if (not isinstance(request, dict) or request.get('instance_id') != ident
                or not isinstance(episode, dict)
                or request.get('candidate_patch') != episode.get('patch')
                or type(request.get('timeout_seconds')) is not int or request['timeout_seconds'] <= 0
                or not isinstance(request.get('request_sha256'), str)):
            raise JobError('correctness request differs from the timed episode of this parent')

    # -- execution -----------------------------------------------------------
    def execute(self, job, prepared):
        try:
            if job.document['kind'] == 'batch':
                payload = self.batch(job, *prepared)
            else:
                payload = self.correctness(job, *prepared)
            signed = self.signer.sign(kind=job.document['kind'], sequence=job.sequence,
                                      request_sha256=job.request_sha256, payload=payload)
            stream_payload = dict(payload)
            if 'batch' in stream_payload:
                header = dict(stream_payload['batch'])
                episodes = header.pop('episodes')
                stream_payload['batch'] = header
                stream_payload['episode_sha256'] = {
                    row['instance_id']: canonical_sha256(attestation.canonical_bytes(row))
                    for row in episodes}
                stream_payload['episode_instance_ids'] = [row['instance_id'] for row in episodes]
            stream_signed = self.signer.sign(kind=job.document['kind'], sequence=job.sequence,
                request_sha256=job.request_sha256, payload=stream_payload)
            encoded = json.dumps({**job.view(), 'signed': signed}, allow_nan=False, separators=(',', ':'))
            if len(encoded.encode()) > self.max_response_bytes:
                raise JobError('signed job result exceeds the publisher response bound')
            with self.lock:
                job.signed = signed
                job.stream_signed = stream_signed
                job.state = 'complete'
        except BaseException as error:
            with self.lock:
                job.state = 'failed:' + type(error).__name__
                if (type(error) is SidecarReadinessError
                        and error.cause in scheduler_failure_contract.READINESS_CAUSES):
                    job.failure = {'stage': 'preinput_sidecar_ready',
                        'sidecar_id': error.sidecar_id, 'candidate_input_forwarded': False}
                    terminal_failure = {**job.failure, 'cause': error.cause}
                else:
                    terminal_failure = {'stage': 'scheduler_control', 'cause': 'unclassified',
                        'sidecar_id': None, 'candidate_input_forwarded': None}
                failure_payload = {'schema': scheduler_failure_contract.SCHEMA,
                    'variant': job.document['variant'], 'failure': terminal_failure}
                scheduler_failure_contract.validate(failure_payload)
                job.signed = self.signer.sign(kind=job.document['kind'], sequence=job.sequence,
                    request_sha256=job.request_sha256, payload=failure_payload)
                job.stream_signed = job.signed
            if not isinstance(error, Exception):
                raise

    def batch(self, job, group, groups):
        variant = job.document['variant']
        with self.lock:
            self.workload = group.workload
        if group.workload == 'swe':
            # All three complete testbed sets must exist before the first timing;
            # later batches only need their own (earlier sets were shut down).
            barrier = groups if job.sequence == 0 else [group]
            for item in barrier:
                for bed in item.testbeds:
                    try:
                        readiness = ({'deadline': job.created + job.document['containment']['owner_seconds']}
                                     if 'containment' in job.document else {})
                        self.testbed_client(item, bed, [], signer=self.tool_signer).ready(**readiness)
                    except Exception as error:
                        raise SidecarReadinessError(bed.child_id,
                            error.code if type(error) is TestbedTransportError else 'unclassified') from error
        arguments = dict(jps=job.document['jps'], calibration=job.document['calibration'],
                         tool_signer=self.tool_signer)
        if self.journal is not None:
            arguments['on_progress'] = lambda event, record: self.progress(job, event, record)
        if 'containment' in job.document:
            limits = job.document['containment']
            arguments['execution'] = {**limits, 'run_id': self.identity['run_id'],
                'attempt_id': self.identity['attempt_id'],
                'request_sha256': job.request_sha256, 'variant': variant,
                'deadline': job.created + limits['owner_seconds'],
                'journal_path': self.journal / f'job-{job.sequence:02d}-episodes.sqlite'}
        batch = self.run_group(group, **arguments)
        shutdown = {}
        for bed in group.testbeds:
            try:
                self.testbed_client(group, bed, [], signer=self.tool_signer).shutdown()
                shutdown[bed.child_id] = 'confirmed'
            except Exception as error:
                shutdown[bed.child_id] = 'uncertain:' + type(error).__name__
        with self.lock:
            self.batches[variant] = (group, batch)
            self.batch_jobs[variant] = job
        return {'variant': variant, 'batch': batch, 'testbed_shutdown': shutdown}

    def correctness(self, job, group, requests):
        variant = job.document['variant']
        ids = expected_groups(group.mode)['swe/public']
        summaries, failures = {}, {}
        lock = threading.Lock()

        def run_one(ident):
            record = None
            try:
                if ident not in requests:
                    # An absent or rejected request has no measured correctness.
                    # Do not invent an empty patch or reuse its timed testbed.
                    with lock:
                        failures[ident] = 'TimedCaseFailed'
                    record = {'instance_id': ident, 'episode': None, 'trace': [],
                        'execution_error': {'origin': 'timed_case_boundary', 'type': 'TimedCaseFailed'}}
                    return
                client = self.check_client(group, ident, signer=self.tool_signer)
                if 'containment' in job.document:
                    from terminal_contract import Slot
                    limits = job.document['containment']
                    slot = Slot(self.identity['run_id'], self.identity['attempt_id'],
                        limits['verification_id'], group.operation_id, group.mode,
                        group.workload + '/' + group.rate + '/correctness', variant,
                        ident, group.model_source_digest, job.request_sha256)
                    client.transport = {'slot': slot,
                        'deadline': job.created + limits['owner_seconds'],
                        'cleanup_seconds': limits['cleanup_seconds'],
                        'maximum_frame_bytes': limits['frame_bytes'],
                        'maximum_stderr_bytes': limits['stderr_bytes']}
                if self.journal is not None:
                    self.progress(job, 'dispatch', {'instance_id': ident})
                result = client.correctness(variant, requests[ident])
                raw = json.dumps(result['receipt'], sort_keys=True, separators=(',', ':'),
                                 ensure_ascii=False, allow_nan=False).encode()
                with lock:
                    self.receipts[(variant, instance_index(ident))] = raw
                    summaries[ident] = {'request_sha256': requests[ident]['request_sha256'],
                                        'receipt_sha256': canonical_sha256(raw), 'receipt_bytes': len(raw),
                                        'reset': result.get('reset'), 'run_index': result.get('run_index'),
                                        'worker': client.name}
                    record = {'instance_id': ident, 'episode': summaries[ident], 'trace': []}
            except Exception as error:
                with lock:
                    failures[ident] = type(error).__name__
                record = {'instance_id': ident, 'episode': None, 'trace': [],
                          'execution_error': {'origin': 'correctness_boundary',
                                              'type': type(error).__name__}}
            finally:
                if self.journal is not None:
                    self.progress(job, 'terminal', record or {
                        'instance_id': ident, 'episode': None, 'trace': [],
                        'execution_error': {'origin': 'correctness_boundary', 'type': 'BaseException'}})

        with ThreadPoolExecutor(max_workers=SWE_CORRECTNESS_WORKERS) as pool:
            list(pool.map(run_one, ids))
        return {'variant': variant, 'instance_ids': list(ids), 'workers': SWE_CORRECTNESS_WORKERS,
                'receipts': summaries}

    # -- views ---------------------------------------------------------------
    def ready(self):
        return {'ready': True, 'identity': self.identity, 'jobs': len(self.jobs)}

    def job_view(self, sequence):
        with self.lock:
            if sequence >= len(self.jobs):
                return None
            return self.jobs[sequence].view()

    def progress(self, job, event, record):
        if self.journal is None or event not in {'dispatch', 'terminal'}:
            raise JobError('durable scheduler progress requires its configured journal')
        with self.lock:
            index = len(job.events)
            if job.document['kind'] == 'correctness' and event == 'terminal':
                record = dict(record, correctness_cursor=len(job.correctness_events))
            payload = {'variant': job.document['variant'], 'event_index': index,
                       'event': event, 'record': record}
            signed = self.signer.sign(kind=job.document['kind'], sequence=job.sequence,
                request_sha256=job.request_sha256, payload=payload)
            raw = attestation.canonical_bytes(signed)
            if job.event_bytes + len(raw) > self.max_response_bytes:
                raise JobError('incremental scheduler evidence exceeds publisher allocation')
            path = self.journal / f'job-{job.sequence:02d}-event-{index:05d}.json'
            with path.open('xb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            descriptor = os.open(self.journal, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            job.events.append(signed)
            job.event_bytes += len(raw)
            if job.document['kind'] == 'correctness' and event == 'terminal':
                job.correctness_events.append(signed)

    def correctness_view(self, sequence, cursor):
        with self.lock:
            if sequence >= len(self.jobs):
                return None
            job = self.jobs[sequence]
            if job.document['kind'] != 'correctness':
                return None
            return {'sequence': sequence, 'request_sha256': job.request_sha256,
                    'state': job.state, 'cursor': cursor,
                    'event': job.correctness_events[cursor]
                        if cursor < len(job.correctness_events) else None}

    def event_view(self, sequence, index):
        with self.lock:
            if sequence >= len(self.jobs):
                return None
            job = self.jobs[sequence]
            return {'sequence': sequence, 'request_sha256': job.request_sha256,
                    'state': job.state, 'event_count': len(job.events),
                    'failure': job.failure,
                    'event': job.events[index] if index < len(job.events) else None,
                    'signed': job.stream_signed}

    def receipt(self, variant, index):
        with self.lock:
            return self.receipts.get((variant, index))


def make_server(service, *, host, port, max_request_bytes):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, format, *args):
            pass

        def respond(self, status, body, content_type='application/json'):
            encoded = body if isinstance(body, bytes) else json.dumps(body, allow_nan=False,
                                                                       separators=(',', ':')).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        def do_GET(self):
            if self.path == '/ready':
                self.respond(200, service.ready())
                return
            match = CORRECTNESS_NEXT_PATH.fullmatch(self.path)
            if match:
                view = service.correctness_view(int(match.group(1)), int(match.group(2)))
                self.respond(404 if view is None else 200,
                             {'error': 'unknown correctness cursor'} if view is None else view)
                return
            match = EVENT_PATH.fullmatch(self.path)
            if match:
                view = service.event_view(int(match.group(1)), int(match.group(2)))
                self.respond(404 if view is None else 200,
                             {'error': 'unknown job'} if view is None else view)
                return
            match = JOB_PATH.fullmatch(self.path)
            if match:
                view = service.job_view(int(match.group(1)))
                if view is None:
                    self.respond(404, {'error': 'unknown job'})
                else:
                    self.respond(200, view)
                return
            match = RECEIPT_PATH.fullmatch(self.path)
            if match:
                raw = service.receipt(match.group(1), int(match.group(2)))
                if raw is None:
                    self.respond(404, {'error': 'unknown receipt'})
                else:
                    self.respond(200, raw)
                return
            self.respond(404, {'error': 'unknown scheduler route'})

        def do_POST(self):
            if self.path != '/jobs':
                self.respond(404, {'error': 'unknown scheduler route'})
                return
            lengths = self.headers.get_all('Content-Length', [])
            if self.headers.get('Transfer-Encoding') or len(lengths) != 1:
                self.respond(400, {'error': 'one Content-Length is required'})
                return
            try:
                length = int(lengths[0])
            except ValueError:
                self.respond(400, {'error': 'invalid Content-Length'})
                return
            if length <= 0 or length > max_request_bytes:
                self.respond(413, {'error': 'job exceeds publisher limit or is empty'})
                return
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError('incomplete body')
            except (ValueError, TimeoutError):
                self.respond(400, {'error': 'incomplete job body'})
                return
            status, body = service.submit(raw)
            self.respond(status, body)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--max-request-bytes', required=True, type=int)
    parser.add_argument('--max-response-bytes', required=True, type=int)
    parser.add_argument('--journal-directory', default='/tmp/frontier-scheduler-journal')
    parser.add_argument('--variants', choices=('legacy', 'candidate'), default='legacy')
    args = parser.parse_args()
    if min(args.max_request_bytes, args.max_response_bytes) <= 0:
        raise ValueError('explicit publisher stream bounds required')
    identity = sidecar_env.install('signer')
    tool_signer = ToolAuthSigner.from_environment()
    signer = attestation.Signer(os.environ['FRONTIER_TOOL_AUTH_PRIVATE_KEY_OPENSSH'],
                                os.environ['FRONTIER_TOOL_AUTH_PUBLIC_KEY_OPENSSH'], identity)
    sidecar_env.scrub()
    service = SchedulerService(identity, signer=signer, tool_signer=tool_signer,
                               max_response_bytes=args.max_response_bytes,
                               journal_directory=args.journal_directory,
                               variants=('candidate',) if args.variants == 'candidate' else VARIANTS)
    with make_server(service, host=args.host, port=args.port, max_request_bytes=args.max_request_bytes) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
