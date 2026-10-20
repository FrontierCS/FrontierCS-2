"""Verifier-scoped Driver client for one owned GPU parent.

Provider credentials and lifecycle authority stay in Driver. This consumes an
already funded parent; it selects no resource budget and exposes no sidecar
operation: declared sidecars are created and ended by the Driver with the parent.
"""
from __future__ import annotations

from contextlib import contextmanager
import http.client
import hashlib
import json
from pathlib import Path
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse


class SidecarExecutionError(RuntimeError):
    """Unscored execution/transport evidence, without candidate attribution."""

    def __init__(self, message, *, error_code=None, http_status=None, driver_error_code=None):
        super().__init__(message)
        self.error_code = error_code
        self.http_status = http_status
        self.driver_error_code = driver_error_code


class RemoteExecutionError(SidecarExecutionError):
    """CPU observed a bounded remote operation fail after local setup succeeded.

    This conveys no infrastructure exemption; the enclosing C1/C2 owner decides
    the affected scope. Local filesystem, command and allocation errors use the
    original trusted error channel instead.
    """

    def __init__(self, message, *, cause):
        if cause not in {'deadline', 'output_limit', 'child_exit', 'protocol_error'}:
            raise ValueError('unknown bounded remote operation failure')
        super().__init__(message)
        self.cause = cause


class AuthenticatedProvisioningError(SidecarExecutionError):
    """Compensated pre-input capacity failure from the owned Driver response."""

    def __init__(self, provenance):
        super().__init__('Driver confirmed an owned pre-input capacity failure',
                         error_code='driver_platform_capacity')
        self.provenance = provenance


class DriverQuotaError(SidecarExecutionError):
    """Authenticated local resource ledger feedback; never an infra waiver."""
    def __init__(self, quota):
        fields = {'gpu_instances': {'limit', 'current', 'requested'},
                  'gpu_seconds': {'limit', 'used', 'reserved', 'remaining',
                                  'requested_count', 'requested_seconds'}}
        if (not isinstance(quota, dict) or quota.get('resource') not in fields
                or quota.get('principal') not in {'agent', 'verifier'}
                or set(quota) != fields[quota['resource']] | {'resource', 'principal'}
                or any(type(quota[key]) is not int or quota[key] < 0
                       or quota[key] > (2 ** (32 if quota['resource'] == 'gpu_instances'
                                                or key == 'requested_count' else 64) - 1)
                       for key in fields[quota['resource']])):
            raise ValueError('invalid typed Driver quota ledger')
        super().__init__('Declared GPU resource allocation exhausted', error_code='quota_exhausted',
                         http_status=429, driver_error_code='quota_exhausted')
        self.quota = dict(quota)


class DriverWorkIntervalError(SidecarExecutionError):
    """Accepted lifetime leaves no work after unchanged cleanup/capture reserves."""
    def __init__(self, *, phase, granted_seconds):
        if phase not in {'train', 'eval'} or type(granted_seconds) is not int or granted_seconds <= 0:
            raise ValueError('invalid accepted work interval feedback')
        super().__init__('Accepted GPU grant has no remaining work interval',
                         error_code='granted_work_interval_exhausted')
        self.quota = {'resource': 'work_interval', 'phase': phase,
                      'granted_seconds': granted_seconds,
                      'cause': 'accepted_grant_after_original_cleanup_capture_reserves'}


class SchedulerPreinputInfrastructureError(SidecarExecutionError):
    """CPU-verified exact signed scheduler readiness failure, never stderr."""
    def __init__(self, signal):
        super().__init__('Authenticated scheduler pre-input readiness failed')
        self.signal = dict(signal)


class DriverCreateInfrastructureError(SidecarExecutionError):
    """CPU-observed owned CREATE failure, not a compensated provisioning proof.

    This channel is constructed only inside the authenticated Driver CREATE
    adapter before returning a parent. It never accepts remote child output.
    Without a confirmed parent identity cleanup remains Driver-owned/uncertain.
    """
    def __init__(self, *, cause, verification_id, request_key, request_body,
                 observation_seconds, owner_identity, http_status=None, driver_error_code=None):
        if cause not in {'driver_create_capacity_timeout', 'driver_create_transport_timeout',
                         'driver_create_cleanup_uncertain'}:
            raise ValueError('unknown Driver CREATE infrastructure signal')
        super().__init__('Owned Driver CREATE could not complete before candidate input',
                         error_code=cause, http_status=http_status,
                         driver_error_code=driver_error_code)
        self.signal = {'schema': 'frontier.vllm-driver-create-observation/v1',
            'cause': cause, 'verification_id': verification_id,
            'owner_identity': dict(owner_identity),
            'origin': 'cpu_owned_authenticated_driver_create',
            'idempotency_sha256': 'sha256:' + hashlib.sha256(request_key.encode()).hexdigest(),
            'request_sha256': 'sha256:' + hashlib.sha256(request_body).hexdigest(),
            'observation_seconds': observation_seconds, 'http_status': http_status,
            'driver_error_code': driver_error_code, 'parent_response_accepted': False,
            'candidate_input_forwarded': False, 'cleanup_required': True,
            'proof_kind': 'cpu_owned_control_observation_not_compensated_proof'}


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate Driver response field')
        value[key] = item
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


@contextmanager
def _open_before_deadline(request, deadline):
    # urllib's timeout is per socket read. A slow peer can otherwise extend the
    # whole operation by dripping headers/body. Interrupt only this connection.
    sockets = []
    lock = threading.Lock()
    expired = threading.Event()

    def stop():
        expired.set()
        with lock:
            for connection in sockets:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()

    class BoundConnection:
        def connect(self):
            self.timeout = max(0.001, deadline - time.monotonic())
            super().connect()
            with lock:
                sockets.append(self.sock)
                if expired.is_set():
                    self.sock.close()
                    raise TimeoutError('connection completed after observation deadline')

    class PlainConnection(BoundConnection, http.client.HTTPConnection):
        pass

    class TLSConnection(BoundConnection, http.client.HTTPSConnection):
        pass

    class PlainHandler(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(PlainConnection, req)

    class TLSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(TLSConnection, req, context=self._context,
                                check_hostname=self._check_hostname)

    timer = threading.Timer(max(0, deadline - time.monotonic()), stop)
    timer.daemon = True
    timer.start()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect(),
                                            PlainHandler(), TLSHandler())
        with opener.open(request, timeout=max(0.001, deadline - time.monotonic())) as response:
            yield response, sockets[-1]
    finally:
        timer.cancel()
        stop()


_COMPONENT = re.compile(r'^[A-Za-z0-9_-]+$')
_CONTROL_BYTES = 1024 * 1024  # Existing Ready verifier Driver-response bound.
# Observe the Driver's bounded 3600-second provider create plus response/cleanup
# grace. The earlier absolute owner deadline still wins; no grant is extended.
_CREATE_OBSERVATION_SECONDS = 3720
_CREATE_BACKOFF_SECONDS = 15
_TERMINAL_CREATE_CODES = frozenset({
    'invalid_dispatch', 'config_digest_mismatch', 'invalid_state_transition',
    'payload_purpose_mismatch', 'payload_digest_mismatch', 'payload_too_large',
    'payload_not_found', 'idempotency_conflict', 'quota_exhausted',
    'gpu_type_not_allowed', 'gpu_runner_not_found', 'verification_not_found',
    'verifier_invalid_result', 'management_result_rejected', 'traffic_audit_failed',
    'driver_cancelled',
})


def _component(value):
    if not isinstance(value, str) or not _COMPONENT.fullmatch(value):
        raise SidecarExecutionError('invalid scoped Driver path identity')
    return value


class DriverClient:
    """Bounded, deadline-owned requests to the Verifier's GPU runner routes."""

    def __init__(self, driver_url, verification_id, token, *, deadline):
        parsed = urlparse(driver_url)
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ('', '/')):
            raise SidecarExecutionError('Driver URL must be the existing closed HTTP(S) origin')
        if not token or len(token.encode()) > 4096 or any(c.isspace() for c in token):
            raise SidecarExecutionError('invalid verification-scoped Driver token')
        self.verification_id = _component(verification_id)
        self.driver_url = driver_url.rstrip('/')
        self.url = self.driver_url + '/internal/v1/verifications/' + verification_id + '/gpu-runners'
        self._token = token
        self.deadline = deadline
        self._diagnostic_path = None

    def enable_control_diagnostics(self, path):
        self._diagnostic_path = Path(path)
        self._diagnostics_started = time.monotonic()
        self._diagnostics = {'events': [], 'omitted_events': 0}
        self._diagnostics_lock = threading.Lock()

    def _control_event(self, method, outcome, *, status=None, code=None):
        if self._diagnostic_path is None:
            return
        row = {'method': method, 'outcome': outcome,
               'elapsed_seconds': time.monotonic() - self._diagnostics_started}
        if status is not None:
            row['http_status'] = status
        if code is not None:
            row['driver_error_code'] = code
        with self._diagnostics_lock:
            events = self._diagnostics['events']
            events.append(row)
            # Keep the first and most recent events without URLs, bodies or tokens.
            if len(events) > 32:
                del events[16]
                self._diagnostics['omitted_events'] += 1
            pending = self._diagnostic_path.with_suffix('.pending')
            try:
                pending.write_text(json.dumps(self._diagnostics, indent=2) + '\n')
                pending.replace(self._diagnostic_path)
            except OSError:
                # Optional diagnostics must not replay an accepted control operation.
                self._diagnostics['write_failed'] = True
                try:
                    pending.unlink(missing_ok=True)
                except OSError:
                    pass

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise SidecarExecutionError('Driver operation observation deadline reached')
        return remaining

    def request(self, method, suffix, *, body=None, key=None, limit=_CONTROL_BYTES,
                provisioning_owner=None):
        creating = method == 'POST' and suffix == ''
        if creating and not key:
            raise SidecarExecutionError('GPU CREATE requires a stable idempotency key')
        if provisioning_owner is not None:
            if not creating:
                raise ValueError('provisioning authority is only valid on owned create')
            provisioning_owner.check_request(body, key, self.verification_id)
            body = {**body, 'controller_binding': provisioning_owner.claim}
        deadline = min(self.deadline, time.monotonic() + _CREATE_OBSERVATION_SECONDS) if creating else self.deadline
        delay = 1.0 if creating else 0.2
        create_started = time.monotonic()
        create_observation = None

        def create_failure(cause, *, status=None, code=None):
            return DriverCreateInfrastructureError(cause=cause,
                verification_id=self.verification_id, request_key=key, request_body=body,
                observation_seconds=max(0., time.monotonic() - create_started),
                owner_identity=provisioning_owner.identity,
                http_status=status, driver_error_code=code)

        def remaining():
            value = deadline - time.monotonic()
            if value <= 0:
                if creating and provisioning_owner is not None and create_observation is not None:
                    cause, status, code = create_observation
                    raise create_failure(cause, status=status, code=code)
                raise SidecarExecutionError('Driver operation observation deadline reached',
                                           error_code='driver_observation_timeout')
            return value

        headers = {'Authorization': 'Bearer ' + self._token, 'Accept': 'application/json'}
        if body is not None:
            headers['Content-Type'] = 'application/json'
            body = json.dumps(body, separators=(',', ':')).encode()
            headers['Content-Length'] = str(len(body))
        if key is not None:
            headers['Idempotency-Key'] = key
        url = self.url + suffix
        self._control_event(method, 'started')
        while True:
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                remaining()
                with _open_before_deadline(request, deadline) as (response, connection):
                    if response.status not in ({202} if method == 'POST' else {200}) or response.geturl() != url:
                        raise SidecarExecutionError('Driver did not accept the fixed GPU operation')
                    data = bytearray()
                    while True:
                        if response.isclosed():
                            break
                        connection.settimeout(remaining())
                        chunk = response.read1(min(65536, limit + 1 - len(data)))
                        if not chunk:
                            break
                        data.extend(chunk)
                        if len(data) > limit:
                            raise SidecarExecutionError('Driver response exceeds its declared byte bound')
                remaining()
                value = json.loads(data, object_pairs_hook=_unique_object)
                if not isinstance(value, dict):
                    raise SidecarExecutionError('Driver returned a non-object control response')
                if provisioning_owner is not None:
                    provisioning_owner.accept_success(value)
                self._control_event(method, 'accepted', status=response.status)
                return value
            except urllib.error.HTTPError as error:
                problem = None
                problem_code = None
                cleanup_uncertain = False
                try:
                    raw_problem = error.read(_CONTROL_BYTES + 1)
                except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
                    self._control_event(method, 'transport_error')
                    raw_problem = b''
                try:
                    if len(raw_problem) > _CONTROL_BYTES:
                        raise ValueError('Driver problem exceeds its byte bound')
                    problem = json.loads(raw_problem, object_pairs_hook=_unique_object)
                    if (isinstance(problem, dict) and problem.get('status') == error.code
                            and isinstance(problem.get('code'), str)
                            and re.fullmatch(r'[a-z][a-z0-9_]{0,63}', problem['code'])):
                        problem_code = problem['code']
                        title = problem.get('title', '')
                        cleanup_uncertain = (problem_code == 'gpu_cleanup_uncertain' or
                            isinstance(title, str) and any(marker in title.lower() for marker in
                                ('pending gpu cleanup failed', 'pending cleanup failed')))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
                self._control_event(method, 'http_error', status=error.code, code=problem_code)
                if (creating and error.code == 429 and problem_code == 'quota_exhausted'
                        and isinstance(problem, dict) and 'quota_failure' in problem
                        and error.headers.get_content_type() == 'application/problem+json'):
                    raise DriverQuotaError(problem['quota_failure']) from None
                if creating and cleanup_uncertain:
                    if provisioning_owner is not None and problem_code == 'gpu_cleanup_uncertain':
                        raise create_failure('driver_create_cleanup_uncertain',
                            status=error.code, code=problem_code) from None
                    raise SidecarExecutionError('Driver GPU CREATE cleanup is uncertain',
                        error_code='driver_cleanup_uncertain', http_status=error.code,
                        driver_error_code=problem_code) from None
                if provisioning_owner is not None and isinstance(problem, dict) and 'platform_failure' in problem:
                    if (set(problem) != {'status', 'code', 'title', 'platform_failure'}
                            or type(problem['status']) is not int or problem['status'] != error.code
                            or not isinstance(problem['title'], str)
                            or error.headers.get_content_type() != 'application/problem+json'):
                        raise ValueError('invalid authenticated Driver problem framing')
                    provenance = provisioning_owner.consume_failure(problem['platform_failure'],
                        status=error.code, code=problem['code'])
                    self._control_event(method, 'platform_capacity', status=error.code, code=problem['code'])
                    raise AuthenticatedProvisioningError(provenance) from None
                if (creating and provisioning_owner is not None
                        and 500 <= error.code <= 599 and problem_code == 'gpu_backend_unavailable'):
                    create_observation = ('driver_create_capacity_timeout', error.code, problem_code)
                # An unclassified 500 can hide failed compensation. Only the
                # Driver's compensated backend failure is safe to replay there.
                retryable = error.code in (425, 502, 503, 504)
                if creating and error.code == 429 and problem_code == 'mutation_rate_exhausted':
                    retryable = True
                if creating and 500 <= error.code <= 599 and problem_code == 'gpu_backend_unavailable':
                    retryable = True
                if not retryable or creating and problem_code in _TERMINAL_CREATE_CODES:
                    raise SidecarExecutionError(
                        f'Driver GPU endpoint returned HTTP {error.code}',
                        error_code='driver_http_error', http_status=error.code,
                        driver_error_code=problem_code) from None
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
                self._control_event(method, 'transport_error')
                if creating and provisioning_owner is not None:
                    create_observation = ('driver_create_transport_timeout', None, None)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise SidecarExecutionError('Driver control response is not JSON') from None
            # Reuse the same idempotency key/body after a lost response. Observing
            # a network error never creates a replacement parent.
            time.sleep(min(delay, remaining()))
            if creating:
                delay = min(delay * 2, _CREATE_BACKOFF_SECONDS)

    def pause(self):
        time.sleep(min(0.2, self.remaining()))
