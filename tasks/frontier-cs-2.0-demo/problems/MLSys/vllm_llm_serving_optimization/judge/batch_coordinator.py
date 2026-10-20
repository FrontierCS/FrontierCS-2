"""Local batch adapter only; Driver lifecycle and trusted transport are absent.

Descriptors must eventually come from the owning Driver operation. Their fields
and readiness echoes are identity checks, not authentication or proof of locality.
"""
from __future__ import annotations

import base64
import http.client
import copy
import hashlib
import json
import math
import random
import threading
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler

from instance_manifest import expected_groups
from serving_eval import agent_runner, bfcl
from serving_eval.tool_auth import ToolAuthSigner
from serving_eval.settings import EpisodeSettings
from serving_eval.swe_inputs import load_swe_inputs


TOOL_TRANSPORT_GRACE_SECONDS = 30.0
# Equal to swe_tool_service.MAX_PATCH_SECONDS (that module is not importable
# here: it lives in the testbed image). Worst case per episode with the single
# retry: 2 x (180 + 30) s = 420 s of transport.
PATCH_TIMEOUT_SECONDS = 180
TESTBED_READY_SECONDS = 660
TRACE_PREVIEW_BYTES = 2048
# Final (eval) hidden arrival rates in jobs/s, configured by the task author
# for evaluation. Public rates stay 0.5/1.0.
FINAL_HIDDEN_JPS = {'swe': 1.0, 'bfcl': 1.25}


class BackendError(RuntimeError):
    pass


class TestbedTransportError(BackendError):
    """Typed read-transport failure; no child payload or text attribution."""
    def __init__(self, code):
        if code not in {'connection', 'timeout', 'http_5xx'}:
            raise ValueError('unknown testbed transport code')
        super().__init__('declared testbed read transport failed')
        self.code = code


def _digest(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _trace_receipt(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False,
    ).encode()
    receipt = {
        'bytes': len(raw),
        'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(),
    }
    if len(raw) <= TRACE_PREVIEW_BYTES:
        receipt['content'] = value
    else:
        half = TRACE_PREVIEW_BYTES // 2
        receipt['encoding'] = 'base64'
        receipt['prefix'] = base64.b64encode(raw[:half]).decode()
        receipt['suffix'] = base64.b64encode(raw[-half:]).decode()
        receipt['truncated'] = True
    return receipt


def _compact_trace_event(event):
    if not isinstance(event, dict) or not isinstance(event.get('kind'), str):
        raise BackendError('trace event is malformed')
    compact = {
        key: event[key]
        for key in (
            'kind', 'path', 'started_monotonic', 'ended_monotonic',
            'error_origin', 'error_type',
        )
        if key in event
    }
    raw = json.dumps(
        event, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False,
    ).encode()
    compact['event_bytes'] = len(raw)
    compact['event_sha256'] = 'sha256:' + hashlib.sha256(raw).hexdigest()
    for field in ('request', 'response', 'calls', 'results'):
        if field in event:
            compact[field] = _trace_receipt(event[field])
    return compact


def _compact_record_trace(record):
    """Replace transport-heavy trace data with bounded, verifiable receipts."""
    try:
        record['trace'] = [_compact_trace_event(event) for event in record['trace']]
    except Exception as exc:
        record['episode'] = None
        record['execution_error'] = {
            'origin': 'trusted_trace_compaction',
            'type': type(exc).__name__,
        }
        record['requires_error_attribution'] = True
        record['trace'] = []


def _endpoint(value):
    parsed = urlsplit(value)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BackendError('invalid prepared endpoint')
    return value.rstrip('/')


@dataclass(frozen=True)
class PreparedTestbed:
    """One declared interactive testbed: its bridge endpoint and Driver name.

    The Driver starts the container from the TaskConfig profile of that name,
    so the name identifies the declared image and command; the phase binds it
    to the parent's train/eval generation.
    """
    instance_id: str
    endpoint: str
    child_id: str
    phase: str


@dataclass(frozen=True)
class PreparedGroup:
    operation_id: str
    group_id: str
    placement_id: str
    mode: str
    workload: str
    rate: str
    model_endpoint: str
    model_source_digest: str
    testbeds: tuple[PreparedTestbed, ...] = ()

    def identity(self):
        return {'operation_id': self.operation_id, 'group_id': self.group_id, 'placement_id': self.placement_id}


def _request(endpoint, path, body=None, timeout=60, *, signer, target_sidecar_id, transport=None):
    data = b'' if body is None else json.dumps(body).encode()
    method = 'GET' if body is None else 'POST'
    try:
        headers = signer.headers(target_sidecar_id, method, path, data)
    except Exception as error:
        from instance_supervisor import TrustedControllerError
        raise TrustedControllerError('trusted tool signer failed') from error
    if body is not None:
        headers['Content-Type'] = 'application/json'
    if transport is not None:
        from bounded_http import request_signed
        return request_signed(_endpoint(endpoint) + path, method, headers,
            None if body is None else data.decode(), **transport)
    request = Request(_endpoint(endpoint) + path, data=None if body is None else data,
                      headers=headers, method=method)
    # Local bridge traffic must not silently follow an operator's HTTP proxy.
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        if 500 <= exc.code <= 599:
            raise TestbedTransportError('http_5xx') from exc
        raise BackendError('testbed HTTP request rejected') from exc
    except TimeoutError as exc:
        raise TestbedTransportError('timeout') from exc
    except (URLError, ConnectionError, http.client.HTTPException) as exc:
        raise TestbedTransportError('connection') from exc
    except Exception as exc:
        raise BackendError('testbed transport failed: ' + type(exc).__name__) from exc
    if not isinstance(result, dict):
        raise BackendError('testbed returned a malformed response')
    return result


class TestbedClient:
    def __init__(self, group, descriptor, trace, *, signer=None):
        self.endpoint = _endpoint(descriptor.endpoint)
        self.identity = {**group.identity(), **{k: v for k, v in asdict(descriptor).items() if k != 'endpoint'}}
        self.trace = trace
        self.signer = signer if signer is not None else ToolAuthSigner.from_environment()
        self.transport = None
        self.tainted = False

    def _call(self, path, body=None, *, timeout=None):
        start = time.perf_counter()
        event = {'kind': 'testbed', 'path': path, 'started_monotonic': start}
        if body is not None:
            event['request'] = copy.deepcopy(body)
        try:
            transport_timeout = (60 if body is None else body['timeout_seconds'] + TOOL_TRANSPORT_GRACE_SECONDS) if timeout is None else timeout
            transport = None if self.transport is None else dict(self.transport,
                deadline=min(self.transport['deadline'], time.monotonic() + transport_timeout))
            result = _request(self.endpoint, path, None if body is None else {'identity': self.identity, **body},
                              timeout=transport_timeout, signer=self.signer,
                              target_sidecar_id=self.identity['child_id'], transport=transport)
            if result.get('identity') != self.identity:
                raise BackendError('testbed response identity mismatch')
            if 'case_failure' in result:
                from serving_eval.swe_case_failure import validate_failure
                validate_failure(result['case_failure'])
                self.tainted = True
                raise BackendError('authenticated SWE service returned case-local failure')
            event['response'] = copy.deepcopy(result)
            return result
        except Exception as exc:
            if self.transport is not None:
                self.tainted = True
            event['error_origin'] = 'testbed_transport_or_protocol'
            event['error_type'] = type(exc).__name__
            raise
        finally:
            event['ended_monotonic'] = time.perf_counter()
            self.trace.append(event)

    def ready(self, *, timeout=None, deadline=None):
        # Preparation is outside the timed batch. Keep each transport bounded
        # while allowing a cold testbed to finish its once-only index refresh.
        local_deadline = time.monotonic() + TESTBED_READY_SECONDS
        deadline = local_deadline if deadline is None else min(deadline, local_deadline)
        first_transport_error = None
        delay = .1
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if first_transport_error is not None:
                    raise first_transport_error
                raise BackendError('testbed is not ready')
            try:
                result = self._call('/ready', timeout=min(remaining, 60 if timeout is None else timeout))
            except TestbedTransportError as error:
                if first_transport_error is None:
                    first_transport_error = error
                time.sleep(min(delay, max(0, deadline - time.monotonic())))
                delay = min(2., delay * 2)
                continue
            if result.get('ready') is True:
                return
            if result.get('preparation_complete') is True:
                raise BackendError('testbed index preparation failed')
            if result.get('preparation_complete') is not False:
                raise BackendError('testbed is not ready')
            time.sleep(0.1)

    def run(self, command, *, timeout):
        result = self._call('/execute', {'command': command, 'timeout_seconds': timeout})
        code, output = result.get('exit_code'), result.get('output')
        if isinstance(code, bool) or not isinstance(code, int) or not isinstance(output, str):
            raise BackendError('testbed command result is malformed')
        return code, output

    def read_patch(self):
        result = self._call('/patch', {'timeout_seconds': PATCH_TIMEOUT_SECONDS})
        diagnostics = result.get('diagnostics')
        if (result.get('exit_code') == 124 and isinstance(diagnostics, dict)
                and diagnostics.get('stage') == 'git_diff'):
            # One retry of a timed-out read only: /patch diffs against a
            # disposable copy of the index and never writes the worktree or
            # .git/index, so a second read is idempotent (and page-cache warm).
            result = self._call('/patch', {'timeout_seconds': PATCH_TIMEOUT_SECONDS})
        if result.get('exit_code') != 0 or not isinstance(result.get('patch'), str):
            raise BackendError('testbed patch extraction failed')
        return result['patch']

    def shutdown(self):
        """Ask a used testbed to kill its model-owned processes and exit."""
        result = self._call('/shutdown', {'timeout_seconds': 5}, timeout=15)
        if result.get('shutdown') is not True:
            raise BackendError('testbed did not confirm shutdown')


class ModelClient:
    def __init__(self, endpoint, timeout, trace):
        try:
            from openai import APIStatusError, OpenAI, DefaultHttpxClient
        except ImportError as exc:
            raise BackendError('original OpenAI client dependency is unavailable') from exc
        self.client = OpenAI(base_url=_endpoint(endpoint), api_key='EMPTY', timeout=timeout, http_client=DefaultHttpxClient(trust_env=False))
        self.api_status_error = APIStatusError
        self.trace = trace
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        event = {'kind': 'model', 'started_monotonic': time.perf_counter(), 'request': copy.deepcopy(kwargs)}
        try:
            result = self.client.chat.completions.create(**kwargs)
            event['response'] = result.model_dump()
            return result
        except Exception as exc:
            event['error_origin'] = ('model_api_response' if isinstance(exc, self.api_status_error)
                                     else 'model_transport_or_client_unclassified')
            event['error_type'] = type(exc).__name__
            raise
        finally:
            event['ended_monotonic'] = time.perf_counter()
            self.trace.append(event)

    def ready(self, model):
        if model not in {entry.id for entry in self.client.models.list().data}:
            raise BackendError('prepared inference endpoint lacks the fixed model')

    def close(self):
        self.client.close()


def poisson_schedule(workload, count, jps):
    if workload not in {'swe', 'bfcl'} or not math.isfinite(jps) or jps <= 0:
        raise ValueError('invalid original workload arrival rate')
    rng = random.Random(20260604 if workload == 'swe' else 20260605)
    offsets, elapsed = [], 0.0
    for _ in range(count):
        elapsed += rng.expovariate(jps)
        offsets.append(elapsed)
    return offsets


def run_group(group: PreparedGroup, *, swe_instances=None, jps=None, calibration=False,
              tool_signer=None, on_progress=None, execution=None):
    """Run one prepared original group, returning evidence only, never a score.

    A hidden group runs only at its configured FINAL_HIDDEN_JPS rate, in final
    evaluation or in labelled calibration. One thread per original arrival is
    retained. The separate fresh-testbed correctness phase and independent final adapter
    still need integration; SWE patches here are not resolve-rate results.
    """
    settings = EpisodeSettings()
    key = group.workload + '/' + group.rate
    groups = expected_groups(group.mode)
    if key not in groups or not all((group.operation_id, group.group_id, group.placement_id, group.model_source_digest)):
        raise BackendError('incomplete or invalid prepared group identity')
    ids = groups[key]
    public_jps = {'swe': 0.5, 'bfcl': 1.0}[group.workload]
    if group.rate == 'public':
        if jps is not None and jps != public_jps:
            raise BackendError('public arrival rate differs from original task')
        jps = public_jps
    elif type(calibration) is not bool or jps != FINAL_HIDDEN_JPS[group.workload]:
        raise BackendError('hidden arrival rate differs from the configured final rate')
    setup_start = time.perf_counter()
    traces = {ident: [] for ident in ids}
    episode_ledger, slots = None, {}
    if execution is not None:
        from terminal_contract import Slot
        from terminal_ledger import Ledger
        episode_ledger = Ledger(execution['journal_path'])
        for ident in ids:
            slots[ident] = Slot(execution['run_id'], execution['attempt_id'], execution['verification_id'],
                group.operation_id, group.mode, key, execution['variant'], ident,
                group.model_source_digest, execution['request_sha256'])
            episode_ledger.register(slots[ident], group.operation_id + '/' + execution['variant'] + '/' + ident)
    if group.workload == 'swe':
        fixed_inputs = [{'instance_id': row['instance_id'], 'problem_statement': row['problem_statement']} for row in load_swe_inputs(group.mode)]
        supplied = fixed_inputs if swe_instances is None else list(swe_instances)
        if supplied != fixed_inputs:
            raise BackendError('SWE problem statements differ from frozen original inputs')
        if tuple(row.get('instance_id') for row in supplied) != ids or any(not isinstance(row.get('problem_statement'), str) for row in supplied):
            raise BackendError('prepared SWE inputs differ from the fixed ordered instance list')
        if tuple(item.instance_id for item in group.testbeds) != ids:
            raise BackendError('all fixed fresh SWE testbeds must be prepared first')
        if (len({item.child_id for item in group.testbeds}) != len(ids)
                or any(not item.child_id or item.phase != group.mode for item in group.testbeds)):
            raise BackendError('testbed identities are missing, reused or from another phase')
        instances = supplied
        testbeds = {item.instance_id: TestbedClient(group, item, traces[item.instance_id], signer=tool_signer) for item in group.testbeds}
        for client in testbeds.values():
            client.ready(**({'deadline': execution['deadline']} if execution is not None else {}))
    else:
        if group.testbeds or swe_instances is not None:
            raise BackendError('BFCL uses its original in-process memory backend')
        instances = bfcl.load_memory_instances(ids)
        # Verify every original scenario snapshot and search dependency before
        # the barrier, without changing its per-episode fresh-memory semantics.
        from serving_eval.bfcl_vendor.memory_kv import BM25Plus
        if BM25Plus is None:
            raise BackendError('original BFCL search backend is unavailable')
        for instance in instances:
            bfcl.load_snapshot(instance['scenario'])
        testbeds = {}
    model = ModelClient(group.model_endpoint, 900 if group.workload == 'swe' else 300, [])
    try:
        model.ready(settings.model)
    finally:
        model.close()
    setup_end = time.perf_counter()
    schedule = poisson_schedule(group.workload, len(ids), jps)
    origin = time.perf_counter()
    results, lock = {}, threading.Lock()

    def run(instance, offset):
        ident = instance['instance_id'] if group.workload == 'swe' else instance['id']
        delay = offset - (time.perf_counter() - origin)
        if delay > 0:
            time.sleep(delay)
        dispatch = time.perf_counter()
        record = {'instance_id': ident, 'scheduled_offset_seconds': offset, 'dispatched_monotonic': dispatch, 'arrival_lateness_seconds': max(0, dispatch - origin - offset), 'trace': traces[ident]}
        client = None
        compacted = False
        try:
            if on_progress is not None:
                on_progress('dispatch', {'instance_id': ident})
            if execution is not None:
                from dispatch_broker import run_episode
                observed = run_episode(episode_ledger, slots[ident], instance,
                    workload=group.workload, endpoint=group.model_endpoint, settings=settings,
                    testbed=testbeds.get(ident), deadline=execution['deadline'],
                    cleanup_seconds=execution['cleanup_seconds'], maximum_frame_bytes=execution['frame_bytes'],
                    maximum_stderr_bytes=execution['stderr_bytes'])
                if observed['status'] == 'complete':
                    record.update(observed['result'])
                    model_trace = record['trace']
                    record['trace'] = sorted(model_trace + [_compact_trace_event(event) for event in traces[ident]],
                                             key=lambda event: event.get('started_monotonic', 0))
                else:
                    # Preserve the local supervisor observation before it is signed
                    # by scheduler progress; never invent a completed episode.
                    record['supervised_terminal'] = observed
                    record['episode'] = None
                    record['execution_error'] = {'origin': 'supervised_episode', 'type': observed['cause']}
                    record['failure_context'] = observed['failure_context']
                    record['requires_error_attribution'] = True
                    _compact_record_trace(record)
                compacted = True
                if group.workload == 'swe' and record.get('episode') is not None:
                    record['patch_sha256'] = 'sha256:' + hashlib.sha256(record['episode']['patch'].encode()).hexdigest()
            else:
                client = ModelClient(group.model_endpoint, 900 if group.workload == 'swe' else 300, traces[ident])
                if group.workload == 'swe':
                    outcome = agent_runner.run_instance(instance, client=client, prepared_testbed=testbeds[ident], settings=settings)
                else:
                    outcome = bfcl.run_memory_instance(instance, client=client, settings=settings, tool_trace=traces[ident])
                record['episode'] = asdict(outcome)
                if group.workload == 'swe':
                    record['patch_sha256'] = 'sha256:' + hashlib.sha256(outcome.patch.encode()).hexdigest()
                record['requires_error_attribution'] = any('error_origin' in event for event in traces[ident])
        except Exception as exc:
            record['episode'] = None
            record['execution_error'] = {'origin': 'coordinator_or_testbed_unclassified', 'type': type(exc).__name__}
            record['requires_error_attribution'] = True
        finally:
            if client is not None:
                client.close()
            if not compacted:
                _compact_record_trace(record)
            if on_progress is not None:
                on_progress('terminal', record)
            with lock:
                results[ident] = record

    threads = [threading.Thread(target=run, args=(instance, offset), daemon=True) for instance, offset in zip(instances, schedule)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if set(results) != set(ids):
        raise BackendError('batch lost an expected episode')
    return {
        'schema': 'frontier.vllm-local-batch/v1',
        'identity': group.identity(),
        'mode': group.mode,
        'group': key,
        'model_source_digest': group.model_source_digest,
        'input_digest': _digest(instances),
        'calibration_only': calibration,
        'jps': jps,
        'arrival_seed': 20260604 if group.workload == 'swe' else 20260605,
        'setup_started_monotonic': setup_start,
        'setup_ended_monotonic': setup_end,
        'schedule_origin_monotonic': origin,
        'expected_instances': len(ids),
        'episodes': [results[i] for i in ids],
        'score_ready': False,
        'remaining': [
            'deployed Driver-owned lifecycle and trusted same-Sandbox invocation',
            'SWE worker isolation and separate fresh correctness-phase integration',
            'trusted final measurement projection and error attribution',
        ],
    }
