"""CPU controller child containment draft; production wiring/attestation deferred."""
from dataclasses import dataclass
import hashlib
import json
import math
import os
import selectors
import signal
import struct
import subprocess
import time

from terminal_contract import terminal
from containment_policy import Boundary, child_failure_metadata
from scheduler_stream import Decoder, decode

CHILD_SCHEMA = 'frontier.vllm-child-result-draft/v1'
RPC_SCHEMA = 'frontier.vllm-episode-rpc-draft/v1'


class TrustedControllerError(RuntimeError):
    """Local signing/persistence defects must never become candidate science."""


class CaseBoundaryFailure(RuntimeError):
    """Trusted broker observed a failed model-used service request."""


@dataclass(frozen=True)
class Limits:
    total_seconds: float
    cleanup_seconds: float
    frame_bytes: int
    stderr_bytes: int
    json_depth: int

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (self.total_seconds, self.cleanup_seconds)):
            raise ValueError('explicit positive finite injected deadlines required')
        if any(type(v) is not int or v <= 0 for v in (self.frame_bytes, self.stderr_bytes, self.json_depth)):
            raise ValueError('explicit positive byte bounds required')


class Capture:
    def __init__(self, limit):
        self.limit, self.prefix, self.observed = limit, bytearray(), 0
        self.complete = False

    def add(self, data):
        self.observed += len(data)
        self.prefix.extend(data[:max(0, self.limit - len(self.prefix))])
        return self.observed > self.limit

    def summary(self):
        return {'limit_bytes': self.limit, 'observed_bytes': self.observed, 'retained_bytes': len(self.prefix),
                'retained_prefix_sha256': 'sha256:' + hashlib.sha256(self.prefix).hexdigest(),
                'stream_fully_observed': self.complete, 'truncated': not self.complete or self.observed > len(self.prefix)}


def _reap(process, seconds):
    if process is None:
        return True
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        return False
    return True


def _exit_observation(process):
    # Keep the owned session leader unreaped until group cleanup, preventing
    # PID/PGID reuse between result selection and signalling its descendants.
    value = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if value is None:
        return None
    if value.si_code == os.CLD_EXITED:
        return value.si_status
    return -value.si_status


def _cleanup(process, selection, seconds):
    errors = []
    try:
        if not _reap(process, seconds):
            errors.append('reap')
    except Exception:
        errors.append('reap')
    if selection is not None:
        try:
            selection.close()
        except Exception:
            errors.append('selector_close')
    if process is not None:
        for name in ('stdin', 'stdout', 'stderr'):
            try:
                pipe = getattr(process, name)
                if pipe is not None:
                    pipe.close()
            except Exception:
                errors.append(name + '_close')
    return {'confirmed': not errors, 'errors': errors}


def _decode_frame(raw, limit):
    if len(raw) < 4:
        raise ValueError('missing frame')
    length, = struct.unpack('!I', raw[:4])
    if length > limit or len(raw) != length + 4:
        raise ValueError('invalid frame length')
    def reject_constant(value):
        raise ValueError('nonfinite child JSON')
    value = json.loads(raw[4:].decode('utf-8'), parse_constant=reject_constant)
    if not isinstance(value, dict) or set(value) != {'schema', 'result'} or value['schema'] != CHILD_SCHEMA or not isinstance(value['result'], dict):
        raise ValueError('invalid child result protocol')
    return value['result']


def run(ledger, slot, command, limits, *, boundary=None, input_document=None, rpc=None):
    boundary = Boundary('candidate_service', 'execution') if boundary is None else boundary
    if boundary is not None and (not isinstance(boundary, Boundary) or boundary.kind == 'platform'):
        raise ValueError('child execution requires a candidate service or SWE case boundary')
    existing = ledger.read(slot)
    if existing is not None:
        return existing
    if not ledger.mark_dispatched(slot):
        existing = ledger.read(slot)
        if existing is None:
            raise RuntimeError('owned slot was already dispatched; reconcile instead of replaying')
        return existing
    started = time.monotonic()
    deadline = started + limits.total_seconds
    process, selection = None, None
    captures = {'stdout': Capture(limits.frame_bytes + 4), 'stderr': Capture(limits.stderr_bytes)}
    status, cause, result = 'runtime_error', 'unknown_exception', None
    observed_exit = None
    interactive = input_document is not None
    pending_input = bytearray()
    final_results = []
    rpc_sequence = 0
    if interactive:
        raw = json.dumps(input_document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if len(raw) > limits.frame_bytes or not callable(rpc):
            raise ValueError('bounded interactive input and trusted broker required')
        pending_input.extend(struct.pack('!I', len(raw)) + raw)
    def receive_frame(raw):
        nonlocal rpc_sequence
        value = decode(raw)
        if not isinstance(value, dict) or final_results:
            raise ValueError('invalid episode worker frame lifecycle')
        if value.get('schema') == CHILD_SCHEMA:
            if set(value) != {'schema', 'result'} or not isinstance(value['result'], dict):
                raise ValueError('invalid final episode worker result')
            final_results.append(value['result'])
            return
        if (set(value) != {'schema', 'sequence', 'method', 'arguments'} or value['schema'] != RPC_SCHEMA
                or type(value['sequence']) is not int or value['sequence'] != rpc_sequence
                or value['method'] not in {'run', 'read_patch'} or not isinstance(value['arguments'], dict)):
            raise ValueError('invalid closed episode worker RPC')
        response = rpc(value['method'], value['arguments'], deadline)
        encoded = json.dumps({'schema': RPC_SCHEMA, 'sequence': rpc_sequence, 'result': response},
                             sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if len(encoded) > limits.frame_bytes or pending_input:
            raise ValueError('episode broker response exceeds its bounded exchange')
        pending_input.extend(struct.pack('!I', len(encoded)) + encoded)
        selection.register(process.stdin, selectors.EVENT_WRITE, 'stdin')
        rpc_sequence += 1
    decoder = (Decoder(maximum_frame_bytes=limits.frame_bytes,
                       maximum_total_bytes=limits.frame_bytes, accept=receive_frame) if interactive else None)
    try:
        selection = selectors.DefaultSelector()
        process = subprocess.Popen(command, stdin=subprocess.PIPE if interactive else subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
                                   env={'PATH': os.defpath, 'PYTHONIOENCODING': 'utf-8'})
        for name, pipe in (('stdout', process.stdout), ('stderr', process.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selection.register(pipe, selectors.EVENT_READ, name)
        if interactive:
            os.set_blocking(process.stdin.fileno(), False)
            selection.register(process.stdin, selectors.EVENT_WRITE, 'stdin')
        while selection.get_map() or observed_exit is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status, cause = 'timeout', 'deadline'
                break
            exceeded = False
            for key, _ in selection.select(min(remaining, 0.02)):
                if key.data == 'stdin':
                    size = os.write(key.fd, pending_input[:65536])
                    del pending_input[:size]
                    if not pending_input:
                        selection.unregister(key.fileobj)
                    continue
                capture = captures[key.data]
                data = os.read(key.fd, min(65536, max(1, capture.limit + 1 - capture.observed)))
                if not data:
                    capture.complete = True
                    selection.unregister(key.fileobj)
                elif capture.add(data):
                    exceeded = True
                    break
                elif interactive and key.data == 'stdout':
                    decoder.feed(data)
            if exceeded:
                cause = 'output_limit'
                break
            if not interactive and captures['stdout'].observed >= 4:
                declared, = struct.unpack('!I', captures['stdout'].prefix[:4])
                if declared > limits.frame_bytes:
                    cause = 'output_limit'
                    break
            observed_exit = _exit_observation(process)
        else:
            if observed_exit != 0:
                cause = 'child_exit'
            else:
                try:
                    if interactive:
                        decoder.finish()
                        if len(final_results) != 1 or pending_input:
                            raise ValueError('episode worker did not complete its bounded protocol')
                        result = final_results[0]
                    else:
                        result = _decode_frame(bytes(captures['stdout'].prefix), limits.frame_bytes)
                    status, cause = 'complete', 'normal_completion'
                except Exception:
                    cause = 'protocol_error'
    except TrustedControllerError:
        _cleanup(process, selection, limits.cleanup_seconds)
        raise
    except CaseBoundaryFailure:
        status, cause, result = 'runtime_error', 'child_exit', None
        boundary = Boundary('swe_case', 'tool')
    except Exception:
        status, cause, result = 'runtime_error', 'unknown_exception', None
    ended = time.monotonic()
    if ended >= deadline and status == 'complete':
        status, cause, result = 'timeout', 'deadline', None
    committed = False
    try:
        try:
            record = terminal(slot, status=status, cause=cause, elapsed=ended - started,
                              cap=limits.total_seconds,
                              streams={name: capture.summary() for name, capture in captures.items()}, result=result,
                              result_limit_bytes=limits.frame_bytes if status == 'complete' else None,
                              result_limit_depth=limits.json_depth if status == 'complete' else None)
        except Exception:
            # Decoding alone is not sufficient: child nesting/encoding can fail
            # when wrapped in the controller record. No child data enters fallback.
            record = terminal(slot, status='runtime_error', cause='protocol_error',
                              elapsed=ended - started, cap=limits.total_seconds)
        record['candidate_input_dispatched'] = ledger.dispatched(slot)
        record['exit_code'] = observed_exit
        if record['status'] != 'complete':
            record['failure_context'] = child_failure_metadata(
                slot, boundary, record['candidate_input_dispatched'])
        winner = ledger.commit_once(slot, record)[1]
        committed = True
    finally:
        # The immutable primary result precedes every cleanup action. Ledger failures
        # remain explicit trusted-controller defects, but still kill owned children.
        cleanup = _cleanup(process, selection, limits.cleanup_seconds)
    if committed:
        ledger.record_cleanup(slot, cleanup)
    return winner
