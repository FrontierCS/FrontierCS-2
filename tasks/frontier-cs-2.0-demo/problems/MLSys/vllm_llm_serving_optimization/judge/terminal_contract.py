"""Local controller terminal records; no scores, infra inference or attestation."""
from dataclasses import asdict, dataclass
import json
import math

SCHEMA = 'frontier.vllm-controller-terminal-draft/v1'
STATUSES = frozenset({'complete', 'incorrect', 'runtime_error', 'timeout'})
CAUSES = frozenset({'normal_completion', 'child_exit', 'protocol_error',
                    'output_limit', 'deadline', 'unknown_exception', 'collector_failure'})


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


@dataclass(frozen=True)
class Slot:
    run_id: str
    attempt_id: str
    verification_id: str
    operation_id: str
    phase: str
    group: str
    variant: str
    instance_id: str
    source_digest: str
    request_digest: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v or len(v.encode()) > 256 for v in asdict(self).values()):
            raise ValueError('bounded nonempty slot identities required')
        if self.phase not in {'train', 'eval'} or self.variant not in {'vanilla', 'reference', 'candidate'}:
            raise ValueError('unknown phase or variant')
        for digest in (self.source_digest, self.request_digest):
            if len(digest) != 71 or not digest.startswith('sha256:') or any(c not in '0123456789abcdef' for c in digest[7:]):
                raise ValueError('exact SHA256 binding required')

    def key(self):
        return canonical(asdict(self)).decode()


def _bounded_result(result, max_bytes, max_depth):
    if type(max_bytes) is not int or max_bytes <= 0 or type(max_depth) is not int or max_depth <= 0:
        raise ValueError('explicit child result resource bounds required')
    stack, visited = [(result, 0)], 0
    while stack:
        item, depth = stack.pop()
        visited += 1
        if depth > max_depth or visited > max_bytes + 1:
            raise ValueError('child result nesting or node allocation exceeds injected bound')
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError('result object keys must be strings')
            stack.extend((value, depth + 1) for value in item.values())
        elif isinstance(item, list):
            stack.extend((value, depth + 1) for value in item)
    if len(canonical(result)) > max_bytes:
        raise ValueError('child result exceeds injected byte bound')


def terminal(slot, *, status, cause, elapsed, cap, streams=None, result=None, result_limit_bytes=None, result_limit_depth=None):
    if status not in STATUSES or cause not in CAUSES:
        raise ValueError('unknown terminal disposition')
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in (elapsed, cap)) or cap == 0:
        raise ValueError('finite elapsed and positive injected cap required')
    if (status == 'complete') != (cause == 'normal_completion') or (status == 'timeout') != (cause == 'deadline'):
        raise ValueError('inconsistent terminal cause')
    if status == 'complete' and not isinstance(result, dict):
        raise ValueError('completed child requires a result object')
    if status == 'complete':
        _bounded_result(result, result_limit_bytes, result_limit_depth)
    if status != 'complete' and result is not None:
        raise ValueError('failed execution cannot carry successful output')
    if status != 'complete' and (result_limit_bytes is not None or result_limit_depth is not None):
        raise ValueError('failed execution has no successful result allocation')
    streams = {} if streams is None else streams
    if not isinstance(streams, dict) or set(streams) - {'stdout', 'stderr'}:
        raise ValueError('unknown stream metadata')
    for stream in streams.values():
        keys = {'limit_bytes', 'observed_bytes', 'retained_bytes', 'retained_prefix_sha256',
                'stream_fully_observed', 'truncated'}
        if not isinstance(stream, dict) or set(stream) != keys:
            raise ValueError('invalid bounded stream metadata')
        limit, observed, retained = (stream[k] for k in ('limit_bytes', 'observed_bytes', 'retained_bytes'))
        if any(type(n) is not int for n in (limit, observed, retained)) or limit <= 0 or not 0 <= retained <= min(observed, limit) or not 0 <= observed <= limit + 1:
            raise ValueError('invalid stream byte counts')
        digest = stream['retained_prefix_sha256']
        if not isinstance(digest, str) or len(digest) != 71 or not digest.startswith('sha256:') or any(c not in '0123456789abcdef' for c in digest[7:]):
            raise ValueError('invalid stream prefix digest')
        if any(type(stream[k]) is not bool for k in ('stream_fully_observed', 'truncated')):
            raise ValueError('invalid stream observation flags')
        if stream['truncated'] != (not stream['stream_fully_observed'] or observed > retained):
            raise ValueError('inconsistent stream observation flags')
    value = {'schema': SCHEMA, 'slot': asdict(slot), 'status': status, 'cause': cause,
             'elapsed_seconds': elapsed, 'cap_seconds': cap, 'streams': streams or {},
             'result': result, 'result_limit_bytes': result_limit_bytes, 'result_limit_depth': result_limit_depth,
             'authority': 'local-controller-draft-unattested'}
    canonical(value)
    return value


def validate_record(value):
    if not isinstance(value, dict) or value.get('schema') != SCHEMA or value.get('authority') != 'local-controller-draft-unattested':
        raise ValueError('unknown local draft record')
    terminal(Slot(**value['slot']), status=value['status'], cause=value['cause'],
             elapsed=value['elapsed_seconds'], cap=value['cap_seconds'],
             streams=value['streams'], result=value['result'], result_limit_bytes=value['result_limit_bytes'], result_limit_depth=value['result_limit_depth'])
    allowed = {'schema', 'slot', 'status', 'cause', 'elapsed_seconds', 'cap_seconds',
               'streams', 'result', 'result_limit_bytes', 'result_limit_depth', 'authority', 'candidate_input_dispatched',
               'exit_code', 'failure_context'}
    required = allowed - {'candidate_input_dispatched', 'exit_code', 'failure_context'}
    if set(value) - allowed or required - set(value):
        raise ValueError('unrecognized terminal authority fields')
    for field in ('candidate_input_dispatched',):
        if field in value and type(value[field]) is not bool:
            raise ValueError('invalid controller observation')
    if value.get('exit_code') is not None and type(value['exit_code']) is not int:
        raise ValueError('invalid child exit code')
    if 'failure_context' in value:
        from containment_policy import Boundary, child_failure_metadata
        context = value['failure_context']
        if value['status'] == 'complete' or not isinstance(context, dict) or set(context) != {'boundary', 'decision'}:
            raise ValueError('invalid failed execution boundary')
        boundary = Boundary(**context['boundary'])
        if boundary.kind == 'platform' or context != child_failure_metadata(
                Slot(**value['slot']), boundary, value.get('candidate_input_dispatched')):
            raise ValueError('child failure cannot claim platform attribution')
