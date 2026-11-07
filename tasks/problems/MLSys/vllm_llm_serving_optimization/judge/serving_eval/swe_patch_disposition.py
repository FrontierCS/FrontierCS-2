"""Bounded patch metadata shared by the trusted scheduler and correctness worker.

No gold tests or answers are needed by the scheduler. It binds patch metadata to
its retained timed batch; the worker independently checks the full wire size
against its own fixed instructions before issuing an incorrect disposition.
"""
import hashlib
import json
import re

SWE_CORRECTNESS_REQUEST_BYTES = 2 * 1024**2
OVERSIZE_REQUEST_SCHEMA = 'frontier.vllm-swe-oversize-request/v1'
SWE_TEST_TIMEOUT_SECONDS = 1200


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def correctness_envelope(request, identity, variant):
    return json.dumps({'identity': identity, 'variant': variant, 'request': request}).encode()


def patch_metadata(patch):
    if not isinstance(patch, str):
        raise ValueError('timed patch is not explicit text')
    return {'candidate_patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
            'candidate_patch_bytes': len(patch.encode()),
            'candidate_patch_json_bytes': len(json.dumps(patch).encode())}


def oversize_request(instance_id, patch, *, identity, variant, original_request_bytes):
    request = {'schema': OVERSIZE_REQUEST_SCHEMA, 'instance_id': instance_id,
               'correctness_identity': identity, 'variant': variant, **patch_metadata(patch),
               'original_request_bytes': original_request_bytes,
               'request_limit_bytes': SWE_CORRECTNESS_REQUEST_BYTES,
               'timeout_seconds': SWE_TEST_TIMEOUT_SECONDS}
    request['request_sha256'] = digest(request)
    return request


def validate_disposition(request, *, patch=None):
    fields = {'schema', 'instance_id', 'correctness_identity', 'variant', 'candidate_patch_sha256',
              'candidate_patch_bytes', 'candidate_patch_json_bytes', 'original_request_bytes',
              'request_limit_bytes', 'timeout_seconds', 'request_sha256'}
    identity = request.get('correctness_identity') if isinstance(request, dict) else None
    identity_fields = {'operation_id', 'group_id', 'placement_id', 'instance_id', 'child_id', 'phase'}
    if (not isinstance(request, dict) or set(request) != fields
            or request['schema'] != OVERSIZE_REQUEST_SCHEMA
            or request['variant'] not in ('vanilla', 'reference', 'candidate')
            or not isinstance(identity, dict) or set(identity) != identity_fields
            or any(not isinstance(v, str) or not v for v in identity.values())
            or identity['instance_id'] != request['instance_id'] or identity['phase'] not in ('train', 'eval')
            or type(request['timeout_seconds']) is not int or request['timeout_seconds'] != SWE_TEST_TIMEOUT_SECONDS
            or type(request['request_limit_bytes']) is not int
            or request['request_limit_bytes'] != SWE_CORRECTNESS_REQUEST_BYTES
            or any(type(request[k]) is not int or request[k] <= 0 for k in
                   ('candidate_patch_bytes', 'candidate_patch_json_bytes', 'original_request_bytes'))
            or request['original_request_bytes'] <= SWE_CORRECTNESS_REQUEST_BYTES
            or any(not isinstance(request[k], str) or not re.fullmatch('[0-9a-f]{64}', request[k])
                   for k in ('candidate_patch_sha256', 'request_sha256'))
            or request['request_sha256'] != digest({k:v for k,v in request.items() if k!='request_sha256'})):
        raise ValueError('oversize disposition is malformed or unbound')
    if patch is not None and any(request[k] != value for k,value in patch_metadata(patch).items()):
        raise ValueError('oversize disposition differs from the retained timed patch')
    return {k:request[k] for k in ('candidate_patch_sha256', 'candidate_patch_bytes',
            'candidate_patch_json_bytes', 'original_request_bytes', 'request_limit_bytes')}
