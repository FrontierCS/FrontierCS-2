"""Evidence envelopes from the trusted scheduler.

Whatever the scheduler reports is wrapped here with its identity, its job and
the digest of the job's exact request bytes, and the CPU Verifier binds each
envelope to its own parent, phase, job sequence and request.
"""
from __future__ import annotations

import hashlib
import json

SCHEMA = 'frontier.vllm-attested-envelope/v1'
IDENTITY_FIELDS = frozenset({'run_id', 'attempt_id', 'operation_id', 'phase', 'name', 'profile_digest'})
KINDS = frozenset({'batch', 'correctness'})


class AttestationError(ValueError):
    pass


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def canonical_sha256(value):
    return 'sha256:' + hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_identity(identity):
    if (not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS
            or any(not isinstance(value, str) or not value for value in identity.values())
            or identity['phase'] not in {'agent', 'train', 'eval'}):
        raise AttestationError('invalid sidecar identity')
    return dict(identity)


def _envelope(identity, kind, sequence, request_sha256, payload):
    if kind not in KINDS:
        raise AttestationError('unknown envelope kind')
    if type(sequence) is not int or sequence < 0:
        raise AttestationError('envelope sequence must be a non-negative integer')
    if (not isinstance(request_sha256, str) or not request_sha256.startswith('sha256:')
            or len(request_sha256) != 71):
        raise AttestationError('envelope must bind one canonical request digest')
    if not isinstance(payload, dict):
        raise AttestationError('envelope payload must be an object')
    return {'schema': SCHEMA, 'identity': validate_identity(identity), 'kind': kind,
            'sequence': sequence, 'request_sha256': request_sha256, 'payload': payload}


class Envelopes:
    def __init__(self, identity):
        self.identity = validate_identity(identity)

    def wrap(self, *, kind, sequence, request_sha256, payload):
        return {'schema': SCHEMA, 'envelope': _envelope(self.identity, kind, sequence, request_sha256, payload)}


def unwrap(wrapped):
    """Return the envelope after checking its shape; binding is the caller's."""
    if (not isinstance(wrapped, dict) or set(wrapped) != {'schema', 'envelope'}
            or wrapped['schema'] != SCHEMA):
        raise AttestationError('invalid envelope shape')
    envelope = wrapped['envelope']
    if (not isinstance(envelope, dict)
            or set(envelope) != {'schema', 'identity', 'kind', 'sequence', 'request_sha256', 'payload'}
            or envelope['schema'] != SCHEMA):
        raise AttestationError('invalid envelope fields')
    checked = _envelope(envelope['identity'], envelope['kind'], envelope['sequence'],
                        envelope['request_sha256'], envelope['payload'])
    if checked != envelope:
        raise AttestationError('envelope fields differ from their canonical form')
    return envelope


def bind(envelope, *, operation_id, phase, name, kind, sequence, request_sha256):
    """Bind an envelope to the owned parent, phase, scheduler, job and request."""
    identity = envelope['identity']
    if (identity['operation_id'] != operation_id or identity['phase'] != phase
            or identity['name'] != name):
        raise AttestationError('envelope identity belongs to another operation or sidecar')
    if (envelope['kind'] != kind or envelope['sequence'] != sequence
            or envelope['request_sha256'] != request_sha256):
        raise AttestationError('envelope answers another job or request')
    return envelope['payload']
