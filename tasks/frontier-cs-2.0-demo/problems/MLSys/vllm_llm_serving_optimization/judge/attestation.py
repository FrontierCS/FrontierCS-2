"""Signed evidence envelopes from the trusted scheduler sidecar.

The Driver generates one Ed25519 key per GPU operation, hands the private half
only to sidecars declared with ``attestation_key`` and reports the public half
on the runner view. Whatever the scheduler returns through the untrusted parent
is signed here and verified by the CPU Verifier against that view, so the parent
can delay or drop evidence but never alter or replay it.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives.serialization import load_ssh_private_key, load_ssh_public_key

SCHEMA = 'frontier.vllm-attested-envelope/v1'
DOMAIN = b'frontier-vllm-attestation-v1\n'
IDENTITY_FIELDS = frozenset({'run_id', 'attempt_id', 'operation_id', 'phase', 'name', 'profile_digest'})
KINDS = frozenset({'batch', 'correctness'})
B64URL = re.compile(r'[A-Za-z0-9_-]+')


class AttestationError(ValueError):
    pass


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def canonical_sha256(value):
    return 'sha256:' + hashlib.sha256(canonical_bytes(value)).hexdigest()


def key_id(public_key_openssh):
    fields = public_key_openssh.strip().split()
    if len(fields) not in {2, 3} or fields[0] != 'ssh-ed25519':
        raise AttestationError('expected one OpenSSH Ed25519 public key')
    try:
        blob = base64.b64decode(fields[1], validate=True)
    except (ValueError, TypeError) as error:
        raise AttestationError('invalid OpenSSH public key') from error
    return 'SHA256:' + base64.b64encode(hashlib.sha256(blob).digest()).rstrip(b'=').decode()


def validate_identity(identity):
    if (not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS
            or any(not isinstance(value, str) or not value for value in identity.values())
            or identity['phase'] not in {'agent', 'train', 'eval'}):
        raise AttestationError('invalid Driver sidecar identity')
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


class Signer:
    def __init__(self, private_key_openssh, public_key_openssh, identity):
        self.private_key = load_ssh_private_key(private_key_openssh.encode(), password=None)
        public = load_ssh_public_key(public_key_openssh.encode())
        if self.private_key.public_key().public_bytes_raw() != public.public_bytes_raw():
            raise AttestationError('attestation private/public key mismatch')
        self.key_id = key_id(public_key_openssh)
        self.identity = validate_identity(identity)

    def sign(self, *, kind, sequence, request_sha256, payload):
        envelope = _envelope(self.identity, kind, sequence, request_sha256, payload)
        message = DOMAIN + canonical_bytes(envelope)
        signature = base64.urlsafe_b64encode(self.private_key.sign(message)).rstrip(b'=').decode()
        return {'schema': SCHEMA, 'envelope': envelope, 'key_id': self.key_id, 'signature': signature}


def verify(signed, public_key_openssh):
    """Return the envelope after checking its signature; binding is the caller's."""
    if (not isinstance(signed, dict) or set(signed) != {'schema', 'envelope', 'key_id', 'signature'}
            or signed['schema'] != SCHEMA):
        raise AttestationError('invalid signed envelope shape')
    if signed['key_id'] != key_id(public_key_openssh):
        raise AttestationError('envelope was signed by another key')
    signature = signed['signature']
    if not isinstance(signature, str) or not B64URL.fullmatch(signature) or '=' in signature:
        raise AttestationError('invalid envelope signature encoding')
    raw = base64.urlsafe_b64decode(signature + '=' * (-len(signature) % 4))
    if len(raw) != 64:
        raise AttestationError('invalid envelope signature length')
    envelope = signed['envelope']
    if (not isinstance(envelope, dict)
            or set(envelope) != {'schema', 'identity', 'kind', 'sequence', 'request_sha256', 'payload'}
            or envelope['schema'] != SCHEMA):
        raise AttestationError('invalid envelope fields')
    checked = _envelope(envelope['identity'], envelope['kind'], envelope['sequence'],
                        envelope['request_sha256'], envelope['payload'])
    if checked != envelope:
        raise AttestationError('envelope fields differ from their canonical form')
    try:
        load_ssh_public_key(public_key_openssh.encode()).verify(raw, DOMAIN + canonical_bytes(envelope))
    except (InvalidSignature, UnsupportedAlgorithm, ValueError, TypeError) as error:
        raise AttestationError('invalid envelope signature') from error
    return envelope


def bind(envelope, *, operation_id, phase, name, kind, sequence, request_sha256):
    """Bind a verified envelope to the owned parent, phase, scheduler, job and request."""
    identity = envelope['identity']
    if (identity['operation_id'] != operation_id or identity['phase'] != phase
            or identity['name'] != name):
        raise AttestationError('envelope identity belongs to another operation or sidecar')
    if (envelope['kind'] != kind or envelope['sequence'] != sequence
            or envelope['request_sha256'] != request_sha256):
        raise AttestationError('envelope answers another job or request')
    return envelope['payload']
