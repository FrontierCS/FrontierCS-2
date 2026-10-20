"""Per-operation Ed25519 authentication for trusted scheduler -> SWE HTTP."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import threading
import time
from cryptography.hazmat.primitives.serialization import load_ssh_private_key, load_ssh_public_key

VERSION = '1'
DOMAIN = 'frontier-tool-auth-v1'
MAX_VALIDITY_SECONDS = 30
MAX_NONCES = 8192
CONTEXT_FIELDS = frozenset({'schema_version', 'run_id', 'attempt_id', 'operation_id', 'sidecar_id'})
HEADERS = {
    'version': 'X-Frontier-Auth-Version',
    'key_id': 'X-Frontier-Key-Id',
    'issued': 'X-Frontier-Issued-At',
    'expires': 'X-Frontier-Expires-At',
    'nonce': 'X-Frontier-Nonce',
    'signature': 'X-Frontier-Signature',
}
B64URL = re.compile(r'[A-Za-z0-9_-]+')
DECIMAL = re.compile(r'0|[1-9][0-9]*')


class ToolAuthError(ValueError):
    pass


def _context(raw):
    value = json.loads(raw) if isinstance(raw, str) else raw
    if (not isinstance(value, dict) or set(value) != CONTEXT_FIELDS
            or value.get('schema_version') != 'frontier.tool-auth-context/v1'
            or any(not isinstance(value[k], str) or not value[k] or '\r' in value[k] or '\n' in value[k]
                   for k in CONTEXT_FIELDS - {'schema_version'})):
        raise ToolAuthError('invalid tool authentication context')
    return value


def _decode_url(value, name, *, expected_bytes=None):
    if not isinstance(value, str) or not B64URL.fullmatch(value) or '=' in value:
        raise ToolAuthError('invalid ' + name)
    try:
        raw = base64.urlsafe_b64decode(value + '=' * (-len(value) % 4))
    except Exception as error:
        raise ToolAuthError('invalid ' + name) from error
    if base64.urlsafe_b64encode(raw).rstrip(b'=').decode() != value or (expected_bytes is not None and len(raw) != expected_bytes):
        raise ToolAuthError('invalid ' + name)
    return raw


def _key_id(public_key):
    fields = public_key.strip().split()
    if len(fields) not in {2, 3} or fields[0] != 'ssh-ed25519':
        raise ToolAuthError('expected one OpenSSH Ed25519 public key')
    try:
        blob = base64.b64decode(fields[1], validate=True)
    except Exception as error:
        raise ToolAuthError('invalid OpenSSH public key') from error
    return 'SHA256:' + base64.b64encode(hashlib.sha256(blob).digest()).rstrip(b'=').decode()


def _canonical(key_id, context, target_sidecar_id, method, request_target, body_sha256,
               issued_at, expires_at, nonce):
    values = [DOMAIN, key_id, context['run_id'], context['attempt_id'], context['operation_id'],
              target_sidecar_id, method, request_target, body_sha256, str(issued_at),
              str(expires_at), nonce]
    if any(not isinstance(value, str) or not value or '\r' in value or '\n' in value for value in values):
        raise ToolAuthError('authentication values must be nonempty single lines')
    if method != method.upper() or not request_target.startswith('/') or not re.fullmatch(r'[0-9a-f]{64}', body_sha256):
        raise ToolAuthError('invalid canonical request')
    return ('\n'.join(values) + '\n').encode()


class ToolAuthSigner:
    def __init__(self, private_key_openssh, public_key_openssh, key_id, context, *, now=time.time):
        self.context = _context(context)
        if _key_id(public_key_openssh) != key_id:
            raise ToolAuthError('tool key fingerprint mismatch')
        self.private_key = load_ssh_private_key(private_key_openssh.encode(), password=None)
        public = load_ssh_public_key(public_key_openssh.encode())
        if self.private_key.public_key().public_bytes_raw() != public.public_bytes_raw():
            raise ToolAuthError('tool private/public key mismatch')
        self.key_id = key_id
        self.now = now

    @classmethod
    def from_environment(cls):
        required = ('FRONTIER_TOOL_AUTH_PRIVATE_KEY_OPENSSH', 'FRONTIER_TOOL_AUTH_PUBLIC_KEY_OPENSSH',
                    'FRONTIER_TOOL_AUTH_KEY_ID', 'FRONTIER_TOOL_AUTH_CONTEXT_JSON')
        try:
            values = [os.environ[key] for key in required]
        except KeyError as error:
            raise ToolAuthError('trusted scheduler authentication is unavailable') from error
        return cls(*values)

    def headers(self, target_sidecar_id, method, request_target, body):
        issued = int(self.now())
        expires = issued + MAX_VALIDITY_SECONDS
        nonce = base64.urlsafe_b64encode(secrets.token_bytes(16)).rstrip(b'=').decode()
        body_hash = hashlib.sha256(body).hexdigest()
        message = _canonical(self.key_id, self.context, target_sidecar_id, method,
                             request_target, body_hash, issued, expires, nonce)
        signature = base64.urlsafe_b64encode(self.private_key.sign(message)).rstrip(b'=').decode()
        return {HEADERS['version']: VERSION, HEADERS['key_id']: self.key_id,
                HEADERS['issued']: str(issued), HEADERS['expires']: str(expires),
                HEADERS['nonce']: nonce, HEADERS['signature']: signature}


class ToolAuthVerifier:
    def __init__(self, public_key_openssh, key_id, context, *, now=time.time, max_nonces=MAX_NONCES):
        self.context = _context(context)
        if _key_id(public_key_openssh) != key_id:
            raise ToolAuthError('tool key fingerprint mismatch')
        self.public_key = load_ssh_public_key(public_key_openssh.encode())
        self.key_id = key_id
        self.now = now
        if type(max_nonces) is not int or max_nonces <= 0:
            raise ToolAuthError('invalid nonce cache bound')
        self.max_nonces = max_nonces
        self.nonces = {}
        self.lock = threading.Lock()

    @classmethod
    def from_environment(cls):
        if 'FRONTIER_TOOL_AUTH_PRIVATE_KEY_OPENSSH' in os.environ:
            raise ToolAuthError('tool verifier received scheduler private key')
        required = ('FRONTIER_TOOL_AUTH_PUBLIC_KEY_OPENSSH', 'FRONTIER_TOOL_AUTH_KEY_ID',
                    'FRONTIER_TOOL_AUTH_CONTEXT_JSON')
        try:
            values = [os.environ[key] for key in required]
        except KeyError as error:
            raise ToolAuthError('tool verifier authentication is unavailable') from error
        return cls(*values)

    def verify(self, headers, method, request_target, body):
        try:
            version = headers[HEADERS['version']]
            key_id = headers[HEADERS['key_id']]
            issued_raw = headers[HEADERS['issued']]
            expires_raw = headers[HEADERS['expires']]
            nonce = headers[HEADERS['nonce']]
            signature = headers[HEADERS['signature']]
        except (KeyError, TypeError) as error:
            raise ToolAuthError('missing tool authentication header') from error
        if version != VERSION or key_id != self.key_id or not DECIMAL.fullmatch(issued_raw) or not DECIMAL.fullmatch(expires_raw):
            raise ToolAuthError('invalid tool authentication header')
        issued, expires, current = int(issued_raw), int(expires_raw), int(self.now())
        if issued > current or current > expires or not 0 < expires - issued <= MAX_VALIDITY_SECONDS:
            raise ToolAuthError('expired or invalid tool authentication interval')
        _decode_url(nonce, 'nonce', expected_bytes=16)
        raw_signature = _decode_url(signature, 'signature', expected_bytes=64)
        message = _canonical(key_id, self.context, self.context['sidecar_id'], method,
                             request_target, hashlib.sha256(body).hexdigest(), issued, expires, nonce)
        try:
            self.public_key.verify(raw_signature, message)
        except Exception as error:
            raise ToolAuthError('invalid tool request signature') from error
        with self.lock:
            self.nonces = {value: until for value, until in self.nonces.items() if until >= current}
            if nonce in self.nonces:
                raise ToolAuthError('replayed tool request nonce')
            if len(self.nonces) >= self.max_nonces:
                raise ToolAuthError('tool request nonce cache is full')
            self.nonces[nonce] = expires
