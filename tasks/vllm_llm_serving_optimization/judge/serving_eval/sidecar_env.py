"""The identity and the secret that the Verifier gives each container of a parent's pod.

The Verifier starts every service with ``FRONTIER_SIDECAR_IDENTITY_JSON``, and
this module reads it once at start. Every container of the pod reaches every
other's ports, each caller from the same address, so the Verifier also gives
the scheduler a key of the lease, ``FRONTIER_SIDECAR_KEY``, and each SWE
service its token, ``FRONTIER_SIDECAR_TOKEN``, which the scheduler derives
from the key and the service's container name. A service answers only a
request that carries its token.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re

IDENTITY_ENV = 'FRONTIER_SIDECAR_IDENTITY_JSON'
IDENTITY_FIELDS = frozenset({'run_id', 'attempt_id', 'operation_id', 'phase', 'name', 'profile_digest'})
KEY_ENV = 'FRONTIER_SIDECAR_KEY'
TOKEN_ENV = 'FRONTIER_SIDECAR_TOKEN'
SECRET = re.compile(r'[0-9a-f]{64}')


class SidecarEnvironmentError(ValueError):
    pass


def read_identity(environment=None):
    environment = os.environ if environment is None else environment
    try:
        raw = json.loads(environment[IDENTITY_ENV])
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SidecarEnvironmentError('sidecar identity is unavailable') from error
    if (not isinstance(raw, dict) or set(raw) != IDENTITY_FIELDS
            or any(not isinstance(value, str) or not value or '\r' in value or '\n' in value
                   for value in raw.values())
            or raw['phase'] not in {'agent', 'train', 'eval'}):
        raise SidecarEnvironmentError('invalid sidecar identity')
    return dict(raw)


def service_token(key, name):
    """The token of the service in container ``name``: one container's token opens no other's."""
    return hmac.new(bytes.fromhex(key), name.encode(), hashlib.sha256).hexdigest()


def read_secret(variable, environment=None):
    """The key or the token in ``variable``, 64 hexadecimal digits."""
    environment = os.environ if environment is None else environment
    value = environment.get(variable, '')
    if not SECRET.fullmatch(value):
        raise SidecarEnvironmentError('sidecar secret is unavailable')
    return value


def take_token():
    """This service's token, which no process that it starts inherits."""
    token = read_secret(TOKEN_ENV)
    del os.environ[TOKEN_ENV]
    return token
