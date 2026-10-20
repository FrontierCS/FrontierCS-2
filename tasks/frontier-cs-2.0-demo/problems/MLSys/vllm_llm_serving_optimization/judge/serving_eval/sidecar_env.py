"""Map the Driver's declared-sidecar environment onto the existing tool auth.

The Driver gives every declared sidecar ``FRONTIER_SIDECAR_IDENTITY_JSON`` and
the operation's attestation public key; only sidecars declared with
``attestation_key`` also receive the private key. This module validates that
contract once at start, derives the ``FRONTIER_TOOL_AUTH_*`` variables that
``tool_auth`` already consumes, and removes the Driver variables so model
commands and test scripts never inherit them.
"""
from __future__ import annotations

import json
import os

IDENTITY_ENV = 'FRONTIER_SIDECAR_IDENTITY_JSON'
PRIVATE_KEY_ENV = 'FRONTIER_SIDECAR_ATTESTATION_PRIVATE_KEY'
PUBLIC_KEY_ENV = 'FRONTIER_SIDECAR_ATTESTATION_PUBLIC_KEY'
IDENTITY_FIELDS = frozenset({'run_id', 'attempt_id', 'operation_id', 'phase', 'name', 'profile_digest'})
TOOL_AUTH_VARIABLES = ('FRONTIER_TOOL_AUTH_PRIVATE_KEY_OPENSSH', 'FRONTIER_TOOL_AUTH_PUBLIC_KEY_OPENSSH',
                       'FRONTIER_TOOL_AUTH_KEY_ID', 'FRONTIER_TOOL_AUTH_CONTEXT_JSON')


class SidecarEnvironmentError(ValueError):
    pass


def read_identity(environment=None):
    environment = os.environ if environment is None else environment
    try:
        raw = json.loads(environment[IDENTITY_ENV])
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SidecarEnvironmentError('Driver sidecar identity is unavailable') from error
    if (not isinstance(raw, dict) or set(raw) != IDENTITY_FIELDS
            or any(not isinstance(value, str) or not value or '\r' in value or '\n' in value
                   for value in raw.values())
            or raw['phase'] not in {'agent', 'train', 'eval'}):
        raise SidecarEnvironmentError('invalid Driver sidecar identity')
    return dict(raw)


def install(role, environment=None):
    """Return the Driver identity after deriving tool-auth variables for ``role``.

    ``signer`` is the attesting scheduler and requires the private key;
    ``verifier`` is any other sidecar and refuses to start with one.
    """
    from .tool_auth import _key_id  # local import keeps this module importable without cryptography

    environment = os.environ if environment is None else environment
    if role not in {'signer', 'verifier'}:
        raise SidecarEnvironmentError('unknown sidecar auth role')
    identity = read_identity(environment)
    public = environment.get(PUBLIC_KEY_ENV)
    if not isinstance(public, str) or not public.strip():
        raise SidecarEnvironmentError('attestation public key is unavailable')
    private = environment.get(PRIVATE_KEY_ENV)
    if role == 'signer' and not private:
        raise SidecarEnvironmentError('attesting sidecar started without its private key')
    if role == 'verifier' and private is not None:
        raise SidecarEnvironmentError('non-attesting sidecar received a private key')
    if any(name in environment for name in TOOL_AUTH_VARIABLES):
        raise SidecarEnvironmentError('tool auth variables must come from the Driver identity only')
    context = {'schema_version': 'frontier.tool-auth-context/v1', 'run_id': identity['run_id'],
               'attempt_id': identity['attempt_id'], 'operation_id': identity['operation_id'],
               'sidecar_id': identity['name']}
    environment['FRONTIER_TOOL_AUTH_PUBLIC_KEY_OPENSSH'] = public.strip()
    environment['FRONTIER_TOOL_AUTH_KEY_ID'] = _key_id(public)
    environment['FRONTIER_TOOL_AUTH_CONTEXT_JSON'] = json.dumps(context, separators=(',', ':'))
    if role == 'signer':
        environment['FRONTIER_TOOL_AUTH_PRIVATE_KEY_OPENSSH'] = private
    for name in (IDENTITY_ENV, PRIVATE_KEY_ENV, PUBLIC_KEY_ENV):
        environment.pop(name, None)
    return identity


def scrub(environment=None):
    """Drop the derived tool-auth variables once the service holds its objects."""
    environment = os.environ if environment is None else environment
    for name in TOOL_AUTH_VARIABLES:
        environment.pop(name, None)
