"""CPU trust binding from Driver-injected identities and image-local content."""
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

from controller_envelope import Context, _bounded, digest, slots_digest
from terminal_contract import canonical

CONTRACT_SCHEMA = 'frontier.vllm-controller-content-contract-draft/v1'
LIMITS_SCHEMA = 'frontier.vllm-controller-source-allocation-draft/v1'
ENVIRONMENT = {
    'run_id': 'FRONTIER_RUN_ID',
    'attempt_id': 'FRONTIER_ATTEMPT_ID',
    'verification_id': 'FRONTIER_VERIFICATION_ID',
    'task_config_digest': 'FRONTIER_TASK_CONFIG_DIGEST',
    'executable_image_digest': 'FRONTIER_VERIFIER_IMAGE_DIGEST',
}


def load_limits(path):
    """Load a readonly publisher allocation, not a scientific runtime budget."""
    maximum = len(canonical({'schema': LIMITS_SCHEMA,
        'maximum_contract_bytes': sys.maxsize, 'maximum_source_bytes': sys.maxsize}))
    with Path(path).open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('source allocation artifact exceeds its structural bound')
    value = json.loads(raw)
    if (not isinstance(value, dict) or set(value) !=
            {'schema', 'maximum_contract_bytes', 'maximum_source_bytes'}
            or value['schema'] != LIMITS_SCHEMA
            or any(type(value[key]) is not int or not 0 < value[key] <= sys.maxsize
                   for key in ('maximum_contract_bytes', 'maximum_source_bytes'))
            or canonical(value) != raw):
        raise ValueError('noncanonical immutable source allocation artifact')
    return {key: value[key] for key in ('maximum_contract_bytes', 'maximum_source_bytes')}


def _source(path, root, maximum):
    relative = PurePosixPath(path)
    if (not isinstance(path, str) or not path or relative.is_absolute()
            or '..' in relative.parts or str(relative) != path):
        raise ValueError('contract requires exact relative source paths')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('controller source cannot traverse symlinks')
    if not current.is_file():
        raise ValueError('controller contract source is not a regular file')
    hashed, observed = hashlib.sha256(), 0
    with current.open('rb') as stream:
        while True:
            part = stream.read(min(65536, maximum + 1 - observed))
            if not part:
                break
            observed += len(part)
            if observed > maximum:
                raise ValueError('controller source exceeds publisher allocation')
            hashed.update(part)
    return 'sha256:' + hashed.hexdigest(), observed


def content_contract(root, paths, *, maximum_contract_bytes, maximum_source_bytes):
    """Build-time helper; no image digest is embedded into its own image."""
    if (not isinstance(paths, (tuple, list)) or not paths or
            any(type(n) is not int or n <= 0 for n in (maximum_contract_bytes, maximum_source_bytes))
            or len(paths) > maximum_contract_bytes or len(set(paths)) != len(paths)):
        raise ValueError('explicit bounded unique controller source list required')
    files, remaining = {}, maximum_source_bytes
    for path in sorted(paths):
        files[path], size = _source(path, Path(root), remaining)
        remaining -= size
    contract = {'schema': CONTRACT_SCHEMA, 'files': files}
    _bounded(contract, maximum_contract_bytes)
    return contract


def load_content_binding(contract_path, source_root, *,
                         maximum_contract_bytes, maximum_source_bytes):
    """Validate image-local content before either provisioning or scientific work."""
    if any(type(n) is not int or n <= 0 for n in (maximum_contract_bytes, maximum_source_bytes)):
        raise ValueError('explicit CPU contract allocation required')
    with Path(contract_path).open('rb') as stream:
        raw = stream.read(maximum_contract_bytes + 1)
    if len(raw) > maximum_contract_bytes:
        raise ValueError('CPU content contract exceeds publisher allocation')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate CPU content-contract key')
            value[key] = item
        return value
    contract = json.loads(raw, object_pairs_hook=unique)
    _bounded(contract, maximum_contract_bytes)
    if (not isinstance(contract, dict) or set(contract) != {'schema', 'files'}
            or contract['schema'] != CONTRACT_SCHEMA or not isinstance(contract['files'], dict)):
        raise ValueError('unknown CPU content contract')
    observed = content_contract(source_root, list(contract['files']),
        maximum_contract_bytes=maximum_contract_bytes, maximum_source_bytes=maximum_source_bytes)
    if observed != contract:
        raise ValueError('CPU source content differs from immutable image contract')
    return {'source_content_digest': digest(canonical(contract['files'])),
            'contract_digest': digest(canonical(contract))}


def load_context(environment, contract_path, source_root, slots, *,
                 maximum_contract_bytes, maximum_source_bytes):
    """All errors here are CPU integrity defects, never candidate attribution."""
    binding = load_content_binding(contract_path, source_root,
        maximum_contract_bytes=maximum_contract_bytes, maximum_source_bytes=maximum_source_bytes)
    values = {field: environment[name] for field, name in ENVIRONMENT.items()}
    return Context(**values, **binding,
                   expected_slots_digest=slots_digest(slots))
