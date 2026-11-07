"""Publisher-owned runtime fields projected from the actual TaskConfig.

This selects no scientific quota. Publication must compare the projection with
its final TaskConfig. Image identities (the CPU/GPU images and every sidecar
image digest) are deliberately omitted: the CPU image embeds this projection
and is built in the same importer pass as the sidecar images, so including
their digests would be a cycle. The Driver pins each sidecar image itself.
"""
from __future__ import annotations

import copy
import hashlib
import json

from sidecar_catalog import (CHECK_PORT, MAX_SIDECARS_PER_PARENT, NAME, SCHEDULER, SCHEDULER_PORT,
                             TESTBED_PORT, VARIANTS, check_name, expected_sidecars, instance_index,
                             phases_for_instance, testbed_name, CANDIDATE_ONLY)
from instance_manifest import expected_groups


SCHEMA = 'frontier.vllm-runtime-config/v3'
TASK_CONFIG_SCHEMA = 'frontier.task-config/v1'
POLICY_SCHEMA = 'frontier.frontiersmith-task-policy/v1'
SIDECAR_MIN_CPU_MILLIS = 32   # Modal admission floor per sidecar container.
SIDECAR_MIN_MEMORY_BYTES = 32 * 1024 * 1024


class RuntimeConfigurationError(ValueError):
    pass


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise RuntimeConfigurationError(name + ' must be an explicit positive integer')
    return value


def canonical_digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def _catalog(task):
    """Sidecar declarations of a TaskConfig (`images.gpu_sidecars`) or a policy (`sidecars`)."""
    try:
        if task.get('schema_version') == TASK_CONFIG_SCHEMA:
            profiles = task['images']['gpu_sidecars']
            resources = task['images']['gpu_parent_resources']['verifier']
        elif task.get('schema_version') == POLICY_SCHEMA:
            profiles = task['sidecars']
            resources = task['gpu_parent_resources']['verifier']
        else:
            raise RuntimeConfigurationError('expected the TaskConfig v1 or task policy v1 schema')
    except (KeyError, TypeError) as error:
        raise RuntimeConfigurationError('task lacks its declared sidecar catalog or parent allocation') from error
    if not isinstance(profiles, dict):
        raise RuntimeConfigurationError('sidecar catalog must be an object')
    catalog = {}
    for name, profile in profiles.items():
        if not isinstance(profile, dict) or not set(profile) <= {'image', 'args', 'workdir', 'phases', 'attestation_key'}:
            raise RuntimeConfigurationError('invalid or unknown sidecar profile fields')
        try:
            projected = {'args': profile['args'], 'workdir': profile['workdir'], 'phases': profile['phases']}
        except KeyError as error:
            raise RuntimeConfigurationError('sidecar profile lacks its fixed command') from error
        if profile.get('attestation_key', False) is True:
            projected['attestation_key'] = True
        elif profile.get('attestation_key', False) is not False:
            raise RuntimeConfigurationError('attestation_key must be a boolean')
        catalog[name] = projected
    return catalog, resources


def project_task_config(task):
    """Project only existing platform fields; do not put provider data in a task."""
    if not isinstance(task, dict):
        raise RuntimeConfigurationError('expected a TaskConfig or policy object')
    catalog, resources = _catalog(task)
    try:
        value = {
            'schema': SCHEMA,
            'verifier_quota': task['quota']['verifier'],
            'audit': task['network']['audit'],
            'max_evidence_bytes': task['verifier']['max_evidence_bytes'],
            'gpu_parent_resources': resources,
            'sidecars': catalog,
        }
    except (KeyError, TypeError) as error:
        raise RuntimeConfigurationError('TaskConfig is missing an explicit vLLM runtime field') from error
    value = copy.deepcopy(value)
    validate_projection(value)
    return value


def verify_projection(value, task):
    validate_projection(value)
    if value != project_task_config(task):
        raise RuntimeConfigurationError('runtime projection differs from publication TaskConfig')


def _sha256(value):
    return (isinstance(value, str) and value.startswith('sha256:') and len(value) == 71
            and all(c in '0123456789abcdef' for c in value[7:]))


def _command(profile):
    args, workdir = profile['args'], profile['workdir']
    if (not isinstance(args, list) or not args or not args[0]
            or any(not isinstance(arg, str) or '\0' in arg for arg in args)
            or not isinstance(workdir, str) or not workdir.startswith('/') or '\0' in workdir):
        raise RuntimeConfigurationError('profile command and absolute workdir must be fixed')


def fixed_arg(profile, name):
    args = profile['args']
    if args.count(name) != 1 or args.index(name) + 1 >= len(args):
        raise RuntimeConfigurationError('publisher must declare one fixed ' + name)
    return args[args.index(name) + 1]


def execution_variants(config):
    """Policy comes from immutable publisher scheduler args, never a request."""
    scheduler = config['sidecars'][SCHEDULER]
    if '--variants' not in scheduler['args']:
        return VARIANTS  # Old calibration versions and internal legacy fixtures.
    if fixed_arg(scheduler, '--variants') != 'candidate':
        raise RuntimeConfigurationError('unknown scheduler variant policy')
    return CANDIDATE_ONLY


def _validate_catalog(profiles, resources):
    if not isinstance(profiles, dict) or not profiles:
        raise RuntimeConfigurationError('missing publisher sidecar catalog')
    for name, profile in profiles.items():
        if not isinstance(name, str) or not NAME.fullmatch(name) or name == 'main' or not isinstance(profile, dict):
            raise RuntimeConfigurationError('invalid fixed sidecar name or profile')
        if set(profile) not in ({'args', 'workdir', 'phases'}, {'args', 'workdir', 'phases', 'attestation_key'}):
            raise RuntimeConfigurationError('unknown or incomplete sidecar profile fields')
        if 'attestation_key' in profile and profile['attestation_key'] is not True:
            raise RuntimeConfigurationError('attestation_key is only recorded when true')
        _command(profile)
        phases = profile['phases']
        if (not isinstance(phases, list) or not phases
                or any(not isinstance(p, str) or p not in {'agent', 'train', 'eval'} for p in phases)
                or len(set(phases)) != len(phases)):
            raise RuntimeConfigurationError('invalid sidecar phases')
    attesting = {name for name, profile in profiles.items() if 'attestation_key' in profile}
    if attesting != {SCHEDULER}:
        raise RuntimeConfigurationError('exactly the scheduler sidecar holds the attestation key')
    scheduler = profiles[SCHEDULER]
    variants = execution_variants({'sidecars': profiles})
    if set(scheduler['phases']) != {'train', 'eval'} or fixed_arg(scheduler, '--port') != str(SCHEDULER_PORT):
        raise RuntimeConfigurationError('scheduler must serve both Verifier phases on its fixed port')
    largest = 0
    for mode in ('train', 'eval'):
        declared = {name for name, profile in profiles.items() if mode in profile['phases']}
        if declared != set(expected_sidecars(mode, variants=variants)):
            raise RuntimeConfigurationError('declared sidecars differ from the fixed catalog of ' + mode)
        largest = max(largest, len(declared))
    if any('agent' in profile['phases'] for profile in profiles.values()):
        raise RuntimeConfigurationError('this task declares no Agent-phase sidecars')
    for ident in expected_groups('eval')['swe/public']:
        index = instance_index(ident)
        names = [testbed_name(index, variant) for variant in variants] + [check_name(index)]
        for name in names:
            profile = profiles[name]
            port = CHECK_PORT if name == check_name(index) else TESTBED_PORT
            if (profile['phases'] != phases_for_instance(ident) or profile['workdir'] != '/testbed'
                    or fixed_arg(profile, '--instance-id') != ident or fixed_arg(profile, '--port') != str(port)):
                raise RuntimeConfigurationError('SWE sidecar profile does not bind its fixed instance, port or phases')
            if name == check_name(index):
                if variants == CANDIDATE_ONLY:
                    if fixed_arg(profile, '--variants') != 'candidate':
                        raise RuntimeConfigurationError('candidate-only correctness order is not pinned')
                elif '--variants' in profile['args']:
                    raise RuntimeConfigurationError('legacy correctness order cannot be overridden')
    if largest > MAX_SIDECARS_PER_PARENT:
        raise RuntimeConfigurationError('catalog exceeds the provider sidecar cap')
    if not isinstance(resources, dict) or set(resources) != {'cpu_millis', 'memory_bytes'}:
        raise RuntimeConfigurationError('Verifier GPU host allocation must be explicit')
    _positive(resources['cpu_millis'], 'GPU host CPU')
    if _positive(resources['memory_bytes'], 'GPU host memory') % (1024 * 1024):
        raise RuntimeConfigurationError('GPU host memory must be whole MiB')
    if (resources['cpu_millis'] < largest * SIDECAR_MIN_CPU_MILLIS
            or resources['memory_bytes'] < largest * SIDECAR_MIN_MEMORY_BYTES):
        raise RuntimeConfigurationError('parent reservation cannot admit every declared sidecar')


def validate_projection(value):
    keys = {'schema', 'verifier_quota', 'audit', 'max_evidence_bytes', 'gpu_parent_resources', 'sidecars'}
    if not isinstance(value, dict) or set(value) != keys or value['schema'] != SCHEMA:
        raise RuntimeConfigurationError('invalid closed runtime projection')
    quota = value['verifier_quota']
    quota_keys = {'max_train_calls', 'max_train_wall_seconds', 'max_gpu_instances',
                  'max_gpu_seconds', 'eval_timeout_seconds'}
    if not isinstance(quota, dict) or set(quota) != quota_keys:
        raise RuntimeConfigurationError('incomplete explicit Verifier quota')
    for key in quota_keys:
        _positive(quota[key], key)
    audit = value['audit']
    if not isinstance(audit, dict):
        raise RuntimeConfigurationError('missing audit limits')
    for key in ('max_model_sessions', 'max_ssh_sessions', 'max_ssh_bytes_per_direction',
                'max_bundle_bytes'):
        _positive(audit.get(key), key)
    if audit['max_model_sessions'] + audit['max_ssh_sessions'] > 2048:
        raise RuntimeConfigurationError('combined model and SSH session limit exceeds 2048')
    pools = audit.get('ssh_session_budgets')
    if (not isinstance(pools, dict) or set(pools) != {'agent', 'verifier_train', 'verifier_eval'}
            or any(type(n) is not int or n < 0 for n in pools.values())
            or sum(pools.values()) != audit['max_ssh_sessions']):
        raise RuntimeConfigurationError('explicit separate SSH pools must equal the audit allocation')
    _positive(value['max_evidence_bytes'], 'max_evidence_bytes')
    _validate_catalog(value['sidecars'], value['gpu_parent_resources'])
    return value


def phase_limits(value, mode, *, requested_gpu_seconds):
    """Validate an externally selected request; never derive a new task budget."""
    validate_projection(value)
    if mode not in {'train', 'eval'}:
        raise RuntimeConfigurationError('unknown Verifier phase')
    _positive(requested_gpu_seconds, 'requested GPU seconds')
    quota, audit = value['verifier_quota'], value['audit']
    timeout = quota['max_train_wall_seconds' if mode == 'train' else 'eval_timeout_seconds']
    if requested_gpu_seconds > min(timeout, quota['max_gpu_seconds']):
        raise RuntimeConfigurationError('requested GPU lifetime exceeds declared phase or total quota')
    sessions = audit['ssh_session_budgets']['verifier_' + mode]
    _positive(sessions, 'Verifier phase SSH sessions')
    return {'timeout_seconds': timeout, 'gpu_seconds': requested_gpu_seconds,
            'ssh_sessions': sessions, 'ssh_bytes_per_direction': audit['max_ssh_bytes_per_direction'],
            'max_evidence_bytes': value['max_evidence_bytes']}
