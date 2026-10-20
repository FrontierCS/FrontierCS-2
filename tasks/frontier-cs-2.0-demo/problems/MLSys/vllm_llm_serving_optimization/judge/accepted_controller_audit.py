"""Post-acceptance operator audit; never a live verifier or scientific writer.

Core401 exports are authoritative only across the separately verified OS/exec
boundary. Private file modes and self-signed JSON alone do not establish it.
"""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile

from controller_audit_plan import (GROUP_SCHEMA, PHASE_SCHEMA, VARIANTS,
                                   controller_directory, original_slots)
from controller_envelope import Context, _manifest_value, slots_digest, verify_terminal_bytes

BINDING_SCHEMA = 'frontier.accepted-controller-export-binding/v1'
BINDING_SOURCE = 'current_accepted_original_selected_final'
BINDING_FIELDS = frozenset({'schema_version', 'source', 'run_id', 'attempt_id', 'result_id',
    'receipt_sha256', 'task_version_id', 'task_config_digest', 'cpu_verifier_image_digest',
    'outer_artifact', 'selected_final', 'nested_artifact'})
LIMIT_FIELDS = frozenset({'maximum_binding_bytes', 'maximum_archive_bytes',
    'maximum_expanded_bytes', 'maximum_plan_bytes', 'maximum_manifest_bytes',
    'maximum_terminal_bytes', 'maximum_members'})


def unique_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate operator audit JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite audit JSON')))


def _sha(value):
    if (not isinstance(value, str) or len(value) != 71 or not value.startswith('sha256:')
            or any(c not in '0123456789abcdef' for c in value[7:])):
        raise ValueError('exact SHA256 audit identity required')
    return value


def _fields(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError('unknown or missing operator audit fields')


def _positive(value):
    if type(value) is not int or value <= 0:
        raise ValueError('explicit positive audit allocation required')
    return value


def validate_binding(value):
    _fields(value, BINDING_FIELDS)
    if value['schema_version'] != BINDING_SCHEMA or value['source'] != BINDING_SOURCE:
        raise ValueError('current original accepted selected-final source required')
    for key in ('run_id', 'attempt_id', 'result_id', 'task_version_id'):
        if not isinstance(value[key], str) or not 0 < len(value[key].encode()) <= 256:
            raise ValueError('bounded immutable accepted identity required')
    for key in ('receipt_sha256', 'task_config_digest', 'cpu_verifier_image_digest'):
        _sha(value[key])
    outer, final, nested = (value[key] for key in ('outer_artifact', 'selected_final', 'nested_artifact'))
    _fields(outer, {'bucket', 'key', 'version_id', 'sha256', 'size_bytes'})
    _fields(final, {'payload_id', 'payload_sha256', 'verification_id', 'payload_size_bytes'})
    _fields(nested, {'member_path', 'sha256', 'size_bytes'})
    for item in (outer, nested):
        _sha(item['sha256'])
        _positive(item['size_bytes'])
    _sha(final['payload_sha256'])
    _positive(final['payload_size_bytes'])
    for item in (final['payload_id'], final['verification_id'], outer['bucket'], outer['key']):
        if not isinstance(item, str) or not 0 < len(item.encode()) <= 4096 or '\x00' in item:
            raise ValueError('bounded accepted artifact identity required')
    if outer['version_id'] is not None and not isinstance(outer['version_id'], str):
        raise ValueError('invalid outer artifact version')
    if nested['member_path'] != f"verifications/{final['verification_id']}/evidence.tar":
        raise ValueError('nested source differs from the accepted selected final')
    return value


def _private_directory(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('absolute private operator export path required')
    descriptor = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022:
            raise ValueError('operator export directory must be privately owned')
        parent = os.open('..', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
        try:
            info = os.fstat(parent)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022:
                raise ValueError('operator export parent must be privately owned')
        finally:
            os.close(parent)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _private_file(directory, name):
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o077 or info.st_nlink != 1):
            raise ValueError('private regular operator export file required')
        return os.fdopen(descriptor, 'rb')
    except BaseException:
        os.close(descriptor)
        raise


def _members(snapshot, selectors, limits, *, closed_controllers=False):
    """Select source-derived paths from already snapshotted accepted bytes."""
    import zstandard
    snapshot.seek(0)
    found, names = {}, set()
    with zstandard.ZstdDecompressor().stream_reader(snapshot, closefd=False) as expanded:
        class Bounded:
            count = 0
            def read(self, size=-1):
                remain = limits['maximum_expanded_bytes'] - self.count
                value = expanded.read(min(65536 if size < 0 else size, remain + 1))
                self.count += len(value)
                if self.count > limits['maximum_expanded_bytes']:
                    raise ValueError('expanded accepted artifact exceeds allocation')
                return value
        bounded = Bounded()
        with tarfile.open(fileobj=bounded, mode='r|') as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if (member.name in names or not member.name or path.is_absolute()
                        or '..' in path.parts or str(path) != member.name
                        or '\\' in member.name or '\x00' in member.name
                        or not member.isfile() or member.issparse()):
                    raise ValueError('unsafe, duplicate or nonregular accepted member')
                names.add(member.name)
                if (closed_controllers and '/cpu-controller/' in member.name
                        and (member.name.endswith('/manifest.json') or member.name.endswith('.terminal.json'))
                        and member.name not in selectors):
                    raise ValueError('accepted artifact contains unplanned controller ownership')
                if len(names) > limits['maximum_members']:
                    raise ValueError('accepted artifact member count exceeds allocation')
                if member.name in selectors:
                    maximum = selectors[member.name]
                    if not 0 < member.size <= maximum:
                        raise ValueError('accepted audit member exceeds allocation')
                    with archive.extractfile(member) as stream:
                        value = stream.read(maximum + 1)
                    if len(value) != member.size:
                        raise ValueError('truncated accepted audit member')
                    found[member.name] = value
        while bounded.read(65536):
            pass
    if set(found) != set(selectors):
        raise ValueError('accepted artifact lacks planned CPU audit members')
    return found


def consume_export(directory, *, expected_binding, expected_content, expected_sources, limits,
                   expected_variants=VARIANTS, expected_fixed_targets_sha256=None):
    """Caller verifies operator authority and independent release/run pins first.

    This audit returns signature/coverage evidence or an unscored blocker. It
    never authorizes a score, retry, scientific result write or remote execution.
    """
    from sidecar_catalog import validate_variants, CANDIDATE_ONLY
    expected_variants = validate_variants(expected_variants)
    if expected_variants == CANDIDATE_ONLY:
        if (not isinstance(expected_fixed_targets_sha256, str) or len(expected_fixed_targets_sha256) != 64
                or any(c not in '0123456789abcdef' for c in expected_fixed_targets_sha256)):
            raise ValueError('operator must independently pin fixed-target SHA')
    elif expected_fixed_targets_sha256 is not None:
        raise ValueError('legacy evidence cannot carry a fixed-target release pin')
    validate_binding(expected_binding)
    _fields(limits, LIMIT_FIELDS)
    for value in limits.values():
        _positive(value)
    _fields(expected_content, {'source_content_digest', 'contract_digest'})
    _fields(expected_sources, expected_variants)
    for value in (*expected_content.values(), *expected_sources.values()):
        _sha(value)
    fd = _private_directory(directory)
    try:
        with _private_file(fd, 'binding.json') as stream:
            raw = stream.read(limits['maximum_binding_bytes'] + 1)
        if len(raw) > limits['maximum_binding_bytes']:
            raise ValueError('accepted binding exceeds allocation')
        binding = validate_binding(unique_json(raw))
        if binding != expected_binding:
            raise ValueError('export differs from independent operator accepted binding')
        nested = binding['nested_artifact']
        if nested['size_bytes'] > limits['maximum_archive_bytes']:
            raise ValueError('accepted nested artifact exceeds allocation')
        with _private_file(fd, 'evidence.tar') as stream, tempfile.TemporaryFile() as snapshot:
            hashed, count = hashlib.sha256(), 0
            while True:
                part = stream.read(min(65536, nested['size_bytes'] + 1 - count))
                if not part:
                    break
                count += len(part)
                if count > nested['size_bytes']:
                    raise ValueError('export exceeds accepted nested size')
                hashed.update(part)
                snapshot.write(part)
            if count != nested['size_bytes'] or 'sha256:' + hashed.hexdigest() != nested['sha256']:
                raise ValueError('export differs from original accepted nested bytes')
            return _consume_snapshot(snapshot, binding, expected_content, expected_sources, limits,
                                     expected_variants, expected_fixed_targets_sha256)
    finally:
        os.close(fd)


def allocation_exceeds_owner(start, end, cap):
    # Match the producer's deadline arithmetic without subtractive roundoff.
    return start + cap > end


def _consume_snapshot(snapshot, binding, content, sources, limits, variants=VARIANTS, fixed_sha=None):
    fixed_binding = {'fixed_targets_sha256': fixed_sha} if fixed_sha is not None else {}
    from instance_manifest import expected_groups
    groups = expected_groups('eval')
    identity = {'run_id': binding['run_id'], 'attempt_id': binding['attempt_id'],
                'verification_id': binding['selected_final']['verification_id'],
                'task_config_digest': binding['task_config_digest'],
                'executable_image_digest': binding['cpu_verifier_image_digest']}
    master = 'phase/controller-audit-plan.json'
    selectors = {master: limits['maximum_plan_bytes'], **{
        f"phase/{group.replace('/', '-')}/controller-audit-group.json": limits['maximum_plan_bytes']
        for group in groups}}
    plans = _members(snapshot, selectors, limits)
    phase = unique_json(plans[master])
    if phase != {'schema': PHASE_SCHEMA, 'mode': 'eval', 'identity': identity,
                  'content': content, 'groups': {g: list(ids) for g, ids in groups.items()},
                  'variants': list(variants), **fixed_binding}:
        raise ValueError('CPU audit membership differs from the original reviewed phase')
    controllers, blockers, ownership = [], [], {}
    for group, ids in groups.items():
        prefix = f"phase/{group.replace('/', '-')}"
        plan = unique_json(plans[prefix + '/controller-audit-group.json'])
        _fields(plan, {'schema', 'group', 'mode', 'identity', 'content', 'instances', 'variants',
                       'state', 'operation_id', 'controllers', 'blocker', *fixed_binding})
        for name, expected in {'schema': GROUP_SCHEMA, 'group': group, 'mode': 'eval',
                'identity': identity, 'content': content, 'instances': list(ids),
                'variants': list(variants), **fixed_binding}.items():
            if plan[name] != expected:
                raise ValueError('CPU group plan identity/membership mismatch')
        if plan['state'] != 'complete':
            if plan['state'] not in {'provisioning', 'allocated', 'blocked'}:
                raise ValueError('unknown CPU group lifecycle state')
            blocker = plan['blocker']
            if plan['state'] == 'blocked':
                _fields(blocker, {'stage', 'error_type', 'disposition'})
                if (blocker['stage'] not in {'preallocation', 'execution'}
                        or blocker['disposition'] != 'unscored_trusted_blocker'
                        or not isinstance(blocker['error_type'], str) or not blocker['error_type']):
                    raise ValueError('invalid trusted CPU blocker')
            elif blocker is not None:
                raise ValueError('unfinished CPU plan cannot invent a blocker')
            blockers.append({'group': group, 'state': plan['state'], 'blocker': blocker})
            continue
        if plan['blocker'] is not None or not isinstance(plan['operation_id'], str) or not plan['operation_id']:
            raise ValueError('complete CPU group lacks its actual allocation')
        if not isinstance(plan['controllers'], list) or len(plan['controllers']) > len(variants) * 5:
            raise ValueError('CPU controller plan exceeds original phase layout')
        seen = set()
        for entry in plan['controllers']:
            _fields(entry, {'kind', 'variant', 'source_digest', 'request_digest', 'context',
                            'started_monotonic', 'deadline_monotonic'})
            start, end = entry['started_monotonic'], entry['deadline_monotonic']
            if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, end))
                    or start < 0 or end <= start):
                raise ValueError('CPU controller lacks its actual positive owned interval')
            key = entry['kind'], entry['variant']
            if key in seen or entry['source_digest'] != sources.get(entry['variant']):
                raise ValueError('duplicate or foreign CPU controller ownership')
            seen.add(key)
            _sha(entry['request_digest'])
            slots = original_slots(identity, 'eval', group, entry['variant'], entry['kind'],
                plan['operation_id'], entry['source_digest'], entry['request_digest'])
            context = Context(**identity, **content, expected_slots_digest=slots_digest(slots))
            if entry['context'] != asdict(context):
                raise ValueError('CPU controller context differs from original slots')
            path = prefix + '/' + controller_directory('eval', group, entry['variant'], entry['kind'])
            controllers.append((path, context, slots, group, entry['kind'], entry['variant'], start, end))
        if not all(('startup', variant) in seen for variant in variants):
            raise ValueError('complete group omitted an original serving variant')
        ownership[group] = seen
    if blockers:
        return {'status': 'UNSCORED_TRUSTED_BLOCKER', 'groups': list(groups), 'blockers': blockers,
                'accepted_receipt_sha256': binding['receipt_sha256'], 'scientific_write': False}
    selectors = {}
    for path, _, slots, _, _, _, _, _ in controllers:
        selectors[path + '/manifest.json'] = limits['maximum_manifest_bytes']
        for slot in slots:
            name = hashlib.sha256(slot.key().encode()).hexdigest() + '.terminal.json'
            selectors[path + '/' + name] = limits['maximum_terminal_bytes']
    files = _members(snapshot, selectors, limits, closed_controllers=True)
    statuses, runtime_failures = Counter(), set()
    for path, context, slots, group, kind, variant, start, end in controllers:
        manifest = _manifest_value(files[path + '/manifest.json'], context, limits['maximum_manifest_bytes'])
        if manifest['expected_slots'] != [asdict(slot) for slot in slots]:
            raise ValueError('accepted manifest reordered or substituted the original CPU slots')
        for slot in slots:
            name = hashlib.sha256(slot.key().encode()).hexdigest() + '.terminal.json'
            record = verify_terminal_bytes(files[path + '/' + name], context=context, slots=slots,
                public_key=manifest['public_key'], maximum_bytes=limits['maximum_terminal_bytes'])
            if record['slot'] != asdict(slot):
                raise ValueError('terminal path does not name its exact original slot')
            if allocation_exceeds_owner(start, end, record['cap_seconds']):
                raise ValueError('terminal allocation exceeds its pre-input CPU owner interval')
            statuses[record['status']] += 1
            failure = record.get('failure_context', {})
            decision = failure.get('decision', {})
            if (record['status'] != 'complete'
                    and failure.get('boundary', {}).get('kind') == 'candidate_service'
                    and decision.get('attribution') == 'candidate'
                    and decision.get('disposition') == 'runtime_error'
                    and decision.get('affected_scope') == 'workload_group'
                    and decision.get('retry_eligible') is False):
                runtime_failures.add((group, variant, kind))
    for group, seen in ownership.items():
        fixed = {'startup', 'batch', 'group_control', 'shutdown'}
        if group.startswith('swe/'):
            fixed.add('correctness')
        for variant in variants:
            actual = {kind for kind, owner in seen if owner == variant}
            if not any(g == group and owner == variant for g, owner, _ in runtime_failures):
                if actual != fixed:
                    raise ValueError('complete healthy variant omitted an original controller')
            elif actual != {'startup'} or (group, variant, 'startup') not in runtime_failures:
                required = fixed - {'correctness'}
                if not required.issubset(actual):
                    raise ValueError('runtime failure omitted an owned lifecycle controller')
    return {'status': 'ACCEPTED_CONTROLLER_SIGNATURES_AND_COVERAGE_VERIFIED',
            'accepted_receipt_sha256': binding['receipt_sha256'], 'groups': list(groups),
            'controllers': len(controllers), 'terminal_counts': dict(statuses),
            'scientific_write': False}
