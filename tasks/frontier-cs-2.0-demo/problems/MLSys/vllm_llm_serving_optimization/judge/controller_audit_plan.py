"""CPU-owned audit layout, persisted before any remote operation is dispatched."""
from dataclasses import asdict
import os
from pathlib import Path
import tempfile

from controller_envelope import Context, slots_digest
from controller_identity import ENVIRONMENT, load_content_binding, load_limits
from instance_manifest import expected_groups
from terminal_contract import Slot, canonical

PHASE_SCHEMA = 'frontier.vllm-controller-audit-phase/v1'
GROUP_SCHEMA = 'frontier.vllm-controller-audit-group/v1'
VARIANTS = ('vanilla', 'reference', 'candidate')
KINDS = ('startup', 'batch', 'group_control', 'shutdown', 'correctness')


def durable_json(path, value, *, create=False):
    path = Path(path)
    if create and path.exists():
        raise ValueError('CPU audit ownership cannot be replaced')
    raw = canonical(value)
    descriptor, temporary = tempfile.mkstemp(prefix='.audit-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if create:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def controller_directory(mode, group, variant, kind):
    groups = expected_groups(mode)
    if group not in groups or variant not in VARIANTS or kind not in KINDS:
        raise ValueError('controller selector is outside the original phase layout')
    if kind == 'correctness' and not group.startswith('swe/'):
        raise ValueError('BFCL has no separate correctness controller')
    prefix = f"phase/group-{tuple(groups).index(group):02d}"
    if kind in {'startup', 'shutdown'}:
        action = 'start' if kind == 'startup' else 'stop'
        return f'{prefix}/{variant}/parent-{action}-ownership/cpu-controller'
    suffix = '/group-control' if kind == 'group_control' else ''
    operation = 'batch' if kind == 'group_control' else kind
    return f'{prefix}/relay/{variant}-{operation}{suffix}/cpu-controller'


def original_slots(identity, mode, group, variant, kind, operation_id,
                   source_digest, request_digest):
    controller_directory(mode, group, variant, kind)
    name = group if kind == 'batch' else group + '/' + kind
    return tuple(Slot(identity['run_id'], identity['attempt_id'], identity['verification_id'],
        operation_id, mode, name, variant, instance, source_digest, request_digest)
        for instance in expected_groups(mode)[group])


class PhaseAuditPlan:
    def __init__(self, evidence, mode, *, environment, source_root,
                 variants=VARIANTS, fixed_targets_sha256=None):
        from sidecar_catalog import validate_variants, CANDIDATE_ONLY
        self.variants = validate_variants(variants)
        self.fixed_binding = {}
        if variants == CANDIDATE_ONLY:
            if (not isinstance(fixed_targets_sha256, str) or len(fixed_targets_sha256) != 64
                    or any(c not in '0123456789abcdef' for c in fixed_targets_sha256)):
                raise ValueError('candidate-only audit requires trusted fixed-target SHA')
            self.fixed_binding = {'fixed_targets_sha256': fixed_targets_sha256}
        elif fixed_targets_sha256 is not None:
            raise ValueError('legacy audit cannot masquerade as fixed-target evaluation')
        self.evidence, self.mode = Path(evidence), mode
        self.identity = {key: environment[name] for key, name in ENVIRONMENT.items()}
        limits = load_limits(Path(source_root) / 'controller-contract-limits.json')
        self.content = load_content_binding(Path(source_root) / 'controller-contract.json',
                                           source_root, **limits)
        self.groups = expected_groups(mode)
        durable_json(self.evidence / 'controller-audit-plan.json', {
            'schema': PHASE_SCHEMA, 'mode': mode, 'identity': self.identity,
            'content': self.content, 'groups': {group: list(ids) for group, ids in self.groups.items()},
            'variants': list(self.variants), **self.fixed_binding}, create=True)

    def group(self, group, evidence):
        if group not in self.groups or Path(evidence) != self.evidence / group.replace('/', '-'):
            raise ValueError('CPU group audit location differs from original phase')
        return GroupAuditPlan(self, group, evidence)


class GroupAuditPlan:
    def __init__(self, phase, group, evidence):
        self.phase, self.group, self.evidence = phase, group, Path(evidence)
        self.path = self.evidence / 'controller-audit-group.json'
        self.value = {'schema': GROUP_SCHEMA, 'group': group, 'mode': phase.mode,
                      'identity': phase.identity, 'content': phase.content,
                      'instances': list(phase.groups[group]), 'variants': list(phase.variants),
                      **phase.fixed_binding,
                      'state': 'provisioning', 'operation_id': None,
                      'controllers': [], 'blocker': None}
        durable_json(self.path, self.value, create=True)

    def allocated(self, operation_id):
        if self.value['state'] != 'provisioning' or not isinstance(operation_id, str) or not operation_id:
            raise ValueError('CPU group may bind one actual allocated operation only')
        self.value['operation_id'] = operation_id
        self.value['state'] = 'allocated'
        durable_json(self.path, self.value)

    def register(self, kind, variant, request_digest, source_digest, controller):
        if variant not in self.phase.variants:
            raise ValueError('controller variant is excluded by immutable phase policy')
        if self.value['state'] != 'allocated':
            raise ValueError('controller requires a real CPU-owned group allocation')
        expected = original_slots(self.phase.identity, self.phase.mode, self.group, variant, kind,
            self.value['operation_id'], source_digest, request_digest)
        relative = controller_directory(self.phase.mode, self.group, variant, kind)
        if controller.slots != expected or controller.evidence != self.evidence / relative:
            raise ValueError('actual controller differs from source-derived audit ownership')
        context = Context(**self.phase.identity, **self.phase.content,
                          expected_slots_digest=slots_digest(expected))
        if controller.signer.context != context:
            raise ValueError('controller content/identity differs from pre-input CPU plan')
        entry = {'kind': kind, 'variant': variant, 'source_digest': source_digest,
                 'request_digest': request_digest, 'context': asdict(context),
                 'started_monotonic': controller.started, 'deadline_monotonic': controller.deadline}
        if any(e['kind'] == kind and e['variant'] == variant for e in self.value['controllers']):
            raise ValueError('CPU controller audit ownership cannot be registered twice')
        self.value['controllers'].append(entry)
        durable_json(self.path, self.value)

    def complete(self):
        if self.value['state'] != 'allocated':
            raise ValueError('only an allocated CPU group may complete')
        self.value['state'] = 'complete'
        durable_json(self.path, self.value)

    def blocked(self, error):
        self.value['blocker'] = {'stage': 'preallocation' if self.value['state'] == 'provisioning'
                                 else 'execution', 'error_type': type(error).__name__,
                                 'disposition': 'unscored_trusted_blocker'}
        self.value['state'] = 'blocked'
        durable_json(self.path, self.value)
