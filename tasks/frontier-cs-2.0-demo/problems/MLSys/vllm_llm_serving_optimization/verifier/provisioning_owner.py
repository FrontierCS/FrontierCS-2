"""CPU precreate ownership; only the authenticated Driver client consumes proofs."""
from dataclasses import dataclass
import re

from controller_envelope import digest
from controller_identity import ENVIRONMENT, load_content_binding, load_limits
from instance_manifest import expected_groups
from terminal_contract import canonical

CLAIM_SCHEMA = 'frontier.verifier-provisioning-claim/v1'
FAILURE_SCHEMA = 'frontier.verifier-platform-failure/v1'
_DIGEST = re.compile(r'^sha256:[0-9a-f]{64}$')
_ULID = r'[0-7][0-9A-HJKMNP-TV-Z]{25}'
_PRODUCERS = {'gpu_create_failed': 'fixed_gpu_create_capacity',
              'sandbox_create_failed': 'fixed_sandbox_create_capacity',
              'sidecar_unavailable': 'declared_sidecar_create_capacity'}


def _id(value, prefix):
    return isinstance(value, str) and re.fullmatch(prefix + _ULID, value) is not None


def _hash(value):
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


@dataclass(frozen=True)
class AuthenticatedDriverFailure:
    """Constructed only by the CPU owner's authenticated-response consumption."""
    operation_id: str
    cause: str
    encoded: bytes


class ProvisioningOwner:
    def __init__(self, environment, plan, slots, request, key):
        self.identity = {field: environment[name] for field, name in ENVIRONMENT.items()}
        for name, prefix in (('run_id', 'run_'), ('attempt_id', 'attempt_'),
                             ('verification_id', 'verification_')):
            if not _id(self.identity[name], prefix):
                raise ValueError('strict Driver-injected provisioning identity required')
        if any(not _hash(self.identity[name]) for name in
               ('task_config_digest', 'executable_image_digest')):
            raise ValueError('exact frozen Driver digest required')
        if (not isinstance(key, str) or not key or len(key.encode()) > 256
                or set(request) != {'gpu_type', 'gpu_count', 'max_duration_seconds'}
                or request['gpu_type'] != 'H100' or type(request['gpu_count']) is not int
                or request['gpu_count'] != 1 or type(request['max_duration_seconds']) is not int
                or request['max_duration_seconds'] <= 0 or not slots):
            raise ValueError('exact owned original provisioning request required')
        self.plan_bytes, self.slot_bytes = canonical(plan), canonical(slots)
        self.request_bytes = canonical([request['gpu_type'], request['gpu_count'], request['max_duration_seconds']])
        self.key = key
        self._claim_bytes = canonical({'schema_version': CLAIM_SCHEMA,
            'create_request_digest': digest(self.request_bytes),
            'create_idempotency_digest': digest(key.encode()),
            'frozen_group_variant_plan_digest': digest(self.plan_bytes),
            'expected_provisioning_slot_set_digest': digest(self.slot_bytes)})
        self.operation_id = None
        self._terminal = None

    @property
    def claim(self):
        import json
        return json.loads(self._claim_bytes)

    def check_request(self, request, key, verification_id):
        if (key != self.key or verification_id != self.identity['verification_id']
                or canonical([request.get('gpu_type'), request.get('gpu_count'),
                              request.get('max_duration_seconds')]) != self.request_bytes
                or set(request) != {'gpu_type', 'gpu_count', 'max_duration_seconds'}):
            raise ValueError('provisioning owner differs from authenticated request')

    def _bind(self, operation_id, terminal):
        if not _id(operation_id, 'gpurunner_'):
            raise ValueError('Driver did not supply its actual typed GPU operation')
        if self.operation_id is not None and (self.operation_id != operation_id or self._terminal != terminal):
            raise ValueError('provisioning first terminal is single assignment')
        self.operation_id, self._terminal = operation_id, terminal

    def accept_success(self, value):
        if value.get('controller_binding') != self.claim:
            raise ValueError('successful Driver create did not echo its exact CPU claim')
        self._bind(value.get('runner_id'), ('success', canonical(value)))

    def consume_failure(self, value, *, status, code):
        # Defense in depth against a producer accidentally attaching proof after
        # compensation failed: only the compensated outer error is admissible.
        if status != 502 or code != 'gpu_backend_unavailable':
            raise ValueError('platform proof cannot override cleanup or control failure')
        fields = {'schema_version', 'run_id', 'attempt_id', 'verification_id',
            'task_config_digest', 'verifier_image_digest', 'operation_id',
            'controller_binding', 'cause', 'proof', 'candidate_independent', 'input_forwarded_bytes'}
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError('closed Driver platform failure required')
        expected = dict(self.identity)
        expected['verifier_image_digest'] = expected.pop('executable_image_digest')
        if (value['schema_version'] != FAILURE_SCHEMA
                or any(value[name] != item for name, item in expected.items())
                or value['controller_binding'] != self.claim
                or value['candidate_independent'] is not True
                or type(value['input_forwarded_bytes']) is not int
                or value['input_forwarded_bytes'] != 0):
            raise ValueError('Driver platform failure differs from frozen CPU ownership')
        proof = value['proof']
        producer = _PRODUCERS.get(value['cause']) if isinstance(value['cause'], str) else None
        fields = {'producer', 'profile_digest'} if value['cause'] == 'sidecar_unavailable' else {'producer'}
        if (producer is None or not isinstance(proof, dict) or set(proof) != fields
                or proof.get('producer') != producer
                or ('profile_digest' in proof and not _hash(proof['profile_digest']))):
            raise ValueError('unknown or mismatched Driver capacity producer')
        # Core authenticates sidecar profile membership against its own frozen
        # phase-selected dispatch. Child JSON has no route to this method.
        encoded = canonical(value)
        self._bind(value['operation_id'], ('failure', encoded))
        return AuthenticatedDriverFailure(value['operation_id'], value['cause'], encoded)


def for_phase(environment, config, phase_plan, mode, selected_groups, candidate_patch,
              source_root, request, key):
    from pathlib import Path
    from phase_run import source_digest
    from runtime_config import execution_variants
    execution = execution_variants(config)
    source_root = Path(source_root)
    limits = load_limits(source_root / 'controller-contract-limits.json')
    binding = load_content_binding(source_root / 'controller-contract.json', source_root, **limits)
    expected = expected_groups(mode)
    selected = tuple(selected_groups)
    if not selected or len(set(selected)) != len(selected) or any(g not in expected for g in selected):
        raise ValueError('provisioning must own original phase groups')
    patches = {'candidate': candidate_patch}
    if execution != ('candidate',):
        patches.update(vanilla=b'', reference=(source_root/'judge/reference.patch').read_bytes())
    variants = {variant: source_digest(patches[variant]) for variant in execution}
    plan = {'schema': 'frontier.vllm-provisioning-plan/v1', 'mode': mode,
        'groups': {group: list(expected[group]) for group in selected},
        'variants': variants, 'content_binding': binding,
        'runtime_config_digest': digest(canonical(config)), 'phase_plan_digest': digest(canonical(phase_plan))}
    slots = [{'mode': mode, 'group': group, 'variant': variant, 'instance_id': ident,
              'model_source_digest': variants[variant]}
             for group in selected for variant in execution for ident in expected[group]]
    return ProvisioningOwner(environment, plan, slots, request, key)
