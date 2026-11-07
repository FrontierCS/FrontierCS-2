"""CPU-frozen watchdog bounds for model cases without measured conversation time."""
import base64
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from controller_envelope import _bounded, _key_id, _public, digest, slots_digest
from remote_terminal import _retain_prefix
from terminal_contract import canonical

SCHEMA = 'frontier.vllm-cpu-timing-allocation/v1'
DOMAIN = b'frontier-vllm-cpu-timing-allocation-v1\n'
SOURCE = 'verifier/runtime-plan.json'


def _positive(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class TimingAllocation:
    envelope: dict
    context: object
    slots: tuple
    public_key: str
    maximum_bytes: int

    def validate(self):
        value = self.envelope
        _bounded(value, self.maximum_bytes)
        if (set(value) != {'payload', 'signature'} or not isinstance(value['payload'], dict)):
            raise ValueError('invalid CPU timing allocation envelope')
        payload = value['payload']
        if (set(payload) != {'schema', 'context', 'owner_seconds', 'started_monotonic', 'deadline_monotonic', 'configured_limit_seconds',
                'source_path', 'source_sha256', 'config_pointer', 'key_id'}
                or payload['schema'] != SCHEMA or payload['context'] != asdict(self.context)
                or slots_digest(self.slots) != self.context.expected_slots_digest
                or payload['source_path'] != SOURCE
                or payload['config_pointer'] != 'relay.batch_timeout_seconds.' + self.slots[0].phase
                or payload['key_id'] != _key_id(self.public_key)
                or not _positive(payload['started_monotonic'])
                or not _positive(payload['deadline_monotonic'])
                or payload['deadline_monotonic'] - payload['started_monotonic'] != payload['owner_seconds']
                or not _positive(payload['owner_seconds'])
                or not _positive(payload['configured_limit_seconds'])
                or payload['owner_seconds'] > payload['configured_limit_seconds']):
            raise ValueError('CPU timing allocation differs from its frozen source and ownership')
        _public(self.public_key).verify(base64.b64decode(value['signature'], validate=True),
                                        DOMAIN + canonical(payload))
        return payload

    def for_slot(self, slot):
        self.validate()
        if slot not in self.slots or '/correctness' in slot.group or not slot.group.startswith('swe/'):
            raise ValueError('censored timing requires an original owned timed SWE slot')
        return CensoredTiming(self, slot)


@dataclass(frozen=True)
class CensoredTiming:
    allocation: TimingAllocation
    slot: object

    @property
    def bound_seconds(self):
        return self.allocation.validate()['owner_seconds']

    def validate(self, *, instance_id, variant, group=None):
        self.allocation.for_slot(self.slot)
        if (self.slot.instance_id != instance_id or self.slot.variant != variant
                or (group is not None and self.slot.group != group)):
            raise ValueError('censored timing belongs to another original case or variant')
        return self.bound_seconds

    def provenance(self):
        payload = self.allocation.validate()
        return {'kind': 'upper_bound', 'bound_seconds': payload['owner_seconds'],
            'basis': 'pre_input_cpu_owner_watchdog', 'measured': False,
            'source_path': payload['source_path'], 'source_sha256': payload['source_sha256'],
            'config_pointer': payload['config_pointer'],
            'configured_limit_seconds': payload['configured_limit_seconds'],
            'allocation_sha256': digest(canonical(self.allocation.envelope)),
            'controller_context': payload['context'], 'slot': asdict(self.slot)}


def freeze(controller, *, source_root, contract_path, maximum_source_bytes, maximum_contract_bytes):
    """Persist before job dispatch, using only image-local inputs and the CPU key.

    The full owner interval conservatively covers every original case. Receiving
    remote dispatch progress later must not shorten this bound by network delay.
    """
    root = Path(source_root)
    with (root / SOURCE).open('rb') as stream:
        raw = stream.read(maximum_source_bytes + 1)
    with Path(contract_path).open('rb') as stream:
        contract_raw = stream.read(maximum_contract_bytes + 1)
    if len(raw) > maximum_source_bytes or len(contract_raw) > maximum_contract_bytes:
        raise ValueError('CPU timing source exceeds its image allocation')
    contract = json.loads(contract_raw)
    if (digest(canonical(contract)) != controller.signer.context.contract_digest
            or contract['files'].get(SOURCE) != digest(raw)):
        raise ValueError('CPU timing plan differs from the immutable image source contract')
    phase = controller.slots[0].phase
    configured = json.loads(raw)['relay']['batch_timeout_seconds'][phase]
    owner = controller.deadline - controller.started
    if not _positive(configured) or not _positive(owner) or owner > configured:
        raise ValueError('CPU timing owner exceeds the existing configured watchdog')
    payload = {'schema': SCHEMA, 'context': asdict(controller.signer.context),
        'owner_seconds': owner, 'started_monotonic': controller.started,
        'deadline_monotonic': controller.deadline, 'configured_limit_seconds': configured,
        'source_path': SOURCE, 'source_sha256': digest(raw),
        'config_pointer': 'relay.batch_timeout_seconds.' + phase,
        'key_id': _key_id(controller.signer.public_key)}
    signature = controller.signer.private_key.sign(DOMAIN + canonical(payload))
    envelope = {'payload': payload, 'signature': base64.b64encode(signature).decode()}
    allocation = TimingAllocation(envelope, controller.signer.context, controller.slots,
        controller.signer.public_key, controller.signer.maximum_manifest_bytes)
    allocation.validate()
    _retain_prefix(controller.evidence / 'timing-allocation.json',
                   _bounded(envelope, controller.signer.maximum_manifest_bytes))
    return allocation


def owned_case_failure_seconds(row, *, bound, group, variant):
    """Bind authenticated remote timing to the original CPU allocation and group."""
    from supervised_case_timing import measured_case_failure_seconds
    if (not isinstance(bound, CensoredTiming)
            or bound.slot.operation_id != group.operation_id
            or bound.slot.phase != group.mode
            or bound.slot.source_digest != group.model_source_digest):
        raise ValueError('supervised timing lacks the original CPU group allocation')
    bound.validate(instance_id=row['instance_id'], variant=variant,
                   group=group.workload + '/' + group.rate)
    return measured_case_failure_seconds(row, slot=bound.slot)
