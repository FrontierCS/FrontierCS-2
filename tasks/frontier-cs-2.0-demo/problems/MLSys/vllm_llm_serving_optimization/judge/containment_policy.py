"""Local attribution draft; platform adapter authentication is not wired here."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Boundary:
    kind: str
    stage: str
    affected_domain: str | None = None

    def __post_init__(self):
        stages = {
            'candidate_service': {'startup', 'health', 'request', 'response', 'execution'},
            'swe_case': {'setup', 'tool', 'patch', 'correctness', 'execution'},
            'platform': {'provisioning', 'control', 'sidecar_preflight'},
        }
        if self.kind not in stages or self.stage not in stages[self.kind]:
            raise ValueError('unknown execution boundary')
        domain = self.affected_domain or (self.kind if self.kind != 'platform' else None)
        if domain not in {'candidate_service', 'swe_case'} or (self.kind != 'platform' and domain != self.kind):
            raise ValueError('explicit affected domain required for platform adapters')
        object.__setattr__(self, 'affected_domain', domain)


@dataclass(frozen=True)
class LocalPlatformSignal:
    """Normalized trusted-adapter test input, not authenticated accepted evidence.

    Production must construct this outside child data after verifying the Driver
    signal, binding and candidate independence. Python typing alone proves none
    of those properties; this draft cannot authorize a live retry.
    """
    run_id: str
    attempt_id: str
    operation_id: str
    code: str
    channel: str
    candidate_independent: bool
    preinput_never_forwarded: bool = False


_PLATFORM = {
    ('gpu_create_failed', 'driver_gpu_provisioning'): 'provisioning',
    ('sandbox_create_failed', 'driver_sandbox_provisioning'): 'provisioning',
    ('driver_transport_failed', 'driver_control'): 'control',
    ('network_control_failed', 'driver_control'): 'control',
    ('sidecar_unavailable', 'sidecar_preflight'): 'sidecar_preflight',
}


def classify_failure(slot, boundary, *, candidate_input_dispatched, platform_signal=None,
                     dispatch_state=None):
    if not isinstance(boundary, Boundary) or type(candidate_input_dispatched) is not bool:
        raise ValueError('controller boundary and durable input state required')
    signal = platform_signal
    state = ('observed_dispatched' if candidate_input_dispatched else 'unknown') if dispatch_state is None else dispatch_state
    if state not in {'never_forwarded', 'observed_dispatched', 'unknown'}:
        raise ValueError('unknown input dispatch state')
    if candidate_input_dispatched:
        state = 'observed_dispatched'
    matched = (type(signal) is LocalPlatformSignal and
               (signal.run_id, signal.attempt_id, signal.operation_id) ==
               (slot.run_id, slot.attempt_id, slot.operation_id) and
               signal.candidate_independent is True and
               _PLATFORM.get((signal.code, signal.channel)) == boundary.stage)
    # A candidate engine that never becomes ready is still candidate execution.
    # Pre-input sidecar readiness is a different owned platform boundary.
    common = {'dispatch_state': state}
    if (boundary.kind == 'platform' and matched and
            (boundary.stage != 'sidecar_preflight' or
             (state == 'never_forwarded' and signal.preinput_never_forwarded is True))):
        return {'attribution': 'platform', 'disposition': 'infrastructure',
                'affected_scope': 'run', 'scientific_effect': 'unscored',
                'retry_eligible': True, 'platform_code': signal.code, **common}
    if boundary.affected_domain == 'swe_case':
        return {'attribution': 'candidate', 'disposition': 'case_fail',
                'affected_scope': 'case', 'scientific_effect': 'case_incorrect',
                'retry_eligible': False, 'platform_code': None, **common}
    return {'attribution': 'candidate', 'disposition': 'runtime_error',
            'affected_scope': 'workload_group',
            'scientific_effect': 'evaluation_fail_affected_scope_no_credit',
            'retry_eligible': False, 'platform_code': None, **common}


def child_failure_metadata(slot, boundary, candidate_input_dispatched):
    # Child processing has no platform-signal input or infrastructure branch.
    return {'boundary': asdict(boundary),
            'decision': classify_failure(slot, boundary,
                                        candidate_input_dispatched=candidate_input_dispatched)}
