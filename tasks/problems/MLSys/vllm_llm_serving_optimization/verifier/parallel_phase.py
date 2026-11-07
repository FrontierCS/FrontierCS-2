"""Independent GPU parents per workload group; join only complete results."""
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from calibration_clock import begin
from safe_error_frames import safe_error_frames
from controller_audit_plan import PhaseAuditPlan
from gpu_session import DriverGPU, GPUSession
from driver_client import (AuthenticatedProvisioningError, DriverCreateInfrastructureError,
                           DriverQuotaError, DriverWorkIntervalError)
from driver_client import SchedulerPreinputInfrastructureError
from instance_manifest import expected_groups
from phase_run import VARIANTS, run_phase
from provisioning_owner import for_phase
from runtime_config import canonical_digest, execution_variants
from cpu_reference_verdict import TrustedMeasurementError, evaluate_final, evaluate_train


class PhaseInfrastructureError(RuntimeError):
    """Only allowlisted CPU/Driver pre-input signals; no scientific score."""
    def __init__(self, failures):
        super().__init__('Owned platform provisioning could not complete')
        self.failures = failures
        self.cleanup_required = any(row.get('cleanup_required', True) for row in failures.values())


class PhaseQuotaError(RuntimeError):
    """Original task allocation exhausted: no score and no automatic retry."""
    def __init__(self, failures):
        super().__init__('Declared GPU resource allocation exhausted')
        self.failures = failures


def infrastructure_signal(error):
    # Exact types prevent unknown remote/child exceptions from gaining infra
    # status through a shared RuntimeError or SidecarExecutionError base class.
    if type(error) is DriverCreateInfrastructureError:
        return dict(error.signal)
    if type(error) is SchedulerPreinputInfrastructureError:
        return dict(error.signal)
    if type(error) is AuthenticatedProvisioningError:
        return {'schema': 'frontier.vllm-authenticated-provisioning-failure/v1',
                'cause': error.provenance.cause, 'operation_id': error.provenance.operation_id,
                'cleanup_required': False, 'candidate_input_forwarded': False,
                'proof_kind': 'driver_bound_compensated_provisioning_failure'}
    return None


def join_groups(mode, completed, *, fixed_targets=None):
    expected = set(expected_groups(mode))
    if not isinstance(completed, Mapping) or set(completed) != expected:
        raise TrustedMeasurementError(None, None, 'incomplete_group_join')
    variants = ('candidate',) if fixed_targets is not None else VARIANTS
    merged = {variant: {} for variant in variants}
    for group in expected_groups(mode):
        observations = completed[group]
        if not isinstance(observations, Mapping) or set(observations) != set(variants):
            raise TrustedMeasurementError(None, group, 'missing_serving_variant')
        for variant in variants:
            if not isinstance(observations[variant], Mapping) or set(observations[variant]) != {group}:
                raise TrustedMeasurementError(variant, group, 'misbound_group_observation')
            merged[variant][group] = observations[variant][group]
    if fixed_targets is not None:
        from fixed_targets import evaluate_fixed
        return evaluate_fixed(merged['candidate'], fixed_targets, mode=mode)
    return (evaluate_train if mode == 'train' else evaluate_final)(
        merged['vanilla'], merged['candidate'], merged['reference'])


def run_parallel_phase(config, plan, mode, *, environment, candidate_patch, root,
                       evidence, deadline, calibration=False, hidden_jps=None,
                       clock_directory=None, fixed_targets=None):
    """All parent sessions close independently before any result is scored."""
    groups = expected_groups(mode)
    variants = execution_variants(config)
    if (variants == ('candidate',)) != (fixed_targets is not None):
        raise TrustedMeasurementError(None, None, 'fixed_target_variant_binding_mismatch')
    fixed_sha = None
    if fixed_targets is not None:
        from fixed_targets import digest, validate_targets
        fixed_targets = validate_targets(fixed_targets, expected_environment=fixed_targets['environment'])
        fixed_sha = digest(fixed_targets)
    phase = plan['phases'][mode]
    evidence = Path(evidence)
    root = Path(root)
    completed = {}
    errors = {}
    trusted_errors = []
    error_details = {}
    infrastructure = {}
    evidence.mkdir(parents=True, exist_ok=True)
    audit = PhaseAuditPlan(evidence, mode, environment=environment,
                          source_root=Path(__file__).parents[1], variants=variants,
                          fixed_targets_sha256=fixed_sha)
    clocks = Path(clock_directory) if clock_directory is not None else root / 'parent-clocks'
    clocks.mkdir(parents=True, exist_ok=False)

    def execute_group(group):
        group_key = group.replace('/', '-')
        group_evidence = evidence / group_key
        group_evidence.mkdir(parents=True, exist_ok=False)
        group_audit = audit.group(group, group_evidence)
        try:
            observations = execute_owned_group(group, group_evidence, group_audit)
        except BaseException as error:
            group_audit.blocked(error)
            raise
        group_audit.complete()
        return observations

    def execute_owned_group(group, group_evidence, group_audit):
        group_key = group.replace('/', '-')
        parent_seconds = phase['groups'][group]['gpu_seconds']
        request_key = hashlib.sha256((environment['FRONTIER_VERIFICATION_ID'] + '\0' + mode + '\0' + group +
                                      '\0' + hashlib.sha256(candidate_patch).hexdigest()).encode()).hexdigest()
        clock = begin({'driver_origin': environment['FRONTIER_DRIVER_INTERNAL_URL'],
                       'verification_id': environment['FRONTIER_VERIFICATION_ID'],
                       'request_key': request_key,
                       'request': {'gpu_type': 'H100', 'gpu_count': 1,
                                   'max_duration_seconds': parent_seconds},
                       'config_sha256': canonical_digest(config),
                       'plan_sha256': canonical_digest(plan)},
                      clocks / (mode + '-' + group_key + '.json'), cap_seconds=parent_seconds,
                      cleanup_seconds=phase['cleanup_seconds'],
                      capture_seconds=phase['capture_seconds'])
        cutoffs = clock.initial_deadlines
        client = DriverGPU(environment['FRONTIER_DRIVER_INTERNAL_URL'],
                           environment['FRONTIER_VERIFICATION_ID'],
                           environment['FRONTIER_VERIFIER_TOKEN'],
                           deadline=min(deadline - phase['cleanup_seconds'] - phase['capture_seconds'],
                                        cutoffs['work_deadline']))
        owner = for_phase(environment, config, plan, mode, (group,), candidate_patch,
            Path(__file__).parents[1], {'gpu_type': 'H100', 'gpu_count': 1,
                                      'max_duration_seconds': parent_seconds}, request_key)
        with GPUSession(client, config, mode, gpu_seconds=parent_seconds,
                        request_key=request_key, scratch=root / ('ssh-' + group_key),
                        evidence=group_evidence / 'parent',
                        cleanup_deadline=min(deadline - phase['capture_seconds'],
                                             cutoffs['delete_deadline']),
                        calibration_clock=clock, provisioning_owner=owner) as session:
            group_audit.allocated(session.runner_id)
            observations = run_phase(session, config, plan, candidate_patch=candidate_patch,
                evidence=group_evidence / 'phase', calibration=calibration,
                hidden_jps=hidden_jps, selected_group=group,
                controller_environment=environment, controller_audit=group_audit,
                **{key: phase[key] for key in ('startup_seconds', 'stop_seconds',
                                              'max_stdout_bytes', 'max_stderr_bytes')})
        if time.monotonic() >= clock.deadlines['capture_deadline']:
            raise TimeoutError('group receipt capture exceeded its own grant')
        (group_evidence / 'parent-clock.json').write_bytes(clock.path.read_bytes())
        (group_evidence / 'parent-cutoffs.json').write_text(json.dumps(clock.deadlines,
            sort_keys=True, allow_nan=False) + '\n')
        return observations

    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = {pool.submit(execute_group, group): group for group in groups}
        for future in as_completed(futures):
            group = futures[future]
            try:
                completed[group] = future.result()
            except BaseException as error:
                errors[group] = type(error).__name__
                if isinstance(error, TrustedMeasurementError):
                    trusted_errors.append(error)
                signal = infrastructure_signal(error)
                if signal is not None:
                    infrastructure[group] = signal
                error_details[group] = {'phase': mode, 'group': group, 'variant': None,
                                        'frames': safe_error_frames(error),
                                        **(error.private_classification() if isinstance(error, TrustedMeasurementError) else {})}
                if type(error) in {DriverQuotaError, DriverWorkIntervalError}:
                    error_details[group]['quota_failure'] = error.quota
    if errors:
        try:
            (evidence / 'incomplete-groups.json').write_text(json.dumps({
                'failed_groups': errors, 'completed_groups': sorted(completed),
                'failure_details': error_details,
                'infrastructure_signals': infrastructure,
                'score_ready': False}, sort_keys=True) + '\n')
        except Exception:
            pass
        quotas = {group: detail['quota_failure'] for group, detail in error_details.items()
                  if 'quota_failure' in detail}
        if quotas:
            raise PhaseQuotaError(quotas)
        if trusted_errors:
            raise trusted_errors[0]
        if set(infrastructure) == set(errors):
            raise PhaseInfrastructureError(infrastructure)
        raise RuntimeError('independent workload parent failed; all parents were closed')
    result = join_groups(mode, completed, fixed_targets=fixed_targets)
    (evidence / 'operator-result.json').write_text(json.dumps({
        'mode': mode, 'groups': list(groups), 'calibration_only': calibration,
        'selected_patch_sha256': 'sha256:' + hashlib.sha256(candidate_patch).hexdigest(),
        'result': result, 'scope': 'operator-only complete independently joined phase',
    }, indent=2) + '\n')
    return result
