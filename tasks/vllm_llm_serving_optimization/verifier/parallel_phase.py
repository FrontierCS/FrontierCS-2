"""Independent GPU parents per workload group; join only complete results."""
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from safe_error_frames import safe_error_frames
from instance_manifest import expected_groups
from phase_run import VARIANTS, run_phase
from pod import PlatformError, PodSession, SchedulerPreinputInfrastructureError
from runtime_config import execution_variants
from cpu_reference_verdict import TrustedMeasurementError, evaluate_final, evaluate_train


class PhaseInfrastructureError(RuntimeError):
    """Only allowlisted platform and pre-input signals; no scientific score."""
    def __init__(self, failures):
        super().__init__('Owned platform provisioning could not complete')
        self.failures = failures
        self.cleanup_required = any(row.get('cleanup_required', True) for row in failures.values())


def infrastructure_signal(error):
    # Exact types prevent unknown remote/child exceptions from gaining infra
    # status through a shared RuntimeError or SidecarExecutionError base class.
    if type(error) in {PlatformError, SchedulerPreinputInfrastructureError}:
        return dict(error.signal)
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


def run_parallel_phase(config, plan, mode, *, environment, candidate_patch,
                       evidence, deadline, calibration=False, hidden_jps=None,
                       fixed_targets=None):
    """All parent sessions close independently before any result is scored."""
    groups = expected_groups(mode)
    variants = execution_variants(config)
    if (variants == ('candidate',)) != (fixed_targets is not None):
        raise TrustedMeasurementError(None, None, 'fixed_target_variant_binding_mismatch')
    if fixed_targets is not None:
        from fixed_targets import validate_targets
        fixed_targets = validate_targets(fixed_targets, expected_environment=fixed_targets['environment'])
    phase = plan['phases'][mode]
    evidence = Path(evidence)
    completed = {}
    errors = {}
    trusted_errors = []
    error_details = {}
    infrastructure = {}
    evidence.mkdir(parents=True, exist_ok=True)
    identity = {field: environment[variable] for field, variable in
                (('run_id', 'FRONTIER_RUN_ID'), ('attempt_id', 'FRONTIER_ATTEMPT_ID'),
                 ('verification_id', 'FRONTIER_VERIFICATION_ID'))}

    def execute_group(group):
        group_evidence = evidence / group.replace('/', '-')
        group_evidence.mkdir(parents=True, exist_ok=False)
        with PodSession(config, mode, group.split('/')[0], identity=identity,
                        deadline=deadline - phase['cleanup_seconds'] - phase['capture_seconds'],
                        cleanup_seconds=phase['cleanup_seconds'],
                        capture_seconds=phase['capture_seconds'],
                        evidence=group_evidence / 'parent') as session:
            return run_phase(session, config, plan, candidate_patch=candidate_patch,
                evidence=group_evidence / 'phase', calibration=calibration,
                hidden_jps=hidden_jps, selected_group=group,
                controller_environment=environment,
                **{key: phase[key] for key in ('startup_seconds', 'stop_seconds',
                                              'max_stdout_bytes', 'max_stderr_bytes')})

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
    if errors:
        try:
            (evidence / 'incomplete-groups.json').write_text(json.dumps({
                'failed_groups': errors, 'completed_groups': sorted(completed),
                'failure_details': error_details,
                'infrastructure_signals': infrastructure,
                'score_ready': False}, sort_keys=True) + '\n')
        except Exception:
            pass
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
