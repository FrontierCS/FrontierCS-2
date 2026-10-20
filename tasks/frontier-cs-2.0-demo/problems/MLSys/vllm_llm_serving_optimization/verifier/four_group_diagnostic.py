"""Four Qwen workload groups on independent H100 parents; diagnostic only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parents[1] / 'judge'))
from batch_coordinator import FINAL_HIDDEN_JPS
from calibration_clock import begin
from evidence import write_evidence
from gpu_session import DriverGPU
from instance_manifest import expected_groups
from parallel_phase import run_parallel_phase
from runtime_config import canonical_digest
from runtime_plan import validate_plan
from verify import dataset_selector, extract_patch

# Configured by the task author for evaluation.
HIDDEN_JPS = FINAL_HIDDEN_JPS


def validate_diagnostic(plan, config, mode='eval'):
    validate_plan(plan, config)
    groups = expected_groups(mode)
    return {'mode': mode, 'groups': list(groups),
            'execution_topology': 'declared_sidecars',
            'declared_sidecars_per_parent': len(expected_groups(mode)['swe/public']) * 4 + 1,
            'grants': {group: plan['phases'][mode]['groups'][group]['gpu_seconds']
                       for group in groups},
            'aggregate_gpu_seconds': config['verifier_quota']['max_gpu_seconds'],
            'diagnostic_only': True, 'formal_rates_approved': True,
            'acceptance': False, 'launch_authorized': False}


def cleanup_only(config, plan, *, environment, candidate_patch, clocks, deadline, mode='eval'):
    """Attempt every recorded parent; never replay work to resolve a missing clock."""
    groups = expected_groups(mode)
    records = {}
    for group in groups:
        try:
            key = group.replace('/', '-')
            grant = plan['phases'][mode]['groups'][group]['gpu_seconds']
            request_key = hashlib.sha256((environment['FRONTIER_VERIFICATION_ID'] + '\0' + mode + '\0' + group +
                                          '\0' + hashlib.sha256(candidate_patch).hexdigest()).encode()).hexdigest()
            clock = begin({'driver_origin': environment['FRONTIER_DRIVER_INTERNAL_URL'],
                           'verification_id': environment['FRONTIER_VERIFICATION_ID'],
                           'request_key': request_key,
                           'request': {'gpu_type': 'H100', 'gpu_count': 1,
                                       'max_duration_seconds': grant},
                           'config_sha256': canonical_digest(config),
                           'plan_sha256': canonical_digest(plan)},
                          clocks / (mode + '-' + key + '.json'), cap_seconds=grant,
                          cleanup_seconds=plan['phases'][mode]['cleanup_seconds'],
                          capture_seconds=plan['phases'][mode]['capture_seconds'],
                          resume=True)
            if clock.runner_id is None:
                records[group] = {'state': 'unbound_create_requires_driver_reconciliation'}
                continue
            client = DriverGPU(environment['FRONTIER_DRIVER_INTERNAL_URL'],
                               environment['FRONTIER_VERIFICATION_ID'],
                               environment['FRONTIER_VERIFIER_TOKEN'],
                               deadline=min(deadline, clock.deadlines['delete_deadline']))
            deleted = client.request('DELETE', '/' + clock.runner_id)
            if deleted.get('runner_id') != clock.runner_id or deleted.get('state') != 'terminated':
                records[group] = {'state': 'cleanup_uncertain', 'runner_id': clock.runner_id}
            else:
                records[group] = {'state': 'terminated', 'runner_id': clock.runner_id}
        except Exception as error:
            # One missing receipt or failed delete must not strand other parents.
            # Do not leak provider messages/tokens or invent a termination receipt.
            records[group] = {'state': 'cleanup_uncertain', 'error_type': type(error).__name__}
    return {'mode': mode, 'diagnostic_only': True, 'scientific_execution_replayed': False,
            'score_ready': False,
            'cleanup_complete': all(row['state'] == 'terminated' for row in records.values()),
            'parents': records}


def platform_wrapper(config, plan, environment):
    """Native train/eval entry: zero scientific score, independent evidence."""
    result_path = Path(environment['FRONTIER_RESULT_PATH'])
    evidence_path = Path(environment['FRONTIER_EVIDENCE_PATH'])
    result = {'status': 'failed', 'failure': {'code': 'vllm_four_group_diagnostic_failed',
        'message': 'Four-group diagnostic did not complete and retain evidence.',
        'retryable': False}}
    try:
        mode = environment.get('FRONTIER_VERIFICATION_MODE')
        validate_diagnostic(plan, config, mode)
        dataset_selector(Path(environment['FRONTIER_DATASET_PATH']), mode)
        root = result_path.parent / 'four-group-diagnostic'
        root.mkdir(parents=True, exist_ok=False)
        patch = extract_patch(Path(environment['FRONTIER_PAYLOAD_PATH']),
                              root / 'solution.patch', plan['max_payload_bytes'])
        timeout_key = 'max_train_wall_seconds' if mode == 'train' else 'eval_timeout_seconds'
        end = time.monotonic() + config['verifier_quota'][timeout_key]
        completed = False
        try:
            operator = run_parallel_phase(config, plan, mode, environment=environment,
                candidate_patch=patch, root=root, evidence=root / 'operator-evidence',
                deadline=end, calibration=True,
                hidden_jps=HIDDEN_JPS if mode == 'eval' else None)
            (root / 'operator-summary.json').write_text(json.dumps({
                'mode': mode, 'diagnostic_only': True, 'acceptance': False,
                'formal_rates_approved': True, 'operator_result': operator},
                indent=2, allow_nan=False) + '\n')
            completed = True
        except BaseException as error:
            (root / 'failure.json').write_text(json.dumps({
                'error_type': type(error).__name__, 'score_ready': False}) + '\n')
        captured = []
        missing = []
        groups = expected_groups(mode)
        for group in groups:
            receipt = root / 'operator-evidence' / group.replace('/', '-')
            if ((receipt / 'parent-clock.json').is_file()
                    and (receipt / 'parent-cutoffs.json').is_file()):
                captured.append(group)
            else:
                missing.append(group)
        (root / 'capture-status.json').write_text(json.dumps({
            'mode': mode, 'captured_groups': captured, 'missing_groups': missing,
            'operator_complete': completed, 'score_ready': False},
            sort_keys=True) + '\n')
        if time.monotonic() >= end:
            raise TimeoutError('Verifier evidence capture deadline expired')
        write_evidence(root, evidence_path, config['max_evidence_bytes'])
        if time.monotonic() >= end:
            raise TimeoutError('global diagnostic evidence exceeded Verifier cutoff')
        if completed and not missing:
            result = {'status': 'passed', 'raw_score': 0.0,
                      'summary': 'Workload-group diagnostic evidence captured; no formal acceptance.',
                      'metrics': {'diagnostic_complete': 1, 'calibration_only': 1,
                                  'workload_groups': len(groups)}}
        else:
            result['failure']['message'] = (
                'Workload-group diagnostic incomplete; bounded failure evidence captured.')
    except Exception:
        pass
    result_path.write_text(json.dumps(result, allow_nan=False) + '\n')
    return 0


def main():
    if sys.argv[1:] == ['--platform-run']:
        config = json.loads(Path('/opt/frontier/runtime-config.json').read_bytes())
        plan = json.loads(Path('/opt/frontier/runtime-plan.json').read_bytes())
        raise SystemExit(platform_wrapper(config, plan, os.environ))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--candidate-patch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('train', 'eval'), default='eval')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preview', action='store_true')
    action.add_argument('--execute', action='store_true')
    action.add_argument('--cleanup-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    plan = json.loads(args.plan.read_bytes())
    preview = validate_diagnostic(plan, config, args.mode)
    patch = args.candidate_patch.read_bytes()
    if len(patch) > 1_500_000:
        raise ValueError('candidate patch exceeds original payload bound')
    if args.preview:
        print(json.dumps(preview, sort_keys=True))
        return
    args.output.mkdir(parents=True, exist_ok=args.cleanup_only)
    clocks = args.output / 'parent-clocks'
    environment = {key: os.environ[key] for key in ('FRONTIER_DRIVER_INTERNAL_URL',
        'FRONTIER_VERIFICATION_ID', 'FRONTIER_VERIFIER_TOKEN')}
    timeout_key = 'max_train_wall_seconds' if args.mode == 'train' else 'eval_timeout_seconds'
    deadline = time.monotonic() + config['verifier_quota'][timeout_key]
    if args.cleanup_only:
        result = cleanup_only(config, plan, environment=environment, candidate_patch=patch,
                              clocks=clocks, deadline=deadline, mode=args.mode)
    else:
        result = run_parallel_phase(config, plan, args.mode, environment=environment,
            candidate_patch=patch, root=args.output, evidence=args.output / 'operator-evidence',
            deadline=deadline, clock_directory=clocks, calibration=True,
            hidden_jps=HIDDEN_JPS if args.mode == 'eval' else None)
        result = {'diagnostic_only': True, 'acceptance': False,
                  'formal_rates_approved': True, 'score_publishable': False,
                  'operator_result': result}
    (args.output / 'diagnostic-result.json').write_text(json.dumps(result,
        indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__': main()
