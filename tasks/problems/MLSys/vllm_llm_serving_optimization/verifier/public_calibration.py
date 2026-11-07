"""Operator-only full public calibration; never a native submission/final route.

The trusted launcher retains the earliest-create marker across retries/replays.
Driver-clamped grant minus the rounding guard bounds attempts conservatively;
this is not an exact owner clock or a guarantee of durable evidence upload.
No cloud action occurs without --execute; no provider client/credential is used.
"""
import argparse
import fcntl
from contextlib import contextmanager
import stat
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time

sys.path.insert(0, str(Path(__file__).parents[1] / 'judge'))
from gpu_session import DriverGPU, GPUSession
from calibration_clock import begin
from instance_manifest import expected_groups
from phase_run import run_phase, VARIANTS
from provisioning_owner import for_phase
from runtime_config import canonical_digest, phase_limits, validate_projection
from runtime_plan import group_operations, validate_transport


SCHEMA = 'frontier.vllm-public-calibration/v3'


def validate_calibration(plan, config):
    validate_projection(config)
    if set(plan) != {'schema', 'inputs', 'phase', 'relay', 'correctness'} or plan['schema'] != SCHEMA:
        raise ValueError('closed public-only calibration plan required')
    inputs = plan['inputs']
    if set(inputs) != {'payload', 'dataset'}:
        raise ValueError('exact calibration payload and dataset bindings required')
    for name, binding in inputs.items():
        if (not isinstance(binding, dict) or set(binding) != {'sha256', 'bytes'}
                or not isinstance(binding['sha256'], str) or not binding['sha256'].startswith('sha256:')
                or len(binding['sha256']) != 71 or any(c not in '0123456789abcdef' for c in binding['sha256'][7:])
                or type(binding['bytes']) is not int or binding['bytes'] <= 0):
            raise ValueError('invalid immutable calibration input binding: ' + name)
    phase = plan['phase']
    fields = {'gpu_seconds', 'cleanup_seconds', 'capture_seconds', 'collector_seconds', 'startup_seconds', 'stop_seconds', 'max_stdout_bytes', 'max_stderr_bytes'}
    if set(phase) != fields or any(type(n) is not int or n <= 0 for n in phase.values()):
        raise ValueError('every calibration limit must be explicit and positive')
    limits = phase_limits(config, 'train', requested_gpu_seconds=phase['gpu_seconds'])
    if (phase['cleanup_seconds'] + phase['capture_seconds'] + phase['collector_seconds'] + phase['startup_seconds'] + phase['stop_seconds'] >= phase['gpu_seconds']
            or phase['max_stdout_bytes'] + phase['max_stderr_bytes'] > limits['ssh_bytes_per_direction']
            or limits['ssh_sessions'] < group_operations('train', 'swe/public')
            or limits['timeout_seconds'] < phase['gpu_seconds'] + phase['cleanup_seconds'] + phase['capture_seconds'] + phase['collector_seconds']):
        raise ValueError('calibration limits do not fund the complete public execution path')
    # The declared catalog itself is validated by the projection: five public
    # testbeds per variant plus five check workers and the scheduler for train.
    validate_transport(plan['relay'], plan['correctness'], config,
                       {'train': {'max_stderr_bytes': phase['max_stderr_bytes'],
                                  'budget_seconds': limits['timeout_seconds'] - phase['cleanup_seconds'] - phase['capture_seconds']}})
    return plan


def validate_bound_inputs(plan, environment):
    """Bind wrapper-downloaded bytes before any Driver or provider operation."""
    paths = {'payload': Path(environment['FRONTIER_PAYLOAD_PATH']),
             'dataset': Path(environment['FRONTIER_DATASET_PATH'])}
    observed = {}
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file() or path.stat().st_size != plan['inputs'][name]['bytes']:
            raise ValueError('calibration ' + name + ' differs from its immutable binding')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if 'sha256:' + digest != plan['inputs'][name]['sha256']:
            raise ValueError('calibration ' + name + ' differs from its immutable binding')
        observed[name] = {'sha256': 'sha256:' + digest, 'bytes': path.stat().st_size}
    return observed


@contextmanager
def execution_lock(clock_path):
    descriptor = os.open(str(clock_path)+'.execution.lock', os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError('execution lock must be a regular file')
        fcntl.flock(descriptor, fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def execute(plan, config, *, output, environment, clock_path, resume=False,
            owner_deadline=None, owner_clock_receipt=None, margin_seconds=1.0):
    with execution_lock(clock_path):
        return _execute_locked(plan, config, output=output, environment=environment, clock_path=clock_path,
                               resume=resume, owner_deadline=owner_deadline, owner_clock_receipt=owner_clock_receipt, margin_seconds=margin_seconds)


def _execute_locked(plan, config, *, output, environment, clock_path, resume=False,
            owner_deadline=None, owner_clock_receipt=None, margin_seconds=1.0):
    started = time.monotonic()
    validate_calibration(plan, config)
    phase = plan['phase']
    if owner_deadline is not None:
        if (not math.isfinite(owner_deadline)
                or (not resume and owner_deadline < started + phase['gpu_seconds'] + phase['cleanup_seconds'] + phase['capture_seconds'] + phase['collector_seconds'])
                or owner_clock_receipt is None or not owner_clock_receipt.is_file()
                or not owner_clock_receipt.stat().st_size):
            raise ValueError('optional owner cutoff requires funded grant/headroom and provenance receipt')
    elif owner_clock_receipt is not None:
        raise ValueError('owner receipt without its independently justified cutoff')
    verification_id = environment['FRONTIER_VERIFICATION_ID']
    key = hashlib.sha256((verification_id + '\0public-calibration').encode()).hexdigest()
    clock = begin({'driver_origin': environment['FRONTIER_DRIVER_INTERNAL_URL'],
                   'verification_id': verification_id, 'request_key': key,
                   'request': {'gpu_type': 'H100', 'gpu_count': 1, 'max_duration_seconds': phase['gpu_seconds']},
                   'config_sha256': canonical_digest(config), 'plan_sha256': canonical_digest(plan)},
                  clock_path, cap_seconds=phase['gpu_seconds'], cleanup_seconds=phase['cleanup_seconds'],
                  capture_seconds=phase['capture_seconds'] + phase['collector_seconds'], margin_seconds=margin_seconds, resume=resume)
    cutoffs = clock.initial_deadlines
    if owner_deadline is not None:
        cutoffs = {key: min(value, owner_deadline - ({'work_deadline': phase['cleanup_seconds'] + phase['capture_seconds'] + phase['collector_seconds'],
                    'delete_deadline': phase['capture_seconds'] + phase['collector_seconds'], 'capture_deadline': 0}[key])) for key,value in cutoffs.items()}
    if time.monotonic() >= cutoffs['delete_deadline' if resume else 'work_deadline']:
        raise ValueError('persisted calibration budget already exhausted')
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / 'evidence'; evidence.mkdir()
    record = {'schema': SCHEMA, 'calibration_only': True, 'acceptance': False,
              'runtime_projection_sha256': canonical_digest(config), 'plan_sha256': canonical_digest(plan),
              'clock_marker': str(clock_path), 'earliest_create_monotonic': clock.t0,
              'clock_scope': 'conservative comparable-elapsed-seconds bound; not exact owner time',
              'rounding_and_supported_error_margin_seconds': margin_seconds, 'initial_cutoffs': cutoffs,
              'normal_clock_assumptions': ['ordinary comparable elapsed-clock rates',
                                           'launcher host does not suspend during the diagnostic'],
              'candidate_patch_sha256': 'sha256:' + hashlib.sha256(b'').hexdigest(),
              'accounting_expected': {'parent_creates': 1, 'declared_sidecars': 21,
                                      'parent_ssh': group_operations('train', 'swe/public'),
                                      'scheduler_jobs': 6, 'receipt_fetches': 15},
              'status': 'running', 'score_ready': False}
    (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    collector_deadline = cutoffs['capture_deadline']
    capture_deadline = collector_deadline - phase['collector_seconds']
    session = None
    try:
        client = DriverGPU(environment['FRONTIER_DRIVER_INTERNAL_URL'], verification_id,
                           environment['FRONTIER_VERIFIER_TOKEN'], deadline=cutoffs['work_deadline'])
        if resume:
            # Never replay potentially executed batches. Recover only the known
            # parent; absent grant remains Driver-owned uncertain cleanup.
            if clock.runner_id is None:
                record.update(status='cleanup_uncertain', reason='no persisted parent identity; Driver reconciliation required')
                return record
            client.deadline = cutoffs['delete_deadline']
            cleanup = client.request('DELETE', '/' + clock.runner_id)
            (evidence/'resume-cleanup.json').write_text(json.dumps(cleanup)+'\n')
            if cleanup.get('runner_id') != clock.runner_id or cleanup.get('state') != 'terminated':
                raise ValueError('resumed parent cleanup unconfirmed')
            record.update(status='cleanup_only', scientific_execution_replayed=False)
            return record
        owner = for_phase(environment, config, plan, 'train', tuple(expected_groups('train')), b'',
            Path(__file__).parents[1], {'gpu_type': 'H100', 'gpu_count': 1,
                                      'max_duration_seconds': phase['gpu_seconds']}, key)
        session = GPUSession(client, config, 'train', gpu_seconds=phase['gpu_seconds'], request_key=key,
                             scratch=output / 'ssh', evidence=evidence / 'parent',
                             cleanup_deadline=cutoffs['delete_deadline'], calibration_clock=clock,
                             provisioning_owner=owner)
        with session:
            result = run_phase(session, config, {'relay': plan['relay'], 'correctness': plan['correctness'],
                               'phases': {'train': {'cleanup_seconds': phase['cleanup_seconds']}}},
                               candidate_patch=b'', evidence=evidence / 'phase', calibration=True,
                               controller_environment=environment,
                               **{k: phase[k] for k in ('startup_seconds', 'stop_seconds',
                                                        'max_stdout_bytes', 'max_stderr_bytes')})
        record.update(status='complete', operator_result=result)
    except BaseException as error:
        record.update(status='runtime_failure', error_type=type(error).__name__)
        raise
    finally:
        collector_deadline = min(collector_deadline, clock.deadlines['capture_deadline'])
        if session is not None and session.capture_deadline is not None:
            collector_deadline = min(collector_deadline, session.capture_deadline)
        capture_deadline = min(capture_deadline, collector_deadline - phase['collector_seconds'])
        record['final_cutoffs'] = clock.deadlines
        record['local_capture_deadline_monotonic'] = capture_deadline
        record['collector_deadline_monotonic'] = collector_deadline
        try:
            if time.monotonic() >= capture_deadline:
                raise TimeoutError('capture reserve exhausted')
            (evidence/'clock-marker.json').write_bytes(clock_path.read_bytes())
            (evidence/'calibration-summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
            code = 'from pathlib import Path; import sys; from evidence import write_evidence; write_evidence(Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]))'
            subprocess.run([sys.executable, '-c', code, str(evidence), str(output/'evidence.tar.zst'),
                            str(config['max_evidence_bytes'])], check=True,
                           timeout=max(0.001,capture_deadline-time.monotonic()),
                           env={'PATH':'/usr/bin:/bin','PYTHONPATH':str(Path(__file__).parent),
                                'PYTHONDONTWRITEBYTECODE':'1'})
            if time.monotonic() >= capture_deadline:
                raise TimeoutError('local capture finished after cutoff')
            record['local_capture_before_cutoff'] = True
        except Exception as error:
            record.update(status='capture_failure', capture_error_type=type(error).__name__,
                          local_capture_before_cutoff=False)
            record.pop('operator_result', None)
        record['elapsed_seconds'] = time.monotonic() - started
        record['evidence_scope'] = ('bounded local package; the official Verifier wrapper owns authenticated upload '
                                    'within the separately reserved collector interval')
        (output / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        if time.monotonic() >= capture_deadline:
            raise TimeoutError('result/capture cutoff exceeded; no acceptance')
        if record['status'] in {'capture_failure','cleanup_uncertain'}:
            raise RuntimeError('calibration capture or cleanup incomplete; inspect operator result')
    return record




def platform_wrapper(plan, config, environment):
    """Write standard output for the existing trusted Verifier wrapper.

    Platform ``passed`` means only that a bounded diagnostic record was
    collected. The score is fixed at zero and scientific acceptance is disabled.
    """
    result_path = Path(environment['FRONTIER_RESULT_PATH'])
    evidence_path = Path(environment['FRONTIER_EVIDENCE_PATH'])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'failed', 'failure': {'code': 'vllm_calibration_collection_failed',
              'message': 'Calibration cleanup or evidence collection could not be confirmed.',
              'retryable': False}}
    if environment.get('FRONTIER_VERIFICATION_MODE') != 'train':
        result_path.write_text(json.dumps(result, allow_nan=False) + '\n')
        return 0
    try:
        validate_bound_inputs(plan, environment)
        output = result_path.parent / 'public-calibration'
        marker = result_path.parent / 'public-calibration-create-clock.json'
        runtime_error = None
        try:
            execute(plan, config, output=output, environment=environment, clock_path=marker)
        except Exception as error:
            runtime_error = error
        record_path = output / 'result.json'
        record = json.loads(record_path.read_bytes()) if record_path.is_file() else None
        archive = output / 'evidence.tar.zst'
        collected = (isinstance(record, dict) and record.get('local_capture_before_cutoff') is True
                     and archive.is_file() and not evidence_path.exists()
                     and time.monotonic() < record.get('collector_deadline_monotonic', -1))
        if collected:
            os.replace(archive, evidence_path)
            complete = record.get('status') == 'complete'
            result = {'status': 'passed', 'raw_score': 0.0,
                      'summary': ('Calibration diagnostic complete; no scientific score or acceptance.' if complete else
                                  'Calibration diagnostic incomplete; bounded failure evidence retained; no scientific score or acceptance.'),
                      'metrics': {'diagnostic_complete': int(complete)}}
        elif runtime_error is not None:
            result['failure']['message'] = 'Calibration failed before bounded evidence collection completed.'
    except Exception:
        # The platform wrapper retains task stdout/stderr for this closed failure;
        # do not copy exception text or credentials into the result envelope.
        pass
    result_path.write_text(json.dumps(result, allow_nan=False) + '\n')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--execute', action='store_true', help='only after separate exact-scope authorization')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--clock-marker', type=Path)
    parser.add_argument('--clock-margin-seconds', type=float, default=1.0,
                        help='at least1s rounding plus any operator-supported elapsed-clock error allowance')
    parser.add_argument('--resume', action='store_true', help='cleanup only; requires original marker, never reruns scientific work')
    parser.add_argument('--owner-deadline-monotonic', type=float)
    parser.add_argument('--owner-clock-receipt', type=Path)
    parser.add_argument('--platform-wrapper', action='store_true',
                        help='write standard output for the existing trusted Verifier wrapper')
    args = parser.parse_args()
    config = json.loads(args.runtime_config.read_bytes())
    plan = validate_calibration(json.loads(args.plan.read_bytes()), config)
    if args.platform_wrapper:
        if (args.execute or args.output or args.clock_marker or args.resume
                or args.owner_deadline_monotonic or args.owner_clock_receipt):
            parser.error('--platform-wrapper owns execution paths and cannot use operator path arguments')
        raise SystemExit(platform_wrapper(plan, config, os.environ))
    if not args.execute:
        print(json.dumps({'scope': 'offline plan validation only', 'gpu_seconds_cap': plan['phase']['gpu_seconds'],
                          'scheduler_jobs': 6, 'receipt_fetches': 15, 'acceptance': False}))
        return
    if any(v is None for v in (args.output, args.clock_marker)):
        parser.error('execution requires output and a durable clock marker; explicit resume must retain its original marker')
    execute(plan, config, output=args.output, owner_deadline=args.owner_deadline_monotonic,
            owner_clock_receipt=args.owner_clock_receipt, environment=os.environ, clock_path=args.clock_marker, resume=args.resume, margin_seconds=args.clock_margin_seconds)


if __name__ == '__main__':
    main()
