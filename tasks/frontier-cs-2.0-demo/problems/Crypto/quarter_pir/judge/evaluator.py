from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import secrets
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import yaml

from submission_patch import apply_submission_patch
from table2_metrics import (GATED_METRICS, check_result, compare_experiment, continuous_score, audit_failure,
                            probe_failure, reference_message_bytes, reference_volume_storage, score_online,
                            storage_peak, timing_values, validity_failure)


TASK = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((TASK / 'config.yaml').read_text())
CODE = Path('quarter_pir/code')
EDITABLE = CODE / 'pir'
SIZES = (67108864,)
QUERY_COUNT = 1000
THREADS = 4
PLAN_SCHEMA = 'quarter-pir.fresh-seed-plan/v1'
VISIBLE = CONFIG['evaluation']['solver_visible_metrics']
LIMIT = CONFIG['runtime']['timeout_seconds']
RUN_LIMIT = 3000  # one conductor run, the volume phase of the probe included
# Explicit import allowlist of the editable package, checked on the compiled
# package's import graph here and on every file's syntax tree by the trusted policy tool.
ALLOWED_IMPORTS = frozenset((
    'bytes', 'encoding/binary', 'math', 'math/bits', 'math/rand', 'sort', 'sync', 'sync/atomic',
    'example.com/util', 'easypir/refwire', 'easypir/safeaes'))
# Solver-visible validity codes (candidate_public_validity_code).
VALIDITY_CODES = {
    'valid': 0, 'patch': 1, 'source_policy': 2, 'build': 3, 'client': 4, 'hint_export': 5,
    'request_reduction': 6, 'wrong_value': 7, 'process_memory': 8, 'server_replay': 9,
    'time_limit': 10, 'coverage_probe': 11, 'hint_audit': 12,
}
CONDUCTOR_FAILURES = {
    'client_crashed_or_protocol': 'client', 'hint_export_length': 'hint_export',
    'request_not_reduction': 'request_reduction', 'reduction_nondeterministic': 'request_reduction',
    'reducer_crashed': 'request_reduction', 'server_rpc_failed': 'wrong_value',
    'wrong_value': 'wrong_value',
}


class Rejected(Exception):
    def __init__(self, message: str, category: str = 'patch', details: dict | None = None):
        super().__init__(message)
        self.category = category
        self.details = details  # private diagnostics only


class TaskConfigurationError(RuntimeError):
    """Trusted task data is invalid; this is not a submission verdict."""
    pass


def state_root() -> Path:
    path = Path(os.environ.get('FRONTIER_PRIVATE_STATE', str(TASK.parent / 'scratch/quarter-pir-private'))).resolve()
    if TASK == path or TASK in path.parents:
        raise Rejected('private state must be outside the task package')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def safe_env() -> dict:
    environment = {
        'PATH': os.environ.get('PATH', '/usr/local/go/bin:/usr/bin:/bin'),
        'HOME': str(state_root()), 'TMPDIR': str(state_root() / 'tmp'),
        'GOCACHE': str(state_root() / 'go-build'),
        'GOMODCACHE': os.environ.get('GOMODCACHE', '/opt/frontier/go-mod'),
        'GOPROXY': 'off', 'GOSUMDB': 'off', 'CGO_ENABLED': '0',
        'GOFLAGS': '-mod=readonly -buildvcs=false', 'GOMAXPROCS': str(THREADS), 'GOGC': '100',
        'LC_ALL': 'C', 'GOTOOLCHAIN': 'local',
    }
    Path(environment['TMPDIR']).mkdir(exist_ok=True)
    return environment


def remaining(deadline: float, cap: float = LIMIT) -> float:
    duration = min(cap, deadline - time.monotonic())
    if duration <= 0:
        raise Rejected('evaluation time limit exceeded', 'time_limit')
    return duration


def command(arguments, cwd, deadline, *, environment=None, cap=LIMIT, category='build'):
    try:
        process = subprocess.Popen([str(item) for item in arguments], cwd=cwd,
                                   env=environment or safe_env(), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        try:
            output, error = process.communicate(timeout=remaining(deadline, cap))
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        if process.returncode != 0:
            raise Rejected('build, policy, or protocol validation failed', category)
        return output
    except subprocess.TimeoutExpired as error:
        raise Rejected('evaluation time limit exceeded', 'time_limit') from error


def signature() -> str:
    digest = hashlib.sha256()
    for path in sorted([TASK / 'reference.patch', TASK / 'baseline_naive.patch', TASK / 'beat_spec.json',
                        TASK / 'evaluator.py', TASK / 'table2_metrics.py',
                        *list((TASK / 'harbor/app').rglob('*'))]):
        if path.is_file():
            digest.update(path.relative_to(TASK).as_posix().encode())
            digest.update(path.read_bytes())
    digest.update(repr(SIZES).encode())
    return digest.hexdigest()


def apply(patch: Path) -> tuple[Path, Path]:
    result = apply_submission_patch(patch, TASK, CONFIG)
    if not result.ok or result.workdir is None:
        raise Rejected('patch rejected: invalid format, path, or application', 'patch')
    return result.workdir, result.cleanup_root


def inventory(root: Path) -> dict:
    """Relative path -> content digest for every entry; non-regular entries are marked."""
    entries = {}
    for directory, names, files in os.walk(root, followlinks=False):
        for name in names + files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                entries[relative] = 'special'
            elif path.is_file():
                entries[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return entries


def check_protected(tree: Path):
    """Everything outside the editable package is byte-identical to the trusted tree."""
    editable = EDITABLE.as_posix() + '/'
    trusted = {name: value for name, value in inventory(TASK / 'harbor/app').items()
               if not name.startswith(editable)}
    candidate = inventory(tree)
    if any(value == 'special' for value in candidate.values()):
        raise Rejected('patch rejected: only regular files are allowed', 'patch')
    outside = {name: value for name, value in candidate.items() if not name.startswith(editable)}
    if outside != trusted:
        raise Rejected('patch rejected: protected harness files differ', 'patch')


def check_source_policy(tree: Path, deadline: float):
    """Syntax-tree policy on every file, then the compiled package's real import graph."""
    state = state_root()
    report = json.loads(command([state / 'policy', TASK / 'harbor/app' / CODE, tree / CODE], tree, deadline,
                                cap=60, category='source_policy'))
    listed = json.loads(command(['go', 'list', '-json', './pir'], tree / CODE, deadline,
                                cap=120, category='source_policy'))
    go_files = sorted(listed.get('GoFiles') or [])
    expected = sorted(report['editable_go_files'])
    forbidden = ('CgoFiles', 'IgnoredGoFiles', 'IgnoredOtherFiles', 'SFiles', 'CFiles', 'CXXFiles', 'MFiles',
                 'HFiles', 'FFiles', 'SwigFiles', 'SwigCXXFiles', 'SysoFiles', 'EmbedPatterns',
                 'EmbedFiles', 'TestGoFiles', 'XTestGoFiles', 'InvalidGoFiles')
    if (listed.get('ImportPath') != 'easypir/pir' or listed.get('Name') != 'pir'
            or go_files != expected
            or any(listed.get(key) for key in forbidden)
            or not set(listed.get('Imports') or []) <= ALLOWED_IMPORTS
            or listed.get('Error') or listed.get('DepsErrors')):
        raise Rejected('source policy rejected the compiled package', 'source_policy')


def build(tree: Path, target: Path, deadline: float):
    target.mkdir(parents=True, exist_ok=True)
    code = tree / CODE
    limit = CONFIG['environment']['build_timeout_seconds']
    for name, package in (('server', './server'), ('client', './ec24'), ('reducer', './reducer')):
        command(['go', 'build', '-trimpath', '-o', target / name, package], code, deadline, cap=limit)


def port() -> int:
    with socket.socket() as connection:
        connection.bind(('127.0.0.1', 0))
        return connection.getsockname()[1]


def stop(process):
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def fresh_seeds() -> dict:
    """Private per-experiment seeds; never fixed, published or stored in a dataset."""
    return {'db_seed': secrets.randbelow(2**63 - 1) + 1,
            'coin_seed': secrets.randbelow(2**63),
            'query_seed': secrets.randbelow(2**63),
            'probe_seed': secrets.randbelow(2**63)}


def integration(binary: Path, size: int, seeds: dict, target: Path, deadline: float, *,
                skip_volume: bool = False) -> dict:
    """One complete trusted-conductor experiment; returns its result record."""
    target.mkdir(parents=True, exist_ok=True)
    (target / 'config.txt').write_text(f'{size} {seeds["db_seed"]}\n')
    server_port = port()
    settings = {
        'upstream': f'127.0.0.1:{server_port}', 'client': str(binary / 'client'),
        'reducer': str(binary / 'reducer'), 'size': size, 'queries': QUERY_COUNT, 'threads': THREADS,
        'transcript': str(target / 'transcript.json'), 'result': str(target / 'result.json'), **seeds,
        'skip_volume': skip_volume,
    }
    (target / 'conductor.json').write_text(json.dumps(settings))
    processes = []
    try:
        with open(os.devnull, 'wb') as suppressed:
            server = subprocess.Popen([str(binary / 'server'), '-port', str(server_port), '-ignoreInit=false'],
                                      cwd=target, env=safe_env(), stdout=suppressed, stderr=suppressed,
                                      start_new_session=True)
            processes.append(server)
            conductor = subprocess.Popen([str(state_root() / 'conductor'), str(target / 'conductor.json')],
                                         cwd=target, env=safe_env(), stdout=suppressed, stderr=suppressed,
                                         start_new_session=True)
            processes.append(conductor)
            try:
                conductor.wait(timeout=remaining(deadline, RUN_LIMIT))
            except subprocess.TimeoutExpired as error:
                raise Rejected('client experiment timed out', 'time_limit') from error
            if conductor.returncode != 0 or not (target / 'result.json').is_file():
                raise Rejected('client experiment could not run', 'client')
            result = json.loads((target / 'result.json').read_text())
            if result.get('valid') is not True:
                failure = str(result.get('failure'))
                raise Rejected('client experiment invalid: ' + failure,
                               CONDUCTOR_FAILURES.get(failure, 'client'))
            if server.poll() is not None:
                raise Rejected('server exited during the experiment', 'wrong_value')
            try:
                return check_result(result, size=size, queries=QUERY_COUNT, threads=THREADS)
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                raise Rejected('client experiment record invalid', 'client') from error
    finally:
        for process in reversed(processes):
            stop(process)
        for name in ('config.txt', 'conductor.json'):
            (target / name).unlink(missing_ok=True)


def replay(binary: Path, size: int, seed: int, transcript: Path, output: Path, deadline: float, *, validate=False):
    environment = safe_env()
    environment['FRONTIER_REPLAY'] = '1'
    command([binary / 'server', size, seed, transcript, output, 'validate' if validate else 'measure'],
            output.parent, deadline, environment=environment, category='server_replay')
    result = json.loads(output.read_text())
    if any(len(result[key]) != QUERY_COUNT for key in ('times', 'upload', 'download')):
        raise Rejected('incomplete RPC measurements', 'server_replay')
    if not validate:
        if any(len(row) != 5 or any(type(value) is not int or value <= 0 for value in row) for row in result['times']):
            raise Rejected('invalid RPC measurements', 'server_replay')
        result['median_ns'] = [statistics.median(row) for row in result['times']]
    return result


def experiment_plan(final: bool) -> list:
    """Database sizes of the paired experiments for this role, from the dataset plan."""
    expected = list(SIZES) * (2 if final else 1)
    dataset_path = os.environ.get('FRONTIER_DATASET_PATH')
    if not dataset_path:
        return expected
    try:
        with tarfile.open(dataset_path, 'r:') as archive:
            members = archive.getmembers()
            if len(members) != 1 or members[0].name != 'plan.json' or not members[0].isfile() or members[0].size > 65536:
                raise ValueError('unexpected dataset members')
            plan = json.loads(archive.extractfile(members[0]).read())
        if (plan.get('schema') != PLAN_SCHEMA or plan.get('queries_per_experiment') != QUERY_COUNT
                or plan.get('seed_policy') != 'fresh-private-per-evaluation'
                or [row['db_size'] for row in plan['experiments']] != expected):
            raise ValueError('plan differs from the evaluation role')
    except (OSError, ValueError, TypeError, KeyError, tarfile.TarError) as error:
        raise TaskConfigurationError('experiment plan dataset is invalid for this role') from error
    return expected


def _prepare(deadline: float):
    """Build the trusted tools, the reference and the expansion baseline once per task version."""
    state = state_root()
    fingerprint = signature()
    tools_marker = state / 'tools.json'
    if tools_marker.exists() and json.loads(tools_marker.read_text()).get('signature') == fingerprint:
        return
    try:
        code = TASK / 'harbor/app' / CODE
        limit = CONFIG['environment']['build_timeout_seconds']
        command(['go', 'build', '-trimpath', '-o', state / 'policy', './policy'], code, deadline, cap=limit)
        command(['go', 'build', '-trimpath', '-o', state / 'conductor', './conductor'], code, deadline, cap=limit)
        for label, patch in (('reference', 'reference.patch'), ('naive', 'baseline_naive.patch')):
            tree, cleanup = apply(TASK / patch)
            try:
                check_protected(tree)
                check_source_policy(tree, deadline)
                build(tree, state / label, deadline)
            finally:
                shutil.rmtree(cleanup, ignore_errors=True)
    except Rejected as error:
        raise TaskConfigurationError('trusted reference build failed: ' + str(error)) from error
    tools_marker.write_text(json.dumps({'signature': fingerprint}))


def set_memory_page_policy():
    # Inherited across fork/exec: every client, build and replay uses one policy.
    set_thp_disable, get_thp_disable = 41, 42
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
        prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
        prctl.restype = ctypes.c_int
        if prctl(set_thp_disable, 1, 0, 0, 0) != 0 or prctl(get_thp_disable, 0, 0, 0, 0) != 1:
            raise TaskConfigurationError('required memory page policy unavailable')
    except (AttributeError, OSError) as error:
        raise TaskConfigurationError('required memory page policy unavailable') from error


def setup_process():
    os.umask(0o077)
    set_memory_page_policy()
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:THREADS])
    temporary = state_root() / 'tmp'
    temporary.mkdir(exist_ok=True)
    tempfile.tempdir = str(temporary)


def prepare() -> dict:
    setup_process()
    _prepare(time.monotonic() + LIMIT)
    return {'ready': True}


def failure_metrics(final: bool):
    experiments = len(SIZES) * (2 if final else 1)
    return {
        'beats_reference': 0, 'reference_score': 0.0, 'margin': 0.0,
        'beat_spec_suite_role': 'final' if final else 'train',
        'beat_spec_workload_passes': [
            {'workload': f'e{index:05d}', 'metrics': {name: False for name in GATED_METRICS}}
            for index in range(experiments)],
        **{name: -1 for name in VISIBLE},
    }


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    final = os.environ.get('FRONTIER_SUBMISSION_ROLE') == 'final'
    metrics = failure_metrics(final)
    cleanup = None
    work = None
    started = time.monotonic()
    deadline = started + LIMIT
    try:
        setup_process()
        plan = experiment_plan(final)
        _prepare(deadline)
        state = state_root()
        tree, cleanup = apply(Path(solution_path))
        check_protected(tree)
        check_source_policy(tree, deadline)
        work = Path(tempfile.mkdtemp(prefix='evaluate-', dir=state))
        candidate = work / 'candidate'
        build(tree, candidate, deadline)
        binaries = {'candidate': candidate, 'reference': state / 'reference', 'naive': state / 'naive'}
        candidate_times, reference_times, naive_times = [], [], []
        rows, private_measurements, private_experiments = [], [], []
        score_rows = {'candidate': [], 'reference': [], 'naive': []}
        public = {'online_ms': [], 'preprocessing_s': {}, 'communication': {}, 'storage': {}}
        for ordinal, size in enumerate(plan):
            seeds = fresh_seeds()
            directory = work / f'experiment-{ordinal}'
            directory.mkdir()
            results = {}
            order = ('candidate', 'reference') if ordinal % 2 == 0 else ('reference', 'candidate')
            for implementation in order:
                try:
                    results[implementation] = integration(binaries[implementation], size, seeds,
                                                          directory / implementation, deadline,
                                                          skip_volume=implementation == 'reference')
                except Rejected as error:
                    if implementation == 'reference' and error.category != 'time_limit':
                        raise TaskConfigurationError('paired reference experiment failed') from error
                    raise
            # The reference transcript is checked word for word against the expanded-set oracle.
            try:
                replay(state / 'reference', size, seeds['db_seed'], directory / 'reference/transcript.json',
                       directory / 'reference/valid.json', deadline, validate=True)
                expected = reference_message_bytes(size)
                if (set(results['reference']['upload_bytes']) != {expected[0]}
                        or set(results['reference']['download_bytes']) != {expected[1]}):
                    raise Rejected('reference message sizes differ from the tight encoding', 'server_replay')
            except Rejected as error:
                if error.category != 'time_limit':
                    raise TaskConfigurationError('trusted reference transcript failed validation') from error
                raise
            failure = validity_failure(results['candidate'], results['reference'])
            if failure is not None:
                raise Rejected('paired validity check failed: ' + failure,
                               'hint_export' if failure == 'hint_export_mismatch' else 'process_memory')
            # Coverage probe: the additional queries after the measured ones, identical
            # in both runs. It is a validity condition and enters no metric.
            try:
                failure = probe_failure(results['candidate'], results['reference'])
            except ValueError as error:
                raise TaskConfigurationError('paired coverage probes differ') from error
            if failure is not None:
                raise Rejected('additional query failed', 'coverage_probe', details={
                    'workload': f'e{ordinal:05d}', 'size': size, 'check': failure,
                    'probe_queries': results['candidate']['probe_queries'],
                    'candidate_answered': results['candidate']['probe_answered'],
                    'reference_answered': results['reference']['probe_answered'],
                    'reference_check': results['reference']['probe_failure']})
            # Primary-hint audit after the online phase, same sample in both runs.
            try:
                failure = audit_failure(results['candidate'], results['reference'])
            except ValueError as error:
                raise TaskConfigurationError('paired primary-hint audits differ') from error
            if failure is not None:
                raise Rejected('primary-hint audit failed', 'hint_audit', details={
                    'workload': f'e{ordinal:05d}', 'size': size, 'check': failure,
                    'audit_hints': results['candidate']['audit_hints'],
                    'candidate_mismatched': len(results['candidate']['audit_mismatched']),
                    'reference_mismatched': len(results['reference']['audit_mismatched'])})
            measurements = {}
            for implementation in (*order, 'naive'):
                source = 'candidate' if implementation == 'candidate' else 'reference'
                try:
                    measurements[implementation] = replay(
                        binaries[implementation], size, seeds['db_seed'], directory / source / 'transcript.json',
                        directory / f'{implementation}.json', deadline)
                    if (measurements[implementation]['upload'] != results[source]['upload_bytes']
                            or measurements[implementation]['download'] != results[source]['download_bytes']):
                        raise Rejected('replayed message sizes differ from the experiment', 'server_replay')
                except Rejected as error:
                    if implementation != 'candidate' and error.category != 'time_limit':
                        raise TaskConfigurationError('trusted reference replay failed') from error
                    raise
            for index in range(QUERY_COUNT):
                candidate_ns = measurements['candidate']['median_ns'][index]
                reference_ns = measurements['reference']['median_ns'][index]
                candidate_bytes = results['candidate']['upload_bytes'][index] + results['candidate']['download_bytes'][index]
                reference_bytes = results['reference']['upload_bytes'][index] + results['reference']['download_bytes'][index]
                private_measurements.append({
                    'workload': f'w{len(candidate_times):05d}',
                    'candidate_ns': measurements['candidate']['times'][index],
                    'reference_ns': measurements['reference']['times'][index],
                    'naive_ns': measurements['naive']['times'][index],
                    'candidate_bytes': candidate_bytes, 'reference_bytes': reference_bytes,
                })
                candidate_times.append(candidate_ns)
                reference_times.append(reference_ns)
                naive_times.append(measurements['naive']['median_ns'][index])
            for implementation in score_rows:
                # The expansion baseline only replaces the server handler.
                source = 'candidate' if implementation == 'candidate' else 'reference'
                online_ns, online_queries = score_online(results[source])
                storage = (storage_peak(results['candidate']) if source == 'candidate' else
                           max(results['reference']['client_storage_bytes'],
                               reference_volume_storage(results['reference'])))
                score_rows[implementation].append({
                    'db_size': size, 'queries': online_queries,
                    'bytes': [up + down for up, down in zip(results[source]['upload_bytes'],
                                                            results[source]['download_bytes'])],
                    'server_ns': measurements[implementation]['median_ns'],
                    'offline_ns': results[source]['offline_elapsed_ns'],
                    'online_ns': online_ns, 'storage': storage})
            experiment_id = f'e{ordinal:05d}'
            rows.append({'workload': experiment_id,
                         'metrics': compare_experiment(results['candidate'], results['reference'],
                                                       measurements['candidate']['median_ns'],
                                                       measurements['reference']['median_ns'])})
            summary = {}
            for implementation in ('candidate', 'reference'):
                record = {key: value for key, value in results[implementation].items()
                          if key not in ('upload_bytes', 'download_bytes')}
                record['values'] = timing_values(results[implementation])
                record['upload_bytes_per_query'] = statistics.mean(results[implementation]['upload_bytes'])
                record['download_bytes_per_query'] = statistics.mean(results[implementation]['download_bytes'])
                summary[implementation] = record
            private_experiments.append({
                'workload': experiment_id, 'size': size, **seeds,
                'transcript_identical': results['candidate']['transcript_sha256'] == results['reference']['transcript_sha256'],
                **summary})
            if ordinal < len(SIZES):
                label = 'large'
                public['online_ms'].append(summary['candidate']['values']['end_to_end_online_time_ms'])
                public['preprocessing_s'][label] = results['candidate']['offline_elapsed_ns'] / 1e9
                public['communication'][label] = (summary['candidate']['upload_bytes_per_query']
                                                  + summary['candidate']['download_bytes_per_query'])
                public['storage'][label] = storage_peak(results['candidate'])
            shutil.rmtree(directory, ignore_errors=True)
        try:
            score = continuous_score(score_rows['candidate'])
            reference_score = continuous_score(score_rows['reference'])
            naive_score = continuous_score(score_rows['naive'])
        except ValueError as error:
            raise Rejected('invalid score measurements', 'server_replay') from error
        metrics.update({
            'beats_reference': int(all(decision for row in rows for decision in row['metrics'].values())),
            'beat_spec_workload_passes': rows, 'reference_score': reference_score, 'margin': score-reference_score,
            'naive_score': naive_score,
            'elapsed_seconds': time.monotonic()-started,
            'private_workload_measurements': private_measurements,
            'private_experiment_measurements': private_experiments,
        })
        if not final:
            metrics.update({
                'candidate_public_server_compute_time_ns_median': statistics.median(candidate_times),
                'candidate_public_online_time_ms_median': statistics.median(public['online_ms']),
                'candidate_public_preprocessing_seconds_large': public['preprocessing_s']['large'],
                'candidate_public_online_communication_bytes_large': public['communication']['large'],
                'candidate_public_client_storage_bytes_large': public['storage']['large'],
                'candidate_public_validity_code': VALIDITY_CODES['valid'],
            })
        return score, score, 'All validity checks passed.', metrics
    except Rejected as error:
        if not final:
            metrics['candidate_public_validity_code'] = VALIDITY_CODES.get(error.category, 1)
        metrics['private_rejection'] = {'category': error.category, 'message': str(error)}
        if error.details is not None:
            metrics['private_rejection']['details'] = error.details
        return 0.0, 0.0, str(error), metrics
    except TaskConfigurationError:
        raise
    finally:
        for path in (cleanup, work):
            if path is not None:
                shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--prepare':
        print(json.dumps(prepare()))
    elif len(sys.argv) == 2:
        print(json.dumps(evaluate(sys.argv[1]), allow_nan=False))
    else:
        print('usage: evaluator.py solution.patch | --prepare', file=sys.stderr)
        sys.exit(2)
