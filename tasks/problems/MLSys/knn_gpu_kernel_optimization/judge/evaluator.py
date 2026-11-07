"""Evaluator for GPU brute-force k-NN kernel optimization (one H100).

Each selected workload samples fresh SIFT/GIST rows per iteration and runs the frozen naive
search, the optimized reference and the candidate (`tree`: the starter with the submission
overlaid) with two warmups and seven timed iterations. Train: the four public workloads. Final:
the six original workloads scored, plus two strict-only v3 workloads (PASS only).

raw_score = score_unbounded: S = geometric mean over the scored workloads of the median paired
naive/candidate time ratio; 0 if any timed ball-recall < 0.88 or S <= 1, else
100*log(S)/(log(10)+log(S)).

Metrics per workload (candidate vs reference; all must pass):
  latency_ms            median call latency, lower; passes when candidate < reference.
  worst_ball_recall     worst timed ball-recall, higher; passes when candidate >= reference and
                        every timed iteration >= 0.88.
  worst_exact_recall    worst timed exact recall (fp64 k-th radius), higher; passes when
                        candidate >= reference.
  distance_consistency  every returned distance matches the fp64 recomputation within
                        rtol 1e-6; passes when it holds on every iteration.
beats_reference = 1 when every metric passes on every selected workload (a conjunction).
check_tree rejects (content_policy) external optimized imports in added lines; task.toml
[submission] holds the editable surface. Every implementation runs under the shared isolate launcher
(one fresh instance per workload: the GPU, the public data, no judge file, no network, nothing
kept).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TASK_DIR / "scoring_config.json"
REFERENCE_PATCH = TASK_DIR / "reference.patch"
PRISTINE = TASK_DIR / "pristine"  # the starter repo; the shared verifier overlays the submission
STARTER = PRISTINE / "knnlib"
DATA_ROOT = Path("/data")  # the public SIFT/GIST corpora (data_manifest.json)
# What a worker instance gets as read-only input: `evaluator.py --worker` and its imports.
WORKER_FILES = ("evaluator.py", "knn_worker.py", "refknn.py")
WORKER_TIMEOUT_SECONDS = 3600
# Judge-only content rule over each changed file's added lines, unchanged from the previous
# checker (it joins a file's added lines, so a match may span lines).
FORBIDDEN_IMPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:import|from)\s+"
    r"(flashlib|cuml|cudf|cupy|faiss|sklearn|scikit|cutlass|cuvs)\b"
)


def check_tree(tree: str, pristine: str, changes: list) -> tuple[str, list[str]] | None:
    """Shared-verifier hook: no external optimized library imports in the lines a file adds
    (the editable surface itself is task.toml's; isolate bounds what the code can
    reach)."""
    hits = []
    for change in changes:
        if change.status == "D":
            continue
        added = "\n".join(change.added)
        if FORBIDDEN_IMPORT_RE.search(added):
            hits.append(change.path)
    return ("content_policy", hits) if hits else None


def geometric_mean(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('speedups must be positive finite values')
    return math.exp(sum(map(math.log, values)) / len(values))


def score_from_speedup(speedup):
    if not math.isfinite(speedup) or speedup <= 1:
        return 0.0
    value = math.log(speedup)
    return 100 * value / (math.log(10) + value)


def select_suite(config, role):
    if role not in ('development', 'final'):
        raise ValueError('trusted submission role is required')
    mode = 'train' if role == 'development' else 'eval'
    workloads = {w['id']: (index, w) for index, w in enumerate(config['workloads'])}
    suites, domains = config['suites'], config['seed_domains']
    if (set(suites) != {'train', 'eval'} or len(workloads) != len(config['workloads'])
            or any(not s['workload_ids'] or len(s['workload_ids']) != len(set(s['workload_ids'])) for s in suites.values())
            or set(suites['eval']['workload_ids']) != set(workloads)
            or not set(suites['train']['workload_ids']) < set(workloads)):
        raise ValueError('final must cover all workloads and train must be a public subset')
    ids = [wid for domain in domains.values() for wid in domain['workload_ids']]
    if (set(domains) != {'public', 'hidden'} or len(ids) != len(set(ids))
            or set(ids) != set(workloads)
            or domains['public']['workload_ids'] != suites['train']['workload_ids']):
        raise ValueError('seed domains must partition public and hidden workloads')
    seeds = [domain['base_seed'] for domain in domains.values()]
    if any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != 2:
        raise ValueError('public and hidden require distinct seed domains')
    bases = {wid: domain['base_seed'] for domain in domains.values() for wid in domain['workload_ids']}
    return {**config, 'suite_mode': mode, 'workloads': [
        {**workloads[wid][1], 'base_seed': bases[wid], 'seed_index': workloads[wid][0]}
        for wid in suites[mode]['workload_ids']]}


def distance_tolerances(config):
    policy = config['distance_consistency']
    values = [policy.get('absolute_tolerance'), policy.get('relative_tolerance')]
    if (policy.get('status') != 'approved'
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)):
        raise RuntimeError('distance consistency tolerances require reference calibration and user approval')
    return values


def summarize_measurement(row, iterations):
    samples = [row[name] for name in ('naive_latency_ms', 'latency_ms', 'ball_recall', 'exact_recall')]
    if any(len(values) != iterations for values in samples):
        raise ValueError('each workload requires every timed iteration')
    for values in samples[:2]:
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('invalid latency measurement')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in samples[2]):
        raise ValueError('invalid ball-recall measurement')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in samples[3]):
        raise ValueError('invalid exact-recall measurement')
    distance_checks = row['distance_errors']
    if (len(distance_checks) != iterations
            or any(check.get('tolerance_configured') is not True or type(check.get('violations')) is not int or check['violations'] < 0 for check in distance_checks)):
        raise ValueError('every timed iteration requires a configured distance consistency check')
    return {'distance_consistent': all(check['violations'] == 0 for check in distance_checks),
            'latency_ms': statistics.median(samples[1]),
            'worst_ball_recall': min(samples[2]),
            'worst_exact_recall': min(samples[3]),
            'speedup': statistics.median([naive / candidate for naive, candidate in zip(*samples[:2])])}


def compare_measurements(reference, candidate, config):
    ids = [w['id'] for w in config['workloads']]
    tables = [{r['id']: r for r in rows} for rows in (reference, candidate)]
    if any(len(rows) != len(ids) or set(table) != set(ids) for rows, table in zip((reference, candidate), tables)):
        raise ValueError('measurements must cover every selected workload exactly once')
    details, passes, speedups, reference_speedups = [], [], [], []
    valid = True
    # Beat spec v3: strict_only workloads enter PASS but not the continuous score.
    scored = {w['id'] for w in config['workloads'] if not w.get('strict_only')}
    for wid in ids:
        ref = summarize_measurement(tables[0][wid], config['timed_iterations'])
        cand = summarize_measurement(tables[1][wid], config['timed_iterations'])
        if not ref['distance_consistent']:
            raise RuntimeError('frozen optimized reference fails distance consistency')
        if ref['worst_ball_recall'] < config['recall_threshold']:
            raise RuntimeError('frozen optimized reference fails the absolute ball-recall gate')
        gate = cand['worst_ball_recall'] >= config['recall_threshold'] and cand['distance_consistent']
        valid = valid and gate
        decisions = {'latency_ms': cand['latency_ms'] < ref['latency_ms'],
                     'worst_ball_recall': cand['worst_ball_recall'] >= ref['worst_ball_recall'] and cand['worst_ball_recall'] >= config['recall_threshold'],
                     'distance_consistency': cand['distance_consistent'],
                     'worst_exact_recall': cand['worst_exact_recall'] >= ref['worst_exact_recall']}
        passes.append({'workload': wid, 'metrics': decisions})
        details.append({'workload': wid, 'reference': ref, 'candidate': cand,
                        'metrics': decisions, 'absolute_quality_gate': gate,
                        'reference_iterations': tables[0][wid], 'candidate_iterations': tables[1][wid]})
        if wid in scored:
            speedups.append(cand['speedup']); reference_speedups.append(ref['speedup'])
    speedup = geometric_mean(speedups)
    score = score_from_speedup(speedup) if valid else 0.0
    reference_score = score_from_speedup(geometric_mean(reference_speedups))
    return score, {'geomean_speedup': speedup, 'absolute_quality_gate': int(valid),
                   'beats_reference': int(all(v for row in passes for v in row['metrics'].values())),
                   'reference_score': reference_score, 'margin': score - reference_score,
                   'latency_ms': sorted(row['candidate']['latency_ms'] for row in details),
                   'worst_ball_recall': sorted(row['candidate']['worst_ball_recall'] for row in details),
                   'worst_exact_recall': sorted(row['candidate']['worst_exact_recall'] for row in details),
                   'beat_spec_workload_passes': passes, 'comparison_details': details}


def validate_data_manifest(manifest_path, data_root, datasets):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('schema_version') != 'frontier.knn-data-manifest/v1':
        raise RuntimeError('KNN dataset manifest schema is invalid')
    for name in sorted(set(datasets)):
        record = manifest['datasets'][name]
        expected = record.get('sha256')
        if (not isinstance(expected, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', expected)
                or type(record.get('bytes')) is not int or record['bytes'] <= 0):
            raise RuntimeError('KNN datasets are unready: verified size and SHA256 are required')
        path = Path(data_root) / record['filename']
        if path.parent != Path(data_root) or not path.is_file() or path.is_symlink():
            raise RuntimeError('KNN dataset is absent or not a regular local file')
        with path.open('rb') as stream:
            actual = 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != record['bytes'] or actual != expected:
            raise RuntimeError('KNN dataset content does not match its frozen manifest')
    return manifest


def _apply_patch(root, patch):
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=root, check=True,
                   capture_output=True, timeout=60)
    subprocess.run(['git', 'apply', str(patch)], cwd=root, check=True,
                   capture_output=True, timeout=60)


def _isolate():
    """The shared launcher, from gpu-tools. Imported here, not at module level: a worker
    instance runs this file too and need not reach /opt/frontier."""
    import isolate
    return isolate


def _worker_options():
    """The isolate knobs of one worker: the GPU, the public ANN data read in place, the
    toolchain environment. Reference and candidate run alike, so their latencies stay paired."""
    isolate = _isolate()
    env = {key: os.environ[key] for key in ('LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH', 'CC', 'CXX') if key in os.environ}
    env.update(TRITON_CACHE_DIR='{tmp}/triton-cache', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    return {'devices': ['nvidia'], 'env': env, 'path': os.environ.get('PATH', isolate.DEFAULT_PATH),
            'dev_shm': True, 'read_paths': [str(DATA_ROOT)], 'timeout_seconds': WORKER_TIMEOUT_SECONDS}


def _measure(source_root, config):
    """One isolate instance per workload: `source_root` (holding knnlib/) is its writable
    work dir; the worker (this file, knn_worker.py, refknn.py) is read-only input. It sees no
    other judge file, no other implementation's run, no FRONTIER_* variable."""
    isolate = _isolate()
    rows = []
    for workload in config['workloads']:
        command = [os.path.realpath(sys.executable), '{in}/evaluator.py', '--worker', '{work}',
                   json.dumps({**workload, 'distance_tolerances': distance_tolerances(config)}, separators=(',', ':')), str(config['warmup_iterations']),
                   str(config['timed_iterations']), '{out}/result.json']
        result = isolate.run(command, tree=source_root, outputs=['result.json'],
                                  inputs={name: TASK_DIR / name for name in WORKER_FILES},
                                  **_worker_options())
        output = result.outputs.get('result.json')
        if not result.ok or output is None:
            raise RuntimeError('KNN GPU worker failed; private worker evidence retained only locally')
        row = json.loads(output)
        if row.get('id') != workload['id']:
            raise RuntimeError('worker workload identity mismatch')
        summarize_measurement(row, config['timed_iterations'])
        rows.append(row)
    return rows


def evaluate(tree):
    """`tree` is the starter with only the submission's allowed files overlaid (shared verifier)."""
    config = select_suite(json.loads(CONFIG_PATH.read_text()), os.environ.get('FRONTIER_SUBMISSION_ROLE', ''))
    distance_tolerances(config)
    validate_data_manifest(TASK_DIR / 'data_manifest.json', Path('/data'),
                           [w['dataset'] for w in config['workloads']])
    if not STARTER.is_dir() or not REFERENCE_PATCH.is_file():
        raise RuntimeError('Verifier source bundle is incomplete')
    with tempfile.TemporaryDirectory(prefix='frontier-knn-') as directory:
        scratch = Path(directory)
        roots = {kind: scratch / kind for kind in ('reference', 'candidate')}
        shutil.copytree(STARTER.parent, roots['reference'])
        shutil.copytree(Path(tree), roots['candidate'])
        _apply_patch(roots['reference'], REFERENCE_PATCH)
        reference = _measure(roots['reference'], config)
        candidate = _measure(roots['candidate'], config)
        score, metrics = compare_measurements(reference, candidate, config)
    metrics['evaluation_suite'] = {'mode': config['suite_mode'],
        'workload_ids': [w['id'] for w in config['workloads']],
        'workload_base_seeds': {w['id']: w['base_seed'] for w in config['workloads']}}
    return score, score, f'KNN score {score:.2f}/100', metrics


def _worker(argv):
    if len(argv) != 5:
        raise ValueError('--worker requires source, workload, warmups, iterations and output')
    source, workload_json, warmup, iterations, output = argv
    workload = json.loads(workload_json)
    from knn_worker import measure_workload
    row = measure_workload(Path(source), workload, int(warmup), int(iterations),
                           TASK_DIR / 'refknn.py', Path('/data'),
                           distance_policy=workload['distance_tolerances'])
    Path(output).write_text(json.dumps(row, allow_nan=False))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '--worker':
        return _worker(argv[1:])
    if len(argv) != 1:
        raise ValueError('one solution patch is required')
    print(json.dumps(evaluate(argv[0]), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
