"""Native FrontierCS2 evaluator for GPU KNN kernel optimization."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import importlib.util
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TASK_DIR / "scoring_config.json"
REFERENCE_PATCH = TASK_DIR / "reference.patch"
STARTER = TASK_DIR / "repo" / "knnlib"
MAX_PATCH_BYTES = 2_000_000
MAX_CHANGED_FILES = 80
ALLOWED_PATTERNS = ("knnlib/**",)
DENIED_PATTERNS = (
    "evaluator.py", "reference.patch", "refknn.py", "**/refknn.py",
    "**/conftest.py", "pyproject.toml", "setup.py",
    "setup.cfg",
)
FORBIDDEN_IMPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:import|from)\s+"
    r"(flashlib|cuml|cudf|cupy|faiss|sklearn|scikit|cutlass|cuvs)\b"
)
FORBIDDEN_PATTERN_RE = re.compile(
    r"\bsubprocess\b|\bsocket\b|\burllib\b|\brequests\b|os\.system|"
    r"torch\.cuda\.synchronize\s*=|time\.perf_counter\s*=|"
    r"setattr\(\s*torch\.cuda"
)


@dataclass(frozen=True)
class PatchFile:
    old_path: str
    new_path: str
    added_lines: tuple[str, ...]

    @property
    def path(self) -> str:
        return self.new_path if self.new_path != "/dev/null" else self.old_path


def _match(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _parse_patch(text: str) -> list[PatchFile]:
    files: list[PatchFile] = []
    old = new = ""
    added: list[str] = []
    active = False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            if active:
                files.append(PatchFile(old, new, tuple(added)))
            active = True
            header = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", line)
            if header is None:
                raise ValueError("unsupported patch header")
            old, new = header.groups()
            for path in (old, new):
                ok, message = _validate_path(path)
                if not ok:
                    raise ValueError(message)
            added = []
        elif active and line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            raise ValueError("rename/copy patch operations are not supported")
        elif active and line.startswith("deleted file mode "):
            new = "/dev/null"
        elif active and line.startswith("--- "):
            old = line[4:].strip()
            old = old[2:] if old.startswith("a/") else old
        elif active and line.startswith("+++ "):
            new = line[4:].strip()
            new = new[2:] if new.startswith("b/") else new
        elif active and line.startswith("+") and not line.startswith("+++ "):
            added.append(line[1:])
    if active:
        files.append(PatchFile(old, new, tuple(added)))
    return files


def _validate_path(path: str) -> tuple[bool, str]:
    if not path or path == "/dev/null":
        return True, ""
    if path.startswith("/") or ".." in Path(path).parts:
        return False, f"unsafe patch path: {path}"
    if _match(path, DENIED_PATTERNS):
        return False, f"reserved or trusted file may not change: {path}"
    if not _match(path, ALLOWED_PATTERNS):
        return False, f"changed file is outside knnlib: {path}"
    if not path.endswith(".py"):
        return False, f"only Python files may change: {path}"
    return True, ""


def validate_patch(path: Path) -> tuple[bool, str, dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        return False, "solution patch is not a regular file", {}
    size = path.stat().st_size
    metrics: dict[str, Any] = {"patch_bytes": size}
    if size <= 0 or size > MAX_PATCH_BYTES:
        return False, "solution patch is empty or exceeds 2000000 bytes", metrics
    body = path.read_bytes()
    if (b"GIT binary patch" in body or b"\x00" in body
            or any(line.startswith(b"Binary files ") for line in body.splitlines())):
        return False, "binary patches are not allowed", metrics
    text = body.decode("utf-8", errors="replace")
    try:
        files = _parse_patch(text)
    except ValueError as error:
        return False, str(error), metrics
    metrics.update(
        patch_sha256="sha256:" + hashlib.sha256(body).hexdigest(),
        changed_files=len(files),
    )
    if not files or len(files) > MAX_CHANGED_FILES:
        return False, "patch has no changes or too many changed files", metrics
    for item in files:
        if item.new_path == "/dev/null":
            return False, f"deleting files is outside the task boundary: {item.old_path}", metrics
        if item.old_path != "/dev/null" and item.old_path != item.new_path:
            ok, error = _validate_path(item.old_path)
            if not ok:
                return False, error, metrics
        ok, error = _validate_path(item.path)
        if not ok:
            return False, error, metrics
        added = "\n".join(item.added_lines)
        forbidden_import = FORBIDDEN_IMPORT_RE.search(added)
        if forbidden_import:
            return False, f"forbidden external optimized import: {forbidden_import.group(1)}", metrics
        forbidden = FORBIDDEN_PATTERN_RE.search(added)
        if forbidden:
            return False, f"forbidden process, network, or timing pattern: {forbidden.group(0)}", metrics
    metrics["valid_patch"] = 1
    return True, "patch accepted by static policy", metrics



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


def _measure(source_root, config, scratch):
    rows = []
    environment = {key: os.environ[key] for key in ('PATH', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'CC', 'CXX') if key in os.environ}
    environment.update(HOME=str(scratch), TMPDIR=str(scratch), TRITON_CACHE_DIR=str(scratch / 'triton-cache'),
                       PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', LANG='C.UTF-8', LC_ALL='C.UTF-8')
    for workload in config['workloads']:
        output = scratch / f'{source_root.name}-{workload["id"]}.json'
        error_path = scratch / f'{source_root.name}-{workload["id"]}.log'
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(source_root),
                   json.dumps({**workload, 'distance_tolerances': distance_tolerances(config)}, separators=(',', ':')), str(config['warmup_iterations']),
                   str(config['timed_iterations']), str(output)]
        with error_path.open('wb') as stream:
            result = subprocess.run(command, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=stream, timeout=3600, check=False)
        if result.returncode != 0 or not output.is_file():
            raise RuntimeError('KNN GPU worker failed; private worker evidence retained only locally')
        row = json.loads(output.read_text())
        if row.get('id') != workload['id']:
            raise RuntimeError('worker workload identity mismatch')
        summarize_measurement(row, config['timed_iterations'])
        rows.append(row)
    return rows


def evaluate(solution_path):
    patch = Path(solution_path)
    valid, message, patch_metrics = validate_patch(patch)
    if not valid:
        return 0.0, 0.0, message, {}
    config = select_suite(json.loads(CONFIG_PATH.read_text()), os.environ.get('FRONTIER_SUBMISSION_ROLE', ''))
    distance_tolerances(config)
    validate_data_manifest(TASK_DIR / 'data_manifest.json', Path('/data'),
                           [w['dataset'] for w in config['workloads']])
    if not STARTER.is_dir() or not REFERENCE_PATCH.is_file():
        raise RuntimeError('Verifier source bundle is incomplete')
    with tempfile.TemporaryDirectory(prefix='frontier-knn-') as directory:
        scratch = Path(directory)
        roots = {kind: scratch / kind for kind in ('reference', 'candidate')}
        for kind, root in roots.items():
            shutil.copytree(STARTER.parent, root)
        _apply_patch(roots['reference'], REFERENCE_PATCH)
        try:
            _apply_patch(roots['candidate'], patch)
        except subprocess.CalledProcessError:
            return 0.0, 0.0, 'Submission patch does not apply to the clean package tree', {}
        reference = _measure(roots['reference'], config, scratch)
        candidate = _measure(roots['candidate'], config, scratch)
        score, metrics = compare_measurements(reference, candidate, config)
    metrics.update(patch_metrics)
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
