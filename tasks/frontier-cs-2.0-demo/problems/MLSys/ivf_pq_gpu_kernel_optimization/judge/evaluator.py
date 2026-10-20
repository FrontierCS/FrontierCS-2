"""Native FrontierCS2 evaluator for GPU IVF-PQ kernel optimization."""
from __future__ import annotations

import fnmatch
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TASK_DIR / "scoring_config.json"
WORKER_DIR = TASK_DIR / "repo" / "benchmark"
sys.path.insert(0, str(WORKER_DIR))
REFERENCE_PATCH = TASK_DIR / "reference.patch"
STARTER = TASK_DIR / "repo" / "ivfpqlib"
MAX_PATCH_BYTES = 2_000_000
MAX_CHANGED_FILES = 80
ALLOWED_PATTERNS = ("ivfpqlib/**",)
DENIED_PATTERNS = (
    "evaluator.py", "reference.patch", "refivfpq.py", "**/refivfpq.py",
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
        return False, f"changed file is outside ivfpqlib: {path}"
    if not path.endswith(".py"):
        return False, f"only Python files may change: {path}"
    return True, ""


def validate_patch(path: Path) -> tuple[bool, str, dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        return False, "solution patch is not a regular file", {}
    size = path.stat().st_size
    metrics: dict[str, Any] = {"patch_bytes": size}
    if size > MAX_PATCH_BYTES:
        return False, "solution patch exceeds 2000000 bytes", metrics
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
    if (size > 0 and not files) or len(files) > MAX_CHANGED_FILES:
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


def geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def score_from_speedup(speedup: float, midpoint: float) -> float:
    if not math.isfinite(speedup) or speedup <= 0:
        return 0.0
    if not math.isfinite(midpoint) or midpoint <= 1.0:
        raise ValueError("continuous score speedup midpoint must be greater than one")
    if speedup <= 1.0:
        return 0.0
    improvement = math.log(speedup)
    return 100.0 * improvement / (math.log(midpoint) + improvement)


def _apply_patch(root: Path, patch: Path) -> None:
    if patch.stat().st_size == 0:
        return
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(root), "LANG": "C", "LC_ALL": "C"}
    for arguments in (["git", "apply", "--check", str(patch)], ["git", "apply", str(patch)]):
        subprocess.run(arguments, cwd=root, env=environment, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)


def _worker_environment(source_root: Path, scratch: Path) -> dict[str, str]:
    allowed = (
        "PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES", "CC", "CXX",
    )
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment.update({
        "HOME": str(scratch), "TMPDIR": str(scratch),
        "TRITON_CACHE_DIR": str(scratch / "triton-cache"),
        "PYTHONPATH": str(source_root), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    })
    return environment


def select_suite(config: dict[str, Any], role: str) -> dict[str, Any]:
    modes = {"development": "train", "final": "eval"}
    if role not in modes:
        raise ValueError("a trusted development or final submission role is required")
    mode = modes[role]
    suites = config["suites"]
    workloads = {row["id"]: row for row in config["workloads"]}
    if (set(suites) != {"train", "eval"}
            or len(workloads) != len(config["workloads"])
            or any(not suite["workload_ids"]
                   or len(suite["workload_ids"]) != len(set(suite["workload_ids"]))
                   for suite in suites.values())
            or set(suites["eval"]["workload_ids"]) != set(workloads)
            or not set(suites["train"]["workload_ids"]) < set(workloads)):
        raise ValueError("final must cover all workloads and train must be a public subset")
    domains = config["seed_domains"]
    domain_ids = [workload_id for domain in domains.values()
                  for workload_id in domain["workload_ids"]]
    if (set(domains) != {"public", "hidden"}
            or any(not domain["workload_ids"] for domain in domains.values())
            or len(domain_ids) != len(set(domain_ids))
            or set(domain_ids) != set(workloads)
            or domains["public"]["workload_ids"] != suites["train"]["workload_ids"]):
        raise ValueError("public and hidden seed domains must partition the workloads")
    seeds = [domain["base_seed"] for domain in domains.values()]
    if any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != 2:
        raise ValueError("public and hidden require distinct deterministic seed domains")
    base_seeds = {workload_id: domain["base_seed"] for domain in domains.values()
                  for workload_id in domain["workload_ids"]}
    # Seed identity belongs to the workload, so public data is identical in both modes.
    return {
        **config,
        "suite_mode": mode,
        "workloads": [{**workloads[workload_id], "base_seed": base_seeds[workload_id]}
                      for workload_id in suites[mode]["workload_ids"]],
    }


def compare_measurements(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    expected = [w['id'] for w in config['workloads']]
    mappings = [{row['id']: row for row in rows} for rows in (reference, candidate)]
    if any(len(rows) != len(expected) or set(table) != set(expected)
           for rows, table in zip((reference, candidate), mappings)):
        raise ValueError('measurements must cover each selected workload exactly once')
    if config['comparisons'] != {
        'search_latency_ms': {'better': 'lower', 'strictly_better': True},
        'iso_recall': {'better': 'higher', 'absolute_non_inferiority_band': 0.0},
        'nn_recall': {'better': 'higher', 'absolute_non_inferiority_band': 0.0},
    }:
        raise ValueError('comparison configuration differs from the approved IVF rules')
    ref_map, cand_map = mappings
    passes, details = [], []
    for wid in expected:
        ref, cand = ref_map[wid], cand_map[wid]
        for row in (ref, cand):
            if row.get('valid') is not True:
                raise ValueError('invalid measurement cannot enter score aggregation')
            if len(row['timed_samples']) != int(config['timed_iterations']):
                raise ValueError('wrong timed repetition count')
            if any(sample.get('distance_errors', {}).get('distance_gate_passed') is not True
                   or sample['distance_errors'].get('mode') != 'enforce'
                   for sample in row['timed_samples']):
                raise ValueError('distance calibration or unchecked samples cannot enter final scoring')
            # Recompute aggregates from paired raw timings, never trust submitted scalar metrics.
            from ivfpq_worker import summarize_samples
            actual = summarize_samples(wid, row['timed_samples'])
            if any(actual[key] != row[key] for key in ('search_latency_ms', 'naive_latency_ms', 'iso_recall', 'nn_recall', 'speedup')):
                raise ValueError('measurement aggregates disagree with the raw samples')
            if actual['iso_recall'] < float(config['recall_threshold']):
                raise ValueError('invalid recall cannot enter score aggregation')
        decisions = {'search_latency_ms': cand['search_latency_ms'] < ref['search_latency_ms'],
                     'iso_recall': cand['iso_recall'] >= ref['iso_recall'],
                     'nn_recall': cand['nn_recall'] >= ref['nn_recall']}
        passes.append({'workload': wid, 'metrics': decisions})
        details.append({'workload': wid, 'reference': ref, 'candidate': cand, 'metrics': decisions})
    # Beat spec v3 workloads marked strict_only enter PASS but not the
    # continuous score, which keeps its original six-workload definition.
    scored = [w['id'] for w in config['workloads'] if not w.get('strict_only')]
    speedup = geometric_mean([cand_map[wid]['speedup'] for wid in scored])
    reference_speedup = geometric_mean([ref_map[wid]['speedup'] for wid in scored])
    midpoint = float(config['continuous_score']['speedup_midpoint'])
    score = score_from_speedup(speedup, midpoint)
    reference_score = score_from_speedup(reference_speedup, midpoint)
    return score, {
        'geomean_speedup': speedup,
        'search_latency_ms': sorted(cand_map[wid]['search_latency_ms'] for wid in expected),
        'iso_recall': sorted(cand_map[wid]['iso_recall'] for wid in expected),
        'nn_recall': sorted(cand_map[wid]['nn_recall'] for wid in expected),
        'beats_reference': int(all(value for row in passes for value in row['metrics'].values())),
        'reference_score': reference_score,
        'margin': score - reference_score,
        'beat_spec_workload_passes': passes,
        'comparison_details': details,
        'calibration_status': config['calibration_status'],
    }


def _measure(source_root: Path, config: dict[str, Any], scratch: Path) -> list[dict[str, Any]]:
    rows = []
    for workload in config['workloads']:
        result_path = scratch / f"result-{source_root.name}-{workload['id']}.json"
        error_path = scratch / f"error-{source_root.name}-{workload['id']}.log"
        worker_config = {name: config[name] for name in ('data', 'distance_check', 'recall_threshold', 'warmup_iterations', 'timed_iterations')}
        command = [sys.executable, str(WORKER_DIR / 'ivfpq_worker.py'), str(source_root), str(TASK_DIR),
                   json.dumps(workload), json.dumps(worker_config), str(result_path)]
        with error_path.open('wb') as stderr:
            completed = subprocess.run(command, env=_worker_environment(source_root, scratch),
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr,
                                       check=False, timeout=int(config['worker_timeout_seconds']))
        if completed.returncode or not result_path.is_file():
            raise RuntimeError('IVF GPU worker failed; ' + error_path.read_bytes()[-700:].decode('utf-8', errors='replace'))
        row = json.loads(result_path.read_text())
        if row.get('id') != workload['id'] or type(row.get('valid')) is not bool:
            raise RuntimeError('IVF GPU worker returned an invalid result envelope')
        rows.append(row)
    return rows


def evaluate(solution_path: str) -> tuple[float, float, str, dict[str, Any]]:
    patch = Path(solution_path)
    valid, message, patch_metrics = validate_patch(patch)
    if not valid:
        return 0.0, 0.0, message, {}
    if not STARTER.is_dir() or not REFERENCE_PATCH.is_file():
        raise RuntimeError('IVF verifier source bundle is incomplete')
    config = select_suite(json.loads(CONFIG_PATH.read_text()), os.environ.get('FRONTIER_SUBMISSION_ROLE', ''))
    from ivfpq_worker import verified_data_path, validate_distance_config
    validate_distance_config(config['distance_check'])
    verified_data_path(config['data'])
    with tempfile.TemporaryDirectory(prefix='frontier-ivfpq-') as directory:
        scratch = Path(directory)
        (scratch / 'triton-cache').mkdir()
        roots = {kind: scratch / kind for kind in ('reference', 'candidate')}
        for root in roots.values():
            shutil.copytree(STARTER.parent, root)
        _apply_patch(roots['reference'], REFERENCE_PATCH)
        try:
            _apply_patch(roots['candidate'], patch)
        except subprocess.CalledProcessError:
            return 0.0, 0.0, 'Submission patch does not apply to the clean package tree', {}
        reference = _measure(roots['reference'], config, scratch)
        if any(row['valid'] is not True for row in reference):
            raise RuntimeError('Frozen optimized reference failed its validity gate')
        candidate = _measure(roots['candidate'], config, scratch)
        if any(row['valid'] is not True for row in candidate):
            return 0.0, 0.0, 'Submission failed the search output or recall validity gate', {}
        score, metrics = compare_measurements(reference, candidate, config)
    metrics.update(patch_metrics)
    metrics['evaluation_suite'] = {
        'mode': config['suite_mode'], 'workload_ids': [w['id'] for w in config['workloads']],
        'workload_seeds': {w['id']: w['base_seed'] + 1000*(w['ordinal']+1) for w in config['workloads']},
        'data_sha256': config['data']['sha256'],
    }
    return score, score, f"IVF-PQ score {score:.2f}/100; independent speedup {metrics['geomean_speedup']:.4f}x", metrics


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise ValueError('expected one solution patch path')
    score, unbounded, message, metrics = evaluate(argv[0])
    print(json.dumps({'score': score, 'score_unbounded': unbounded, 'message': message, 'metrics': metrics}, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
