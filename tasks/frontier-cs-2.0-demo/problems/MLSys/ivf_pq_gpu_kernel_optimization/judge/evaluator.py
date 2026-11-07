"""Evaluator for GPU IVF-PQ search kernel optimization (one H100, SIFT-128).

Each selected workload runs the frozen naive search and then the optimized reference or the
candidate (`tree`: the starter with the submission overlaid) on the same fresh index and queries,
two warmups and seven timed calls. Train: the four public workloads. Final: the six original
workloads scored, plus two strict-only v3 workloads (PASS only). Any candidate timed call that
fails output validation or iso-result recall >= 0.99 makes the result 0.

raw_score = score_unbounded: S = geometric mean over the scored workloads of the median paired
naive/candidate time ratio; 0 if S <= 1, else 100*log(S)/(log(10)+log(S)) (midpoint from
scoring_config.json).

Metrics per workload (candidate vs reference; all must pass):
  search_latency_ms  median search latency, lower; passes when candidate < reference.
  iso_recall         worst timed recall against the full-nprobe naive ADC top-k, higher; passes
                     when candidate >= reference.
  nn_recall          worst timed exact nearest-neighbor recall@k, higher; passes when
                     candidate >= reference.
beats_reference = 1 when every metric passes on every selected workload (a conjunction).
check_tree rejects (content_policy) external optimized imports in added lines; task.toml
[submission] holds the editable surface. Every worker (naive + reference, naive + candidate)
runs under the shared isolate launcher: one fresh instance per workload with the GPU and the SIFT file,
no judge file, no network, nothing kept.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import isolate  # noqa: E402  (/opt/frontier, on the import path by frontier.pth)

TASK_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TASK_DIR / "scoring_config.json"
REFERENCE_PATCH = TASK_DIR / "reference.patch"
PRISTINE = TASK_DIR / "pristine"  # the starter repo; the shared verifier overlays the submission
STARTER = PRISTINE / "ivfpqlib"
WORKER_DIR = PRISTINE / "benchmark"  # outside the editable surface: the judge's trusted copy
WORKER = WORKER_DIR / "ivfpq_worker.py"  # runs one tree and workload in an isolate instance
ORACLE = TASK_DIR / "refivfpq.py"  # the frozen naive search, timed beside every implementation


def _load_worker():
    """The measurement worker from the judge's own pristine copy, loaded by path: never from
    the candidate tree and never through sys.path (the shared verifier restores sys.path after
    importing this module, so a deferred `import ivfpq_worker` would not resolve)."""
    spec = importlib.util.spec_from_file_location("frontier_ivfpq_worker", WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("IVF verifier bundle has no benchmark/ivfpq_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ivfpq_worker = _load_worker()
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


def _worker_options(config: dict[str, Any]) -> dict[str, Any]:
    """The isolate knobs of one worker: the GPU, the SIFT file (read in place), the
    toolchain environment, the time bound. Reference and candidate runs are set up alike."""
    allowed = ("LD_LIBRARY_PATH", "LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CC", "CXX")
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment.update({
        "TRITON_CACHE_DIR": "{tmp}/triton-cache", "PYTHONPATH": "{work}",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    })
    return {"devices": ["nvidia"], "env": environment, "dev_shm": True,
            "read_paths": [config["data"]["path"]],
            "path": os.environ.get("PATH", isolate.DEFAULT_PATH),
            "timeout_seconds": int(config["worker_timeout_seconds"])}


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
    passes, details, samples = [], [], []
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
            actual = ivfpq_worker.summarize_samples(wid, row['timed_samples'])
            if any(actual[key] != row[key] for key in ('search_latency_ms', 'naive_latency_ms', 'iso_recall', 'nn_recall', 'speedup')):
                raise ValueError('measurement aggregates disagree with the raw samples')
            if actual['iso_recall'] < float(config['recall_threshold']):
                raise ValueError('invalid recall cannot enter score aggregation')
        decisions = {'search_latency_ms': cand['search_latency_ms'] < ref['search_latency_ms'],
                     'iso_recall': cand['iso_recall'] >= ref['iso_recall'],
                     'nn_recall': cand['nn_recall'] >= ref['nn_recall']}
        passes.append({'workload': wid, 'metrics': decisions})
        # Evidence stays shallow (the shared verifier bounds metrics nesting): per-workload
        # aggregates here, every timed sample as one flat row of `timed_samples`.
        details.append({'workload': wid, 'metrics': decisions, **{
            role: {key: value for key, value in row.items() if key != 'timed_samples'}
            for role, row in (('reference', ref), ('candidate', cand))}})
        samples.extend({'workload': wid, 'role': role, 'iteration': iteration, **sample}
                       for role, row in (('reference', ref), ('candidate', cand))
                       for iteration, sample in enumerate(row['timed_samples']))
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
        'timed_samples': samples,
        'calibration_status': config['calibration_status'],
    }


def _measure(source_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """One isolate instance per workload: `source_root` (the tree) is its writable work dir;
    the worker and the frozen naive oracle (refivfpq.py, imported from {in}) are its inputs."""
    rows = []
    for workload in config['workloads']:
        worker_config = {name: config[name] for name in ('data', 'distance_check', 'recall_threshold', 'warmup_iterations', 'timed_iterations')}
        command = [os.path.realpath(sys.executable), '{in}/ivfpq_worker.py', '{work}', '{in}',
                   json.dumps(workload), json.dumps(worker_config), '{out}/result.json']
        result = isolate.run(command, tree=source_root, outputs=['result.json'],
                                  inputs={'ivfpq_worker.py': WORKER, 'refivfpq.py': ORACLE},
                                  **_worker_options(config))
        output = result.outputs.get('result.json')
        if not result.ok or output is None:
            raise RuntimeError('IVF GPU worker failed; ' + result.stderr[-700:].decode('utf-8', errors='replace'))
        row = json.loads(output)
        if row.get('id') != workload['id'] or type(row.get('valid')) is not bool:
            raise RuntimeError('IVF GPU worker returned an invalid result envelope')
        rows.append(row)
    return rows


def evaluate(tree: str) -> tuple[float, float, str, dict[str, Any]]:
    """`tree` is the starter with only the submission's allowed files overlaid (shared verifier)."""
    if not STARTER.is_dir() or not REFERENCE_PATCH.is_file():
        raise RuntimeError('IVF verifier source bundle is incomplete')
    config = select_suite(json.loads(CONFIG_PATH.read_text()), os.environ.get('FRONTIER_SUBMISSION_ROLE', ''))
    ivfpq_worker.validate_distance_config(config['distance_check'])
    ivfpq_worker.verified_data_path(config['data'])
    with tempfile.TemporaryDirectory(prefix='frontier-ivfpq-') as directory:
        scratch = Path(directory)
        roots = {kind: scratch / kind for kind in ('reference', 'candidate')}
        shutil.copytree(STARTER.parent, roots['reference'])
        shutil.copytree(Path(tree), roots['candidate'])
        _apply_patch(roots['reference'], REFERENCE_PATCH)
        reference = _measure(roots['reference'], config)
        if any(row['valid'] is not True for row in reference):
            raise RuntimeError('Frozen optimized reference failed its validity gate')
        candidate = _measure(roots['candidate'], config)
        if any(row['valid'] is not True for row in candidate):
            return 0.0, 0.0, 'Submission failed the search output or recall validity gate', {}
        score, metrics = compare_measurements(reference, candidate, config)
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
