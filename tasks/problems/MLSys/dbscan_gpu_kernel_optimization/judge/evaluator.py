"""Evaluator for GPU DBSCAN kernel optimization (one H100).

Each selected workload runs the frozen naive starter, the optimized reference and the candidate
(`tree`: the starter with the submission overlaid) with two warmups and three timed
repetitions on fresh inputs. Train: public w0-w1. Final: w0-w3 scored, plus strict-only
w4-w5 (PASS only).

raw_score: S = geometric mean over the scored workloads of median(naive_i / candidate_i) per
workload; 0 if any timed candidate ARI < 0.99 or S <= 1, else 100*log(S)/(log(10)+log(S)).
score_unbounded = S (geomean_speedup).

Metrics per workload (candidate vs reference; all three must pass):
  dbscan_latency_ms        median call latency, lower; passes when candidate < reference.
  adjusted_rand_index      minimum timed ARI against the naive labels, higher; passes when
                           candidate >= reference and every candidate ARI >= 0.99.
  noise_mismatch_fraction  maximum timed noise-mask mismatch against naive, lower; passes when
                           candidate <= reference (zero if the reference is zero).
beats_reference = 1 when every metric passes on every selected workload (a conjunction). A
reference missing the 0.99 ARI gate is an evaluator error, not a comparison.
check_tree rejects (content_policy) external optimized imports in added lines; task.toml
[submission] holds the editable surface. Every implementation runs under the shared isolate launcher
(one fresh instance per workload: the GPU, no judge file, no network, nothing kept).
"""
from __future__ import annotations

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
from typing import Any

from checker import adjusted_rand_index, noise_mask_mismatch, validate_labels

import isolate  # noqa: E402  (/opt/frontier, on the import path by frontier.pth)

TASK_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TASK_DIR / "scoring_config.json"
REFERENCE_PATCH = TASK_DIR / "reference.patch"
PRISTINE = TASK_DIR / "pristine"  # the starter repo; the shared verifier overlays the submission
STARTER = PRISTINE / "dbscanlib"
WORKER = TASK_DIR / "worker.py"  # runs one implementation and workload in an isolate instance
# Judge-only content rule over each changed file's added lines, unchanged from the previous
# checker (they join a file's added lines, so a match may span lines; task.toml cannot say that).
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


def geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("speedups must be finite and positive")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def score_from_speedup(speedup: float) -> float:
    if not math.isfinite(speedup) or speedup <= 0:
        raise ValueError("speedup must be finite and positive")
    if speedup <= 1:
        return 0.0
    improvement = math.log(speedup)
    return 100.0 * improvement / (math.log(10.0) + improvement)


def _validate_samples(row: dict, workload: dict, iterations: int) -> None:
    if row.get("id") != workload["id"] or len(row.get("samples", [])) != iterations:
        raise ValueError("measurement does not cover every required timed iteration")
    for sample in row["samples"]:
        latency = sample.get("latency_ms")
        if type(latency) not in (float, int) or not math.isfinite(latency) or latency <= 0:
            raise ValueError("latency must be finite and strictly positive")
        validate_labels(sample.get("labels"), workload["N"])


def compare_measurements(naive, reference, candidate, config):
    workloads = config["workloads"]
    ids = {w["id"] for w in workloads}
    maps = [{row["id"]: row for row in rows} for rows in (naive, reference, candidate)]
    if any(set(table) != ids or len(table) != len(rows)
           for table, rows in zip(maps, (naive, reference, candidate))):
        raise ValueError("measurements must cover exactly the selected workloads")
    iterations = config["timed_iterations"]
    threshold = config["correctness"]["minimum_adjusted_rand_index"]
    speedups, reference_speedups, passes, details = [], [], [], []
    candidate_correct = True
    # Beat spec v3: strict_only workloads enter PASS but not the continuous
    # score, which keeps its original definition over the other workloads.
    scored = {w["id"] for w in workloads if not w.get("strict_only")}
    for workload in workloads:
        ident = workload["id"]
        rows = [table[ident] for table in maps]
        for row in rows:
            _validate_samples(row, workload, iterations)
        baseline, ref, cand = [row["samples"] for row in rows]
        # Preserve the legacy median of paired per-iteration speedups.
        speedup = statistics.median(n["latency_ms"] / c["latency_ms"] for n, c in zip(baseline, cand))
        reference_speedup = statistics.median(n["latency_ms"] / r["latency_ms"] for n, r in zip(baseline, ref))
        if ident in scored:
            speedups.append(speedup)
            reference_speedups.append(reference_speedup)
        quality = []
        for samples in (ref, cand):
            aris = [adjusted_rand_index(s["labels"], n["labels"])
                    for s, n in zip(samples, baseline)]
            noise = [noise_mask_mismatch(s["labels"], n["labels"])
                     for s, n in zip(samples, baseline)]
            quality.append({
                "dbscan_latency_ms": statistics.median(s["latency_ms"] for s in samples),
                "adjusted_rand_index": min(aris),
                "noise_mismatch_fraction": max(noise),
                "iteration_latency_ms": [s["latency_ms"] for s in samples],
                "iteration_adjusted_rand_index": aris,
                "iteration_noise_mismatch_fraction": noise,
            })
        ref_metrics, cand_metrics = quality
        if ref_metrics["adjusted_rand_index"] < threshold:
            # Beat spec v3: an incorrect reference is an evaluator error; it
            # never sets the comparison bar or zeroes the reference score.
            raise RuntimeError("frozen optimized reference fails the original ARI correctness gate")
        candidate_correct &= cand_metrics["adjusted_rand_index"] >= threshold
        decisions = {
            "dbscan_latency_ms": cand_metrics["dbscan_latency_ms"] < ref_metrics["dbscan_latency_ms"],
            "adjusted_rand_index": (cand_metrics["adjusted_rand_index"] >= threshold
                                   and cand_metrics["adjusted_rand_index"] >= ref_metrics["adjusted_rand_index"]),
            "noise_mismatch_fraction": cand_metrics["noise_mismatch_fraction"] <= ref_metrics["noise_mismatch_fraction"],
        }
        passes.append({"workload": ident, "metrics": decisions})
        details.append({"workload": ident, "naive_iteration_latency_ms": [s["latency_ms"] for s in baseline],
                        "reference": ref_metrics, "candidate": cand_metrics,
                        "speedup": speedup, "reference_speedup": reference_speedup,
                        "metrics": decisions})
    geomean = geometric_mean(speedups)
    reference_geomean = geometric_mean(reference_speedups)
    score = score_from_speedup(geomean) if candidate_correct else 0.0
    reference_score = score_from_speedup(reference_geomean)
    metrics = {
        "geomean_speedup": geomean,
        "dbscan_latency_ms": sorted(row["candidate"]["dbscan_latency_ms"] for row in details),
        "adjusted_rand_index": sorted(row["candidate"]["adjusted_rand_index"] for row in details),
        "noise_mismatch_fraction": sorted(row["candidate"]["noise_mismatch_fraction"] for row in details),
        "beats_reference": int(all(value for row in passes for value in row["metrics"].values())),
        "reference_score": reference_score,
        "margin": score - reference_score,
        "beat_spec_workload_passes": passes,
        "comparison_details": details,
        "legacy_ari_gate_passed": bool(candidate_correct),
        "reference_legacy_ari_gate_passed": True,
    }
    return score, metrics


def _apply_patch(root: Path, patch: Path) -> None:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(root), "LANG": "C", "LC_ALL": "C"}
    for args in (["git", "apply", "--check", str(patch)], ["git", "apply", str(patch)]):
        subprocess.run(args, cwd=root, env=env, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=60)


def _result_bound(config: dict, workload: dict) -> int:
    return workload["N"] * config["timed_iterations"] * 24 + 4096


def _worker_options(config: dict, workload: dict) -> dict:
    """The isolate knobs of one worker: the GPU, the toolchain environment, the time and
    output bounds. Naive, reference and candidate run alike, so their latencies stay paired."""
    env = {name: os.environ[name] for name in (
        "LD_LIBRARY_PATH", "LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CC", "CXX") if name in os.environ}
    env.update(TRITON_CACHE_DIR="{tmp}/triton-cache", PYTHONPATH="{work}", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1")
    return {"devices": ["nvidia"], "env": env, "path": os.environ.get("PATH", isolate.DEFAULT_PATH),
            "dev_shm": True, "timeout_seconds": config["worker_timeout_seconds"],
            # one byte past the result bound, so an oversized result is seen, never cut
            "output_bytes": max(isolate.DEFAULT_LIMITS["output_bytes"],
                                _result_bound(config, workload) + 1)}


def _measure(source: Path, config: dict) -> list[dict]:
    """One isolate instance per workload: `source` (holding dbscanlib/) is its writable
    work dir; it sees no judge file, no other implementation's run, no FRONTIER_* variable."""
    rows = []
    for workload in config["workloads"]:
        result = isolate.run(
            [os.path.realpath(sys.executable), "{in}/worker.py", "{work}",
             json.dumps(workload, separators=(",", ":")), str(config["warmup_iterations"]),
             str(config["timed_iterations"]), "{out}/result.json"],
            tree=source, inputs={"worker.py": WORKER}, outputs=["result.json"],
            **_worker_options(config, workload))
        output = result.outputs.get("result.json")
        if not result.ok or output is None:
            raise RuntimeError(f"DBSCAN workload {workload['id']} did not produce a measurement")
        if len(output) > _result_bound(config, workload):
            raise RuntimeError("DBSCAN worker output exceeds its bounded label/measurement size")
        row = json.loads(output)
        _validate_samples(row, workload, config["timed_iterations"])
        rows.append(row)
    return rows


def evaluate(tree: str) -> tuple[float, float, str, dict]:
    """`tree` is the starter with only the submission's allowed files overlaid (shared verifier)."""
    if not STARTER.is_dir() or not REFERENCE_PATCH.is_file():
        raise RuntimeError("DBSCAN Verifier source bundle is incomplete")
    config = select_suite(json.loads(CONFIG_PATH.read_text()), os.environ.get("FRONTIER_SUBMISSION_ROLE", ""))
    with tempfile.TemporaryDirectory(prefix="frontier-dbscan-") as directory:
        scratch = Path(directory)
        roots = {kind: scratch / kind for kind in ("naive", "reference", "candidate")}
        for kind in ("naive", "reference"):
            shutil.copytree(STARTER, roots[kind] / "dbscanlib")
        shutil.copytree(Path(tree) / "dbscanlib", roots["candidate"] / "dbscanlib")
        # The naive oracle is frozen independently of the Agent's starter tree.
        shutil.copyfile(TASK_DIR / "refdbscan.py", roots["naive"] / "dbscanlib" / "dbscan.py")
        _apply_patch(roots["reference"], REFERENCE_PATCH)
        measurements = [_measure(roots[kind], config) for kind in roots]
        score, metrics = compare_measurements(*measurements, config)
    metrics["evaluation_suite"] = {"mode": config["suite_mode"],
                                    "workloads": [{"id": w["id"], "base_seed": w["base_seed"]} for w in config["workloads"]]}
    return score, metrics["geomean_speedup"], "DBSCAN paired speedup and per-workload clustering agreement measured", metrics
