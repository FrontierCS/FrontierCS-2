"""Evaluator for GPU K-Means kernel optimization (one H100): one exact Lloyd step.

Each selected workload runs the naive starter, the optimized reference and the candidate
(`tree`: the starter with the submission overlaid) over a two-step Lloyd loop on planted
bfloat16 clusters. Train: the four public workloads. Final: all six (public plus hidden).

raw_score = score_unbounded: S = geometric mean over the workloads of naive median loop latency
divided by candidate median loop latency; 0 if S <= 1, else 100*log(S)/(log(10)+log(S)).

Metrics per workload (candidate vs reference; all must pass):
  lloyd_loop_latency_ms    synchronized two-step loop latency, lower; passes when
                           candidate < reference.
  labeled_inertia          inertia of the final labels/centroids, lower; passes when
                           candidate <= 1.05 * reference.
  label_mismatch_fraction  share of final labels that are not an exact nearest centroid, lower;
                           passes when candidate <= reference + 0.002.
beats_reference = 1 when every metric passes on every selected workload (a conjunction).
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

TASK_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TASK_DIR / "scoring_config.json"
REFERENCE_PATCH = TASK_DIR / "reference.patch"
PRISTINE = TASK_DIR / "pristine"  # the starter repo; the shared verifier overlays the submission
STARTER = PRISTINE / "kmeanslib"
WORKER = Path(__file__).resolve()  # this file runs one implementation and workload (--worker)
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


def compare_measurements(
    naive: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    by_kind = [{row["id"]: row for row in rows} for rows in (naive, reference, candidate)]
    workload_ids = [row["id"] for row in config["workloads"]]
    if any(set(table) != set(workload_ids) for table in by_kind):
        raise ValueError("implementation measurements do not cover the configured workloads")
    naive_map, reference_map, candidate_map = by_kind
    comparison = config["comparisons"]
    latency_rule = comparison["lloyd_loop_latency_ms"]
    latency_margin = float(latency_rule["minimum_improvement_fraction"])
    latency_strict = latency_rule.get("strictly_better")
    inertia_band = float(comparison["labeled_inertia"]["relative_non_inferiority_band"])
    mismatch_band = float(comparison["label_mismatch_fraction"]["absolute_non_inferiority_band"])
    if latency_strict is not True or not 0 <= latency_margin < 1 or inertia_band < 0 or mismatch_band < 0:
        raise ValueError("scoring comparison bands are invalid")

    speedups: list[float] = []
    reference_speedups: list[float] = []
    passes: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for workload_id in workload_ids:
        baseline = naive_map[workload_id]
        ref = reference_map[workload_id]
        cand = candidate_map[workload_id]
        speedups.append(float(baseline["lloyd_loop_latency_ms"]) / float(cand["lloyd_loop_latency_ms"]))
        reference_speedups.append(float(baseline["lloyd_loop_latency_ms"]) / float(ref["lloyd_loop_latency_ms"]))
        decisions = {
            "lloyd_loop_latency_ms": float(cand["lloyd_loop_latency_ms"]) < float(ref["lloyd_loop_latency_ms"]) * (1.0 - latency_margin),
            "labeled_inertia": float(cand["labeled_inertia"]) <= float(ref["labeled_inertia"]) * (1.0 + inertia_band),
            "label_mismatch_fraction": float(cand["label_mismatch_fraction"]) <= float(ref["label_mismatch_fraction"]) + mismatch_band,
        }
        passes.append({"workload": workload_id, "metrics": decisions})
        details.append({"workload": workload_id, "naive": baseline, "reference": ref, "candidate": cand, "metrics": decisions})

    geomean_speedup = geometric_mean(speedups)
    reference_geomean_speedup = geometric_mean(reference_speedups)
    score_midpoint = float(config["continuous_score"]["speedup_midpoint"])
    score = score_from_speedup(geomean_speedup, score_midpoint)
    reference_score = score_from_speedup(reference_geomean_speedup, score_midpoint)
    metrics = {
        "geomean_speedup": geomean_speedup,
        "lloyd_loop_latency_ms": sorted(float(row["lloyd_loop_latency_ms"]) for row in candidate),
        "labeled_inertia": sorted(float(row["labeled_inertia"]) for row in candidate),
        "label_mismatch_fraction": sorted(float(row["label_mismatch_fraction"]) for row in candidate),
        "beats_reference": int(all(value for row in passes for value in row["metrics"].values())),
        "reference_score": reference_score,
        "margin": score - reference_score,
        "beat_spec_workload_passes": passes,
        "comparison_details": details,
        "calibration_status": config["calibration_status"],
    }
    return score, metrics


def _apply_patch(root: Path, patch: Path) -> None:
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(root), "LANG": "C", "LC_ALL": "C"}
    for arguments in (["git", "apply", "--check", str(patch)], ["git", "apply", str(patch)]):
        subprocess.run(arguments, cwd=root, env=environment, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)


def _isolate():
    """The shared isolate launcher (from gpu-tools). Imported on use: this file is also the
    worker that runs inside the sandbox, where the launcher is neither needed nor readable."""
    import isolate
    return isolate


def _worker_options(config: dict[str, Any]) -> dict[str, Any]:
    """The isolate knobs of one worker: the GPU, the toolchain environment, the time bound.
    Naive, reference and candidate run alike, so their latencies stay paired."""
    allowed = ("LD_LIBRARY_PATH", "LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CC", "CXX")
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment.update({
        "TRITON_CACHE_DIR": "{tmp}/triton-cache", "PYTHONPATH": "{work}",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    })
    return {"devices": ["nvidia"], "env": environment, "dev_shm": True, "timeout_seconds": 900,
            "path": os.environ.get("PATH", _isolate().DEFAULT_PATH)}


def _measure(source_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """One isolate instance per workload: `source_root` (holding kmeanslib/) is its
    writable work dir; it sees no judge file, no other implementation's run, no FRONTIER_*
    variable. The worker is this file (`--worker`), given as an input."""
    isolate = _isolate()
    rows: list[dict[str, Any]] = []
    for workload in config["workloads"]:
        arguments = [
            os.path.realpath(sys.executable), "{in}/evaluator.py", "--worker",
            "{work}", json.dumps(workload, separators=(",", ":")),
            str(workload["base_seed"]), str(config["warmup_iterations"]),
            str(config["timed_iterations"]), "{out}/result.json",
        ]
        result = isolate.run(arguments, tree=source_root, inputs={"evaluator.py": WORKER},
                                  outputs=["result.json"], **_worker_options(config))
        if result.timed_out:
            raise RuntimeError(f"workload {workload['id']} exceeded its GPU worker timeout")
        output = result.outputs.get("result.json")
        if not result.ok or output is None:
            tail = result.stderr[-1000:].decode("utf-8", errors="replace")
            raise RuntimeError(f"workload {workload['id']} failed: {tail}")
        row = json.loads(output)
        if row.get("id") != workload["id"]:
            raise RuntimeError("GPU worker returned the wrong workload identity")
        for key in ("lloyd_loop_latency_ms", "labeled_inertia", "label_mismatch_fraction"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                raise RuntimeError(f"GPU worker returned invalid {key}")
        rows.append(row)
    return rows


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


def evaluate(tree: str) -> tuple[float, float, str, dict[str, Any]]:
    """`tree` is the starter with only the submission's allowed files overlaid (shared verifier)."""
    if not STARTER.is_dir() or not REFERENCE_PATCH.is_file():
        raise RuntimeError("KMeans Verifier source bundle is incomplete")
    # The native Verifier maps its trusted train/eval mode to this role.
    config = select_suite(json.loads(CONFIG_PATH.read_text()),
                          os.environ.get("FRONTIER_SUBMISSION_ROLE", ""))
    # The verifier reports execution errors as failed verification, not a score.
    with tempfile.TemporaryDirectory(prefix="frontier-kmeans-") as directory:
        scratch = Path(directory)
        roots = {name: scratch / name for name in ("naive", "reference", "candidate")}
        for name in ("naive", "reference"):
            shutil.copytree(STARTER.parent, roots[name])
        shutil.copytree(Path(tree), roots["candidate"])
        _apply_patch(roots["reference"], REFERENCE_PATCH)
        naive = _measure(roots["naive"], config)
        reference = _measure(roots["reference"], config)
        candidate = _measure(roots["candidate"], config)
        score, metrics = compare_measurements(naive, reference, candidate, config)
    metrics["evaluation_suite"] = {
        "mode": config["suite_mode"],
        "workload_base_seeds": {row["id"]: row["base_seed"] for row in config["workloads"]},
        "workload_ids": [row["id"] for row in config["workloads"]],
    }
    return (
        score,
        score,
        f"KMeans score {score:.2f}/100 "
        f"(geometric-mean speedup over naive: {metrics['geomean_speedup']:.4f}x)",
        metrics,
    )


def _generate(workload: dict[str, Any], seed: int):
    import torch
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)
    n, d, k = int(workload["N"]), int(workload["D"]), int(workload["K"])
    noise = torch.randn(n, d, generator=generator, device=device, dtype=torch.float32)
    centers = torch.randn(k, d, generator=generator, device=device, dtype=torch.float32) * (4.0 * (2.0 / d) ** 0.5)
    assignments = torch.randint(0, k, (n,), generator=generator, device=device)
    x = (centers.index_select(0, assignments) + noise).to(torch.bfloat16)
    permutation = torch.randperm(n, generator=generator, device=device)[:k]
    return x, x.index_select(0, permutation).clone()


def _run_loop(module, x, initial, iterations: int):
    centroids = initial.clone()
    labels = None
    input_centroids = None
    for _ in range(iterations):
        input_centroids = centroids
        labels, centroids = module.step(x, centroids)
    return labels, centroids, input_centroids


def _validate_output(workload: dict[str, Any], labels, centroids, input_centroids) -> None:
    import torch
    if labels is None or centroids is None or input_centroids is None:
        raise ValueError("step returned no output")
    if tuple(labels.shape) != (int(workload["N"]),) or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("labels have the wrong shape or dtype")
    if tuple(centroids.shape) != (int(workload["K"]), int(workload["D"])):
        raise ValueError("centroids have the wrong shape")
    if not torch.isfinite(centroids.float()).all():
        raise ValueError("centroids are non-finite")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= int(workload["K"]):
        raise ValueError("labels are out of range")


def _labeled_inertia(x, centroids, labels) -> float:
    total = 0.0
    centers = centroids.float()
    labels = labels.long()
    for start in range(0, x.shape[0], 16384):
        points = x[start:start + 16384].float()
        selected = centers.index_select(0, labels[start:start + 16384])
        total += float(((points - selected) ** 2).sum().item())
    return total


def _label_mismatch(x, centroids, labels) -> float:
    centers = centroids.float()
    center_norm = (centers * centers).sum(1)
    labels = labels.long()
    mismatches = 0
    for start in range(0, x.shape[0], 16384):
        points = x[start:start + 16384].float()
        distances = (points * points).sum(1, keepdim=True) - 2.0 * (points @ centers.t()) + center_norm[None, :]
        exact = distances.argmin(1)
        best = distances.gather(1, exact[:, None]).squeeze(1)
        returned = distances.gather(1, labels[start:start + 16384, None]).squeeze(1)
        mismatches += int((returned > best * (1.0 + 1e-6) + 1e-6).sum().item())
    return mismatches / max(1, x.shape[0])


def _worker(argv: list[str]) -> int:
    import importlib
    import time
    import torch
    source_root = Path(argv[0]).resolve()
    workload = json.loads(argv[1])
    base_seed, warmups, timed = map(int, argv[2:5])
    result_path = Path(argv[5])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    synchronize = torch.cuda.synchronize
    perf_counter = time.perf_counter
    sys.path.insert(0, str(source_root))
    module = importlib.import_module("kmeanslib")
    workload_seed = base_seed + 1000 * (int(workload["id"][1:]) + 1)
    for index in range(warmups):
        x, initial = _generate(workload, workload_seed + 100 + index)
        _run_loop(module, x, initial, int(workload["max_iters"]))
        synchronize()
        del x, initial
        torch.cuda.empty_cache()
    latencies: list[float] = []
    inertias: list[float] = []
    mismatches: list[float] = []
    for index in range(timed):
        x, initial = _generate(workload, workload_seed + 10000 + index)
        synchronize()
        started = perf_counter()
        labels, centroids, input_centroids = _run_loop(module, x, initial, int(workload["max_iters"]))
        synchronize()
        latencies.append((perf_counter() - started) * 1000.0)
        _validate_output(workload, labels, centroids, input_centroids)
        inertias.append(_labeled_inertia(x, centroids, labels))
        mismatches.append(_label_mismatch(x, input_centroids, labels))
        del x, initial, labels, centroids, input_centroids
        torch.cuda.empty_cache()
    result = {
        "id": workload["id"],
        "lloyd_loop_latency_ms": statistics.median(latencies),
        "labeled_inertia": max(inertias),
        "label_mismatch_fraction": max(mismatches),
        "timed_samples": timed,
    }
    result_path.write_text(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--worker":
        if len(argv) != 7:
            raise ValueError("--worker requires source, workload, seed, warmups, repetitions and result path")
        return _worker(argv[1:])
    if len(argv) > 1:
        raise ValueError("expected at most one solution patch path")
    path = argv[0] if argv else str(REFERENCE_PATCH)
    score, unbounded, message, metrics = evaluate(path)
    print(json.dumps({"score": score, "score_unbounded": unbounded, "message": message, "metrics": metrics}, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
