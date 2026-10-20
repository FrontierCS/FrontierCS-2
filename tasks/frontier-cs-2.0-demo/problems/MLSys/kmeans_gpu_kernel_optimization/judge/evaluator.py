"""Native FrontierCS2 evaluator for GPU K-Means kernel optimization."""
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
REFERENCE_PATCH = TASK_DIR / "reference.patch"
STARTER = TASK_DIR / "repo" / "kmeanslib"
MAX_PATCH_BYTES = 2_000_000
MAX_CHANGED_FILES = 80
ALLOWED_PATTERNS = ("kmeanslib/**",)
DENIED_PATTERNS = (
    "evaluator.py", "reference.patch", "refkmeans.py", "**/refkmeans.py",
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
            raise ValueError("rename and copy patch operations are not supported")
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
        return False, f"changed file is outside kmeanslib: {path}"
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


def _measure(source_root: Path, config: dict[str, Any], scratch: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload in config["workloads"]:
        result_path = scratch / f"result-{source_root.name}-{workload['id']}.json"
        error_path = scratch / f"error-{source_root.name}-{workload['id']}.log"
        arguments = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            str(source_root), json.dumps(workload, separators=(",", ":")),
            str(workload["base_seed"]), str(config["warmup_iterations"]),
            str(config["timed_iterations"]), str(result_path),
        ]
        with error_path.open("wb") as error_stream:
            try:
                completed = subprocess.run(
                    arguments, env=_worker_environment(source_root, scratch),
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=error_stream, check=False, timeout=900,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"workload {workload['id']} exceeded its GPU worker timeout") from error
        if completed.returncode != 0 or not result_path.is_file():
            tail = error_path.read_bytes()[-1000:].decode("utf-8", errors="replace")
            raise RuntimeError(f"workload {workload['id']} failed: {tail}")
        row = json.loads(result_path.read_text())
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


def evaluate(solution_path: str) -> tuple[float, float, str, dict[str, Any]]:
    patch = Path(solution_path)
    valid, message, patch_metrics = validate_patch(patch)
    if not valid:
        return 0.0, 0.0, message, {}
    if not STARTER.is_dir() or not REFERENCE_PATCH.is_file():
        raise RuntimeError("KMeans Verifier source bundle is incomplete")
    # The native Verifier maps its trusted train/eval mode to this role.
    config = select_suite(json.loads(CONFIG_PATH.read_text()),
                          os.environ.get("FRONTIER_SUBMISSION_ROLE", ""))
    # The verifier reports execution errors as failed verification, not a score.
    with tempfile.TemporaryDirectory(prefix="frontier-kmeans-") as directory:
        scratch = Path(directory)
        (scratch / "triton-cache").mkdir()
        roots = {name: scratch / name for name in ("naive", "reference", "candidate")}
        for root in roots.values():
            shutil.copytree(STARTER.parent, root)
        _apply_patch(roots["reference"], REFERENCE_PATCH)
        try:
            _apply_patch(roots["candidate"], patch)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return 0.0, 0.0, "candidate patch does not apply to the starter", {}
        naive = _measure(roots["naive"], config, scratch)
        reference = _measure(roots["reference"], config, scratch)
        candidate = _measure(roots["candidate"], config, scratch)
        score, metrics = compare_measurements(naive, reference, candidate, config)
    metrics.update(patch_metrics)
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
