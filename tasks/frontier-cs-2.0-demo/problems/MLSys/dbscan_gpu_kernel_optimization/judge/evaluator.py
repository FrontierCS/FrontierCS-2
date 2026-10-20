"""Native FrontierCS2 evaluator for GPU DBSCAN kernel optimization."""
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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from checker import adjusted_rand_index, noise_mask_mismatch, validate_labels

TASK_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TASK_DIR / "scoring_config.json"
REFERENCE_PATCH = TASK_DIR / "reference.patch"
STARTER = TASK_DIR / "repo" / "dbscanlib"
MAX_PATCH_BYTES = 2_000_000
MAX_CHANGED_FILES = 80
ALLOWED_PATTERNS = ("dbscanlib/**",)
DENIED_PATTERNS = (
    "evaluator.py", "reference.patch", "refdbscan.py", "**/refdbscan.py",
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
        return False, f"changed file is outside dbscanlib: {path}"
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


def _worker_environment(source: Path, scratch: Path) -> dict[str, str]:
    env = {name: os.environ[name] for name in (
        "PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "CC", "CXX") if name in os.environ}
    env.update(HOME=str(scratch), TMPDIR=str(scratch), TRITON_CACHE_DIR=str(scratch / "triton-cache"),
               PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", LANG="C.UTF-8", LC_ALL="C.UTF-8")
    return env


def _measure(source: Path, config: dict, scratch: Path) -> list[dict]:
    rows = []
    for workload in config["workloads"]:
        output = scratch / f"result-{source.name}-{workload['id']}.json"
        stderr = scratch / f"error-{source.name}-{workload['id']}.log"
        args = [sys.executable, str(TASK_DIR / "worker.py"), str(source),
                json.dumps(workload, separators=(",", ":")), str(config["warmup_iterations"]),
                str(config["timed_iterations"]), str(output)]
        with stderr.open("wb") as errors:
            completed = subprocess.run(args, env=_worker_environment(source, scratch), stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=errors, check=False,
                                       timeout=config["worker_timeout_seconds"])
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(f"DBSCAN workload {workload['id']} did not produce a measurement")
        if output.stat().st_size > workload["N"] * config["timed_iterations"] * 24 + 4096:
            raise RuntimeError("DBSCAN worker output exceeds its bounded label/measurement size")
        row = json.loads(output.read_text())
        _validate_samples(row, workload, config["timed_iterations"])
        rows.append(row)
    return rows


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    patch = Path(solution_path)
    valid, message, patch_metrics = validate_patch(patch)
    if not valid:
        return 0.0, 0.0, message, {}
    if not STARTER.is_dir() or not REFERENCE_PATCH.is_file():
        raise RuntimeError("DBSCAN Verifier source bundle is incomplete")
    config = select_suite(json.loads(CONFIG_PATH.read_text()), os.environ.get("FRONTIER_SUBMISSION_ROLE", ""))
    with tempfile.TemporaryDirectory(prefix="frontier-dbscan-") as directory:
        scratch = Path(directory)
        roots = {kind: scratch / kind for kind in ("naive", "reference", "candidate")}
        for root in roots.values():
            shutil.copytree(STARTER, root / "dbscanlib")
        # The naive oracle is frozen independently of the Agent's starter tree.
        shutil.copyfile(TASK_DIR / "refdbscan.py", roots["naive"] / "dbscanlib" / "dbscan.py")
        _apply_patch(roots["reference"], REFERENCE_PATCH)
        try:
            _apply_patch(roots["candidate"], patch)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return 0.0, 0.0, "candidate patch does not apply to the starter", {}
        measurements = [_measure(roots[kind], config, scratch) for kind in roots]
        score, metrics = compare_measurements(*measurements, config)
    metrics.update(patch_metrics)
    metrics["evaluation_suite"] = {"mode": config["suite_mode"],
                                    "workloads": [{"id": w["id"], "base_seed": w["base_seed"]} for w in config["workloads"]]}
    return score, metrics["geomean_speedup"], "DBSCAN paired speedup and per-workload clustering agreement measured", metrics
