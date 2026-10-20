#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import gzip
import math
import os
import re
import signal
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
from submission_patch import apply_submission_patch

CONFIG = {
    "submission": {
        "kind": "patch",
        "path": "/app/solution.patch",
        "repo_root": "/app",
        "allow_paths": [
            "counting/forest.h", "counting/forest.cpp",
            "counting/execution.h", "counting/execution.cpp",
            "counting/aggregation.h", "counting/aggregation.cpp",
            "executable/scope.cpp", "executable/command.cpp",
            "plan/tree.h", "plan/tree.cpp",
        ],
        "max_changed_files": 12,
        "max_patch_bytes": 524288,
    }
}

# Fixed independent halves of the upstream queries for which the artifact's
# independent and optimized paths agree exactly on every evaluated graph.
_COMPATIBLE_5 = tuple(n for n in range(15, 73)
                      if n not in {22, 23, 49, 50, 54, 55, 68, 69, 70, 71})
_COMPATIBLE_4 = (0, 3, 4, 5, 9, 10, 11, 14)
DEV_5 = _COMPATIBLE_5[::2]
FINAL_5 = _COMPATIBLE_5[1::2]
DEV_4 = _COMPATIBLE_4[::2]
FINAL_4 = _COMPATIBLE_4[1::2]
RUNS = 5
FORBIDDEN_ADDED = ("web-spam", "rec-movielens", "bio-grid-yeast",
                   "4voc", "5voc", "frontier_", "submission_role")


CASE_NAMES = tuple(f"{graph}-{size}" for graph in
                   ("web-spam", "rec-movielens", "bio-grid-yeast") for size in (5, 4))


def _metrics(candidate_score: float, reference_score: float, rows: list[dict] | None = None) -> dict:
    if rows is None:
        rows = [{"workload": name, "metrics": {
            "exact_counts": False, "batch_latency_seconds": False,
        }} for name in CASE_NAMES]
    if tuple(row["workload"] for row in rows) != CASE_NAMES:
        raise ValueError("incomplete fixed workload decisions")
    return {
        "beat_spec_workload_passes": rows,
        "beats_reference": int(all(value for row in rows for value in row["metrics"].values())),
        "reference_score": reference_score,
        "margin": candidate_score - reference_score,
    }


def _score(naive_seconds: float, candidate_seconds: float) -> float:
    if naive_seconds <= 0 or candidate_seconds <= 0:
        return 0.0
    return _score_log_ratio(math.log2(naive_seconds / candidate_seconds))


def _score_log_ratio(log_ratio: float) -> float:
    improvement = max(0.0, log_ratio)
    return 100.0 * (improvement / (2.0 + improvement))


def _score_log_ratios(log_ratios: list[float]) -> tuple[float, float]:
    median_log_ratio = statistics.median(log_ratios)
    score = _score_log_ratio(median_log_ratio)
    return score, 2.0 ** median_log_ratio


def _added_source_uses_forbidden_token(patch: Path) -> str | None:
    text = patch.read_text(encoding="utf-8", errors="replace")
    added = "\n".join(line[1:] for line in text.splitlines()
                      if line.startswith("+") and not line.startswith("+++"))
    folded = added.casefold()
    return next((token for token in FORBIDDEN_ADDED if token in folded), None)


def _run(args: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        args, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        text=True, env={**os.environ, "LC_ALL": "C"}, start_new_session=True)
    try:
        proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise
    finally:
        # A submitted binary must not leave workers consuming CPU or perturbing
        # the next member of an interleaved timing round.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return subprocess.CompletedProcess(args, proc.returncode)


def _build(tree: Path) -> Path:
    auto = tree / "utility" / "automorphism"
    if _run(["./configure"], auto, 120).returncode != 0:
        raise RuntimeError("nauty configuration failed")
    makefile = auto / "makefile"
    content = makefile.read_text(encoding="utf-8")
    content = re.sub(r"^CFLAGS=", "CFLAGS= -fPIC ", content, count=1, flags=re.M)
    makefile.write_text(content, encoding="utf-8")
    if _run(["make", "-j1", "nauty.a"], auto, 300).returncode != 0:
        raise RuntimeError("nauty build failed")
    (auto / "nauty.a").replace(auto / "libnauty.a")
    build = tree / ".frontier-build"
    if _run(["cmake", "-S", str(tree), "-B", str(build),
             "-DCMAKE_BUILD_TYPE=Release"], tree, 120).returncode != 0:
        raise RuntimeError("CMake configuration failed")
    if _run(["cmake", "--build", str(build), "--target", "scope.out", "-j1"],
            tree, 600).returncode != 0:
        raise RuntimeError("C++ build failed")
    binary = build / "executable" / "scope.out"
    if not binary.is_file():
        raise RuntimeError("scope.out was not produced")
    return binary


def _make_workload(tree: Path, size: int, ids: tuple[int, ...], name: str) -> Path:
    workload = tree / f".frontier-workload-{name}-{size}"
    workload.mkdir()
    source = tree / "exp" / "pattern_graph" / f"{size}voc"
    for number in ids:
        shutil.copy2(source / f"{number}.txt", workload / f"{number}.txt")
    return workload


def _inflate_cache(tree: Path, stem: str) -> Path:
    source = tree / "exp" / "data_graph" / f"{stem}.bin.gz"
    destination = tree / ".frontier-assets" / f"{stem}.bin"
    destination.parent.mkdir(exist_ok=True)
    with gzip.open(source, "rb") as compressed, destination.open("wb") as expanded:
        shutil.copyfileobj(compressed, expanded)
    return destination


def _case_specs(tree: Path, final: bool) -> list[tuple[str, Path, Path, Path]]:
    ids5 = FINAL_5 if final else DEV_5
    ids4 = FINAL_4 if final else DEV_4
    data = tree / "exp" / "data_graph"
    graphs = [
        ("web-spam", data / "web-spam.txt", data / "web-spamt.bin"),
        ("rec-movielens", data / "rec-movielens.txt", _inflate_cache(tree, "rec-movielenst")),
        ("bio-grid-yeast", data / "bio-grid-yeast.txt", _inflate_cache(tree, "bio-grid-yeastt")),
    ]
    cases: list[tuple[str, Path, Path, Path]] = []
    for name, graph, triangle in graphs:
        cases.append((f"{name}-5", _make_workload(tree, 5, ids5, name), graph, triangle))
        cases.append((f"{name}-4", _make_workload(tree, 4, ids4, name), graph, triangle))
    return cases


def _digest_outputs(output: Path) -> str:
    files = sorted(p for p in output.iterdir() if p.is_file())
    if not files:
        raise RuntimeError("counter emitted no result files")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _prepare_patch(patch: Path, final: bool) -> dict:
    result = apply_submission_patch(patch, TASK_DIR, CONFIG)
    if not result.ok or result.workdir is None:
        raise RuntimeError(result.message or "patch rejected")
    tree = result.workdir
    binary = _build(tree)
    cases = []
    for name, workload, graph, triangle in _case_specs(tree, final):
        output = tree / f".frontier-results-{name}"
        command = [
            str(binary), "-q", str(workload), "-d", str(graph),
            "-t", str(triangle), "-r", str(output), "-b", "-share",
        ]
        cases.append((name, output, command))
    return {
        "tree": tree,
        "cases": cases,
        "digest": "",
        "times": [],
        "case_times": {name: [] for name, _, _ in cases},
        "case_digests": {},
    }


def _run_prepared(prepared: dict) -> float:
    start = time.monotonic()
    digest = hashlib.sha256()
    for name, output, command in prepared["cases"]:
        if output.exists():
            shutil.rmtree(output)
        output.mkdir()
        case_start = time.monotonic()
        proc = _run(command, prepared["tree"], 180)
        prepared["case_times"][name].append(time.monotonic() - case_start)
        if proc.returncode != 0:
            raise RuntimeError("counting run failed")
        digest.update(name.encode())
        case_digest = _digest_outputs(output)
        previous = prepared["case_digests"].get(name)
        if previous is not None and previous != case_digest:
            raise RuntimeError("counts changed between repeated runs")
        prepared["case_digests"][name] = case_digest
        digest.update(case_digest.encode())
    seconds = time.monotonic() - start
    current = digest.hexdigest()
    if prepared["digest"] and current != prepared["digest"]:
        raise RuntimeError("counts changed between repeated runs")
    prepared["digest"] = current
    prepared["times"].append(seconds)
    return seconds


def _patch_key(patch: Path) -> str:
    return hashlib.sha256(patch.read_bytes()).hexdigest()


def _workload_decisions(naive: dict, reference: dict, candidate: dict) -> list[dict]:
    rows = []
    for name in CASE_NAMES:
        ref_times, cand_times = reference["case_times"][name], candidate["case_times"][name]
        if len(ref_times) != RUNS or len(cand_times) != RUNS:
            raise ValueError("incomplete per-workload timing rounds")
        if any(not math.isfinite(t) or t <= 0 for t in ref_times + cand_times):
            raise ValueError("invalid per-workload time")
        if naive["case_digests"][name] != reference["case_digests"][name]:
            raise ValueError("reference count validation failed")
        correct = candidate["case_digests"][name] == reference["case_digests"][name]
        # The existing five interleaved rounds are paired observations. Require
        # consistent strict improvement, rather than inventing a percentage
        # band from the old aggregate score. A tie (including a deduplicated
        # exact reference replay) cannot be a timing win.
        faster = all(c < r for c, r in zip(cand_times, ref_times, strict=True))
        rows.append({"workload": name, "metrics": {
            "exact_counts": bool(correct), "batch_latency_seconds": bool(correct and faster),
        }})
    return rows


def _paired_protocol(candidate_patch: Path, final: bool) -> dict:
    patches = {
        "naive": TASK_DIR / "baseline_naive.patch",
        "reference": TASK_DIR / "reference.patch",
        "candidate": candidate_patch,
    }
    label_keys = {label: _patch_key(patch) for label, patch in patches.items()}
    unique_keys: list[str] = []
    key_patches: dict[str, Path] = {}
    for label in ("naive", "reference", "candidate"):
        key = label_keys[label]
        if key not in key_patches:
            unique_keys.append(key)
            key_patches[key] = patches[label]

    prepared: dict[str, dict] = {}
    try:
        for key in unique_keys:
            prepared[key] = _prepare_patch(key_patches[key], final)

        reference_logs: list[float] = []
        candidate_logs: list[float] = []
        for round_id in range(RUNS):
            shift = round_id % len(unique_keys)
            order = unique_keys[shift:] + unique_keys[:shift]
            round_times: dict[str, float] = {}
            for key in order:
                round_times[key] = _run_prepared(prepared[key])
            naive_time = round_times[label_keys["naive"]]
            reference_logs.append(math.log2(
                naive_time / round_times[label_keys["reference"]]))
            candidate_logs.append(math.log2(
                naive_time / round_times[label_keys["candidate"]]))

        naive = prepared[label_keys["naive"]]
        reference = prepared[label_keys["reference"]]
        candidate = prepared[label_keys["candidate"]]
        reference_score, reference_speedup = _score_log_ratios(reference_logs)
        candidate_score, candidate_speedup = _score_log_ratios(candidate_logs)
        return {
            "correct": (naive["digest"] == reference["digest"]
                        and candidate["digest"] == reference["digest"]),
            "reference_score": reference_score,
            "candidate_score": candidate_score,
            "reference_speedup": reference_speedup,
            "candidate_speedup": candidate_speedup,
            "candidate_median_seconds": statistics.median(candidate["times"]),
            "naive_median_seconds": statistics.median(naive["times"]),
            "beat_spec_workload_passes": _workload_decisions(naive, reference, candidate),
        }
    finally:
        for item in prepared.values():
            shutil.rmtree(item["tree"].parent, ignore_errors=True)


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    try:
        scratch = TASK_DIR / ".evaluator-work"
        scratch.mkdir(exist_ok=True)
        tempfile.tempdir = str(scratch)
        patch = Path(solution_path)
        final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
        applied = apply_submission_patch(patch, TASK_DIR, CONFIG)
        if not applied.ok:
            metrics = dict(applied.metrics)
            metrics.update(_metrics(0.0, 0.0))
            return 0.0, 0.0, applied.message, metrics
        if applied.workdir is not None:
            shutil.rmtree(applied.workdir.parent, ignore_errors=True)
        token = _added_source_uses_forbidden_token(patch)
        if token:
            metrics = _metrics(0.0, 0.0)
            return 0.0, 0.0, "submission contains workload-specific or judge-specific source", metrics

        result = _paired_protocol(patch, final)
        reference_score = result["reference_score"]
        candidate_score = result["candidate_score"]
        if not result["correct"]:
            return (0.0, 0.0, "per-query counts do not exactly match",
                    _metrics(0.0, reference_score, result["beat_spec_workload_passes"]))
        metrics = _metrics(candidate_score, reference_score, result["beat_spec_workload_passes"])
        metrics.update({
            "valid_patch": 1,
            "median_seconds": result["candidate_median_seconds"],
            "naive_median_seconds": result["naive_median_seconds"],
            "paired_speedup": result["candidate_speedup"],
        })
        return candidate_score, candidate_score, "exact counts; timed batch accepted", metrics
    except subprocess.TimeoutExpired:
        return 0.0, 0.0, "build or counting timed out", _metrics(0.0, 0.0)
    except Exception as exc:
        reason = str(exc).strip() or exc.__class__.__name__
        return 0.0, 0.0, reason[:180], _metrics(0.0, 0.0)


if __name__ == "__main__":
    chosen = sys.argv[1] if len(sys.argv) > 1 else str(TASK_DIR / "reference.patch")
    print(evaluate(chosen))
