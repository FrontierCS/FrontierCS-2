"""Black-box judge for SeerEVM's reproduced branch-prediction experiment."""
from __future__ import annotations

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

from submission_patch import apply_submission_patch

TASK_DIR = Path(__file__).resolve().parent
CONFIG = {"submission": {"allow_paths": ["core/vm/predictor.go"]}}
DATASET = TASK_DIR / "judge_data" / "ethereumdata_1465_small"
REFERENCE_SOURCE = TASK_DIR / "upstream_predictor.go"
DEV_BLOCKS = 200
FINAL_BLOCKS = 400
RUN_TIMEOUT = 420
SAFE_IMPORTS = {"math", "math/bits", "sort", "sync", "sync/atomic"}
FORBIDDEN = (
    "\"os\"", "\"syscall\"", "\"unsafe\"", "\"net/", "os/exec",
    "//go:", "func init(", "FRONTIER_", "evaluator", "reference.patch",
)
# Preserve the historical finite 0.xx/1.xx domain for transaction satisfaction.
# The upstream normal denominator subtracts all high-gas receipts from contract
# calls, including creations that were never counted as calls; it is not a
# probability. Branch accuracy still has matching numerator/denominator domains.
PAIR_RE = re.compile(r"^([01]\.\d{2})\s*&&\s*(0\.\d{2}|1\.00)$")
HEADER_RE = re.compile(r"^===== Block height (\d+) =====$")
WORKLOADS = ("complex", "normal")
BEAT_METRICS = ("branch_accuracy", "transaction_path_accuracy", "report_coverage")
# Beat spec v3: the paper reports prediction accuracy per 200-block interval
# (Section 5.2, Figure 10). Development grades its single epoch; final grades
# every epoch of its replay, i.e. the freshly replayed public prefix (blocks
# 1-200) and the held-out epoch (blocks 201-400), each category separately.
# The continuous score is unchanged and still uses only the scored epoch.
# Final replays 600 blocks so a second held-out interval (blocks 401-600) is
# graded as well; FINAL_BLOCKS stays the continuous-score epoch.
FINAL_REPLAY_BLOCKS = 600
DEV_EPOCHS = (DEV_BLOCKS,)
FINAL_EPOCHS = (DEV_BLOCKS, FINAL_BLOCKS, FINAL_REPLAY_BLOCKS)


class ReferenceFailure(RuntimeError):
    """The paired reference or calibration replay did not produce a valid report.

    An evaluator error, never a candidate result: a failed reference cannot set
    a bar and the candidate is not scored against it.
    """


def _graded_epochs(blocks: int) -> tuple[int, ...]:
    return FINAL_EPOCHS if blocks == FINAL_REPLAY_BLOCKS else DEV_EPOCHS


def _workload_name(epoch: int, category: str) -> str:
    return f"epoch{epoch}-{category}"
_REFERENCE_CACHE: dict[int, tuple[float, dict[str, Any]]] = {}
_BASELINE_CACHE: dict[int, tuple[float, dict[str, Any]]] = {}


def _invalid(reason: str) -> tuple[float, float, str, dict]:
    return 0.0, 0.0, reason, {
        "beats_reference": 0,
        "reference_score": 0.0,
        "margin": 0.0,
        "beat_spec_workload_passes": _failed_rows(),
    }


def _role_blocks() -> int:
    return FINAL_REPLAY_BLOCKS if os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final" else DEV_BLOCKS


def _failed_rows() -> list[dict[str, Any]]:
    return [{"workload": _workload_name(epoch, name), "metrics": dict.fromkeys(BEAT_METRICS, False)}
            for epoch in _graded_epochs(_role_blocks()) for name in WORKLOADS]


def _beat_rows(candidate: dict[str, Any], reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare each graded epoch and transaction class using exact printed ticks."""
    candidate_epochs = candidate["epoch_measurements"]
    reference_epochs = reference["epoch_measurements"]
    if set(candidate_epochs) != set(reference_epochs):
        raise ValueError("prediction report lacks a required epoch")
    rows = []
    for epoch in sorted(reference_epochs, key=int):
        candidate_rows = candidate_epochs[epoch]
        reference_rows = reference_epochs[epoch]
        if set(candidate_rows) != set(WORKLOADS) or set(reference_rows) != set(WORKLOADS):
            raise ValueError("prediction report lacks a required transaction category")
        for name in WORKLOADS:
            own, ref = candidate_rows[name], reference_rows[name]
            coverage = own["observations"] == ref["observations"] > 0
            rows.append({"workload": _workload_name(int(epoch), name), "metrics": {
                "branch_accuracy": coverage and own["branch_ticks"] >= ref["branch_ticks"] + 1,
                "transaction_path_accuracy": coverage and own["transaction_ticks"] >= ref["transaction_ticks"],
                "report_coverage": coverage,
            }})
    return rows


def _category_ticks(categories: list[list[tuple[float, float]]]) -> dict[str, dict[str, int]]:
    return {
        name: {"observations": len(records),
               "transaction_ticks": sum(round(tx * 100) for tx, _ in records),
               "branch_ticks": sum(round(branch * 100) for _, branch in records)}
        for name, records in zip(WORKLOADS, categories)
    }


def _validate_source(source: str) -> str | None:
    if not re.match(r"\s*package\s+vm\b", source):
        return "predictor must remain in package vm"
    if len(source.encode("utf-8")) > 100_000:
        return "predictor source exceeds 100,000 bytes"
    for token in FORBIDDEN:
        if token in source:
            return "predictor uses a forbidden runtime capability"
    imports = set(re.findall(r'["`]([A-Za-z0-9_./-]+)["`]', source))
    if not imports <= SAFE_IMPORTS:
        return "predictor imports a package outside the safe allowlist"
    return None


def _adapt_upstream_source(source: str) -> str:
    """Rename integration symbols without changing the upstream algorithm."""
    return (
        source.replace("Perceptron", "Predictor")
        .replace("perceptron", "predictor")
        .replace("HISTORY_LENGTH_MIN", "WARMUP_OBSERVATIONS")
        .replace("historyLength", "observationCount")
    )


def _copy_dataset(destination: Path) -> None:
    if not (DATASET / "copychain").is_dir():
        raise ValueError("benchmark dataset is unavailable")
    proc = subprocess.run(
        ["cp", "-a", "--reflink=auto", str(DATASET), str(destination)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise ValueError("benchmark dataset could not be prepared")


def _parse_report(report: Path, blocks: int = DEV_BLOCKS) -> tuple[float, dict[str, Any]]:
    if not report.is_file() or report.stat().st_size > 2_000_000:
        raise ValueError("prediction report is missing or invalid")
    groups: dict[int, list[list[tuple[float, float]]]] = {}
    current_height: int | None = None
    section = 0
    for line in report.read_text("utf-8", errors="strict").splitlines():
        header = HEADER_RE.fullmatch(line.strip())
        if header:
            current_height = int(header.group(1))
            if current_height in groups:
                raise ValueError("prediction report repeats an epoch")
            groups[current_height] = [[], []]
            section = 0
            continue
        match = PAIR_RE.fullmatch(line.strip())
        if match:
            tx_accuracy, branch_accuracy = map(float, match.groups())
            if section == 0 and tx_accuracy > 1.0:
                raise ValueError("prediction report contains an invalid accuracy record")
            if current_height is None:
                current_height = blocks
                groups[current_height] = [[], []]
            groups[current_height][section].append((tx_accuracy, branch_accuracy))
        elif not line.strip() and current_height is not None and groups[current_height][0]:
            section = 1
        elif "&&" in line:
            raise ValueError("prediction report contains an invalid accuracy record")
    final = blocks == FINAL_REPLAY_BLOCKS
    # The continuous score keeps its v2 epoch: 200 for development and 400
    # (blocks 201-400) for final, even though final now replays 600 blocks.
    scored = FINAL_BLOCKS if final else blocks
    if any(epoch not in groups for epoch in _graded_epochs(blocks)):
        raise ValueError("prediction report lacks a required epoch")
    pairs = groups[scored][0] + groups[scored][1]
    scored_start = DEV_BLOCKS + 1 if final else 1
    if len(pairs) < 20:
        raise ValueError("prediction report contains too few observations")
    categories = groups[scored]
    if not all(categories):
        raise ValueError("prediction report lacks a required transaction category")
    if any(not all(groups[epoch]) for epoch in _graded_epochs(blocks)):
        raise ValueError("prediction report lacks a required transaction category")
    mean_tx = sum(x for x, _ in pairs) / len(pairs)
    mean_branch = sum(x for _, x in pairs) / len(pairs)
    # Branch accuracy is the paper experiment's primary measure. Preserve the
    # upstream transaction satisfaction ratio as the smaller quality term,
    # including normal-category values above one, without clamping.
    quality = 100.0 * (0.8 * mean_branch + 0.2 * mean_tx)
    return quality, {
        "branch_accuracy": mean_branch,
        "transaction_path_accuracy": mean_tx,
        "observations": len(pairs),
        "scored_block_start": scored_start,
        "scored_block_end": scored,
        # Private exact hundredth sums retain the upstream printer's resolution.
        # Only aggregate candidate metrics are allowed through train feedback.
        "category_measurements": _category_ticks(categories),
        # Beat spec v3: exact ticks for every graded epoch (string keys keep
        # the evidence JSON stable). The prefix epoch must be complete too.
        "epoch_measurements": {
            str(epoch): _category_ticks(groups[epoch]) for epoch in _graded_epochs(blocks)
        },
    }


def _run_real(source: str, blocks: int) -> tuple[float, dict[str, Any]]:
    problem = _validate_source(source)
    if problem:
        raise ValueError(problem)
    root = Path(tempfile.mkdtemp(prefix="seer-real-"))
    app = root / "app"
    try:
        shutil.copytree(TASK_DIR / "harbor" / "app", app)
        (app / "core" / "vm" / "predictor.go").write_text(source, "utf-8")
        shutil.copytree(TASK_DIR / "judge_data" / "vendor", app / "vendor")
        _copy_dataset(app / "ethereumdata_1465_small")
        env = os.environ.copy()
        env.update({
            "GOTOOLCHAIN": "local",
            "GOCACHE": str(root / "go-cache"),
            "GOPATH": str(root / "go-path"),
            "GOFLAGS": "-mod=vendor",
        })
        build = subprocess.run(
            ["go", "build", "-buildvcs=false", "-o", str(root / "seerevm"), "."],
            cwd=app,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if build.returncode != 0:
            raise ValueError("predictor does not compile in SeerEVM")
        run = subprocess.run(
            [str(root / "seerevm"), "--indicator", "1", "--start", "14650000", "--blockNum", str(blocks)],
            cwd=app / "experiments",
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=RUN_TIMEOUT,
        )
        if run.returncode != 0:
            raise ValueError("SeerEVM replay failed")
        return _parse_report(app / "experiments" / "prediction_height.txt", blocks)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("SeerEVM build or replay timed out") from exc
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _reference(blocks: int) -> tuple[float, dict[str, Any]]:
    if blocks not in _REFERENCE_CACHE:
        try:
            source = _adapt_upstream_source(REFERENCE_SOURCE.read_text("utf-8"))
            _REFERENCE_CACHE[blocks] = _run_real(source, blocks)
        except Exception as error:
            raise ReferenceFailure(f"reference replay failed: {type(error).__name__}") from error
    return _REFERENCE_CACHE[blocks]


def _baseline(blocks: int) -> tuple[float, dict[str, Any]]:
    if blocks not in _BASELINE_CACHE:
        try:
            source = (TASK_DIR / "baseline_naive.go").read_text("utf-8")
            _BASELINE_CACHE[blocks] = _run_real(source, blocks)
        except Exception as error:
            raise ReferenceFailure(f"calibration replay failed: {type(error).__name__}") from error
    return _BASELINE_CACHE[blocks]


def _bounded_score(quality: float, baseline_quality: float) -> float:
    if baseline_quality >= 100.0:
        return 0.0
    normalized = max(0.0, (quality - baseline_quality) / (100.0 - baseline_quality))
    return max(0.0, min(100.0, 100.0 * normalized * normalized))


def prepare() -> dict:
    blocks = _role_blocks()
    quality, _ = _reference(blocks)
    baseline_quality, _ = _baseline(blocks)
    score = _bounded_score(quality, baseline_quality)
    return {"ready": True, "blocks": blocks, "reference_valid": math.isfinite(score)}


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    applied = None
    try:
        applied = apply_submission_patch(solution_path, TASK_DIR, CONFIG)
        if not applied.ok:
            return _invalid(applied.message)
        source = (applied.workdir / "core" / "vm" / "predictor.go").read_text("utf-8")
        blocks = _role_blocks()
        reference_source = _adapt_upstream_source(REFERENCE_SOURCE.read_text("utf-8"))
        if source == reference_source:
            candidate_quality, public = _reference(blocks)
            reference_quality = candidate_quality
            reference_public = public
        else:
            candidate_quality, public = _run_real(source, blocks)
            reference_quality, reference_public = _reference(blocks)
        baseline_quality, _ = _baseline(blocks)
        candidate = _bounded_score(candidate_quality, baseline_quality)
        reference = _bounded_score(reference_quality, baseline_quality)
        rows = _beat_rows(public, reference_public)
        metrics = {
            "beats_reference": int(all(all(row["metrics"].values()) for row in rows)),
            "beat_spec_workload_passes": rows,
            "reference_score": reference,
            "margin": candidate - reference,
            "raw_quality": candidate_quality,
            **public,
        }
        return candidate, candidate, f"scored {blocks}-block Ethereum replay", metrics
    except ReferenceFailure:
        raise
    except Exception as exc:
        message = str(exc) if isinstance(exc, ValueError) else "submission could not be evaluated"
        return _invalid(message[:120])
    finally:
        if applied is not None and applied.workdir is not None:
            shutil.rmtree(applied.workdir.parent, ignore_errors=True)


if __name__ == "__main__":
    try:
        result = evaluate(sys.argv[1])
        print(json.dumps({"score": result[0], "score_unbounded": result[1], "message": result[2], "metrics": result[3]}, sort_keys=True))
    except Exception:
        result = _invalid("invalid invocation")
        print(json.dumps({"score": result[0], "score_unbounded": result[1], "message": result[2], "metrics": result[3]}))
