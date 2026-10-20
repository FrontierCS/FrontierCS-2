"""Offline CLI regressions using invented timings, never scientific run evidence."""
import deployment_test_guard  # noqa: F401  # isort: skip

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("parbutterfly", "parallel_radix_sort")


def curve(speedup):
    return 100.0 * math.log(speedup) / (math.log(10.0) + math.log(speedup))


def evidence(task, final=False, candidate_valid=True):
    if task == "parbutterfly":
        names = [f"graph-{i + 1}" for i in range(8 if final else 4)]
        metrics = {
            "private_timing_samples": {
                name: {"candidate_reference_pair": [1.0] * 31,
                       "candidate_baseline_pair": [1.0] * 31,
                       "reference": [2.0] * 31, "baseline": [4.0] * 31}
                for name in names},
            "private_counts_per_workload": {
                name: {"candidate": 1 if candidate_valid else 0, "oracle": 1,
                       "reference": 1, "baseline": 1} for name in names}}
        return {"metrics": metrics, "raw_score": curve(4) if candidate_valid else 0}, curve(2)
    names = ["d-int-uniform", "d-pair-zipf", "d-int-duplicates", "d-pair-almost"]
    if final:
        names += ["f-int-uniform", "f-pair-zipf", "f-int-duplicates", "f-pair-almost"]
    metrics = {"measurements": [
        {"workload": name, "candidate_ns": 200, "reference_ns": 250,
         "baseline_ns": 400, "candidate_aux_bytes": 0 if candidate_valid else 64 * 1024 * 1024 + 1}
        for name in names]}
    return {"metrics": metrics, "score": curve(2) if candidate_valid else 0}, curve(1.6)


def run_cli(task, document, tmp_path):
    path = tmp_path / "synthetic-evidence.json"
    path.write_text(json.dumps(document))
    return subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/maintenance" / task / "reference_score.py"), str(path)],
        capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("final", [False, True])
def test_complete_scored_evidence_matches_independent_formula(task, final, tmp_path):
    document, reference = evidence(task, final)
    result = run_cli(task, document, tmp_path)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["score"] == pytest.approx(document.get("raw_score", document.get("score")))
    assert output["reference_score"] == pytest.approx(reference)
    assert output["margin"] == pytest.approx(output["score"] - reference)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("bare_metrics", [False, True])
def test_missing_recorded_score_refuses_backfill(task, bare_metrics, tmp_path):
    document, _ = evidence(task)
    document.pop("raw_score" if task == "parbutterfly" else "score")
    if bare_metrics:
        document = document["metrics"]
    result = run_cli(task, document, tmp_path)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "finite numeric recorded candidate" in result.stderr


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("recorded", [None, float("nan"), float("inf"), -float("inf"), True, False, "37.58", [], {}])
def test_nonfinite_or_nonnumeric_recorded_score_refuses_backfill(task, recorded, tmp_path):
    document, _ = evidence(task)
    document["raw_score" if task == "parbutterfly" else "score"] = recorded
    result = run_cli(task, document, tmp_path)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "finite numeric recorded candidate" in result.stderr


@pytest.mark.parametrize("task", TASKS)
def test_wrong_recorded_formula_refuses_backfill(task, tmp_path):
    document, _ = evidence(task, final=True)
    document["raw_score" if task == "parbutterfly" else "score"] = 999
    result = run_cli(task, document, tmp_path)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "differs from recorded" in result.stderr


@pytest.mark.parametrize("task", TASKS)
def test_genuine_recorded_zero_is_accepted(task, tmp_path):
    document, reference = evidence(task, candidate_valid=False)
    result = run_cli(task, document, tmp_path)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["score"] == 0
    assert output["reference_score"] == pytest.approx(reference)
    assert output["margin"] == pytest.approx(-reference)
