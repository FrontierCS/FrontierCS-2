from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from submission_patch import apply_submission_patch

TASK_DIR = Path(__file__).resolve().parent
CONFIG = {"submission": {"allow_paths": ["performance_model/src/main.py"],
                         "max_patch_bytes": 65536}}
DEV_INDICES = tuple(range(0, 112))
FINAL_INDICES = tuple(range(112, 224))
WORKLOADS = ("triangle", "4clique")
BEAT_METRICS = ("selected_runtime_seconds", "accuracy_qualified")
RUNTIME_ROUNDOFF = 1e-12


def _failed_metrics() -> dict:
    return {"beats_reference": 0, "reference_score": 0.0, "margin": 0.0,
            "beat_spec_workload_passes": [
                {"workload": pattern, "metrics": dict.fromkeys(BEAT_METRICS, False)}
                for pattern in WORKLOADS]}


def _load_suite(final: bool) -> list[dict]:
    rows = json.loads((TASK_DIR / "measured_workloads.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 224:
        raise ValueError("invalid measured workload table")
    indices = FINAL_INDICES if final else DEV_INDICES
    suite = [rows[index] for index in indices]
    if len(suite) != 112 or {case["pattern"] for case in suite} != set(WORKLOADS):
        raise ValueError("invalid measured workload table")
    for case in suite:
        for key in ("gs_seconds", "ns_seconds", "gs_error", "ns_error", "error"):
            value = case[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError("invalid measured workload value")
        if not 0 <= min(case["gs_error"], case["ns_error"]) or not 0 < case["error"] <= 1:
            raise ValueError("invalid measured error")
        if case["gs_error"] > case["error"] or case["ns_error"] > case["error"]:
            raise ValueError("stored backend result misses its accuracy target")
        if min(case["gs_seconds"], case["ns_seconds"]) <= 0:
            raise ValueError("invalid measured runtime")
    return suite


def _measure_patterns(choices: list[str], suite: list[dict]) -> dict:
    if len(choices) != len(suite) or any(choice not in ("GS", "NS") for choice in choices):
        raise ValueError("selector returned invalid decisions")
    measurements = {}
    for pattern in WORKLOADS:
        selected = [(case, choice.lower()) for case, choice in zip(suite, choices)
                    if case["pattern"] == pattern]
        if not selected:
            raise ValueError("measured suite lacks a required pattern")
        times = [case[f"{engine}_seconds"] for case, engine in selected]
        measurements[pattern] = {
            "selected_runtime_seconds": math.exp(sum(map(math.log, times)) / len(times)),
            "accuracy_qualified": all(case[f"{engine}_error"] <= case["error"]
                                      for case, engine in selected),
            "cases": len(selected),
        }
    return measurements


def _beat_rows(choices: list[str], ref_choices: list[str], suite: list[dict]) -> list[dict]:
    own = _measure_patterns(choices, suite)
    reference = _measure_patterns(ref_choices, suite)
    rows = []
    for pattern in WORKLOADS:
        candidate, ref = own[pattern], reference[pattern]
        if not ref["accuracy_qualified"]:
            raise ValueError("reference selection misses the accuracy target")
        runtime, ref_runtime = candidate["selected_runtime_seconds"], ref["selected_runtime_seconds"]
        rows.append({"workload": pattern, "metrics": {
            "selected_runtime_seconds": runtime < ref_runtime - RUNTIME_ROUNDOFF * max(runtime, ref_runtime),
            "accuracy_qualified": candidate["accuracy_qualified"],
        }})
    return rows


def _public_cases(suite: list[dict]) -> list[dict]:
    keys = ("v", "e", "triangles", "max_degree", "pattern", "error", "c", "ns", "cores")
    return [{key: case[key] for key in keys} for case in suite]


def _constant_regret(suite: list[dict]) -> float:
    """The better of the two constant selectors, measured on this same suite.

    This is what a solver gets for free: answer "GS" every time, or "NS" every
    time, and pick whichever is better. Anchoring the scale here rather than at
    `100 / regret` is the whole point of the metric. Under the old formula the
    free constant already scored 79.7 of 100 on the feedback suite and 73.0 on
    the graded one, so four fifths of the range measured nothing, and a selector
    that was *worse* than a constant still scored 55.3 — which is how this task
    came to be filed `very_hard` on agents landing at 89.

    Computed per suite, so the feedback and graded halves each carry their own
    zero. They are not equally hard — a constant wins 81 of 112 cases on one and
    73 of 112 on the other — and a shared constant would silently make one suite
    score higher than the other for reasons that have nothing to do with the
    submission.
    """
    totals = []
    for pick in ("GS", "NS"):
        logs = [math.log((case["gs_seconds"] if pick == "GS" else case["ns_seconds"])
                         / min(case["gs_seconds"], case["ns_seconds"])) for case in suite]
        totals.append(math.exp(sum(logs) / len(logs)))
    return min(totals)


def _score(choices: list[str], suite: list[dict]) -> tuple[float, float, float, float]:
    regrets = []
    correct = 0
    for choice, case in zip(choices, suite):
        gs, ns = case["gs_seconds"], case["ns_seconds"]
        selected = gs if choice == "GS" else ns
        regrets.append(selected / min(gs, ns))
        correct += choice == ("GS" if gs < ns else "NS")
    regret = math.exp(sum(math.log(value) for value in regrets) / len(regrets))
    base = _constant_regret(suite)
    # A suite on which the free constant is already optimal cannot grade anything;
    # say so rather than dividing by zero and reporting a number.
    if base - 1.0 < 1e-9:
        raise ValueError("the constant selector is already optimal on this suite; "
                         "the workload cannot separate a selector from a constant")
    unbounded = 100.0 * (base - regret) / (base - 1.0)
    return min(100.0, max(0.0, unbounded)), unbounded, regret, correct / len(suite)


def _run_candidate(workdir: str, suite: list[dict]) -> tuple[bool, list[str], str]:
    runner = r'''
import json, sys
sys.path.insert(0, sys.argv[1] + "/performance_model/src")
import data, main
cases = json.loads(sys.stdin.read())
answers = []
for x in cases:
    data.graphs["workload"] = data.Graph("measured", x["v"], x["e"], x["triangles"], x["max_degree"], 1.0)
    value = main.threshold("workload", x["pattern"], x["error"], x["ns"], x["c"], x["cores"])
    if value not in ("GS", "NS"):
        raise ValueError("selector must return GS or NS")
    answers.append(value)
print("FRONTIER_RESULT=" + json.dumps(answers))
'''
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", runner, workdir],
                              input=json.dumps(_public_cases(suite)), text=True,
                              capture_output=True, timeout=12, cwd=workdir,
                              env={"PATH": os.environ.get("PATH", "")})
        if proc.returncode:
            return False, [], "selector crashed"
        marker = next((line for line in reversed(proc.stdout.splitlines())
                       if line.startswith("FRONTIER_RESULT=")), None)
        if marker is None:
            return False, [], "selector produced no valid result"
        answers = json.loads(marker.split("=", 1)[1])
        if len(answers) != len(suite) or any(x not in ("GS", "NS") for x in answers):
            return False, [], "selector returned invalid decisions"
        return True, answers, "ok"
    except subprocess.TimeoutExpired:
        return False, [], "selector timed out"
    except Exception:
        return False, [], "selector output was invalid"


def _evaluate_patch(path: str, suite: list[dict]) -> tuple[bool, list[str], str]:
    result = apply_submission_patch(path, TASK_DIR, CONFIG)
    if not result.ok:
        return False, [], result.message
    try:
        return _run_candidate(result.workdir, suite)
    finally:
        if result.workdir:
            shutil.rmtree(result.workdir, ignore_errors=True)


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    try:
        final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
        suite = _load_suite(final)
        ref_ok, ref_choices, _ = _evaluate_patch(str(TASK_DIR / "reference.patch"), suite)
        if not ref_ok:
            return 0.0, 0.0, "reference evaluation failed", _failed_metrics()
        reference_score, _, reference_regret, _ = _score(ref_choices, suite)
        base_metrics = {**_failed_metrics(), "reference_score": reference_score,
                        "margin": -reference_score}
        ok, choices, reason = _evaluate_patch(solution_path, suite)
        if not ok:
            return 0.0, 0.0, reason, base_metrics
        score, unbounded, regret, accuracy = _score(choices, suite)
        rows = _beat_rows(choices, ref_choices, suite)
        selected_times = [case[f"{choice.lower()}_seconds"] for case, choice in zip(suite, choices)]
        metrics = {"beats_reference": int(all(all(row["metrics"].values()) for row in rows)),
                   "beat_spec_workload_passes": rows,
                   "reference_score": reference_score, "margin": score - reference_score,
                   "geomean_runtime_regret": regret, "decision_accuracy": accuracy,
                   "geomean_selected_runtime_seconds": math.exp(sum(map(math.log, selected_times)) / len(selected_times)),
                   "max_selected_relative_error": max(case[f"{choice.lower()}_error"] for case, choice in zip(suite, choices)),
                   # The anchor and the ceiling, so a reader can see how much of the
                   # frontier is left without re-deriving it from the score. On the
                   # graded suite the reference sits at 4.8% regret against an oracle
                   # at 0%, and that is the whole of what is unclaimed here.
                   "constant_selector_regret": _constant_regret(suite),
                   "reference_regret": reference_regret,
                   "max_gs_reference_error": max(case["gs_error"] for case in suite),
                   "max_ns_reference_error": max(case["ns_error"] for case in suite),
                   "cases": len(suite)}
        message = f"score={score:.3f}; regret={regret:.4f}; accuracy={accuracy:.3f}"
        return score, unbounded, message, metrics
    except Exception:
        return 0.0, 0.0, "evaluation failed safely", _failed_metrics()


if __name__ == "__main__":
    result = evaluate(sys.argv[1] if len(sys.argv) > 1 else "")
    print(json.dumps({"score": result[0], "score_unbounded": result[1],
                      "message": result[2], "metrics": result[3]}, sort_keys=True))
