"""Evaluator logic on invented measurements: validity, early-ended runs,
feedback privacy, scoring.  No program is built or run here."""
import deployment_test_guard  # noqa: F401  # isort: skip

import json
from pathlib import Path
import types

import pytest
from tests.maintenance.cute_crest.runtime_imports import MODULES, TASK

e = MODULES["evaluator"]
gate = MODULES["gate"]
runner = MODULES["crest_runner"]
feedback = MODULES["beat_spec_feedback"]
SPEC = json.loads((TASK / "judge/beat_spec.json").read_text())
K = gate.K_SEEDS


def run(covered, overhead_seconds=70.0, status="budget_exhausted", attempts=16000, killed=0):
    return {"status": status, "covered": covered, "iterations": 4000, "solver_attempts": attempts,
            "strategy_cpu_seconds": 1.0, "solver_cpu_seconds": overhead_seconds - 1.0,
            "killed_executions": killed, "wall_seconds": 100.0,
            "curve": [[100, covered // 2], [4000, covered]] if status == "budget_exhausted" else [[100, covered // 2]]}


def measured(programs, candidate, reference):
    """candidate/reference: function (program index, seed index) -> run dict."""
    return {(i, role): [f(i, n) for n in range(K)]
            for i in range(len(programs)) for role, f in (("candidate", candidate), ("reference", reference))}


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Replaces building and measuring; returns a setter for the fake runs."""
    state = {}
    tools = (Path("/nonexistent"), runner, types.SimpleNamespace(
        environment=lambda tree, out, tmp=None: ({}, None)), MODULES["solution_policy"])
    monkeypatch.setattr(e, "load_tools", lambda: tools)
    monkeypatch.setattr(e, "build_candidate", lambda tools, solution, out, env: Path("/nonexistent/host"))
    monkeypatch.setattr(e, "measure", lambda tools, programs, hosts, scratch, env:
                        measured(programs, state["candidate"], state["reference"]))

    def evaluate(candidate, reference, role="development", patch="reference.patch"):
        state.update(candidate=candidate, reference=reference)
        monkeypatch.setenv("FRONTIER_SUBMISSION_ROLE", role)
        return e.evaluate(str(TASK / "judge" / patch))
    return evaluate


def spread(base):
    return lambda i, n: run(base + (n % 5) * 4)


def test_reference_equal_candidate_fails_with_zero_margin(harness):
    score, unbounded, message, metrics = harness(spread(1586), spread(1586))
    assert message == "evaluation complete" and score == unbounded
    assert metrics["beats_reference"] == 0 and metrics["margin"] == 0
    assert metrics["beat_spec_workload_passes"] == [
        {"workload": "w0", "metrics": {"branches_covered": False, "search_overhead_seconds": False}}]


def test_win_needs_coverage_and_overhead(harness):
    better = lambda i, n: run(1700 + (n % 5) * 4, overhead_seconds=50.0 + n % 3)  # noqa: E731
    reference = lambda i, n: run(1586 + (n % 5) * 4, overhead_seconds=70.0 + n % 3)  # noqa: E731
    _, _, _, metrics = harness(better, reference)
    assert metrics["beats_reference"] == 1 and metrics["margin"] > 0
    covered_only = lambda i, n: run(1700 + (n % 5) * 4, overhead_seconds=70.0 + n % 3)  # noqa: E731
    _, _, _, metrics = harness(covered_only, reference)
    assert metrics["beats_reference"] == 0
    assert metrics["beat_spec_workload_passes"][0]["metrics"] == {
        "branches_covered": True, "search_overhead_seconds": False}
    cheaper_only = lambda i, n: run(1586 + (n % 5) * 4, overhead_seconds=50.0 + n % 3)  # noqa: E731
    _, _, _, metrics = harness(cheaper_only, reference)
    assert metrics["beat_spec_workload_passes"][0]["metrics"] == {
        "branches_covered": False, "search_overhead_seconds": True}


def test_final_grades_two_programs_and_returns_no_development_feedback(harness):
    better = lambda i, n: run((8500, 760)[i] + n % 7, overhead_seconds=20.0 + n % 3)  # noqa: E731
    reference = lambda i, n: run((7700, 660)[i] + n % 7, overhead_seconds=40.0 + n % 3)  # noqa: E731
    _, _, _, metrics = harness(better, reference, role="final")
    assert metrics["beats_reference"] == 1
    assert [row["workload"] for row in metrics["beat_spec_workload_passes"]] == ["w0", "w1"]
    assert not [key for key in metrics if key.startswith("dev_")]
    # One program short of the margin: no compensation by the other.
    partial = lambda i, n: run((8500, 661)[i] + n % 7, overhead_seconds=20.0 + n % 3)  # noqa: E731
    _, _, _, metrics = harness(partial, reference, role="final")
    assert metrics["beats_reference"] == 0
    assert feedback.private_metrics(metrics, beat_spec_path=TASK / "judge/beat_spec.json")["beats_reference"] == 0


def test_early_ended_runs_are_valid_and_keep_their_coverage(harness):
    def candidate(i, n):
        status = ("compute_budget_exhausted", "solver_attempt_budget_exhausted", "negate_request_budget_exhausted",
                  "run_time_budget_exhausted", "strategy_returned")[n % 5] if n < 8 else "budget_exhausted"
        return run(900 if n < 8 else 1586, status=status)
    score, _, message, metrics = harness(candidate, spread(1586))
    assert message == "evaluation complete" and score > 0
    assert metrics["dev_runs_ended_early"] == 8
    assert metrics["dev_branches_covered_mean"] == round((8 * 900 + 22 * 1586) / 30, 3)
    assert metrics["beats_reference"] == 0
    # A run that ended before a checkpoint counts there with its final coverage.
    assert metrics["dev_coverage_curve"][-1] == [4000, round((8 * 900 + 22 * 1586) / 30, 2)]


def test_killed_executions_are_reported_not_penalised_twice(harness):
    _, _, _, metrics = harness(lambda i, n: run(1586, killed=n), spread(1586))
    assert metrics["dev_killed_executions_mean"] == round(sum(range(30)) / 30, 2)


@pytest.mark.parametrize("status,code", [
    ("strategy_fault:crashed", "strategy_fault_crashed"),
    ("strategy_fault:sandbox_kill", "strategy_fault_sandbox_kill"),
    ("strategy_fault:protocol", "strategy_fault_protocol"),
    ("strategy_fault:api_misuse", "strategy_fault_api_misuse"),
])
def test_strategy_fault_voids_the_submission(harness, status, code):
    candidate = lambda i, n: run(5000, overhead_seconds=1.0, status=status if n == 17 else "budget_exhausted")  # noqa: E731
    score, unbounded, message, metrics = harness(candidate, spread(1586))
    assert (score, unbounded, message) == (0.0, 0.0, "invalid candidate: " + code)
    assert metrics["beats_reference"] == 0
    assert not any(metrics["beat_spec_workload_passes"][0]["metrics"].values())
    assert feedback.public_metrics(metrics, beat_spec_path=TASK / "judge/beat_spec.json")


def test_reference_or_harness_failure_is_never_a_candidate_result(harness):
    with pytest.raises(e.EvaluatorFailure):
        harness(spread(1586), lambda i, n: run(1586, status="compute_budget_exhausted" if n == 3 else "budget_exhausted"))
    with pytest.raises(e.EvaluatorFailure):
        harness(lambda i, n: run(0, status="harness_failure"), spread(1586))
    with pytest.raises(e.EvaluatorFailure):
        harness(lambda i, n: run(0, status="harness_error:sandbox_unavailable"), spread(1586))


def test_invalid_patches_are_invalid_candidates(harness, tmp_path, monkeypatch):
    outside = tmp_path / "outside.patch"
    outside.write_text("diff --git a/src/run_crest/search_harness.cc b/src/run_crest/search_harness.cc\n"
                       "--- a/src/run_crest/search_harness.cc\n+++ b/src/run_crest/search_harness.cc\n"
                       "@@ -1,1 +1,1 @@\n-// Copyright (c) 2008, Jacob Burnim (jburnim@cs.berkeley.edu)\n+// x\n")
    monkeypatch.setenv("FRONTIER_SUBMISSION_ROLE", "development")
    assert e.evaluate(str(outside))[2] == "invalid candidate: invalid_patch"
    assert e.evaluate(str(tmp_path / "missing.patch"))[2] == "invalid candidate: missing_submission"


def test_policy_violation_reports_its_category(monkeypatch, tmp_path):
    policy = MODULES["solution_policy"]

    def refuse(*args, **kwargs):
        raise policy.PolicyViolation("forbidden_symbol", "time")
    tools = (Path("/nonexistent"), runner, types.SimpleNamespace(
        environment=lambda tree, out, tmp=None: ({}, None), build_strategy=refuse, BuildError=RuntimeError), policy)
    monkeypatch.setattr(e, "load_tools", lambda: tools)
    monkeypatch.setenv("FRONTIER_SUBMISSION_ROLE", "final")
    score, _, message, metrics = e.evaluate(str(TASK / "judge/reference.patch"))
    assert (score, message) == (0.0, "invalid candidate: forbidden_symbol")
    assert len(metrics["beat_spec_workload_passes"]) == 2


def test_development_feedback_is_exactly_the_declared_candidate_only_set(harness):
    _, _, _, metrics = harness(spread(1500), lambda i, n: run(1234567, overhead_seconds=7654.321))
    visible = feedback.public_metrics(metrics, beat_spec_path=TASK / "judge/beat_spec.json")
    declared = {entry["name"] for entry in SPEC["solver_feedback"]}
    assert set(visible) == declared | {"beat_spec_progress"}
    text = json.dumps({k: v for k, v in visible.items() if k != "beat_spec_progress"})
    assert "1234567" not in text and "7654" not in text
    assert set(visible["beat_spec_progress"]) == {"branches_covered", "search_overhead_seconds"}
    assert visible["beat_spec_progress"]["branches_covered"]["total"] == 1


def test_score_is_reference_free_and_monotone(harness):
    program = e.PROGRAMS["development"][0]
    assert e.program_score(program, program["baseline_mean_coverage"]) == 0.0
    assert e.program_score(program, program["baseline_mean_coverage"] - 50) == 0.0
    assert e.program_score(program, program["branch_count"]) == 100.0
    low = harness(spread(1500), spread(1586))[0]
    same = harness(spread(1500), spread(1900))[0]
    high = harness(spread(1650), spread(1586))[0]
    assert low == same and high > low


def test_seeds_keys_and_interleaving(monkeypatch):
    programs = e.PROGRAMS["final"]
    keys = {(p["id"], s): e.relabel_key(p, s) for p in programs for s in e.seeds_of(p)}
    assert len(set(keys.values())) == 2 * K and all(0 <= k < 2 ** 63 for k in keys.values())
    assert e.relabel_key(programs[0], 2001) == e.relabel_key(programs[0], 2001)
    captured = {}

    def run_many(jobs, workers):
        captured["jobs"], captured["workers"] = jobs, workers
        return {key: run(1) for key, _ in jobs}
    fake_runner = types.SimpleNamespace(ProgramSpec=runner.ProgramSpec, run_many=run_many)
    e.measure((Path("/t"), fake_runner, None, None), programs, {"candidate": "c", "reference": "r"}, Path("/s"), {})
    jobs = captured["jobs"]
    assert captured["workers"] == 8 and len(jobs) == 2 * 2 * K
    assert [key[1] for key, _ in jobs[:4]] == ["candidate", "reference", "reference", "candidate"]
    for (index, role, seed), kwargs in jobs:
        assert kwargs["relabel_key"] == keys[(programs[index]["id"], seed)]
        assert kwargs["run_wall_seconds"] == programs[index]["run_wall_seconds"]
        assert kwargs["strategy_host"] == {"candidate": "c", "reference": "r"}[role]


def test_program_manifest_does_not_reach_the_solver_tree():
    public = TASK / "agent-base/frontier/task"
    names = {p.name for p in public.rglob("*")}
    assert "programs.json" not in names and "reference.patch" not in names and "gate.py" not in names
    statement = (public / "README.md").read_text().lower() + (TASK / "judge/readme").read_text().lower()
    for word in ("vim", "sed 3", "sed-3"):
        assert word not in statement
