"""Integration tests of the real harness: budgets, early ends, the sandbox.

They need a built tree and the instrumented development program, as
judge/build_benchmarks.py produces them (OUT/tree, OUT/build, OUT/bench/dev):

    CUTE_CREST_BUILT_ROOT=OUT pytest tests/maintenance/cute_crest/test_harness_limits.py

Without that variable they are skipped.  Every run here is short and capped.
"""
import deployment_test_guard  # noqa: F401  # isort: skip

import os
from pathlib import Path
import subprocess
import time

import pytest
from tests.maintenance.cute_crest.runtime_imports import MODULES, TASK

BUILT = os.environ.get("CUTE_CREST_BUILT_ROOT")
pytestmark = pytest.mark.skipif(not BUILT, reason="set CUTE_CREST_BUILT_ROOT to a built tree")
runner = MODULES["crest_runner"]
fb = MODULES["frontier_build"]

HEAD = '#include <vector>\n#include "strategy/strategy_api.h"\nnamespace frontier {\nnamespace {\n' \
       'class S : public Strategy { public: virtual void Run(Host& host) {\n'
TAIL = '\n} };\n}\nStrategy* MakeStrategy() { return new S(); }\n}\n'
STRATEGIES = {
    "restart": "while (true) host.Release(host.Restart());",
    "busy": "host.Release(host.Restart()); volatile unsigned long x = 0; while (true) x = x + 1;",
    "spam": """const Execution* e = host.Restart();
        for (size_t k = 0; k < e->constraint_pos.size(); k++) {
          const Execution* r = host.Negate(e, k);
          if (r) { host.Release(r); continue; }
          while (true) (void)host.Negate(e, k);   // infeasible: a solver attempt, no iteration
        }""",
    "returns": "host.Release(host.Restart());",
    "hoard": "while (true) (void)host.Restart();",
    "memory": "host.Release(host.Restart()); std::vector<std::vector<char> > v;"
              " while (true) v.push_back(std::vector<char>(1 << 24, 1));",
    "by_id": """const Execution* e = host.Restart();
        while (true) {
          if (e->constraint_pos.empty()) { host.Release(e); e = host.Restart(); continue; }
          size_t k = static_cast<size_t>(e->path[e->constraint_pos[0]]) % e->constraint_pos.size();
          const Execution* r = host.Negate(e, k);
          host.Release(e);
          e = r ? r : host.Restart();
        }""",
}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = Path(BUILT)
    tree = root / "tree"
    work = tmp_path_factory.mktemp("strategies")
    # The built root may be read-only (it is in the verifier image).
    env, _ = fb.environment(tree, root / "build", tmp=work / "tmp")
    hosts = {}
    for name, body in STRATEGIES.items():
        directory = work / name / "frontier_solution"
        directory.mkdir(parents=True)
        (directory / "strategy.cc").write_text(HEAD + body + TAIL)
        hosts[name] = fb.build_strategy(tree, directory, work / name / "out", env)
    # A strategy that slipped past the static checks: built without them.
    rogue = work / "rogue"
    rogue.mkdir()
    (rogue / "strategy.cc").write_text(
        '#include "strategy/strategy_api.h"\nextern "C" int open(const char*, int, ...);\n' + HEAD.split("\n", 2)[2]
        + 'host.Release(host.Restart()); (void)open("/etc/passwd", 0);' + TAIL)
    flags = fb.CXX_STRATEGY + ["-I" + str(tree / "src")]
    subprocess.run(flags + ["-c", str(rogue / "strategy.cc"), "-o", str(rogue / "s.o")], check=True)
    subprocess.run(flags + ["-c", str(tree / "src/strategy/strategy_host.cc"), "-o", str(rogue / "h.o")], check=True)
    subprocess.run(flags + ["-static", str(rogue / "h.o"), str(rogue / "s.o"), "-o", str(rogue / "host")], check=True)
    hosts["rogue"] = rogue / "host"
    spec = runner.ProgramSpec(root / "bench/dev", ["grep", "aaaaaaaaaaaaaaaaaaaa", "in.txt"])
    scratch = tmp_path_factory.mktemp("runs")

    def run(name, **kwargs):
        kwargs.setdefault("budget", 60)
        kwargs.setdefault("hard_timeout", 120)
        started = time.time()
        result = runner.run_one(tree, spec, hosts[name], kwargs.pop("seed", 5), scratch, toolchain_env=env, **kwargs)
        result["elapsed"] = time.time() - started
        return result
    return run


def test_budget_is_enforced_exactly(setup):
    result = setup("restart", budget=37)
    assert (result["status"], result["iterations"]) == ("budget_exhausted", 37)
    assert result["covered"] > 0


def test_busy_loop_ends_early_with_partial_coverage(setup):
    result = setup("busy", compute_cpu_seconds=3)
    assert result["status"] == "compute_budget_exhausted"
    assert result["status"] in runner.EARLY_END_STATUSES
    assert result["iterations"] == 1 and result["covered"] > 0
    assert result["elapsed"] < 30
    assert result["children_peak_kb"] < 4 * 1024 * 1024


def test_solver_spam_ends_early_with_partial_coverage(setup):
    # The strategy repeats one infeasible request for ever.  Whether the harness
    # answers it with a solver query or without one, a deterministic cap ends the run.
    result = setup("spam", max_solver_attempts=300, max_negate_requests=2000, budget=4000)
    assert result["status"] in ("solver_attempt_budget_exhausted", "negate_request_budget_exhausted")
    assert result["status"] in runner.EARLY_END_STATUSES
    assert result["solver_attempts"] <= 300 and result["negate_requests"] <= 2000
    assert result["solver_attempts"] == 300 or result["negate_requests"] == 2000
    assert 0 < result["iterations"] < 4000 and result["covered"] > 0
    assert result["elapsed"] < 60
    assert result["children_peak_kb"] < 4 * 1024 * 1024


def test_request_spam_cannot_outlast_the_run(setup):
    # No solver cap at all: the request cap alone stops it, quickly.
    result = setup("spam", max_solver_attempts=0, max_negate_requests=5000, compute_cpu_seconds=0, budget=4000)
    assert result["status"] in ("negate_request_budget_exhausted", "budget_exhausted")
    assert result["negate_requests"] <= 5000 and result["elapsed"] < 110


def test_solver_spam_is_also_stopped_by_the_compute_budget(setup):
    result = setup("spam", compute_cpu_seconds=3, max_solver_attempts=0, max_negate_requests=0, budget=4000,
                   hard_timeout=300)
    assert result["status"] == "compute_budget_exhausted"
    assert result["covered"] > 0 and result["elapsed"] < 290
    assert runner.search_overhead_seconds(result) < 3 + 12   # one bounded query may straddle the cap


def test_busy_loop_ends_at_the_run_wall_budget(setup):
    # Without a compute cap the wall-clock budget alone ends a strategy that
    # computes without asking for anything.
    result = setup("busy", run_wall_seconds=6, compute_cpu_seconds=0, hard_timeout=60)
    assert result["status"] == "run_time_budget_exhausted"
    assert result["status"] in runner.EARLY_END_STATUSES
    assert result["iterations"] == 1 and result["covered"] > 0
    assert result["elapsed"] < 20


def test_a_long_run_ends_at_the_run_wall_budget_with_its_coverage(setup):
    result = setup("restart", run_wall_seconds=8, budget=4000, hard_timeout=60)
    assert result["status"] == "run_time_budget_exhausted"
    assert 0 < result["iterations"] < 4000 and result["covered"] > 0
    assert result["elapsed"] < 30


def test_strategy_that_returns_ends_the_run(setup):
    result = setup("returns")
    assert (result["status"], result["iterations"]) == ("strategy_returned", 1)


@pytest.mark.parametrize("name", ["hoard", "memory", "rogue"])
def test_faults_do_not_hang_and_stay_inside_the_memory_cap(setup, name):
    result = setup(name, budget=4000, compute_cpu_seconds=60)
    assert result["status"].startswith("strategy_fault:"), result["status"]
    assert result["status"] not in runner.COMPLETE_STATUSES
    assert result["elapsed"] < 110
    assert result["children_peak_kb"] < 4 * 1024 * 1024
    if name == "rogue":
        assert result["status"] == "strategy_fault:sandbox_kill"
    if name == "hoard":
        assert result["iterations"] <= 64


def test_runs_are_reproducible_and_ids_depend_on_the_secret_key(setup):
    first = setup("by_id", relabel_key=11, budget=150)
    again = setup("by_id", relabel_key=11, budget=150)
    other = [setup("by_id", relabel_key=key, budget=150)["covered"] for key in (12, 13, 14)]
    assert first["covered"] == again["covered"]
    assert any(c != first["covered"] for c in other)      # an id-dependent choice changes with the key
    plain = {setup("restart", relabel_key=key)["covered"] for key in (11, 12)}
    assert len(plain) == 1                                # an id-blind strategy does not
