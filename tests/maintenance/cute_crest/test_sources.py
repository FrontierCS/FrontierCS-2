"""Static checks of what the images fetch and build, and of the one place
each limit is defined."""
import deployment_test_guard  # noqa: F401  # isort: skip

import json
import re

from tests.maintenance.cute_crest.runtime_imports import MODULES, REPO_TREE, TASK

runner = MODULES["crest_runner"]
gate = MODULES["gate"]
SOURCES = json.loads((TASK / "judge/benchmark_sources.json").read_text())
PROGRAMS = json.loads((TASK / "judge/programs.json").read_text())
POLICY = json.loads((TASK / "policy.json").read_text())
STATEMENT = (TASK / "judge/readme").read_text()


def test_sources_are_pinned_public_and_licensed():
    for source in SOURCES["sources"]:
        assert source["url"].startswith("https://")
        assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
        assert set(source["licences"]) == set(source["provides"])
    crest = SOURCES["sources"][0]
    assert "3e5ececd3793c88f18af47853236af34ffde4569" in crest["url"]
    fetch = (TASK / "agent-base/frontier/task/bin/fetch_benchmark.py").read_text()
    assert crest["url"] in fetch and crest["sha256"] in fetch
    # The solver image keeps the development program only.
    assert "benchmarks/grep-2.2/" in fetch and "vim" not in fetch.lower() and "sed" not in fetch.lower()


def test_no_program_source_is_committed():
    committed = {p.name for p in TASK.rglob("*") if p.is_file()}
    assert not {"grep.c", "sed.c", "vim.h", "replace.c"} & committed
    assert not [n for n in committed if n.endswith((".tar.gz", ".tgz")) and n != "z3.tar.gz"]


def test_program_manifest():
    assert [p["id"] for p in PROGRAMS["development"]] == ["dev"]
    assert len(PROGRAMS["final"]) == 2
    programs = PROGRAMS["development"] + PROGRAMS["final"]
    assert len({p["directory"] for p in programs}) == 3 and len({p["seed_base"] for p in programs}) == 3
    for p in programs:
        assert 0 < p["baseline_mean_coverage"] < p["branch_count"]
    assert not {p["argv"][0] for p in PROGRAMS["development"]} & {p["argv"][0] for p in PROGRAMS["final"]}


def test_images_build_what_the_evaluator_reads():
    verifier = (TASK / "verifier/Dockerfile").read_text()
    assert "python3 /opt/frontier/task/build_benchmarks.py /opt/crest-task" in verifier
    assert "libncurses-dev" in verifier and "patch" in verifier
    agent = (TASK / "agent-base/Dockerfile").read_text()
    assert "fetch_benchmark.py /frontier/task/benchmark-src/grep-2.2" in agent
    assert "/frontier/task/benchmark-src/grep-2.2" in (REPO_TREE / "frontier_build.py").read_text()


def test_harness_tree_is_identical_on_both_sides():
    public = TASK / "agent-base/frontier/task/repo"
    names = lambda root: sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())  # noqa: E731
    assert [n for n in names(public) if n != "README.md"] == names(REPO_TREE)
    for name in names(REPO_TREE):
        assert (public / name).read_bytes() == (REPO_TREE / name).read_bytes(), name
    # The reference search of the original tester is not in the public tree.
    assert not (public / "src/run_crest/concolic_search.cc").exists()
    text = "".join(p.read_text(errors="replace") for p in (public / "src").rglob("*") if p.is_file())
    assert "CfgHeuristicSearch" not in text and "UpdateBranchDistances" not in text


def test_statement_states_the_limits_the_code_enforces():
    for value in (runner.BUDGET_ITERATIONS, runner.EXEC_TIMEOUT_SECONDS):
        assert re.search(r"\b%d\b" % value, STATEMENT), value
    assert "{:,}".format(runner.MAX_SOLVER_ATTEMPTS) in STATEMENT
    assert "{:,}".format(runner.MAX_NEGATE_REQUESTS) in STATEMENT
    assert "%d seconds" % PROGRAMS["development"][0]["run_wall_seconds"] in STATEMENT
    assert runner.RUN_WALL_SECONDS == PROGRAMS["development"][0]["run_wall_seconds"]
    assert "sqrt(2 / %d)" % gate.K_SEEDS in STATEMENT and "3 * s_cov" in STATEMENT and "3 * s_ovh" in STATEMENT
    assert "%d-second timeout" % POLICY["quota"]["eval_timeout_seconds"] in STATEMENT
    assert (TASK / "agent-base/frontier/task/repo/README.md").read_text() == STATEMENT.replace("/app", "/workspace/repo")


FIXED_SECONDS = 600   # judge start, two strategy builds, scratch clean-up


def worst_case_seconds(programs):
    """Every candidate and every reference run of every program at its wall
    budget plus the harness's stopping slack, eight at a time."""
    runs = 2 * gate.K_SEEDS
    per_run = sum(p["run_wall_seconds"] + runner.HARD_TIMEOUT_SLACK_SECONDS for p in programs)
    return runs * per_run / 8 + FIXED_SECONDS


def test_every_judgment_fits_the_timeout_with_every_run_at_its_caps():
    timeout = POLICY["quota"]["eval_timeout_seconds"]
    assert timeout == 10800
    assert worst_case_seconds(PROGRAMS["final"]) <= 0.95 * timeout
    assert worst_case_seconds(PROGRAMS["development"]) <= 0.95 * timeout
    for p in PROGRAMS["development"] + PROGRAMS["final"]:
        # The reference must never meet the budget: at least 1.15 times its slowest measured run.
        assert p["run_wall_seconds"] >= 1.15 * p["reference_max_run_wall_seconds"]
        assert p["reference_run_wall_seconds"] < p["reference_max_run_wall_seconds"]
    assert POLICY["verifier"]["cpu_millis"] == 8000 and POLICY["verifier"]["memory_bytes"] == 32 * 1024 ** 3
    assert POLICY["agent"]["cpu_millis"] == 8000 and POLICY["agent"]["memory_bytes"] == 16 * 1024 ** 3
