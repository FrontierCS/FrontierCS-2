"""Implement CREST's concolic search strategy so that it covers more branches within 4000 executions, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 8000, "memory_bytes": 16 * GiB},
    judge = {"cpu_millis": 8000, "memory_bytes": 32 * GiB},
    judge_leases = {
        "search": {
            "trusted": True,
            "machine": {"cpu_millis": 8000, "memory_bytes": 32 * GiB},
            "instances": 1,
            "instance_seconds": {"train": 3600, "eval": 10500},
            "transfer_bytes": 512 * MiB,
        },
    },
    train_calls = 100,
    train_cumulative_seconds = 18000,
    judge_timeout_seconds = 10800,
    submission_max_bytes = 256 * KiB,
    visible_metrics = [
        "dev_branches_covered_mean",
        "dev_coverage_curve",
        "dev_search_overhead_seconds_mean",
        "dev_solver_attempts_mean",
        "dev_runs_ended_early",
        "dev_killed_executions_mean",
    ],
    source = [
        paper(
            "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-123.pdf",
            title = "Heuristics for Scalable Dynamic Test Generation",
            venue = "ASE 2008",
            authors = ["Jacob Burnim", "Koushik Sen"],
            year = 2008,
            doi = "10.1109/ASE.2008.69",
            summary = (
                "Design a search strategy for the CREST concolic tester that covers more " +
                "branches."
            ),
        ),
    ],
    display_name = "Search Strategy for Concolic Testing",
    category = "Software Engineering",
)
