"""Implement CREST's concolic search strategy so that it covers more branches within 4000 executions, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = package_agent(ubuntu_agent_base),
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
    visible_metrics = {
        "dev_branches_covered_mean": (
            "Mean over the 30 development runs of the branches covered when\n" +
            "  the run ended."
        ),
        "dev_coverage_curve": (
            "List of [iteration, mean branches covered] at fixed checkpoints over the\n" +
            "  30 development runs; a run that ended before a checkpoint counts with its final coverage."
        ),
        "dev_search_overhead_seconds_mean": (
            "Mean over the 30 development runs of the search overhead\n" +
            "  (strategy CPU plus constraint-solving CPU), in seconds."
        ),
        "dev_solver_attempts_mean": (
            "Mean over the 30 development runs of the constraint-solver\n" +
            "  attempts, feasible and infeasible."
        ),
        "dev_runs_ended_early": (
            "Number of the 30 development runs that ended before the iteration\n" +
            "  budget was spent (a cap was reached or Run returned)."
        ),
        "dev_killed_executions_mean": (
            "Mean over the 30 development runs of the program launches that\n" +
            "  crashed or hit the per-launch time limit."
        ),
    },
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
