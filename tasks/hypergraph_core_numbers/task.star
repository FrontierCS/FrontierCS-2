"""Compute exact hypergraph core numbers in Hygra as fast as possible, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 8000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 8000, "memory_bytes": 4 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 30000,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 256 * KiB,
    visible_metrics = [
        "candidate_runtime_seconds_median",
        "candidate_sequential_runtime_seconds_median",
    ],
    source = [
        paper(
            "https://people.csail.mit.edu/jshun/hygra.pdf",
            title = "Practical Parallel Hypergraph Algorithms",
            venue = "PPoPP 2020",
            authors = ["Julian Shun"],
            year = 2020,
            doi = "10.1145/3332466.3374527",
            summary = "Speed up hypergraph k-core decomposition in the Hygra/Ligra framework.",
        ),
        artifacts(
            "https://github.com/jshun/ligra",
            commit = "8763202ef09ebf8c5f7d7787cf72f3ca55efbe4c",
        ),
    ],
    display_name = "Hypergraph Core Numbers",
    category = "HPC",
)
