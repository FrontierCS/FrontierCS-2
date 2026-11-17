"""Implement a fast exact butterfly count for ParButterfly on bipartite graphs, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 4000, "memory_bytes": 8 * GiB},
    judge = {"cpu_millis": 4000, "memory_bytes": 8 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 40000,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 1 * MiB,
    visible_metrics = [
        "dev_counting_seconds_median",
        "dev_counting_seconds_p90",
    ],
    source = [
        paper(
            "https://jshun.csail.mit.edu/butterfly.pdf",
            title = "Parallel Algorithms for Butterfly Computations",
            venue = "APOCS 2020",
            summary = "Speed up parallel butterfly (2x2 biclique) counting in bipartite graphs.",
        ),
    ],
    display_name = "ParButterfly Exact Counting",
    category = "HPC",
)
