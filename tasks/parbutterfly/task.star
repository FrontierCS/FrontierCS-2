"""Implement a fast exact butterfly count for ParButterfly on bipartite graphs, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper", "ubuntu_agent", "ubuntu_verifier")

fcs_task(
    agent_recipe = ubuntu_agent("build-essential ca-certificates git python3 tar", executable = "*.sh"),
    verifier_base = ubuntu_verifier("build-essential ca-certificates git python3"),
    agent = {"cpu_millis": 4000, "memory_bytes": 8 * GiB},
    judge = {"cpu_millis": 4000, "memory_bytes": 8 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 40000,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 1 * MiB,
    visible_metrics = {
        "dev_counting_seconds_median": (
            "Median over the development graphs of the candidate's per-graph\n" +
            "  median counting time, in seconds."
        ),
        "dev_counting_seconds_p90": (
            "Nearest-rank 90th percentile of those per-graph median counting\n" +
            "  times, in seconds."
        ),
    },
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
