"""Implement Glenside's access-pattern analysis so that equality saturation maps convolutions onto a systolic array, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 12000,
    judge_timeout_seconds = 300,
    submission_max_bytes = 1 * MiB,
    visible_metrics = [
        "development_im2col_discovered_count",
        "development_rewrite_time_seconds_median",
        "development_egraph_node_growth_median",
    ],
    source = [
        paper(
            "https://arxiv.org/abs/2105.09377",
            title = "Pure Tensor Program Rewriting via Access Patterns (Representation Pearl)",
            venue = "MAPS 2021",
            summary = (
                "Find better access-pattern rewrites that map tensor convolutions onto matrix " +
                "hardware in Glenside."
            ),
        ),
    ],
    display_name = "Access-Pattern Rewriting",
    category = "Programming Language",
)
