"""Reconstruct Polygon's search for small databases on which two SQL queries differ, validated and fast, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 2000, "memory_bytes": 8 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 8 * GiB},
    judge_leases = {
        "search": {
            "trusted": True,
            "machine": {"cpu_millis": 2000, "memory_bytes": 8 * GiB},
            "instances": 1,
            "instance_seconds": {"train": 1500, "eval": 1500},
            "transfer_bytes": 64 * MiB,
        },
    },
    train_calls = 100,
    train_cumulative_seconds = 12000,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 256 * KiB,
    visible_metrics = ["refuted_pairs", "wall_time_seconds", "search_iterations", "backtracks"],
    source = [
        paper(
            "https://arxiv.org/pdf/2504.06542v1",
            title = "Polygon: Symbolic Reasoning for SQL using Conflict-Driven Under-Approximation Search",
            venue = "PLDI 2025",
            authors = ["Pinhan Zhao", "Yuepeng Wang", "Xinyu Wang"],
            year = 2025,
            doi = "10.1145/3729303",
            summary = (
                "Find small counterexample databases that distinguish non-equivalent SQL queries " +
                "faster with Polygon."
            ),
        ),
    ],
    display_name = "SQL Counterexample Search",
    category = "Programming Language",
)
