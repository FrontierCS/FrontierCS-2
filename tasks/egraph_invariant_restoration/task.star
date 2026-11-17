"""Restore the e-graph rebuilding pipeline of egg and make equality saturation fast, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 8000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 8000, "memory_bytes": 4 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 120000,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 1 * MiB,
    visible_metrics = [
        "candidate_dev_wall_clock_seconds_median",
        "candidate_dev_congruence_seconds_median",
    ],
    mode_stages = False,
    source = [
        paper(
            "https://arxiv.org/pdf/2004.03082",
            title = "egg: Fast and Extensible Equality Saturation",
            venue = "POPL 2021",
            authors = [
                "Max Willsey",
                "Chandrakana Nandi",
                "Yisu Remy Wang",
                "Oliver Flatt",
                "Zachary Tatlock",
                "Pavel Panchekha",
            ],
            year = 2021,
            summary = "Speed up invariant restoration (rebuilding) in the egg e-graph library.",
        ),
        artifacts(
            "https://github.com/egraphs-good/egg",
            commit = "2f31b28e3f9d78e02273b6c6d4201b5b0720b343",
        ),
    ],
    display_name = "E-Graph Invariant Restoration",
    category = "Programming Language",
)
