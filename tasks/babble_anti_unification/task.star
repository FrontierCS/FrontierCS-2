"""Supply BABBLE's library-learning candidates and state pairing so that program corpora shrink further, faster, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 2000, "memory_bytes": 8 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 2 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 1800,
    judge_timeout_seconds = 600,
    submission_max_bytes = 256 * KiB,
    visible_metrics = [
        "compression_ratio",
        "runtime_seconds",
    ],
    source = [
        paper(
            "https://arxiv.org/abs/2212.04596",
            title = "babble: Learning Better Abstractions with E-Graphs and Anti-unification",
            venue = "POPL 2023",
            summary = (
                "Improve BABBLE's library learning so refactored program corpora become smaller."
            ),
        ),
        artifacts(
            "https://github.com/dcao/babble",
            commit = "09ff03fd485a3e24178c1d706a5944a230123ba1",
        ),
    ],
    display_name = "Reusable Library Candidates",
    category = "Programming Language",
)
