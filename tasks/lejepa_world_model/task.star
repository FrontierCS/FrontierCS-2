"""Implement a LeJEPA training objective whose encoder recovers a synthetic world's hidden latent state, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 2000, "memory_bytes": 12 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 2 * GiB},
    train_calls = 8,
    train_cumulative_seconds = 3200,
    judge_timeout_seconds = 400,
    submission_max_bytes = 256 * KiB,
    visible_metrics = [
        "r2_zh",
        "r2_hz",
        "orth_err_normalized",
        "epsilon",
        "delta",
        "procrustes_mse",
    ],
    source = [
        paper(
            "https://arxiv.org/abs/2605.26379",
            title = "When Does LeJEPA Learn a World Model?",
            venue = "arXiv 2026",
            summary = (
                "Recover a self-supervised (LeJEPA-style) training objective whose encoder " +
                "recovers the hidden latent state of a synthetic world."
            ),
        ),
    ],
    display_name = "Latent Objective Recovery",
    category = "Machine Learning",
)
