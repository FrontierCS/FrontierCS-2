"""Implement the Polar Express iterations that approximate the matrix polar factor with matrix products, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 24000,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 1 * MiB,
    visible_metrics = [
        "worst_case_scalar_error",
        "relative_polar_factor_error",
        "paper_regime_relative_polar_factor_error",
        "spectral_polar_error",
        "polar_cosine_similarity",
        "top_subspace_polar_error",
    ],
    source = [
        paper(
            "https://arxiv.org/abs/2505.16932",
            title = "The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm",
            venue = "ICLR 2026",
            summary = (
                "Compute optimal polynomial iterations that approximate the matrix polar factor."
            ),
        ),
    ],
    display_name = "Polar-Factor Approximation",
    category = "Machine Learning",
)
