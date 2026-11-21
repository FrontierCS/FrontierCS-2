"""Implement the Polar Express iterations that approximate the matrix polar factor with matrix products, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = package_agent(ubuntu_agent_base),
    agent = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 24000,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 1 * MiB,
    visible_metrics = {
        "worst_case_scalar_error": (
            "Mean over the six original development workloads of the maximum\n" +
            "  scalar grid error."
        ),
        "relative_polar_factor_error": (
            "Mean over the six original development workloads of the\n" +
            "  relative Frobenius polar error."
        ),
        "paper_regime_relative_polar_factor_error": (
            "Mean over the six paper-regime development members\n" +
            "  of the relative Frobenius polar error."
        ),
        "spectral_polar_error": (
            "Mean over all twelve development workloads of the spectral-norm polar\n" +
            "  error."
        ),
        "polar_cosine_similarity": (
            "Mean over all twelve development workloads of the Frobenius cosine\n" +
            "  similarity to the polar factor."
        ),
        "top_subspace_polar_error": (
            "Mean over all twelve development workloads of the top-subspace\n" +
            "  relative Frobenius error."
        ),
    },
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
