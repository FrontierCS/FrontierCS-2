"""Implement a LeJEPA training objective whose encoder recovers a synthetic world's hidden latent state, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = package_agent(ubuntu_agent_base),
    agent = {"cpu_millis": 2000, "memory_bytes": 12 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 2 * GiB},
    train_calls = 8,
    train_cumulative_seconds = 3200,
    judge_timeout_seconds = 400,
    submission_max_bytes = 256 * KiB,
    visible_metrics = {
        "r2_zh": "Mean of r2_zh over the three development workloads.",
        "r2_hz": "Mean of r2_hz over the three development workloads.",
        "orth_err_normalized": "Mean of orth_err_normalized over the three development workloads.",
        "epsilon": "Mean of epsilon over the three development workloads.",
        "delta": "Mean of delta over the three development workloads.",
        "procrustes_mse": "Mean of procrustes_mse over the three development workloads.",
    },
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
