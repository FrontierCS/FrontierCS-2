"""Implement the differentiable population statistics that post-train an image generator against Frechet distances, ported from FrontierCS."""

load("//lib/fcs.star", "A100_80", "GPU_RUNNER", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 3000, "memory_bytes": 12 * GiB},
    agent_lease = {
        "machine": GPU_RUNNER | {"gpu": A100_80},
        "instances": 1,
        "instance_seconds": 7200,
        "cumulative_seconds": 7200,
        "commands": 1024,
        "transfer_bytes": 8 * MiB,
    },
    judge = GPU_RUNNER | {"gpu": A100_80},
    train_calls = 3,
    train_cumulative_seconds = 21600,
    judge_timeout_seconds = 10800,
    submission_max_bytes = 256 * KiB,
    visible_metrics = [
        "development_mean_inception_fdr",
        "development_mean_convnext_fdr",
        "development_mean_dinov2_fdr",
        "development_mean_mae_fdr",
        "development_mean_siglip_fdr",
        "development_mean_clip_fdr",
    ],
    contexts = {"runtime": "runtime/Dockerfile"},
    source = [
        paper(
            "https://arxiv.org/abs/2604.28190",
            title = "Representation Fréchet Loss for Visual Generation",
            venue = "arXiv 2026",
            summary = (
                "Implement differentiable population mean/covariance statistics (FeatureQueue) " +
                "for post-training an image generator with representation Fréchet distances."
            ),
        ),
    ],
    display_name = "Differentiable Population Statistics",
    category = "Machine Learning",
)
