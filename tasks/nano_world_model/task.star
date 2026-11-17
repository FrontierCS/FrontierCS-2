"""Implement the latent-video sampler of NanoWM so predicted future frames match the recorded ones, ported from FrontierCS."""

load("//lib/fcs.star", "A100_80", "GPU_RUNNER", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 3000, "memory_bytes": 12 * GiB},
    agent_lease = {
        "machine": GPU_RUNNER | {"gpu": A100_80},
        "instances": 1,
        "instance_seconds": 1800,
        "cumulative_seconds": 1800,
        "commands": 1024,
        "transfer_bytes": 512 * MiB,
    },
    judge = {"cpu_millis": 3000, "memory_bytes": 12 * GiB},
    judge_leases = {
        "candidate": {
            "trusted": False,
            "machine": GPU_RUNNER | {"gpu": A100_80},
            "instances": 1,
            "instance_seconds": {"train": 5700, "eval": 5700},
            "transfer_bytes": 512 * MiB,
        },
        "trusted": {
            "trusted": True,
            "machine": GPU_RUNNER | {"gpu": A100_80},
            "instances": 1,
            "instance_seconds": {"train": 1800, "eval": 1800},
            "transfer_bytes": 512 * MiB,
        },
    },
    train_calls = 3,
    train_cumulative_seconds = 10800,
    judge_timeout_seconds = 10200,
    submission_max_bytes = 256 * KiB,
    visible_metrics = ["psnr", "ssim", "lpips"],
    contexts = {"runtime": "runtime/Dockerfile"},
    source = [
        paper(
            "https://arxiv.org/abs/2605.23993",
            title = "Nano World Models: A Minimalist Implementation of Future Video Prediction",
            venue = "arXiv 2026",
            summary = (
                "Speed up future-video sampling in the NanoWM action-conditioned video world " +
                "model."
            ),
        ),
    ],
    display_name = "Future Video Sampling",
    category = "Machine Learning",
)
