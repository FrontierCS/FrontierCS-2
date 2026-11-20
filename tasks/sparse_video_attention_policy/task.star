"""Design a sparse-attention policy for Sparse VideoGen text-to-video diffusion that keeps the video's quality while reducing GPU time, ported from FrontierCS."""

load("//lib/fcs.star", "GPU_RUNNER", "H100", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 3000, "memory_bytes": 12 * GiB},
    agent_lease = {
        "machine": GPU_RUNNER | {"gpu": H100},
        "instances": 1,
        "instance_seconds": 18000,
        "cumulative_seconds": 54000,
        "commands": 1024,
        "transfer_bytes": 512 * MiB,
    },
    # The judge runs on four H100s at once (`[gpu.verifier]`, `judge/evaluator.py`'s
    # `EXPECTED_VERIFIER_GPUS`), with a GPU runner's CPUs and memory, as FrontierCS starts it.
    judge = GPU_RUNNER | {"gpu": {"count": 4, "model": H100["model"]}},
    # The evaluator reads its GPU count from `FRONTIER_VERIFIER_GPU_COUNT` (`judge/evaluator.py`,
    # `_gpu_count`), which FrontierCS's Driver set.
    judge_env = ["FRONTIER_VERIFIER_GPU_COUNT=4"],
    train_calls = 25,
    train_cumulative_seconds = 225000,
    judge_timeout_seconds = 12600,
    submission_max_bytes = 256 * KiB,
    visible_metrics = ["e2e_latency_seconds", "psnr_db", "ssim", "lpips"],
    contexts = {"runtime": "runtime/Dockerfile"},
    source = [
        paper(
            "https://arxiv.org/abs/2603.08982",
            title = "SVG-EAR: Parameter-Free Linear Compensation for Sparse Video Generation via Error-aware Routing",
            venue = "ECCV 2026",
            summary = (
                "Design a sparse-attention policy for Sparse VideoGen text-to-video diffusion " +
                "that preserves quality while reducing GPU time."
            ),
        ),
    ],
    display_name = "Sparse Video Attention Policy",
    category = "MLSys",
)
