"""Design a sparse-attention policy for Sparse VideoGen text-to-video diffusion that keeps the video's quality while reducing GPU time, ported from FrontierCS."""

load("//lib/fcs.star", "GPU_RUNNER", "H100", "fcs_task", "paper", "ubuntu_agent")
load("//tasks/sparse_video_attention_policy/images.star", "gpu_runner_image", "gpu_verifier_image", "reference_image")

fcs_task(
    agent_recipe = ubuntu_agent("ca-certificates git python3 tar", executable = "*.sh"),
    agent_runner_image = gpu_runner_image,
    reference_image = reference_image,
    verifier_image = gpu_verifier_image,
    agent = {"cpu_millis": 3000, "memory_bytes": 12 * GiB},
    agent_env = ["FRONTIER_GPU_SECONDS=18000"],
    agent_lease = {
        "machine": GPU_RUNNER | {"gpu": H100},
        "instances": 1,
        "instance_seconds": 18000,
        "cumulative_seconds": 54000,
        "commands": 1024,
        "transfer_bytes": 512 * MiB,
    },
    # The judge runs on four H100s at once (`[gpu.verifier]`, `judge/evaluator.py`'s
    # `EXPECTED_VERIFIER_GPUS`), one workload worker on each, so it has four GPU runners' CPUs
    # and memory: a sandbox's memory is its hard limit, and the four workers' models exceed one
    # runner's 64 GiB.
    judge = {
        "cpu_millis": 4 * GPU_RUNNER["cpu_millis"],
        "memory_bytes": 4 * GPU_RUNNER["memory_bytes"],
        "gpu": {"count": 4, "model": H100["model"]},
    },
    # The evaluator reads its GPU count from `FRONTIER_VERIFIER_GPU_COUNT` (`judge/evaluator.py`,
    # `_gpu_count`), which FrontierCS's Driver set.
    judge_env = ["FRONTIER_VERIFIER_GPU_COUNT=4"],
    train_calls = 25,
    train_cumulative_seconds = 225000,
    judge_timeout_seconds = 12600,
    submission_max_bytes = 256 * KiB,
    visible_metrics = {
        "e2e_latency_seconds": (
            "Candidate latency per train workload, in seconds. Each metric list is\n" +
            "  sorted independently; positions match neither other metrics nor workloads."
        ),
        "psnr_db": (
            "Candidate PSNR per train workload, in dB, sorted; an exact match is reported as\n" +
            "  \"positive_infinity\"."
        ),
        "ssim": "Candidate SSIM per train workload, sorted.",
        "lpips": "Candidate LPIPS per train workload, sorted.",
    },
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
