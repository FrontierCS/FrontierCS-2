"""Supply the l-DeTok tokenizer's latent corruption and masked-token behaviour for better, faster reconstructions, ported from FrontierCS."""

load("//lib/fcs.star", "A100_80", "GPU_RUNNER", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 3000, "memory_bytes": 12 * GiB},
    agent_lease = {
        "machine": GPU_RUNNER | {"gpu": A100_80},
        "instances": 1,
        "instance_seconds": 5400,
        "cumulative_seconds": 5400,
        "commands": 1024,
        "transfer_bytes": 8 * MiB,
    },
    judge = GPU_RUNNER | {"gpu": A100_80},
    train_calls = 6,
    train_cumulative_seconds = 32400,
    judge_timeout_seconds = 5400,
    submission_max_bytes = 256 * KiB,
    visible_metrics = [
        "development_mean_ordinary_quantized_png_psnr_db",
        "development_mean_ema_quantized_png_psnr_db",
        "development_mean_ordinary_quantized_png_lpips",
        "development_mean_ema_quantized_png_lpips",
        "development_mean_method_process_wall_seconds",
    ],
    contexts = {"runtime": "runtime/Dockerfile"},
    source = [
        paper(
            "https://arxiv.org/abs/2507.15856",
            title = "Latent Denoising Makes Good Tokenizers",
            venue = "ICLR 2026",
            summary = "Train a more robust image tokenizer by denoising latent tokens.",
        ),
    ],
    display_name = "Robust Image Tokenizer",
    category = "Machine Learning",
)
