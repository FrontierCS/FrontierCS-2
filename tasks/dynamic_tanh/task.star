"""Implement Dynamic Tanh as the replacement for every LayerNorm in a ViT, faster than the authors' version, ported from FrontierCS."""

load("//lib/fcs.star", "A100_80", "GPU_RUNNER", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 3000, "memory_bytes": 12 * GiB},
    agent_lease = {
        "machine": GPU_RUNNER | {"gpu": A100_80},
        "instances": 1,
        "instance_seconds": 900,
        "cumulative_seconds": 1800,
        "commands": 1024,
        "transfer_bytes": 8 * MiB,
    },
    judge = GPU_RUNNER | {"gpu": A100_80},
    train_calls = 3,
    train_cumulative_seconds = 2700,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 256 * KiB,
    visible_metrics = [
        "top1_accuracy_pct",
        "training_latency_ms",
        "inference_latency_ms",
    ],
    source = [
        paper(
            "https://arxiv.org/abs/2503.10622",
            title = "Transformers without Normalization",
            venue = "CVPR 2025",
            summary = (
                "Train ViT image classifiers whose normalization layers are replaced by Dynamic " +
                "Tanh-style element-wise mappings without losing accuracy."
            ),
        ),
    ],
    display_name = "Learned Transformer Layer Mapping",
    category = "Machine Learning",
)
