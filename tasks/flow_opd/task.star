"""Implement the trajectory sampler of an on-policy distillation trainer that teaches Stable Diffusion 3.5 to render text, ported from FrontierCS."""

load("//lib/fcs.star", "GPU_RUNNER", "H100", "artifacts", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 3000, "memory_bytes": 12 * GiB},
    agent_lease = {
        "machine": GPU_RUNNER | {"gpu": H100},
        "instances": 1,
        "instance_seconds": 3000,
        "cumulative_seconds": 3000,
        "commands": 1024,
        "transfer_bytes": 512 * MiB,
    },
    judge = GPU_RUNNER | {"gpu": H100},
    train_calls = 4,
    train_cumulative_seconds = 12000,
    judge_timeout_seconds = 3000,
    submission_max_bytes = 256 * KiB,
    visible_metrics = [
        "dev_eval_reward_ocr",
        "dev_verified_optimizer_updates",
    ],
    source = [
        paper(
            "https://arxiv.org/abs/2605.08063",
            title = "Flow-OPD: On-Policy Distillation for Flow Matching Models",
            venue = "arXiv 2026",
            summary = (
                "Implement the trajectory sampler and SDE step with log-probabilities for " +
                "on-policy distillation of a flow-matching text-to-image model."
            ),
        ),
        artifacts(
            "https://github.com/CostaliyA/Flow-OPD",
            commit = "434ab16911fc580f51e22107151915846222eb6a",
            license = "MIT",
        ),
    ],
    display_name = "OCR Distillation Trajectory Sampling",
    category = "Machine Learning",
)
