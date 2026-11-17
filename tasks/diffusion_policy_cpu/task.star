"""Implement the action sampler of a Push-T diffusion policy on CPU for more reward at lower latency, ported from FrontierCS."""

load("//lib/fcs.star", "XEON_8375C", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 4 * GiB, "cpu_model": XEON_8375C},
    train_calls = 100,
    train_cumulative_seconds = 372000,
    judge_timeout_seconds = 3720,
    submission_max_bytes = 256 * KiB,
    visible_metrics = [
        "dev_mean_episode_max_reward",
        "dev_episode_success_rate",
        "dev_mean_predict_action_seconds",
        "dev_total_workload_wall_seconds",
    ],
    source = [
        paper(
            "https://arxiv.org/abs/2303.04137",
            title = "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion",
            venue = "RSS 2023",
            summary = (
                "Implement the conditional action sampler of a frozen diffusion-policy " +
                "controller for a T-pushing robot task, evaluated closed-loop on CPU."
            ),
        ),
    ],
    display_name = "Closed-loop action sampling",
    category = "Robotics",
)
