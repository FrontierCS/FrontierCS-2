"""Implement the action sampler of a Push-T diffusion policy on CPU for more reward at lower latency, ported from FrontierCS."""

load("//lib/fcs.star", "XEON_8375C", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = package_agent(ubuntu_agent_base),
    agent = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 4 * GiB, "cpu_model": XEON_8375C},
    train_calls = 100,
    train_cumulative_seconds = 372000,
    judge_timeout_seconds = 3720,
    submission_max_bytes = 256 * KiB,
    visible_metrics = {
        "dev_mean_episode_max_reward": "Mean episode-maximum reward over all train episodes (0 to 1).",
        "dev_episode_success_rate": (
            "Fraction of train episodes that reach the original environment\n" +
            "  success."
        ),
        "dev_mean_predict_action_seconds": (
            "Total predict_action time divided by the number of calls,\n" +
            "  in seconds."
        ),
        "dev_total_workload_wall_seconds": (
            "Sum over the train workloads of each workload's median\n" +
            "  episode-process seconds."
        ),
    },
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
