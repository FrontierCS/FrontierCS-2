"""Train a better Unitree Go1 joystick locomotion policy than the reference in a fixed 720-second GPU budget, ported from FrontierCS."""

load("//lib/fcs.star", "GPU_RUNNER", "H100", "artifacts", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 3000, "memory_bytes": 12 * GiB},
    agent_lease = {
        "machine": GPU_RUNNER | {"gpu": H100},
        "instances": 1,
        "instance_seconds": 14400,
        "cumulative_seconds": 14400,
        "commands": 1024,
        "transfer_bytes": 8 * MiB,
    },
    judge = GPU_RUNNER | {"gpu": H100},
    train_calls = 3,
    train_cumulative_seconds = 16200,
    judge_timeout_seconds = 14400,
    submission_max_bytes = 256 * KiB,
    visible_metrics = [
        "development_mean_episode_reward",
        "development_mean_tracking_lin_vel",
        "development_mean_tracking_ang_vel",
        "development_mean_energy",
        "development_mean_action_rate",
    ],
    contexts = {"runtime": "runtime/Dockerfile"},
    reference_patch = False,
    noop_patch_file = "frontier_solution/train.py",
    source = [
        paper(
            "https://arxiv.org/abs/2502.08844",
            title = "Demonstrating MuJoCo Playground",
            venue = "RSS 2025",
            summary = (
                "Train a Unitree Go1 joystick locomotion policy in MuJoCo Playground within a " +
                "fixed GPU budget."
            ),
        ),
        artifacts(
            "https://github.com/google-deepmind/mujoco_playground",
            commit = "254c38f9c240012518dbe61d1267bc375ff827e4",
        ),
    ],
    display_name = "Go1 Joystick Policy Training",
    category = "Robotics",
)
