"""Build RAGEN's multi-turn PPO training batches from a language-model agent's Sokoban interactions, ported from FrontierCS."""

load("//lib/fcs.star", "A100_80", "GPU_RUNNER", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 3000, "memory_bytes": 12 * GiB},
    agent_lease = {
        "machine": GPU_RUNNER | {"gpu": A100_80},
        "instances": 1,
        "instance_seconds": 1800,
        "cumulative_seconds": 1800,
        "commands": 1024,
        "transfer_bytes": 8 * MiB,
    },
    judge = {"cpu_millis": 3000, "memory_bytes": 12 * GiB},
    judge_leases = {
        "candidate": {
            "trusted": False,
            "machine": GPU_RUNNER | {"gpu": A100_80},
            "instances": 2,
            "instance_seconds": {"train": 1500, "eval": 1500},
            "transfer_bytes": 8 * MiB,
        },
    },
    train_calls = 3,
    train_cumulative_seconds = 5400,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 256 * KiB,
    visible_metrics = ["success_rate", "average_episodic_reward"],
    contexts = {"runtime": "runtime/Dockerfile"},
    source = [
        paper(
            "https://arxiv.org/abs/2504.20073",
            title = "RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning",
            venue = "arXiv 2025",
            summary = (
                "Build correct multi-turn trajectory batches for RAGEN PPO training of " +
                "language-model agents."
            ),
        ),
    ],
    display_name = "Multi-turn Trajectory Batches",
    category = "Machine Learning",
)
