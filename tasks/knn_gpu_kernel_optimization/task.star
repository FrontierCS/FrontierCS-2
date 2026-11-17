"""Make exact brute-force k-nearest-neighbour search on one H100 faster than the naive search, ported from FrontierCS."""

load("//lib/fcs.star", "GPU_RUNNER", "H100", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 4000, "memory_bytes": 8 * GiB},
    agent_lease = {
        "machine": GPU_RUNNER | {"gpu": H100},
        "instances": 1,
        "instance_seconds": 10800,
        "cumulative_seconds": 10800,
        "commands": 1024,
        "transfer_bytes": 64 * MiB,
    },
    judge = GPU_RUNNER | {"gpu": H100},
    train_calls = 25,
    train_cumulative_seconds = 10800,
    judge_timeout_seconds = 5400,
    submission_max_bytes = 2 * MiB,
    visible_metrics = [
        "geomean_speedup",
        "latency_ms",
        "worst_ball_recall",
        "worst_exact_recall",
    ],
    source = [
        paper(
            "https://flashml-org.github.io/",
            title = "FlashLib: Bringing Flash Magic to Classical Machine Learning Operators",
            venue = "Blog",
            summary = "Optimize exact brute-force k-nearest-neighbour search on one H100.",
        ),
    ],
    display_name = "GPU Brute-Force k-NN Kernel Optimization",
    category = "MLSys",
)
