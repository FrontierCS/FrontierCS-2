"""Make Euclidean DBSCAN on one H100 faster than the naive starter with identical clusterings, ported from FrontierCS."""

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
    judge_timeout_seconds = 2400,
    submission_max_bytes = 2 * MiB,
    visible_metrics = [
        "geomean_speedup",
        "dbscan_latency_ms",
        "adjusted_rand_index",
        "noise_mismatch_fraction",
    ],
    source = [
        paper(
            "https://flashml-org.github.io/",
            title = "FlashLib: Bringing Flash Magic to Classical Machine Learning Operators",
            venue = "Blog",
            summary = "Optimize an exact GPU DBSCAN clustering kernel on one H100.",
        ),
    ],
    display_name = "GPU DBSCAN Kernel Optimization",
    category = "MLSys",
)
