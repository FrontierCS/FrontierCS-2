"""Make exact brute-force k-nearest-neighbour search on one H100 faster than the naive search, ported from FrontierCS."""

load("//lib/fcs.star", "GPU_RUNNER", "H100", "PYTORCH_KERNEL_AGENT", "fcs_task", "paper")

fcs_task(
    agent_recipe = PYTORCH_KERNEL_AGENT,
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
    visible_metrics = {
        "geomean_speedup": (
            "Geometric mean over the train workloads of the median paired\n" +
            "  naive/candidate time ratio (unitless)."
        ),
        "latency_ms": (
            "Candidate median call latency per train workload, in milliseconds; sorted,\n" +
            "  without workload labels."
        ),
        "worst_ball_recall": (
            "Candidate worst timed ball-recall per train workload; sorted, without\n" +
            "  workload labels."
        ),
        "worst_exact_recall": (
            "Candidate worst timed exact recall per train workload; sorted, without\n" +
            "  workload labels."
        ),
    },
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
