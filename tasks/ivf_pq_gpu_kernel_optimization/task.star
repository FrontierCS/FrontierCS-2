"""Make IVF-PQ approximate nearest-neighbour search on one H100 faster than the naive search, ported from FrontierCS."""

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
    judge_timeout_seconds = 14400,
    submission_max_bytes = 2 * MiB,
    visible_metrics = {
        "geomean_speedup": (
            "Geometric mean over the train workloads of the median paired speedup over\n" +
            "  the naive search (unitless)."
        ),
        "search_latency_ms": (
            "Candidate median search latency per train workload, in milliseconds;\n" +
            "  sorted, without workload labels."
        ),
        "iso_recall": (
            "Candidate worst timed iso-result recall per train workload; sorted, without\n" +
            "  workload labels."
        ),
        "nn_recall": (
            "Candidate worst timed nearest-neighbor recall@k per train workload; sorted,\n" +
            "  without workload labels."
        ),
    },
    source = [
        paper(
            "https://flashml-org.github.io/",
            title = "FlashLib: Bringing Flash Magic to Classical Machine Learning Operators",
            venue = "Blog",
            summary = (
                "Optimize IVF-PQ approximate nearest-neighbour search on one H100 with " +
                "Triton/Python kernels."
            ),
        ),
    ],
    display_name = "GPU IVF-PQ Search Kernel Optimization",
    category = "MLSys",
)
