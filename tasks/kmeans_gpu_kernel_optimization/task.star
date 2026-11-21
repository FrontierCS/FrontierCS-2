"""Make one exact Lloyd k-means step on one H100 faster than the naive starter, ported from FrontierCS."""

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
    train_cumulative_seconds = 16200,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 2 * MiB,
    visible_metrics = {
        "geomean_speedup": (
            "Geometric mean over the train workloads of the naive/candidate median\n" +
            "  loop-latency ratio (unitless)."
        ),
        "lloyd_loop_latency_ms": (
            "Candidate median two-step loop latency per train workload, in\n" +
            "  milliseconds; sorted, without workload labels."
        ),
        "labeled_inertia": (
            "Candidate labeled inertia of the final labels and centroids per train\n" +
            "  workload; sorted, without workload labels."
        ),
        "label_mismatch_fraction": (
            "Candidate fraction of final labels that are not an exact nearest\n" +
            "  centroid per train workload; sorted, without workload labels."
        ),
    },
    source = [
        paper(
            "https://arxiv.org/abs/2603.09229",
            title = "Flash-KMeans: Fast and Memory-Efficient Exact K-Means",
            venue = "arXiv 2026",
            summary = "Optimize one exact Lloyd k-means step on one H100.",
        ),
    ],
    display_name = "GPU K-Means Kernel Optimization",
    category = "MLSys",
)
