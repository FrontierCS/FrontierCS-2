"""Search for sparse traffic matrices on which a combined traffic-engineering heuristic loses the most throughput, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 4000, "memory_bytes": 8 * GiB},
    judge = {"cpu_millis": 4000, "memory_bytes": 8 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 18000,
    judge_timeout_seconds = 3600,
    submission_max_bytes = 4 * MiB,
    visible_metrics = [
        "mean_gap",
        "instance_count",
    ],
    reference_patch = False,
    source = [
        paper(
            "https://www.usenix.org/system/files/nsdi24-namyar-finding.pdf",
            title = "Finding Adversarial Inputs for Heuristics using Multi-level Optimization",
            venue = "NSDI 2024",
            summary = (
                "Find sparse traffic matrices that expose a large throughput gap between optimal " +
                "traffic engineering and a combined POP + demand-pinning heuristic."
            ),
        ),
    ],
    display_name = "MetaOpt Combined-TE Gap Search",
    category = "Network",
)
