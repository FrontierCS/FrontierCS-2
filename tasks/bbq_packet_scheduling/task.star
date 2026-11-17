"""Implement BBQ's missing priority-queue level so the generated hardware queue reaches a higher packet rate, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 60000,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 8 * MiB,
    visible_metrics = [
        "dev_min_throughput_ops_per_second",
        "dev_max_latency_picoseconds",
        "dev_max_clock_period_picoseconds",
        "dev_min_throughput_ops_per_cycle",
        "dev_max_latency_cycles",
        "dev_valid_fraction",
    ],
    source = [
        paper(
            "https://www.usenix.org/system/files/nsdi24-atre.pdf",
            title = "BBQ: A Fast and Scalable Integer Priority Queue for Hardware Packet Scheduling",
            venue = "NSDI 2024",
            summary = (
                "Improve the BBQ hardware integer priority queue generator's bitmap traversal " +
                "for packet scheduling."
            ),
        ),
    ],
    display_name = "BBQ bitmap traversal",
    category = "Network",
)
