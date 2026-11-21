"""Implement BBQ's missing priority-queue level so the generated hardware queue reaches a higher packet rate, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = package_agent(ubuntu_agent_base),
    agent = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 60000,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 8 * MiB,
    visible_metrics = {
        "dev_min_throughput_ops_per_second": (
            "Minimum candidate throughput over the public train\n" +
            "  workloads, in operations per second; 0 if any workload is invalid."
        ),
        "dev_max_latency_picoseconds": (
            "Maximum candidate worst response latency (cycles times clock\n" +
            "  period) over the public train workloads, in picoseconds; -1 if any workload is invalid."
        ),
        "dev_max_clock_period_picoseconds": (
            "Maximum candidate modelled clock period over the public\n" +
            "  train configurations, in picoseconds; -1 if any workload is invalid."
        ),
        "dev_min_throughput_ops_per_cycle": (
            "Minimum candidate completed operations per simulated cycle\n" +
            "  over the public train workloads; 0 if any workload is invalid."
        ),
        "dev_max_latency_cycles": (
            "Maximum candidate worst response latency over the public train\n" +
            "  workloads, in simulated cycles; -1 if any workload is invalid."
        ),
        "dev_valid_fraction": "Fraction of the public train workloads that are fully correct.",
    },
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
