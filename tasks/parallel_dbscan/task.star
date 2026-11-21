"""Supply the core-connectivity stage of an exact parallel DBSCAN and make it faster than the reference, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = package_agent(ubuntu_agent_base),
    agent = {"cpu_millis": 8000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 8000, "memory_bytes": 4 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 24000,
    judge_timeout_seconds = 600,
    submission_max_bytes = 280 * KiB,
    visible_metrics = {
        "candidate_elapsed_seconds_median": (
            "Median over the train workloads (four small, two\n" +
            "  200,000-point) of the candidate's workload time, in seconds."
        ),
        "candidate_elapsed_seconds_p90": (
            "Nearest-rank 90th percentile of the same six candidate\n" +
            "  workload times, in seconds (the largest of the six)."
        ),
    },
    source = [
        paper(
            "https://doi.org/10.1145/3318464.3380582",
            title = "Theoretically-Efficient and Practical Parallel DBSCAN",
            venue = "SIGMOD 2020",
            authors = ["Yiqiu Wang", "Yan Gu", "Julian Shun"],
            year = 2020,
            doi = "10.1145/3318464.3380582",
            summary = "Speed up exact parallel DBSCAN cluster connectivity.",
        ),
        artifacts(
            "https://github.com/wangyiqiu/dbscan",
        ),
    ],
    display_name = "Exact DBSCAN Connectivity",
    category = "HPC",
)
