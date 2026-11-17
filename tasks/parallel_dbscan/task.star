"""Supply the core-connectivity stage of an exact parallel DBSCAN and make it faster than the reference, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 8000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 8000, "memory_bytes": 4 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 24000,
    judge_timeout_seconds = 600,
    submission_max_bytes = 280 * KiB,
    visible_metrics = [
        "candidate_elapsed_seconds_median",
        "candidate_elapsed_seconds_p90",
    ],
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
