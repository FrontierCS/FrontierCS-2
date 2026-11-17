"""Generate FARGO's global probing sequence and stopping rule for approximate maximum-inner-product search, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 4000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 4000, "memory_bytes": 4 * GiB},
    train_calls = 50,
    train_cumulative_seconds = 18000,
    judge_timeout_seconds = 900,
    submission_max_bytes = 64 * KiB,
    visible_metrics = [
        "mean_recall",
        "mean_ratio",
        "mean_kratio",
        "mean_time_ms",
        "mean_candidates",
    ],
    source = [
        paper(
            "https://www.vldb.org/pvldb/vol16/p1100-zheng.pdf",
            title = "FARGO: Fast Maximum Inner Product Search via Global Multi-Probing",
            venue = "PVLDB 2023",
            authors = [
                "Xi Zhao",
                "Bolong Zheng",
                "Xiaomeng Yi",
                "Xiaofan Luan",
                "Charles Xie",
                "Xiaofang Zhou",
                "Christian S. Jensen",
            ],
            year = 2023,
            doi = "10.14778/3579075.3579084",
            summary = (
                "Rebuild FARGO's global multi-probing query core to beat the reference latency " +
                "at equal recall."
            ),
        ),
    ],
    display_name = "FARGO paper-aligned query core",
    category = "Database",
)
