"""Decide when FARGO stops probing buckets in approximate maximum-inner-product search, ported from FrontierCS."""

load("//lib/fcs.star", "AVX512", "XEON_8375C", "artifacts", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 4000, "memory_bytes": 4 * GiB, "cpu_features": AVX512, "cpu_model": XEON_8375C},
    judge = {"cpu_millis": 4000, "memory_bytes": 4 * GiB, "cpu_features": AVX512, "cpu_model": XEON_8375C},
    train_calls = 100,
    train_cumulative_seconds = 90000,
    judge_timeout_seconds = 900,
    submission_max_bytes = 64 * KiB,
    visible_metrics = [
        "mean_recall",
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
            summary = (
                "Design a query-time early-stopping rule for FARGO approximate " +
                "maximum-inner-product search."
            ),
        ),
        artifacts(
            "https://github.com/Jacyhust/FARGO_VLDB23",
        ),
    ],
    display_name = "Approximate-MIPS Query Stopping",
    category = "Database",
)
