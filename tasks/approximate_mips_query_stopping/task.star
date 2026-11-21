"""Decide when FARGO stops probing buckets in approximate maximum-inner-product search, ported from FrontierCS."""

load("//lib/fcs.star", "AVX512", "XEON_8375C", "artifacts", "fcs_task", "paper", "ubuntu_agent", "ubuntu_verifier")

fcs_task(
    agent_recipe = ubuntu_agent("build-essential ca-certificates git python3 tar", executable = "*.sh"),
    verifier_base = ubuntu_verifier("build-essential ca-certificates git python3 tar"),
    agent = {"cpu_millis": 4000, "memory_bytes": 4 * GiB, "cpu_features": AVX512, "cpu_model": XEON_8375C},
    judge = {"cpu_millis": 4000, "memory_bytes": 4 * GiB, "cpu_features": AVX512, "cpu_model": XEON_8375C},
    train_calls = 100,
    train_cumulative_seconds = 90000,
    judge_timeout_seconds = 900,
    submission_max_bytes = 64 * KiB,
    visible_metrics = {
        "mean_recall": "Arithmetic mean over the train workloads of the candidate's recall at 50.",
        "mean_kratio": (
            "Arithmetic mean over the train workloads of the candidate's last-result\n" +
            "  inner-product ratio."
        ),
        "mean_time_ms": (
            "Arithmetic mean over the train workloads of the candidate's mean query time,\n" +
            "  in milliseconds."
        ),
        "mean_candidates": (
            "Arithmetic mean over the train workloads of the candidate's distinct\n" +
            "  verified candidates per query."
        ),
    },
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
