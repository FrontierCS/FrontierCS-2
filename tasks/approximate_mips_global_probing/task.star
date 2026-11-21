"""Generate FARGO's global probing sequence and stopping rule for approximate maximum-inner-product search, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper", "ubuntu_agent", "ubuntu_verifier")

fcs_task(
    package_context = True,
    agent_recipe = ubuntu_agent("build-essential ca-certificates git python3", executable = "*.sh"),
    verifier_base = ubuntu_verifier("build-essential ca-certificates git python3"),
    agent = {"cpu_millis": 4000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 4000, "memory_bytes": 4 * GiB},
    train_calls = 50,
    train_cumulative_seconds = 18000,
    judge_timeout_seconds = 900,
    submission_max_bytes = 64 * KiB,
    visible_metrics = {
        "mean_recall": "Candidate recall averaged over the train operating points.",
        "mean_ratio": (
            "Candidate overall (mean rankwise inner-product) ratio averaged over the train\n" +
            "  operating points."
        ),
        "mean_kratio": (
            "Candidate kth-result inner-product ratio averaged over the train operating\n" +
            "  points (diagnostic)."
        ),
        "mean_time_ms": (
            "Candidate query time averaged over the train operating points, in\n" +
            "  milliseconds."
        ),
        "mean_candidates": (
            "Candidate verified candidates per query averaged over the train operating\n" +
            "  points."
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
