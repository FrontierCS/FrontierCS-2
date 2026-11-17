"""Supply SCOPE's batch path for exact local subgraph counting and make the six-case batch faster, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 1000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 1000, "memory_bytes": 4 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 180000,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 512 * KiB,
    visible_metrics = [
        "median_seconds",
    ],
    source = [
        paper(
            "https://www.vldb.org/pvldb/vol17/p1967-li.pdf",
            title = "Fast Local Subgraph Counting",
            venue = "PVLDB 2024",
            authors = ["Qiyan Li", "Jeffrey Xu Yu"],
            year = 2024,
            summary = (
                "Implement the batched (shared-computation) path of the SCOPE exact local " +
                "subgraph counter and minimize single-core wall time."
            ),
        ),
        artifacts(
            "https://github.com/magic62442/subgraph-counting",
        ),
    ],
    display_name = "Batch Local Counting",
    category = "Database",
)
