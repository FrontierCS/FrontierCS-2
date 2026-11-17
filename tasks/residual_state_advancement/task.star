"""Advance JoinBoost's residual state each boosting round inside DuckDB and make the fit faster, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 2000, "memory_bytes": 16 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 16 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 720000,
    judge_timeout_seconds = 7200,
    submission_max_bytes = 128 * KiB,
    visible_metrics = [],
    source = [
        paper(
            "https://www.vldb.org/pvldb/vol16/p3071-huang.pdf",
            title = "JoinBoost: Grow Trees over Normalized Data Using Only SQL",
            venue = "PVLDB 2023",
            authors = ["Zezhou Huang", "Rathijit Sen", "Jiaxiang Liu", "Eugene Wu"],
            year = 2023,
            summary = (
                "Speed up residual updates in JoinBoost's SQL-only gradient-boosted tree " +
                "training over normalized data."
            ),
        ),
        artifacts(
            "https://github.com/JoinBoost/JoinBoost",
        ),
    ],
    display_name = "Residual State Advancement",
    category = "Database",
)
