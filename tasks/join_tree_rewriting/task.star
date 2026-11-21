"""Rewrite binary join trees into plans that lower to fast two-phase Yannakakis execution, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "paper", "ubuntu_agent", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    package_context = True,
    agent_recipe = ubuntu_agent("python3-pip python-is-python3"),
    agent = {"cpu_millis": 2000, "memory_bytes": 8 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 8 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 24000,
    judge_timeout_seconds = 900,
    submission_max_bytes = 64 * KiB,
    visible_metrics = {
        "all_metrics_progress": (
            "Train queries clearing both exact_result and median_latency: passed,\n" +
            "  failed and total counts with pass_percent and fail_percent."
        ),
        "candidate_median_latency_us": (
            "Median over the train queries of the candidate's per-query\n" +
            "  median latency, in microseconds; null unless every query was measured."
        ),
    },
    source = [
        paper(
            "https://www.vldb.org/pvldb/vol18/p2413-vansummeren.pdf",
            title = "Instance-Optimal Acyclic Join Processing Without Regret: Engineering the Yannakakis Algorithm in Column Stores",
            venue = "PVLDB 2025",
            authors = ["Liese Bekkers", "Frank Neven", "Stijn Vansummeren", "Remy Wang"],
            year = 2025,
            summary = (
                "Rewrite optimizer join trees into faster nested-semijoin plans for " +
                "instance-optimal acyclic join processing in a column store."
            ),
        ),
        artifacts(
            "https://github.com/UHasselt-DSI-Data-Systems-Lab/code-reproducability-yannakakis-vldb2025",
            commit = "2d75c9670cd693378dc5f4fc72edc359dd4bb6d1",
        ),
    ],
    display_name = "Join-Tree Rewriting",
    category = "Database",
)
