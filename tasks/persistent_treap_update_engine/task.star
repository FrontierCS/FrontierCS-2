"""Make concurrent transactional updates to Contreap's persistent treap faster with every version correct, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 8000, "memory_bytes": 16 * GiB},
    judge = {"cpu_millis": 8000, "memory_bytes": 16 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 180000,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 256 * KiB,
    visible_metrics = [],
    source = [
        paper(
            "https://www.vldb.org/pvldb/vol18/p1481-wang.pdf",
            title = "Efficient Concurrent Updates to Persistent Randomized Binary Search Trees",
            venue = "PVLDB 2025",
            authors = ["Guanhao Hou", "Jinchao Huang", "Fangyuan Zhang", "Sibo Wang"],
            year = 2025,
            summary = (
                "Make concurrent transactional updates to Contreap, a persistent treap index, " +
                "faster."
            ),
        ),
        artifacts(
            "https://github.com/CUHK-DBGroup/Contreap",
        ),
    ],
    display_name = "Persistent Treap Update Engine",
    category = "Database",
)
