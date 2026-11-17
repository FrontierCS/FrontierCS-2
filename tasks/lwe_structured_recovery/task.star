"""Recover secret vectors for as many public structured-LWE instances as possible, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 8000, "memory_bytes": 32 * GiB},
    judge = {"cpu_millis": 8000, "memory_bytes": 2 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 12000,
    judge_timeout_seconds = 120,
    submission_max_bytes = 4 * MiB,
    visible_metrics = [
        "instance_count",
        "solved_count",
        "submitted_count",
        "invalid_count",
        "duplicate_count",
        "conflict_count",
        "unknown_count",
        "rejection_code_counts",
    ],
    mode_stages = False,
    source = [
        paper(
            "https://eprint.iacr.org/2024/1589",
            title = "A Systematic Study of Sparse LWE",
            venue = "CRYPTO 2024",
            summary = (
                "Recover valid secret vectors for as many public structured-LWE instances as " +
                "possible."
            ),
        ),
    ],
    display_name = "Structured-LWE Public Witness Recovery",
    category = "Crypto",
)
