"""Recover secret vectors for as many public structured-LWE instances as possible, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = package_agent(ubuntu_agent_base),
    agent = {"cpu_millis": 8000, "memory_bytes": 32 * GiB},
    judge = {"cpu_millis": 8000, "memory_bytes": 2 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 12000,
    judge_timeout_seconds = 120,
    submission_max_bytes = 4 * MiB,
    visible_metrics = {
        "instance_count": "Number of instances in the public catalog.",
        "solved_count": (
            "Distinct catalog instances whose unique admissible ledger record passes every\n" +
            "  validity predicate."
        ),
        "submitted_count": "Ledger records read, including records rejected as malformed.",
        "invalid_count": (
            "Rejected record occurrences: malformed, duplicated, unknown or failing a\n" +
            "  predicate."
        ),
        "duplicate_count": (
            "Distinct syntactically valid instance IDs that occur more than once in the\n" +
            "  ledger."
        ),
        "conflict_count": (
            "Distinct instance IDs that carry more than one distinct syntactically valid\n" +
            "  vector."
        ),
        "unknown_count": "Distinct syntactically valid instance IDs that are not in the public catalog.",
        "rejection_code_counts": "Number of rejected record occurrences per stable rejection code.",
    },
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
