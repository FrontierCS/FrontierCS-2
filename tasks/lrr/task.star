"""Write the instruction policy that lets a fixed language model repair regular expressions that backtrack catastrophically, ported from FrontierCS."""

load("//lib/fcs.star", "A100_80", "GPU_RUNNER", "artifacts", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 4000, "memory_bytes": 12 * GiB},
    judge = GPU_RUNNER | {"gpu": A100_80},
    train_calls = 100,
    train_cumulative_seconds = 5400,
    judge_timeout_seconds = 3600,
    submission_max_bytes = 256 * KiB,
    visible_metrics = [
        "train_vulnerability_elimination_rate",
        "train_repair_rate",
        "train_well_formed_rate",
        "train_mean_rli",
        "train_mean_nls",
        "train_mean_jaccard",
        "train_mean_precision",
        "train_mean_recall",
    ],
    source = [
        paper(
            "https://aclanthology.org/2026.eacl-long.379.pdf",
            title = "Repairing Regex Vulnerabilities via Localization-Guided Instructions",
            venue = "EACL 2026",
            summary = (
                "Write localization-guided repair prompts that let a fixed local LLM rewrite " +
                "ReDoS-vulnerable regular expressions so they are safe while preserving their " +
                "language."
            ),
        ),
        artifacts(
            "https://github.com/cdltlehf/LRR",
            commit = "99fde1327e389b9564615ecc7a51b89e6188494f",
        ),
    ],
    display_name = "Localized Regex Repair",
    category = "Security",
)
