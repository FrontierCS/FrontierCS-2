"""Write the instruction policy that lets a fixed language model repair regular expressions that backtrack catastrophically, ported from FrontierCS."""

load("//lib/fcs.star", "A100_80", "GPU_RUNNER", "artifacts", "fcs_task", "paper", "ubuntu_agent")

fcs_task(
    agent_recipe = ubuntu_agent("ca-certificates git python3 tar", executable = "*.sh"),
    agent = {"cpu_millis": 4000, "memory_bytes": 12 * GiB},
    judge = GPU_RUNNER | {"gpu": A100_80},
    train_calls = 100,
    train_cumulative_seconds = 5400,
    judge_timeout_seconds = 3600,
    submission_max_bytes = 256 * KiB,
    visible_metrics = {
        "train_vulnerability_elimination_rate": (
            "Share of train regexes whose repair has a completed\n" +
            "  detector result with no successful attack, regardless of semantics."
        ),
        "train_repair_rate": "Share of train regexes with a credited repair.",
        "train_well_formed_rate": "Share of train regexes with a well-formed repair.",
        "train_mean_rli": (
            "Mean relative length increase of the repairs over all train regexes; lower\n" +
            "  is better."
        ),
        "train_mean_nls": (
            "Mean normalized Levenshtein similarity of the repairs over all train\n" +
            "  regexes."
        ),
        "train_mean_jaccard": "Mean Xeger Jaccard estimate of the repairs over all train regexes.",
        "train_mean_precision": "Mean Xeger precision estimate of the repairs over all train regexes.",
        "train_mean_recall": "Mean Xeger recall estimate of the repairs over all train regexes.",
    },
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
