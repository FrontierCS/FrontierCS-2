"""Replace Spectrum's checkpoint-capable EVM stack with a faster one that restores state exactly, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = package_agent(ubuntu_agent_base),
    agent = {"cpu_millis": 4000, "memory_bytes": 8 * GiB},
    judge = {"cpu_millis": 4000, "memory_bytes": 8 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 360000,
    judge_timeout_seconds = 3600,
    submission_max_bytes = 64 * KiB,
    visible_metrics = {
        "committed_tps": (
            "Geometric mean over the train Spectrum workloads of the candidate's median\n" +
            "  committed transactions per second."
        ),
        "checkpoint_create_ns": (
            "Median over train seeds of the candidate's first checkpoint-micro\n" +
            "  creation time, in nanoseconds per checkpoint."
        ),
        "checkpoint_restore_ns": (
            "Median over train seeds of the candidate's first checkpoint-micro\n" +
            "  restoration time, in nanoseconds."
        ),
        "retained_bytes_per_checkpoint": (
            "Maximum over train seeds of the candidate's first\n" +
            "  checkpoint-micro retained bytes per live checkpoint."
        ),
    },
    source = [
        paper(
            "https://www.vldb.org/pvldb/vol17/p2541-zhang.pdf",
            title = "Spectrum: Speedy and Strictly-Deterministic Smart Contract Transactions for Blockchain Ledgers",
            venue = "PVLDB 2024",
            authors = [
                "Zhihao Chen",
                "Tianji Yang",
                "Yixiao Zheng",
                "Zhao Zhang",
                "Cheqing Jin",
                "Aoying Zhou",
            ],
            year = 2024,
            summary = (
                "Implement fast EVM execution-state checkpoint and restore in the Spectrum " +
                "deterministic concurrent transaction engine."
            ),
        ),
        artifacts(
            "https://github.com/jacklightChen/spectrum",
        ),
    ],
    display_name = "EVM State Checkpoints",
    category = "Database",
)
