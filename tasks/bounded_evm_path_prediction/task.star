"""Predict the directions of state-dependent EVM branches in SeerEVM's replay of Ethereum blocks, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier")

fcs_task(
    agent_recipe = package_agent(ubuntu_agent_base),
    package_context = True,
    verifier_base = ubuntu_verifier("ca-certificates git golang-go python3"),
    agent = {"cpu_millis": 8000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 8000, "memory_bytes": 8 * GiB},
    train_calls = 50,
    train_cumulative_seconds = 90000,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 128 * KiB,
    visible_metrics = {
        "raw_quality": (
            "Candidate quality Q = 100 * (0.8 * mean branch accuracy + 0.2 * mean\n" +
            "  transaction-path ratio) on the scored epoch."
        ),
        "branch_accuracy": (
            "Candidate mean branch-direction accuracy over the scored records, in [0,\n" +
            "  1]."
        ),
        "transaction_path_accuracy": (
            "Candidate mean transaction-path satisfaction ratio over the\n" +
            "  scored records; normal-category ratios may exceed 1."
        ),
        "observations": "Number of scored report records.",
    },
    source = [
        paper(
            "https://www.vldb.org/pvldb/vol18/p822-xiao.pdf",
            title = "Seer: Accelerating Blockchain Transaction Execution by Fine-Grained Branch Prediction",
            venue = "PVLDB 2024",
            authors = ["Shijie Zhang", "Ru Cheng", "Xinpeng Liu", "Jiang Xiao", "Hai Jin", "Bo Li"],
            year = 2025,
            summary = (
                "Speed up SeerEVM's fine-grained branch prediction for blockchain transaction " +
                "execution."
            ),
        ),
        artifacts(
            "https://github.com/CGCL-codes/SeerEVM",
        ),
    ],
    display_name = "Bounded EVM Path Prediction",
    category = "Database",
)
