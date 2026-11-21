"""Select the faster of ScaleGPM's two approximate engines for each graph-pattern-mining configuration, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "paper", "ubuntu_agent", "ubuntu_verifier")

fcs_task(
    agent_recipe = ubuntu_agent("build-essential ca-certificates git python3", executable = "*.sh"),
    verifier_base = ubuntu_verifier("ca-certificates git python3"),
    agent = {"cpu_millis": 8000, "memory_bytes": 16 * GiB},
    judge = {"cpu_millis": 8000, "memory_bytes": 16 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 90000,
    judge_timeout_seconds = 900,
    submission_max_bytes = 64 * KiB,
    visible_metrics = {
        "geomean_selected_runtime_seconds": (
            "Geometric mean of the stored runtimes of the\n" +
            "  candidate-selected engines over the 112 train cases, in seconds."
        ),
        "max_selected_relative_error": (
            "Largest stored relative error among the candidate-selected\n" +
            "  engine estimates."
        ),
        "cases": "Number of train cases evaluated.",
    },
    source = [
        paper(
            "https://www.vldb.org/pvldb/vol18/p93-chen.pdf",
            title = "Accurate and Fast Approximate Graph Pattern Mining at Scale",
            venue = "PVLDB 2024",
            authors = ["Anna A Arpaci-Dusseau", "Zixiang Zhou", "Xuhao Chen"],
            year = 2025,
            summary = (
                "Choose and tune ScaleGPM's approximate graph-pattern mining engine per query to " +
                "cut runtime within the error bound."
            ),
        ),
        artifacts(
            "https://anonymous.4open.science/r/scale-gpm-B987/",
        ),
    ],
    display_name = "Graph Mining Method Selection",
    category = "Database",
)
