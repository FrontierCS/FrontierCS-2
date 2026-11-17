"""Supply the register-value component of a TAGE-SC-L branch predictor to cut mispredictions and wrong-path cycles, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 1000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 8000, "memory_bytes": 32 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 18000,
    judge_timeout_seconds = 7200,
    final_timeout_seconds = 12000,
    submission_max_bytes = 256 * KiB,
    visible_metrics = [
        "train_br_mis_pki_mean",
        "train_cyc_wp_pki_mean",
        "all_metrics_progress",
        "category_means",
        "candidate_failure",
    ],
    source = [
        paper(
            "https://ericrotenberg.wordpress.ncsu.edu/files/2025/06/cbp2025-final44-Koizumi.pdf",
            title = "RUNLTS: Register-value-aware Predictor Utilizing Nested Large Tables",
            venue = "CBP 2025",
            summary = (
                "Design a register-aware conditional branch predictor evaluated in a cycle-aware " +
                "trace simulator."
            ),
        ),
    ],
    display_name = "Register-aware Branch Prediction",
    category = "Arch",
)
