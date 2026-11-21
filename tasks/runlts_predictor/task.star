"""Supply the register-value component of a TAGE-SC-L branch predictor to cut mispredictions and wrong-path cycles, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper", "ubuntu_agent", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = ubuntu_agent("build-essential ca-certificates git python3 zlib1g-dev", executable = "*.sh"),
    agent = {"cpu_millis": 1000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 8000, "memory_bytes": 32 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 18000,
    judge_timeout_seconds = 7200,
    final_timeout_seconds = 12000,
    submission_max_bytes = 256 * KiB,
    visible_metrics = {
        "train_br_mis_pki_mean": (
            "Candidate mean BrMisPKI (four-decimal printed rate) over the train\n" +
            "  traces."
        ),
        "train_cyc_wp_pki_mean": (
            "Candidate mean CycWpPKI (four-decimal printed rate) over the train\n" +
            "  traces."
        ),
        "all_metrics_progress": (
            "Counts and percentages of train traces on which both prediction\n" +
            "  comparisons clear together: passed, failed, total, pass_percent, fail_percent."
        ),
        "category_means": (
            "Per application category (compress, fp, infra, int, media, web): number of\n" +
            "  train traces and the candidate's mean br_mis_pki and cyc_wp_pki from exact counters."
        ),
        "candidate_failure": (
            "0 for a measured result; otherwise the failing stage of a scored zero: 1\n" +
            "  C++ parse error in the source audit, 2 logical storage over 1,572,864 bits, 3 source audit\n" +
            "  over its limit, 4 build failed or timed out, 5 a simulation failed or timed out, 6 malformed\n" +
            "  simulator output, 7 instruction counts differ from the trace's, 8 another candidate failure."
        ),
    },
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
