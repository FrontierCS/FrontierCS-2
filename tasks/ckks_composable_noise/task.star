"""Predict the numerical precision of CKKS homomorphic circuits accurately and cheaply, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = package_agent(ubuntu_agent_base),
    agent = {"cpu_millis": 2000, "memory_bytes": 2 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 7200,
    judge_timeout_seconds = 600,
    submission_max_bytes = 256 * KiB,
    visible_metrics = {
        "candidate_avg_log2_prec_l2_abs_discrepancy_mean_bits": (
            "Candidate mean absolute AVGLog2Prec.L2\n" +
            "  discrepancy over the six train configurations, in bits."
        ),
        "candidate_std_log2_prec_l2_abs_discrepancy_mean_bits": (
            "Candidate mean absolute STDLog2Prec.L2\n" +
            "  discrepancy over the six train configurations, in bits."
        ),
        "candidate_estimator_seconds_mean": (
            "Candidate mean estimator time per invocation over the six\n" +
            "  train configurations, in seconds."
        ),
    },
    context_images = {"go-toolchain": ubuntu_verifier_image("toolchain/Dockerfile")},
    source = [
        paper(
            "https://eprint.iacr.org/2024/853.pdf",
            title = "Accurate and Composable Noise Estimates for CKKS with Application to Exact HE Computation",
            venue = "IACR CiC 2025",
            authors = [
                "Jean-Philippe Bossuat",
                "Anamaria Costache",
                "Christian Mouchet",
                "Lea Nürnberger",
                "Juan Ramón Troncoso-Pastoriza",
            ],
            year = 2025,
            summary = (
                "Build a fast estimator that predicts the precision of CKKS " +
                "homomorphic-encryption computations."
            ),
        ),
        artifacts(
            "https://github.com/tuneinsight/ckks-noise-estimator",
            commit = "53dd2367a25a0ae45298dd4086ef50ae03aa778a",
        ),
    ],
    display_name = "CKKS Precision Propagation at 2,048 Slots",
    category = "Crypto",
)
