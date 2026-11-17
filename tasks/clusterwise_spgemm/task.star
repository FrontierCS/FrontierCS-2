"""Reconstruct and speed up clusterwise sparse matrix multiplication on 60 CPUs with exact results, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 2000, "memory_bytes": 16 * GiB},
    judge = {"cpu_millis": 60000, "memory_bytes": 64 * GiB},
    train_calls = 3,
    train_cumulative_seconds = 21600,
    judge_timeout_seconds = 7200,
    submission_max_bytes = 260 * KiB,
    visible_metrics = [
        "candidate_median_execution_time_ms",
        "candidate_median_throughput_mflops",
        "candidate_max_output_bytes",
    ],
    datasets = {
        "train": "/opt/frontier/datasets/train.tar",
        "eval": "/opt/frontier/datasets/eval.tar",
    },
    source = [
        paper(
            "https://arxiv.org/pdf/2507.21253v1",
            title = "Improving SpGEMM Performance Through Matrix-Reordering and Cluster-wise Computation",
            venue = "SC 2025",
            authors = ["Abdullah Al Raqibul Islam", "Helen Xu", "Dong Dai", "Aydin Buluc"],
            year = 2025,
            summary = (
                "Reconstruct and speed up fixed-length clustered sparse matrix-matrix " +
                "multiplication (SpGEMM)."
            ),
        ),
    ],
    display_name = "Fixed-Length Clustered Sparse Multiplication",
    category = "Arch",
)
