"""Send fewer bytes per online query than QuarterPIR while beating it on every cost it measures, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 4000, "memory_bytes": 8 * GiB},
    judge = {"cpu_millis": 4000, "memory_bytes": 8 * GiB},
    train_calls = 8,
    train_cumulative_seconds = 48000,
    judge_timeout_seconds = 6000,
    submission_max_bytes = 1 * MiB,
    visible_metrics = [
        "candidate_public_server_compute_time_ns_median",
        "candidate_public_online_time_ms_median",
        "candidate_public_preprocessing_seconds_large",
        "candidate_public_online_communication_bytes_large",
        "candidate_public_client_storage_bytes_large",
        "candidate_public_validity_code",
    ],
    datasets = {
        "train": "/opt/frontier/datasets/train.tar",
        "eval": "/opt/frontier/datasets/eval.tar",
    },
    mode_stages = False,
    source = [
        paper(
            "https://eprint.iacr.org/2023/1574",
            title = "Efficient Pre-processing PIR Without Public-Key Cryptography",
            venue = "EUROCRYPT 2024",
            summary = (
                "Reduce per-query communication in QuarterPIR, a preprocessing " +
                "private-information-retrieval scheme with client-side hints."
            ),
        ),
    ],
    display_name = "Private Set List Decoding",
    category = "Crypto",
)
