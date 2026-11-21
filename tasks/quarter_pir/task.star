"""Send fewer bytes per online query than QuarterPIR while beating it on every cost it measures, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = package_agent(ubuntu_agent_base),
    agent = {"cpu_millis": 4000, "memory_bytes": 8 * GiB},
    judge = {"cpu_millis": 4000, "memory_bytes": 8 * GiB},
    train_calls = 8,
    train_cumulative_seconds = 48000,
    judge_timeout_seconds = 6000,
    submission_max_bytes = 1 * MiB,
    visible_metrics = {
        "candidate_public_server_compute_time_ns_median": (
            "Median candidate server compute time over\n" +
            "  the 1,000 measured queries of the train experiment, in nanoseconds; -1 when unavailable."
        ),
        "candidate_public_online_time_ms_median": (
            "Candidate end-to-end online time of the train\n" +
            "  experiment, in milliseconds per query; -1 when unavailable."
        ),
        "candidate_public_preprocessing_seconds_large": (
            "Candidate client preprocessing time of the\n" +
            "  train experiment, in seconds; -1 when unavailable."
        ),
        "candidate_public_online_communication_bytes_large": (
            "Candidate mean request-plus-response\n" +
            "  bytes per query in the train experiment; -1 when unavailable."
        ),
        "candidate_public_client_storage_bytes_large": (
            "Candidate client storage in the train\n" +
            "  experiment (the larger of the peak live heap readings), in bytes; -1 when unavailable."
        ),
        "candidate_public_validity_code": (
            "First failed check: 0 valid; 3 build; 4 client crashed,\n" +
            "  broke the harness protocol or timed out; 5 hint export differs; 6 request is not the reduction\n" +
            "  of the reference request; 7 wrong retrieved value or failed server call; 8 client process\n" +
            "  memory limit; 9 server replay check; 10 evaluation time limit; 11 failed an additional query;\n" +
            "  12 primary-hint audit failed; -1 no verdict (codes 1 and 2 are now shared rejections)."
        ),
    },
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
