"""Compact a long document's KV cache for Qwen3-4B so that questions are answered about as well, faster or smaller, ported from FrontierCS."""

load("//lib/fcs.star", "GPU_RUNNER", "H100", "artifacts", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 4000, "memory_bytes": 16 * GiB},
    agent_lease = {
        "machine": GPU_RUNNER | {"gpu": H100},
        "instances": 1,
        "instance_seconds": 18000,
        "cumulative_seconds": 18000,
        "commands": 1024,
        "transfer_bytes": 512 * MiB,
    },
    judge = {"cpu_millis": 3000, "memory_bytes": 12 * GiB},
    judge_leases = {
        "shard": {
            "trusted": True,
            "machine": GPU_RUNNER | {"gpu": H100},
            "instances": 8,
            "instance_seconds": {"train": 3600, "eval": 4500},
            "transfer_bytes": 512 * MiB,
        },
    },
    train_calls = 8,
    train_cumulative_seconds = 28800,
    judge_timeout_seconds = 5400,
    submission_max_bytes = 256 * KiB,
    visible_metrics = [
        "dev_qa_accuracy",
        "dev_compaction_seconds",
        "dev_compact_cache_bytes",
    ],
    contexts = {"runtime": "runtime/Dockerfile"},
    source = [
        paper(
            "https://arxiv.org/pdf/2602.16284",
            title = "Fast KV Compaction via Attention Matching",
            venue = "ICML 2026",
            summary = (
                "Compress a long document's KV cache for Qwen3-4B so QuALITY questions are " +
                "answered from the compacted cache as well as by attention matching, but faster " +
                "or smaller."
            ),
        ),
        artifacts(
            "https://github.com/adamzweiger/compaction",
            commit = "2020f6d37940fe013e7e7f1ebf409e5d107c5670",
            license = "MIT",
        ),
    ],
    display_name = "KV Cache Compaction",
    category = "Machine Learning",
)
