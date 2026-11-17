"""Build static binary fuse filters that beat the upstream filters on memory, construction and lookup time, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 12000,
    judge_timeout_seconds = 180,
    submission_max_bytes = 1 * MiB,
    visible_metrics = [
        "mean_resident_bytes",
        "mean_build_ns",
        "mean_lookup_ns",
        "mean_effective_fpr",
        "space_overhead_defined_workloads",
    ],
    source = [
        paper(
            "https://arxiv.org/abs/2201.01174",
            title = "Binary Fuse Filters: Fast and Smaller Than Xor Filters",
            venue = "ACM JEA 2022",
            doi = "10.1145/3510449",
            summary = (
                "Implement and optimize construction and lookup of static binary fuse " +
                "approximate-membership filters."
            ),
        ),
        artifacts(
            "https://github.com/FastFilter/xor_singleheader",
            commit = "3e10fb185a0c51e5866518d13b403e07db63d8d8",
            license = "Apache-2.0",
        ),
    ],
    display_name = "Binary Fuse Filter Construction and Lookup",
    category = "Theory",
)
