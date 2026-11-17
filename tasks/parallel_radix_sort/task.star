"""Supply Regions Sort's exchange planner so the in-place radix sorter beats std::sort, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 4000, "memory_bytes": 2 * GiB},
    judge = {"cpu_millis": 4000, "memory_bytes": 2 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 18000,
    judge_timeout_seconds = 900,
    submission_max_bytes = 270 * KiB,
    visible_metrics = [
        "dev_candidate_median_sort_time_ns",
        "dev_candidate_max_auxiliary_bytes",
        "dev_candidate_median_region_graph_ns",
    ],
    source = [
        paper(
            "https://people.csail.mit.edu/jshun/RegionsSort.pdf",
            title = "Theoretically-Efficient and Practical Parallel In-Place Radix Sorting",
            venue = "SPAA 2019",
            authors = ["Omar Obeya", "Endrias Kahssay", "Edward Fan", "Julian Shun"],
            year = 2019,
            doi = "10.1145/3323165.3323198",
            summary = (
                "Speed up Regions Sort, a parallel in-place most-significant-digit radix sorter."
            ),
        ),
        artifacts(
            "https://github.com/omarobeya/parallel-inplace-radixsort",
            commit = "4f5884733748b51550e9ef7d69791fb9decb922f",
        ),
    ],
    display_name = "In-place Integer Redistribution",
    category = "HPC",
)
