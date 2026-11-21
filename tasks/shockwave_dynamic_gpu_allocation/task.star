"""Write the round-by-round GPU allocation policy of a cluster scheduler for jobs with changing batch sizes, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = package_agent(ubuntu_agent_base),
    agent = {"cpu_millis": 4000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 4000, "memory_bytes": 4 * GiB},
    train_calls = 3,
    train_cumulative_seconds = 5400,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 256 * KiB,
    visible_metrics = {
        "makespan_seconds": "Candidate makespan per train trace, in seconds; anonymous order.",
        "average_jct_seconds": (
            "Candidate average job completion time per train trace, in seconds;\n" +
            "  anonymous order."
        ),
        "worst_finish_time_fairness_rho": (
            "Candidate maximum job-level finish-time-fairness rho per\n" +
            "  train trace; anonymous order."
        ),
        "unfair_job_fraction_percent": (
            "Candidate percentage of jobs with rho above 1.05 per train\n" +
            "  trace; anonymous order."
        ),
    },
    source = [
        paper(
            "https://arxiv.org/pdf/2210.00093",
            title = "Shockwave: Fair and Efficient Cluster Scheduling for Dynamic Adaptation in Machine Learning",
            venue = "NSDI 2023",
            authors = [
                "Pengfei Zheng",
                "Rui Pan",
                "Tarannum Khan",
                "Shivaram Venkataraman",
                "Aditya Akella",
            ],
            year = 2023,
            summary = (
                "Improve a round-based GPU cluster scheduler for ML jobs whose batch size and " +
                "throughput change during training."
            ),
        ),
        artifacts(
            "https://github.com/uw-mad-dash/shockwave",
            commit = "8a44f4a27a45760520386809b695d51b665b3679",
        ),
    ],
    display_name = "Dynamic GPU Allocation",
    category = "Network",
)
