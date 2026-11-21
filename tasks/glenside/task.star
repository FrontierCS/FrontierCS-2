"""Implement Glenside's access-pattern analysis so that equality saturation maps convolutions onto a systolic array, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = package_agent(ubuntu_agent_base),
    agent = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 12000,
    judge_timeout_seconds = 300,
    submission_max_bytes = 1 * MiB,
    visible_metrics = {
        "development_im2col_discovered_count": (
            "Number of development workloads with a valid im2col\n" +
            "  discovery (0 to 14)."
        ),
        "development_rewrite_time_seconds_median": (
            "Median over the development workloads of the\n" +
            "  candidate's three-run median rewrite time, in seconds."
        ),
        "development_egraph_node_growth_median": (
            "Median over the development workloads of the\n" +
            "  candidate's maximum e-graph node growth."
        ),
    },
    source = [
        paper(
            "https://arxiv.org/abs/2105.09377",
            title = "Pure Tensor Program Rewriting via Access Patterns (Representation Pearl)",
            venue = "MAPS 2021",
            summary = (
                "Find better access-pattern rewrites that map tensor convolutions onto matrix " +
                "hardware in Glenside."
            ),
        ),
    ],
    display_name = "Access-Pattern Rewriting",
    category = "Programming Language",
)
