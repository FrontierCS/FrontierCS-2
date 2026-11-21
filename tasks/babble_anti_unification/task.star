"""Supply BABBLE's library-learning candidates and state pairing so that program corpora shrink further, faster, ported from FrontierCS."""

load("//lib/fcs.star", "artifacts", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = package_agent(ubuntu_agent_base),
    agent = {"cpu_millis": 2000, "memory_bytes": 8 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 2 * GiB},
    train_calls = 100,
    train_cumulative_seconds = 1800,
    judge_timeout_seconds = 600,
    submission_max_bytes = 256 * KiB,
    visible_metrics = {
        "compression_ratio": (
            "The candidate's compression ratio (initial / final AST cost) for each\n" +
            "  development corpus, in suite order."
        ),
        "runtime_seconds": (
            "The candidate's median end-to-end benchmark runtime for each development\n" +
            "  corpus, in seconds, in suite order."
        ),
    },
    source = [
        paper(
            "https://arxiv.org/abs/2212.04596",
            title = "babble: Learning Better Abstractions with E-Graphs and Anti-unification",
            venue = "POPL 2023",
            summary = (
                "Improve BABBLE's library learning so refactored program corpora become smaller."
            ),
        ),
        artifacts(
            "https://github.com/dcao/babble",
            commit = "09ff03fd485a3e24178c1d706a5944a230123ba1",
        ),
    ],
    display_name = "Reusable Library Candidates",
    category = "Programming Language",
)
