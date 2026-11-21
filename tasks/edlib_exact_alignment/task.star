"""Implement Edlib's Hirschberg path reconstruction so exact alignments are found as fast as possible, ported from FrontierCS."""

load("//lib/fcs.star", "fcs_task", "paper", "ubuntu_agent", "ubuntu_verifier_image")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = ubuntu_agent("build-essential cmake git python3", executable = "*.sh"),
    agent = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    judge = {"cpu_millis": 2000, "memory_bytes": 4 * GiB},
    train_calls = 20,
    train_cumulative_seconds = 12000,
    judge_timeout_seconds = 600,
    submission_max_bytes = 256 * KiB,
    source = [
        paper(
            "https://doi.org/10.1093/bioinformatics/btw753",
            title = "Edlib: a C/C++ library for fast, exact sequence alignment using edit distance",
            venue = "Bioinformatics 2017",
            summary = "Speed up exact edit-distance sequence alignment in the Edlib C++ library.",
        ),
    ],
    display_name = "Exact Sequence Alignment",
    category = "Bioinformatics",
)
