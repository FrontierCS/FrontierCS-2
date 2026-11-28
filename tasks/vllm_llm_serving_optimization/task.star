"""Speed up vLLM 0.11.0's Python scheduling and serving of multi-turn agent conversations on one H100 while keeping their correctness, ported from FrontierCS."""

load("//lib/fcs.star", "GPU_RUNNER", "H100", "fcs_task", "package_agent", "paper", "ubuntu_agent_base", "ubuntu_verifier_image")
load("//tasks/vllm_llm_serving_optimization/images.star", "MODEL_ASSETS", "gpu_runner_image", "parents")

fcs_task(
    verifier_image = ubuntu_verifier_image,
    agent_recipe = package_agent(ubuntu_agent_base),
    agent_runner_image = gpu_runner_image,
    # Both judges run the one stage of the verifier's recipe, as the source's control image is one.
    mode_stages = False,
    agent = {"cpu_millis": 8000, "memory_bytes": 32 * GiB},
    agent_lease = {
        "machine": GPU_RUNNER | {"gpu": H100},
        "instances": 1,
        "instance_seconds": 18000,
        "cumulative_seconds": 18000,
        "commands": 512,
        "transfer_bytes": 256 * MiB,
        "assets": MODEL_ASSETS,
    },
    judge = {"cpu_millis": 8000, "memory_bytes": 32 * GiB},
    judge_leases = {"swe": parents("swe"), "bfcl": parents("bfcl")},
    # The verifier bounds the evidence it keeps by this (`[verifier]`'s `max_evidence_bytes`, which
    # FrontierCS's Driver set).
    judge_env = ["FRONTIER_MAX_EVIDENCE_BYTES=67108864"],
    train_calls = 25,
    train_cumulative_seconds = 7620,
    judge_timeout_seconds = 5400,
    final_timeout_seconds = 7200,
    submission_max_bytes = 2 * MiB,
    visible_metrics = {
        "swe_public_geometric_mean_latency_seconds": (
            "Geometric mean of the candidate's end-to-end conversation latency\n" +
            "  over the public SWE-bench cases, in seconds; omitted unless every case has usable timing."
        ),
        "swe_public_speedup_over_vanilla": (
            "Paired geometric mean of unmodified-vLLM latency (per-case median of three\n" +
            "  fixed captures) over candidate latency on the public SWE-bench cases; omitted unless every\n" +
            "  case has usable timing."
        ),
        "swe_public_p90_latency_seconds": (
            "Candidate p90 end-to-end latency on the public SWE-bench cases, in seconds,\n" +
            "  integer nearest rank; omitted unless every case has usable timing."
        ),
        "swe_public_p95_latency_seconds": (
            "Candidate p95 end-to-end latency on the public SWE-bench cases, in seconds,\n" +
            "  integer nearest rank; omitted unless every case has usable timing."
        ),
        "swe_public_p99_latency_seconds": (
            "Candidate p99 end-to-end latency on the public SWE-bench cases, in seconds,\n" +
            "  integer nearest rank; omitted unless every case has usable timing."
        ),
        "swe_public_correct": (
            "Number of the 5 public SWE-bench cases the candidate resolved; 0 when the candidate\n" +
            "  server failed."
        ),
        "bfcl_public_geometric_mean_latency_seconds": (
            "Geometric mean of the candidate's end-to-end conversation latency\n" +
            "  over the public BFCL cases, in seconds; omitted unless every case has usable timing."
        ),
        "bfcl_public_speedup_over_vanilla": (
            "Paired geometric mean of unmodified-vLLM latency (per-case median of three\n" +
            "  fixed captures) over candidate latency on the public BFCL cases; omitted unless every case\n" +
            "  has usable timing."
        ),
        "bfcl_public_p90_latency_seconds": (
            "Candidate p90 end-to-end latency on the public BFCL cases, in seconds, integer\n" +
            "  nearest rank; omitted unless every case has usable timing."
        ),
        "bfcl_public_p95_latency_seconds": (
            "Candidate p95 end-to-end latency on the public BFCL cases, in seconds, integer\n" +
            "  nearest rank; omitted unless every case has usable timing."
        ),
        "bfcl_public_p99_latency_seconds": (
            "Candidate p99 end-to-end latency on the public BFCL cases, in seconds, integer\n" +
            "  nearest rank; omitted unless every case has usable timing."
        ),
        "bfcl_public_correct": (
            "Number of the 8 public BFCL cases the candidate answered correctly; 0 when the\n" +
            "  candidate server failed."
        ),
    },
    source = [
        paper(
            "https://arxiv.org/pdf/2511.02230",
            title = "Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live",
            venue = "arXiv 2025",
            summary = (
                "Optimize vLLM 0.11.0's Python scheduling and serving for multi-turn SWE-bench and " +
                "BFCL agent conversations on Qwen3-Coder-30B-A3B, cutting end-to-end latency on one " +
                "H100 while keeping correctness."
            ),
        ),
    ],
    display_name = "GPU vLLM Serving Optimization",
    category = "MLSys",
)
