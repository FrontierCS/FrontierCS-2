"""Implement the FP64 KAMI tensor-core block-GEMM kernel for one H100 and make it fast, ported from FrontierCS."""

load("//lib/fcs.star", "GPU_RUNNER", "H100", "fcs_task", "paper")

fcs_task(
    agent = {"cpu_millis": 3000, "memory_bytes": 4 * GiB},
    agent_lease = {
        "machine": GPU_RUNNER | {"gpu": H100},
        "instances": 1,
        "instance_seconds": 900,
        "cumulative_seconds": 900,
        "commands": 1024,
        "transfer_bytes": 512 * MiB,
    },
    judge = GPU_RUNNER | {"gpu": H100},
    train_calls = 12,
    train_cumulative_seconds = 21600,
    judge_timeout_seconds = 1800,
    submission_max_bytes = 256 * KiB,
    visible_metrics = {
        "tflops": (
            "Candidate crossover-adjusted validated batched throughput per train workload, in\n" +
            "  TFLOPS; an array in suite order, without workload labels."
        ),
    },
    source = [
        paper(
            "https://www.ssslab.cn/assets/papers/2025-wang-KAMI.pdf",
            title = "KAMI: Communication-Avoiding General Matrix Multiplication within a Single GPU",
            venue = "SC 2025",
            authors = [
                "Hemeng Wang",
                "Yang Du",
                "Sidu Li",
                "Xiaowen Tian",
                "Qingxiao Sun",
                "Weifeng Liu",
            ],
            year = 2025,
            summary = (
                "Implement the missing FP64 KAMI tensor-core block-GEMM device kernel on one " +
                "H100."
            ),
        ),
    ],
    display_name = "KAMI FP64 Block GEMM — trusted batched harness adaptation",
    category = "Arch",
)
