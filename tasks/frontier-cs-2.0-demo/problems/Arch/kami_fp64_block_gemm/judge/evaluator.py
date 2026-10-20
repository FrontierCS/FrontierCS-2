"""Release evaluator for the KAMI trusted batched harness adaptation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from submission_patch import apply_submission_patch

TASK_DIR = Path(__file__).resolve().parent
APP_DIR = TASK_DIR / "harbor/app"
HARNESS_DIR = APP_DIR / "frontier_harness"
sys.path.insert(0, str(APP_DIR))

from frontier_harness.adapter import (  # noqa: E402
    FIXED_ENTRY,
    build_trusted_host,
    compile_cubin,
    split_device_region,
    transform_kernel_to_helper,
)
from frontier_harness.protocol import (  # noqa: E402
    ALPHA,
    BETA,
    batch_count,
    memory_budget,
    protocol_manifest,
    scored_operation_count,
)

SOURCE = Path("src/block_gemm/KAMI/1d/fp64/block_gemm_1d_double_mma.cu")
REPETITIONS = 6
BUILD_TIMEOUT_SECONDS = 300
WORKLOAD_TIMEOUT_SECONDS = 900
SCORE_SCALE_TFLOPS = 20.0
STRICT_IMPROVEMENT_RATIO: float | None = 1.024
EXACT_GPU_NAME = "NVIDIA H100 80GB HBM3"

STUBBED_SOURCE_SHA256 = "54ca1476848d70bedc469a04e24c85224305bf4a5926bc798ec807b81b7ee08e"
REFERENCE_PATCH_SHA256 = "e104c534ba929b73d0ca1a7104aebdfc6bdb2a0a5a646d7119b14e2859d2228a"
RESTORED_UPSTREAM_SOURCE_SHA256 = "2b822cd274b0ff600d574c06cb0e52da97216c8d01b15d0b5932d97edaafebf9"

CONFIG: dict[str, Any] = {
    "submission": {
        "kind": "patch",
        "repo_root": "/app",
        "allow_paths": [
            str(SOURCE),
            "src/block_gemm/KAMI/1d/fp64/solver_extensions/",
        ],
        "max_changed_files": 12,
        "max_patch_bytes": 262144,
    }
}


class ReferenceVerificationError(RuntimeError):
    """The source-locked reference could not be restored, built or validated.

    A failed verification, never a candidate score: it propagates to the
    verifier adapter (and, on the Verifier GPU, to the GPU control result).
    """


class PlatformVerificationError(RuntimeError):
    """The Verifier GPU runner is not the declared single exact H100.

    A wrong or missing GPU assignment is a platform problem, never a candidate
    score: like ReferenceVerificationError it propagates as a failed
    verification.
    """


# Trusted-host messages raised by its own device-identity check, before any
# candidate or reference module is loaded.
HOST_PLATFORM_ERRORS = (
    "trusted harness error: trusted host requires exactly one GPU",
    f"trusted harness error: trusted host requires exactly one {EXACT_GPU_NAME}",
)


@dataclass(frozen=True)
class Workload:
    m: int
    n: int
    k: int
    active_ranks: int
    allocated_ranks: int
    input_profile: int
    input_seed: int
    batch_size: int

    @property
    def opaque_id(self) -> str:
        raw = (
            f"{self.m}:{self.n}:{self.k}:{self.active_ranks}:"
            f"{self.allocated_ranks}:{self.input_profile}:{self.input_seed}"
        )
        return "w-" + hashlib.sha256(raw.encode()).hexdigest()[:10]

    @property
    def batch_count(self) -> int:
        return self.batch_size

    @property
    def operation_count(self) -> int:
        return scored_operation_count(self.m, self.n, self.k, self.batch_count)


DEV_SUITE = (
    Workload(16, 16, 24, 1, 1, 0, 11, 65536),
    Workload(32, 64, 48, 2, 4, 1, 23, 8192),
    Workload(64, 64, 32, 4, 4, 2, 37, 8192),
    Workload(128, 64, 64, 8, 8, 3, 53, 4096),
)
HIDDEN_SUITE = (
    Workload(16, 16, 48, 1, 1, 2, 71, 65536),
    Workload(32, 64, 24, 2, 4, 3, 89, 16384),
    Workload(64, 64, 64, 4, 4, 0, 101, 8192),
    Workload(128, 64, 32, 8, 8, 1, 127, 4096),
)
FINAL_SUITE = DEV_SUITE + HIDDEN_SUITE


@dataclass(frozen=True)
class WorkloadMeasurement:
    candidate_samples: tuple[float, ...]
    reference_samples: tuple[float, ...]
    preprocess_milliseconds: tuple[float, ...]
    cpu_oracle_milliseconds: tuple[float, ...]
    pair_orders: tuple[str, ...]
    candidate_rendered_sha256: str
    reference_rendered_sha256: str


@dataclass(frozen=True)
class CrossoverAggregate:
    adjusted_tflops: float
    candidate_reference_order_median: float
    reference_candidate_order_median: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_orders(repetitions: int = REPETITIONS) -> tuple[str, ...]:
    if repetitions <= 0 or repetitions % 2:
        raise RuntimeError("paired repetitions must be positive and even")
    return tuple(
        "candidate-reference" if repetition % 2 == 0 else "reference-candidate"
        for repetition in range(repetitions)
    )


def _crossover_adjusted_tflops(
    samples: tuple[float, ...],
    pair_orders: tuple[str, ...],
) -> CrossoverAggregate:
    if len(samples) != len(pair_orders) or not samples:
        raise RuntimeError("crossover samples and pair orders must have equal nonzero length")
    if any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise RuntimeError("crossover samples must be finite and positive")
    candidate_reference = [
        value
        for value, order in zip(samples, pair_orders, strict=True)
        if order == "candidate-reference"
    ]
    reference_candidate = [
        value
        for value, order in zip(samples, pair_orders, strict=True)
        if order == "reference-candidate"
    ]
    if (
        len(candidate_reference) != len(reference_candidate)
        or len(candidate_reference) * 2 != len(samples)
        or any(
            order not in {"candidate-reference", "reference-candidate"}
            for order in pair_orders
        )
    ):
        raise RuntimeError("crossover pair orders must be exactly balanced")
    candidate_reference_median = statistics.median(candidate_reference)
    reference_candidate_median = statistics.median(reference_candidate)
    return CrossoverAggregate(
        adjusted_tflops=math.sqrt(
            candidate_reference_median * reference_candidate_median
        ),
        candidate_reference_order_median=candidate_reference_median,
        reference_candidate_order_median=reference_candidate_median,
    )


def _scratch_root() -> Path:
    configured = os.environ.get("FRONTIER_EVAL_SCRATCH")
    if configured:
        parent = Path(configured)
        parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="kami-batched-", dir=parent))
    return Path(tempfile.mkdtemp(prefix="kami-batched-eval-"))


def _score(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        return 0.0
    geomean = math.exp(sum(math.log(value) for value in values) / len(values))
    return max(0.0, min(100.0, 100.0 * geomean / (geomean + SCORE_SCALE_TFLOPS)))


def _validate_hardware() -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise PlatformVerificationError("nvidia-smi is unavailable on the verifier GPU runner")
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PlatformVerificationError("nvidia-smi could not query the verifier GPU") from error
    names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or names != [EXACT_GPU_NAME]:
        observed = ", ".join(names) if names else "none"
        raise PlatformVerificationError(
            f"expected exactly one {EXACT_GPU_NAME} device; observed {observed}")


def _validate_canonical_locks() -> None:
    if _sha256(APP_DIR / SOURCE) != STUBBED_SOURCE_SHA256:
        raise RuntimeError("public excised CUDA source lock does not match")
    if _sha256(TASK_DIR / "reference.patch") != REFERENCE_PATCH_SHA256:
        raise RuntimeError("source-locked reference patch does not match")


def _validate_solution_structure(tree: Path) -> None:
    pristine = (APP_DIR / SOURCE).read_text(encoding="utf-8")
    candidate = (tree / SOURCE).read_text(encoding="utf-8")
    pristine_prefix, _stub, pristine_suffix = split_device_region(pristine)
    candidate_prefix, device_region, candidate_suffix = split_device_region(candidate)
    if candidate_prefix != pristine_prefix or candidate_suffix != pristine_suffix:
        raise RuntimeError("bytes outside the original device region must remain source-locked")
    # A lexical structure check supports the one deterministic wrapper
    # conversion. It does not blacklist source or infer performed work.
    transform_kernel_to_helper(device_region)
    extension = tree / "src/block_gemm/KAMI/1d/fp64/solver_extensions"
    if extension.exists():
        for path in extension.rglob("*"):
            if path.is_symlink() or (
                path.exists() and not path.is_file() and not path.is_dir()
            ):
                raise RuntimeError("solver extension entries must be regular files")


def _apply_reference() -> Any:
    return apply_submission_patch(TASK_DIR / "reference.patch", TASK_DIR, CONFIG)


def _compile_module(tree: Path, workload: Workload, output: Path) -> tuple[str, str]:
    rendered, proc = compile_cubin(
        tree / SOURCE,
        output,
        m=workload.m,
        n=workload.n,
        k=workload.k,
        active_ranks=workload.active_ranks,
        allocated_ranks=workload.allocated_ranks,
    )
    if proc.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise RuntimeError(
            "device-only sm_90a compilation failed"
            + (f": {detail[-1][:180]}" if detail else "")
        )
    return rendered.rendered_sha256, proc.stderr


def _compile_trusted_host(output: Path) -> None:
    proc = build_trusted_host(HARNESS_DIR, output)
    if proc.returncode != 0 or not output.is_file():
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise RuntimeError(
            "trusted host compilation failed"
            + (f": {detail[-1][:180]}" if detail else "")
        )


def _parse_host_result(stdout: str) -> dict[str, Any]:
    records = [
        line.removeprefix("FCS_RESULT=")
        for line in stdout.splitlines()
        if line.startswith("FCS_RESULT=")
    ]
    if len(records) != 1:
        raise RuntimeError("trusted host emitted an unexpected result count")
    try:
        return json.loads(records[0])
    except json.JSONDecodeError as error:
        raise RuntimeError("trusted host emitted malformed JSON") from error


def _measure_workload(
    candidate_tree: Path,
    reference_tree: Path,
    trusted_host: Path,
    workload: Workload,
    scratch: Path,
) -> WorkloadMeasurement:
    pair_orders = _paired_orders()
    build = scratch / workload.opaque_id
    candidate_cubin = build / "candidate.cubin"
    reference_cubin = build / "reference.cubin"
    candidate_rendered, _candidate_compile_log = _compile_module(
        candidate_tree, workload, candidate_cubin
    )
    try:
        reference_rendered, _reference_compile_log = _compile_module(
            reference_tree, workload, reference_cubin
        )
    except Exception as error:
        raise ReferenceVerificationError("source-locked reference did not compile") from error
    command = [
        str(trusted_host),
        "--candidate", str(candidate_cubin),
        "--reference", str(reference_cubin),
        "--m", str(workload.m), "--n", str(workload.n), "--k", str(workload.k),
        "--active-ranks", str(workload.active_ranks),
        "--allocated-ranks", str(workload.allocated_ranks),
        "--batch", str(workload.batch_count),
        "--profile", str(workload.input_profile),
        "--seed", str(workload.input_seed),
        "--repetitions", str(REPETITIONS),
    ]
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=WORKLOAD_TIMEOUT_SECONDS,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        # The trusted host labels which module failed full-output validation.
        # An invalid reference never sets the bar and is not a candidate score.
        if detail and detail[-1].startswith("trusted harness error: reference "):
            raise ReferenceVerificationError("source-locked reference failed full-output validation")
        if detail and detail[-1].strip() in HOST_PLATFORM_ERRORS:
            raise PlatformVerificationError("trusted host did not find exactly one exact H100")
        raise RuntimeError(
            "trusted batched execution or full-output validation failed"
            + (f": {detail[-1][:180]}" if detail else "")
        )
    result = _parse_host_result(proc.stdout)
    required = {
        "batch_count": workload.batch_count,
        "operation_count": workload.operation_count,
        "num_iter": 1,
        "alpha": ALPHA,
        "beta": BETA,
        "warm_validated": True,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise RuntimeError(f"trusted host protocol mismatch for {key}")
    candidate_samples = tuple(map(float, result.get("candidate_tflops", [])))
    reference_samples = tuple(map(float, result.get("reference_tflops", [])))
    preprocess = tuple(map(float, result.get("preprocess_milliseconds", [])))
    oracle = tuple(map(float, result.get("cpu_oracle_milliseconds", [])))
    sequences = (candidate_samples, reference_samples, preprocess, oracle)
    if any(len(values) != REPETITIONS for values in sequences):
        raise RuntimeError("trusted host did not return every paired repetition")
    if any(
        not math.isfinite(value) or value <= 0.0
        for values in sequences
        for value in values
    ):
        raise RuntimeError("trusted host returned invalid timing or throughput")
    return WorkloadMeasurement(
        candidate_samples=candidate_samples,
        reference_samples=reference_samples,
        preprocess_milliseconds=preprocess,
        cpu_oracle_milliseconds=oracle,
        pair_orders=pair_orders,
        candidate_rendered_sha256=candidate_rendered,
        reference_rendered_sha256=reference_rendered,
    )


def _strictly_beats_reference(
    candidate_tflops: float,
    reference_tflops: float,
    required_ratio: float,
) -> bool:
    if not math.isfinite(required_ratio) or required_ratio <= 1.0:
        raise RuntimeError("strict improvement ratio must be finite and greater than one")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (candidate_tflops, reference_tflops)
    ):
        return False
    return candidate_tflops / reference_tflops > required_ratio


def _clears_reference(candidate_tflops: float, reference_tflops: float) -> bool:
    if STRICT_IMPROVEMENT_RATIO is None:
        return False
    return _strictly_beats_reference(candidate_tflops, reference_tflops, STRICT_IMPROVEMENT_RATIO)


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    candidate_result = None
    reference_result = None
    scratch = None
    try:
        _validate_hardware()
        _validate_canonical_locks()
        scratch = _scratch_root()
        candidate_result = apply_submission_patch(solution_path, TASK_DIR, CONFIG)
        if not candidate_result.ok or candidate_result.workdir is None:
            return 0.0, 0.0, candidate_result.message or "invalid submission", {}
        reference_result = _apply_reference()
        if not reference_result.ok or reference_result.workdir is None:
            raise ReferenceVerificationError("judge reference could not be restored")
        _validate_solution_structure(candidate_result.workdir)
        try:
            _validate_solution_structure(reference_result.workdir)
        except RuntimeError as error:
            raise ReferenceVerificationError("judge reference structure is invalid") from error
        if _sha256(reference_result.workdir / SOURCE) != RESTORED_UPSTREAM_SOURCE_SHA256:
            raise ReferenceVerificationError("reference patch did not restore the source-locked upstream CUDA")

        trusted_host = scratch / "trusted_host"
        _compile_trusted_host(trusted_host)
        suite = FINAL_SUITE if os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final" else DEV_SUITE
        candidate_values: list[float] = []
        reference_values: list[float] = []
        private_rows: list[dict[str, Any]] = []
        pass_rows: list[dict[str, Any]] = []
        for workload in suite:
            measured = _measure_workload(
                candidate_result.workdir,
                reference_result.workdir,
                trusted_host,
                workload,
                scratch,
            )
            candidate_aggregate = _crossover_adjusted_tflops(
                measured.candidate_samples, measured.pair_orders
            )
            reference_aggregate = _crossover_adjusted_tflops(
                measured.reference_samples, measured.pair_orders
            )
            candidate_tflops = candidate_aggregate.adjusted_tflops
            reference_tflops = reference_aggregate.adjusted_tflops
            candidate_values.append(candidate_tflops)
            reference_values.append(reference_tflops)
            pass_rows.append(
                {
                    "workload": workload.opaque_id,
                    "metrics": {"tflops": _clears_reference(candidate_tflops, reference_tflops)},
                }
            )
            budget = memory_budget(workload.m, workload.n, workload.k, workload.batch_count)
            private_rows.append(
                {
                    "workload": workload.opaque_id,
                    "shape": [workload.m, workload.n, workload.k],
                    "active_ranks": workload.active_ranks,
                    "allocated_ranks": workload.allocated_ranks,
                    "batch_count": workload.batch_count,
                    "operation_count": workload.operation_count,
                    "device_bytes": budget.device_bytes,
                    "host_bytes": budget.host_bytes,
                    "candidate_tflops": candidate_tflops,
                    "reference_tflops": reference_tflops,
                    "candidate_reference_ratio": candidate_tflops / reference_tflops,
                    "crossover_aggregation": {
                        "method": "geometric mean of per-order medians",
                        "candidate_first_candidate_median_tflops": (
                            candidate_aggregate.candidate_reference_order_median
                        ),
                        "candidate_second_candidate_median_tflops": (
                            candidate_aggregate.reference_candidate_order_median
                        ),
                        "reference_second_reference_median_tflops": (
                            reference_aggregate.candidate_reference_order_median
                        ),
                        "reference_first_reference_median_tflops": (
                            reference_aggregate.reference_candidate_order_median
                        ),
                    },
                    "candidate_samples": list(measured.candidate_samples),
                    "reference_samples": list(measured.reference_samples),
                    "preprocess_milliseconds": list(measured.preprocess_milliseconds),
                    "cpu_oracle_milliseconds": list(measured.cpu_oracle_milliseconds),
                    "pair_orders": list(measured.pair_orders),
                    "candidate_rendered_sha256": measured.candidate_rendered_sha256,
                    "reference_rendered_sha256": measured.reference_rendered_sha256,
                }
            )

        score = _score(candidate_values)
        reference_score = _score(reference_values)
        workload_wins = [row["metrics"]["tflops"] for row in pass_rows]
        metrics = {
            "beat_spec_workload_passes": pass_rows,
            "beats_reference": int(bool(workload_wins) and all(workload_wins)),
            "reference_score": reference_score,
            "reference_improvement_ratio": STRICT_IMPROVEMENT_RATIO,
            "margin": score - reference_score,
            "tflops": candidate_values,
            "paired_reference_tflops": private_rows,
            "harness_protocol": protocol_manifest(),
            "fixed_device_entry": FIXED_ENTRY,
        }
        return (
            score,
            score,
            "valid KAMI trusted batched results",
            metrics,
        )
    except (ReferenceVerificationError, PlatformVerificationError):
        # Failed verification, not a candidate score.
        raise
    except subprocess.TimeoutExpired:
        return 0.0, 0.0, "trusted build, preprocessing, oracle, or workload timed out", {}
    except Exception as error:
        return 0.0, 0.0, f"evaluation failed: {str(error)[:220]}", {}
    finally:
        for result in (candidate_result, reference_result):
            workdir = getattr(result, "workdir", None)
            if workdir is not None:
                shutil.rmtree(workdir.parent, ignore_errors=True)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"score": 0.0, "message": "usage: evaluator.py SOLUTION_PATCH"}))
        raise SystemExit(0)
    score, unbounded, message, metrics = evaluate(sys.argv[1])
    print(json.dumps({
        "score": score,
        "score_unbounded": unbounded,
        "message": message,
        "metrics": metrics,
    }))
