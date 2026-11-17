"""Public semantics shared by debug and private KAMI harness execution.

Final judging reruns the public workloads and adds verifier-only shapes and seeds.
Pointer strides, NUM_ITER=1, alpha/beta, operation accounting, and validation
remain identical in public debug and private evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
WARM_PHASE_SALT = 0x6A09E667F3BCC909
TIMED_PHASE_SALT = 0xBB67AE8584CAA73B
TENSOR_SALTS = (
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
)

VALUE_SCALE_LOG2 = 10
VALUE_SCALE = 1 << VALUE_SCALE_LOG2
VALUE_MASK = (1 << 11) - 1
VALUE_BIAS = 1 << 10

ALPHA = 1.0
BETA = 0.5
ABSOLUTE_TOLERANCE = 1e-12
RELATIVE_TOLERANCE = 2e-13
MIN_BATCH_COUNT = 4096
HOST_DATA_BUDGET_BYTES = 1 << 30
DEVICE_DATA_BUDGET_BYTES = 1 << 30

# The table is public and intentionally shape-adaptive.  It is a pre-GPU
# calibration starting point, not a frozen performance threshold.
BATCH_COUNTS = {
    (16, 16, 24): 65536,
    (32, 64, 48): 8192,
    (64, 64, 32): 8192,
    (128, 64, 64): 4096,
}


@dataclass(frozen=True)
class MemoryBudget:
    batch_count: int
    device_bytes: int
    host_bytes: int
    bytes_per_device_cta: int
    bytes_per_host_cta: int


def batch_count(m: int, n: int, k: int) -> int:
    try:
        count = BATCH_COUNTS[(m, n, k)]
    except KeyError as error:
        raise ValueError(f"unapproved batched shape {(m, n, k)!r}") from error
    budget = memory_budget(m, n, k, count)
    if count < MIN_BATCH_COUNT:
        raise ValueError("batch count is below the public scheduling floor")
    if budget.device_bytes > DEVICE_DATA_BUDGET_BYTES:
        raise ValueError("device batch exceeds the 1 GiB data budget")
    if budget.host_bytes > HOST_DATA_BUDGET_BYTES:
        raise ValueError("host batch exceeds the 1 GiB data budget")
    return count


def memory_budget(m: int, n: int, k: int, count: int) -> MemoryBudget:
    if min(m, n, k, count) <= 0:
        raise ValueError("shape and batch dimensions must be positive")
    a_elements = m * k
    b_elements = n * k
    c_elements = m * n
    device_per_cta = 8 * (a_elements + b_elements + c_elements)
    # C input/output and a separately trusted golden C are both budgeted.
    host_per_cta = 8 * (a_elements + b_elements + 2 * c_elements)
    return MemoryBudget(
        batch_count=count,
        device_bytes=device_per_cta * count,
        host_bytes=host_per_cta * count,
        bytes_per_device_cta=device_per_cta,
        bytes_per_host_cta=host_per_cta,
    )


def scored_operation_count(m: int, n: int, k: int, count: int) -> int:
    """Count only real, independent GEMM work; NUM_ITER is exactly one."""
    if min(m, n, k, count) <= 0:
        raise ValueError("shape and batch dimensions must be positive")
    return 2 * m * n * k * count


def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def derive_phase(input_seed: int, input_profile: int, repetition: int, *, warm: bool) -> int:
    if not 0 <= input_profile <= 3:
        raise ValueError("input profile must be in [0, 3]")
    if repetition < 0:
        raise ValueError("repetition must be nonnegative")
    phase_salt = WARM_PHASE_SALT if warm else TIMED_PHASE_SALT
    value = input_seed & MASK64
    value ^= phase_salt
    value ^= ((input_profile + 1) * 0xD1B54A32D192ED03) & MASK64
    value ^= ((repetition + 1) * 0x94D049BB133111EB) & MASK64
    return splitmix64(value)


def raw_sample(phase: int, tensor: int, batch_index: int, element_index: int) -> int:
    if tensor not in (0, 1, 2):
        raise ValueError("tensor must identify A, B, or C")
    if batch_index < 0 or element_index < 0:
        raise ValueError("sample coordinates must be nonnegative")
    # Each tensor/phase uses a 64-bit counter permutation.  There is no small
    # modulo period to recognize; the unquantized stream has period 2^64.
    linear = ((batch_index << 32) | element_index) & MASK64
    domain = splitmix64((phase ^ TENSOR_SALTS[tensor]) & MASK64)
    return splitmix64((domain + linear * SPLITMIX_GAMMA) & MASK64)


def sample_numerator(
    phase: int, tensor: int, batch_index: int, element_index: int
) -> int:
    return int(raw_sample(phase, tensor, batch_index, element_index) & VALUE_MASK) - VALUE_BIAS


def sample_value(phase: int, tensor: int, batch_index: int, element_index: int) -> float:
    return math.ldexp(
        float(sample_numerator(phase, tensor, batch_index, element_index)),
        -VALUE_SCALE_LOG2,
    )


def expected_element(
    m: int,
    n: int,
    k: int,
    phase: int,
    batch_index: int,
    row: int,
    col: int,
) -> float:
    del m  # row bounds are checked by callers; the A stride is K.
    dot_numerator = 0
    a_base = row * k
    b_base = col * k
    for inner in range(k):
        dot_numerator += (
            sample_numerator(phase, 0, batch_index, a_base + inner)
            * sample_numerator(phase, 1, batch_index, b_base + inner)
        )
    c_numerator = sample_numerator(phase, 2, batch_index, row * n + col)
    return (
        math.ldexp(float(dot_numerator), -(2 * VALUE_SCALE_LOG2))
        + BETA * math.ldexp(float(c_numerator), -VALUE_SCALE_LOG2)
    )


def iter_expected_output(
    m: int, n: int, k: int, count: int, phase: int
) -> Iterable[float]:
    for batch_index in range(count):
        for row in range(m):
            for col in range(n):
                yield expected_element(m, n, k, phase, batch_index, row, col)


def verify_complete_output(
    observed: Sequence[float],
    m: int,
    n: int,
    k: int,
    count: int,
    phase: int,
) -> tuple[bool, int | None]:
    expected_size = count * m * n
    if len(observed) != expected_size:
        return False, min(len(observed), expected_size)
    for index, (actual, expected) in enumerate(
        zip(observed, iter_expected_output(m, n, k, count, phase))
    ):
        tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(expected)
        if not math.isfinite(actual) or abs(actual - expected) > tolerance:
            return False, index
    return True, None


def protocol_manifest() -> dict:
    rows = []
    for shape, count in BATCH_COUNTS.items():
        m, n, k = shape
        budget = memory_budget(m, n, k, count)
        rows.append(
            {
                "shape": [m, n, k],
                "batch_count": count,
                "scored_operations": scored_operation_count(m, n, k, count),
                "device_bytes": budget.device_bytes,
                "host_bytes": budget.host_bytes,
            }
        )
    return {
        "name": "KAMI kernel trusted batched harness adaptation",
        "num_iter": 1,
        "alpha": ALPHA,
        "beta": BETA,
        "minimum_batch_count": MIN_BATCH_COUNT,
        "device_data_budget_bytes": DEVICE_DATA_BUDGET_BYTES,
        "host_data_budget_bytes": HOST_DATA_BUDGET_BYTES,
        "workloads": rows,
    }
