"""Deterministic device-only compiler adapter for KAMI solver sources."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEVICE_REGION_START = "#define BLOCKS_PER_GRID 16384\n\n"
DEVICE_REGION_END = "\n\nint main(int argc, char *argv[])"
FIXED_ENTRY = "frontier_batched_block_gemm_entry"

_KERNEL_RE = re.compile(
    r"\b__global__\s+void\s+block_gemm_1d_double_mma\s*\(\s*"
    r"const\s+double\s*\*\s*A\s*,\s*"
    r"const\s+double\s*\*\s*B\s*,\s*"
    r"double\s*\*\s*C\s*,\s*"
    r"const\s+double\s+alpha\s*,\s*"
    r"const\s+double\s+beta\s*\)"
)

_HELPER_SIGNATURE = (
    "__device__ __forceinline__ void frontier_solver_block_gemm_helper("
    "const double *A, const double *B, double *C, "
    "const double alpha, const double beta)"
)


@dataclass(frozen=True)
class RenderedDeviceSource:
    text: str
    original_sha256: str
    device_region_sha256: str
    rendered_sha256: str


def split_device_region(text: str) -> tuple[str, str, str]:
    start = text.find(DEVICE_REGION_START)
    end = text.find(DEVICE_REGION_END)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("the source-locked device boundary is missing")
    body_start = start + len(DEVICE_REGION_START)
    return text[:body_start], text[body_start:end], text[end:]


def _masked_cpp(text: str) -> str:
    """Mask comments and literals while preserving offsets for signature search."""
    out = list(text)
    index = 0
    length = len(text)
    while index < length:
        if text.startswith("//", index):
            end = text.find("\n", index + 2)
            end = length if end < 0 else end
            for pos in range(index, end):
                out[pos] = " "
            index = end
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                raise ValueError("unterminated block comment")
            for pos in range(index, end + 2):
                if out[pos] != "\n":
                    out[pos] = " "
            index = end + 2
            continue
        if text.startswith('R"', index):
            delimiter_end = text.find("(", index + 2)
            if delimiter_end < 0 or delimiter_end - (index + 2) > 16:
                raise ValueError("invalid C++ raw-string delimiter")
            delimiter = text[index + 2:delimiter_end]
            closing = ")" + delimiter + '"'
            end = text.find(closing, delimiter_end + 1)
            if end < 0:
                raise ValueError("unterminated C++ raw string")
            end += len(closing)
            for pos in range(index, end):
                if out[pos] != "\n":
                    out[pos] = " "
            index = end
            continue
        if text[index] in ('"', "'"):
            quote = text[index]
            pos = index + 1
            while pos < length:
                if text[pos] == "\\":
                    pos += 2
                    continue
                if text[pos] == quote:
                    pos += 1
                    break
                pos += 1
            else:
                raise ValueError("unterminated C++ literal")
            for masked in range(index, min(pos, length)):
                if out[masked] != "\n":
                    out[masked] = " "
            index = pos
            continue
        index += 1
    return "".join(out)


def transform_kernel_to_helper(device_region: str) -> str:
    masked = _masked_cpp(device_region)
    matches = list(_KERNEL_RE.finditer(masked))
    if len(matches) != 1:
        raise ValueError("required original kernel signature must occur exactly once as code")
    match = matches[0]
    cursor = match.end()
    while cursor < len(masked) and masked[cursor].isspace():
        cursor += 1
    if cursor >= len(masked) or masked[cursor] != "{":
        raise ValueError("required kernel signature is not a definition")
    return device_region[:match.start()] + _HELPER_SIGNATURE + device_region[match.end():]


def render_device_source(source: str) -> RenderedDeviceSource:
    prefix, device_region, _host_suffix = split_device_region(source)
    transformed = transform_kernel_to_helper(device_region)
    wrapper = f"""

extern "C" __global__ void {FIXED_ENTRY}(
    const double *batch_A, const double *batch_B, double *batch_C,
    const double alpha, const double beta)
{{
    const unsigned long long batch_index =
        static_cast<unsigned long long>(blockIdx.x);
    const double *A = batch_A + batch_index *
        static_cast<unsigned long long>(M_BLOCK) * K_BLOCK;
    const double *B = batch_B + batch_index *
        static_cast<unsigned long long>(N_BLOCK) * K_BLOCK;
    double *C = batch_C + batch_index *
        static_cast<unsigned long long>(M_BLOCK) * N_BLOCK;
    frontier_solver_block_gemm_helper(A, B, C, alpha, beta);
}}
"""
    rendered = (
        prefix
        + "#undef NUM_ITER\n#define NUM_ITER 1\n"
        + transformed
        + wrapper
    )
    return RenderedDeviceSource(
        text=rendered,
        original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        device_region_sha256=hashlib.sha256(device_region.encode()).hexdigest(),
        rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
    )


def compile_cubin(
    source_path: Path,
    output_path: Path,
    *,
    m: int,
    n: int,
    k: int,
    active_ranks: int,
    allocated_ranks: int,
    nvcc: str = "nvcc",
    timeout_seconds: int = 300,
) -> tuple[RenderedDeviceSource, subprocess.CompletedProcess[str]]:
    rendered = render_device_source(source_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = output_path.with_suffix(".device.cu")
    generated.write_text(rendered.text, encoding="utf-8")
    command = [
        nvcc,
        "--cubin",
        "-std=c++17",
        "-arch=sm_90a",
        "-O3",
        "-Xptxas",
        "-O3",
        "--ptxas-options=-v",
        "-lineinfo",
        f"-DM_BLOCK={m}",
        f"-DN_BLOCK={n}",
        f"-DK_BLOCK={k}",
        f"-DNUM_RANK_BLOCK={active_ranks}",
        f"-DNUM_ALLOC_RANK_BLOCK={allocated_ranks}",
        "-I",
        str(source_path.parent),
        "-o",
        str(output_path),
        str(generated),
    ]
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return rendered, proc


def build_trusted_host(
    harness_dir: Path,
    output_path: Path,
    *,
    nvcc: str = "nvcc",
    timeout_seconds: int = 300,
) -> subprocess.CompletedProcess[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        nvcc,
        "-std=c++17",
        "-O2",
        "-Xcompiler=-fopenmp",
        "-I",
        str(harness_dir),
        str(harness_dir / "trusted_host.cpp"),
        "-L/usr/local/cuda/lib64/stubs",
        "-lcuda",
        "-lgomp",
        "-o",
        str(output_path),
    ]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
