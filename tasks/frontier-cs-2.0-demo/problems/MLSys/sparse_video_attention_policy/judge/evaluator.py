#!/usr/bin/env python3
"""GPU evaluator for the stripped Wan + HunyuanVideo SVG attention task.

The authoritative path invokes the retained 720p shell entrypoint of the
workload's model (Wan2.1 or HunyuanVideo) for the candidate. Dense and
exact-upstream videos prepared once through that same path are hash-checked,
decoded, and scored as the paired target and frontier. Unit tests replace only
the expensive GPU runner; patch application, bookkeeping, and score arithmetic
remain real.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
from submission_patch import apply_submission_patch


TASK_DIR = _MODULE_DIR
REFERENCE_PATCH = TASK_DIR / "reference.patch"
DENSE_TARGET_DIR = TASK_DIR / "dense_targets"
DENSE_MANIFEST_SCHEMA_VERSION = 2
TARGET_READY = "ready"
TARGET_PENDING = "PENDING"

CONFIG = {
    "submission": {
        "kind": "patch",
        "repo_root": "/app",
        "allow_paths": [
            "svg/models/wan/attention.py",
            "svg/models/hyvideo/attention.py",
            "svg/kmeans_utils.py",
            "svg/solver_extensions/",
        ],
        "max_changed_files": 12,
        "max_patch_bytes": 262144,
    }
}

EXPECTED_FRAMES = 81
EXPECTED_HEIGHT = 720
EXPECTED_WIDTH = 1280
# HunyuanVideo runs at the same 81-frame (21 latent frame) scale as Wan. This is a
# task deviation from the upstream/paper HunyuanVideo setting of 129 frames; see
# docs/task-authoring/sparse_video_attention_policy/hunyuan-workloads.md.
HUNYUAN_NUM_FRAMES = 81


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    entrypoint: Path
    program: str
    pipeline_class: str
    model_target: str
    num_frames: int
    # (transformer attribute, attention attribute) lists holding the scored
    # self-attention processors that the retained integration installs.
    processor_blocks: tuple[tuple[str, str], ...]
    # Class-level processor caches that the retained integration keys by layer
    # and that must not carry warm-up state into the scored call.
    class_state_caches: tuple[str, ...]


MODELS = {
    "wan": ModelSpec(
        name="wan",
        display_name="Wan2.1",
        entrypoint=Path("scripts/wan/wan_t2v_720p_ear.sh"),
        program="wan_t2v_inference.py",
        pipeline_class="WanPipeline",
        model_target="Wan2.1-T2V-14B-Diffusers",
        num_frames=81,
        processor_blocks=(("blocks", "attn1"),),
        class_state_caches=(),
    ),
    "hunyuan": ModelSpec(
        name="hunyuan",
        display_name="HunyuanVideo",
        entrypoint=Path("scripts/hyvideo/hyvideo_t2v_720p_ear.sh"),
        program="hyvideo_t2v_inference.py",
        pipeline_class="HunyuanVideoPipeline",
        model_target="HunyuanVideo",
        num_frames=HUNYUAN_NUM_FRAMES,
        processor_blocks=(("transformer_blocks", "attn"), ("single_transformer_blocks", "attn")),
        class_state_caches=("centroids_init", "q_centroids", "k_centroids"),
    ),
}
WORKLOAD_PREFIX = {"wan": "wan_t2v_720p", "hunyuan": "hunyuan_t2v_720p"}
# Compatibility names for authoring utilities written against the Wan-only task.
ENTRYPOINT = MODELS["wan"].entrypoint
MODEL_TARGET = MODELS["wan"].model_target

# (model, prompt_id, seed). Each clip is its own workload. Order fixes the
# four-GPU waves: graded runs Wan 0-3, Wan 4-7, then HunyuanVideo 0-3.
FEEDBACK_SUITE = (
    ("wan", 1, 104729),
    ("wan", 2, 104759),
    ("hunyuan", 3, 104779),
    ("hunyuan", 4, 104789),
)
GRADED_SUITE = (
    ("wan", 1, 130363), ("wan", 2, 130367), ("wan", 3, 130399), ("wan", 4, 130409),
    ("wan", 1, 130411), ("wan", 2, 130423), ("wan", 3, 130439), ("wan", 4, 130447),
    ("hunyuan", 1, 130457), ("hunyuan", 2, 130469), ("hunyuan", 3, 130477), ("hunyuan", 4, 130483),
)
FEEDBACK_SEEDS = tuple(seed for _model, _prompt, seed in FEEDBACK_SUITE)
GRADED_SEEDS = tuple(seed for _model, _prompt, seed in GRADED_SUITE)
FEEDBACK_PROMPT_IDS = tuple(prompt for _model, prompt, _seed in FEEDBACK_SUITE)
GRADED_PROMPT_IDS = tuple(prompt for _model, prompt, _seed in GRADED_SUITE)
FEEDBACK_WORKLOADS = 4
GRADED_WORKLOADS = 12
EXPECTED_VERIFIER_GPUS = 4
BUILD_TIMEOUT_SECONDS = 600
GENERATION_TIMEOUT_SECONDS = 3600
WORKER_TIMEOUT_SECONDS = GENERATION_TIMEOUT_SECONDS + 300
# 600 s build + three 3,900 s graded worker waves = 12,300 s, leaving 300 s for
# patch application, target checks, orchestration and cleanup.
OUTER_TIMEOUT_SECONDS = 12600
MAX_WORKER_OUTPUT_BYTES = 64 * 1024
MAX_OPERATOR_DIAGNOSTIC_BYTES = 4 * 1024

# Score calibration.  All are challenge constants (not from materials).
PSNR_FLOOR_DB = 15.0
PSNR_SPAN_DB = 20.0
SSIM_FLOOR = 0.50
SSIM_SPAN = 0.50
LPIPS_CEILING = 0.35
MIN_QUALITY_CREDIT = 0.25
EFFICIENCY_CREDIT = 0.75
SPEEDUP_SPAN = 1.0

# Beat-reference calibration. Latency must improve by more than the largest
# observed repeated-run timing spread, while fidelity is treated as
# non-inferior inside small decoded-video measurement bands. These thresholds
# affect only the binary beat criterion, not the continuous score.
MIN_LATENCY_SPEEDUP_FRACTION = 0.04
PSNR_NONINFERIOR_TOLERANCE_DB = 0.25
SSIM_NONINFERIOR_TOLERANCE = 0.01
LPIPS_NONINFERIOR_TOLERANCE = 0.005

SPEC_METRICS = (
    "e2e_latency_seconds",
    "psnr_db",
    "ssim",
    "lpips",
)

# Added code may implement arbitrary tensor policies, but it may not reach into
# the private runner or replace the class whose call boundary is timed.
FORBIDDEN_ADDED_PATTERNS = (
    re.compile(r"FRONTIER_(?:SUBMISSION_ROLE|MODEL_ROOT)"),
    re.compile(r"HARBOR_"),
    re.compile(r"(?:^|\W)(?:inspect|subprocess|socket|runpy)(?:\W|$)"),
    re.compile(r"(?:^|\W)(?:open|listdir|scandir|walk|glob)(?:\W|$)"),
    re.compile(r"(?:__closure__|__dict__|__import__|_getframe|/proc/|perf_counter|WanPipeline|HunyuanVideoPipeline|sys\.modules)"),
)


@dataclass(frozen=True)
class Workload:
    workload_id: str
    seed: int
    prompt_id: int
    model: str = "wan"

    @property
    def spec(self) -> ModelSpec:
        return MODELS[self.model]


class SuiteTargetsUnavailable(Exception):
    """Prepared dense/reference targets for the selected suite are missing.

    This is deliberately not a RuntimeError: the evaluator re-raises it so the
    verifier reports an evaluator failure instead of a scored 0 verdict.
    """


class WorkloadWorkerFailure(RuntimeError):
    """A generic public failure plus bounded operator-only process evidence."""

    def __init__(self, workload_id: str, process: subprocess.CompletedProcess):
        super().__init__(f"GPU worker failed for {workload_id}")
        stdout = process.stdout or b""
        stderr = process.stderr or b""
        tail = stderr[-MAX_OPERATOR_DIAGNOSTIC_BYTES:].decode("utf-8", "replace")
        stage_context = {}
        for line in stderr.decode("utf-8", "replace").splitlines():
            if line.startswith("FRONTIER_OPERATOR_STAGE="):
                try:
                    stage_context = json.loads(line.split("=", 1)[1])
                except json.JSONDecodeError:
                    pass
        self.operator_diagnostic = {
            "workload": workload_id,
            "returncode": process.returncode,
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "stderr_tail": tail,
            "stderr_tail_truncated": len(stderr) > MAX_OPERATOR_DIAGNOSTIC_BYTES,
            "stage_context": stage_context,
        }


_LAST_GENERATION_DIAGNOSTIC = None
_GENERATION_PHASES = (
    "imports_start", "imports_complete", "entrypoint_start", "pipeline_enter",
    "warmup_complete", "inference_complete", "entrypoint_complete",
)


def _generation_diagnostic(phase_path, started, stderr_bytes, stderr_hash, tail, capture_complete):
    phases = []
    phase_path = Path(phase_path)
    try:
        with phase_path.open("rb") as handle:
            raw = handle.read(4097)
        if len(raw) <= 4096:
            records = json.loads(raw)
            if isinstance(records, list) and len(records) <= len(_GENERATION_PHASES):
                previous_elapsed = 0.0
                for record in records:
                    if (not isinstance(record, dict) or set(record) != {"stage", "elapsed_seconds"}
                            or record["stage"] != _GENERATION_PHASES[len(phases)]
                            or isinstance(record["elapsed_seconds"], bool)
                            or not isinstance(record["elapsed_seconds"], (int, float))
                            or not math.isfinite(record["elapsed_seconds"])
                            or record["elapsed_seconds"] < previous_elapsed):
                        phases = []
                        break
                    phases.append(record)
                    previous_elapsed = record["elapsed_seconds"]
    except (OSError, ValueError, TypeError, RecursionError):
        pass
    elapsed = time.monotonic() - started
    if phases and phases[-1]["elapsed_seconds"] > elapsed:
        phases = []
    return {
        "subprocess_elapsed_seconds": elapsed,
        "stderr_observed_bytes": stderr_bytes,
        "stderr_observed_sha256": stderr_hash.hexdigest(),
        "stderr_tail": bytes(tail).decode("utf-8", "replace"),
        "stderr_tail_truncated": stderr_bytes > len(tail),
        "stderr_capture_complete": capture_complete,
        "phases": phases,
    }


def _run_generation_process(
    command: list[str], *, cwd: Path, env: dict[str, str], timeout: int
) -> subprocess.CompletedProcess:
    """Run generation while retaining only the last diagnostic stderr bytes."""
    global _LAST_GENERATION_DIAGNOSTIC
    started = time.monotonic()
    stderr_count = 0
    stderr_hash = hashlib.sha256()
    stderr_lock = threading.Lock()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    if process.stderr is None:
        raise RuntimeError("generation stderr pipe unavailable")
    stderr_tail = bytearray()
    reader_errors: list[BaseException] = []

    def read_stderr() -> None:
        nonlocal stderr_count
        try:
            while chunk := process.stderr.read(64 * 1024):
                with stderr_lock:
                    stderr_count += len(chunk)
                    stderr_hash.update(chunk)
                    if len(chunk) >= 1024:
                        stderr_tail[:] = chunk[-1024:]
                    else:
                        excess = max(
                            0,
                            len(stderr_tail) + len(chunk)
                            - 1024,
                        )
                        if excess:
                            del stderr_tail[:excess]
                        stderr_tail.extend(chunk)
        except BaseException as error:
            reader_errors.append(error)

    reader = threading.Thread(
        target=read_stderr,
        name="svg-generation-stderr",
        daemon=True,
    )
    reader.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, 9)
        except ProcessLookupError:
            pass
        process.wait()
        reader.join(timeout=5)
        if not reader.is_alive():
            process.stderr.close()
        with stderr_lock:
            captured_tail = bytes(stderr_tail)
            _LAST_GENERATION_DIAGNOSTIC = _generation_diagnostic(env.get("FRONTIER_GENERATION_PHASE_PATH", "/nonexistent"), started, stderr_count, stderr_hash, stderr_tail, not reader.is_alive() and not reader_errors)
        expired = subprocess.TimeoutExpired(command, timeout, stderr=captured_tail)
        expired.operator_generation = _LAST_GENERATION_DIAGNOSTIC
        raise expired from error
    reader.join(timeout=5)
    if not reader.is_alive():
        process.stderr.close()
    if reader.is_alive() or reader_errors:
        raise RuntimeError("generation stderr capture failed")
    with stderr_lock:
        _LAST_GENERATION_DIAGNOSTIC = _generation_diagnostic(env.get("FRONTIER_GENERATION_PHASE_PATH", "/nonexistent"), started, stderr_count, stderr_hash, stderr_tail, not reader.is_alive() and not reader_errors)
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout=b"",
        stderr=bytes(stderr_tail),
    )


def _generation_failure_message(process: subprocess.CompletedProcess) -> str:
    """Return bounded detail that remains inside the outer worker traceback."""
    stderr = process.stderr or b""
    tail = stderr[-MAX_OPERATOR_DIAGNOSTIC_BYTES:].decode("utf-8", "replace")
    return (
        f"Video generation stderr tail:\n{tail}\n"
        f"Video generation process exited with code {process.returncode}"
    )


_LPIPS_MODEL: Any = None


def _workloads(final: bool) -> tuple[Workload, ...]:
    role = "graded" if final else "feedback"
    suite = GRADED_SUITE if final else FEEDBACK_SUITE
    counters = {name: 0 for name in MODELS}
    workloads = []
    for model, prompt_id, seed in suite:
        index = counters[model]
        counters[model] += 1
        workloads.append(Workload(f"{WORKLOAD_PREFIX[model]}_{role}_{index}", seed, prompt_id, model))
    return tuple(workloads)


def _waves(workloads: tuple[Workload, ...]) -> tuple[tuple[Workload, ...], ...]:
    """Consecutive EXPECTED_VERIFIER_GPUS-sized waves, one workload per GPU."""
    return tuple(
        workloads[start:start + EXPECTED_VERIFIER_GPUS]
        for start in range(0, len(workloads), EXPECTED_VERIFIER_GPUS)
    )


def _workload(final: bool) -> Workload:
    """Compatibility accessor for authoring utilities that need one sample."""
    return _workloads(final)[0]


def _scratch_root() -> Path:
    requested = Path(os.environ.get("FRONTIER_EVAL_SCRATCH", "/workspace/frontier-eval"))
    try:
        requested.mkdir(parents=True, exist_ok=True)
        probe = requested / ".write-probe"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
        return requested
    except OSError:
        fallback = TASK_DIR.parent / "scratch" / "evaluator"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _apply_patch(path: str | Path):
    # Keep the 31 MiB source copies and CUDA builds off RAM-backed /tmp.
    tempfile.tempdir = str(_scratch_root())
    old_ceiling = os.environ.get("GIT_CEILING_DIRECTORIES")
    os.environ["GIT_CEILING_DIRECTORIES"] = str(_scratch_root())
    try:
        return apply_submission_patch(str(path), TASK_DIR, CONFIG)
    finally:
        if old_ceiling is None:
            os.environ.pop("GIT_CEILING_DIRECTORIES", None)
        else:
            os.environ["GIT_CEILING_DIRECTORIES"] = old_ceiling


def _empty_metrics() -> dict[str, Any]:
    workloads = _workloads(os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final")
    return {
        "beats_reference": 0,
        "beat_spec_workload_passes": [
            {"workload": workload.workload_id,
             "metrics": {name: False for name in SPEC_METRICS}}
            for workload in workloads
        ],
    }


def _added_source(patch_path: str | Path) -> str:
    text = Path(patch_path).read_text(encoding="utf-8", errors="replace")
    return "\n".join(
        line[1:]
        for line in text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def _task_policy_error(patch_path: str | Path) -> str | None:
    added = _added_source(patch_path)
    for pattern in FORBIDDEN_ADDED_PATTERNS:
        match = pattern.search(added)
        if match:
            return f"submitted source uses forbidden harness-access token: {match.group(0)[:40]}"
    return None


def _check_public_contract(root: Path) -> None:
    expected: dict[Path, set[str]] = {
        root / "svg/models/wan/attention.py": {"WanAttn_EARAttn_Processor"},
        root / "svg/models/hyvideo/attention.py": {"Hunyuan_EARAttn_Processor"},
        root / "svg/kmeans_utils.py": {
            "identify_dynamic_map_estimated",
            "scatter_mean_fused",
            "dynamic_block_sparse_prune_fwd_flashinfer",
        },
    }
    for path, names in expected.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing = names - found
        if missing:
            raise RuntimeError(f"required callable is missing from {path.name}")


def _model_path(model: str = "wan") -> Path:
    spec = MODELS[model]
    model_root = os.environ.get("FRONTIER_MODEL_ROOT")
    if not model_root:
        raise RuntimeError(f"offline {spec.display_name} checkpoint is unavailable")
    path = Path(model_root) / spec.model_target
    if not (path / "model_index.json").is_file():
        raise RuntimeError(f"offline {spec.display_name} checkpoint is incomplete")
    return path


def _load_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads((DENSE_TARGET_DIR / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SuiteTargetsUnavailable("cached target manifest is unreadable") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != DENSE_MANIFEST_SCHEMA_VERSION
        or manifest.get("parallel_gpu_workers") != EXPECTED_VERIFIER_GPUS
        or not isinstance(manifest.get("models"), dict)
        or not isinstance(manifest.get("targets"), dict)
    ):
        raise SuiteTargetsUnavailable("cached target manifest schema is invalid")
    for name, spec in MODELS.items():
        model = manifest["models"].get(name)
        if (
            not isinstance(model, dict)
            or model.get("entrypoint") != str(spec.entrypoint)
            or model.get("num_frames") != spec.num_frames
        ):
            raise SuiteTargetsUnavailable("cached target manifest model record is invalid")
    return manifest


def _target_record(manifest: dict[str, Any], workload: Workload) -> dict[str, Any]:
    """Return the ready per-workload record or fail closed.

    A target that has not been generated yet is an explicit PENDING placeholder.
    Any non-ready record stops the evaluation before a verdict can be produced.
    """
    record = manifest["targets"].get(workload.workload_id)
    if (
        not isinstance(record, dict)
        or record.get("model") != workload.model
        or record.get("prompt_id") != workload.prompt_id
        or record.get("seed") != workload.seed
    ):
        raise SuiteTargetsUnavailable("cached target record does not match the suite")
    if record.get("status") != TARGET_READY:
        raise SuiteTargetsUnavailable("cached targets for this suite are not prepared")
    for kind in ("dense", "reference"):
        if not isinstance(record.get(kind), dict):
            raise SuiteTargetsUnavailable("cached target record is incomplete")
    return record


def _require_suite_targets(workloads: tuple[Workload, ...]) -> None:
    """Check every target record before any build or GPU work starts."""
    manifest = _load_manifest()
    for workload in workloads:
        _target_record(manifest, workload)


def _cached_target(workload: Workload, kind: str) -> tuple[float, Any]:
    """Load one immutable output prepared through the retained entrypoint."""
    if kind not in {"dense", "reference"}:
        raise RuntimeError("cached target kind is invalid")
    try:
        workload_record = _target_record(_load_manifest(), workload)
    except SuiteTargetsUnavailable as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        record = workload_record[kind]
        relative = str(record["path"])
        expected_sha256 = str(record["sha256"])
        latency = float(record["e2e_latency_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("cached target manifest is invalid") from exc
    path = DENSE_TARGET_DIR / relative
    if (
        Path(relative).name != relative
        or len(expected_sha256) != 64
        or not math.isfinite(latency)
        or latency <= 0
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256
    ):
        raise RuntimeError("cached target is invalid")
    return latency, _load_video(path)


def _build_extension(root: Path) -> None:
    kernels = root / "svg/kernels"
    env = os.environ.copy()
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = "8"
    proc = subprocess.run(
        ["bash", "setup.sh"],
        cwd=kernels,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=BUILD_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0 or not list((kernels / "build").glob("_kernels*.so")):
        raise RuntimeError("CUDA extension build failed")


def _runner_source(
    target: Path,
    model_path: Path,
    output_path: Path,
    timing_path: Path,
    logging_path: Path,
    seed: int,
    prompt: str,
    pattern: str,
    spec: ModelSpec = MODELS["wan"],
) -> str:
    # Values are literals in a private, per-run shim.  Candidate code sees only
    # the normal upstream argv, while the timer path remains in this closure.
    return f'''#!/usr/bin/env python3
import json
import os
import random
import runpy
import sys
import time
PHASE_PATH = {str(timing_path.with_name("phases.json"))!r}
_operator_phase_origin = time.monotonic()
_operator_phases = []
def _operator_phase(stage):
    if len(_operator_phases) >= 7:
        return
    _operator_phases.append({{"stage": stage, "elapsed_seconds": time.monotonic() - _operator_phase_origin}})
    try:
        temporary = PHASE_PATH + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(_operator_phases, handle)
        os.replace(temporary, PHASE_PATH)
    except OSError:
        pass

_operator_phase("imports_start")
import numpy as np
import torch
from diffusers import {spec.pipeline_class} as TimedPipeline
_operator_phase("imports_complete")

TARGET = {str(target)!r}
PROGRAM = {spec.program!r}
MODEL = {str(model_path)!r}
OUTPUT = {str(output_path)!r}
TIMING = {str(timing_path)!r}
LOGGING = {str(logging_path)!r}
SEED = {int(seed)!r}
PROMPT = {prompt!r}
PATTERN = {pattern!r}
NUM_FRAMES = {int(spec.num_frames)!r}
PROCESSOR_BLOCKS = {spec.processor_blocks!r}
CLASS_STATE_CACHES = {spec.class_state_caches!r}

args = list(sys.argv[1:])
if not args or os.path.basename(args[0]) != PROGRAM:
    raise SystemExit("frontier python shim only runs the retained " + PROGRAM + " entrypoint")

def set_arg(flag, value):
    if flag in args:
        args[args.index(flag) + 1] = str(value)
    else:
        args.extend([flag, str(value)])

set_arg("--model_id", MODEL)
set_arg("--seed", SEED)
set_arg("--prompt", PROMPT)
set_arg("--pattern", PATTERN)
set_arg("--num_frames", NUM_FRAMES)
set_arg("--output_file", OUTPUT)
set_arg("--logging_file", LOGGING)
sys.argv = [TARGET] + args[1:]

original_call = TimedPipeline.__call__
clock = time.perf_counter
sync = torch.cuda.synchronize

def reseed():
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rebuild_processors(pipe):
    # The retained integration's public constructor is exactly layer_idx -> processor.
    classes = set()
    for blocks_name, attention_name in PROCESSOR_BLOCKS:
        for block in getattr(pipe.transformer, blocks_name):
            attention = getattr(block, attention_name)
            current = attention.processor
            classes.add(type(current))
            attention.set_processor(type(current)(layer_idx=current.layer_idx))
    # HunyuanVideo's retained k-means processors keep per-layer caches in
    # class-level dicts; a new instance alone would still see warm-up centroids.
    for processor_class in classes:
        for klass in processor_class.__mro__:
            for name in CLASS_STATE_CACHES:
                cache = vars(klass).get(name)
                if isinstance(cache, dict):
                    cache.clear()

def measured_call(self, *call_args, **call_kwargs):
    # Two steps traverse both the initial exact path and the later sparse path,
    # compiling every timed kernel without repeating a full 50-step video.
    _operator_phase("pipeline_enter")
    warm_kwargs = dict(call_kwargs)
    warm_kwargs["num_inference_steps"] = 2
    warm = original_call(self, *call_args, **warm_kwargs)
    del warm
    _operator_phase("warmup_complete")
    # Warm-up must not donate clustering state to the scored trajectory.
    if PATTERN == "EAR":
        rebuild_processors(self)
    reseed()
    sync()
    start = clock()
    result = original_call(self, *call_args, **call_kwargs)
    sync()
    elapsed = clock() - start
    _operator_phase("inference_complete")
    with open(TIMING, "w", encoding="utf-8") as handle:
        json.dump({{"seconds": elapsed}}, handle)
    return result

TimedPipeline.__call__ = measured_call
_operator_phase("entrypoint_start")
runpy.run_path(TARGET, run_name="__main__")
_operator_phase("entrypoint_complete")
'''


def _run_generation(
    root: Path, workload: Workload, pattern: str, keep_video: Path | None = None
) -> tuple[float, Any]:
    """Run one fresh generation process; optionally retain its MP4 for authoring."""
    spec = workload.spec
    run_dir = Path(tempfile.mkdtemp(prefix=f"{spec.name}-{pattern.lower()}-", dir=_scratch_root()))
    try:
        output_path = run_dir / "video.mp4"
        timing_path = run_dir / "timing.json"
        logging_path = run_dir / "attention.jsonl"
        shim_dir = run_dir / "bin"
        shim_dir.mkdir()
        shim = shim_dir / "python"
        shim.write_text(
            _runner_source(
                root / spec.program,
                _model_path(spec.name),
                output_path,
                timing_path,
                logging_path,
                workload.seed,
                (root / f"examples/{workload.prompt_id}/prompt.txt").read_text(
                    encoding="utf-8"),
                pattern,
                spec,
            ),
            encoding="utf-8",
        )
        shim.chmod(0o755)
        env = os.environ.copy()
        env["PATH"] = str(shim_dir) + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = str(root)
        # Each GPU worker compiles independently; sharing Triton's writable cache
        # across four processes can corrupt a first-run artifact.
        env["TRITON_CACHE_DIR"] = str(run_dir / "triton-cache")
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["FRONTIER_GENERATION_PHASE_PATH"] = str(run_dir / "phases.json")
        proc = _run_generation_process(
            ["bash", str(spec.entrypoint)],
            cwd=root,
            env=env,
            timeout=GENERATION_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            raise RuntimeError(_generation_failure_message(proc))
        payload = json.loads(timing_path.read_text(encoding="utf-8"))
        latency = float(payload["seconds"])
        if not math.isfinite(latency) or latency <= 0:
            raise RuntimeError("video generation produced invalid timing")
        frames = _load_video(output_path)
        if keep_video is not None:
            shutil.copyfile(output_path, keep_video)
        return latency, frames
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def _load_video(path: Path):
    import imageio.v2 as imageio
    import numpy as np

    reader = imageio.get_reader(str(path))
    frames = []
    try:
        for frame in reader:
            frames.append(np.asarray(frame, dtype=np.uint8))
    finally:
        reader.close()
    if len(frames) != EXPECTED_FRAMES:
        raise RuntimeError("generated video has the wrong frame count")
    result = np.stack(frames)
    if result.shape != (EXPECTED_FRAMES, EXPECTED_HEIGHT, EXPECTED_WIDTH, 3):
        raise RuntimeError("generated video has the wrong decoded shape")
    return result


def _lpips_model():
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips

        _LPIPS_MODEL = lpips.LPIPS(net="vgg").cuda().eval()
    return _LPIPS_MODEL


def _fidelity(candidate_frames: Any, dense_frames: Any) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    if candidate_frames.shape != dense_frames.shape:
        raise RuntimeError("candidate and dense videos have different shapes")
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    model = _lpips_model()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    with torch.inference_mode():
        for index in range(candidate_frames.shape[0]):
            frame1 = torch.from_numpy(candidate_frames[index]).cuda().permute(2, 0, 1).unsqueeze(0).float() / 255.0
            frame2 = torch.from_numpy(dense_frames[index]).cuda().permute(2, 0, 1).unsqueeze(0).float() / 255.0
            mse = F.mse_loss(frame1, frame2, reduction="mean")
            psnr_values.append(float((10 * torch.log10(1.0 / mse)).item()))

            mu1 = F.avg_pool2d(frame1, kernel_size=11, stride=1, padding=5)
            mu2 = F.avg_pool2d(frame2, kernel_size=11, stride=1, padding=5)
            sigma1_sq = F.avg_pool2d(frame1 * frame1, 11, 1, 5) - mu1 ** 2
            sigma2_sq = F.avg_pool2d(frame2 * frame2, 11, 1, 5) - mu2 ** 2
            sigma12 = F.avg_pool2d(frame1 * frame2, 11, 1, 5) - mu1 * mu2
            ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
                (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
            )
            ssim_values.append(float(ssim_map.mean().item()))
            lpips_values.append(float(model(frame1, frame2).item()))
    values = {
        "psnr_db": sum(psnr_values) / len(psnr_values),
        "ssim": sum(ssim_values) / len(ssim_values),
        "lpips": sum(lpips_values) / len(lpips_values),
    }
    # An exactly matching decoded video has +inf PSNR under the artifact's
    # 10*log10(1/MSE) definition and is valid.  NaN, -inf, or non-finite
    # bounded perceptual metrics are not.
    if (
        math.isnan(values["psnr_db"])
        or values["psnr_db"] == -math.inf
        or not math.isfinite(values["ssim"])
        or not math.isfinite(values["lpips"])
    ):
        raise RuntimeError("fidelity evaluation produced an invalid value")
    return values


def _measure_tree(root: Path, workload: Workload, dense_frames: Any | None = None) -> tuple[dict[str, float], Any]:
    latency, frames = _run_generation(root, workload, "EAR")
    if dense_frames is None:
        return {"e2e_latency_seconds": latency}, frames
    values = _fidelity(frames, dense_frames)
    values["e2e_latency_seconds"] = latency
    return values, frames


def _operator_stage(stage: str, function, *args):
    try:
        return function(*args)
    except Exception as error:
        context = {"stage": stage, "exception_type": type(error).__name__}
        if isinstance(error, subprocess.TimeoutExpired):
            context["timeout_seconds"] = error.timeout
            if hasattr(error, "operator_generation"):
                context["generation"] = error.operator_generation
        print("FRONTIER_OPERATOR_STAGE=" + json.dumps(context, sort_keys=True), file=sys.stderr)
        raise


def _evaluate_paired(candidate_root: Path, workload: Workload) -> tuple[dict[str, float], dict[str, float], float]:
    dense_latency, dense_frames = _operator_stage("dense_target", _cached_target, workload, "dense")
    reference_latency, reference_frames = _operator_stage("reference_target", _cached_target, workload, "reference")
    # The timed model runs in a fresh child before LPIPS creates a CUDA context
    # or occupies memory on that GPU. Target creation uses the same ordering.
    candidate_latency, candidate_frames = _operator_stage("candidate_generation", _run_generation, candidate_root, workload, "EAR")
    candidate = _operator_stage("candidate_metrics", _fidelity, candidate_frames, dense_frames)
    candidate["e2e_latency_seconds"] = candidate_latency
    candidate["_operator_generation"] = _LAST_GENERATION_DIAGNOSTIC
    reference = _operator_stage("reference_metrics", _fidelity, reference_frames, dense_frames)
    reference["e2e_latency_seconds"] = reference_latency
    return candidate, reference, dense_latency


def _gpu_count() -> int:
    try:
        count = int(os.environ.get("FRONTIER_VERIFIER_GPU_COUNT", "0"))
    except ValueError as exc:
        raise RuntimeError("Verifier GPU count is invalid") from exc
    if count != EXPECTED_VERIFIER_GPUS:
        raise RuntimeError(
            f"this suite requires exactly {EXPECTED_VERIFIER_GPUS} Verifier GPUs")
    return count


def _worker_result(candidate_root: Path, workload: Workload, gpu_index: int) -> tuple[dict[str, float], dict[str, float], float]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    role = "graded" if "_graded_" in workload.workload_id else "feedback"
    index = _workloads(role == "graded").index(workload)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--workload-worker",
         str(candidate_root), role, str(index)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=WORKER_TIMEOUT_SECONDS,
    )
    if (
        proc.returncode != 0
        or not proc.stdout
        or len(proc.stdout) > MAX_WORKER_OUTPUT_BYTES
        or len(proc.stderr) > MAX_WORKER_OUTPUT_BYTES
    ):
        raise WorkloadWorkerFailure(workload.workload_id, proc)
    try:
        # LPIPS reports its fixed weight initialization on stdout. The worker's
        # bounded final line is the machine-readable record.
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        candidate = {name: float(payload["candidate"][name]) for name in SPEC_METRICS}
        candidate["_operator_generation"] = payload["candidate"].get("_operator_generation")
        reference = {name: float(payload["reference"][name]) for name in SPEC_METRICS}
        dense_latency = float(payload["dense_latency_seconds"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"GPU worker returned invalid output for {workload.workload_id}") from exc
    if not math.isfinite(dense_latency) or dense_latency <= 0:
        raise RuntimeError(f"GPU worker returned invalid output for {workload.workload_id}")
    return candidate, reference, dense_latency


def _evaluate_suite(
    candidate_root: Path,
    workloads: tuple[Workload, ...],
) -> list[tuple[dict[str, float], dict[str, float], float]]:
    if not workloads:
        raise RuntimeError("suite workload count is invalid")
    graded = all("_graded_" in workload.workload_id for workload in workloads)
    feedback = all("_feedback_" in workload.workload_id for workload in workloads)
    expected = GRADED_WORKLOADS if graded else FEEDBACK_WORKLOADS
    if not (graded or feedback) or len(workloads) != expected:
        raise RuntimeError("suite workload count is invalid")
    gpu_count = _gpu_count()
    # Fail closed before any build or GPU work when a suite target is missing
    # or still an explicit PENDING placeholder.
    _require_suite_targets(workloads)
    # The patch cannot touch svg/kernels. Build the immutable extension once,
    # then give each workload an isolated process and one visible H100.  The
    # twelve-workload graded suite runs as three explicit four-GPU waves (Wan
    # 0-3, Wan 4-7, HunyuanVideo 0-3) so no GPU is shared by concurrent workers.
    _build_extension(candidate_root)
    results = []
    for wave in _waves(workloads):
        with ThreadPoolExecutor(max_workers=gpu_count) as pool:
            futures = [
                pool.submit(_worker_result, candidate_root, workload, gpu_index)
                for gpu_index, workload in enumerate(wave)
            ]
            results.extend(future.result() for future in futures)
    return results


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score(metrics: dict[str, float], dense_latency_seconds: float) -> float:
    quality = (
        _clip01((float(metrics["psnr_db"]) - PSNR_FLOOR_DB) / PSNR_SPAN_DB)
        + _clip01((float(metrics["ssim"]) - SSIM_FLOOR) / SSIM_SPAN)
        + _clip01((LPIPS_CEILING - float(metrics["lpips"])) / LPIPS_CEILING)
    ) / 3.0
    speedup = float(dense_latency_seconds) / max(1e-12, float(metrics["e2e_latency_seconds"]))
    efficiency = _clip01((speedup - 1.0) / SPEEDUP_SPAN)
    # Exact dense attention earns full quality but only the minimum credit when
    # it provides no speedup. A high score requires both fidelity and efficiency.
    return 100.0 * quality * (MIN_QUALITY_CREDIT + EFFICIENCY_CREDIT * efficiency)


def _beat_decisions(
    candidate: dict[str, float], reference: dict[str, float]
) -> dict[str, bool]:
    return {
        "e2e_latency_seconds": bool(
            candidate["e2e_latency_seconds"]
            <= reference["e2e_latency_seconds"]
            * (1.0 - MIN_LATENCY_SPEEDUP_FRACTION)
        ),
        "psnr_db": bool(
            candidate["psnr_db"]
            >= reference["psnr_db"] - PSNR_NONINFERIOR_TOLERANCE_DB
        ),
        "ssim": bool(
            candidate["ssim"]
            >= reference["ssim"] - SSIM_NONINFERIOR_TOLERANCE
        ),
        "lpips": bool(
            candidate["lpips"]
            <= reference["lpips"] + LPIPS_NONINFERIOR_TOLERANCE
        ),
    }


def _complete_metrics(
    workloads: tuple[Workload, ...],
    results: list[tuple[dict[str, float], dict[str, float], float]],
    candidate_score: float,
    reference_score: float,
) -> dict[str, Any]:
    rows = []
    paired = []
    candidate_metrics = {name: [] for name in SPEC_METRICS}
    for workload, (candidate, reference, _dense_latency) in zip(
            workloads, results, strict=True):
        paired.append({
            "workload": workload.workload_id,
            "candidate": {name: ("positive_infinity" if name == "psnr_db" and candidate[name] == math.inf else float(candidate[name])) for name in SPEC_METRICS},
            "reference": {name: ("positive_infinity" if name == "psnr_db" and reference[name] == math.inf else float(reference[name])) for name in SPEC_METRICS},
            "dense_latency_seconds": float(_dense_latency),
        })
        decisions = _beat_decisions(candidate, reference)
        rows.append({"workload": workload.workload_id, "metrics": decisions})
        for name in SPEC_METRICS:
            candidate_metrics[name].append(float(candidate[name]))
    return {
        "beat_spec_workload_passes": rows,
        "beats_reference": int(all(value for row in rows for value in row["metrics"].values())),
        "_operator_generation_workload_diagnostics": [
            {"workload": workload.workload_id, "generation": candidate["_operator_generation"]}
            for workload, (candidate, _reference, _dense) in zip(workloads, results, strict=True)
            if candidate.get("_operator_generation") is not None
        ] or None,
        "_operator_paired_workload_metrics": paired,
        "reference_score": float(reference_score),
        "margin": float(candidate_score - reference_score),
        **{name: [("positive_infinity" if name == "psnr_db" and value == math.inf else value)
                  for value in sorted(values)]
           for name, values in candidate_metrics.items()},
    }


def prepare() -> dict[str, Any]:
    return {
        "workloads": GRADED_WORKLOADS,
        "feedback_workloads": FEEDBACK_WORKLOADS,
        "graded_workloads": GRADED_WORKLOADS,
        "parallel_gpu_workers": EXPECTED_VERIFIER_GPUS,
        "paired_reference": True,
        "prepared_suite_targets": True,
        "entrypoint": str(ENTRYPOINT),
        "entrypoints": {name: str(spec.entrypoint) for name, spec in MODELS.items()},
        "accelerator_required": "nvidia_gpu",
    }


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    applied = None
    final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
    workloads = _workloads(final)
    # Without every prepared target there is no valid verdict for any
    # submission. Raise instead of returning a scored zero.
    _require_suite_targets(workloads)
    try:
        # The judge-only exact upstream patch contains its original logging
        # `open`; solver submissions do not receive that patch or this exemption.
        is_reference = Path(solution_path).read_bytes() == REFERENCE_PATCH.read_bytes()
        policy_error = None if is_reference else _task_policy_error(solution_path)
        if policy_error:
            return 0.0, 0.0, policy_error, _empty_metrics()
        applied = _apply_patch(solution_path)
        if not applied.ok or applied.workdir is None:
            return 0.0, 0.0, applied.message or "submission patch was rejected", _empty_metrics()
        root = Path(applied.workdir)
        _check_public_contract(root)
        results = _evaluate_suite(root, workloads)

        candidate_scores = [
            _score(candidate, dense_latency)
            for candidate, _reference, dense_latency in results
        ]
        reference_scores = [
            _score(reference, dense_latency)
            for _candidate, reference, dense_latency in results
        ]
        candidate_score = max(
            0.0, min(100.0, sum(candidate_scores) / len(candidate_scores)))
        reference_score = max(
            0.0, min(100.0, sum(reference_scores) / len(reference_scores)))
        metrics = _complete_metrics(
            workloads, results, candidate_score, reference_score
        )
        wave_count = len(_waves(workloads))
        counts = {name: sum(1 for workload in workloads if workload.model == name) for name in MODELS}
        return (
            candidate_score,
            candidate_score,
            f"Scored {len(workloads)} 81-frame workloads "
            f"({counts['wan']} Wan, {counts['hunyuan']} HunyuanVideo) in "
            f"{wave_count} four-GPU wave(s): {candidate_score:.2f}/100.",
            metrics,
        )
    except SuiteTargetsUnavailable:
        raise
    except WorkloadWorkerFailure as exc:
        # The public result remains generic. The trusted adapter removes this
        # reserved record before score/beat-spec validation and retains it only
        # in operator evidence. Candidate code has no provider credentials.
        metrics = _empty_metrics()
        metrics["_operator_diagnostics"] = [exc.operator_diagnostic]
        return 0.0, 0.0, str(exc)[:180], metrics
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError, SyntaxError) as exc:
        message = str(exc).strip() or "evaluation failed"
        return 0.0, 0.0, message[:180], _empty_metrics()
    except Exception:
        return 0.0, 0.0, "evaluation failed", _empty_metrics()
    finally:
        if applied is not None and applied.workdir is not None:
            shutil.rmtree(Path(applied.workdir).parent, ignore_errors=True)
        gc.collect()


def _workload_worker_main() -> int:
    if len(sys.argv) != 5 or sys.argv[3] not in {"feedback", "graded"}:
        raise RuntimeError("GPU worker arguments are invalid")
    candidate_root = Path(sys.argv[2]).resolve()
    workloads = _workloads(sys.argv[3] == "graded")
    try:
        index = int(sys.argv[4])
        workload = workloads[index]
    except (ValueError, IndexError) as exc:
        raise RuntimeError("GPU worker index is invalid") from exc
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"[0-9]+", visible):
        raise RuntimeError("each workload worker requires exactly one visible GPU")
    candidate, reference, dense_latency = _evaluate_paired(candidate_root, workload)
    print(json.dumps({
        "candidate": candidate,
        "reference": reference,
        "dense_latency_seconds": dense_latency,
    }, sort_keys=True))
    return 0


def _main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--workload-worker":
        return _workload_worker_main()
    path = sys.argv[1] if len(sys.argv) > 1 else str(REFERENCE_PATCH)
    score, unbounded, message, metrics = evaluate(path)
    print(json.dumps({"score": score, "score_unbounded": unbounded, "message": message, "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
