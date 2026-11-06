#!/usr/bin/env python3
"""

# The authoritative path runs the workload model's retained 720p shell entrypoint for the
# candidate. Dense and reference videos prepared once through the same path are hash-checked,
# decoded and scored as the paired target and reference. Unit tests replace only the GPU runner.Judge of sparse_video_attention_policy for the shared verifier (#757): evaluate(tree), check_tree.

Score: per workload, with T the candidate latency, Td the fixed pre-measured dense latency and
P, S, L the 81-frame means of PSNR, SSIM and VGG-LPIPS against the fixed dense video,
quality = (clip((P-15)/20) + clip((S-0.5)/0.5) + clip((0.35-L)/0.35)) / 3,
efficiency = clip(Td/max(1e-12, T) - 1), score = 100 * quality * (0.25 + 0.75 * efficiency),
clip to [0, 1]. raw_score = score_unbounded = the mean over the suite (4 train, 12 final
workloads, clamped to [0, 100]); any correctness failure on any workload scores 0.

Beat-spec metrics (beat_spec_workload_passes, one row per workload; the withheld SVG-EAR
reference is measured on the same workload in the same run):
- e2e_latency_seconds, lower is better: the synchronized 50-step pipeline call; passes when
  candidate <= 0.96 * reference (at least 4% lower).
- psnr_db, higher is better: mean frame PSNR 10*log10(1/MSE) on [0, 1] frames; passes when
  candidate >= reference - 0.25 dB.
- ssim, higher is better: mean frame SSIM (11x11 average pool, C1 = 0.01^2, C2 = 0.03^2);
  passes when candidate >= reference - 0.01.
- lpips, lower is better: mean frame VGG-LPIPS on [0, 1] frames; passes when
  candidate <= reference + 0.005.
beats_reference = 1 iff every one of the 12 final workloads (every workload of the evaluated
suite) passes all four metrics; equality at a bound passes; otherwise 0. It does not change the
score. The prior specification is archived at
docs/task-authoring/sparse_video_attention_policy/provenance/beat_spec.json.
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
import time
import traceback
from typing import Any

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
import isolate  # noqa: E402  (/opt/frontier, on the import path by frontier.pth)


TASK_DIR = _MODULE_DIR
PRISTINE = TASK_DIR / "pristine"  # the starter repo; the shared verifier overlays the submission
REFERENCE_PATCH = TASK_DIR / "reference.patch"
DENSE_TARGET_DIR = TASK_DIR / "dense_targets"
DENSE_MANIFEST_SCHEMA_VERSION = 2
TARGET_READY = "ready"
TARGET_PENDING = "PENDING"

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
# A workload worker whose generation sandbox could not be built exits with this code; the
# evaluation then fails as infrastructure (never a scored zero).
SANDBOX_UNAVAILABLE_EXIT = 75
# A workload worker whose candidate generation or candidate metrics failed exits with this code:
# the only worker failure scored as the candidate's (zero). Any other worker failure (an import
# error, a target or reference stage, invalid output) is the judge's and raises.
CANDIDATE_FAILED_EXIT = 65
# Limits of one generation's isolate instance: the MP4 + timing + phases and its stdout/stderr;
# its private disk (tree copy, Triton/flashinfer caches, HOME, TMPDIR).
GENERATION_OUTPUT_BYTES = 1 << 30
GENERATION_DISK_BYTES = 64 << 30
# Image runtime variables a generation keeps (it ran with the judge's whole environment
# before isolate); everything else, FRONTIER_* included, is not passed on.
GENERATION_ENV_PASSTHROUGH = (
    "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH", "CUDA_HOME", "CUDA_PATH", "CC", "CXX",
    "NVIDIA_DRIVER_CAPABILITIES", "TORCH_HOME", "HF_HOME", "HF_HUB_CACHE",
    "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
)
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
# the private runner or replace the class whose call boundary is timed: the timer runs in
# the generation's own process (after the warm-up, around the synchronized pipeline call),
# so isolate, which keeps judge files, environment and network out of reach,
# cannot protect it. These tokens guard that in-process boundary and its timing file.
FORBIDDEN_ADDED_PATTERNS = (
    re.compile(r"(?:^|\W)(?:inspect|subprocess|runpy|open)(?:\W|$)"),
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


class SandboxUnavailable(Exception):
    """The generation's isolate instance could not be built: an infrastructure failure.

    Not a RuntimeError, so evaluate() re-raises it instead of scoring a zero."""


class CandidateStageFailure(RuntimeError):
    """Inside a workload worker: the candidate's generation or its metrics failed."""


class WorkerFault(Exception):
    """A workload worker failed for a reason other than the candidate (the judge or its image).

    Not a RuntimeError, so evaluate() re-raises it instead of scoring a zero."""


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


def _generation_diagnostic(phase_bytes, elapsed, stderr, capture_complete):
    """Operator-only record of one generation: its phase marks and the stderr tail."""
    phases = []
    try:
        if phase_bytes is not None and len(phase_bytes) <= 4096:
            records = json.loads(phase_bytes)
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
    except (ValueError, TypeError, RecursionError):
        pass
    if phases and phases[-1]["elapsed_seconds"] > elapsed:
        phases = []
    tail = stderr[-1024:]
    return {
        "subprocess_elapsed_seconds": elapsed,
        "stderr_observed_bytes": len(stderr),
        "stderr_observed_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_tail": tail.decode("utf-8", "replace"),
        "stderr_tail_truncated": len(stderr) > len(tail),
        "stderr_capture_complete": capture_complete,
        "phases": phases,
    }


def _generation_failure_message(result) -> str:
    """Return bounded detail that remains inside the outer worker traceback."""
    stderr = result.stderr or b""
    tail = stderr[-MAX_OPERATOR_DIAGNOSTIC_BYTES:].decode("utf-8", "replace")
    limit = f" (isolate {result.limit} limit)" if result.limit else ""
    return (
        f"Video generation stderr tail:\n{tail}\n"
        f"Video generation process exited with code {result.returncode}{limit}"
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


def _reference_tree() -> Path:
    """The pristine starter with judge/reference.patch applied (trusted, judge-owned)."""
    scratch = Path(tempfile.mkdtemp(prefix="svg-reference-", dir=_scratch_root()))
    tree = scratch / "reference"
    shutil.copytree(PRISTINE, tree)
    for path in (tree, *tree.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(scratch.resolve())}
    subprocess.run(["git", "apply", "--whitespace=nowarn", str(REFERENCE_PATCH)], cwd=tree,
                   check=True, capture_output=True, timeout=120, env=env)
    return tree


def check_tree(tree: str, pristine: str, changes: list) -> tuple[str, list[str]] | None:
    """Shared-verifier hook: no harness-access tokens in added source, and the public
    callables still exist. The judge's exact reference is exempt from the token screen (its
    upstream code keeps a logging `open`); a submission that is not byte-identical to it is not."""
    tree = Path(tree)
    reference = _reference_tree()
    try:
        is_reference = all(
            (tree / c.path).read_bytes() == (reference / c.path).read_bytes()
            if c.status != "D" else not (reference / c.path).exists()
            for c in changes)
    finally:
        shutil.rmtree(reference.parent, ignore_errors=True)
    if not is_reference:
        hits = [c.path for c in changes
                if any(p.search("\n".join(c.added)) for p in FORBIDDEN_ADDED_PATTERNS)]
        if hits:
            return "content_policy", hits
    try:
        _check_public_contract(tree)
    except (RuntimeError, SyntaxError, OSError):
        return "content_policy", [c.path for c in changes if c.status != "D"]
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


# The shim's own environment names: the instance's tree and output directory (isolate
# placeholders, expanded by the launcher). The shim removes them before the program starts.
SHIM_WORK_ENV = "SVG_SHIM_WORK"
SHIM_OUT_ENV = "SVG_SHIM_OUT"
GENERATION_OUTPUTS = ("video.mp4", "timing.json", "phases.json")


def _runner_source(
    model_path: Path,
    seed: int,
    prompt: str,
    pattern: str,
    spec: ModelSpec = MODELS["wan"],
) -> str:
    # Values are literals in a per-run shim inside the generation's isolate instance (the tree and
    # output directory come from its environment). Candidate code sees only the normal
    # upstream argv, while the timer path remains in this closure.
    return f'''#!/usr/bin/env python3
import json
import os
import random
import runpy
import sys
import time
WORK = os.environ.pop({SHIM_WORK_ENV!r})
OUT = os.environ.pop({SHIM_OUT_ENV!r})
PHASE_PATH = os.path.join(OUT, "phases.json")
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

PROGRAM = {spec.program!r}
TARGET = os.path.join(WORK, PROGRAM)
MODEL = {str(model_path)!r}
OUTPUT = os.path.join(OUT, "video.mp4")
TIMING = os.path.join(OUT, "timing.json")
LOGGING = os.path.join(os.environ["TMPDIR"], "attention.jsonl")
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


def _generation_env() -> dict[str, str]:
    """The generation's environment inside its isolate instance: the image's runtime variables
    only (the launcher adds PATH, HOME, TMPDIR, LANG, CUDA_VISIBLE_DEVICES; no FRONTIER_*)."""
    env = {name: os.environ[name] for name in GENERATION_ENV_PASSTHROUGH if name in os.environ}
    env.update({
        # The shim first: the retained entrypoint's `python` is the judge's timed runner.
        # (In env, not the launcher's `path`, so the {in} placeholder is expanded.)
        "PATH": "{in}/bin:" + os.environ.get("PATH", isolate.DEFAULT_PATH),
        "PYTHONPATH": "{work}",
        # Each generation compiles independently into its own private cache.
        "TRITON_CACHE_DIR": "{tmp}/triton-cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        SHIM_WORK_ENV: "{work}",
        SHIM_OUT_ENV: "{out}",
    })
    return env


def _run_generation(
    root: Path, workload: Workload, pattern: str, keep_video: Path | None = None
) -> tuple[float, Any]:
    """Run one fresh generation in its own isolate instance; optionally retain its MP4 for authoring.

    The instance gets a private copy of `root`, the shim, one GPU and the offline models;
    the judge, the targets, the scratch and every other run are unreadable to it. Its video,
    the shim's timing and phase marks come back as declared outputs."""
    global _LAST_GENERATION_DIAGNOSTIC
    spec = workload.spec
    model_path = _model_path(spec.name)
    run_dir = Path(tempfile.mkdtemp(prefix=f"{spec.name}-{pattern.lower()}-", dir=_scratch_root()))
    try:
        shim_dir = run_dir / "bin"
        shim_dir.mkdir()
        shim = shim_dir / "python"
        shim.write_text(
            _runner_source(
                model_path,
                workload.seed,
                (root / f"examples/{workload.prompt_id}/prompt.txt").read_text(
                    encoding="utf-8"),
                pattern,
                spec,
            ),
            encoding="utf-8",
        )
        shim.chmod(0o755)
        try:
            result = isolate.run(
                ["bash", str(spec.entrypoint)],
                timeout_seconds=GENERATION_TIMEOUT_SECONDS,
                tree=root,
                inputs={"bin": shim_dir},
                outputs=GENERATION_OUTPUTS,
                read_paths=[str(model_path.parent)],
                env=_generation_env(),
                devices=["nvidia"],
                dev_shm=True,
                output_bytes=GENERATION_OUTPUT_BYTES,
                disk_bytes=GENERATION_DISK_BYTES,
            )
        except isolate.IsolateError as error:
            raise SandboxUnavailable(str(error)) from error
        _LAST_GENERATION_DIAGNOSTIC = _generation_diagnostic(
            result.outputs.get("phases.json"), result.command_seconds, result.stderr,
            result.limit != "output")
        if result.timed_out:
            expired = subprocess.TimeoutExpired(
                ["bash", str(spec.entrypoint)], GENERATION_TIMEOUT_SECONDS,
                stderr=result.stderr[-1024:])
            expired.operator_generation = _LAST_GENERATION_DIAGNOSTIC
            raise expired
        if not result.ok:
            raise RuntimeError(_generation_failure_message(result))
        try:
            latency = float(json.loads(result.outputs.get("timing.json") or b"")["seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("video generation produced invalid timing") from exc
        if not math.isfinite(latency) or latency <= 0:
            raise RuntimeError("video generation produced invalid timing")
        video = result.outputs.get("video.mp4")
        if not video:
            raise RuntimeError("video generation produced no video")
        output_path = run_dir / "video.mp4"
        output_path.write_bytes(video)
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
    try:
        candidate_latency, candidate_frames = _operator_stage("candidate_generation", _run_generation, candidate_root, workload, "EAR")
        candidate = _operator_stage("candidate_metrics", _fidelity, candidate_frames, dense_frames)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        raise CandidateStageFailure(str(error)) from error
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
    if proc.returncode == SANDBOX_UNAVAILABLE_EXIT:
        raise SandboxUnavailable(f"generation isolate launcher unavailable for {workload.workload_id}: "
                                 + proc.stderr[-MAX_OPERATOR_DIAGNOSTIC_BYTES:].decode("utf-8", "replace"))
    if proc.returncode == CANDIDATE_FAILED_EXIT:
        raise WorkloadWorkerFailure(workload.workload_id, proc)
    if (
        proc.returncode != 0
        or not proc.stdout
        or len(proc.stdout) > MAX_WORKER_OUTPUT_BYTES
        or len(proc.stderr) > MAX_WORKER_OUTPUT_BYTES
    ):
        raise WorkerFault(
            f"GPU worker for {workload.workload_id} exited {proc.returncode}: "
            + proc.stderr[-MAX_OPERATOR_DIAGNOSTIC_BYTES:].decode("utf-8", "replace"))
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


def evaluate(tree: str) -> tuple[float, float, str, dict]:
    final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
    workloads = _workloads(final)
    # Without every prepared target there is no valid verdict for any
    # submission. Raise instead of returning a scored zero.
    _require_suite_targets(workloads)
    try:
        # The shared verifier has applied the patch and run check_tree (token screen and
        # public callables); `tree` is the starter with only the allowed files overlaid.
        root = Path(tree)
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
    except (SuiteTargetsUnavailable, SandboxUnavailable, WorkerFault):
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
        try:
            return _workload_worker_main()
        except SandboxUnavailable as error:
            print(f"isolate launcher unavailable: {error}", file=sys.stderr)
            return SANDBOX_UNAVAILABLE_EXIT
        except CandidateStageFailure:
            traceback.print_exc()
            return CANDIDATE_FAILED_EXIT
    path = sys.argv[1] if len(sys.argv) > 1 else str(REFERENCE_PATCH)
    score, unbounded, message, metrics = evaluate(path)
    print(json.dumps({"score": score, "score_unbounded": unbounded, "message": message, "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
