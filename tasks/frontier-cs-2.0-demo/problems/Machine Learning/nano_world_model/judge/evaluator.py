"""GPU evaluator for action-conditioned Point Maze future-video sampling."""

from __future__ import annotations

import json
import ast
import math
import os
import statistics
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

_EVALUATOR_DIR = Path(__file__).resolve().parent
if str(_EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALUATOR_DIR))

from submission_patch import apply_submission_patch
from judge_runtime.execution_audit import validate as validate_execution_audit


TASK_DIR = _EVALUATOR_DIR
CONFIG = yaml.safe_load((TASK_DIR / "config.yaml").read_text(encoding="utf-8"))
REFERENCE_PATCH = TASK_DIR / "reference.patch"
MODEL_REVISION = "b84303b7151cbd1836841c3a87bf095e1015b150"
VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
# Beat spec v3: the clip count comes from each suite manifest (dev 4, final 6).
# Every clip enters both the strict per-clip comparison and the continuous score
# (the manifests' strict_only flag is historical and no longer used).
# Release refresh 2026-10-11: 825 s per graded clip as before (1650 s covered two clips).
RUN_TIMEOUT_SECONDS = 4950
# The immutable entrypoint declared by ARTIFACT_PLAN.json/artifact_manifest.json.
AUTHENTIC_ENTRYPOINT = "src/main.py"
_REFERENCE_CACHE: dict[str, list[dict[str, float]]] = {}

# Continuous score: reference-relative, 50 for the trusted reference measured
# in the same verification, bounded to (0, 100).  Each metric is first averaged
# over every clip of the suite; the strict per-clip comparison is separate.
REFERENCE_SCORE = 50.0
PSNR_SCALE_DB = 2.5        # mean PSNR difference, dB
SSIM_SCALE_LOG2 = 0.3      # log2 ratio of mean (1 - SSIM), reference / candidate
LPIPS_SCALE_LOG2 = 0.4     # log2 ratio of mean LPIPS, reference / candidate
GATE_FULL = -0.01          # gains count fully while every normalised gain >= this
GATE_ZERO = -0.06          # ... and not at all once any normalised gain <= this
_ERROR_FLOOR = 1e-6        # keeps the log ratios finite for perfect predictions


class CandidateExecutionError(Exception):
    """A candidate failed after the trusted reference completed successfully."""


def _private_process_tail(stdout: str | bytes | None, stderr: str | bytes | None) -> None:
    parts = [part.decode("utf-8", errors="replace") if isinstance(part, bytes)
             else (part or "") for part in (stdout, stderr)]
    tail = "\n".join(parts).encode("utf-8", errors="replace")[-8192:]
    print("nanowm private subprocess tail:\n" + tail.decode("utf-8", errors="ignore"),
          file=sys.stderr, flush=True)


def _apply(path: str):
    return apply_submission_patch(path, TASK_DIR, CONFIG)

def _policy_check(repo: Path) -> None:
    forbidden_text = (
        "judge_data", "frontier_submission_role", "frontier_model_root",
        "validation_fixed_subset", "torch.load", "numpy.load", "np.load",
        "getenv", "environ",
    )
    allowed_import_roots = {
        "diffusion", "collections", "dataclasses", "enum", "functools", "itertools",
        "math", "numpy", "operator", "torch", "tqdm", "typing",
    }
    forbidden_names = {
        "__builtins__", "__import__", "builtins", "compile", "eval", "exec",
        "getattr", "globals", "input", "locals", "open", "setattr", "vars",
    }
    forbidden_attributes = {
        "_getframe", "ag_frame", "cr_frame", "f_back", "f_code", "f_globals",
        "f_locals", "fromfile", "genfromtxt", "gi_frame", "load", "loadtxt",
        "memmap", "save", "tb_frame", "with_traceback",
    }
    paths = [
        repo / "src/diffusion/df_sample.py",
        repo / "src/diffusion/gaussian_diffusion.py",
        repo / "src/diffusion/flow_matching.py",
    ]
    extra = repo / "src/diffusion/frontier_solution"
    if extra.exists():
        paths.extend(p for p in extra.rglob("*") if p.is_file())
    for path in paths:
        if path.is_symlink():
            raise ValueError("solver source must not be a symbolic link")
        if not path.is_file():
            raise ValueError(f"required sampler source is missing: {path.name}")
        if path.suffix not in (".py", ".pyi"):
            raise ValueError("solver extensions must be Python source")
        module_parts = path.relative_to(repo / "src").with_suffix("").parts
        if not all(part.isidentifier() for part in module_parts):
            raise ValueError("solver source paths must use valid Python module names")
        source = path.read_text(encoding="utf-8")
        text = source.lower()
        hit = next((token for token in forbidden_text if token in text), None)
        if hit:
            raise ValueError(f"forbidden data or runtime access token: {hit}")
        try:
            tree = ast.parse(source, filename=str(path))
            compile(tree, str(path), "exec")
        except SyntaxError as exc:
            raise ValueError(f"invalid Python source in {path.name}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
                denied = sorted(roots - allowed_import_roots)
                if denied:
                    raise ValueError(f"forbidden import in sampler source: {denied[0]}")
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    root = (node.module or "").split(".", 1)[0]
                    if root not in allowed_import_roots:
                        raise ValueError(f"forbidden import in sampler source: {root}")
            elif isinstance(node, ast.Name) and node.id in forbidden_names:
                raise ValueError(f"forbidden process-state name: {node.id}")
            elif isinstance(node, ast.Attribute):
                attr = node.attr
                if attr in forbidden_attributes or (attr.startswith("__") and attr.endswith("__")):
                    raise ValueError(f"forbidden frame or runtime attribute: {attr}")


def _suite_slices(suite: str) -> list[dict[str, Any]]:
    slices = json.loads((TASK_DIR / "judge_data" / f"{suite}_subset.json").read_text(encoding="utf-8"))["slices"]
    if not slices:
        raise RuntimeError("suite manifest has no clips")
    return slices


def _expected_workloads(suite: str) -> int:
    return len(_suite_slices(suite))


def _suite_name() -> str:
    return "final" if os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final" else "dev"


def _checkpoint_path() -> Path:
    cache_root = Path(os.environ.get("FRONTIER_EVAL_CACHE", "/workspace/.nanowm-eval-cache"))
    cache_root.mkdir(parents=True, exist_ok=True)
    converted = cache_root / "point-maze-30k.pt"
    if converted.exists():
        return converted
    model_root = Path(os.environ.get("FRONTIER_MODEL_ROOT", "/opt/frontier/models"))
    source = model_root / "point-maze" / "model.safetensors"
    if not source.is_file():
        raise FileNotFoundError("pinned Point Maze checkpoint is unavailable")
    from safetensors.torch import load_file
    import torch

    state = load_file(str(source), device="cpu")
    torch.save(state, converted)
    return converted


def _run_entrypoint(repo: Path, suite: str, *, isolated: bool = False) -> list[dict[str, float]]:
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("an NVIDIA GPU is required")
    model_root = Path(os.environ.get("FRONTIER_MODEL_ROOT", "/opt/frontier/models"))
    vae = model_root / "sd-vae"
    if not (vae / "config.json").is_file():
        raise FileNotFoundError("pinned SD-VAE is unavailable")
    subset = TASK_DIR / "judge_data" / f"{suite}_subset.json"
    data_root = TASK_DIR / "judge_data"
    checkpoint = _checkpoint_path()
    run_root = Path(tempfile.mkdtemp(prefix=f"nanowm-{suite}-"))
    command = [
        sys.executable, str(repo / AUTHENTIC_ENTRYPOINT),
        "experiment=evaluate_only",
        "dataset=dino_wm/point_maze",
        "model=nanowm_b2",
        f"experiment.resume_from_checkpoint={checkpoint}",
        f"dataset.loader.validation_fixed_subset_path={subset}",
        "experiment.evaluation.batch_size=1",
        "experiment.evaluation.save_videos=true",
        "experiment.infra.num_workers=0",
        "experiment.infra.compile=false",
        "experiment.infra.mixed_precision=false",
        "wandb.enabled=false",
        "logger.name=none",
        f"hydra.run.dir={run_root}",
    ]
    env = os.environ.copy()
    # Do not inherit bridge settings from an enclosing evaluation process.
    for name in list(env):
        if name.startswith("NANOWM_"):
            del env[name]
    env.update({
        "DATASET_DIR": str(data_root),
        "VAE_MODEL_PATH": str(vae),
        "RESULTS_DIR": str(run_root),
        "PYTHONPATH": os.pathsep.join((
            str(TASK_DIR / "judge_runtime"), str(repo / "frontier_runtime"), str(repo / "src")
        )),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONHASHSEED": "3407",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    trusted = None
    bridge_audit = run_root / "bridge-audit.jsonl"
    try:
        if isolated:
            trusted = _apply(str(REFERENCE_PATCH))
            if not trusted.ok:
                raise RuntimeError("trusted native caller could not be restored")
            command[1] = str(Path(trusted.workdir) / AUTHENTIC_ENTRYPOINT)
            env["PYTHONPATH"] = os.pathsep.join((
                str(TASK_DIR / "judge_runtime"),
                str(Path(trusted.workdir) / "frontier_runtime"),
                str(Path(trusted.workdir) / "src")
            ))
            env.update({"NANOWM_CANDIDATE_REPO": str(repo),
                        "NANOWM_PUBLIC_REPO": str(TASK_DIR / "harbor/app"),
                        "NANOWM_BRIDGE_AUDIT": str(bridge_audit),
                        "NANOWM_CALLER_DEADLINE": str(time.monotonic() + RUN_TIMEOUT_SECONDS - 10),
                        "FRONTIER_MODEL_ROOT": str(model_root)})
        execution_audit = run_root / "native-execution.json"
        env["NANOWM_ENTRYPOINT"] = command[1]
        env["NANOWM_EXECUTION_AUDIT"] = str(execution_audit)
        try:
            completed = subprocess.run(
                command, cwd=Path(trusted.workdir) if trusted else repo,
                env=env, capture_output=True, text=True,
                timeout=RUN_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            _private_process_tail(exc.stdout, exc.stderr)
            if isolated:
                raise RuntimeError("trusted isolated native caller exceeded its time limit") from exc
            raise CandidateExecutionError("candidate evaluation exceeded its time limit") from exc
        events = []
        if isolated and bridge_audit.exists():
            events = [json.loads(line) for line in bridge_audit.read_text().splitlines()]
        if any(event["event"] == "infrastructure_failure" for event in events):
            _private_process_tail(completed.stdout, completed.stderr)
            raise RuntimeError("isolated candidate runtime is unavailable")
        if isolated and not any(event["event"] == "installed" for event in events):
            _private_process_tail(completed.stdout, completed.stderr)
            raise RuntimeError("trusted sampling isolation hook did not install")
        if completed.returncode != 0:
            _private_process_tail(completed.stdout, completed.stderr)
            if isolated and not any(event["event"] == "candidate_failure" for event in events):
                raise RuntimeError("trusted native caller or writer failed")
            raise CandidateExecutionError("candidate evaluation subprocess failed")
        if isolated and sum(event["event"] == "call_complete" for event in events) != _expected_workloads(suite):
            raise RuntimeError("trusted native caller did not complete isolated sampling")
        # Saved arrays alone are insufficient: require the retained entrypoint,
        # native data loader, checkpoint load, VAE and sampling integration.
        validate_execution_audit(execution_audit, isolated=isolated)
        rows = _measure_saved_predictions(run_root, subset)
        if len(rows) != _expected_workloads(suite):
            raise CandidateExecutionError("candidate did not produce every expected clip")
        return rows
    finally:
        shutil.rmtree(run_root, ignore_errors=True)
        if trusted is not None and getattr(trusted, "workdir", None):
            shutil.rmtree(Path(trusted.workdir).parent, ignore_errors=True)


def _measure_saved_predictions(run_root: Path, subset_path: Path) -> list[dict[str, float]]:
    import lpips
    import piqa
    import torch
    import numpy as np

    device = torch.device("cuda")
    psnr_metric = piqa.PSNR(epsilon=1e-8, value_range=1.0, reduction="none").to(device).eval()
    ssim_metric = piqa.SSIM(window_size=11, sigma=1.5, n_channels=3, reduction="none").to(device).eval()
    lpips_metric = lpips.LPIPS(net="vgg").to(device).eval()
    specs = json.loads(subset_path.read_text(encoding="utf-8"))["slices"]
    rows: list[dict[str, float]] = []
    for spec in specs:
        tag = f"traj_{spec['traj_idx']:04d}_start_{spec['start_frame']:04d}"
        matches = list(run_root.rglob(f"{tag}/raw_data.npz"))
        if len(matches) != 1:
            raise CandidateExecutionError("candidate prediction output is incomplete")
        with np.load(matches[0]) as payload:
            pred = torch.from_numpy(payload["pred"][:, 1:]).permute(1, 0, 2, 3).to(device)
            gt = torch.from_numpy(payload["gt"][:, 1:]).permute(1, 0, 2, 3).to(device)
        with torch.no_grad():
            psnr = psnr_metric(pred, gt).mean().item()
            ssim = ssim_metric(pred, gt).mean().item()
            lpips_value = lpips_metric(pred * 2 - 1, gt * 2 - 1).mean().item()
        rows.append({"psnr": psnr, "ssim": ssim, "lpips": lpips_value})
    return rows


def _suite_means(rows: list[dict[str, float]]) -> dict[str, float]:
    return {name: statistics.fmean(row[name] for row in rows) for name in ("psnr", "ssim", "lpips")}


def metric_gains(candidate: list[dict[str, float]], reference: list[dict[str, float]]) -> tuple[float, float, float]:
    """Normalised PSNR, SSIM and LPIPS gains of the suite means over the reference (0 = equal)."""
    if not candidate or len(candidate) != len(reference):
        raise ValueError("candidate and reference must cover the same clips")
    c, r = _suite_means(candidate), _suite_means(reference)
    return (
        (c["psnr"] - r["psnr"]) / PSNR_SCALE_DB,
        math.log2(max(1.0 - r["ssim"], _ERROR_FLOOR) / max(1.0 - c["ssim"], _ERROR_FLOOR)) / SSIM_SCALE_LOG2,
        math.log2(max(r["lpips"], _ERROR_FLOOR) / max(c["lpips"], _ERROR_FLOOR)) / LPIPS_SCALE_LOG2,
    )


def continuous_score(candidate: list[dict[str, float]], reference: list[dict[str, float]]) -> float:
    """Reference-relative quality score; the reference scores exactly 50."""
    gains = metric_gains(candidate, reference)
    # A metric that is worse than the reference withdraws the credit for the others.
    gate = min(1.0, max(0.0, (min(gains) - GATE_ZERO) / (GATE_FULL - GATE_ZERO)))
    terms = [math.tanh(value) for value in gains]
    return REFERENCE_SCORE + (50.0 / 3.0) * sum(gate * t if t > 0 else t for t in terms)


def _reference_rows(suite: str) -> list[dict[str, float]]:
    if suite in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[suite]
    result = _apply(str(REFERENCE_PATCH))
    if not result.ok:
        raise RuntimeError("hidden reference patch could not be applied")
    try:
        try:
            rows = _run_entrypoint(Path(result.workdir), suite)
            _validate_rows(rows, suite)
        except Exception as exc:
            raise RuntimeError("trusted reference evaluation failed") from exc
        _REFERENCE_CACHE[suite] = rows
        return rows
    finally:
        shutil.rmtree(Path(result.workdir).parent, ignore_errors=True)


def _validate_rows(rows: list[dict[str, float]], suite: str) -> None:
    if len(rows) != _expected_workloads(suite):
        raise CandidateExecutionError("workload result count mismatch")
    for row in rows:
        if set(row) != {"psnr", "ssim", "lpips"} or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in row.values()
        ):
            raise CandidateExecutionError("invalid or non-finite workload metrics")


def prepare() -> dict[str, Any]:
    suite = _suite_name()
    _reference_rows(suite)
    return {"ready": True, "suite": suite, "workloads": _expected_workloads(suite)}


def _rejected(message: str) -> tuple[float, float, str, dict[str, Any]]:
    # A rejected candidate has no workload measurements to aggregate.
    return 0.0, 0.0, message, {"candidate_rejection": True, "beats_reference": 0}


def evaluate(solution_path: str) -> tuple[float, float, str, dict[str, Any]]:
    result = None
    try:
        if not isinstance(solution_path, (str, os.PathLike)):
            return _rejected("invalid submission path")
        if "\x00" in os.fspath(solution_path):
            return _rejected("invalid submission path")
        suite = _suite_name()
        if not (TASK_DIR / "harbor/app" / AUTHENTIC_ENTRYPOINT).is_file():
            raise RuntimeError("trusted task substrate is unavailable")
        result = _apply(solution_path)
        if not result.ok:
            if result.candidate_rejection is True:
                return _rejected(result.message)
            raise RuntimeError("submission preparation failed") from RuntimeError(result.message)
        repo = Path(result.workdir)
        try:
            _policy_check(repo)
        except ValueError as exc:
            return _rejected(str(exc))
        # Establish the trusted runtime before classifying candidate execution failures.
        reference = _reference_rows(suite)
        try:
            candidate = _run_entrypoint(repo, suite, isolated=True)
            _validate_rows(candidate, suite)
        except CandidateExecutionError:
            return _rejected("candidate evaluation failed")
        candidate_score = continuous_score(candidate, reference)
        ids = [f"clip-{i + 1:02d}" for i in range(_expected_workloads(suite))]
        passes = []
        for workload, cand, ref in zip(ids, candidate, reference):
            passes.append({
                "workload": workload,
                "metrics": {
                    "psnr": bool(cand["psnr"] > ref["psnr"]),
                    "ssim": bool(cand["ssim"] > ref["ssim"]),
                    "lpips": bool(cand["lpips"] < ref["lpips"]),
                },
            })
        beats_reference = int(all(v for row in passes for v in row["metrics"].values()))
        metrics: dict[str, Any] = {
            "psnr": sum(row["psnr"] for row in candidate) / len(candidate),
            "ssim": sum(row["ssim"] for row in candidate) / len(candidate),
            "lpips": sum(row["lpips"] for row in candidate) / len(candidate),
            "beat_spec_workload_passes": passes,
            "beats_reference": beats_reference,
            "reference_score": REFERENCE_SCORE,
            "margin": candidate_score - REFERENCE_SCORE,
            # Operator evidence only: not in beat_spec solver_feedback.
            "per_workload_measurements": [
                {"workload": workload, "candidate": dict(cand), "reference": dict(ref)}
                for workload, cand, ref in zip(ids, candidate, reference)
            ],
        }
        if not math.isfinite(candidate_score):
            raise RuntimeError("non-finite quality score")
        score = min(100.0, max(0.0, candidate_score))
        return score, candidate_score, f"evaluated {len(ids)} action-conditioned clips", metrics
    finally:
        if result is not None and getattr(result, "workdir", None):
            shutil.rmtree(Path(result.workdir).parent, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: evaluator.py SOLUTION_PATCH", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({"result": evaluate(sys.argv[1])}, indent=2))
