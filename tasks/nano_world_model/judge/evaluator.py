"""Judge of nano_world_model: action-conditioned Point Maze future-video sampling.

Runs on the CPU control image (output-scored with a deadline). The
shared verifier applies the patch with the task's edit-scope rules, runs check_tree (the sampler
source policy) and calls evaluate(tree): the starter with only the allowed files overlaid.

1. The candidate's machine, a [test_sandboxes.candidate] runner of the Agent GPU image, receives
   only the submitted tree, the runner side of this judge (native_run.py, judge_runtime/
   sitecustomize.py) and the suite's public data (public_inputs.py: actions, and per clip only
   the context frame; no scored frame, no state). It runs the device check and checkpoint
   conversion, then the tree's own src/main.py over all clips in one step of 4950 s, which
   counts from the runner's readiness (the control side enforces it); each dfot_sample result
   (the latent video of one clip) comes back.
2. A [test_sandboxes.trusted] runner of the Verifier GPU image (this judge baked in) runs the
   reference (judge/reference.patch) natively on the real judge data and decodes the candidate's
   latents through the same native pipeline (dfot_sample returns them in call order); both are
   measured there against the ground truth.

Scoring. Each suite clip (4 development, 6 final) is sampled once by the candidate and by the
trusted reference in the same verification. Per clip: PSNR, SSIM and LPIPS (VGG) over the three
future frames; the means P, S, L over all clips against the reference's P*, S*, L*:
  g_P = (P - P*)/2.5, g_S = log2((1-S*)/(1-S))/0.3, g_L = log2(L*/L)/0.4,
  G = clamp((min g + 0.06)/0.05, 0, 1), t = tanh(g),
  score_unbounded = 50 + 50/3 * (sum of G*t over positive t + sum of t over negative t);
  raw_score = clamp(score_unbounded, 0, 100). The reference scores 50.
Metrics (per clip, beat_spec_workload_passes):
  psnr   mean PSNR (dB) of the future frames, higher is better; passes if > reference.
  ssim   mean SSIM, higher is better; passes if > reference.
  lpips  mean LPIPS, lower is better; passes if < reference.
beats_reference = 1 iff every clip passes all three (no ties). A candidate whose sampling fails
(a crash, the deadline, a lost session or an output-cap overrun on its own machine, or a missing
or malformed latent) scores 0 and fails every clip once the reference run succeeded; a failure of
the device check, of the trusted runner or of the reference raises (an unscored verification).
"""

from __future__ import annotations

import ast
import json
import math
import os
import shutil
import statistics
import struct
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

_EVALUATOR_DIR = Path(__file__).resolve().parent
if str(_EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALUATOR_DIR))

from fcs2_verifier_gpu import SandboxResult, SandboxRun, SandboxStep, run_sandboxes, sandbox_workdir
from native_run import command, environment


TASK_DIR = _EVALUATOR_DIR
PUBLIC_DATA = TASK_DIR / "public_data"  # judge/public_inputs.py, run by the image build
# Both GPU images' interpreter; the trusted runner's judge is baked at /opt/frontier/task.
PYTHON = "/opt/frontier/runtime-python"
TRUSTED_NATIVE_RUN = "/opt/frontier/task/native_run.py"
# Beat spec v3: the clip count comes from each suite manifest (dev 4, final 6).
# Release refresh 2026-10-11: 825 s per graded clip (4950 s for all clips of a suite).
RUN_TIMEOUT_SECONDS = 4950
PREPARE_SECONDS = 300      # device check and checkpoint conversion, before submitted code runs
SAMPLE_SHAPE = (1, 4, 4, 32, 32)
_NPY_HEADER = {"descr": "<f4", "fortran_order": False, "shape": SAMPLE_SHAPE}
_NPY_MAGIC = b"\x93NUMPY\x01\x00"

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
    """The candidate's returned results are not valid workload metrics."""


class InfrastructureError(RuntimeError):
    """The fixed evaluation environment, the trusted runner or the reference could not run."""


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


def _log(result: SandboxResult, step: int) -> None:
    if result.logs[step]:
        print(result.logs[step].decode("utf-8", errors="replace"), file=sys.stderr)


def _writable(directory: Path) -> None:
    for path in (directory, *directory.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)


def _candidate_run(tree: Path, suite: str, files: Path) -> SandboxRun:
    """The candidate's machine: files/tree (the submitted tree), files/data (the suite's public
    data), files/runtime (native_run.py, sitecustomize.py); the device check and checkpoint
    conversion, then the tree's src/main.py with each dfot_sample result saved to out/latents."""
    root = PurePosixPath(sandbox_workdir("candidate"))
    shutil.copytree(tree, files / "tree", symlinks=True)
    shutil.copytree(PUBLIC_DATA / suite, files / "data")
    (files / "runtime").mkdir()
    shutil.copy2(TASK_DIR / "native_run.py", files / "runtime" / "native_run.py")
    shutil.copy2(TASK_DIR / "judge_runtime" / "sitecustomize.py", files / "runtime" / "sitecustomize.py")
    (files / "out" / "latents").mkdir(parents=True)
    _writable(files)
    repo, checkpoint = root / "tree", root / "cache" / "point-maze-30k.pt"
    sample = command(PYTHON, str(repo), str(root / "data" / "subset.json"), str(checkpoint),
                     str(root / "run"))
    return SandboxRun("candidate", files, [
        SandboxStep([PYTHON, str(root / "runtime" / "native_run.py"), "prepare", str(checkpoint)],
                    PREPARE_SECONDS),
        SandboxStep(["/bin/sh", "-c", f'cd {repo} && exec "$0" "$@"', *sample], RUN_TIMEOUT_SECONDS,
                    {**environment(str(root / "runtime"), str(repo), str(root / "data"),
                                   str(root / "run")),
                     "NANOWM_LATENTS_RECORD": str(root / "out" / "latents")}),
    ], ["out/latents"])


def _latent(path: Path) -> bool:
    """A float32 C-order .npy of SAMPLE_SHAPE with finite values (read without numpy)."""
    if not path.is_file():
        return False
    raw = path.read_bytes()
    if raw[:8] != _NPY_MAGIC or len(raw) < 10:
        return False
    size = int.from_bytes(raw[8:10], "little")
    try:
        fields = ast.literal_eval(raw[10:10 + size].decode("latin-1"))
    except (ValueError, SyntaxError):
        return False
    data = raw[10 + size:]
    return (fields == _NPY_HEADER and len(data) == math.prod(SAMPLE_SHAPE) * 4
            and all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", data)))


def _candidate_latents(result: SandboxResult, suite: str, destination: Path) -> str | None:
    """Copy the candidate's latents, one per clip, to `destination`; or the reason its sampling
    failed (a scored zero once the reference run succeeds)."""
    prepare = result.steps[0]
    if prepare.lost_session or prepare.timed_out or prepare.exit_code != 0:
        # Before any submitted code ran: the fixed environment.
        _log(result, 0)
        raise InfrastructureError("the candidate machine's device check or checkpoint conversion failed")
    step = result.steps[1]
    if step.lost_session:
        # Owner 2026-11-05: a lost session on the candidate's own machine is its scored zero.
        return "candidate evaluation lost its session"
    if result.outputs_exceeded:
        return "candidate evaluation returned more than the output cap"
    if step.timed_out:
        return f"candidate evaluation exceeded its {RUN_TIMEOUT_SECONDS}-second deadline"
    if step.exit_code != 0:
        _log(result, 1)
        return "candidate evaluation failed"
    returned = result.outputs / "out" / "latents"
    names = [f"{index:04d}.npy" for index in range(_expected_workloads(suite))]
    if (not returned.is_dir() or sorted(p.name for p in returned.iterdir()) != names
            or not all(_latent(returned / name) for name in names)):
        return "candidate did not return a valid latent video for every clip"
    shutil.copytree(returned, destination)
    return None


def _trusted_rows(suite: str, files: Path, scratch: Path, with_candidate: bool) -> dict:
    """The reference's rows and, `with_candidate`, the candidate's (decoded from files/latents),
    from the trusted runner."""
    root = PurePosixPath(sandbox_workdir("trusted"))
    argv = [PYTHON, TRUSTED_NATIVE_RUN, "score", suite, str(root / "scores.json")]
    if with_candidate:
        argv.append(str(root / "latents"))
    outputs = scratch / "trusted-outputs"
    outputs.mkdir()
    (result,) = run_sandboxes([SandboxRun("trusted", files, [SandboxStep(argv, None)],
                                          ["scores.json"])], into=outputs)
    step = result.steps[0]
    if step.lost_session or step.timed_out or step.exit_code != 0 or result.outputs is None:
        _log(result, 0)
        raise InfrastructureError("the trusted reference and decode run failed")
    return json.loads((result.outputs / "scores.json").read_text(encoding="utf-8"))


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


def _validate_rows(rows: list[dict[str, float]], suite: str) -> None:
    if len(rows) != _expected_workloads(suite):
        raise CandidateExecutionError("workload result count mismatch")
    for row in rows:
        if set(row) != {"psnr", "ssim", "lpips"} or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in row.values()
        ):
            raise CandidateExecutionError("invalid or non-finite workload metrics")


def _failed(suite: str, message: str) -> tuple[float, float, str, dict[str, Any]]:
    """A candidate whose run failed after the trusted reference completed: a scored zero
    that fails every workload of the suite (no measurements to report)."""
    ids = [f"clip-{i + 1:02d}" for i in range(_expected_workloads(suite))]
    return 0.0, 0.0, message, {
        "beat_spec_workload_passes": [
            {"workload": workload, "metrics": {"psnr": False, "ssim": False, "lpips": False}}
            for workload in ids
        ],
        "beats_reference": 0,
        "reference_score": REFERENCE_SCORE,
        "margin": -REFERENCE_SCORE,
    }


def check_tree(tree: str, pristine: str, changes: list) -> tuple[str, list[str]] | None:
    """Shared-verifier hook: the sampler source policy (allowed imports, no data, runtime or
    frame access) over the three sampler modules and every helper below
    src/diffusion/frontier_solution/, which must be importable Python modules."""
    root = Path(tree)
    changed = [c.path for c in changes if c.status != "D"] or [c.path for c in changes]
    extra = root / "src/diffusion/frontier_solution"
    if extra.exists():
        bad = [str(p.relative_to(root)) for p in extra.rglob("*") if p.is_file()
               and not all(part.isidentifier()
                           for part in p.relative_to(root / "src").with_suffix("").parts)]
        if bad:
            return "out_of_scope", bad
    try:
        _policy_check(root)
    except (ValueError, OSError, UnicodeDecodeError):
        return "content_policy", changed
    return None


def evaluate(tree: str) -> tuple[float, float, str, dict[str, Any]]:
    suite = _suite_name()
    scratch = Path(tempfile.mkdtemp(prefix="nanowm-"))
    try:
        (scratch / "candidate").mkdir()
        (scratch / "candidate-outputs").mkdir()
        (sampled,) = run_sandboxes([_candidate_run(Path(tree), suite, scratch / "candidate")],
                                   into=scratch / "candidate-outputs")
        trusted = scratch / "trusted"
        trusted.mkdir()
        failure = _candidate_latents(sampled, suite, trusted / "latents")
        rows = _trusted_rows(suite, trusted, scratch, with_candidate=failure is None)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    reference = rows["reference"]
    try:
        _validate_rows(reference, suite)
    except CandidateExecutionError as exc:
        raise InfrastructureError("trusted reference evaluation failed") from exc
    if failure is not None:
        return _failed(suite, failure)
    candidate = rows["candidate"]
    try:
        _validate_rows(candidate, suite)
    except CandidateExecutionError:
        return _failed(suite, "candidate evaluation failed")
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

