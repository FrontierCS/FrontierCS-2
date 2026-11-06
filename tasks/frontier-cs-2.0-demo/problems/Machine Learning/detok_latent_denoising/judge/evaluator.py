"""Judge of detok_latent_denoising: latent-denoising tokenizer continuation, on one GPU (#757 scoring note).

Score: per workload, with dO, dE the ordinary / EMA quantized-PNG PSNR differences to the
reference (dB), LO, LE the LPIPS values and T the method-process wall times,
    g = [dO/1.25, dE/1.25, log2(LO_r/LO_c)/0.25, log2(LE_r/LE_c)/0.25]
    speed = 15 tanh(log2(T_r/T_c)/0.5); if speed > 0: speed *= clamp((0.096 + min(g)) / 0.08, 0, 1)
    score_w = 50 + 8.75 * sum(tanh(g)) + speed;
raw_score = mean_w(score_w) clipped to 0..100, score_unbounded unclipped; the reference scores 50.
A crash, a 1,500 s overrun (or a disk/output limit), an incomplete evaluation or a non-finite
measurement of the submission's method process is a scored zero naming it; a reference or asset
failure raises. Every method process (warm-up, candidate, reference) runs in the shared test
sandbox (isolate.run: no judge files, no FRONTIER_* environment, the GPU, a private
loopback for torch.distributed); its time is the sandboxed command's own wall time.
LPIPS ratios use a floor of 1e-6.

Metrics, per workload (train feedback: development_mean_<name>):
- ordinary_quantized_png_psnr_db, ema_quantized_png_psnr_db: mean PSNR of the ordinary / EMA
  reconstructions after PNG quantization; higher is better; pass when strictly above the reference.
- ordinary_quantized_png_lpips, ema_quantized_png_lpips: mean LPIPS (v0.1, VGG16); lower is
  better; pass when strictly below the reference.
- method_process_wall_seconds: complete method-process wall time; lower is better; passes when
  strictly below the reference.

beats_reference = 1 iff all five metrics pass on every workload.
The archived beat spec is docs/task-authoring/detok_latent_denoising/provenance/beat_spec.json.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import isolate  # noqa: E402  (/opt/frontier, on the import path by frontier.pth)

TASK_DIR = Path(__file__).resolve().parent
PRISTINE = TASK_DIR / "pristine"  # the starter repo; the shared verifier overlays the submission
AUTHENTIC_ENTRYPOINT = "main_reconstruction.py"
METHOD_RUNNER = TASK_DIR / "method_runner.py"
RESULT_TAR = "result.tar"
OFFLINE_MODEL_ROOT = Path(os.environ.get("FRONTIER_MODEL_ROOT", "/opt/frontier/models"))
PSNR_O = "ordinary_quantized_png_psnr_db"
PSNR_E = "ema_quantized_png_psnr_db"
LPIPS_O = "ordinary_quantized_png_lpips"
LPIPS_E = "ema_quantized_png_lpips"
TIME = "method_process_wall_seconds"
SPEC_METRICS = (PSNR_O, PSNR_E, LPIPS_O, LPIPS_E, TIME)
# The train feedback names (task.toml [feedback]), in SPEC_METRICS order.
VISIBLE = ("development_mean_ordinary_quantized_png_psnr_db", "development_mean_ema_quantized_png_psnr_db",
           "development_mean_ordinary_quantized_png_lpips", "development_mean_ema_quantized_png_lpips",
           "development_mean_method_process_wall_seconds")
HIGHER_IS_BETTER = {PSNR_O: True, PSNR_E: True, LPIPS_O: False, LPIPS_E: False, TIME: False}
REFERENCE_SCORE = 50.0
QUALITY_POINTS = 8.75        # per quality metric: 4 metrics x (+/-8.75) span the 70-point quality share
PSNR_SCALE_DB = 1.25
LPIPS_SCALE_LOG2 = 0.25
SPEED_POINTS = 15.0
SPEED_SCALE_DOUBLINGS = 0.5
# Positive speed credit fades once any quality metric is worse than the reference by more than
# GATE_START of its scale (0.02 dB PSNR, ~0.28 % LPIPS) and is gone GATE_WIDTH scales later.
GATE_START = 0.016
GATE_WIDTH = 0.08
LPIPS_FLOOR = 1e-6
METHOD_TIMEOUT_SECONDS = 1500
# The method process's private disk (tree copy, workload copy, the one-epoch checkpoint and the
# evaluation PNGs) and its output (stdout, stderr and the result tar, each).
METHOD_DISK_BYTES = 64 << 30
METHOD_OUTPUT_BYTES = 512 << 20
METHOD_PATH = "/usr/local/bin:/usr/bin:/bin"


def method_options() -> dict:
    """The isolate knobs of one method process (tests drop the userns-only loopback)."""
    return {"devices": ["nvidia"], "network": "loopback", "dev_shm": True,
            "read_paths": [str(OFFLINE_MODEL_ROOT)], "disk_bytes": METHOD_DISK_BYTES,
            "output_bytes": METHOD_OUTPUT_BYTES, "path": METHOD_PATH}


def workload_score(cand: dict[str, float], ref: dict[str, float]) -> float:
    """Score one workload against its paired reference run; the reference itself scores 50."""
    values = [cand[k] for k in SPEC_METRICS] + [ref[k] for k in SPEC_METRICS]
    if not all(math.isfinite(v) for v in values) or min(cand[TIME], ref[TIME], cand[LPIPS_O], cand[LPIPS_E],
                                                         ref[LPIPS_O], ref[LPIPS_E]) < 0 or min(cand[TIME], ref[TIME]) <= 0:
        raise RuntimeError("nonfinite measurement")
    # Normalised quality changes: positive is better, one unit = one metric scale.
    gains = [
        (cand[PSNR_O] - ref[PSNR_O]) / PSNR_SCALE_DB,
        (cand[PSNR_E] - ref[PSNR_E]) / PSNR_SCALE_DB,
        math.log2(max(ref[LPIPS_O], LPIPS_FLOOR) / max(cand[LPIPS_O], LPIPS_FLOOR)) / LPIPS_SCALE_LOG2,
        math.log2(max(ref[LPIPS_E], LPIPS_FLOOR) / max(cand[LPIPS_E], LPIPS_FLOOR)) / LPIPS_SCALE_LOG2,
    ]
    quality = QUALITY_POINTS * sum(math.tanh(g) for g in gains)
    speed = SPEED_POINTS * math.tanh(math.log2(ref[TIME] / cand[TIME]) / SPEED_SCALE_DOUBLINGS)
    if speed > 0:
        speed *= min(1.0, max(0.0, (GATE_START + GATE_WIDTH + min(gains)) / GATE_WIDTH))
    return REFERENCE_SCORE + quality + speed


def continuous_score(candidates: list[dict[str, float]], references: list[dict[str, float]]) -> float:
    if not candidates or len(candidates) != len(references):
        raise RuntimeError("incomplete paired measurements")
    return sum(workload_score(c, r) for c, r in zip(candidates, references)) / len(candidates)


_LPIPS_NET = None


def _lpips_net():
    """LPIPS v0.1 (VGG16 backbone) from the offline assets; trusted judge code, never candidate code."""
    global _LPIPS_NET
    if _LPIPS_NET is None:
        import torch
        import torchvision
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg = torchvision.models.vgg16()
        vgg.load_state_dict(torch.load(OFFLINE_MODEL_ROOT / "torch/hub/checkpoints/vgg16-397923af.pth",
                                       map_location="cpu", weights_only=True))
        features = vgg.features[:30].to(device).eval()
        lin = torch.load(OFFLINE_MODEL_ROOT / "lpips/vgg.pth", map_location="cpu", weights_only=True)
        weights = [lin[f"lin{i}.model.1.weight"].to(device) for i in range(5)]
        shift = torch.tensor([-0.030, -0.088, -0.188], device=device).view(1, 3, 1, 1)
        scale = torch.tensor([0.458, 0.448, 0.450], device=device).view(1, 3, 1, 1)
        _LPIPS_NET = (torch, device, features, weights, shift, scale)
    return _LPIPS_NET


def lpips_values(targets, reconstructions) -> list[float]:
    """Per-image LPIPS between uint8 HxWx3 targets and reconstructions."""
    torch, device, features, weights, shift, scale = _lpips_net()
    taps = (3, 8, 15, 22, 29)  # relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
    out = []
    with torch.no_grad():
        for i in range(0, len(targets), 16):
            pair = []
            for batch in (targets[i:i + 16], reconstructions[i:i + 16]):
                x = torch.stack([torch.from_numpy(a.copy()) for a in batch]).to(device)
                x = x.permute(0, 3, 1, 2).float() / 127.5 - 1.0
                h, feats = (x - shift) / scale, []
                for index, layer in enumerate(features):
                    h = layer(h)
                    if index in taps:
                        feats.append(h / (torch.sqrt(torch.sum(h * h, dim=1, keepdim=True)) + 1e-10))
                pair.append(feats)
            total = 0
            for fa, fb, w in zip(pair[0], pair[1], weights):
                total = total + torch.nn.functional.conv2d((fa - fb) ** 2, w).mean(dim=(1, 2, 3))
            out.extend(float(v) for v in total.cpu())
    return out


def _zero(reason: str):
    return 0.0, 0.0, reason[:240], {}


def _targets(images: Path, data_list: Path) -> list:
    """Center-cropped uint8 targets in list order, as the retained validation loader prepares them."""
    import numpy as np
    from PIL import Image
    targets = []
    for line in data_list.read_text().splitlines():
        if not line.strip():
            continue
        gt = Image.open(images / line.split()[0]).convert("RGB")
        while min(gt.size) >= 512:
            gt = gt.resize(tuple(x // 2 for x in gt.size), Image.Resampling.BOX)
        scale = 256 / min(gt.size)
        gt = gt.resize(tuple(round(x * scale) for x in gt.size), Image.Resampling.BICUBIC)
        a = np.asarray(gt, dtype=np.uint8)
        y, x = (a.shape[0] - 256) // 2, (a.shape[1] - 256) // 2
        targets.append(np.ascontiguousarray(a[y:y + 256, x:x + 256]))
    return targets


class CandidateFailure(Exception):
    """The submission's own method process failed: a scored zero whose message names it."""


def _unpack(data: bytes | None, output: Path) -> bool:
    """Extract the method's result tar (regular files under relative names only) into output."""
    if not data:
        return False
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if not member.isfile() or not parts or member.name.startswith("/") or ".." in parts:
                    return False
                target = output.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
    except (tarfile.TarError, OSError, ValueError):
        return False
    return True


def _run(tree: Path, workload: Path, output: Path, *, candidate: bool = False) -> dict[str, float]:
    """Run the retained entrypoint in an isolate instance on a copy of the tree and of this workload
    (lists, images, FID statistics), with the offline model assets readable in place; the
    evaluation PNGs come back as the instance's result tar, extracted into output.

    In the graded suite the EMA evaluation reconstructs the workload's second held-out set
    (``val_ema``) instead of repeating the ordinary evaluation's images."""
    ema_set = _final()
    if not shutil.which("nvidia-smi") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("the authoritative workload requires an NVIDIA GPU")
    required = ("train", "val", "train.txt", "val.txt", "fid_stats.npz", "seed.txt")
    for name in required + (("val_ema", "val_ema.txt") if ema_set else ()):
        if not (workload / name).exists():
            raise RuntimeError("private workload is incomplete")
    checkpoint = workload / "checkpoint.pth"
    if not checkpoint.exists():
        checkpoint = OFFLINE_MODEL_ROOT / "detok/detok-BB-gamm3.0-m0.7.pth"
    if not checkpoint.is_file():
        raise RuntimeError("offline tokenizer checkpoint is missing")
    for name in ("lpips/vgg.pth", "torch/hub/checkpoints/vgg16-397923af.pth",
                 "torch/hub/checkpoints/convnext_small-0c510722.pth",
                 "torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"):
        if not (OFFLINE_MODEL_ROOT / name).is_file():
            raise RuntimeError("offline auxiliary model asset is missing")
    lpips = tree / "work_dirs/ckpts/lpips/vgg.pth"
    lpips.parent.mkdir(parents=True, exist_ok=True)
    # Keep scratch destinations writable across workloads; offline assets are read-only.
    shutil.copyfile(OFFLINE_MODEL_ROOT / "lpips/vgg.pth", lpips)
    data_dir = tree / "data"
    data_dir.mkdir(exist_ok=True)
    shutil.copyfile(workload / "train.txt", data_dir / "train.txt")
    shutil.copyfile(workload / "val.txt", data_dir / "val.txt")
    seed = (workload / "seed.txt").read_text().strip()
    entrypoint = tree / AUTHENTIC_ENTRYPOINT
    if not entrypoint.is_file():
        raise RuntimeError("authentic reconstruction entrypoint is missing")
    data = "{in}/workload"
    if checkpoint.parent == workload:
        checkpoint_arg = f"{data}/{checkpoint.name}"
    else:
        checkpoint_arg = str(checkpoint)
    ema = ["--ema-set", f"{data}/val_ema", f"{data}/val_ema.txt"] if ema_set else []
    command = [
        os.path.realpath(sys.executable), "{in}/method_runner.py", "{out}/" + RESULT_TAR,
        "{work}/output", *ema, "{work}/" + AUTHENTIC_ENTRYPOINT,
        "--epochs", "1", "--batch_size", "4", "--eval_bsz", "4",
        "--model", "detok_BB", "--img_size", "256", "--patch_size", "16",
        "--token_channels", "16", "--gamma", "3", "--mask_ratio", "0.7",
        "--lr", "1e-6", "--seed", seed,
        "--discriminator_start_epoch", "100", "--num_workers", "4",
        "--class_of_interest", "0",
        "--num_images", "64", "--data_path", f"{data}/train",
        "--fid_stats_path", f"{data}/fid_stats.npz",
        "--load_from", checkpoint_arg, "--keep_eval_folder",
        "--output_dir", "{work}/output", "--save_freq", "1", "--vis_freq", "100",
        "--eval_freq", "100",
    ]
    env = {
        "TORCH_HOME": str(OFFLINE_MODEL_ROOT / "torch"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    }
    # From here on a failure is the submission's when this is its tree; the reference's is a
    # judge fault and raises. An IsolateError (the boundary could not be built) always raises.
    def fail(reason: str):
        raise (CandidateFailure if candidate else RuntimeError)(reason)

    proc = isolate.run(
        command, timeout_seconds=METHOD_TIMEOUT_SECONDS, tree=tree, cwd="{work}",
        inputs={"method_runner.py": METHOD_RUNNER, "workload": workload},
        outputs=[RESULT_TAR], env=env, **method_options())
    if proc.limit == "time":
        fail(f"method process exceeded its {METHOD_TIMEOUT_SECONDS}-second limit")
    if proc.limit:
        fail(f"method process exceeded its {proc.limit} limit")
    elapsed = proc.command_seconds
    if proc.returncode:
        fail("method process failed")
    log = (proc.stdout + proc.stderr).decode("utf-8", "replace")
    if len(re.findall(r"Average PSNR \(all ranks\):", log)) != 2:
        fail("ordinary/EMA evaluation did not complete")
    output.mkdir(parents=True, exist_ok=True)
    if not _unpack(proc.outputs.get(RESULT_TAR), output):
        fail("reconstruction output is missing or unreadable")
    summaries = list(output.rglob("eval_summary.txt"))
    if len(summaries) != 1:
        fail("evaluation summary is missing")
    rows = re.findall(r"use_ema=(True|False).*?psnr=([0-9.eE+-]+)", summaries[0].read_text())
    if len(rows) != 2:
        fail("evaluation metrics are incomplete")
    # Recompute the authoritative metric in the parent from PNG bytes. This is
    # intentionally independent of candidate log output.
    import numpy as np
    from PIL import Image
    eval_dirs = {"ordinary": list(output.rglob("epoch_001_use_ema=False")),
                 "ema": list(output.rglob("epoch_001_use_ema=True"))}
    ordinary_targets = _targets(workload / "val", workload / "val.txt")
    target_sets = {"ordinary": ordinary_targets,
                   "ema": _targets(workload / "val_ema", workload / "val_ema.txt") if ema_set else ordinary_targets}
    q, lp = {}, {}
    for kind, matches in eval_dirs.items():
        targets = target_sets[kind]
        if len(matches) != 1:
            fail("reconstruction PNG directory is missing")
        recons, values = [], []
        for idx, target in enumerate(targets):
            try:
                b = np.asarray(Image.open(matches[0] / f"{idx:06d}.png").convert("RGB"), dtype=np.uint8)
            except OSError:  # missing or unreadable PNG (FileNotFoundError, PIL decode errors)
                fail("reconstruction PNG is missing or unreadable")
            if b.shape != target.shape:
                fail("reconstruction PNG has the wrong shape")
            recons.append(b)
            mse = float(np.mean((target.astype(np.float64) - b.astype(np.float64)) ** 2))
            values.append(float("inf") if mse == 0 else 10.0 * math.log10(255.0 * 255.0 / mse))
        q[kind] = sum(values) / len(values)
        per_image = lpips_values(targets, recons)
        lp[kind] = sum(per_image) / len(per_image)
    result = {
        PSNR_O: float(q["ordinary"]),
        PSNR_E: float(q["ema"]),
        LPIPS_O: float(lp["ordinary"]),
        LPIPS_E: float(lp["ema"]),
        TIME: elapsed,
    }
    if not all(math.isfinite(v) for v in result.values()):
        fail("measurement is not finite")
    return result


def _suite_root() -> Path:
    root = os.environ.get("FRONTIER_DETOK_PRIVATE_ROOT", str(TASK_DIR / "judge_data/private"))
    return Path(root) / ("final" if _final() else "development")


def _final() -> bool:
    return os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"


def evaluate(tree: str):
    """Shared-verifier contract: `tree` is the pristine starter with only the submission's
    allowed files overlaid (the edit scope has already passed)."""
    tree = Path(tree)
    suite = _suite_root()
    workloads = sorted(p for p in suite.iterdir() if p.is_dir())
    if len(workloads) != 3:
        raise RuntimeError("the selected suite must contain three workloads")
    # One untimed warm-up before any timed process (issue #315): the first
    # process of the evaluation pays the cold cost of the first checkpoint
    # read and GPU initialisation, which previously fell only on the
    # candidate at workload-0. Run the reference on the first workload and
    # discard its result so neither timed side on workload-0 runs cold.
    with tempfile.TemporaryDirectory(dir=str(tree.parent)) as td:
        warmup_tree = Path(td) / "repo"
        shutil.copytree(PRISTINE, warmup_tree)
        subprocess.run(["git", "apply", str(TASK_DIR / "reference.patch")], cwd=warmup_tree, check=True)
        _run(warmup_tree, workloads[0], warmup_tree / "warmup-output")
    candidates, references = [], []
    for i, workload in enumerate(workloads):
        # Release the original multi-GB checkpoint outputs before the next process.
        with tempfile.TemporaryDirectory(prefix=f"candidate-output-{i}-", dir=str(tree.parent)) as td:
            try:
                candidates.append(_run(tree, workload, Path(td), candidate=True))
            except CandidateFailure as failure:
                return 0.0, 0.0, f"the submission's {failure}"[:240], {}
        with tempfile.TemporaryDirectory(dir=str(tree.parent)) as td:
            ref_tree = Path(td) / "repo"
            shutil.copytree(PRISTINE, ref_tree)
            subprocess.run(["git", "apply", str(TASK_DIR / "reference.patch")], cwd=ref_tree, check=True)
            references.append(_run(ref_tree, workload, ref_tree / "reference-output"))
    passes = []
    for i, (cand, ref) in enumerate(zip(candidates, references)):
        decisions = {k: (cand[k] > ref[k]) if HIGHER_IS_BETTER[k] else (cand[k] < ref[k]) for k in SPEC_METRICS}
        passes.append({"workload": f"workload-{i}", "metrics": decisions})
    beats = int(all(v for row in passes for v in row["metrics"].values()))
    score = continuous_score(candidates, references)
    metrics = {
        "beat_spec_workload_passes": passes,
        "beats_reference": beats,
        "reference_score": REFERENCE_SCORE,
        "margin": score - REFERENCE_SCORE,
        **{name: sum(x[k] for x in candidates) / len(candidates) for name, k in zip(VISIBLE, SPEC_METRICS)},
        # Operator evidence only (never solver-visible): paired per-workload measurements.
        "per_workload_measurements": [
            {"workload": f"workload-{i}", "candidate": dict(c), "reference": dict(r),
             "score": workload_score(c, r)}
            for i, (c, r) in enumerate(zip(candidates, references))
        ],
    }
    return min(100.0, max(0.0, score)), score, "evaluation completed", metrics


if __name__ == "__main__":
    import sys
    print(json.dumps(evaluate(sys.argv[1] if len(sys.argv) > 1 else ""), default=str))
