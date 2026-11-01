"""GPU evaluator for the stripped upstream reconstruction task."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

from submission_patch import apply_submission_patch

TASK_DIR = Path(__file__).resolve().parent
CONFIG = TASK_DIR / "config.yaml"
AUTHENTIC_ENTRYPOINT = "main_reconstruction.py"
OFFLINE_MODEL_ROOT = Path(os.environ.get("FRONTIER_MODEL_ROOT", "/opt/frontier/models"))
PSNR_O = "ordinary_quantized_png_psnr_db"
PSNR_E = "ema_quantized_png_psnr_db"
LPIPS_O = "ordinary_quantized_png_lpips"
LPIPS_E = "ema_quantized_png_lpips"
TIME = "method_process_wall_seconds"
SPEC_METRICS = (PSNR_O, PSNR_E, LPIPS_O, LPIPS_E, TIME)
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


def _run(tree: Path, workload: Path, output: Path) -> dict[str, float]:
    """Run the retained entrypoint; private data supplies lists, images and assets."""
    if not shutil.which("nvidia-smi") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("the authoritative workload requires an NVIDIA GPU")
    for name in ("train", "val", "train.txt", "val.txt", "fid_stats.npz", "seed.txt"):
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
    env = os.environ.copy()
    env.update({
        "TORCH_HOME": str(OFFLINE_MODEL_ROOT / "torch"),
        "FRONTIER_DETOK_DATA": str(workload),
        "FRONTIER_DETOK_FID_STATS": str(workload / "fid_stats.npz"),
        "FRONTIER_DETOK_CHECKPOINT": str(checkpoint),
        "FRONTIER_DETOK_OUTPUT": str(output),
        "FRONTIER_DETOK_SEED": (workload / "seed.txt").read_text().strip(),
    })
    started = time.monotonic()
    entrypoint = tree / AUTHENTIC_ENTRYPOINT
    if not entrypoint.is_file():
        raise RuntimeError("authentic reconstruction entrypoint is missing")
    command = [
        sys.executable, str(entrypoint),
        "--epochs", "1", "--batch_size", "4", "--eval_bsz", "4",
        "--model", "detok_BB", "--img_size", "256", "--patch_size", "16",
        "--token_channels", "16", "--gamma", "3", "--mask_ratio", "0.7",
        "--lr", "1e-6", "--seed", env["FRONTIER_DETOK_SEED"],
        "--discriminator_start_epoch", "100", "--num_workers", "4",
        "--class_of_interest", "0",
        "--num_images", "64", "--data_path", str(workload / "train"),
        "--fid_stats_path", str(workload / "fid_stats.npz"),
        "--load_from", str(checkpoint), "--keep_eval_folder",
        "--output_dir", str(output), "--save_freq", "1", "--vis_freq", "100",
        "--eval_freq", "100",
    ]
    proc = subprocess.run(
        command, cwd=tree, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1500,
    )
    elapsed = time.monotonic() - started
    if proc.returncode:
        raise RuntimeError("authentic reconstruction process failed")
    if len(re.findall(r"Average PSNR \(all ranks\):", proc.stdout)) != 2:
        raise RuntimeError("ordinary/EMA evaluation did not complete")
    summaries = list(output.rglob("eval_summary.txt"))
    if len(summaries) != 1:
        raise RuntimeError("evaluation summary is missing")
    rows = re.findall(r"use_ema=(True|False).*?psnr=([0-9.eE+-]+)", summaries[0].read_text())
    if len(rows) != 2:
        raise RuntimeError("evaluation metrics are incomplete")
    # Recompute the authoritative metric in the parent from PNG bytes. This is
    # intentionally independent of candidate log output.
    import numpy as np
    from PIL import Image
    names = [line.split()[0] for line in (workload / "val.txt").read_text().splitlines() if line.strip()]
    eval_dirs = {"ordinary": list(output.rglob("epoch_001_use_ema=False")),
                 "ema": list(output.rglob("epoch_001_use_ema=True"))}
    targets = []
    for name in names:
        gt = Image.open(workload / "val" / name).convert("RGB")
        while min(gt.size) >= 512:
            gt = gt.resize(tuple(x // 2 for x in gt.size), Image.Resampling.BOX)
        scale = 256 / min(gt.size)
        gt = gt.resize(tuple(round(x * scale) for x in gt.size), Image.Resampling.BICUBIC)
        a = np.asarray(gt, dtype=np.uint8)
        y, x = (a.shape[0] - 256) // 2, (a.shape[1] - 256) // 2
        targets.append(np.ascontiguousarray(a[y:y + 256, x:x + 256]))
    q, lp = {}, {}
    for kind, matches in eval_dirs.items():
        if len(matches) != 1:
            raise RuntimeError("reconstruction PNG directory is missing")
        recons, values = [], []
        for idx, target in enumerate(targets):
            b = np.asarray(Image.open(matches[0] / f"{idx:06d}.png").convert("RGB"), dtype=np.uint8)
            if b.shape != target.shape:
                raise RuntimeError("reconstruction PNG has the wrong shape")
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
        raise RuntimeError("nonfinite measurement")
    return result


def _suite_root() -> Path:
    final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
    root = os.environ.get("FRONTIER_DETOK_PRIVATE_ROOT", str(TASK_DIR / "judge_data/private"))
    return Path(root) / ("final" if final else "development")


def evaluate(solution_path: str):
    applied = None
    try:
        config = yaml.safe_load(CONFIG.read_text())
        if not isinstance(config, dict):
            raise RuntimeError("task config must be a mapping")
        applied = apply_submission_patch(solution_path, TASK_DIR, config)
        if not applied.ok:
            return _zero(applied.message)
        tree = Path(applied.workdir)
        suite = _suite_root()
        workloads = sorted(p for p in suite.iterdir() if p.is_dir())
        if len(workloads) != 3:
            raise RuntimeError("the selected suite must contain three workloads")
        # One untimed warm-up before any timed process: the first
        # process of the evaluation pays the cold cost of the first checkpoint
        # read and GPU initialisation, which previously fell only on the
        # candidate at workload-0. Run the reference on the first workload and
        # discard its result so neither timed side on workload-0 runs cold.
        with tempfile.TemporaryDirectory(dir=str(tree.parent)) as td:
            warmup_tree = Path(td) / "repo"
            shutil.copytree(TASK_DIR / "harbor" / "app", warmup_tree)
            subprocess.run(["git", "apply", str(TASK_DIR / "reference.patch")], cwd=warmup_tree, check=True)
            _run(warmup_tree, workloads[0], warmup_tree / "warmup-output")
        candidates, references = [], []
        for i, workload in enumerate(workloads):
            # Release the original multi-GB checkpoint outputs before the next process.
            with tempfile.TemporaryDirectory(prefix=f"candidate-output-{i}-", dir=str(tree)) as td:
                candidates.append(_run(tree, workload, Path(td)))
            with tempfile.TemporaryDirectory(dir=str(tree.parent)) as td:
                ref_tree = Path(td) / "repo"
                shutil.copytree(TASK_DIR / "harbor" / "app", ref_tree)
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
            **{f"development_mean_{k}": sum(x[k] for x in candidates) / len(candidates) for k in SPEC_METRICS},
            # Operator evidence only (never solver-visible): paired per-workload measurements.
            "per_workload_measurements": [
                {"workload": f"workload-{i}", "candidate": dict(c), "reference": dict(r),
                 "score": workload_score(c, r)}
                for i, (c, r) in enumerate(zip(candidates, references))
            ],
        }
        return min(100.0, max(0.0, score)), score, "evaluation completed", metrics
    finally:
        if applied is not None and hasattr(applied, "cleanup"):
            applied.cleanup()
        elif applied is not None and getattr(applied, "workdir", None) is not None:
            shutil.rmtree(Path(applied.workdir).parent, ignore_errors=True)


if __name__ == "__main__":
    import sys
    print(json.dumps(evaluate(sys.argv[1] if len(sys.argv) > 1 else ""), default=str))
