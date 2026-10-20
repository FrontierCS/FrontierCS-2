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
SPEC_METRICS = (
    "ordinary_quantized_png_psnr_db",
    "ema_quantized_png_psnr_db",
    "method_process_wall_seconds",
)


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
    q = {}
    for kind, matches in eval_dirs.items():
        if len(matches) != 1:
            raise RuntimeError("reconstruction PNG directory is missing")
        values = []
        for idx, name in enumerate(names):
            gt = Image.open(workload / "val" / name).convert("RGB")
            while min(gt.size) >= 512:
                gt = gt.resize(tuple(x // 2 for x in gt.size), Image.Resampling.BOX)
            scale = 256 / min(gt.size)
            gt = gt.resize(tuple(round(x * scale) for x in gt.size), Image.Resampling.BICUBIC)
            a = np.asarray(gt, dtype=np.uint8)
            y, x = (a.shape[0] - 256) // 2, (a.shape[1] - 256) // 2
            a = a[y:y + 256, x:x + 256].astype(np.float64)
            b = np.asarray(Image.open(matches[0] / f"{idx:06d}.png").convert("RGB"), dtype=np.float64)
            mse = float(np.mean((a - b) ** 2))
            values.append(float("inf") if mse == 0 else 10.0 * math.log10(255.0 * 255.0 / mse))
        q[kind] = sum(values) / len(values)
    result = {
        "ordinary_quantized_png_psnr_db": float(q["ordinary"]),
        "ema_quantized_png_psnr_db": float(q["ema"]),
        "method_process_wall_seconds": elapsed,
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
            decisions = {
                SPEC_METRICS[0]: cand[SPEC_METRICS[0]] > ref[SPEC_METRICS[0]],
                SPEC_METRICS[1]: cand[SPEC_METRICS[1]] > ref[SPEC_METRICS[1]],
                SPEC_METRICS[2]: cand[SPEC_METRICS[2]] < ref[SPEC_METRICS[2]],
            }
            passes.append({"workload": f"workload-{i}", "metrics": decisions})
        beats = int(all(v for row in passes for v in row["metrics"].values()))
        quality = sum(c[SPEC_METRICS[0]] + c[SPEC_METRICS[1]] for c in candidates) / (6 * 40.0)
        speed = sum(r[SPEC_METRICS[2]] / c[SPEC_METRICS[2]] for c, r in zip(candidates, references)) / 3
        unbounded = 70.0 * quality + 30.0 * speed
        cand_mean = {k: sum(x[k] for x in candidates) / 3 for k in SPEC_METRICS}
        ref_score = sum(70.0 * ((r[SPEC_METRICS[0]] + r[SPEC_METRICS[1]]) / 80.0) + 30.0 for r in references) / 3
        metrics = {
            "beat_spec_workload_passes": passes,
            "beats_reference": beats,
            "reference_score": ref_score,
            "margin": unbounded - ref_score,
            "development_mean_ordinary_quantized_png_psnr_db": cand_mean[SPEC_METRICS[0]],
            "development_mean_ema_quantized_png_psnr_db": cand_mean[SPEC_METRICS[1]],
            "development_mean_method_process_wall_seconds": cand_mean[SPEC_METRICS[2]],
        }
        return min(100.0, max(0.0, unbounded)), unbounded, "evaluation completed", metrics
    finally:
        if applied is not None and hasattr(applied, "cleanup"):
            applied.cleanup()
        elif applied is not None and getattr(applied, "workdir", None) is not None:
            shutil.rmtree(Path(applied.workdir).parent, ignore_errors=True)


if __name__ == "__main__":
    import sys
    print(json.dumps(evaluate(sys.argv[1] if len(sys.argv) > 1 else ""), default=str))
