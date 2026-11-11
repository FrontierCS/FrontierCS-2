"""Runner side of the nano_world_model judge.

The candidate's machine (Agent GPU image) gets this file, judge_runtime/sitecustomize.py, the
submitted tree and the public data of the suite (judge/public_inputs.py). There it runs
  native_run.py prepare <checkpoint>      the device check and checkpoint conversion, before any
                                          submitted code runs;
and then the tree's own src/main.py (command() and environment() below) with
NANOWM_LATENTS_RECORD set: each dfot_sample result is saved in call order.

The trusted runner (Verifier GPU image, this judge baked at /opt/frontier/task) runs
  native_run.py score <suite> <output> [<latents>]
the reference's native run on the real judge data and, given the candidate's latents, the same
native pipeline (caller: the reference tree) with dfot_sample returning those latents in call
order (NANOWM_LATENTS_REPLAY), so decode, writer and ground truth are the native code paths. It
writes {"reference": rows, "candidate": rows | null}, one {psnr, ssim, lpips} row per clip.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

JUDGE = Path(__file__).resolve().parent
MODEL_ROOT = Path("/opt/frontier/models")
CHECKPOINT_SOURCE = MODEL_ROOT / "point-maze" / "model.safetensors"
VAE = MODEL_ROOT / "sd-vae"
ENTRYPOINT = "src/main.py"


def command(python: str, repo: str, subset: str, checkpoint: str, run_root: str) -> list[str]:
    """The native evaluation of `repo` over the clips of `subset`."""
    return [
        python, f"{repo}/{ENTRYPOINT}",
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


def environment(runtime: str, repo: str, data_root: str, run_root: str) -> dict[str, str]:
    """Its variables besides the image runtime: data, codec, results, the import path
    (judge_runtime's sitecustomize first) and the CPU kernel set."""
    return {
        "DATASET_DIR": data_root,
        "VAE_MODEL_PATH": str(VAE),
        "RESULTS_DIR": run_root,
        "PYTHONPATH": os.pathsep.join((runtime, f"{repo}/frontier_runtime", f"{repo}/src")),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONHASHSEED": "3407",
        "PYTHONDONTWRITEBYTECODE": "1",
        # The candidate's machine and the trusted runner can have different CPUs (Intel, AMD):
        # MKL's CPU-independent code path gives both the same sampler arithmetic bit for bit.
        "MKL_CBWR": "COMPATIBLE",
    }


def prepare(checkpoint: Path) -> int:
    """Exit 1 unless the runner has the one declared A100-80GB; convert the pinned checkpoint."""
    import torch
    from safetensors.torch import load_file

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        print("one NVIDIA GPU is required", file=sys.stderr)
        return 1
    properties = torch.cuda.get_device_properties(0)
    if "A100" not in properties.name.upper() or properties.total_memory < 75 * 1024**3:
        print(f"the runner's GPU is not the declared A100-80GB: {properties.name}", file=sys.stderr)
        return 1
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(load_file(str(CHECKPOINT_SOURCE), device="cpu"), checkpoint)
    return 0


def measure(run_root: Path, subset: Path) -> list[dict[str, float]]:
    """PSNR, SSIM and LPIPS of each clip's saved future frames against its ground truth."""
    import lpips
    import numpy as np
    import piqa
    import torch

    device = torch.device("cuda")
    psnr_metric = piqa.PSNR(epsilon=1e-8, value_range=1.0, reduction="none").to(device).eval()
    ssim_metric = piqa.SSIM(window_size=11, sigma=1.5, n_channels=3, reduction="none").to(device).eval()
    lpips_metric = lpips.LPIPS(net="vgg").to(device).eval()
    rows: list[dict[str, float]] = []
    for spec in json.loads(subset.read_text(encoding="utf-8"))["slices"]:
        tag = f"traj_{spec['traj_idx']:04d}_start_{spec['start_frame']:04d}"
        (match,) = run_root.rglob(f"{tag}/raw_data.npz")
        with np.load(match) as payload:
            pred = torch.from_numpy(payload["pred"][:, 1:]).permute(1, 0, 2, 3).to(device)
            gt = torch.from_numpy(payload["gt"][:, 1:]).permute(1, 0, 2, 3).to(device)
        with torch.no_grad():
            psnr = psnr_metric(pred, gt).mean().item()
            ssim = ssim_metric(pred, gt).mean().item()
            lpips_value = lpips_metric(pred * 2 - 1, gt * 2 - 1).mean().item()
        rows.append({"psnr": psnr, "ssim": ssim, "lpips": lpips_value})
    return rows


def _reference_tree(scratch: Path) -> Path:
    """The starter with judge/reference.patch applied."""
    tree = scratch / "reference"
    shutil.copytree(JUDGE / "pristine", tree, symlinks=True)
    for path in (tree, *tree.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    subprocess.run(["git", "apply", "--whitespace=nowarn", str(JUDGE / "reference.patch")],
                   cwd=tree, check=True, timeout=120,
                   env={**os.environ, "GIT_CEILING_DIRECTORIES": str(scratch.resolve())})
    return tree


def _native(repo: Path, subset: Path, checkpoint: Path, run_root: Path,
            extra: dict[str, str]) -> list[dict[str, float]]:
    env = {**os.environ,
           **environment(str(JUDGE / "judge_runtime"), str(repo), str(JUDGE / "judge_data"),
                         str(run_root)),
           **extra}
    subprocess.run(command(sys.executable, str(repo), str(subset), str(checkpoint), str(run_root)),
                   cwd=repo, env=env, check=True)
    return measure(run_root, subset)


def score(suite: str, output: Path, latents: Path | None) -> None:
    subset = JUDGE / "judge_data" / f"{suite}_subset.json"
    scratch = Path(tempfile.mkdtemp(prefix="nanowm-trusted-", dir=Path.cwd()))
    checkpoint = scratch / "point-maze-30k.pt"
    if prepare(checkpoint) != 0:
        raise RuntimeError("the trusted runner is not the declared A100-80GB")
    reference = _reference_tree(scratch)
    rows = {"reference": _native(reference, subset, checkpoint, scratch / "reference-run", {}),
            "candidate": None}
    if latents is not None:
        rows["candidate"] = _native(reference, subset, checkpoint, scratch / "candidate-run",
                                    {"NANOWM_LATENTS_REPLAY": str(latents)})
    output.write_text(json.dumps(rows), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        sys.exit(prepare(Path(sys.argv[2])))
    score(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]) if len(sys.argv) == 5 else None)
