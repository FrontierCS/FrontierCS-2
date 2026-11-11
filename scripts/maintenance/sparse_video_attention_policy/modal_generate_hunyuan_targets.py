"""Modal launcher for generate_hunyuan_targets.py (authoring only; not run in CI).

One Modal function call = one shard on one 4 x H100 container. Image: the task's
published verifier-gpu image (real runtime, Wan checkpoint, pinned stack) plus the
pinned HunyuanVideo snapshot layer and this branch's judge/ and public repo laid
out exactly as the verifier image lays them out (judge at ROOT, repo at
ROOT/harbor/app). Once a verifier-gpu image built from this branch exists, set
SVG_TARGET_IMAGE to it and SVG_TARGET_IMAGE_HAS_HUNYUAN=1 to skip the extra layer.

Shard outputs (videos, per-process records and logs) are written to the Modal
Volume ``svg-hunyuan-targets`` under ``<run>/shard-NNN``; the calibration pass
writes ``<run>/calibration.json`` there. Fetch them with
``modal volume get svg-hunyuan-targets <run> <local dir>``.

  MODAL_PROFILE=frontier-cs-2 modal run \
    scripts/maintenance/sparse_video_attention_policy/modal_generate_hunyuan_targets.py \
    --gpus 40 --replays 3 --run svg-hunyuan-targets-<date>
"""

import json
import os
import pathlib
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
# Inside the container this module lives at /root/<name>.py; repository paths
# are only meaningful (and only used) on the local side that builds the image.
REPO = HERE.parents[2] if modal.is_local() else None
TASK = REPO / "tasks/problems/MLSys/sparse_video_attention_policy" if REPO else None
PUBLISHED = (
    "docker.io/joyemang33/fcs2-task-verifier-sparse_video_attention_policy-gpu"
    "@sha256:81a52c7461a57e8d0edc075332fe9e2be408e9e3bf94bf73ad564983d3688dcf"
)
BASE = os.environ.get("SVG_TARGET_IMAGE", PUBLISHED)
ROOT = "/opt/svg-hunyuan/task"
SCRIPT = "/opt/svg-hunyuan/generate_hunyuan_targets.py"
VOLUME_PATH = "/vol"
NODE_GPUS = 4


def snapshot_commands() -> list[str]:
    assets = json.loads((TASK / "model-assets.json").read_text())
    model = next(m for m in assets["models"] if m["target"] == "HunyuanVideo")
    commands = ["chmod 0755 /opt/frontier/models"]
    for group in (
        [r["path"] for r in model["files"] if not r["path"].startswith("transformer/")],
        [r["path"] for r in model["files"] if r["path"].startswith("transformer/")],
    ):
        code = (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id={model['repo_id']!r}, revision={model['revision']!r}, "
            f"local_dir='/opt/frontier/models/HunyuanVideo', allow_patterns={group!r})"
        )
        commands.append(
            f"HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c {json.dumps(code)}"
        )
    commands.append(
        "chmod -R a+rX,a-w /opt/frontier/models/HunyuanVideo && chmod 0555 /opt/frontier/models"
    )
    return commands


def build_image():
    if not modal.is_local():
        return modal.Image.debian_slim()
    image = modal.Image.from_registry(BASE)
    if os.environ.get("SVG_TARGET_IMAGE_HAS_HUNYUAN") != "1":
        image = image.run_commands(*snapshot_commands())
    image = image.add_local_dir(str(TASK / "judge"), ROOT, copy=True)
    image = image.add_local_dir(
        str(TASK / "agent-base/frontier/task/repo"), ROOT + "/harbor/app", copy=True
    )
    image = image.add_local_file(
        str(HERE / "generate_hunyuan_targets.py"), SCRIPT, copy=True
    )
    # Re-state the verifier image's runtime environment explicitly.
    return image.run_commands(f"chmod -R a+rX {ROOT}").env(
        {
            "SVG_TARGET_IMAGE": BASE,
            "FRONTIER_MODEL_ROOT": "/opt/frontier/models",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCH_HOME": "/opt/frontier/models/torch",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )


app = modal.App("svg-hunyuan-targets", image=build_image())
volume = modal.Volume.from_name("svg-hunyuan-targets", create_if_missing=True)


def _tail(proc) -> dict:
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-20000:],
        "stderr": proc.stderr[-20000:],
    }


@app.function(
    gpu=f"H100:{NODE_GPUS}",
    cpu=16,
    memory=256 * 1024,
    timeout=24 * 3600,
    volumes={VOLUME_PATH: volume},
)
def shard(index: int, shards: int, replays: int, run: str, pack: str = "waves") -> dict:
    import subprocess
    import sys

    output = pathlib.Path(VOLUME_PATH) / run / f"shard-{index:03d}"
    output.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            SCRIPT,
            "run",
            "--task-root",
            ROOT,
            "--output",
            str(output),
            "--shard",
            str(index),
            "--num-shards",
            str(shards),
            "--replays",
            str(replays),
            "--pack",
            pack,
        ],
        capture_output=True,
        text=True,
    )
    (output / "process.json").write_text(json.dumps(_tail(proc), indent=2) + "\n")
    volume.commit()
    return _tail(proc)


@app.function(gpu="H100", timeout=4 * 3600, volumes={VOLUME_PATH: volume})
def calibrate(run: str) -> dict:
    import subprocess
    import sys

    volume.reload()
    base = pathlib.Path(VOLUME_PATH) / run
    out = base / "calibration.json"
    inputs = sorted(str(p) for p in base.glob("shard-*") if p.is_dir())
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            SCRIPT,
            "calibrate",
            "--judge",
            ROOT,
            "--inputs",
            *inputs,
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    volume.commit()
    return _tail(proc)


@app.local_entrypoint()
def main(gpus: int = 40, replays: int = 3, run: str = "", pack: str = "waves"):
    if gpus % NODE_GPUS:
        raise SystemExit(f"--gpus must be a multiple of {NODE_GPUS}")
    if not run or "/" in run:
        raise SystemExit(
            "--run must name a new directory in the svg-hunyuan-targets volume"
        )
    shards = gpus // NODE_GPUS
    began = time.time()
    results = list(
        shard.starmap([(i, shards, replays, run, pack) for i in range(shards)])
    )
    failed = [i for i, r in enumerate(results) if r["returncode"] != 0]
    summary = {
        "run": run,
        "pack": pack,
        "shards": shards,
        "replays": replays,
        "failed_shards": failed,
        "wall_seconds": round(time.time() - began, 1),
        "image": BASE,
        "results": [
            {k: r[k][-3000:] if isinstance(r[k], str) else r[k] for k in r}
            for r in results
        ],
    }
    if not failed and replays:
        summary["calibration"] = calibrate.remote(run)
    print(json.dumps(summary, indent=2))
    print(f"fetch: modal volume get svg-hunyuan-targets {run} <local-dir>")
