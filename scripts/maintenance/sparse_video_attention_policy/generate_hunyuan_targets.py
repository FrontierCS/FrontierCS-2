#!/usr/bin/env python3
"""Generate the HunyuanVideo dense/reference targets for sparse_video_attention_policy.

Authoring tool, not part of any image. It reuses the task evaluator's own
generation path (`evaluator._run_generation`: the same shell entrypoint, private
python shim, two-step warm-up, processor rebuild, synchronized timing boundary
and MP4 decode check), so prepared targets follow exactly the protocol a
candidate is measured with.

Timing protocol (same as the existing Wan targets): one generation process per
H100, exactly four concurrent processes on one four-H100 node, and the same
wave composition the evaluator uses. A wave that also contains Wan workloads
(the feedback wave: Wan 0, Wan 1, HunyuanVideo 0, HunyuanVideo 1) is generated
with those Wan slots running the same kind of generation as fillers, so every
recorded HunyuanVideo latency saw the evaluator's concurrency. Wan filler
latencies are kept only as a drift check against the existing Wan manifest.

High parallelism comes from many independent four-H100 nodes (shards), never
from more than four concurrent processes on one node.

Subcommands
  plan       (CPU) list the wave instances and the shard assignment for N GPUs.
  run        (GPU node with 4 H100s, inside the task's verifier-gpu image)
             generate this shard's wave instances.
  calibrate  (1 GPU, same image) fidelity of reference and replays vs dense,
             and the replay latency spread used to justify the 4% margin.
  merge      (CPU, repository checkout) validate all shard records and write
             the MP4s plus ready manifest entries into judge/dense_targets.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SCHEMA = "frontier.svg-hunyuan-targets/v1"
NODE_GPUS = 4
KINDS = ("dense", "reference", "replay")
PATTERN = {"dense": "dense", "reference": "EAR", "replay": "EAR"}
_HERE = Path(__file__).resolve()
# Repository defaults only; inside an image the script is copied elsewhere and
# every GPU subcommand takes explicit paths.
REPO = _HERE.parents[3] if len(_HERE.parents) > 3 else _HERE.parent
TASK = REPO / "tasks/problems/MLSys/sparse_video_attention_policy"


def stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_evaluator(judge: Path):
    """Import the task evaluator by path (it imports its sibling submission_patch)."""
    spec = importlib.util.spec_from_file_location("svg_task_evaluator", judge / "evaluator.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- planning


def hunyuan_only_instances(evaluator) -> list[dict]:
    """Owner-approved minimal pack: each HunyuanVideo video generated once.

    6 clips x (dense, reference) = 12 processes in three four-GPU instances,
    four concurrent processes per node and no Wan companion slots: graded
    dense, graded reference, then the two feedback clips' dense and reference.
    """
    jobs = {"graded": [], "feedback": []}
    for role, final in (("feedback", False), ("graded", True)):
        workloads = evaluator._workloads(final)
        for workload in workloads:
            if workload.model == "hunyuan":
                jobs[role].append((role, workloads.index(workload), workload))
    groups = [
        ("graded-dense", [(job, "dense") for job in jobs["graded"]]),
        ("graded-reference", [(job, "reference") for job in jobs["graded"]]),
        ("feedback-dense-reference", [(job, kind) for kind in ("dense", "reference") for job in jobs["feedback"]]),
    ]
    instances = []
    for name, group in groups:
        if len(group) != NODE_GPUS:
            raise ValueError(f"{name} does not fill a four-GPU node")
        kinds = {kind for _job, kind in group}
        instances.append({
            "id": f"hunyuan-only-{name}", "role": group[0][0][0], "wave": None,
            "kind": kinds.pop() if len(kinds) == 1 else "mixed", "repeat": 0,
            "slots": [{"gpu": gpu, "workload": w.workload_id, "model": w.model, "index": index,
                       "role": role, "kind": kind}
                      for gpu, ((role, index, w), kind) in enumerate(group)],
        })
    return instances


def wave_instances(evaluator, replays: int, pack: str = "waves") -> list[dict]:
    """Every (wave, kind, repeat) to run; each instance occupies one 4-GPU node."""
    if pack == "hunyuan-only":
        if replays:
            raise ValueError("--pack hunyuan-only generates each video once; use --replays 0")
        return hunyuan_only_instances(evaluator)
    if pack != "waves":
        raise ValueError("unknown pack")
    if replays < 0:
        raise ValueError("replays must be non-negative")
    instances = []
    for role, final in (("feedback", False), ("graded", True)):
        for wave_index, wave in enumerate(evaluator._waves(evaluator._workloads(final))):
            if not any(workload.model == "hunyuan" for workload in wave):
                continue
            if len(wave) != NODE_GPUS:
                raise ValueError("a HunyuanVideo wave is not a full four-GPU wave")
            kinds = [("dense", 0), ("reference", 0)] + [("replay", r) for r in range(1, replays + 1)]
            for kind, repeat in kinds:
                instances.append({
                    "id": f"{role}-w{wave_index}-{kind}{repeat if kind == 'replay' else ''}",
                    "role": role, "wave": wave_index, "kind": kind, "repeat": repeat,
                    "slots": [{"gpu": gpu, "workload": w.workload_id, "model": w.model,
                               "index": evaluator._workloads(final).index(w)}
                              for gpu, w in enumerate(wave)],
                })
    return instances


def shard_of(instances: list[dict], shard: int, shards: int) -> list[dict]:
    if not 0 <= shard < shards:
        raise ValueError("shard index out of range")
    # Dense and reference instances first so a small pool finishes targets early.
    ordered = sorted(instances, key=lambda row: (row["kind"] == "replay", row["repeat"], row["id"]))
    return ordered[shard::shards]


def estimate(instances: list[dict], shards: int) -> dict:
    # Rough planning numbers (seconds per generation process incl. load/warm-up);
    # Wan values come from the existing manifest, HunyuanVideo ones are estimates
    # (81 frames, one transformer pass per step) to be replaced by measurements.
    seconds = {("wan", "dense"): 2300, ("wan", "EAR"): 1500,
               ("hunyuan", "dense"): 1500, ("hunyuan", "EAR"): 1000}
    walls = [max(seconds[(slot["model"], PATTERN[slot.get("kind", row["kind"])])] for slot in row["slots"])
             for row in instances]
    per_shard = [sum(walls[i::shards]) for i in range(shards)]
    return {"wave_instances": len(instances), "nodes": shards, "gpus": shards * NODE_GPUS,
            "estimated_gpu_seconds": NODE_GPUS * sum(walls),
            "estimated_wall_seconds_excluding_image_pull": max(per_shard) if per_shard else 0}


def command_plan(args) -> int:
    evaluator = load_evaluator(Path(args.task_root) / "judge" if (Path(args.task_root) / "judge").is_dir()
                               else Path(args.task_root))
    if args.gpus % NODE_GPUS:
        raise SystemExit(f"--gpus must be a multiple of {NODE_GPUS} (one shard per four-H100 node)")
    shards = args.gpus // NODE_GPUS
    instances = wave_instances(evaluator, args.replays, args.pack)
    plan = {"schema": SCHEMA, "shards": [[row["id"] for row in shard_of(instances, i, shards)]
                                         for i in range(shards)],
            "estimate": estimate(instances, shards)}
    print(json.dumps(plan, indent=2))
    return 0


# ---------------------------------------------------------------- GPU run


def visible_gpus() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def command_child(args) -> int:
    """One generation process on one visible GPU (invoked by `run`)."""
    evaluator = load_evaluator(Path(args.judge))
    workload = evaluator._workloads(args.role == "graded")[args.index]
    video = Path(args.video)
    started = time.monotonic()
    latency, _frames = evaluator._run_generation(Path(args.tree), workload, PATTERN[args.kind], keep_video=video)
    record = {"workload": workload.workload_id, "model": workload.model, "prompt_id": workload.prompt_id,
              "seed": workload.seed, "kind": args.kind, "repeat": args.repeat,
              "e2e_latency_seconds": latency, "video": video.name, "sha256": sha256(video),
              "process_wall_seconds": time.monotonic() - started,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "finished_at": stamp()}
    write_json(Path(args.record), record)
    return 0


def run_instance(instance: dict, judge: Path, tree: Path, output: Path, scratch: Path) -> dict:
    directory = output / instance["id"]
    directory.mkdir(parents=True, exist_ok=True)
    evaluator = sys.modules["svg_task_evaluator"]

    def slot(entry):
        kind = entry.get("kind", instance["kind"])
        role = entry.get("role", instance["role"])
        stem = f"{entry['workload']}-{kind}{instance['repeat'] or ''}"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(entry["gpu"])
        env["FRONTIER_EVAL_SCRATCH"] = str(scratch / f"gpu{entry['gpu']}")
        Path(env["FRONTIER_EVAL_SCRATCH"]).mkdir(parents=True, exist_ok=True)
        log = directory / f"{stem}.log"
        with log.open("wb") as handle:
            proc = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), "child", "--judge", str(judge),
                 "--tree", str(tree), "--role", role, "--index", str(entry["index"]),
                 "--kind", kind, "--repeat", str(instance["repeat"]),
                 "--video", str(directory / f"{stem}.mp4"), "--record", str(directory / f"{stem}.json")],
                env=env, stdout=handle, stderr=subprocess.STDOUT,
                timeout=evaluator.WORKER_TIMEOUT_SECONDS)
        record_path = directory / f"{stem}.json"
        if proc.returncode != 0 or not record_path.is_file():
            return {"workload": entry["workload"], "kind": kind, "error": proc.returncode,
                    "log_tail": log.read_bytes()[-4000:].decode("utf-8", "replace")}
        record = json.loads(record_path.read_text())
        record.update(gpu=entry["gpu"], filler=entry["model"] != "hunyuan", log=log.name)
        return record

    began = stamp()
    with ThreadPoolExecutor(max_workers=NODE_GPUS) as pool:
        results = list(pool.map(slot, instance["slots"]))
    summary = {"instance": instance, "started_at": began, "finished_at": stamp(),
               "node_concurrency": len(instance["slots"]), "results": results}
    write_json(directory / "instance.json", summary)
    return summary


def command_run(args) -> int:
    task_root = Path(args.task_root)
    judge = task_root / "judge" if (task_root / "judge" / "evaluator.py").is_file() else task_root
    evaluator = load_evaluator(judge)
    gpus = visible_gpus()
    if len(gpus) != NODE_GPUS:
        raise SystemExit(f"this protocol needs exactly {NODE_GPUS} visible H100s per node, saw {len(gpus)}")
    if not all("H100" in line for line in gpus):
        raise SystemExit("all visible GPUs must be H100s")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Builds, Triton caches and temporary videos stay on local disk; only the
    # retained MP4s and records go to --output.
    scratch = Path(args.scratch)
    os.environ["FRONTIER_EVAL_SCRATCH"] = str(scratch / "build")
    Path(os.environ["FRONTIER_EVAL_SCRATCH"]).mkdir(parents=True, exist_ok=True)
    instances = shard_of(wave_instances(evaluator, args.replays, args.pack), args.shard, args.num_shards)
    # Dense and reference targets are produced from the exact upstream tree.
    applied = evaluator._apply_patch(evaluator.REFERENCE_PATCH)
    if not applied.ok or applied.workdir is None:
        raise SystemExit("reference.patch did not apply: " + applied.message)
    tree = Path(applied.workdir)
    evaluator._check_public_contract(tree)
    evaluator._build_extension(tree)
    header = {"schema": SCHEMA, "shard": args.shard, "num_shards": args.num_shards, "replays": args.replays,
              "pack": args.pack,
              "gpus": gpus, "image": os.environ.get("SVG_TARGET_IMAGE"),
              "evaluator_sha256": sha256(judge / "evaluator.py"),
              "reference_patch_sha256": sha256(evaluator.REFERENCE_PATCH),
              "instances": [row["id"] for row in instances], "started_at": stamp()}
    write_json(output / "shard.json", header)
    summaries = [run_instance(instance, judge, tree, output, scratch) for instance in instances]
    header.update(finished_at=stamp(), failures=sum(1 for s in summaries for r in s["results"] if "error" in r))
    write_json(output / "shard.json", header)
    shutil.rmtree(scratch, ignore_errors=True)
    return 1 if header["failures"] else 0


# ---------------------------------------------------------------- collection


def collect(inputs: list[Path], judge: Path) -> dict:
    """All successful HunyuanVideo records keyed by (workload, kind), protocol-checked."""
    evaluator_sha = sha256(judge / "evaluator.py")
    patch_sha = sha256(judge / "reference.patch")
    records: dict[tuple[str, str], list[dict]] = {}
    fillers = []
    for root in inputs:
        header = json.loads((root / "shard.json").read_text())
        if header.get("schema") != SCHEMA:
            raise SystemExit(f"{root}: unknown shard schema")
        if header["evaluator_sha256"] != evaluator_sha or header["reference_patch_sha256"] != patch_sha:
            raise SystemExit(f"{root}: generated with a different evaluator or reference.patch")
        if len(header["gpus"]) != NODE_GPUS or header.get("failures"):
            raise SystemExit(f"{root}: shard did not complete on exactly four GPUs")
        for name in header["instances"]:
            summary = json.loads((root / name / "instance.json").read_text())
            if summary["node_concurrency"] != NODE_GPUS:
                raise SystemExit(f"{root}/{name}: not four concurrent processes")
            for record in summary["results"]:
                if "error" in record:
                    raise SystemExit(f"{root}/{name}: failed slot {record['workload']}")
                video = root / name / record["video"]
                if sha256(video) != record["sha256"]:
                    raise SystemExit(f"{video}: hash mismatch")
                record = dict(record, path=str(video), instance=name)
                if record["filler"]:
                    fillers.append(record)
                else:
                    records.setdefault((record["workload"], record["kind"]), []).append(record)
    return {"records": records, "fillers": fillers}


def hunyuan_workloads(evaluator):
    return [w for final in (False, True) for w in evaluator._workloads(final) if w.model == "hunyuan"]


def command_merge(args) -> int:
    task = Path(args.task)
    judge = task / "judge"
    evaluator = load_evaluator(judge)
    collected = collect([Path(p) for p in args.inputs], judge)
    records = collected["records"]
    manifest_path = judge / "dense_targets/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != evaluator.DENSE_MANIFEST_SCHEMA_VERSION:
        raise SystemExit("unexpected manifest schema")
    updates = {}
    for workload in hunyuan_workloads(evaluator):
        entry = {}
        for kind in ("dense", "reference"):
            rows = records.get((workload.workload_id, kind), [])
            if len(rows) != 1:
                raise SystemExit(f"{workload.workload_id}: expected one {kind} record, found {len(rows)}")
            row = rows[0]
            if (row["model"], row["prompt_id"], row["seed"]) != ("hunyuan", workload.prompt_id, workload.seed):
                raise SystemExit(f"{workload.workload_id}: record does not match the suite")
            latency = float(row["e2e_latency_seconds"])
            if not math.isfinite(latency) or latency <= 0:
                raise SystemExit(f"{workload.workload_id}: invalid latency")
            role, index = workload.workload_id.rsplit("_", 2)[-2:]
            name = f"hunyuan-{role}-{index}-{kind}.mp4"
            entry[kind] = {"path": name, "sha256": row["sha256"], "e2e_latency_seconds": latency,
                           "_source": row["path"]}
        updates[workload.workload_id] = entry
    drift = []
    for row in collected["fillers"]:
        target = manifest["targets"].get(row["workload"], {})
        kind = "dense" if row["kind"] == "dense" else "reference"
        if target.get("status") == evaluator.TARGET_READY:
            prepared = target[kind]["e2e_latency_seconds"]
            drift.append({"workload": row["workload"], "kind": row["kind"], "repeat": row["repeat"],
                          "latency": row["e2e_latency_seconds"], "manifest_latency": prepared,
                          "ratio": row["e2e_latency_seconds"] / prepared})
    report = {"schema": SCHEMA, "merged_at": stamp(), "inputs": [str(p) for p in args.inputs],
              "targets": {k: {kind: {key: v[kind][key] for key in ("path", "sha256", "e2e_latency_seconds")}
                              for kind in ("dense", "reference")} for k, v in updates.items()},
              "wan_filler_drift": drift}
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.write:
        print("dry run: pass --write to copy videos and mark the targets ready", file=sys.stderr)
        return 0
    for workload_id, entry in updates.items():
        record = manifest["targets"][workload_id]
        for kind in ("dense", "reference"):
            source = Path(entry[kind].pop("_source"))
            shutil.copyfile(source, judge / "dense_targets" / entry[kind]["path"])
            record[kind] = entry[kind]
        record["status"] = evaluator.TARGET_READY
    write_json(manifest_path, manifest)
    if args.report:
        write_json(Path(args.report), report)
    return 0


def command_calibrate(args) -> int:
    """Replay spread and fidelity of the unchanged reference (needs one GPU)."""
    judge = Path(args.judge)
    evaluator = load_evaluator(judge)
    collected = collect([Path(p) for p in args.inputs], judge)
    records = collected["records"]
    rows = []
    for workload in hunyuan_workloads(evaluator):
        dense = records[(workload.workload_id, "dense")][0]
        reference = records[(workload.workload_id, "reference")][0]
        dense_frames = evaluator._load_video(Path(dense["path"]))
        reference_values = evaluator._fidelity(evaluator._load_video(Path(reference["path"])), dense_frames)
        reference_values["e2e_latency_seconds"] = reference["e2e_latency_seconds"]
        replays = []
        for replay in sorted(records.get((workload.workload_id, "replay"), []), key=lambda r: r["repeat"]):
            values = evaluator._fidelity(evaluator._load_video(Path(replay["path"])), dense_frames)
            values["e2e_latency_seconds"] = replay["e2e_latency_seconds"]
            decisions = evaluator._beat_decisions(values, reference_values)
            replays.append({"repeat": replay["repeat"], **values,
                            "apparent_speedup": 1 - values["e2e_latency_seconds"] / reference["e2e_latency_seconds"],
                            "decisions": decisions})
        rows.append({"workload": workload.workload_id, "dense_latency": dense["e2e_latency_seconds"],
                     "reference": reference_values,
                     "reference_speedup_over_dense": dense["e2e_latency_seconds"] / reference["e2e_latency_seconds"],
                     "replays": replays})
    speedups = [r["apparent_speedup"] for row in rows for r in row["replays"]]
    margin = evaluator.MIN_LATENCY_SPEEDUP_FRACTION
    verdict = {
        "max_apparent_replay_speedup": max(speedups) if speedups else None,
        "latency_margin": margin,
        # The gate must never pass an unchanged reference by luck, and keeps the
        # original one-percentage-point guard band beyond the observed spread.
        "latency_margin_holds": bool(speedups) and max(speedups) + 0.01 <= margin,
        "replay_latency_gate_passes": sum(r["decisions"]["e2e_latency_seconds"] for row in rows for r in row["replays"]),
        "replay_fidelity_failures": sum(not all(r["decisions"][k] for k in ("psnr_db", "ssim", "lpips"))
                                        for row in rows for r in row["replays"]),
    }
    result = {"schema": SCHEMA, "calibrated_at": stamp(), "workloads": rows, "verdict": verdict}
    write_json(Path(args.output), result)
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["latency_margin_holds"] and not verdict["replay_fidelity_failures"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--task-root", default=str(TASK))
    plan.add_argument("--gpus", type=int, required=True)
    plan.add_argument("--replays", type=int, default=3)
    plan.add_argument("--pack", choices=("waves", "hunyuan-only"), default="waves")
    run = sub.add_parser("run")
    run.add_argument("--task-root", required=True, help="judge directory holding harbor/app (image layout)")
    run.add_argument("--output", required=True)
    run.add_argument("--shard", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--replays", type=int, default=3)
    run.add_argument("--scratch", default="/tmp/svg-hunyuan-scratch", help="local, non-shared scratch")
    run.add_argument("--pack", choices=("waves", "hunyuan-only"), default="waves",
                     help="waves: evaluator wave composition with Wan fillers; "
                          "hunyuan-only: 12 HunyuanVideo videos, once each, 3 x 4 GPUs")
    child = sub.add_parser("child")
    for name in ("--judge", "--tree", "--role", "--kind", "--video", "--record"):
        child.add_argument(name, required=True)
    child.add_argument("--index", type=int, required=True)
    child.add_argument("--repeat", type=int, default=0)
    merge = sub.add_parser("merge")
    merge.add_argument("--task", default=str(TASK))
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--write", action="store_true")
    merge.add_argument("--report")
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--judge", required=True)
    calibrate.add_argument("--inputs", nargs="+", required=True)
    calibrate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    return {"plan": command_plan, "run": command_run, "child": command_child,
            "merge": command_merge, "calibrate": command_calibrate}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
