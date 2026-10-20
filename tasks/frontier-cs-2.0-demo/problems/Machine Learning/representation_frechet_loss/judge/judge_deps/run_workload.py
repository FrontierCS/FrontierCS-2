"""Execute paired FD-SIM workloads with submitted Python only in queue actors."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time

import yaml

from runtime import HERE, ChildRunner, SetupDeadlineExpired, admit_reference, check_queue_runtime, prepare_assets, sha256
from private_evidence import Journal, report_bytes

TASK = HERE.parent
METRICS = ("inception_fdr", "convnext_fdr", "dinov2_fdr", "mae_fdr", "siglip_fdr", "clip_fdr")
TRAIN_MODELS = ("vit_so400m_patch16_siglip_256.v2_webli", "vit_large_patch16_224.mae", "inception")
TRAIN_STATS = ("vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz",
               "vit_large_patch16_224_mae_in256_t224_stats.npz", "guided_diffusion_stats.npz")


def arguments(workload, recipe, models, stats, output, *, checkpoint=None):
    args = ["--model", "pMF_B", "--img_size", "256", "--patch_size", "16",
            "--rope_2d", "--learned_pe", "--disable_v_head", "--dtype", "fp32",
            "--cfg", "8.5", "--interval_min", "0.1", "--interval_max", "0.7",
            "--num_sampling_steps", "1", "--batch_size", str(recipe["batch_size"]),
            "--epochs", "1", "--steps_per_epoch", str(workload["updates"]),
            "--warmup_epochs", "0", "--lr", "1e-6", "--lr_sched", "constant",
            "--min_lr", "0", "--disable_vis", "--vis_freq", "0", "--eval_freq", "0",
            "--disable_wandb", "--milestone_interval", "0", "--save_freq", "1",
            "--print_freq", "1", "--output_dir", str(output), "--project", "workload",
            "--exp_name", "eval" if checkpoint else "train"]
    if checkpoint:
        return args + ["--resume_from", str(checkpoint), "--seed", str(workload["eval_seed"]),
                       "--no_prc", "--cfg_list", "8.5", "--eval_ema_labels", "online",
                       "--num_images", str(recipe["held_out_images"]),
                       "--eval_bsz", str(recipe["batch_size"])]
    args += ["--load_from", str(models / "fd-loss/checkpoints/base/pMF-B_256.pth"),
             "--seed", str(workload["seed"]), "--queue_size", str(recipe["warm_start_images"]),
             "--fd_queue_fill_bsz", str(recipe["batch_size"]), "--fd_eigvalsh",
             "--fd_repr_models", *TRAIN_MODELS, "--fd_repr_pool_types", "cls", "cls", "cls",
             "--fd_target_sizes", "224", "224", "256", "--fd_repr_stats_paths",
             *(str(stats / name) for name in TRAIN_STATS),
             "--fid_stats_path", str(stats / "guided_diffusion_stats.npz")]
    if workload["mode"] == "online":
        args += ["--fd_online_accum"]
    elif workload["mode"] == "ema":
        args += ["--fd_ema_beta", str(workload["beta"])]
    else:
        raise ValueError("unsupported estimator setting")
    return args


def validate_report(train, evaluation, workload, recipe):
    guards = ("gpu", "model_lock", "finite_features", "finite_covariances", "finite_gradients",
              "parameter_update", "queue_state_roundtrip", "authentic_warm_start", "entrypoint_executed")
    if any(train.get(key) is not True for key in guards):
        raise ValueError("training authenticity check failed")
    if train["updates"] != workload["updates"] or train["warm_start_images"] != recipe["warm_start_images"]:
        raise ValueError("incomplete training workload")
    if evaluation.get("gpu") is not True or evaluation.get("model_lock") is not True:
        raise ValueError("evaluation model/device check failed")
    if evaluation.get("finite_features") is not True or evaluation.get("finite_covariances") is not True:
        raise ValueError("evaluation numerical check failed")
    if evaluation.get("num_images") != recipe["held_out_images"]:
        raise ValueError("wrong held-out sample count")
    values = evaluation["metrics"]
    if set(values) != set(METRICS) or any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("invalid held-out metrics")
    return values


def execute(repo, suite, model_root):
    journal = Journal(os.environ.get("FRONTIER_FD_EVIDENCE_PATH"))
    journal.record("suite_started", suite=suite, runner_pid=os.getpid(), runner_uid=os.getuid())
    check_queue_runtime()
    boundary = HERE / "queue_boundary/src"
    sys.path.insert(0, str(boundary))
    from observer_queue import prepare_queue_snapshot

    source = json.loads((HERE / "source.json").read_text())
    if sha256(HERE / "eval_all_fds.py") != source["private_eval_sha256"]:
        raise ValueError("private upstream evaluator source changed")
    recipe = json.loads((HERE / "workloads.json").read_text())
    config = yaml.safe_load((TASK / "config.yaml").read_text())
    local_deadline = os.environ.get("FRONTIER_VERIFIER_LOCAL_DEADLINE")
    try:
        suite_seconds = (config["runtime"].get("final_timeout_seconds", config["runtime"]["timeout_seconds"])
                         if suite == "final" else config["runtime"]["timeout_seconds"])
        runner = ChildRunner(suite_seconds - 90,
                             envelope_deadline=float(local_deadline) if local_deadline is not None else None,
                             on_scientific_start=lambda: journal.record("scientific_started"))
    except TimeoutError:
        # Trusted runner setup expired before any candidate process was launched.
        print(json.dumps({"verifier_failure": "verifier_setup_timeout"}))
        raise SystemExit(75)

    def interrupted(signum, frame):
        runner.stop()
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    with tempfile.TemporaryDirectory(prefix="fd-workloads-") as temp:
        scratch = Path(temp)
        # A root verifier drops the actor UID. Allow traversal to its snapshot;
        # private workload directories below remain owner-only and unmounted.
        scratch.chmod(0o711)
        stats, asset_env = prepare_assets(model_root, scratch)
        # No provider or Driver environment is forwarded to scientific code.
        env = {key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")
               if key in os.environ}
        env.update(asset_env, HOME=str(scratch), TMPDIR=str(scratch),
                   PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
        reference = scratch / "reference"
        shutil.copytree(TASK / "harbor/app", reference,
                        ignore=shutil.ignore_patterns(".git", "__pycache__"))
        for path in [reference, *reference.rglob("*")]:
            path.chmod(path.stat().st_mode | 0o200)
        env["GIT_CEILING_DIRECTORIES"] = str(scratch)
        runner.run(["git", "apply", "--whitespace=nowarn", str(TASK / "reference.patch")],
                   cwd=reference, env=env, log=scratch / "patch.log")
        admit_reference(reference)
        spine_manifest = scratch / "spine.json"
        spine_manifest.write_text(json.dumps({
            p.relative_to(reference).as_posix(): sha256(p)
            for p in reference.rglob("*") if p.is_file()}))
        snapshots = {}
        for side, source_repo in (("candidate", repo), ("reference", reference)):
            snapshots[side] = prepare_queue_snapshot(
                source_repo, reference, spine_manifest, scratch / (side + "-queue"))
            for path in [snapshots[side], *snapshots[side].rglob("*")]:
                path.chmod(0o755 if path.is_dir() else 0o444)
        rows = []
        for workload in recipe["suites"][suite]:
            row = {"id": workload["id"]}
            for side in ("candidate", "reference"):
                side_dir = scratch / (workload["id"] + "-" + side)
                side_dir.mkdir(mode=0o700)
                (side_dir / "data").mkdir()
                (side_dir / "data/fid_stats").symlink_to(stats, target_is_directory=True)
                queue_scratch = side_dir / "queue-workers"
                queue_scratch.mkdir()
                reports = {}
                for phase in ("train", "eval"):
                    report = side_dir / (phase + ".json")
                    checkpoint = (side_dir / "workload/train/checkpoints" /
                                  f"step_{workload['updates'] - 1:07d}.pth")
                    argv = [sys.executable, "-I", "-B", "-m", "torch.distributed.run", "--standalone",
                            "--local-addr=127.0.0.1", "--nproc-per-node=1",
                            str(boundary / "queue_observer_entry.py"), "--phase", phase, "--suite", suite,
                            "--trusted-repo", str(reference), "--spine-manifest", str(spine_manifest),
                            "--queue-source", str(snapshots[side]), "--queue-scratch", str(queue_scratch),
                            "--queue-python", sys.executable, "--observer", str(HERE / "observe.py"),
                            "--observer-sha256", sha256(HERE / "observe.py"),
                            "--seconds", str(max(1, int(runner.deadline - time.monotonic()))),
                            "--device", "cuda:0", "--report", str(report), "--"]
                    argv += arguments(workload, recipe, model_root, stats, side_dir,
                                      checkpoint=checkpoint if phase == "eval" else None)
                    log = side_dir / (phase + ".log")
                    started = time.monotonic()
                    journal.record("phase_started", workload=workload["id"], side=side, phase=phase)
                    try:
                        runner.run(argv, cwd=side_dir, env=env, log=log, scientific=True)
                        body = report_bytes(report)
                        reports[phase] = json.loads(body)
                        journal.record("phase_completed", workload=workload["id"], side=side, phase=phase,
                                       elapsed_seconds=time.monotonic() - started,
                                       report_sha256=sha256(report), report=reports[phase])
                    except BaseException as error:
                        tail = b""
                        if log.is_file():
                            with log.open("rb") as stream:
                                stream.seek(max(0, log.stat().st_size - 4096))
                                tail = stream.read(4096)
                        journal.record("phase_failed", workload=workload["id"], side=side, phase=phase,
                                       elapsed_seconds=time.monotonic() - started,
                                       error_type=type(error).__name__,
                                       report=json.loads(report_bytes(report)) if report.is_file() else None,
                                       untrusted_log_tail=tail.decode("utf-8", errors="replace"))
                        raise
                try:
                    row[side] = validate_report(reports["train"], reports["eval"], workload, recipe)
                except Exception as error:
                    journal.record("phase_failed", workload=workload["id"], side=side,
                                   phase="validation", error_type=type(error).__name__)
                    raise
                shutil.rmtree(side_dir)
            row["validity"] = {key: True for key in (
                "gpu", "source_lock", "model_lock", "finite_features", "finite_covariances",
                "finite_gradients", "parameter_update", "queue_state_roundtrip",
                "authentic_warm_start", "equal_sample_counts", "entrypoint_executed")}
            rows.append(row)
    journal.record("suite_completed", suite=suite)
    return {"workloads": rows, "admission": "isolated_queue_actor"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--suite", choices=("development", "final"), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    opts = parser.parse_args()
    try:
        print(json.dumps(execute(opts.repo.resolve(), opts.suite, opts.model_root.resolve())))
    except SetupDeadlineExpired:
        print(json.dumps({"verifier_failure": "verifier_setup_timeout"}))
        raise SystemExit(75)
