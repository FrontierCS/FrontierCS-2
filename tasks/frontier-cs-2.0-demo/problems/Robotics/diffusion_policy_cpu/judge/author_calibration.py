"""Author-only native collection using the task's existing measurement functions.

This fixed Verifier command produces baseline constants and raw reference
acceptance, not a solver score or a substitute for reference train/final.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

TASK = Path(__file__).resolve().parent


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o400)


def collect(specification, output):
    import yaml
    from baseline_calibration import constants_from_rows
    from measurement_contract import schedule, validate_suite, validate_final_extension
    from measurement_runtime import launch_episode
    from option_b import canonical_sha256
    from private_context import (accepted_calibrations, baseline_schedule, binding,
        file_sha256, hardware_identity, inventory, load_context, require,
        runtime_files, runtime_identity, verify_cgroup, verify_runtime_files)
    from task_guards import METRICS
    from trusted_workload import tensor_state_sha256, validate_measurements, write_json

    require(set(specification) == {"schema", "purpose", "mode", "suites", "maximum_seconds"},
            "Unexpected author collection input")
    require(specification["schema"] == "frontier.diffusion-author-calibration/v1"
            and specification["purpose"] == "fresh_baseline_and_raw_reference"
            and specification["mode"] == "eval", "Wrong author collection purpose")
    require(type(specification["maximum_seconds"]) is int
            and 300 <= specification["maximum_seconds"] <= 3600, "Invalid collection time bound")
    require(set(specification["suites"]) == {"dev", "final"}, "Both private suites are required")
    suites = specification["suites"]
    for mode in suites:
        validate_suite(suites[mode], mode)
    validate_final_extension(suites["dev"], suites["final"])
    deadline = datetime.now(timezone.utc) + timedelta(seconds=specification["maximum_seconds"])
    private = output / "private"
    private.mkdir(mode=0o700)
    declarations = yaml.safe_load((TASK / "config.yaml").read_text())["runtime"]["measurement"]
    identity = runtime_identity(declarations["prefix"])
    require(identity["python"].startswith("3.10.20 ")
            and dict(identity["distributions"])["torch"] == "2.2.2+cpu", "Wrong numeric runtime")
    require(file_sha256(declarations["checkpoint"]) == declarations["checkpoint_sha256"],
            "Checkpoint bytes changed")
    hardware = {"cpus": 2, "memory_bytes": 4 * 1024**3, "identity": hardware_identity(),
                "aggregate_enforcement": "platform_cgroup_v2",
                "process_address_space_bytes": 4 * 1024**3, "threads": 1}
    verify_cgroup(hardware)
    import torch, dill
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    payload = torch.load(declarations["checkpoint"], map_location="cpu", pickle_module=dill)
    require(payload["cfg"].training.use_ema, "Expected frozen EMA checkpoint")
    ema = tensor_state_sha256(payload["state_dicts"]["ema_model"])
    require(ema == declarations["ema_sha256"] and torch.version.cuda is None,
            "EMA or CPU-only runtime differs")
    del payload
    roots = [declarations["prefix"], identity["base_prefix"]]
    closure = runtime_files(roots)
    source = inventory(TASK)
    runtime = {"prefix": declarations["prefix"], "checkpoint": declarations["checkpoint"],
               "checkpoint_sha256": declarations["checkpoint_sha256"],
               "ema_selector": "state_dicts.ema_model", "ema_sha256": ema,
               "identity_sha256": canonical_sha256(identity), "closure_roots": roots,
               "closure_sha256": canonical_sha256(closure), "closure_file_count": len(closure),
               "torch": torch.__version__, "cuda_build": torch.version.cuda}
    write(output / "runtime-identity.json", identity)
    write(output / "source-inventory.json", source)
    write(output / "runtime-files.json", closure)
    suite_bindings = {}
    for mode in suites:
        write(private / (mode + ".json"), suites[mode])
        suite_bindings[mode] = {"file_sha256": file_sha256(private / (mode + ".json")),
            "suite_sha256": canonical_sha256(suites[mode]),
            "schedule_sha256": canonical_sha256({"baseline": baseline_schedule(suites[mode]),
                                                   "evaluation": schedule(suites[mode])})}
    instrumentation = {
        "complete_process": "measurement_runtime.bounded_process: monotonic before Popen through termination and observed-descendant cleanup",
        "prediction": "trusted_workload.execute_episode around original policy.predict_action",
        "episode_seconds": 150, "sampler_seconds": 130, "max_physics_steps": 300,
        "repetitions": 3, "quality": "mean original episode-maximum reward",
        "prediction_latency": "sum prediction_seconds / actual prediction calls",
        "process_wall": "median fresh complete-process elapsed across three repetitions",
        "baseline": "natural original DDPM with num_inference_steps=20",
        "reference": "original DDPM with num_inference_steps=100",
        "execution_policy": "standard_platform_sandbox_with_post_trial_review",
        "measurement_files": {name: source[name]["sha256"] for name in (
            "measurement_runtime.py", "episode_worker.py", "trusted_workload.py", "private_context.py",
            "baseline_calibration.py", "sampler_bridge.py", "sampler_worker.py",
            "bridge_protocol.py", "frozen_model_service.py", "author_calibration.py")}}
    seal = {"version": 1, "sealed_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": canonical_sha256(source), "suites": suite_bindings,
            "runtime": runtime, "hardware_policy": hardware, "instrumentation": instrumentation}
    write(private / "seal.json", seal)
    context = load_context(root=private)
    for profile, steps in (("baseline", 20), ("reference_acceptance", 100)):
        write(private / (profile + "-authorization.json"), {
            "seal_sha256": context["seal_sha256"], "profile": profile,
            "authority": "fixed author-only native command under finite owner allocation",
            "work_cutoff": deadline.isoformat(), "max_trajectories": 18, "steps": steps})
    completed = {}
    for profile in ("baseline", "reference_acceptance"):
        for mode in ("dev", "final"):
            suite = suites[mode]
            cases = (baseline_schedule(suite) if profile == "baseline"
                     else [case for case in schedule(suite) if case["side"] == "reference"])
            require(len(cases) == (6 if mode == "dev" else 12), "Wrong trajectory membership")
            target = output / profile / mode
            target.mkdir(mode=0o700, parents=True)
            rows = []
            for case in cases:
                require((deadline - datetime.now(timezone.utc)).total_seconds() >= 155,
                        "Insufficient collection time for a complete prescribed trajectory")
                row = launch_episode(context, profile, mode, case, target)
                validate_measurements(row["trajectory"])
                require(math.isfinite(row["complete_process_seconds"])
                        and row["complete_process_seconds"] > 0, "Invalid process measurement")
                rows.append(row)
                write_json(target / "rows.json", rows)
                print(json.dumps({"profile": profile, "mode": mode, "completed": len(rows),
                                  "process_seconds": row["complete_process_seconds"]}), flush=True)
            if profile == "baseline":
                write(private / (mode + "-constants.json"),
                      constants_from_rows(suite, rows, binding(context, mode)))
                continue
            measured = {}
            for group in suite["groups"]:
                selected = [row for row in rows if row["case"]["group_id"] == group["id"]]
                require(len(selected) == 3, "Missing reference repetition")
                times = [t for row in selected for t in row["trajectory"]["prediction_seconds"]]
                measured[group["id"]] = dict(zip(METRICS, (
                    statistics.mean(max(row["trajectory"]["rewards"]) for row in selected),
                    statistics.mean(any(row["trajectory"]["original_done"]) for row in selected),
                    sum(times) / len(times),
                    statistics.median(row["complete_process_seconds"] for row in selected))))
                require(all(math.isfinite(v) for v in measured[group["id"]].values()),
                        "Nonfinite reference metric")
            reference = private / (mode + "-reference.json")
            write(reference, {"version": 1, "purpose": "raw_original_reference_acceptance",
                "mode": mode, "steps": 100, "binding": binding(context, mode),
                "complete_trajectories": len(rows), "measurements": measured,
                "rows_sha256": file_sha256(target / "rows.json"), "actual_task_score": None, "margin": None})
            completed[mode] = {"binding": binding(context, mode), "complete_trajectories": len(rows),
                "steps": 100, "reference_sha256": file_sha256(reference),
                "calibration_sha256": file_sha256(private / (mode + "-constants.json"))}
    require(load_context(root=private)["seal_sha256"] == context["seal_sha256"], "Source seal changed")
    verify_runtime_files(context)
    write(private / "reference-acceptance.json", {
        "version": 1, "status": "accepted", "purpose": "native_raw_reference_acceptance",
        "at": datetime.now(timezone.utc).isoformat(), "seal_sha256": context["seal_sha256"],
        "modes": completed, "formal_protocol_evaluation": False, "real_task_scores_computed": False})
    accepted_calibrations(context)
    return {"source_sha256": seal["source_sha256"], "seal_sha256": context["seal_sha256"],
            "baseline_trajectories": 18, "reference_trajectories": 18,
            "formal_reference_train_final": False, "solver_trial": False}


def main():
    os.umask(0o077)
    sys.dont_write_bytecode = True
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    output = Path("/frontier/author-evidence")
    output.mkdir(mode=0o700)
    result_path = Path(os.environ["FRONTIER_RESULT_PATH"])
    evidence_path = Path(os.environ["FRONTIER_EVIDENCE_PATH"])
    assert result_path == Path("/frontier/output/result.json")
    assert evidence_path == Path("/frontier/output/evidence.tar.zst")
    started = time.monotonic()
    passed = False
    record = {"purpose": "fresh_native_calibration", "solver_trial": False}
    try:
        assert os.environ["FRONTIER_VERIFICATION_MODE"] == "eval"
        payload = Path(os.environ["FRONTIER_PAYLOAD_PATH"])
        assert payload.stat().st_size <= 16384
        dataset = Path(os.environ["FRONTIER_DATASET_PATH"])
        assert dataset.stat().st_size <= 65536
        from author_protocol import validate_payload
        specification = validate_payload(payload.read_bytes(), dataset.read_bytes())
        record["collection"] = collect(specification, output)
        passed = True
    except BaseException as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)[:4096],
                             "traceback": traceback.format_exc()[-16384:]}
    record.update(passed=passed, duration_seconds=time.monotonic() - started)
    write(output / "collection-result.json", record)
    # The wrapper uploads this operator evidence, including partial failure rows.
    # Reference/private measurements are never placed in public result metrics.
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    # Compression belongs to the platform runtime; do not modify the sealed
    # numerical environment to add packaging dependencies.
    subprocess.run([
        "/opt/frontier/runtime-python", "-I", "-B", "-c",
        "from pathlib import Path; import sys, tarfile, zstandard\n"
        "source, destination = map(Path, sys.argv[1:])\n"
        "with destination.open('xb') as raw:\n"
        " with zstandard.ZstdCompressor(level=3).stream_writer(raw) as compressed:\n"
        "  with tarfile.open(fileobj=compressed, mode='w|') as archive:\n"
        "   for path in sorted(source.rglob('*')):\n"
        "    if path.is_file() and not path.is_symlink():\n"
        "     archive.add(path, arcname=str(path.relative_to(source)), recursive=False)\n",
        str(output), str(evidence_path),
    ], check=True, timeout=120)
    result = {"status": "passed" if passed else "failed",
              "summary": "Author calibration " + ("completed" if passed else "failed"),
              "metrics": {"author_calibration_complete": int(passed), "solver_trial": 0}}
    if passed:
        result["raw_score"] = 0
    else:
        result["failure"] = {"code": "author_calibration_failed",
                             "message": "See private author collection evidence", "retryable": False}
    result_path.write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
