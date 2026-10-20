"""Trusted fresh-process measurement orchestration, disabled pending review."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from measurement_contract import assemble_measurements, load_private_suite, schedule, validate_final_extension
from task_guards import InvalidSubmission, MAX_PATCH_BYTES, TaskIntegrityError
from trusted_workload import write_json
from sampler_bridge import _owned_descendants, _process_identity

TASK = Path(__file__).resolve().parent


def bounded_process(command, log, env, timeout):
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               env=env, start_new_session=True)
    descendants = {}
    deadline = started + timeout
    timed_out = False
    try:
        while process.poll() is None:
            descendants.update(_owned_descendants(process.pid))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                process.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                pass
    finally:
        if process.poll() is None:
            descendants.update(_owned_descendants(process.pid))
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid, birth in descendants.items():
            try:
                fd = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            try:
                identity = _process_identity(pid)
                if identity is not None and identity[0] == birth:
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                os.close(fd)
        process.wait(timeout=5)
        cleanup_deadline = time.monotonic() + 5
        while any((identity := _process_identity(pid)) is not None
                  and identity[0] == birth and identity[1] != "Z" for pid, birth in descendants.items()):
            if time.monotonic() >= cleanup_deadline:
                raise TaskIntegrityError("Owned measurement descendants did not terminate")
            time.sleep(0.01)
    if timed_out:
        raise subprocess.TimeoutExpired(command, timeout)
    return process.returncode, time.monotonic() - started



def launch_episode(context, profile, mode, case, output, snapshot=None):
    from private_context import authorize, load_context, validate_request
    authorize(context, profile)
    require_context = load_context(root=context["root"], mode=context["mode"])
    if require_context["seal_sha256"] != context["seal_sha256"]:
        raise TaskIntegrityError("Context changed during measurement")
    runtime = context["seal"]["runtime"]
    index = case["run"]["index"]
    request_path = output / f"request-{index:02d}.json"
    episode_dir = output / f"episode-{index:02d}"
    request = {"profile": profile, "mode": mode, "case": case,
               "seal_sha256": context["seal_sha256"],
               "private_context": context["root"],
               "checkpoint": runtime["checkpoint"], "runtime_prefix": runtime["prefix"],
               "output": str(episode_dir), "submission_path": str(snapshot) if case["side"] == "candidate" else None,
               "sampler_seconds": 130,
               "patch_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest() if snapshot else None}
    validate_request(context, request)
    write_json(request_path, request)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (output / f"episode-{index:02d}.log").open("w") as log:
        try:
            returncode, elapsed = bounded_process([str(Path(runtime["prefix"]) / "bin/python"),
                str(TASK / "episode_worker.py"), str(request_path)], log, env, 150)
        except subprocess.TimeoutExpired as error:
            write_json(output / "failure.json", {"case": case, "reason": "process_timeout"})
            raise TaskIntegrityError("A complete measurement process timed out") from error
    write_json(output / f"process-{index:02d}.json", {"case": case, "seconds": elapsed, "returncode": returncode})
    if returncode:
        write_json(output / "failure.json", {"case": case, "reason": "process_failure"})
        failure_path = episode_dir / "failure.json"
        failure = json.loads(failure_path.read_text()) if failure_path.is_file() else {}
        if case["side"] == "candidate" and failure.get("category") == "invalid_submission":
            raise InvalidSubmission("Candidate measurement did not complete")
        raise TaskIntegrityError("A trusted or unclassified measurement process did not complete")
    if load_context(root=context["root"], mode=context["mode"])["seal_sha256"] != context["seal_sha256"]:
        raise TaskIntegrityError("Context changed during a trajectory")
    trajectory = json.loads((episode_dir / "trajectory.json").read_text())
    if (trajectory["run"] != case["run"] or trajectory["execution_kind"] != case["side"]
            or trajectory["ema_sha256"] != runtime["ema_sha256"]):
        raise TaskIntegrityError("Trajectory provenance mismatch")
    return {"case": case, "trajectory": trajectory, "complete_process_seconds": elapsed}


def measure_suite(config, solution_path, output, runtime_prefix, checkpoint, mode, *, context=None):
    from private_context import authorize, load_context, verify_runtime_files
    context = context if context is not None else load_context()
    authorize(context, "evaluation")
    verify_runtime_files(context)
    if (runtime_prefix != context["seal"]["runtime"]["prefix"]
            or checkpoint != context["seal"]["runtime"]["checkpoint"] or mode not in context["suites"]):
        raise TaskIntegrityError("Unbound runtime or mode")
    output = Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    patch = Path(solution_path)
    if patch.is_symlink() or not patch.is_file() or not 0 < patch.stat().st_size <= MAX_PATCH_BYTES:
        raise InvalidSubmission("Provide one bounded regular patch file")
    snapshot = output / "submitted.patch"
    snapshot.write_bytes(patch.read_bytes())
    snapshot.chmod(0o400)
    from evaluator import prepared_submission
    with prepared_submission(snapshot, config):
        pass
    deadline = time.monotonic() + config["evaluation"]["measurement_limits"][mode + "_seconds"]
    rows = []
    for case in schedule(context["suites"][mode]):
        if deadline - time.monotonic() < 155:
            raise TaskIntegrityError("Insufficient suite time for a complete trajectory")
        rows.append(launch_episode(context, "evaluation", mode, case, output, snapshot))
    result = assemble_measurements(context["suites"][mode], rows)
    verify_runtime_files(context)
    write_json(output / "private-result.json", result)
    return result
