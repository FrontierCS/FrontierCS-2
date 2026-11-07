"""Trusted fresh-process measurement orchestration, disabled pending review."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from measurement_contract import assemble_measurements, schedule
from task_guards import CandidateTimeout, InvalidSubmission, TaskIntegrityError
from trusted_workload import write_json
import isolate

TASK = Path(__file__).resolve().parent


def _process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[19], fields[0]
    except (OSError, IndexError):
        return None


def _owned_descendants(pid):
    result = {}
    pending = [pid]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        try:
            children = [int(p) for p in Path(f"/proc/{current}/task/{current}/children").read_text().split()]
        except OSError:
            continue
        for child in children:
            identity = _process_identity(child)
            if identity is not None:
                result[child] = identity[0]
                pending.append(child)
    return result


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



def launch_episode(context, mode, case, output, snapshot=None):
    runtime = context["seal"]["runtime"]
    index = case["run"]["index"]
    request_path = output / f"request-{index:02d}.json"
    episode_dir = output / f"episode-{index:02d}"
    request = {"mode": mode, "case": case,
               "checkpoint": runtime["checkpoint"], "runtime_prefix": runtime["prefix"],
               "output": str(episode_dir), "submission_path": str(snapshot) if case["side"] == "candidate" else None,
               "sampler_seconds": 130}
    write_json(request_path, request)
    # PYTHONPATH: the episode process starts its sampler worker with the shared isolate.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONPATH=str(Path(isolate.__file__).resolve().parent))
    with (output / f"episode-{index:02d}.log").open("w") as log:
        try:
            returncode, elapsed = bounded_process([str(Path(runtime["prefix"]) / "bin/python"),
                str(TASK / "episode_worker.py"), str(request_path)], log, env, 150)
        except subprocess.TimeoutExpired as error:
            write_json(output / "failure.json", {"case": case, "reason": "process_timeout"})
            if case["side"] == "candidate":
                raise CandidateTimeout("A candidate episode process exceeded 150 s") from error
            raise TaskIntegrityError("A complete measurement process timed out") from error
    write_json(output / f"process-{index:02d}.json", {"case": case, "seconds": elapsed, "returncode": returncode})
    if returncode:
        write_json(output / "failure.json", {"case": case, "reason": "process_failure"})
        failure_path = episode_dir / "failure.json"
        failure = json.loads(failure_path.read_text()) if failure_path.is_file() else {}
        if case["side"] == "candidate" and failure.get("category") == "invalid_submission":
            raise InvalidSubmission("Candidate measurement did not complete")
        if case["side"] == "candidate" and failure.get("category") == "candidate_timeout":
            raise CandidateTimeout("The candidate sampler exceeded its 130 s lifetime")
        raise TaskIntegrityError("A trusted or unclassified measurement process did not complete")
    trajectory = json.loads((episode_dir / "trajectory.json").read_text())
    if trajectory["run"] != case["run"] or trajectory["execution_kind"] != case["side"]:
        raise TaskIntegrityError("Trajectory provenance mismatch")
    return {"case": case, "trajectory": trajectory, "complete_process_seconds": elapsed}


def measure_suite(config, solution_path, output, runtime_prefix, checkpoint, mode, *, context=None):
    from private_context import load_context
    context = context if context is not None else load_context()
    if (runtime_prefix != context["seal"]["runtime"]["prefix"]
            or checkpoint != context["seal"]["runtime"]["checkpoint"] or mode not in context["suites"]):
        raise TaskIntegrityError("Unbound runtime or mode")
    output = Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    # A read-only snapshot of the judged tree (the starter with the submission's allowed files),
    # which every candidate episode loads.
    snapshot = output / "submitted-tree"
    shutil.copytree(Path(solution_path), snapshot, symlinks=True)
    for path in sorted(snapshot.rglob("*"), reverse=True) + [snapshot]:
        if not path.is_symlink():
            path.chmod(0o500 if path.is_dir() else 0o400)
    deadline = time.monotonic() + config["evaluation"]["measurement_limits"][mode + "_seconds"]
    rows = []
    for case in schedule(context["suites"][mode]):
        if deadline - time.monotonic() < 155:
            raise CandidateTimeout("The suite measurement budget ran out")
        rows.append(launch_episode(context, mode, case, output, snapshot))
    result = assemble_measurements(context["suites"][mode], rows)
    write_json(output / "private-result.json", result)
    return result
