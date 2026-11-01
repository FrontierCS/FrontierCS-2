"""Verifier-owned source/runtime bindings; private data is not part of sampler RPC."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys

from hardware_admission import admits
from measurement_contract import load_private_suite, schedule, validate_final_extension
from option_b import canonical_sha256, load_calibration
from task_guards import TaskIntegrityError

TASK = Path(__file__).resolve().parent
PRIVATE = TASK.parent / "private"


def require(condition, message):
    if not condition:
        raise TaskIntegrityError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        require(not path.is_symlink(), "Task source contains a link")
        if path.is_file():
            result[str(path.relative_to(root))] = {
                "sha256": file_sha256(path), "mode": stat.S_IMODE(path.stat().st_mode)}
    return result


def private_json(path):
    from native_data import strict_json
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 4 * 1024**2,
            "Missing or invalid private artifact")
    require(path.stat().st_mode & 0o077 == 0, "Private artifact permissions are too broad")
    return strict_json(path.read_text())


def baseline_schedule(suite):
    return [{"side": "baseline", "group_id": group["id"], "run": {
        "index": group_index * 3 + repetition, "group": group_index,
        "repetition": repetition, "env_seed": group["env_seed"],
        "diffusion_seed": seed, "steps": 20}}
        for group_index, group in enumerate(suite["groups"])
        for repetition, seed in enumerate(group["diffusion_seeds"])]


def runtime_identity(prefix=None):
    from runtime_probe import identity
    if prefix is None or Path(prefix).resolve() == Path(sys.prefix).resolve():
        return identity()
    require(Path(prefix).is_absolute(), "Measurement interpreter prefix must be absolute")
    executable = Path(prefix) / "bin/python"
    require(executable.is_file(), "Declared measurement interpreter is missing")
    env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        result = subprocess.run([str(executable), "-I", str(TASK / "runtime_probe.py")],
                                env=env, capture_output=True, timeout=15, check=True)
        require(len(result.stdout) <= 256 * 1024, "Measurement identity exceeds its bound")
        value = json.loads(result.stdout)
        require(isinstance(value, dict) and set(value) == {"python", "executable_sha256", "base_prefix", "distributions"},
                "Malformed measurement interpreter identity")
        return value
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise TaskIntegrityError("Measurement interpreter identity is unavailable") from error


def runtime_files(roots):
    result = {}
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_file():
                result[str(path)] = file_sha256(path)
    return result


def verify_runtime_files(context):
    runtime = context["seal"]["runtime"]
    require(canonical_sha256(runtime_files(runtime["closure_roots"])) == runtime["closure_sha256"],
            "Runtime dependency bytes changed")
    require(file_sha256(runtime["checkpoint"]) == runtime["checkpoint_sha256"], "Checkpoint changed")


def hardware_identity():
    models = {line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
              if line.startswith("model name")}
    return {"architecture": platform.machine(), "kernel": platform.release(),
            "cpu_models": sorted(models), "affinity": sorted(os.sched_getaffinity(0))}


def verify_cgroup(policy):
    membership = Path("/proc/self/cgroup").read_text().strip()
    require(membership.startswith("0::/"), "Unified cgroup enforcement is required")
    group = Path("/sys/fs/cgroup") / membership[3:].lstrip("/")
    cpu = (group / "cpu.max").read_text().split()
    memory = (group / "memory.max").read_text().strip()
    require(cpu[0] != "max" and int(cpu[0]) / int(cpu[1]) <= policy["cpus"],
            "Aggregate CPU limit is absent or too large")
    require(memory != "max" and int(memory) <= policy["memory_bytes"],
            "Aggregate memory limit is absent or too large")
    require(len(os.sched_getaffinity(0)) <= policy["cpus"], "CPU affinity exceeds policy")


def load_context(*, root=None, mode=None, verify_runtime=True):
    root = PRIVATE if root is None else Path(root)
    require(root.is_dir() and not root.is_symlink()
            and root.stat().st_mode & 0o077 == 0, "Private context is absent or exposed")
    seal = private_json(root / "seal.json")
    require(seal["version"] == 1, "Unknown private seal")
    require(canonical_sha256(inventory(TASK)) == seal["source_sha256"], "Sealed task source changed")
    require(mode is None or mode in {"dev", "final"}, "Unknown selected private mode")
    selected = ("dev", "final") if mode is None else (mode,)
    suites = {name: load_private_suite(root / (name + ".json"), seal["suites"][name]["file_sha256"], name)
              for name in selected}
    if mode is None:
        validate_final_extension(suites["dev"], suites["final"])
    else:
        descriptor = private_json(root / "context.json")
        require(descriptor["mode"] == mode and descriptor["purpose"] == "accepted_evaluation",
                "Selected context is not a native evaluation dataset")
        require(set(descriptor["files"]) == {"seal.json", mode + ".json", mode + "-constants.json",
                mode + "-reference.json", "reference-acceptance.json"}, "Selected context membership changed")
        for name, expected in descriptor["files"].items():
            require(file_sha256(root / name) == expected, "Materialized context changed")
    for name, suite in suites.items():
        require(canonical_sha256(suite) == seal["suites"][name]["suite_sha256"], "Suite identity changed")
        require(canonical_sha256({"baseline": baseline_schedule(suite), "evaluation": schedule(suite)})
                == seal["suites"][name]["schedule_sha256"], "Measurement schedule changed")
    if verify_runtime:
        require(canonical_sha256(runtime_identity(seal["runtime"]["prefix"])) == seal["runtime"]["identity_sha256"], "Runtime identity changed")
        require(admits(seal["hardware_policy"]["identity"], hardware_identity()),
                "Hardware is outside the sealed hardware class")
        verify_cgroup(seal["hardware_policy"])
    return {"seal": seal, "seal_sha256": file_sha256(root / "seal.json"), "suites": suites,
            "root": str(root), "mode": mode, "native_dataset": mode is not None}


def binding(context, mode):
    seal = context["seal"]
    return {"source_sha256": seal["source_sha256"],
            "suite_sha256": seal["suites"][mode]["suite_sha256"],
            "runtime_sha256": canonical_sha256(seal["runtime"]),
            "hardware_policy_sha256": canonical_sha256(seal["hardware_policy"]),
            "schedule_sha256": seal["suites"][mode]["schedule_sha256"],
            "instrumentation_sha256": canonical_sha256(seal["instrumentation"])}


def authorize(context, profile):
    require(profile in {"baseline", "reference_acceptance", "evaluation"}, "Unknown trusted execution profile")
    if profile == "evaluation":
        require(context.get("native_dataset") is True, "Production evaluation requires the selected native dataset")
        accepted_calibrations(context)
        return {"profile": profile, "authority": "accepted_bound_native_dataset"}
    root = Path(context.get("root", PRIVATE))
    receipt = private_json(root / (profile + "-authorization.json"))
    require(receipt["seal_sha256"] == context["seal_sha256"] and receipt["profile"] == profile,
            "Authorization belongs to another version/profile")
    require(datetime.now(timezone.utc) < datetime.fromisoformat(receipt["work_cutoff"]), "Execution authorization expired")
    if profile == "baseline":
        require(receipt["max_trajectories"] == 18 and receipt["steps"] == 20,
                "Baseline grant must prescribe exactly 18 natural 20-step trajectories")
    return receipt


def accepted_calibrations(context):
    root = Path(context.get("root", PRIVATE))
    receipt = private_json(root / "reference-acceptance.json")
    require(receipt["seal_sha256"] == context["seal_sha256"] and receipt["status"] == "accepted",
            "Exact-version reference acceptance is required")
    require(set(receipt["modes"]) == set(context["suites"]), "Every selected reference suite must be accepted")
    result = {}
    for mode in context["suites"]:
        item = receipt["modes"][mode]
        require(item["binding"] == binding(context, mode) and item["complete_trajectories"] == len(baseline_schedule(context["suites"][mode]))
                and item["steps"] == 100, "Incomplete or mismatched reference acceptance")
        evidence = root / (mode + "-reference.json")
        require(file_sha256(evidence) == item["reference_sha256"], "Reference evidence changed")
        result[mode] = load_calibration(root / (mode + "-constants.json"), item["calibration_sha256"],
            binding(context, mode), [g["id"] for g in context["suites"][mode]["groups"]])
    return result


def validate_request(context, request):
    require(request["seal_sha256"] == context["seal_sha256"], "Worker request seal mismatch")
    authorize(context, request["profile"])
    mode = request["mode"]
    require(mode in context["suites"], "Unknown trusted suite mode")
    if request["profile"] == "baseline":
        cases = baseline_schedule(context["suites"][mode])
    else:
        cases = schedule(context["suites"][mode])
        if request["profile"] == "reference_acceptance":
            cases = [c for c in cases if c["side"] == "reference"]
    require(request["case"] in cases, "Worker case is outside the frozen schedule")
    require(bool(request["submission_path"]) == (request["case"]["side"] == "candidate"), "Submission role mismatch")
    if request["submission_path"]:
        require(file_sha256(request["submission_path"]) == request["patch_sha256"], "Submitted patch changed")
    require(request["checkpoint"] == context["seal"]["runtime"]["checkpoint"]
            and request["runtime_prefix"] == context["seal"]["runtime"]["prefix"]
            and request["sampler_seconds"] == 130, "Worker runtime/budget mismatch")
