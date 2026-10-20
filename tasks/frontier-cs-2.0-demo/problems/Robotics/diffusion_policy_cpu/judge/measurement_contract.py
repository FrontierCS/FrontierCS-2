"""Private suite and raw-result contract; continuous scoring stays disabled."""

import hashlib
import json
import math
from pathlib import Path
import statistics

from task_guards import DEV_FIELDS, METRICS, TaskIntegrityError, public_feedback
from trusted_workload import validate_measurements


def require(condition, message):
    if not condition:
        raise TaskIntegrityError(message)


def load_private_suite(path, expected_sha256, mode):
    path = Path(path)
    require(mode in {"dev", "final"}, "Invalid trusted suite mode")
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 65536,
            "Private manifest is not a bounded regular file")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Private manifest binding mismatch")
    suite = json.loads(raw)
    return validate_suite(suite, mode)


def validate_suite(suite, mode):
    require(mode in {"dev", "final"}, "Invalid trusted suite mode")
    require(isinstance(suite, dict) and set(suite) == {"version", "mode", "repetitions", "groups"},
            "Invalid private manifest schema")
    require(type(suite["version"]) is int and suite["version"] == 1, "Invalid manifest version")
    require(suite["mode"] == mode and type(suite["repetitions"]) is int and suite["repetitions"] == 3,
            "Unbound suite mode or repetitions")
    groups = suite["groups"]
    require(isinstance(groups, list) and len(groups) == (2 if mode == "dev" else 4),
            "Private suite must retain the declared complete group set")
    identifiers, env_seeds, diffusion_seeds = set(), set(), set()
    for group in groups:
        require(set(group) == {"id", "visibility", "env_seed", "diffusion_seeds"}, "Invalid group schema")
        identifier = group["id"]
        require(isinstance(identifier, str) and identifier.isascii() and identifier.isalnum()
                and 1 <= len(identifier) <= 32 and identifier not in identifiers, "Invalid group identity")
        identifiers.add(identifier)
        require(group["visibility"] in {"public", "private"}, "Invalid group visibility")
        env_seed = group["env_seed"]
        require(type(env_seed) is int and 0 <= env_seed < 2**32 and env_seed not in env_seeds,
                "Initial conditions must be distinct bounded seeds")
        env_seeds.add(env_seed)
        seeds = group["diffusion_seeds"]
        require(isinstance(seeds, list) and len(seeds) == 3, "Missing prescribed repetition")
        for seed in seeds:
            require(type(seed) is int and 0 <= seed < 2**32 and seed not in diffusion_seeds,
                    "Diffusion RNG repetitions must be distinct bounded seeds")
            diffusion_seeds.add(seed)
    require(sum(g["visibility"] == "public" for g in groups) == 2, "Public group membership changed")
    return suite


def validate_final_extension(dev, final):
    require(dev["mode"] == "dev" and final["mode"] == "final", "Wrong suite pairing")
    public_dev = {g["id"]: g for g in dev["groups"]}
    public_final = {g["id"]: g for g in final["groups"] if g["visibility"] == "public"}
    require(set(public_dev) == set(public_final), "Final must freshly include every public group")
    for name, group in public_dev.items():
        require(group["env_seed"] == public_final[name]["env_seed"], "Final public initial condition changed")
    dev_seeds = {seed for group in dev["groups"] for seed in group["diffusion_seeds"]}
    final_seeds = {seed for group in final["groups"] for seed in group["diffusion_seeds"]}
    require(dev_seeds.isdisjoint(final_seeds), "Final diffusion seeds must be independent of development")


def schedule(suite):
    result = []
    for group_index, group in enumerate(suite["groups"]):
        for repetition, seed in enumerate(group["diffusion_seeds"]):
            order = ("reference", "candidate") if (group_index + repetition) % 2 == 0 else ("candidate", "reference")
            for side in order:
                result.append({"side": side, "group_id": group["id"], "run": {
                    "index": len(result), "group": group_index, "repetition": repetition,
                    "env_seed": group["env_seed"], "diffusion_seed": seed, "steps": 100}})
    return result


def strict_comparison(candidate, reference):
    require(set(candidate) == set(reference) and bool(candidate), "Incomplete workload comparison")
    progress = {name: {"passed": 0, "total": len(candidate)} for name in METRICS}
    workload_passes = 0
    audit_rows = []
    for group in candidate:
        require(set(candidate[group]) == set(reference[group]) == set(METRICS),
                "Workload metric membership changed")
        values = []
        metric_passes = {}
        for name in METRICS:
            a, b = candidate[group][name], reference[group][name]
            require(type(a) in (float, int) and type(b) in (float, int)
                    and math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0,
                    "Missing or nonfinite comparison measurement")
            if name in METRICS[:2]:
                require(a <= 1 and b <= 1, "Quality or success outside original range")
                passed = a >= b
            else:
                require(a > 0 and b > 0, "Timing must be strictly positive")
                passed = a < b
            progress[name]["passed"] += int(passed)
            values.append(passed)
            metric_passes[name] = passed
        workload_passes += int(all(values))
        audit_rows.append({"workload": group, "metrics": metric_passes})
    return {"beats_reference": int(workload_passes == len(candidate)),
            "beat_spec_workload_passes": audit_rows,
            "progress": progress, "workload_passes": workload_passes, "workload_total": len(candidate)}


def assemble_measurements(suite, rows):
    expected = schedule(suite)
    require(len(rows) == len(expected), "Complete candidate and reference measurements are required")
    by_index = {}
    for row in rows:
        index = row["case"]["run"]["index"]
        require(type(index) is int and 0 <= index < len(expected) and index not in by_index,
                "Missing or duplicated trajectory")
        require(row["case"] == expected[index], "Result does not match its private manifest")
        require(row["trajectory"]["run"] == expected[index]["run"], "Trajectory seed binding mismatch")
        require(row["trajectory"]["execution_kind"] == expected[index]["side"], "Execution role mismatch")
        wall = row["complete_process_seconds"]
        require(type(wall) in (int, float) and math.isfinite(wall) and wall > 0, "Invalid process duration")
        validate_measurements(row["trajectory"])
        by_index[index] = row
    measured = {"candidate": {}, "reference": {}}
    for side in measured:
        for group in suite["groups"]:
            selected = [row for row in rows if row["case"]["side"] == side and row["case"]["group_id"] == group["id"]]
            require(len(selected) == suite["repetitions"], "Missing group repetition")
            trajectories = [row["trajectory"] for row in selected]
            times = [t for trajectory in trajectories for t in trajectory["prediction_seconds"]]
            measured[side][group["id"]] = dict(zip(METRICS, (
                statistics.mean(max(t["rewards"]) for t in trajectories),
                statistics.mean(any(t["original_done"]) for t in trajectories),
                sum(times) / len(times),
                statistics.median(row["complete_process_seconds"] for row in selected),
            )))
    comparison = strict_comparison(measured["candidate"], measured["reference"])
    candidate_rows = [r for r in rows if r["case"]["side"] == "candidate"]
    all_times = [t for r in candidate_rows for t in r["trajectory"]["prediction_seconds"]]
    absolute = dict(zip(DEV_FIELDS, (
        statistics.mean(max(r["trajectory"]["rewards"]) for r in candidate_rows),
        statistics.mean(any(r["trajectory"]["original_done"]) for r in candidate_rows),
        sum(all_times) / len(all_times),
        sum(g[METRICS[3]] for g in measured["candidate"].values()),
    )))
    return {"score": None, "score_status": "awaiting_fresh_bound_calibration",
            "solver_feedback": public_feedback(suite["mode"], complete=True,
                                                candidate=absolute, progress=comparison["progress"]),
            "private": {"mode": suite["mode"], "measurements": measured, **comparison}}
