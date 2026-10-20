"""Single-use, author-only baseline collection through the measurement worker."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics

from measurement_runtime import launch_episode
from option_b import LATENCY, QUALITY, WALL, validate_calibration
from private_context import (PRIVATE, authorize, baseline_schedule, binding, file_sha256,
                             load_context, require, verify_runtime_files)
from trusted_workload import validate_measurements, write_json


def constants_from_rows(suite, rows, expected_binding):
    cases = baseline_schedule(suite)
    require(len(rows) == len(cases), "Every prescribed baseline trajectory is required")
    for case, row in zip(cases, rows):
        require(row["case"] == case and row["trajectory"]["run"] == case["run"]
                and row["trajectory"]["execution_kind"] == "baseline", "Baseline order or identity changed")
        seconds = row["complete_process_seconds"]
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0, "Invalid process timing")
        validate_measurements(row["trajectory"])
    constants = {}
    for group in suite["groups"]:
        selected = [r for r in rows if r["case"]["group_id"] == group["id"]]
        require(len(selected) == 3, "Missing baseline repetition")
        times = [t for r in selected for t in r["trajectory"]["prediction_seconds"]]
        constants[group["id"]] = {
            QUALITY: statistics.mean(max(r["trajectory"]["rewards"]) for r in selected),
            LATENCY: sum(times) / len(times),
            WALL: statistics.median(r["complete_process_seconds"] for r in selected)}
    payload = {"version": 1, "purpose": "frozen_naive_calibration", "method": "original_ddpm_20_steps",
               "binding": expected_binding, "constants": constants}
    validate_calibration(payload, expected_binding, [g["id"] for g in suite["groups"]])
    return payload


def main():
    os.umask(0o077)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    context = load_context()
    grant = authorize(context, "baseline")
    verify_runtime_files(context)
    output = PRIVATE / "baseline-runs"
    output.mkdir(mode=0o700, exist_ok=False)
    write_json(output / "started.json", {"seal_sha256": context["seal_sha256"], "grant": grant,
               "at": datetime.now(timezone.utc).isoformat()})
    completed = {}
    try:
        for mode in ("dev", "final"):
            mode_dir = output / mode
            mode_dir.mkdir(mode=0o700)
            rows = []
            for case in baseline_schedule(context["suites"][mode]):
                remaining = (datetime.fromisoformat(grant["work_cutoff"]) - datetime.now(timezone.utc)).total_seconds()
                require(remaining >= 155, "Insufficient authorized time for a complete trajectory")
                rows.append(launch_episode(context, "baseline", mode, case, mode_dir))
                write_json(mode_dir / "rows.json", rows)
                print(json.dumps({"mode": mode, "completed": len(rows), "case": case,
                                  "raw_metrics": rows[-1]["trajectory"]["raw_metrics"],
                                  "process_seconds": rows[-1]["complete_process_seconds"]}), flush=True)
                # Reject an invalid group anchor as soon as its three observations exist.
                if len(rows) % 3 == 0:
                    qn = statistics.mean(max(r["trajectory"]["rewards"]) for r in rows[-3:])
                    require(qn < 1, "Naive quality anchor must be strictly below one")
            payload = constants_from_rows(context["suites"][mode], rows, binding(context, mode))
            target = PRIVATE / (mode + "-constants.json")
            with target.open("x") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")
            target.chmod(0o400)
            completed[mode] = {"sha256": file_sha256(target), "trajectories": len(rows)}
        require(load_context()["seal_sha256"] == context["seal_sha256"], "Seal changed")
        verify_runtime_files(context)
        write_json(output / "completed.json", {"at": datetime.now(timezone.utc).isoformat(),
                   "seal_sha256": context["seal_sha256"], "artifacts": completed})
    except BaseException as error:
        write_json(output / "stopped.json", {"at": datetime.now(timezone.utc).isoformat(),
                   "type": type(error).__name__, "error": str(error), "completed": completed})
        raise


if __name__ == "__main__":
    main()
