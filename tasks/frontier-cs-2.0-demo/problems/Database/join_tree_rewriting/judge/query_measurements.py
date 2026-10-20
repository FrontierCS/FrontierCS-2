"""Auditable query measurements; score and reference verdict are independent."""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


class MeasurementError(RuntimeError):
    """A reference, baseline, coverage or measurement configuration is invalid."""


CORRECTNESS_METHODS = ("BinaryJoin", "Reference", "Candidate")
TIMING_METHODS = ("BinaryA", "ReferenceA", "CandidateA", "CandidateB", "ReferenceB", "BinaryB")


def read_measurements(csv_path: Path, audit_path: Path, expected: set[str],
                      methods: tuple[str, ...], repetitions: int) -> dict:
    """Retain raw observations and exact-result checks, including partial failure.

    The engine flushes a start/completion ledger outside its measured interval.
    A successful invocation is validated separately by require_complete().
    Missing measurements are never silently removed from expected workloads.
    """
    records = {q: {"durations_us": {m: [] for m in methods},
                   "matches_original": {m: [] for m in methods}} for q in expected}
    if audit_path.stat().st_size > 1024 * 1024 or csv_path.stat().st_size > 1024 * 1024:
        raise MeasurementError("engine evidence exceeds the bounded capture size")
    pending = None
    next_position = {q: 0 for q in expected}
    with audit_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            q, m, rep = row.get("query"), row.get("method"), row.get("repetition")
            if q not in records or m not in methods or type(rep) is not int or not 0 <= rep < repetitions:
                raise MeasurementError("unexpected engine audit identity")
            identity = (q, m, rep)
            if row.get("state") == "started":
                if pending is not None or next_position[q] != rep * len(methods) + methods.index(m):
                    raise MeasurementError("duplicate or out-of-order engine observation")
                pending = identity
            elif row.get("state") == "completed":
                if pending != identity or type(row.get("matches_first")) is not bool:
                    raise MeasurementError("engine completion lacks its matching start")
                records[q]["matches_original"][m].append(row["matches_first"])
                next_position[q] += 1
                pending = None
            else:
                raise MeasurementError("unknown engine observation state")
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["duration(µs)", "method", "variant", "query"]:
            raise MeasurementError("unexpected timing CSV schema")
        for row in reader:
            q, m = row.get("query"), row.get("method")
            if q not in records or m not in methods or row.get("variant") != "flatten":
                raise MeasurementError("unexpected timing workload or method")
            value = float(row["duration(µs)"])
            if not math.isfinite(value) or value <= 0:
                raise MeasurementError("timing must be finite and positive")
            samples = records[q]["durations_us"][m]
            samples.append(value)
            if len(samples) > repetitions:
                raise MeasurementError("duplicate timing observation")
    # A failed process may stop between writing CSV and completing the audit.
    # Preserve those raw values, but never count them as a complete measurement.
    for record in records.values():
        for method in methods:
            n = len(record["durations_us"][method])
            completed = len(record["matches_original"][method])
            if n < completed or n > completed + 1:
                raise MeasurementError("timing and correctness observation counts disagree")
            if method.startswith(("Reference", "Binary")) and not all(record["matches_original"][method]):
                raise MeasurementError("reference or binary baseline changed the original result")
    return {"queries": records, "pending": pending}


def require_complete(measurements: dict, methods: tuple[str, ...], repetitions: int) -> None:
    if measurements["pending"] is not None:
        raise MeasurementError("successful engine left an incomplete observation")
    for record in measurements["queries"].values():
        for method in methods:
            if (len(record["durations_us"][method]) != repetitions
                    or len(record["matches_original"][method]) != repetitions):
                raise MeasurementError("successful engine omitted a required observation")


def smooth_score(speedup: float) -> float:
    if not math.isfinite(speedup) or speedup <= 0:
        raise MeasurementError("speedup must be finite and positive")
    if speedup <= 1:
        return 0.0
    decades = math.log10(speedup)
    return 100.0 * decades / (1.0 + decades)


def compare_query(*, identity: str, correctness: dict | None, timing: dict | None,
                  identical: bool, repetitions: int, failure: str = "") -> dict:
    result = {"workload": identity, "correctness_pass": False, "latency_pass": False,
              "passed": False, "byte_identical_reference": identical,
              "continuous_score": 0.0, "reference_continuous_score": 0.0,
              "status": failure or "measurement_incomplete"}
    if correctness is not None:
        result["correctness_measurements"] = correctness
    if timing is not None:
        result["timing_measurements"] = timing
    if failure or correctness is None or timing is None:
        return result
    for method in CORRECTNESS_METHODS:
        if (len(correctness["durations_us"][method]) != 1
                or len(correctness["matches_original"][method]) != 1):
            return result
        if method != "Candidate" and correctness["matches_original"][method] != [True]:
            raise MeasurementError("reference or binary correctness failure")
    for method in TIMING_METHODS:
        if (len(timing["durations_us"][method]) != repetitions
                or len(timing["matches_original"][method]) != repetitions):
            return result
    candidate_matches = timing["matches_original"]["CandidateA"] + timing["matches_original"]["CandidateB"]
    result["correctness_pass"] = correctness["matches_original"]["Candidate"] == [True] and all(candidate_matches)
    for role in ("Binary", "Reference", "Candidate"):
        samples = timing["durations_us"][role + "A"] + timing["durations_us"][role + "B"]
        result[role.lower() + "_median_us"] = statistics.median(samples)
    binary, reference, candidate = (result[name + "_median_us"] for name in ("binary", "reference", "candidate"))
    # Fresh measured reference median, never an aggregate score proxy/tolerance.
    result["latency_pass"] = not identical and candidate < reference
    result["passed"] = result["correctness_pass"] and result["latency_pass"]
    result["status"] = "measured" if result["correctness_pass"] else "incorrect_result"
    result["binary_speedup"] = binary / candidate
    result["reference_binary_speedup"] = binary / reference
    if result["correctness_pass"]:
        result["continuous_score"] = smooth_score(binary / candidate)
    result["reference_continuous_score"] = smooth_score(binary / reference)
    return result


def summarize_queries(expected: list[str], queries: dict[str, dict]) -> tuple[float, dict]:
    if not expected or len(set(expected)) != len(expected) or set(queries) != set(expected):
        raise MeasurementError("expected workload coverage changed")
    total = len(expected)
    score = sum(queries[q]["continuous_score"] for q in expected) / total
    metrics = {"beats_reference": int(all(queries[q]["passed"] for q in expected)),
               "expected_case_count": total,
               "case_count": sum(queries[q]["status"] in ("measured", "incorrect_result") for q in expected),
               "correctness_passed": sum(queries[q]["correctness_pass"] for q in expected),
               "latency_passed": sum(queries[q]["latency_pass"] for q in expected),
               "queries": queries, "duration_unit": "microseconds"}
    # Reference score is contextual only; a partial reference score is not made up.
    if all("reference_median_us" in queries[q] for q in expected):
        reference = sum(queries[q]["reference_continuous_score"] for q in expected) / total
        metrics.update(reference_score=reference, margin=score - reference)
    return score, metrics
