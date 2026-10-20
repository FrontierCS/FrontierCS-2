"""Reviewed Option B mathematics and fixture-only native result assembly."""

import hashlib
import json
import math
from pathlib import Path
import re

from measurement_contract import strict_comparison, validate_suite
from task_guards import DEV_FIELDS, METRICS, TaskIntegrityError, public_feedback

QUALITY, SUCCESS, LATENCY, WALL = METRICS
BASELINE_FIELDS = {QUALITY, LATENCY, WALL}
BINDING_FIELDS = {"source_sha256", "suite_sha256", "runtime_sha256",
                  "hardware_policy_sha256", "schedule_sha256", "instrumentation_sha256"}


class ScoreUnavailable(TaskIntegrityError):
    pass


def need(condition, reason):
    if not condition:
        raise ScoreUnavailable(reason)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def number(value, *, positive=False, quality=False, naive=False):
    need(type(value) in (int, float), "Measurement must be numeric")
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ScoreUnavailable("Measurement exceeds the finite domain") from error
    need(math.isfinite(value), "Measurement must be finite")
    need(value > 0 if positive else value >= 0, "Measurement outside its nonnegative domain")
    if quality:
        need(value < 1 if naive else value <= 1, "Quality anchor outside its physical domain")
    return value


def option_b_score(measurements, constants, workload_ids):
    """Pure mathematics; optimized-reference values are deliberately not inputs."""
    need(isinstance(measurements, dict) and isinstance(constants, dict), "Missing measurement or calibration map")
    need(isinstance(workload_ids, (list, tuple)) and bool(workload_ids)
         and all(isinstance(key, str) and key for key in workload_ids)
         and len(set(workload_ids)) == len(workload_ids), "Incomplete workload membership")
    need(set(measurements) == set(constants) == set(workload_ids), "Missing or extra scored workload")
    utilities = []
    for workload in workload_ids:
        observed, naive = measurements[workload], constants[workload]
        need(isinstance(observed, dict) and isinstance(naive, dict)
             and set(observed) == set(METRICS) and set(naive) == BASELINE_FIELDS,
             "Missing or extra measurement/constant field")
        quality = number(observed[QUALITY], quality=True)
        number(observed[SUCCESS], quality=True)
        latency, wall = number(observed[LATENCY], positive=True), number(observed[WALL], positive=True)
        naive_quality = number(naive[QUALITY], quality=True, naive=True)
        naive_latency, naive_wall = number(naive[LATENCY], positive=True), number(naive[WALL], positive=True)
        gain = max(0.0, (quality - naive_quality) / (1.0 - naive_quality))
        utility = 0.0 if gain == 0.0 else gain * min(naive_latency / latency, naive_wall / wall)
        need(math.isfinite(utility), "Utility exceeds finite arithmetic")
        utilities.append(utility)
    # Dividing before summation avoids overflow for a finite positive mean.
    unbounded = math.fsum(value / len(utilities) for value in utilities)
    need(math.isfinite(unbounded), "Mean utility exceeds finite arithmetic")
    display = 100.0 * (unbounded / (1.0 + unbounded))
    return {"display": display, "unbounded": unbounded}


def validate_calibration(payload, expected_binding, workload_ids, *, fixture_only=False):
    need(isinstance(payload, dict) and set(payload) == {"version", "purpose", "method", "binding", "constants"},
         "Invalid calibration schema")
    need(type(payload["version"]) is int and payload["version"] == 1, "Invalid calibration version")
    purpose = "synthetic_fixture" if fixture_only else "frozen_naive_calibration"
    need(payload["purpose"] == purpose and payload["method"] == "original_ddpm_20_steps", "Unbound naive method or purpose")
    need(isinstance(expected_binding, dict) and set(expected_binding) == BINDING_FIELDS, "Incomplete expected calibration binding")
    need(all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in expected_binding.values()),
         "Invalid expected calibration identity")
    need(payload["binding"] == expected_binding, "Calibration source/suite/runtime binding mismatch")
    need(isinstance(payload["constants"], dict) and set(payload["constants"]) == set(workload_ids),
         "Incomplete naive workload constants")
    # Validate every anchor even if a candidate has zero quality gain.
    for values in payload["constants"].values():
        need(isinstance(values, dict) and set(values) == BASELINE_FIELDS, "Invalid naive constant fields")
        number(values[QUALITY], quality=True, naive=True)
        number(values[LATENCY], positive=True)
        number(values[WALL], positive=True)
    return payload["constants"]


def load_calibration(path, expected_sha256, expected_binding, workload_ids, *, fixture_only=False):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 65536,
         "Calibration is missing or not a bounded regular file")
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == expected_sha256, "Calibration artifact hash mismatch")
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise ScoreUnavailable("Malformed calibration data") from error
    validate_calibration(payload, expected_binding, workload_ids, fixture_only=fixture_only)
    return payload


def unavailable(reason):
    return {"status": "unavailable", "reason": reason, "score": None,
            "unbounded": None, "native_result": None, "solver_feedback": {}}


def assemble_scores(measurement, *, baseline=None, expected_binding=None, suite=None, fixture_only=False, execution_context=None):
    if baseline is None or expected_binding is None or suite is None:
        return unavailable("fresh_bound_calibration_and_explicit_manifests_required")
    if not fixture_only and execution_context is None:
        return unavailable("real_score_execution_disabled")
    try:
        if not fixture_only:
            from private_context import accepted_calibrations, binding, load_context
            current = load_context(root=execution_context["root"], mode=execution_context["mode"])
            need(current["seal_sha256"] == execution_context["seal_sha256"], "Execution context changed")
            accepted = accepted_calibrations(current)
            need(suite == current["suites"][suite["mode"]] and baseline == accepted[suite["mode"]]
                 and expected_binding == binding(current, suite["mode"]), "Unaccepted score inputs")
        validate_suite(suite, suite["mode"])
        need(suite["mode"] in {"dev", "final"} and measurement["private"]["mode"] == suite["mode"],
             "Suite routing mismatch")
        workload_ids = [group["id"] for group in suite["groups"]]
        need(len(workload_ids) == (2 if suite["mode"] == "dev" else 4)
             and len(set(workload_ids)) == len(workload_ids), "Incomplete explicit suite")
        need(expected_binding["suite_sha256"] == canonical_sha256(suite), "Suite manifest binding mismatch")
        constants = validate_calibration(baseline, expected_binding, workload_ids, fixture_only=fixture_only)
        measured = measurement["private"]["measurements"]
        candidate = option_b_score(measured["candidate"], constants, workload_ids)
        reference = option_b_score(measured["reference"], constants, workload_ids)
        comparison = strict_comparison(measured["candidate"], measured["reference"])
        feedback = public_feedback(suite["mode"], complete=True,
                                   candidate=measurement["solver_feedback"], progress=comparison["progress"])
        metrics = {**comparison, "reference_score": reference["display"],
                   "margin": candidate["display"] - reference["display"],
                   "private_measurements": measured, "measurement_binding": expected_binding}
        metrics.update({key: feedback[key] for key in DEV_FIELDS if key in feedback})
        message = "Synthetic fixture only; real evaluation is disabled." if fixture_only else "Complete frozen-suite measurement."
        return {"status": "synthetic_fixture_only" if fixture_only else "complete", "score": candidate["display"],
                "unbounded": candidate["unbounded"], "private_metrics": metrics,
                "solver_feedback": feedback,
                "native_result": (candidate["display"], candidate["unbounded"],
                                  message, metrics)}
    except (ScoreUnavailable, TaskIntegrityError, KeyError, TypeError, ValueError):
        return unavailable("invalid_or_incomplete_bound_measurements")
