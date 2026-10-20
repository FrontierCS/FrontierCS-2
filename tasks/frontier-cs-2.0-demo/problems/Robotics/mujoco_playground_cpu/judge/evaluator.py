from __future__ import annotations

import ctypes
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

from submission_patch import apply_submission_patch
from judge_deps.private_evidence import read as read_private_evidence

TASK_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((TASK_DIR / "config.yaml").read_text())
RUNNER = TASK_DIR / "judge_deps" / "run_workload.py"
# Every tunable number of the protocol is in judge_deps/workloads.json.
RECIPE = json.loads((TASK_DIR / "judge_deps" / "workloads.json").read_text())
# All five are "higher is better". The first three are per-episode sums until
# the first termination (mean over environments); energy and action_rate are
# per-alive-step means. See judge_deps/evaluate_policy.py.
SPEC_METRICS = ("episode_reward", "tracking_lin_vel", "tracking_ang_vel", "energy", "action_rate")
VISIBLE = tuple(f"development_mean_{name}" for name in SPEC_METRICS)
TRAIN_BUDGET_SECONDS = RECIPE["train_budget_seconds"]
# Challenge constant, independent of any reference measurement: the mean summed
# upstream episode reward that scores 50. The score has no finite ceiling.
SCORE_HALF_SATURATION_REWARD = RECIPE["score_half_saturation_reward"]
GUARDS = ("gpu", "source_lock", "pristine_evaluation", "train_within_budget",
          "policy_interface", "finite_actions", "equal_evaluation")
MAX_PRIVATE_DIAGNOSTIC_BYTES = 4096

# Deliberately generous anti-pretraining guard over the ADDED lines of a patch:
# long base64/hex-like runs and a very large number of numeric literals.
ENCODED_RUN_RE = re.compile(
    r"[A-Za-z0-9+/=_-]{%d,}" % RECIPE["patch_encoded_run_minimum_length"])
NUMERIC_LITERAL_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(?:0[xXoObB][0-9A-Fa-f_]+|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d+)?)"
    r"(?![A-Za-z0-9_.])")
PARAMETER_DATA_MESSAGE = ("submission embeds parameter-like data; "
                          "the policy must be learned inside the verifier run")


def embeds_parameter_like_data(patch_text):
    """True if the patch's added lines exceed either generous data threshold."""
    added = [line[1:] for line in patch_text.splitlines()
             if line.startswith("+") and not line.startswith("+++ ")]
    encoded = sum(len(run) for line in added for run in ENCODED_RUN_RE.findall(line))
    if encoded > RECIPE["max_patch_encoded_run_characters"]:
        return True
    literals = sum(len(NUMERIC_LITERAL_RE.findall(line)) for line in added)
    return literals > RECIPE["max_patch_numeric_literals"]


if tuple(RECIPE["episode_metrics"]) + tuple(RECIPE["step_metrics"]) != SPEC_METRICS:
    raise ValueError("workloads.json metrics differ from the evaluator's beat-spec metrics")


class VerifierSetupDeadlineExpired(RuntimeError):
    """The platform envelope expired before scientific execution began."""

    verifier_failure_code = "verifier_setup_timeout"


class VerifierScientificFailure(RuntimeError):
    """A trusted (reference/judge) failure or suite exhaustion is not a solver score."""

    def __init__(self, code, message, evidence):
        super().__init__(message)
        self.verifier_failure_code = code
        # Private, operator-only: never part of solver feedback. Child logs are
        # left out; a rejected evaluation keeps its bounded reason so that a
        # trusted-side rejection can be diagnosed from the verifier error alone.
        fields = ("event", "time_unix", "phase", "side", "kind", "stage", "workload",
                  "train_seed", "elapsed_seconds", "error_type", "detail")
        self.private_execution_evidence = ({
            "schema_version": evidence["schema_version"],
            "events": [_private_event(row, fields) for row in evidence["events"]],
        } if evidence else None)


def _private_event(row, fields):
    event = {key: row[key] for key in fields if key in row}
    report = row.get("report")
    if isinstance(report, dict) and "invalid" in report:
        event["invalid"] = str(report["invalid"])[:100]
        event["invalid_detail"] = str(report.get("invalid_detail", ""))[:200]
    return event


def _runtime_evidence(path):
    try:
        return read_private_evidence(path)
    except (OSError, ValueError) as error:
        raise VerifierScientificFailure(
            "verifier_execution_evidence_invalid", "trusted execution evidence is unreadable", None) from error


def _raise_runtime_failure(evidence, *, parent_timeout=False):
    """Every runner failure is the judge's: candidate faults are reported in-band."""
    events = evidence["events"] if evidence else []
    failures = [row for row in events if row.get("event") == "phase_failed"]
    if parent_timeout or any(row.get("error_type") in ("TimeoutExpired", "SuiteDeadlineExpired")
                             for row in failures):
        raise VerifierScientificFailure(
            "verifier_suite_timeout", "shared scientific suite deadline exhausted", evidence)
    if failures and failures[-1].get("side") == "reference":
        raise VerifierScientificFailure(
            "verifier_reference_failed", "trusted reference execution failed", evidence)
    raise VerifierScientificFailure(
        "verifier_execution_evidence_invalid", "trusted execution failed", evidence)


def _invalid(reason, *, stderr=None, returncode=None):
    metrics = {"beats_reference": 0, "reference_diagnostics": {"reason": reason[:200]}}
    if stderr is not None:
        body = stderr if isinstance(stderr, bytes) else str(stderr).encode("utf-8", errors="replace")
        metrics["reference_diagnostics"] = {
            "stderr_tail": body[-MAX_PRIVATE_DIAGNOSTIC_BYTES:].decode("utf-8", errors="ignore"),
            "truncated": len(body) > MAX_PRIVATE_DIAGNOSTIC_BYTES,
            "returncode": returncode,
        }
    return 0.0, 0.0, reason[:200], metrics


def _validate_payload(payload, suite):
    expected = [row["id"] for row in RECIPE["suites"][suite]]
    rows = payload["workloads"]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("GPU runner returned the wrong workload count")
    if [row["id"] for row in rows] != expected:
        raise ValueError("GPU runner returned invalid workload identifiers")
    for row in rows:
        for side in ("candidate", "reference"):
            values = row[side]
            if set(values) != set(SPEC_METRICS) or not all(
                    isinstance(values[k], (int, float)) and not isinstance(values[k], bool)
                    and math.isfinite(float(values[k])) for k in SPEC_METRICS):
                raise ValueError("GPU runner returned incomplete/non-finite measurements")
        guards = row["validity"]
        if not all(guards.get(key) is True for key in GUARDS):
            raise ValueError("an authenticity or numerical validity guard failed")
    return rows


def _score(rows, side):
    """100 * r / (r + half), r = max(0, mean episode reward over the workloads).

    Strictly increasing in r, zero for r <= 0, below 100 for every finite r.
    """
    reward = sum(float(row[side]["episode_reward"]) for row in rows) / len(rows)
    if not math.isfinite(reward) or reward <= 0.0:
        return 0.0
    return 100.0 * reward / (reward + SCORE_HALF_SATURATION_REWARD)


def score_payload(payload, suite):
    """Verdict and score from one validated runner payload (no I/O)."""
    rows = _validate_payload(payload, suite)
    audit = []
    for row in rows:
        # Strict, per workload and per metric, no tolerance: higher is better.
        decisions = {key: bool(float(row["candidate"][key]) > float(row["reference"][key]))
                     for key in SPEC_METRICS}
        audit.append({"workload": row["id"], "metrics": decisions})
    score = _score(rows, "candidate")
    reference_score = _score(rows, "reference")
    metrics = {key: sum(float(row["candidate"][key.removeprefix("development_mean_")])
                        for row in rows) / len(rows) for key in VISIBLE}
    metrics.update({"beat_spec_workload_passes": audit,
                    "beats_reference": int(all(v for row in audit for v in row["metrics"].values())),
                    "reference_score": reference_score, "margin": score - reference_score})
    return score, score, "scored trained policies in the pristine environment", metrics


def _evaluate(solution_path: str, evidence_path: Path):
    try:
        with open(solution_path, encoding="utf-8", errors="replace", newline="") as stream:
            patch_text = stream.read()
    except OSError:
        patch_text = ""  # apply_submission_patch reports the unreadable file
    if embeds_parameter_like_data(patch_text):
        return _invalid(PARAMETER_DATA_MESSAGE)
    applied = apply_submission_patch(solution_path, TASK_DIR, CONFIG)
    if not applied.ok:
        return _invalid(applied.message)
    root = applied.workdir
    try:
        if not RUNNER.is_file():
            return _invalid("authoritative NVIDIA workload runner is unavailable")
        final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
        suite = "final" if final else "development"
        runner_env = {key: os.environ[key] for key in (
            "PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "HOME", "TMPDIR") if key in os.environ}
        runner_env["PYTHONDONTWRITEBYTECODE"] = "1"
        runner_env["FRONTIER_GO1_EVIDENCE_PATH"] = str(evidence_path)
        suite_seconds = (CONFIG["runtime"].get("final_timeout_seconds", CONFIG["runtime"]["timeout_seconds"])
                         if final else CONFIG["runtime"]["timeout_seconds"])
        timeout = suite_seconds - 60
        deadline = os.environ.get("FRONTIER_VERIFIER_LOCAL_DEADLINE")
        if deadline is not None:
            end = float(deadline)
            if not math.isfinite(end):
                raise ValueError("invalid Verifier deadline")
            timeout = min(timeout, end - time.monotonic() - 60)
            if end - time.monotonic() <= 100:
                raise VerifierSetupDeadlineExpired("Verifier setup exhausted its execution envelope")
            runner_env["FRONTIER_VERIFIER_LOCAL_DEADLINE"] = deadline
        proc = subprocess.run(
            [sys.executable, "-B", str(RUNNER), "--repo", str(root), "--suite", suite],
            text=True, capture_output=True, timeout=timeout, env=runner_env,
        )
        if proc.returncode == 75 and proc.stdout.strip() == '{"verifier_failure": "verifier_setup_timeout"}':
            raise VerifierSetupDeadlineExpired("Verifier setup exhausted its execution envelope")
        if proc.returncode:
            sys.stderr.write(proc.stderr[-MAX_PRIVATE_DIAGNOSTIC_BYTES:])
            _raise_runtime_failure(_runtime_evidence(evidence_path))
        payload = json.loads(proc.stdout)
        if "candidate_failure" in payload:
            # Training over budget, a crash in submitted code, or a policy the
            # pristine evaluation rejected: an invalid submission, scored zero.
            return _invalid(str(payload["candidate_failure"]))
        return score_payload(payload, suite)
    except (VerifierSetupDeadlineExpired, VerifierScientificFailure):
        raise
    except subprocess.TimeoutExpired as exc:
        if os.environ.get("FRONTIER_VERIFIER_LOCAL_DEADLINE") is not None:
            evidence = _runtime_evidence(evidence_path)
            if evidence is None or not any(row.get("event") == "scientific_started"
                                           for row in evidence["events"]):
                raise VerifierSetupDeadlineExpired("Verifier setup exhausted its execution envelope") from exc
        _raise_runtime_failure(_runtime_evidence(evidence_path), parent_timeout=True)
    finally:
        if root is not None:
            shutil.rmtree(root.parent, ignore_errors=True)


def evaluate(solution_path: str):
    try:
        # Submitted code may share this UID; refuse same-UID ptrace of the judge.
        ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)
    except (OSError, AttributeError):
        pass
    with tempfile.TemporaryDirectory(prefix="go1-private-evidence-") as temporary:
        path = Path(temporary) / "workloads.json"
        score, unbounded, message, metrics = _evaluate(solution_path, path)
        evidence = read_private_evidence(path)
        if evidence is not None:
            retained = {"runtime_evidence_schema": evidence["schema_version"],
                        "runtime_evidence": evidence["events"]}
            if "beat_spec_workload_passes" in metrics:
                metrics.update(retained)
            else:
                # Preserve the adapter's existing scored-zero diagnostic shape.
                metrics.setdefault("reference_diagnostics", {}).update(retained)
        return score, unbounded, message, metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1] if len(sys.argv) > 1 else ""), default=str))
