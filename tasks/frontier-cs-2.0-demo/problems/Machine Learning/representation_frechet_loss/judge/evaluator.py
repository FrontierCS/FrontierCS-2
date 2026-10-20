from __future__ import annotations

import hashlib
import json
import math
import os
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
MODEL_ROOT = Path(os.environ.get("FRONTIER_MODEL_ROOT", "/opt/frontier/models"))
OFFLINE_TARGETS = ("fd-loss", "mae", "siglip2", "convnextv2", "dinov2", "clip")
SPEC_METRICS = ("inception_fdr", "convnext_fdr", "dinov2_fdr", "mae_fdr", "siglip_fdr", "clip_fdr")
VISIBLE = tuple(f"development_mean_{name}" for name in SPEC_METRICS)
LOCKED = ("main_fd.py", "utils/builders.py", "models/denoiser_pmf.py", "frechet_distance/judges.py")
MAX_PRIVATE_DIAGNOSTIC_BYTES = 4096


class VerifierSetupDeadlineExpired(RuntimeError):
    """The platform envelope expired before scientific execution began."""

    verifier_failure_code = "verifier_setup_timeout"


class VerifierScientificFailure(RuntimeError):
    """Trusted reference failure or shared suite exhaustion is not a solver score."""

    def __init__(self, code, message, evidence):
        super().__init__(message)
        self.verifier_failure_code = code
        # Full reports and child logs remain in the private evidence archive.
        fields = ("event", "time_unix", "workload", "side", "phase",
                  "elapsed_seconds", "error_type", "report_sha256")
        self.private_execution_evidence = ({
            "schema_version": evidence["schema_version"],
            "events": [{key: row[key] for key in fields if key in row}
                       for row in evidence["events"]],
        } if evidence else None)


def _runtime_evidence(path):
    try:
        return read_private_evidence(path)
    except (OSError, ValueError) as error:
        raise VerifierScientificFailure(
            "verifier_execution_evidence_invalid", "trusted execution evidence is unreadable", None) from error


def _raise_runtime_failure(evidence, *, parent_timeout=False):
    events = evidence["events"] if evidence else []
    failures = [row for row in events if row.get("event") == "phase_failed"]
    if failures and failures[-1].get("side") == "reference":
        raise VerifierScientificFailure(
            "verifier_reference_failed", "trusted reference execution failed", evidence)
    if parent_timeout or any(row.get("error_type") in ("TimeoutExpired", "TimeoutError", "SuiteDeadlineExpired")
                             for row in failures):
        raise VerifierScientificFailure(
            "verifier_suite_timeout", "shared scientific suite deadline exhausted", evidence)
    if not failures:
        raise VerifierScientificFailure(
            "verifier_execution_evidence_invalid", "trusted execution failure evidence is missing", evidence)


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


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_payload(payload, final):
    rows = payload["workloads"]
    if not isinstance(rows, list) or len(rows) != (3 if final else 2):
        raise ValueError("GPU runner returned the wrong workload count")
    seen = set()
    for row in rows:
        wid = row["id"]
        if not isinstance(wid, str) or not wid or wid in seen:
            raise ValueError("GPU runner returned invalid workload identifiers")
        seen.add(wid)
        for side in ("candidate", "reference"):
            values = row[side]
            if set(values) != set(SPEC_METRICS) or not all(
                    math.isfinite(float(values[k])) and float(values[k]) >= 0 for k in SPEC_METRICS):
                raise ValueError("GPU runner returned incomplete/non-finite FDr values")
        guards = row["validity"]
        required = ("gpu", "source_lock", "model_lock", "finite_features", "finite_covariances",
                    "finite_gradients", "parameter_update", "queue_state_roundtrip",
                    "authentic_warm_start", "equal_sample_counts", "entrypoint_executed")
        if not all(guards.get(key) is True for key in required):
            raise ValueError("an authenticity or numerical validity guard failed")
    return rows


def _evaluate(solution_path: str, evidence_path: Path):
    applied = apply_submission_patch(solution_path, TASK_DIR, CONFIG)
    if not applied.ok:
        return _invalid(applied.message)
    root = applied.workdir
    try:
        upstream = TASK_DIR / "harbor" / "app"
        if any(_sha(root / p) != _sha(upstream / p) for p in LOCKED):
            return _invalid("retained execution-spine source changed")
        if not RUNNER.is_file():
            return _invalid("authoritative NVIDIA workload runner is unavailable")
        assets = {name: MODEL_ROOT / name for name in OFFLINE_TARGETS}
        missing = [name for name, path in assets.items() if not path.is_dir()]
        if missing:
            return _invalid("offline model snapshot is unavailable")
        final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
        runner_env = {key: os.environ[key] for key in (
            "PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "HOME", "TMPDIR") if key in os.environ}
        runner_env["PYTHONDONTWRITEBYTECODE"] = "1"
        runner_env["FRONTIER_FD_EVIDENCE_PATH"] = str(evidence_path)
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
            [sys.executable, str(RUNNER), "--repo", str(root),
             "--suite", "final" if final else "development",
             "--model-root", str(MODEL_ROOT)],
            text=True, capture_output=True, timeout=timeout,
            env=runner_env,
        )
        if proc.returncode == 75 and proc.stdout.strip() == '{"verifier_failure": "verifier_setup_timeout"}':
            raise VerifierSetupDeadlineExpired("Verifier setup exhausted its execution envelope")
        if proc.returncode:
            _raise_runtime_failure(_runtime_evidence(evidence_path))
            return _invalid("GPU workload failed", stderr=proc.stderr, returncode=proc.returncode)
        payload = json.loads(proc.stdout)
        rows = _validate_payload(payload, final)
        audit = []
        for row in rows:
            decisions = {key: bool(float(row["candidate"][key]) < float(row["reference"][key]))
                         for key in SPEC_METRICS}
            audit.append({"workload": row["id"], "metrics": decisions})
        candidate_fdr6 = [sum(float(row["candidate"][k]) for k in SPEC_METRICS) / 6 for row in rows]
        reference_fdr6 = [sum(float(row["reference"][k]) for k in SPEC_METRICS) / 6 for row in rows]
        score = 100.0 / (1.0 + sum(candidate_fdr6) / len(candidate_fdr6))
        reference_score = 100.0 / (1.0 + sum(reference_fdr6) / len(reference_fdr6))
        metrics = {key: sum(float(row["candidate"][key.removeprefix("development_mean_")])
                            for row in rows) / len(rows) for key in VISIBLE}
        metrics.update({"beat_spec_workload_passes": audit,
                        "beats_reference": int(all(v for row in audit for v in row["metrics"].values())),
                        "reference_score": reference_score, "margin": score - reference_score})
        return score, score, "scored authentic GPU workloads", metrics
    except (VerifierSetupDeadlineExpired, VerifierScientificFailure):
        raise
    except subprocess.TimeoutExpired as exc:
        if os.environ.get("FRONTIER_VERIFIER_LOCAL_DEADLINE") is not None:
            evidence = _runtime_evidence(evidence_path)
            if evidence is None or not any(row.get("event") == "scientific_started"
                                           for row in evidence["events"]):
                raise VerifierSetupDeadlineExpired("Verifier setup exhausted its execution envelope") from exc
        _raise_runtime_failure(_runtime_evidence(evidence_path), parent_timeout=True)
    except Exception:
        return _invalid("evaluation failed")
    finally:
        if root is not None:
            shutil.rmtree(root.parent, ignore_errors=True)


def evaluate(solution_path: str):
    with tempfile.TemporaryDirectory(prefix="fd-private-evidence-") as temporary:
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
    import sys
    print(json.dumps(evaluate(sys.argv[1] if len(sys.argv) > 1 else ""), default=str))
