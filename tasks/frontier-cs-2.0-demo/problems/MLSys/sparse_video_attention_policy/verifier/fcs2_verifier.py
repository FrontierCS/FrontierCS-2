"""Run a Frontier-CS 2.0 evaluator under the FrontierCS2 Verifier contract.

Two platforms, one evaluator. Ours runs `evaluate(solution_path)` inside a judge
server that speaks HTTP; FrontierCS2 runs a command in a Verifier Pod that reads
five environment variables and writes two files. The evaluator does not change —
this is the whole adapter, and it is shared rather than generated per task so
that a fix reaches every exported task at once.

What the platform hands us (see the task-protocol reference):

    FRONTIER_PAYLOAD_PATH     the submitted tar
    FRONTIER_DATASET_PATH     the selected train or eval dataset object
    FRONTIER_VERIFICATION_MODE  "train" or "eval"
    FRONTIER_RESULT_PATH      where to write the task-facing result JSON
    FRONTIER_EVIDENCE_PATH    where to write a zstd-compressed evidence tar

What it must never do: reach S3, Kubernetes, Modal, or any provider endpoint.
For a Verifier-GPU task the shared adapter may use the Verifier's scoped Driver
capability; the Driver still owns the provider credential, SSH gateway, metering,
and cleanup.

Three things this adapter is careful about, because each is a place where a
faithful-looking translation would silently change the task:

**Mode is the suite selector, not a hint.** Our evaluators choose their suite
from `FRONTIER_SUBMISSION_ROLE`, and every one of them tests `== "final"`. The
platform's `eval` maps to `final` and its `train` maps to anything else; passing
the platform's own word through would put every task on its development suite
during grading.

**Private metrics do not go back to the solver.** `beats_reference`, reference
measurements, margins, and exact per-workload pass/fail decisions stay in
operator evidence. Measured eval results also emit the validated operator-only
`private_metrics` envelope for Management. In train mode, tasks with a beat spec also return its approved metric
values and aggregate pass/fail percentages; eval mode returns none of them.

**The evaluator needs a writable task tree.** Ours resolve `TASK_DIR` from
`__file__` and build inside it. The Verifier image is read-only, so the immutable
bundle is copied into a bounded scratch first and the evaluator is imported from
the copy.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

from beat_spec_feedback import public_metrics as beat_spec_public_metrics

# Resolve the evidence codec before importing or executing task-owned Python.
# Evaluators legitimately import a large third-party stack and can also mutate
# import state; neither should be able to shadow or poison a Verifier dependency
# that is part of the trusted image.  Keep the optional import here so a broken
# image still returns the bounded configuration error below.
try:
    import zstandard as _zstandard
except ImportError:  # pragma: no cover - exercised by an intentionally bad image
    _zstandard = None

RESULT_SCHEMA_HINT = "frontier.verifier-result/v1"
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_PAYLOAD_MEMBERS = 4096
MAX_MESSAGE_CHARS = 4096
MAX_METRICS_BYTES = 512 * 1024
# Graded suite: eight Wan2.1 and four HunyuanVideo workloads.
FINAL_WORKLOADS = 12
CHUNK = 1024 * 1024


class VerifierFailure(Exception):
    """A bounded, structured reason the submission could not be scored."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------- environment


def _required_path(name: str) -> Path:
    value = os.environ.get(name, "")
    if not value or not value.startswith("/"):
        raise VerifierFailure("verifier_configuration_error",
                              f"{name} must be an absolute path")
    return Path(value)


def _mode() -> str:
    value = os.environ.get("FRONTIER_VERIFICATION_MODE", "")
    if value not in ("train", "eval"):
        raise VerifierFailure("verifier_configuration_error",
                              "FRONTIER_VERIFICATION_MODE must be train or eval")
    return value


# ------------------------------------------------------------------- payload


def _safe_member(member: tarfile.TarInfo) -> PurePosixPath:
    name = member.name
    if "\\" in name or "\x00" in name or name.startswith("/"):
        raise VerifierFailure("invalid_submission", "payload member path is not relative")
    path = PurePosixPath(name)
    if any(part in ("", ".", "..") for part in path.parts):
        raise VerifierFailure("invalid_submission", "payload member path escapes the archive")
    if not (member.isfile() or member.isdir()):
        raise VerifierFailure("invalid_submission",
                              "payload contains a link, device or FIFO member")
    return path


def extract_payload(payload: Path, destination: Path) -> None:
    """Unpack the tar, refusing every shape the contract forbids.

    Refused rather than sanitised: a payload with an escaping member is not a
    submission with a fixable path, it is a submission that tried something.
    """
    try:
        size = payload.stat().st_size
    except OSError as error:
        raise VerifierFailure("invalid_submission", "payload is not readable") from error
    if not 0 < size <= MAX_PAYLOAD_BYTES:
        raise VerifierFailure("invalid_submission",
                              f"payload must be 1..={MAX_PAYLOAD_BYTES} bytes")
    destination.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    try:
        with tarfile.open(payload, mode="r:") as archive:
            members = []
            for member in archive:
                relative = _safe_member(member)
                if str(relative) in seen:
                    raise VerifierFailure("invalid_submission",
                                          "payload contains duplicate member paths")
                seen.add(str(relative))
                members.append(member)
                if len(members) > MAX_PAYLOAD_MEMBERS:
                    raise VerifierFailure("invalid_submission", "payload has too many members")
            archive.extractall(destination, members=members, filter="tar")
    except tarfile.TarError as error:
        raise VerifierFailure("invalid_submission",
                              "payload is not an uncompressed POSIX tar") from error


def entrypoint_path(root: Path, entrypoint: str) -> Path:
    candidate = (root / entrypoint).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise VerifierFailure("invalid_submission",
                              "submission entrypoint escapes the payload root") from error
    if not candidate.is_file() or candidate.is_symlink():
        raise VerifierFailure("invalid_submission",
                              f"payload does not contain a regular file at {entrypoint}")
    if candidate.stat().st_size == 0:
        raise VerifierFailure("invalid_submission", "submission entrypoint is empty")
    return candidate


# -------------------------------------------------------------- task bundle


def stage_task(bundle: Path, scratch: Path) -> Path:
    """Copy the immutable task bundle somewhere the evaluator can build.

    Ours resolve `TASK_DIR` from `__file__` and compile, unpack and write inside
    it. The Verifier image root filesystem is read-only, and `/frontier` is the
    one bounded writable mount, so the evaluator has to run from a copy.
    """
    if not (bundle / "evaluator.py").is_file():
        raise VerifierFailure("verifier_configuration_error",
                              f"task bundle at {bundle} has no evaluator.py")
    task_dir = scratch / "task"
    shutil.copytree(bundle, task_dir, symlinks=True)
    # `rglob("*")` walks task_dir's descendants, never task_dir itself — so an
    # evaluator that only edits existing files was fine, but one that creates a
    # *new* entry directly under TASK_DIR (several of ours keep a debug patch
    # archive beside themselves, e.g. `TASK_DIR / ".patch-archive"`) hit a
    # PermissionError on the directory's own write bit. Found by running the
    # first task's reference patch through a real build of this image.
    for path in (task_dir, *task_dir.rglob("*")):
        try:
            path.chmod(path.stat().st_mode | 0o200)
        except OSError:
            continue
    return task_dir


def load_evaluator(task_dir: Path):
    spec = importlib.util.spec_from_file_location(
        f"frontier_task_evaluator_{os.getpid()}", task_dir / "evaluator.py")
    if spec is None or spec.loader is None:
        raise VerifierFailure("verifier_configuration_error", "evaluator.py is not importable")
    module = importlib.util.module_from_spec(spec)
    # ``exec_module`` does not register a manually-created module.  Runtime
    # helpers such as ``dataclasses`` resolve postponed annotations through
    # ``sys.modules[cls.__module__]`` while the module is executing, so an
    # otherwise ordinary evaluator containing ``@dataclass`` failed here with
    # an AttributeError even though importing it normally worked.  Install the
    # module only for the duration of execution and restore any prior entry.
    previous_module = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    inserted = str(task_dir)
    sys.path.insert(0, inserted)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001 - a task's own import failure
        raise VerifierFailure("verifier_configuration_error",
                              f"evaluator.py failed to import: {type(error).__name__}") from error
    finally:
        if sys.path and sys.path[0] == inserted:
            sys.path.pop(0)
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
    evaluate = getattr(module, "evaluate", None)
    if not callable(evaluate):
        raise VerifierFailure("verifier_configuration_error",
                              "evaluator.py does not expose evaluate()")
    return evaluate


# ------------------------------------------------------------------- scoring


def _bounded(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        raise VerifierFailure("invalid_evaluator_result", "metrics are nested too deeply")
    if value is None or isinstance(value, (bool, str)):
        return value[:MAX_MESSAGE_CHARS] if isinstance(value, str) else value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VerifierFailure("invalid_evaluator_result", "metrics contain a non-finite number")
        return value
    if isinstance(value, dict):
        return {str(key)[:256]: _bounded(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bounded(item, depth + 1) for item in value]
    raise VerifierFailure("invalid_evaluator_result", "metrics contain an unsupported value")


def run_evaluator(evaluate, submission: Path, mode: str) -> tuple[float, float, str, dict]:
    """Call the task's evaluator with the role its own code expects.

    `eval` is `final`; anything else is a development submission. Our evaluators
    all test `FRONTIER_SUBMISSION_ROLE == "final"`, so passing the platform's own
    word through would silently grade every task on its development suite.
    """
    previous = os.environ.get("FRONTIER_SUBMISSION_ROLE")
    os.environ["FRONTIER_SUBMISSION_ROLE"] = "final" if mode == "eval" else "development"
    try:
        result = evaluate(str(submission))
    except Exception as error:  # noqa: BLE001 - a task's own crash is a task failure
        raise VerifierFailure(
            "evaluator_error",
            f"the task evaluator raised {type(error).__name__}") from error
    finally:
        if previous is None:
            os.environ.pop("FRONTIER_SUBMISSION_ROLE", None)
        else:
            os.environ["FRONTIER_SUBMISSION_ROLE"] = previous

    if not isinstance(result, tuple) or len(result) != 4:
        raise VerifierFailure("invalid_evaluator_result",
                              "the task evaluator did not return a 4-tuple")
    score, unbounded, message, metrics = result
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(float(score)) or float(score) < 0):
        raise VerifierFailure("invalid_evaluator_result",
                              "raw_score must be a finite non-negative number")
    if isinstance(unbounded, bool) or not isinstance(unbounded, (int, float)) \
            or not math.isfinite(float(unbounded)):
        raise VerifierFailure("invalid_evaluator_result", "the unbounded score is not finite")
    if not isinstance(message, str):
        raise VerifierFailure("invalid_evaluator_result", "the evaluator message is not a string")
    scientific, generation_records = split_operator_generations(metrics if isinstance(metrics, dict) else {})
    scientific, generation_failures = split_operator_diagnostics(scientific)
    bounded = _bounded(scientific)
    if generation_records is not None:
        bounded["_operator_generation_workload_diagnostics"] = generation_records
    if generation_failures is not None:
        bounded["_operator_diagnostics"] = generation_failures
    if len(json.dumps(bounded, sort_keys=True).encode("utf-8")) > MAX_METRICS_BYTES:
        raise VerifierFailure("invalid_evaluator_result", "metrics exceed the evidence limit")
    return float(score), float(unbounded), message[:MAX_MESSAGE_CHARS], bounded


# ------------------------------------------------------------------- outputs


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, allow_nan=False, ensure_ascii=True,
                      separators=(",", ":"), sort_keys=True).encode("ascii")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(body)
    temporary.replace(path)


def write_evidence(path: Path, entries: dict[str, bytes]) -> None:
    """A zstd-compressed tar, which is what a new TaskVersion must produce.

    Deterministic: fixed mtime, uid/gid and mode, sorted names. Two runs of the
    same payload against the same dataset produce the same bytes, so a reviewer
    comparing evidence across attempts is comparing the task and not the clock.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(entries):
            info = tarfile.TarInfo(name)
            info.size = len(entries[name])
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(entries[name]))
    raw = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if _zstandard is None:
        # The platform reads the member as `.tar.zst`. Rather than ship bytes
        # whose name lies about their encoding, fail loudly at build time.
        raise VerifierFailure(
            "verifier_configuration_error",
            "the verifier image must provide the `zstandard` package to write "
            "evidence.tar.zst")
    body = _zstandard.ZstdCompressor(level=10).compress(raw)
    path.write_bytes(body)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def final_private_metrics(metrics: dict, *, score: float, unbounded: float,
                          beat_spec_path: Path) -> dict | None:
    """Transport the existing twelve-workload verdict; never derive it from score."""
    if not metrics:
        return None
    try:
        if type(metrics.get("beats_reference")) is not int:
            raise ValueError("invalid verdict type")
        projected = beat_spec_public_metrics(metrics, beat_spec_path=beat_spec_path)
        progress = projected["beat_spec_progress"]
        if not progress or any(row["total"] != FINAL_WORKLOADS for row in progress.values()):
            raise ValueError("incomplete final workload set")
        # SVG's evaluator returns this exact all-false shape for rejected or
        # failed generation. It contains no measured reference score/margin.
        if (set(metrics) == {"beats_reference", "beat_spec_workload_passes"}
                and score == 0 and unbounded == 0 and metrics["beats_reference"] == 0
                and all(row["passed"] == 0 for row in progress.values())):
            return {"beats_reference": 0, "beat_spec_progress": progress}
        result = {"beats_reference": metrics["beats_reference"],
                  "beat_spec_progress": progress}
        for name in ("reference_score", "margin"):
            value = metrics.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise ValueError("invalid reference scalar")
            result[name] = value
        spec = json.loads(beat_spec_path.read_text())
        if "beats_reference_aggregation" in spec:
            result["aggregation"] = spec["beats_reference_aggregation"]
        body = json.dumps(result, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(body) > 16 * 1024:
            raise ValueError("private metrics too large")
        return result
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        # Validation can mention private workload identities. Never echo them.
        raise VerifierFailure("invalid_evaluator_result",
                              "invalid private beat-spec metrics") from error


# ---------------------------------------------------------------------- main


def _validate_operator_generation(value):
    phases = ("imports_start", "imports_complete", "entrypoint_start", "pipeline_enter",
              "warmup_complete", "inference_complete", "entrypoint_complete")
    expected = {"subprocess_elapsed_seconds", "stderr_observed_bytes", "stderr_observed_sha256",
                "stderr_tail", "stderr_tail_truncated", "stderr_capture_complete", "phases"}
    if not isinstance(value, dict) or set(value) != expected:
        raise VerifierFailure("invalid_evaluator_result", "invalid operator generation record")
    elapsed = value["subprocess_elapsed_seconds"]
    count = value["stderr_observed_bytes"]
    digest = value["stderr_observed_sha256"]
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed) or elapsed < 0
            or isinstance(count, bool) or not isinstance(count, int) or count < 0
            or not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or not isinstance(value["stderr_tail"], str)
            or len(value["stderr_tail"].encode("utf-8")) > 3072
            or type(value["stderr_tail_truncated"]) is not bool
            or type(value["stderr_capture_complete"]) is not bool
            or not isinstance(value["phases"], list) or len(value["phases"]) > len(phases)):
        raise VerifierFailure("invalid_evaluator_result", "invalid operator generation record")
    previous_elapsed = 0.0
    for index, row in enumerate(value["phases"]):
        if (not isinstance(row, dict) or set(row) != {"stage", "elapsed_seconds"}
                or row["stage"] != phases[index]
                or isinstance(row["elapsed_seconds"], bool)
                or not isinstance(row["elapsed_seconds"], (int, float))
                or not math.isfinite(row["elapsed_seconds"])
                or row["elapsed_seconds"] < previous_elapsed
                or row["elapsed_seconds"] > elapsed):
            raise VerifierFailure("invalid_evaluator_result", "invalid operator generation phases")
        previous_elapsed = row["elapsed_seconds"]


def split_operator_generations(metrics):
    scientific = dict(metrics)
    rows = scientific.pop("_operator_generation_workload_diagnostics", None)
    if rows is None:
        return scientific, None
    if not isinstance(rows, list) or not 1 <= len(rows) <= FINAL_WORKLOADS:
        raise VerifierFailure("invalid_evaluator_result", "invalid operator generation records")
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"workload", "generation"}
                or not isinstance(row["workload"], str) or not 1 <= len(row["workload"]) <= 128):
            raise VerifierFailure("invalid_evaluator_result", "invalid operator generation records")
        _validate_operator_generation(row["generation"])
    if len(json.dumps(rows, allow_nan=False).encode("utf-8")) > 64 * 1024:
        raise VerifierFailure("invalid_evaluator_result", "operator generation records too large")
    return scientific, rows


def split_operator_pairs(metrics: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Retain paired raw measurements only in operator evidence."""
    scientific = dict(metrics)
    pairs = scientific.pop("_operator_paired_workload_metrics", None)
    if pairs is None:
        return scientific, None
    expected = {"workload", "candidate", "reference", "dense_latency_seconds"}
    names = {"e2e_latency_seconds", "psnr_db", "ssim", "lpips"}
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= FINAL_WORKLOADS:
        raise VerifierFailure("invalid_evaluator_result", "invalid operator paired metrics")
    for row in pairs:
        if not isinstance(row, dict) or set(row) != expected or not isinstance(row["workload"], str):
            raise VerifierFailure("invalid_evaluator_result", "invalid operator paired metrics")
        for role in ("candidate", "reference"):
            values = row[role]
            if not isinstance(values, dict) or set(values) != names:
                raise VerifierFailure("invalid_evaluator_result", "invalid operator paired metrics")
            for name, value in values.items():
                if name == "psnr_db" and value == "positive_infinity":
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise VerifierFailure("invalid_evaluator_result", "invalid operator paired metrics")
        dense = row["dense_latency_seconds"]
        if isinstance(dense, bool) or not isinstance(dense, (int, float)) or not math.isfinite(dense) or dense <= 0:
            raise VerifierFailure("invalid_evaluator_result", "invalid operator paired metrics")
    body = json.dumps(pairs, allow_nan=False).encode("utf-8")
    if len(body) > 16 * 1024:
        raise VerifierFailure("invalid_evaluator_result", "operator paired metrics too large")
    return scientific, pairs


def split_operator_diagnostics(metrics: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Remove the evaluator's reserved diagnostics before public validation."""
    scientific = dict(metrics)
    diagnostics = scientific.pop("_operator_diagnostics", None)
    if diagnostics is None:
        return scientific, None
    if not isinstance(diagnostics, list) or not 1 <= len(diagnostics) <= 8:
        raise VerifierFailure("invalid_evaluator_result",
                              "invalid operator diagnostics")
    expected = {
        "workload", "returncode", "stdout_bytes", "stderr_bytes",
        "stdout_sha256", "stderr_sha256", "stderr_tail",
        "stderr_tail_truncated",
    }
    for row in diagnostics:
        if not isinstance(row, dict) or set(row) not in (expected, expected | {"stage_context"}):
            raise VerifierFailure("invalid_evaluator_result",
                                  "invalid operator diagnostics")
        if (not isinstance(row["workload"], str)
                or not 1 <= len(row["workload"]) <= 128
                or isinstance(row["returncode"], bool)
                or not isinstance(row["returncode"], int)
                or any(isinstance(row[name], bool) or not isinstance(row[name], int)
                       or row[name] < 0
                       for name in ("stdout_bytes", "stderr_bytes"))
                or any(not isinstance(row[name], str) or len(row[name]) != 64
                       or any(char not in "0123456789abcdef" for char in row[name])
                       for name in ("stdout_sha256", "stderr_sha256"))
                or not isinstance(row["stderr_tail"], str)
                or type(row["stderr_tail_truncated"]) is not bool):
            raise VerifierFailure("invalid_evaluator_result",
                                  "invalid operator diagnostics")
    for row in diagnostics:
        context = row.get("stage_context", {})
        if (not isinstance(context, dict) or set(context) - {"stage", "exception_type", "timeout_seconds", "generation"}
                or context.get("stage", "candidate_generation") not in {"dense_target", "reference_target", "candidate_generation", "candidate_metrics", "reference_metrics"}
                or not isinstance(context.get("exception_type", ""), str)
                or ("timeout_seconds" in context and (isinstance(context["timeout_seconds"], bool)
                    or not isinstance(context["timeout_seconds"], (int, float))
                    or not math.isfinite(context["timeout_seconds"]) or context["timeout_seconds"] <= 0))):
            raise VerifierFailure("invalid_evaluator_result", "invalid operator stage context")
        if "generation" in context:
            _validate_operator_generation(context["generation"])
    try:
        body = json.dumps(diagnostics, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise VerifierFailure("invalid_evaluator_result",
                              "invalid operator diagnostics") from error
    if len(body) > 16 * 1024:
        raise VerifierFailure("invalid_evaluator_result",
                              "operator diagnostics too large")
    return scientific, diagnostics


def verify(*, bundle: Path, entrypoint: str, payload: Path, dataset: Path,
           mode: str, result_path: Path, evidence_path: Path) -> dict:
    started = time.time()
    configured_scratch = os.environ.get("FRONTIER_VERIFIER_SCRATCH", "")
    scratch_parent = (configured_scratch if configured_scratch
                      else "/frontier" if os.access("/frontier", os.W_OK) else None)
    scratch = Path(tempfile.mkdtemp(prefix="frontier-verify-", dir=scratch_parent))
    try:
        extracted = scratch / "payload"
        extract_payload(payload, extracted)
        submission = entrypoint_path(extracted, entrypoint)
        task_dir = stage_task(bundle, scratch)
        evaluate = load_evaluator(task_dir)
        score, unbounded, message, evaluator_metrics = run_evaluator(
            evaluate, submission, mode)
        evaluator_metrics, operator_generations = split_operator_generations(evaluator_metrics)
        evaluator_metrics, operator_pairs = split_operator_pairs(evaluator_metrics)
        metrics, operator_diagnostics = split_operator_diagnostics(evaluator_metrics)

        visible_metrics: dict[str, Any] = {}
        # A rejected/failed candidate is a valid scored result (normally score
        # zero), and task evaluators conventionally return an empty metrics
        # object for it. There are no workload decisions to aggregate in that
        # case. Keep strict beat-spec validation for every non-empty evaluator
        # result so partial or inconsistent private decisions still fail closed.
        if (mode == "train" and metrics
                and (task_dir / "beat_spec.json").is_file()):
            try:
                visible_metrics = beat_spec_public_metrics(
                    metrics,
                    beat_spec_path=task_dir / "beat_spec.json",
                )
            except ValueError as error:
                raise VerifierFailure("invalid_evaluator_result", str(error)) from error

        private_metrics = (final_private_metrics(
            metrics, score=score, unbounded=unbounded,
            beat_spec_path=task_dir / "beat_spec.json") if mode == "eval" else None)

        evidence_record = {
            "mode": mode,
            "raw_score": score,
            "score_unbounded": unbounded,
            "message": message,
            # The full scientific dict, including every reference-relative key,
            # remains operator-only. Reserved process diagnostics are separated
            # so they cannot affect score or beat-spec validation.
            "metrics": metrics,
            "payload_sha256": sha256_file(payload),
            "dataset_sha256": sha256_file(dataset) if dataset.is_file() else None,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        if operator_generations is not None:
            evidence_record["generation_workload_diagnostics"] = operator_generations
        if operator_pairs is not None:
            evidence_record["paired_workload_metrics"] = operator_pairs
        if operator_diagnostics is not None:
            evidence_record["operator_diagnostics"] = operator_diagnostics
        evidence = {
            "metrics.json": json.dumps(
                evidence_record, indent=2, sort_keys=True).encode("utf-8"),
        }
        write_evidence(evidence_path, evidence)
        result = {
            "status": "passed",
            "raw_score": score,
            "summary": message[:MAX_MESSAGE_CHARS],
            # The exact reference-relative decisions stay in evidence. Train may
            # add only the beat spec's approved values and aggregate percentages;
            # eval exposes none of them.
            "metrics": {"score_unbounded": unbounded, **visible_metrics},
        }
        if private_metrics is not None:
            result["private_metrics"] = private_metrics
        write_json(result_path, result)
        return result
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(bundle: Path, entrypoint: str) -> int:
    result_path: Path | None = None
    try:
        result_path = _required_path("FRONTIER_RESULT_PATH")
        # The CPU control Pod inherits the wrapper's scoped bearer. The same
        # image runs on Modal, but sshd starts with a cleared environment and
        # therefore has no bearer; that is the non-recursive execution signal.
        if (os.environ.get("FRONTIER_VERIFIER_GPU_TYPE")
                and os.environ.get("FRONTIER_VERIFIER_TOKEN")):
            try:
                from fcs2_verifier_gpu import VerifierGpuError, run as run_on_gpu

                run_on_gpu(
                    payload=_required_path("FRONTIER_PAYLOAD_PATH"),
                    dataset=_required_path("FRONTIER_DATASET_PATH"),
                    mode=_mode(),
                    result_path=result_path,
                    evidence_path=_required_path("FRONTIER_EVIDENCE_PATH"),
                )
                return 0
            except VerifierGpuError as error:
                # The GPU adapter deliberately emits only bounded lifecycle and
                # transport messages; preserve those in the platform result so
                # an operator can distinguish quota, API, SSH, and evaluator
                # failures without exposing credentials or provider responses.
                raise VerifierFailure(
                    "verifier_gpu_error",
                    f"Verifier GPU execution failed: {error}") from error
            except Exception as error:  # noqa: BLE001 - shared lifecycle adapter
                raise VerifierFailure(
                    "verifier_gpu_error",
                    f"Verifier GPU execution failed: {type(error).__name__}") from error
        verify(
            bundle=bundle,
            entrypoint=entrypoint,
            payload=_required_path("FRONTIER_PAYLOAD_PATH"),
            dataset=_required_path("FRONTIER_DATASET_PATH"),
            mode=_mode(),
            result_path=result_path,
            evidence_path=_required_path("FRONTIER_EVIDENCE_PATH"),
        )
        return 0
    except VerifierFailure as failure:
        if result_path is not None:
            write_json(result_path, {
                "status": "failed",
                "summary": failure.message[:MAX_MESSAGE_CHARS],
                "metrics": {},
                "failure": {"code": failure.code,
                            "message": failure.message[:MAX_MESSAGE_CHARS],
                            "retryable": False},
            })
        print(f"{failure.code}: {failure.message}", file=sys.stderr)
        # A handled GPU lifecycle error is a valid structured task result.  A
        # zero process status lets the platform wrapper upload that result;
        # unknown adapter failures still fail closed below.
        return 0 if failure.code == "verifier_gpu_error" else 1
    except Exception as error:  # noqa: BLE001 - never exit without a result file
        if result_path is not None:
            write_json(result_path, {
                "status": "failed",
                "summary": f"the verifier adapter raised {type(error).__name__}",
                "metrics": {},
                "failure": {"code": "verifier_internal_error",
                            "message": f"{type(error).__name__}", "retryable": False},
            })
        traceback.print_exc()
        return 1
