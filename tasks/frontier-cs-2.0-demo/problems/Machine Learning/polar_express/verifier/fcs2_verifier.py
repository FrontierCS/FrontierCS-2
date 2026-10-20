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
measurements, margins, and exact per-workload pass/fail decisions remain in
evidence. In eval/final mode, scored tasks with a beat spec also emit an
operator-only `private_metrics` result object designed for transport to
Management, never to the solver; platform support ships separately. In train
mode, tasks with a beat spec return only its approved metric values and
aggregate pass/fail percentages.

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
import stat
import sys
import tarfile
import tempfile
import time
import traceback
from contextlib import contextmanager
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any

from beat_spec_feedback import private_metrics as beat_spec_private_metrics
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
MAX_PRIVATE_METRICS_BYTES = 16 * 1024
MAX_PRIVATE_EXCEPTION_BYTES = 8192
MAX_PRIVATE_EXECUTION_BYTES = 128 * 1024
CHUNK = 1024 * 1024


class VerifierFailure(Exception):
    """A bounded, structured reason the submission could not be scored."""

    def __init__(self, code: str, message: str, *, private_evidence: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.private_evidence = private_evidence


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


def load_candidate_diagnostic(bundle: Path):
    """Capture an explicitly enabled, pinned private validator before task code."""
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate diagnostic declaration field")
            result[key] = value
        return result

    def read_regular(name: str, limit: int) -> bytes:
        fd = os.open(bundle / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
                raise ValueError("invalid diagnostic asset")
            body = source.read(limit + 1)
        if len(body) > limit:
            raise ValueError("oversized diagnostic asset")
        return body

    try:
        try:
            declaration = read_regular("candidate-diagnostic.json", 1024)
        except FileNotFoundError:
            return None
        config = json.loads(declaration, object_pairs_hook=unique_fields)
        if (not isinstance(config, dict)
                or set(config) != {"schema", "helper_sha256", "diagnostic_schema"}
                or config["schema"] != "frontier.candidate-diagnostic-validator/v1"
                or not isinstance(config["diagnostic_schema"], str)
                or not config["diagnostic_schema"]):
            raise ValueError("invalid diagnostic declaration")
        body = read_regular("candidate_failure.py", 64 * 1024)
        if hashlib.sha256(body).hexdigest() != config["helper_sha256"]:
            raise ValueError("diagnostic helper digest mismatch")
        namespace = {"__name__": "frontier_private_candidate_diagnostic",
                     "__file__": str(bundle / "candidate_failure.py")}
        exec(compile(body, namespace["__file__"], "exec"), namespace)
        validator = namespace.get("public_diagnostic")
        if (not callable(validator)
                or namespace.get("SCHEMA") != config["diagnostic_schema"]):
            raise ValueError("invalid diagnostic validator")
        return validator
    except Exception as error:
        raise VerifierFailure("verifier_configuration_error",
                              "invalid private candidate diagnostic configuration") from error


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
    missing = object()

    @contextmanager
    def import_context():
        # Keep the trusted task's helpers and module registration available both
        # at import time and during evaluate(): sibling imports and dataclasses
        # may be deferred until the function executes. Never leave this scratch
        # directory or evaluator registration installed after the operation.
        previous_module = sys.modules.get(spec.name, missing)
        previous_path = sys.path[:]
        sys.modules[spec.name] = module
        sys.path.insert(0, str(task_dir))
        try:
            yield
        finally:
            sys.path[:] = previous_path
            if previous_module is missing:
                sys.modules.pop(spec.name, None)
            else:
                sys.modules[spec.name] = previous_module

    try:
        with import_context():
            spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001 - a task's own import failure
        raise VerifierFailure("verifier_configuration_error",
                              f"evaluator.py failed to import: {type(error).__name__}") from error
    evaluate = getattr(module, "evaluate", None)
    if not callable(evaluate):
        raise VerifierFailure("verifier_configuration_error",
                              "evaluator.py does not expose evaluate()")

    @wraps(evaluate)
    def evaluate_with_import_context(*args, **kwargs):
        with import_context():
            return evaluate(*args, **kwargs)

    return evaluate_with_import_context


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
        execution = getattr(error, "private_execution_evidence", None)
        if execution is not None:
            execution = _bounded(execution)
            body = json.dumps(execution, allow_nan=False, sort_keys=True).encode("utf-8")
            if not isinstance(execution, dict) or len(body) > MAX_PRIVATE_EXECUTION_BYTES:
                raise VerifierFailure("invalid_evaluator_result", "private execution evidence exceeds its contract") from error
            raise VerifierFailure(
                "invalid_execution_evidence", "the task did not establish valid numerical execution",
                private_evidence={"private_execution": execution}) from error
        # The public result keeps only the exception class. Preserve the cause
        # in private logs so runtime defects can be diagnosed before a retry.
        tail = bytearray()
        diagnostic = traceback.TracebackException.from_exception(
            error, limit=8, capture_locals=False)
        for part in diagnostic.format(chain=True):
            tail.extend(part.encode("utf-8", errors="replace"))
            del tail[:-MAX_PRIVATE_EXCEPTION_BYTES]
        print("frontier-verifier private evaluator exception tail:\n"
              + tail.decode("utf-8", errors="ignore"), file=sys.stderr, flush=True)
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
    bounded = _bounded(metrics if isinstance(metrics, dict) else {})
    try:
        metrics_size = len(json.dumps(bounded, sort_keys=True).encode("utf-8"))
    except ValueError as error:
        # Huge integers can fail JSON encoding before eval's private validation.
        # Keep train's existing exception path and never echo evaluator values.
        if mode != "eval":
            raise
        raise VerifierFailure(
            "invalid_evaluator_result", "metrics cannot be serialized") from error
    if metrics_size > MAX_METRICS_BYTES:
        raise VerifierFailure("invalid_evaluator_result", "metrics exceed the evidence limit")
    return float(score), float(unbounded), message[:MAX_MESSAGE_CHARS], bounded


# ------------------------------------------------------------------- outputs


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, allow_nan=False, ensure_ascii=True,
                      separators=(",", ":"), sort_keys=True).encode("ascii")


def _private_metrics(metrics: dict, beat_spec_path: Path) -> dict | None:
    try:
        result = beat_spec_private_metrics(metrics, beat_spec_path=beat_spec_path)
    except ValueError as error:
        # Progress validation errors can contain private workload identities.
        raise VerifierFailure(
            "invalid_evaluator_result", "invalid private beat-spec metrics") from error
    if result is not None:
        progress = result["beat_spec_progress"]
        for name in sorted(progress, reverse=True):
            if len(_json_bytes(result)) <= MAX_PRIVATE_METRICS_BYTES:
                break
            del progress[name]
    return result


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _json_bytes(payload)
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


# ---------------------------------------------------------------------- main


def verify(*, bundle: Path, entrypoint: str, payload: Path, dataset: Path,
           mode: str, result_path: Path, evidence_path: Path) -> dict:
    started = time.time()
    configured_scratch = os.environ.get("FRONTIER_VERIFIER_SCRATCH", "")
    scratch_parent = (configured_scratch if configured_scratch
                      else "/frontier" if os.access("/frontier", os.W_OK) else None)
    scratch = Path(tempfile.mkdtemp(prefix="frontier-verify-", dir=scratch_parent))
    # The Verifier root filesystem is read-only and only the scratch mount is
    # writable, so build tools the evaluator spawns (autoconf's config.guess,
    # Go and pip caches, anything honouring HOME) must be pointed at it too.
    for name, subdir in (("TMPDIR", "tmp"), ("TMP", "tmp"), ("TEMP", "tmp"),
                         ("HOME", "home"), ("XDG_CACHE_HOME", "home/.cache")):
        (scratch / subdir).mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(scratch / subdir)
    try:
        candidate_diagnostic = load_candidate_diagnostic(bundle)
        extracted = scratch / "payload"
        extract_payload(payload, extracted)
        submission = entrypoint_path(extracted, entrypoint)
        task_dir = stage_task(bundle, scratch)
        evaluate = load_evaluator(task_dir)
        try:
            score, unbounded, message, metrics = run_evaluator(evaluate, submission, mode)
        except VerifierFailure as failure:
            if failure.private_evidence is None:
                raise
            # Keep invalid execution observations in the existing private archive.
            # A normal command return transports this failed result through SSH.
            write_evidence(evidence_path, {"metrics.json": json.dumps({
                "mode": mode, "execution_valid": False, "metrics": failure.private_evidence,
                "payload_sha256": sha256_file(payload),
                "dataset_sha256": sha256_file(dataset) if dataset.is_file() else None,
                "elapsed_seconds": round(time.time() - started, 3),
            }, allow_nan=False, sort_keys=True).encode("utf-8")})
            result = {"status": "failed", "summary": failure.message, "metrics": {},
                      "failure": {"code": failure.code, "message": failure.message, "retryable": False}}
            write_json(result_path, result)
            return result

        visible_metrics: dict[str, Any] = {}
        private_metrics: dict[str, Any] | None = None
        # Only isolated scored-zero diagnostics bypass workload aggregation.
        # All other non-empty train results retain strict beat-spec validation.
        private = metrics.get("all_final_metrics")
        private_rejection = (
            score == 0.0 and unbounded == 0.0
            and set(metrics) == {"all_final_metrics"}
            and isinstance(private, dict)
            and set(private) == {"candidate_failure", "runtime"}
            and isinstance(private["candidate_failure"], dict)
            and bool(private["candidate_failure"])
        )
        reference_rejection = set(metrics) in (
            {"reference_diagnostics"}, {"reference_diagnostics", "beats_reference"},
        )
        if reference_rejection and not (
            score == 0.0 and unbounded == 0.0
            and isinstance(metrics["reference_diagnostics"], dict)
            and bool(metrics["reference_diagnostics"])
            and ("beats_reference" not in metrics or (
                type(metrics["beats_reference"]) is int and metrics["beats_reference"] == 0
            ))
        ):
            raise VerifierFailure(
                "invalid_evaluator_result",
                "private reference diagnostic requires an isolated scored zero",
            )
        if "candidate_diagnostic" in metrics:
            if (candidate_diagnostic is None or set(metrics) != {"candidate_diagnostic"}
                    or score != 0.0 or unbounded != 0.0):
                raise VerifierFailure("invalid_evaluator_result", "candidate diagnostic requires an enabled isolated scored zero")
            try:
                diagnostic = candidate_diagnostic(metrics["candidate_diagnostic"])
            except (ValueError, TypeError) as error:
                raise VerifierFailure("invalid_evaluator_result", "invalid candidate diagnostic") from error
            if mode == "train":
                visible_metrics = {"candidate_diagnostic": diagnostic}
        elif (mode == "train" and metrics
                and not (private_rejection or reference_rejection)
                and (task_dir / "beat_spec.json").is_file()):
            try:
                visible_metrics = beat_spec_public_metrics(
                    metrics,
                    beat_spec_path=task_dir / "beat_spec.json",
                )
            except ValueError as error:
                raise VerifierFailure("invalid_evaluator_result", str(error)) from error
        elif (mode == "eval" and "beats_reference" in metrics
                and not (private_rejection or reference_rejection)
                and (task_dir / "beat_spec.json").is_file()):
            private_metrics = _private_metrics(metrics, task_dir / "beat_spec.json")

        evidence = {
            "metrics.json": json.dumps({
                "mode": mode,
                "raw_score": score,
                "score_unbounded": unbounded,
                "message": message,
                # Preserve the full dict, including exact workload decisions.
                # The operator result adds only validated aggregate progress
                # and reference measurements; solver metrics stay separate.
                "metrics": metrics,
                "payload_sha256": sha256_file(payload),
                "dataset_sha256": sha256_file(dataset) if dataset.is_file() else None,
                "elapsed_seconds": round(time.time() - started, 3),
            }, indent=2, sort_keys=True).encode("utf-8"),
        }
        write_evidence(evidence_path, evidence)
        result = {
            "status": "passed",
            "raw_score": score,
            "summary": message[:MAX_MESSAGE_CHARS],
            # The exact workload decisions stay in evidence. Train may
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
