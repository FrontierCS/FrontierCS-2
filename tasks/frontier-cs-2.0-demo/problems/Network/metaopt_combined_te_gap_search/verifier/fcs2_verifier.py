"""Task-local CPU MetaOpt adapter for the native result/evidence contract.

The trusted dataset and mode select original development15 or fresh final30.
The staged evaluator executes the bounded C# project and stores case data in a
private evidence file. Train returns absolute metrics and aggregate progress;
final additionally returns the operator-only verdict. Candidate invalidity scores
zero and fails every required comparison without inventing reference measurements.
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


# Load the exported native helper from its fixed sibling path, including when
# this adapter is loaded by file location in task-local tests.
_feedback_spec = importlib.util.spec_from_file_location(
    "metaopt_beat_spec_feedback", Path(__file__).with_name("beat_spec_feedback.py")
)
_feedback = importlib.util.module_from_spec(_feedback_spec)
_feedback_spec.loader.exec_module(_feedback)
_patching_spec = importlib.util.spec_from_file_location(
    "metaopt_patch_payload", Path(__file__).with_name("patch_payload.py")
)
_patching = importlib.util.module_from_spec(_patching_spec)
_patching_spec.loader.exec_module(_patching)


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
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024
# The patch is the only archive member; leave room for tar headers and padding.
MAX_PATCH_BYTES = MAX_PAYLOAD_BYTES - 16 * 1024
MAX_PROJECT_FILES = 96
# The evaluator's editable surface; every other project file is locked.
EDITABLE_PATHS = (
    "project/MetaOptimize/TrafficEngineering/BudgetedDemandSearch.cs",
    "project/MetaOptimize/TrafficEngineering/Candidate/",
)
MAX_PROJECT_BYTES = 2_000_000
MAX_MESSAGE_CHARS = 4096
MAX_METRICS_BYTES = 512 * 1024
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


def reconstruct_project(payload: Path, starter: Path, destination: Path) -> Path:
    """Apply the submitted `payload/solution.patch` to a copy of the starter project.

    `starter` contains `project/`. The patch may edit the search file and edit
    or add helpers under `Candidate/`; the result keeps the original source limits.
    """
    try:
        patch = _patching.read_patch_member(payload, MAX_PAYLOAD_BYTES, MAX_PATCH_BYTES)
        changed = _patching.apply_patch(
            patch, starter, EDITABLE_PATHS,
            max_files=MAX_PROJECT_FILES, max_file_bytes=MAX_PROJECT_BYTES,
        )
    except _patching.PatchRejected as error:
        raise VerifierFailure("invalid_submission", str(error)) from error
    except OSError as error:
        raise VerifierFailure("verifier_configuration_error",
                              "starter project is unavailable") from error
    project = destination / "project"
    try:
        shutil.copytree(starter / "project", project, symlinks=False)
    except OSError as error:
        raise VerifierFailure("verifier_configuration_error",
                              "starter project is unavailable") from error
    # Image starters are read-only; only their private scratch copies may change.
    for path in (project, *project.rglob("*")):
        path.chmod(path.stat().st_mode | 0o200)
    for relative, body in changed.items():
        target = destination.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    files = [path for path in project.rglob("*") if path.is_file()]
    if len(files) > MAX_PROJECT_FILES or sum(path.stat().st_size for path in files) > MAX_PROJECT_BYTES:
        raise VerifierFailure("invalid_submission", "project exceeds source file or byte limits")
    return project


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
    bounded = _bounded(metrics if isinstance(metrics, dict) else {})
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

    Archive metadata is deterministic; measurements and elapsed time describe
    this particular evaluation.
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


def validate_dataset_selection(dataset: Path, mode: str) -> None:
    expected_roles = {"train": "development", "eval": "final"}
    try:
        selector = json.loads(dataset.read_text())
        valid = (
            mode in expected_roles
            and selector.get("schema_version") == "frontier.metaopt-dataset/v1"
            and selector.get("problem_id") == "metaopt_combined_te_gap_search"
            and selector.get("mode") == mode
            and selector.get("suite_revision") == "original-dev15-public-plus-hidden30-v1"
            and selector.get("selects") == {"FRONTIER_SUBMISSION_ROLE": expected_roles[mode]}
        )
    except (OSError, ValueError, AttributeError):
        valid = False
    if not valid:
        raise VerifierFailure("verifier_configuration_error",
                              "MetaOpt dataset selector does not match the trusted verification mode")


def aggregate_private_evidence(evidence, expected_count):
    """Return aggregate counts and an operator verdict; never return case values or IDs."""
    identities = evidence["case_ids"]
    candidate = evidence["candidate_gaps"]
    reference = evidence["reference_gaps"]
    if (len(identities) != expected_count or len(set(identities)) != expected_count
            or len(candidate) != expected_count or len(reference) != expected_count):
        raise VerifierFailure("invalid_evaluator_result", "private case coverage mismatch")
    for value in [*candidate, *reference]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise VerifierFailure("invalid_evaluator_result", "private replay gap is invalid")
    passed = sum(c >= r for c, r in zip(candidate, reference))
    progress = {"gap": {"passed": passed, "failed": expected_count-passed, "total": expected_count,
                        "pass_percent": round(100.0*passed/expected_count, 2),
                        "fail_percent": round(100.0*(expected_count-passed)/expected_count, 2)}}
    candidate_score = min(100.0, max(0.0, 100.0*(1.0-math.exp(-sum(candidate)/expected_count/40.0))))
    reference_score = min(100.0, max(0.0, 100.0*(1.0-math.exp(-sum(reference)/expected_count/40.0))))
    return {"beats_reference": int(passed == expected_count), "reference_score": reference_score,
            "margin": candidate_score-reference_score, "beat_spec_progress": progress}


def verify(*, bundle: Path, starter: Path, entrypoint: str, payload: Path, dataset: Path,
           mode: str, result_path: Path, evidence_path: Path) -> dict:
    validate_dataset_selection(dataset, mode)
    started = time.time()
    configured_scratch = os.environ.get("FRONTIER_VERIFIER_SCRATCH", "")
    scratch_parent = (configured_scratch if configured_scratch
                      else "/frontier" if os.access("/frontier", os.W_OK) else None)
    scratch = Path(tempfile.mkdtemp(prefix="frontier-verify-", dir=scratch_parent))
    try:
        if entrypoint != "payload/solution.patch":
            raise VerifierFailure("verifier_configuration_error", "unexpected submission entrypoint")
        submission = reconstruct_project(payload, starter, scratch / "payload")
        task_dir = stage_task(bundle, scratch)
        evaluate = load_evaluator(task_dir)
        score, unbounded, message, metrics = run_evaluator(evaluate, submission, mode)

        visible_metrics = {}
        private_metrics = None
        private_case_path = task_dir / "case-evidence.json"
        if metrics:
            if not private_case_path.is_file() or private_case_path.stat().st_size > 1024 * 1024:
                raise VerifierFailure("invalid_evaluator_result", "private case evidence is absent or oversized")
            case_evidence = json.loads(private_case_path.read_text())
            verdict = aggregate_private_evidence(case_evidence, 15 if mode == "train" else 30)
            if mode == "train":
                # Adapt the existing private evidence shape without changing
                # its exact comparison or disclosing workload identities.
                rows = [
                    {"workload": name, "metrics": {"gap": candidate >= reference}}
                    for name, candidate, reference in zip(
                        case_evidence["case_ids"], case_evidence["candidate_gaps"],
                        case_evidence["reference_gaps"])
                ]
                visible_metrics = _feedback.public_metrics(
                    {"mean_gap": metrics["mean_gap"], "instance_count": metrics["instance_count"],
                     "beats_reference": verdict["beats_reference"], "beat_spec_workload_passes": rows},
                    beat_spec_path=task_dir / "beat_spec.json",
                    configured_visible=("mean_gap", "instance_count"),
                )
            else:
                private_metrics = verdict
        else:
            if score != 0 or unbounded != 0 or private_case_path.exists():
                raise VerifierFailure("invalid_evaluator_result", "invalid candidate result is inconsistent")
            count = 15 if mode == "train" else 30
            progress = {"gap": {"passed": 0, "failed": count, "total": count,
                                "pass_percent": 0.0, "fail_percent": 100.0}}
            # Candidate invalidity establishes FAIL; the reference was not run.
            if mode == "train":
                visible_metrics = {"beat_spec_progress": progress}
            else:
                private_metrics = {"beats_reference": 0, "beat_spec_progress": progress}

        evidence = {
            "metrics.json": json.dumps({
                "mode": mode,
                "raw_score": score,
                "score_unbounded": unbounded,
                "message": message,
                # Absolute values accompany the separate private case evidence.
                "metrics": metrics,
                "payload_sha256": sha256_file(payload),
                "dataset_sha256": sha256_file(dataset) if dataset.is_file() else None,
                "elapsed_seconds": round(time.time() - started, 3),
            }, indent=2, sort_keys=True).encode("utf-8"),
        }
        if metrics:
            evidence["case-evidence.json"] = private_case_path.read_bytes()
        rejection_path = task_dir / "rejection-evidence.json"
        if rejection_path.exists():
            if metrics or not rejection_path.is_file() or rejection_path.stat().st_size > 4096:
                raise VerifierFailure("invalid_evaluator_result", "rejection evidence is inconsistent")
            evidence["rejection-evidence.json"] = rejection_path.read_bytes()
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


def main(bundle: Path, starter: Path, entrypoint: str) -> int:
    result_path: Path | None = None
    try:
        result_path = _required_path("FRONTIER_RESULT_PATH")
        verify(
            bundle=bundle,
            starter=starter,
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
        return 1
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
