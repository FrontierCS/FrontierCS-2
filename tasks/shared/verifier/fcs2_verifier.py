"""The shared FrontierCS2 Verifier: every task's judge runs through this one adapter (#757).

The contract with a task's judge (judge/evaluator.py, run from a writable copy of the bundle):

    evaluate(tree) -> (score, score_unbounded, message, metrics)
        `tree` is the pristine starter (bundle pristine/) with only the submission's allowed
        files overlaid. The task builds and measures it; it never sees the patch. A rejection
        only a build or run can find returns (0.0, 0.0, reason, {"rejected": <category>}).
    check_tree(tree, pristine, changes) -> None | (category, paths)      optional
        Task rules the declarative edit-scope rules cannot express (an AST policy, "only this
        function body may change"). It returns one of edit_scope.CATEGORIES to reject.

Before either runs, this adapter applies the patch with the shared edit_scope module and the
task's rules (config.yaml, generated from task.toml). A rejection by edit_scope or check_tree
is a scored zero with one shared public category, the same for every task; evaluate() then
never runs.

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

**The Driver bearer never reaches task code.** The wrapper starts this process with
FRONTIER_VERIFIER_TOKEN in its environment; the same bearer authorizes the result and
evidence PUTs. main() removes it (and every other FRONTIER_* credential) from os.environ
and marks the process non-dumpable before any judge module is imported, so neither an
in-process import of candidate code nor any child process can read it from the
environment or from /proc/<pid>/environ. Trusted shared code asks for it with
verifier_token(): the GPU adapter (which never forwards it to a child) and a judge's
model-gateway call (task.toml additional_keys, inventory token_env). task.toml `[gpu.verifier] placement` picks how (fcs2_verifier_gpu.py has the
contract): "runner" ships this whole verification to one GPU runner; "control" runs it here
and evaluate(tree) runs commands on its declared sandbox runners with
`fcs2_verifier_gpu.run_sandboxes`.

Three things this adapter is careful about, because each is a place where a
faithful-looking translation would silently change the task:

**Mode is the suite selector, not a hint.** Our evaluators choose their suite
from `FRONTIER_SUBMISSION_ROLE`, and every one of them tests `== "final"`. The
platform's `eval` maps to `final` and its `train` maps to anything else; passing
the platform's own word through would put every task on its development suite
during grading.

**Private metrics do not go back to the solver.** `beats_reference`, reference
measurements, margins, and exact per-workload pass/fail decisions remain in
evidence. In eval/final mode, scored tasks also emit an
operator-only `private_metrics` result object designed for transport to
Management, never to the solver; platform support ships separately. In train
mode, a result returns only the metrics task.toml [feedback] names (config.yaml
`evaluation.solver_visible_metrics`) and aggregate pass/fail percentages.

**The evaluator needs a writable task tree.** Ours resolve `TASK_DIR` from
`__file__` and build inside it. The Verifier image is read-only, so the immutable
bundle is copied into a bounded scratch first and the evaluator is imported from
the copy.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
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
        self._failure_evidence_written = False
        self.private_evidence = private_evidence


# ---------------------------------------------------------------- environment

# The verification-scoped Driver bearer: it authorizes the result and evidence PUTs, the input
# GETs and the Verifier GPU runner API, so task code must never hold it.
VERIFIER_TOKEN = "FRONTIER_VERIFIER_TOKEN"
# Any other credential-shaped platform variable is dropped as well, never kept.
_CREDENTIAL_NAME = re.compile(r"FRONTIER_[A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PRIVATE_KEY|API_KEY)")
_PR_SET_DUMPABLE = 4
_sealed: dict[str, str] = {}


def seal_credentials() -> None:
    """Take every FRONTIER_* credential out of the environment before any task code is imported
    or run. Called first by main().

    The Driver bearer is kept, in this module only, when shared code needs it: a Verifier-GPU
    control Pod (FRONTIER_VERIFIER_GPU_TYPE) or a judge with model keys (FRONTIER_LITELLM_INVENTORY
    names it as the gateway caller token). Otherwise it is discarded.

    Popping alone is not enough: /proc/<pid>/environ shows the block this process was started
    with, which any process of the same uid (a candidate binary the judge spawns) may read, like
    /proc/<pid>/mem, while the process is dumpable. So the popped entries are also zeroed in that
    block (libc no longer points at them) and the process is made non-dumpable."""
    keep = bool(os.environ.get("FRONTIER_VERIFIER_GPU_TYPE")
                or os.environ.get("FRONTIER_LITELLM_INVENTORY"))
    names = [name for name in os.environ if _CREDENTIAL_NAME.fullmatch(name)]
    for name in names:
        value = os.environ.pop(name)
        if name == VERIFIER_TOKEN and value and keep:
            _sealed[name] = value
    _wipe_initial_environment(names)
    _make_undumpable()


def _wipe_initial_environment(names: list[str]) -> None:
    """Zero `NAME=value` for each name in the process's initial environment block (the bytes
    /proc/self/environ returns). Safe after os.environ.pop: unsetenv dropped libc's pointers."""
    if not names:
        return
    try:
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        start, end = int(fields[47]), int(fields[48])  # proc(5) stat fields 50, 51: env_start, env_end
        block = ctypes.string_at(start, end - start)
    except (OSError, ValueError, IndexError) as error:
        raise VerifierFailure("verifier_configuration_error",
                              "the Verifier could not protect its Driver credential") from error
    offset = 0
    for entry in block.split(b"\0"):
        if entry.split(b"=", 1)[0].decode("ascii", "replace") in names:
            ctypes.memset(start + offset, 0, len(entry))
        offset += len(entry) + 1


def _make_undumpable() -> None:
    if ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise VerifierFailure("verifier_configuration_error",
                              "the Verifier could not protect its Driver credential")


def verifier_token() -> str | None:
    """The sealed Driver bearer, for trusted shared code only (the GPU adapter, a judge's
    model-gateway header). None where the platform gave none (a GPU runner's sshd session,
    a local run). Never put it in an environment, argv or file."""
    return _sealed.get(VERIFIER_TOKEN)


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
    # An empty patch is the edit scope's decision (a task may allow it), not a payload error.
    return candidate


# -------------------------------------------------------------- task bundle


def load_edit_scope(bundle: Path):
    """The shared edit-scope module and this task's rules (config.yaml `edit_scope`)."""
    # Beside this file in the Verifier base (/opt/frontier); tasks/shared/lib in the repository.
    here = Path(__file__).resolve().parent
    for directory in (here, here.parent / "lib"):
        if (directory / "edit_scope.py").is_file():
            sys.path.insert(0, str(directory))
            break
    import edit_scope

    try:
        config = json.loads((bundle / "config.yaml").read_text(encoding="utf-8"))
        rules = edit_scope.Rules.from_mapping(config["edit_scope"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise VerifierFailure("verifier_configuration_error",
                              "the task's edit-scope rules are missing or invalid") from error
    if not (bundle / "pristine").is_dir():
        raise VerifierFailure("verifier_configuration_error", "the task bundle has no pristine/")
    return edit_scope, rules


def evaluation_limits(bundle: Path) -> tuple[list[str], int]:
    """From config.yaml `evaluation` (generated from task.toml): the evaluator metrics a train
    result may return (`solver_visible_metrics`, task.toml [feedback]) and the cap on the
    evaluator's metrics (`max_metrics_bytes`, task.toml [verifier])."""
    try:
        config = json.loads((bundle / "config.yaml").read_text(encoding="utf-8"))
        visible = config["evaluation"]["solver_visible_metrics"]
        cap = config["evaluation"]["max_metrics_bytes"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise VerifierFailure("verifier_configuration_error",
                              "the task's evaluation limits are missing") from error
    if not isinstance(visible, list) or not all(isinstance(name, str) for name in visible):
        raise VerifierFailure("verifier_configuration_error",
                              "the task's solver-visible metrics are invalid")
    if type(cap) is not int or cap <= 0:
        raise VerifierFailure("verifier_configuration_error",
                              "the task's max_metrics_bytes is invalid")
    return visible, cap


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
    check_tree = getattr(module, "check_tree", None)
    if check_tree is not None and not callable(check_tree):
        raise VerifierFailure("verifier_configuration_error", "check_tree is not callable")

    def in_context(function):
        if function is None:
            return None

        @wraps(function)
        def wrapped(*args, **kwargs):
            with import_context():
                return function(*args, **kwargs)
        return wrapped

    return in_context(evaluate), in_context(check_tree)


# ------------------------------------------------------------------- scoring


def _bounded(value: Any, depth: int = 0, *, max_string_chars: int = MAX_MESSAGE_CHARS) -> Any:
    if depth > 6:
        raise VerifierFailure("invalid_evaluator_result", "metrics are nested too deeply")
    if value is None or isinstance(value, (bool, str)):
        return value[:max_string_chars] if isinstance(value, str) else value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VerifierFailure("invalid_evaluator_result", "metrics contain a non-finite number")
        return value
    if isinstance(value, dict):
        return {str(key)[:256]: _bounded(item, depth + 1, max_string_chars=max_string_chars)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bounded(item, depth + 1, max_string_chars=max_string_chars) for item in value]
    raise VerifierFailure("invalid_evaluator_result", "metrics contain an unsupported value")


def run_evaluator(evaluate, tree: Path, mode: str, *,
                  max_metrics_bytes: int) -> tuple[float, float, str, dict]:
    """Call the task's evaluator with the role its own code expects.

    `eval` is `final`; anything else is a development submission. Our evaluators
    all test `FRONTIER_SUBMISSION_ROLE == "final"`, so passing the platform's own
    word through would silently grade every task on its development suite.
    """
    previous = os.environ.get("FRONTIER_SUBMISSION_ROLE")
    os.environ["FRONTIER_SUBMISSION_ROLE"] = "final" if mode == "eval" else "development"
    try:
        result = evaluate(str(tree))
    except Exception as error:  # noqa: BLE001 - a task's own crash is a task failure
        gpu = sys.modules.get("fcs2_verifier_gpu")
        if gpu is not None and isinstance(error, gpu.VerifierGpuError):
            # A sandbox runner or transport failure: unscored, like the one-runner placement.
            raise VerifierFailure(
                "verifier_gpu_error", "Verifier GPU sandbox run failed",
                private_evidence={"verifier_gpu": error.private_evidence
                                  or {"failure": type(error).__name__[:256]}}) from error
        execution = getattr(error, "private_execution_evidence", None)
        if execution is not None:
            execution = _bounded(execution, max_string_chars=MAX_PRIVATE_EXECUTION_BYTES)
            body = json.dumps(execution, allow_nan=False, sort_keys=True).encode("utf-8")
            if not isinstance(execution, dict) or len(body) > MAX_PRIVATE_EXECUTION_BYTES:
                raise VerifierFailure("invalid_evaluator_result", "private execution evidence exceeds its contract") from error
            raise VerifierFailure(
                "invalid_execution_evidence", "the task did not establish valid numerical execution",
                private_evidence={"private_execution": execution}) from error
        # The public result keeps only the exception class. Preserve the bounded
        # cause in private evidence too: a GPU command must return BOTH artifacts
        # or its missing evidence gets misclassified as a transport failure.
        tail = bytearray()
        diagnostic = traceback.TracebackException.from_exception(
            error, limit=8, capture_locals=False)
        for part in diagnostic.format(chain=True):
            tail.extend(part.encode("utf-8", errors="replace"))
            del tail[:-MAX_PRIVATE_EXCEPTION_BYTES]
        private_tail = tail.decode("utf-8", errors="ignore")
        print("frontier-verifier private evaluator exception tail:\n"
              + private_tail, file=sys.stderr, flush=True)
        raise VerifierFailure(
            "evaluator_error",
            f"the task evaluator raised {type(error).__name__}",
            private_evidence={"evaluator_exception": {
                "exception_type": type(error).__name__[:256],
                "traceback_tail": private_tail,
            }}) from error
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
    if metrics_size > max_metrics_bytes:
        raise VerifierFailure("invalid_evaluator_result", "metrics exceed the evidence limit")
    return float(score), float(unbounded), message[:MAX_MESSAGE_CHARS], bounded


# ------------------------------------------------------------------- outputs


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, allow_nan=False, ensure_ascii=True,
                      separators=(",", ":"), sort_keys=True).encode("ascii")


def _private_metrics(metrics: dict) -> dict | None:
    try:
        result = beat_spec_private_metrics(metrics)
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


def _write_rejection(rejection, edit_scope, metrics: dict, *, mode: str, payload: Path,
                     dataset: Path, started: float, result_path: Path,
                     evidence_path: Path) -> dict:
    """A submission outside the task's rules: a scored zero with one shared public category
    (and the offending paths), the same for every task; the evaluator never runs."""
    category, paths = rejection
    shown = f" ({', '.join(paths[:5])}{', ...' if len(paths) > 5 else ''})" if paths else ""
    message = edit_scope.CATEGORIES[category] + shown
    write_evidence(evidence_path, {"metrics.json": json.dumps({
        "mode": mode, "raw_score": 0.0, "score_unbounded": 0.0, "message": message,
        "rejection": {"category": category, "paths": list(paths)}, "edit_scope": metrics,
        "payload_sha256": sha256_file(payload),
        "dataset_sha256": sha256_file(dataset) if dataset.is_file() else None,
        "elapsed_seconds": round(time.time() - started, 3),
    }, indent=2, sort_keys=True).encode("utf-8")})
    result = {"status": "passed", "raw_score": 0.0, "summary": message[:MAX_MESSAGE_CHARS],
              "metrics": {"score_unbounded": 0.0, "rejected": category}}
    if mode == "eval":
        result["private_metrics"] = {"beats_reference": 0, "rejected": category}
    write_json(result_path, result)
    return result


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
        extracted = scratch / "payload"
        extract_payload(payload, extracted)
        patch = entrypoint_path(extracted, entrypoint).read_bytes()
        edit_scope, rules = load_edit_scope(bundle)
        visible, max_metrics_bytes = evaluation_limits(bundle)
        task_dir = stage_task(bundle, scratch)
        evaluate, check_tree = load_evaluator(task_dir)

        # The edit scope is the verifier's, not the task's: one strict application of the
        # patch, the changed files read back from git, and a judged tree that is the
        # pristine starter with only the allowed files overlaid.
        scope = edit_scope.check(patch, task_dir / "pristine", rules)
        try:
            rejection = None
            if not scope.ok:
                rejection = (scope.category, scope.paths)
            elif check_tree is not None:
                try:
                    verdict = check_tree(str(scope.tree), str(task_dir / "pristine"),
                                         list(scope.changes))
                except Exception as error:  # noqa: BLE001 - the task's own hook failed
                    raise VerifierFailure("invalid_evaluator_result",
                                          f"check_tree raised {type(error).__name__}") from error
                if verdict is not None:
                    category, paths = verdict
                    if category not in edit_scope.CATEGORIES or not isinstance(paths, (list, tuple)):
                        raise VerifierFailure("invalid_evaluator_result",
                                              "check_tree must return (category, paths) or None")
                    rejection = (category, tuple(str(p) for p in paths))
            if rejection is not None:
                return _write_rejection(rejection, edit_scope, scope.metrics, mode=mode,
                                        payload=payload, dataset=dataset, started=started,
                                        result_path=result_path, evidence_path=evidence_path)
            score, unbounded, message, metrics = run_evaluator(
                evaluate, scope.tree, mode, max_metrics_bytes=max_metrics_bytes)
            if "rejected" in metrics:
                # A rejection only the build or a run can find (e.g. a forbidden launch
                # surface): reported with the same shared categories as the edit scope.
                category = metrics["rejected"]
                if (set(metrics) != {"rejected"} or score != 0.0 or unbounded != 0.0
                        or category not in edit_scope.CATEGORIES):
                    raise VerifierFailure("invalid_evaluator_result",
                                          "a rejection is a scored zero whose only metric is a "
                                          "shared category")
                paths = tuple(c.path for c in scope.changes)
                return _write_rejection((category, paths), edit_scope, scope.metrics, mode=mode,
                                        payload=payload, dataset=dataset, started=started,
                                        result_path=result_path, evidence_path=evidence_path)
        except VerifierFailure as failure:
            private_evidence = failure.private_evidence
            if private_evidence is None:
                # Validation has already rejected this result. Never serialize
                # its untrusted/oversized/deep values merely to retain an error.
                private_evidence = {"evaluator_validation": {"code": failure.code}}
            # Keep invalid execution, output and runtime failures in the archive.
            # A normal command return transports this failed result through SSH.
            write_evidence(evidence_path, {"metrics.json": json.dumps({
                "mode": mode, "execution_valid": False, "metrics": private_evidence,
                "payload_sha256": sha256_file(payload),
                "dataset_sha256": sha256_file(dataset) if dataset.is_file() else None,
                "elapsed_seconds": round(time.time() - started, 3),
            }, allow_nan=False, sort_keys=True).encode("utf-8")})
            if failure.private_evidence is None:
                failure._failure_evidence_written = True
                raise
            message = failure.message[:MAX_MESSAGE_CHARS]
            result = {"status": "failed", "summary": message, "metrics": {},
                      "failure": {"code": failure.code, "message": message, "retryable": False}}
            write_json(result_path, result)
            return result
        finally:
            scope.dispose()

        visible_metrics: dict[str, Any] = {}
        private_metrics: dict[str, Any] | None = None
        if mode == "train" and metrics:
            try:
                visible_metrics = beat_spec_public_metrics(metrics, visible=visible)
            except ValueError as error:
                raise VerifierFailure("invalid_evaluator_result", str(error)) from error
        elif mode == "eval" and "beats_reference" in metrics:
            private_metrics = _private_metrics(metrics)
        elif mode == "eval" and score == 0.0:
            # A validity failure the evaluator scored zero without the beat-spec fields: like
            # a rejection, the final result still records that it did not beat the reference.
            private_metrics = {"beats_reference": 0}

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
            # The exact workload decisions stay in evidence. Train may add only
            # the task's [feedback] values and aggregate percentages; eval exposes
            # none of them.
            "metrics": {"score_unbounded": unbounded, **visible_metrics},
        }
        if private_metrics is not None:
            result["private_metrics"] = private_metrics
        write_json(result_path, result)
        return result
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _write_gpu_phase_evidence(path: Path, value: dict) -> None:
    body = json.dumps(value, allow_nan=False, sort_keys=True).encode("utf-8")
    if len(body) > 4096 or len(value.get("events", [])) > 16:
        raise VerifierFailure("verifier_gpu_error", "Verifier phase evidence exceeded its bound")
    temporary = path.with_name(f".{path.name}.tmp")
    write_evidence(temporary, {"control-phases.json": body})
    if temporary.stat().st_size > min(4096, int(os.environ["FRONTIER_MAX_EVIDENCE_BYTES"])):
        raise VerifierFailure("verifier_gpu_error", "Verifier phase archive exceeded its bound")
    temporary.replace(path)


def _gpu_placement() -> str:
    try:
        from fcs2_verifier_gpu import placement
        return placement()
    except Exception as error:  # noqa: BLE001 - the shared adapter's configuration
        raise VerifierFailure("verifier_gpu_error",
                              "Verifier GPU configuration is invalid") from error


def main(bundle: Path, entrypoint: str) -> int:
    result_path: Path | None = None
    try:
        result_path = _required_path("FRONTIER_RESULT_PATH")
        seal_credentials()
        # The CPU control Pod inherits the wrapper's scoped bearer. The same
        # image runs on Modal, but sshd starts with a cleared environment and
        # therefore has no bearer; that is the non-recursive execution signal.
        if (os.environ.get("FRONTIER_VERIFIER_GPU_TYPE")
                and verifier_token()
                and _gpu_placement() == "runner"):
            evidence_path = _required_path("FRONTIER_EVIDENCE_PATH")
            checkpoint = evidence_path.with_name(f".{evidence_path.name}.control")
            try:
                from fcs2_verifier_gpu import VerifierGpuError, run as run_on_gpu

                if _zstandard is None:
                    raise VerifierGpuError("the Verifier control evidence codec is unavailable")
                run_on_gpu(
                    payload=_required_path("FRONTIER_PAYLOAD_PATH"),
                    dataset=_required_path("FRONTIER_DATASET_PATH"),
                    mode=_mode(),
                    result_path=result_path,
                    evidence_path=evidence_path,
                    record_evidence=lambda value: _write_gpu_phase_evidence(checkpoint, value),
                )
                return 0
            except VerifierGpuError as error:
                # Atomically promote the last bounded checkpoint if a final
                # evidence write fails. It may say "started", never a guessed cause.
                try:
                    if error.private_evidence is not None:
                        _write_gpu_phase_evidence(checkpoint, error.private_evidence)
                except Exception:
                    pass  # Preserve the previous complete checkpoint and failure.
                try:
                    if checkpoint.is_file():
                        checkpoint.replace(evidence_path)
                except OSError:
                    pass  # Filesystem loss cannot be represented as retained evidence.
                # The GPU adapter deliberately emits only bounded lifecycle and
                # transport messages; preserve those in the platform result so
                # an operator can distinguish quota, API, SSH, and evaluator
                # failures without exposing credentials or provider responses.
                raise VerifierFailure(
                    "verifier_gpu_error",
                    "Verifier GPU control failed") from error
            except Exception as error:  # noqa: BLE001 - shared lifecycle adapter
                try:
                    if checkpoint.is_file():
                        checkpoint.replace(evidence_path)
                except OSError:
                    pass
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
        # A handled GPU lifecycle error, or evaluator rejection with retained
        # evidence, is a structured failed result. A zero process status lets
        # the wrapper transport it; it is not a score or successful evaluation.
        return 0 if failure.code == "verifier_gpu_error" or failure._failure_evidence_written else 1
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
