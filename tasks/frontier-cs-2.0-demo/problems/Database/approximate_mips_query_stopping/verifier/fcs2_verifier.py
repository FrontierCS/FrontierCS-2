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

What it must never do: call Driver endpoints, reach S3, reuse wrapper
credentials, or write outside `/frontier`. The wrapper owns all of that.

Three things this adapter is careful about, because each is a place where a
faithful-looking translation would silently change the task:

**Mode is the suite selector, not a hint.** Our evaluators choose their suite
from `FRONTIER_SUBMISSION_ROLE`, and every one of them tests `== "final"`. The
platform's `eval` maps to `final` and its `train` maps to anything else; passing
the platform's own word through would put every task on its development suite
during grading.

**Private metrics do not go back to the solver.** `beats_reference`, reference
measurements, margins, and exact per-workload pass/fail decisions go only to
evidence. In train mode, tasks with a beat spec also return its approved metric
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

RESULT_SCHEMA_HINT = "frontier.verifier-result/v1"
MAX_PAYLOAD_BYTES = 65536
MAX_PAYLOAD_MEMBERS = 2
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
                if str(relative) not in {"payload", "payload/solution.patch"}:
                    raise VerifierFailure("invalid_submission", "unexpected payload member")
                if (str(relative) == "payload" and not member.isdir()) or (
                        str(relative) == "payload/solution.patch" and not member.isfile()):
                    raise VerifierFailure("invalid_submission", "invalid payload member type")
                if member.size > MAX_PAYLOAD_BYTES:
                    raise VerifierFailure("invalid_submission", "payload member exceeds limit")
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
    try:
        import zstandard

        body = zstandard.ZstdCompressor(level=10).compress(raw)
    except ImportError:
        # The platform reads the member as `.tar.zst`. Rather than ship bytes
        # whose name lies about their encoding, fail loudly at build time.
        raise VerifierFailure(
            "verifier_configuration_error",
            "the verifier image must provide the `zstandard` package to write "
            "evidence.tar.zst")
    path.write_bytes(body)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


# ---------------------------------------------------------------------- main


def task_public_metrics(metrics: dict[str, Any], *, beat_spec_path: Path,
                        mode: str = "train") -> dict[str, Any]:
    """Keep exact task coverage at the adapter boundary around generic feedback."""
    spec = json.loads(beat_spec_path.read_text())
    workloads = spec.get("workloads")
    if mode == "eval" and isinstance(workloads, list):
        return final_metrics(metrics, spec, beat_spec_path=beat_spec_path)
    result = beat_spec_public_metrics(metrics, beat_spec_path=beat_spec_path)
    if not isinstance(workloads, list) or not workloads or any(
            not isinstance(row, dict) or not isinstance(row.get("name"), str)
            or not row["name"].strip() for row in workloads):
        raise ValueError("beat_spec.json must declare named workloads")
    expected = [row["name"] for row in workloads]
    observed = [row["workload"] for row in metrics["beat_spec_workload_passes"]]
    # The generic helper already rejects duplicate rows and inconsistent verdicts.
    if len(set(expected)) != len(expected) or set(observed) != set(expected):
        raise ValueError("metrics.beat_spec_workload_passes must contain exactly the declared workloads")
    return result


def final_metrics(metrics: dict[str, Any], spec: dict[str, Any], *,
                  beat_spec_path: Path) -> dict[str, Any]:
    """Final coverage: every metric on the disclosed workloads, and only the
    declared deterministic metrics on the final-only workloads."""
    def names(field: str) -> list[str]:
        rows = spec.get(field)
        if not isinstance(rows, list) or not rows or any(
                not isinstance(row, dict) or not isinstance(row.get("name"), str)
                or not row["name"].strip() for row in rows):
            raise ValueError(f"beat_spec.json must declare named {field}")
        found = [row["name"] for row in rows]
        if len(set(found)) != len(found):
            raise ValueError(f"beat_spec.json {field} contains a duplicate name")
        return found

    disclosed, final_only = names("workloads"), names("final_only_workloads")
    metric_names = names("metrics")
    subset = spec.get("final_only_metrics")
    if (not isinstance(subset, list) or not subset or len(set(subset)) != len(subset)
            or not set(subset) <= set(metric_names) or set(disclosed) & set(final_only)):
        raise ValueError("beat_spec.json must declare final-only metrics among its metrics")
    rows = metrics.get("beat_spec_workload_passes")
    if not isinstance(rows, list) or any(
            not isinstance(row, dict) or not isinstance(row.get("workload"), str)
            for row in rows):
        raise ValueError("metrics.beat_spec_workload_passes must be a list of workload rows")
    observed = [row["workload"] for row in rows]
    if len(set(observed)) != len(observed) or set(observed) != set(disclosed) | set(final_only):
        raise ValueError("metrics.beat_spec_workload_passes must contain exactly the declared workloads")
    extra_rows = [row for row in rows if row["workload"] in set(final_only)]
    for row in extra_rows:
        decisions = row.get("metrics")
        if (not isinstance(decisions, dict) or set(decisions) != set(subset)
                or any(not isinstance(value, bool) for value in decisions.values())):
            raise ValueError(
                "final-only workload rows must contain exactly the final-only metrics")
    verdict = metrics.get("beats_reference")
    if type(verdict) is not int or verdict not in (0, 1):
        raise ValueError("metrics.beats_reference must be 0 or 1")
    disclosed_rows = [row for row in rows if row["workload"] in set(disclosed)]
    # The generic helper validates the disclosed rows against their own conjunction.
    disclosed_verdict = int(all(
        isinstance(row.get("metrics"), dict) and all(row["metrics"].values())
        for row in disclosed_rows))
    result = beat_spec_public_metrics(
        {**metrics, "beat_spec_workload_passes": disclosed_rows,
         "beats_reference": disclosed_verdict},
        beat_spec_path=beat_spec_path)
    if verdict != int(disclosed_verdict and all(
            value for row in extra_rows for value in row["metrics"].values())):
        raise ValueError(
            "metrics.beats_reference does not equal the aggregation declared by beat_spec.json")
    progress = result["beat_spec_progress"]
    for name in subset:
        passed = progress[name]["passed"] + sum(row["metrics"][name] for row in extra_rows)
        total = progress[name]["total"] + len(extra_rows)
        progress[name] = {
            "passed": passed, "failed": total - passed, "total": total,
            "pass_percent": round(100.0 * passed / total, 2),
            "fail_percent": round(100.0 * (total - passed) / total, 2),
        }
    return result


def verify(*, bundle: Path, entrypoint: str, payload: Path, dataset: Path,
           mode: str, result_path: Path, evidence_path: Path) -> dict:
    started = time.time()
    scratch = Path(tempfile.mkdtemp(prefix="frontier-verify-", dir="/frontier"
                                    if Path("/frontier").is_dir() else None))
    try:
        if mode not in ("train", "eval"):
            raise VerifierFailure("verifier_configuration_error", "invalid verification mode")
        try:
            selector = json.loads(dataset.read_text())
        except (OSError, ValueError):
            raise VerifierFailure("verifier_configuration_error", "invalid dataset selector")
        expected_role = "final" if mode == "eval" else "development"
        if not isinstance(selector, dict) or (
                selector.get("schema_version") != "frontier.frontiersmith-dataset/v1"
                or selector.get("problem_id") != "approximate_mips_query_stopping"
                or selector.get("mode") != mode
                or selector.get("selects") != {"FRONTIER_SUBMISSION_ROLE": expected_role}):
            raise VerifierFailure("verifier_configuration_error", "dataset selector does not match mode")
        extracted = scratch / "payload"
        extract_payload(payload, extracted)
        submission = entrypoint_path(extracted, entrypoint)
        task_dir = stage_task(bundle, scratch)
        evaluate = load_evaluator(task_dir)
        score, unbounded, message, metrics = run_evaluator(evaluate, submission, mode)

        visible_metrics: dict[str, Any] = {}
        if mode == "train" and (task_dir / "beat_spec.json").is_file():
            try:
                visible_metrics = task_public_metrics(
                    metrics,
                    beat_spec_path=task_dir / "beat_spec.json",
                )
            except ValueError as error:
                raise VerifierFailure("invalid_evaluator_result", str(error)) from error

        evidence = {
            "metrics.json": json.dumps({
                "mode": mode,
                "raw_score": score,
                "score_unbounded": unbounded,
                "message": message,
                # The full dict, including every reference-relative key. It is
                # here and not in the result because evidence is for the people
                # auditing the task, and the result is for the solver.
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
            "summary": "Evaluation completed; detailed measurements are operator-only.",
            # The exact reference-relative decisions stay in evidence. Train may
            # add only the beat spec's approved values and aggregate percentages;
            # eval exposes none of them.
            "metrics": {"score_unbounded": unbounded, **visible_metrics} if mode == "train" else {},
        }
        measured_fields = {"mean_recall", "mean_kratio", "mean_time_ms", "mean_candidates"}
        if mode == "eval" and measured_fields <= metrics.keys():
            verdict = metrics.get("beats_reference")
            values = (metrics.get("reference_score"), metrics.get("margin"))
            aggregation = metrics.get("aggregation")
            if aggregation != "public_plus_hidden_pooled_by_workload":
                raise VerifierFailure(
                    "invalid_evaluator_result", "final metrics do not cover public plus hidden")
            if type(verdict) is not int or verdict not in (0, 1) or any(
                    isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) for v in values):
                raise VerifierFailure("invalid_evaluator_result", "invalid private verdict")
            result["private_metrics"] = {
                "beats_reference": verdict,
                "reference_score": values[0],
                "margin": values[1],
                "beat_spec_progress": task_public_metrics(
                    metrics, beat_spec_path=task_dir / "beat_spec.json", mode="eval"
                )["beat_spec_progress"],
            }
        write_json(result_path, result)
        return result
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def check_cpu_compatibility(cpuinfo: Path = Path("/proc/cpuinfo")) -> None:
    """Classify a known source hardware mismatch before scientific evaluation."""
    try:
        processors = [set(line.split(":", 1)[1].split())
                      for line in cpuinfo.read_text().splitlines()
                      if line.startswith("flags") and ":" in line]
    except OSError as error:
        raise VerifierFailure("verifier_configuration_error", "CPU capabilities are unavailable") from error
    required = {"avx2", "avx512f", "avx512dq", "avx512bw"}
    if not processors or any(not required <= flags for flags in processors):
        raise VerifierFailure(
            "verifier_configuration_error",
            "The preserved FARGO workload requires AVX2 and AVX-512F/DQ/BW CPU support")


def main(bundle: Path, entrypoint: str) -> int:
    result_path: Path | None = None
    try:
        result_path = _required_path("FRONTIER_RESULT_PATH")
        check_cpu_compatibility()
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
