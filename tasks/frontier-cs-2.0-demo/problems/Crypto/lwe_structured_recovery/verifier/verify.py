#!/usr/bin/env python3
"""FrontierCS2 verifier for the public structured-LWE recovery task."""

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
import traceback
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


PROBLEM_ID = "lwe_structured_recovery"
DATASET_SCHEMA = "frontier.lwe-structured-recovery-dataset/v1"
CATALOG_SHA256 = "318f32b68e8a30c6bcd5867c89eb10d87a5472614401f4e6df0f55356cdc98a6"
ENTRYPOINT = PurePosixPath("payload/solution.patch")
ANSWER = "payload/solution.json"
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024
# The patch is the only archive member; leave room for tar headers and padding.
MAX_PATCH_BYTES = MAX_PAYLOAD_BYTES - 16 * 1024
MAX_SOLUTION_BYTES = 2_000_000
MAX_DATASET_BYTES = 4096
MAX_EVIDENCE_BYTES = 1024 * 1024
MAX_SUMMARY_CHARS = 4096
MAX_PRIVATE_METRICS_BYTES = 16 * 1024
JUDGE_ROOT = Path(os.environ.get("FRONTIER_LWE_JUDGE", Path(__file__).resolve().parent.parent / "judge"))
PUBLIC_METRICS = frozenset(
    {
        "instance_count",
        "solved_count",
        "submitted_count",
        "invalid_count",
        "duplicate_count",
        "conflict_count",
        "unknown_count",
        "rejection_code_counts",
        "invalid_examples",
        "solved_ids",
    }
)


class VerificationFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise VerificationFailure("verifier_configuration_error", "missing scoring adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationFailure("invalid_dataset", "dataset contains a duplicate key")
        result[key] = value
    return result


def _required_path(name: str) -> Path:
    value = os.environ.get(name, "")
    if not value.startswith("/"):
        raise VerificationFailure("verifier_configuration_error", f"{name} must be absolute")
    return Path(value)


def _output_path(name: str) -> Path:
    path = _required_path(name)
    try:
        path.relative_to("/frontier/output")
    except ValueError as error:
        raise VerificationFailure(
            "verifier_configuration_error", f"{name} must be below /frontier/output"
        ) from error
    return path


def _mode() -> str:
    mode = os.environ.get("FRONTIER_VERIFICATION_MODE", "")
    if mode not in {"train", "eval"}:
        raise VerificationFailure(
            "verifier_configuration_error", "FRONTIER_VERIFICATION_MODE must be train or eval"
        )
    return mode


def _read_regular(path: Path, maximum: int, code: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise VerificationFailure(code, f"{path.name} is not a regular file")
        size = path.stat().st_size
        if not 0 < size <= maximum:
            raise VerificationFailure(code, f"{path.name} exceeds its byte limit")
        with path.open("rb") as source:
            body = source.read(maximum + 1)
    except VerificationFailure:
        raise
    except OSError as error:
        raise VerificationFailure(code, f"{path.name} is not readable") from error
    if len(body) > maximum:
        raise VerificationFailure(code, f"{path.name} exceeds its byte limit")
    return body


def _load_dataset(path: Path, mode: str) -> dict[str, str]:
    body = _read_regular(path, MAX_DATASET_BYTES, "invalid_dataset")
    try:
        value = json.loads(body, object_pairs_hook=_reject_duplicate_keys)
    except VerificationFailure:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise VerificationFailure("invalid_dataset", "dataset is not valid JSON") from error
    expected = {"schema_version", "problem_id", "mode", "catalog_sha256"}
    if not isinstance(value, dict) or set(value) != expected:
        raise VerificationFailure("invalid_dataset", "dataset fields do not match the schema")
    if value != {
        "schema_version": DATASET_SCHEMA,
        "problem_id": PROBLEM_ID,
        "mode": mode,
        "catalog_sha256": f"sha256:{CATALOG_SHA256}",
    }:
        raise VerificationFailure("invalid_dataset", "dataset identity does not match this run")
    return value


def _extract_solution(payload: Path, starter: Path, destination: Path) -> Path:
    """Apply the submitted patch to the pristine starter ledger."""
    patching = _module(Path(__file__).with_name("patch_payload.py"), "lwe_patch_payload")
    try:
        patch = patching.read_patch_member(payload, MAX_PAYLOAD_BYTES, MAX_PATCH_BYTES)
        files = patching.apply_patch(
            patch, starter, (ANSWER,), max_files=1, max_file_bytes=MAX_SOLUTION_BYTES
        )
    except patching.PatchRejected as error:
        raise VerificationFailure("invalid_submission", str(error)) from error
    except OSError as error:
        raise VerificationFailure(
            "verifier_configuration_error", "starter ledger is unavailable"
        ) from error
    solution = files[ANSWER]
    if not 0 < len(solution) <= MAX_SOLUTION_BYTES:
        raise VerificationFailure("invalid_submission", "solution.json exceeds its byte limit")
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "solution.json"
    path.write_bytes(solution)
    return path


def _catalog(root: Path):
    public = root / "repo"
    catalog_path = public / "catalog.jsonl"
    sidecar_path = public / "catalog.sha256"
    body = _read_regular(catalog_path, 64 * 1024 * 1024, "verifier_configuration_error")
    digest = hashlib.sha256(body).hexdigest()
    expected_sidecar = f"{digest}  catalog.jsonl\n".encode("ascii")
    sidecar = _read_regular(sidecar_path, 80, "verifier_configuration_error")
    if digest != CATALOG_SHA256 or sidecar != expected_sidecar:
        raise VerificationFailure(
            "verifier_configuration_error", "catalog assets do not match the frozen digest"
        )
    public_text = str(public)
    while public_text in sys.path:
        sys.path.remove(public_text)
    sys.path.insert(0, public_text)
    try:
        from lwe_challenge.evaluator_core import evaluate_path
        from lwe_challenge.schema import Catalog

        loaded = Catalog.load(catalog_path)
    except Exception as error:
        raise VerificationFailure(
            "verifier_configuration_error", "catalog could not be loaded"
        ) from error
    finally:
        if sys.path and sys.path[0] == public_text:
            sys.path.pop(0)
    if loaded.catalog_id != CATALOG_SHA256 or len(loaded.instances) != 200:
        raise VerificationFailure(
            "verifier_configuration_error", "catalog identity or instance count is invalid"
        )
    return loaded, evaluate_path


def _plain(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        raise VerificationFailure("invalid_evaluator_result", "metrics are nested too deeply")
    if value is None or isinstance(value, (bool, int, str)):
        return value[:MAX_SUMMARY_CHARS] if isinstance(value, str) else value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VerificationFailure("invalid_evaluator_result", "metrics are not finite")
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth + 1) for item in value]
    raise VerificationFailure("invalid_evaluator_result", "metrics contain an unsupported value")


def _evaluate(
    solution: Path, catalog: Any, evaluate_path: Any
) -> tuple[float, float, str, dict[str, Any]]:
    try:
        evaluated = evaluate_path(solution, catalog=catalog)
    except Exception as error:
        raise VerificationFailure("evaluator_error", "structured-LWE evaluator failed") from error
    score = float(evaluated.score)
    unbounded = float(evaluated.score_unbounded)
    if not math.isfinite(score) or score < 0 or not math.isfinite(unbounded) or unbounded < 0:
        raise VerificationFailure("invalid_evaluator_result", "score is not finite and non-negative")
    metrics = _plain(evaluated.metrics)
    if not isinstance(metrics, dict) or set(metrics) != PUBLIC_METRICS:
        raise VerificationFailure("invalid_evaluator_result", "public metric fields changed")
    return score, unbounded, str(evaluated.message)[:MAX_SUMMARY_CHARS], metrics


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_canonical(value))
    temporary.replace(path)


def _write_evidence(path: Path, value: Any) -> None:
    member = _canonical(value)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo("metrics.json")
        info.size = len(member)
        info.mtime = 0
        info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        archive.addfile(info, io.BytesIO(member))
    try:
        import zstandard
    except ImportError as error:
        raise VerificationFailure(
            "verifier_configuration_error", "Python zstandard module is unavailable"
        ) from error
    body = zstandard.ZstdCompressor(level=10).compress(buffer.getvalue())
    if not 0 < len(body) <= MAX_EVIDENCE_BYTES:
        raise VerificationFailure("invalid_evaluator_result", "evidence exceeds its byte limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def verify(
    *,
    root: Path,
    payload: Path,
    dataset: Path,
    mode: str,
    result_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    _load_dataset(dataset, mode)
    catalog, evaluate_path = _catalog(root)
    # The Driver contract guarantees only /frontier/output is a writable
    # emptyDir.  The verifier image itself and the rest of /frontier may be
    # read-only for this unprivileged process, so keep private scratch beside
    # the validated result path and remove it before returning.
    scratch_root = result_path.parent
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".frontier-lwe-", dir=scratch_root))
    try:
        solution = _extract_solution(payload, root / "starter", scratch / "submission")
        score, unbounded, summary, metrics = _evaluate(solution, catalog, evaluate_path)
        judge = _module(JUDGE_ROOT / "evaluator.py", "lwe_reference_comparison")
        feedback = _module(Path(__file__).with_name("beat_spec_feedback.py"), "lwe_feedback")
        decisions = judge.evaluate_reference_comparison(catalog, metrics, evaluate_path)
        spec_path = JUDGE_ROOT / "beat_spec.json"
        visible = [entry["name"] for entry in json.loads(spec_path.read_text())["solver_feedback"]]
        train_metrics = feedback.public_metrics(
            {**metrics, **decisions}, beat_spec_path=spec_path, configured_visible=visible
        )
        evidence = {
            "schema_version": "frontier.lwe-structured-recovery-evidence/v1",
            "problem_id": PROBLEM_ID,
            "mode": mode,
            "catalog_sha256": f"sha256:{CATALOG_SHA256}",
            "payload_sha256": _sha256(payload),
            "dataset_sha256": _sha256(dataset),
            "raw_score": score,
            "score_unbounded": unbounded,
            "summary": summary,
            "metrics": metrics,
            **decisions,
        }
        _write_evidence(evidence_path, evidence)
        result = {
            "status": "passed",
            "raw_score": score,
            "summary": summary,
            "metrics": (
                {"score_unbounded": unbounded, **train_metrics} if mode == "train" else {}
            ),
        }
        if mode == "eval":
            private_metrics = {
                "beats_reference": decisions["beats_reference"],
                "reference_score": decisions["reference_score"],
                "margin": score - decisions["reference_score"],
                "beat_spec_progress": train_metrics["beat_spec_progress"],
            }
            if len(_canonical(private_metrics)) > MAX_PRIVATE_METRICS_BYTES:
                raise VerificationFailure(
                    "invalid_evaluator_result", "private metrics exceed their byte limit"
                )
            result["private_metrics"] = private_metrics
        _write_json(result_path, result)
        return result
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    result_path: Path | None = None
    try:
        result_path = _output_path("FRONTIER_RESULT_PATH")
        evidence_path = _output_path("FRONTIER_EVIDENCE_PATH")
        if result_path == evidence_path:
            raise VerificationFailure(
                "verifier_configuration_error", "result and evidence paths must differ"
            )
        root = Path(os.environ.get("FRONTIER_LWE_ROOT", "/opt/frontier/lwe"))
        verify(
            root=root,
            payload=_required_path("FRONTIER_PAYLOAD_PATH"),
            dataset=_required_path("FRONTIER_DATASET_PATH"),
            mode=_mode(),
            result_path=result_path,
            evidence_path=evidence_path,
        )
        return 0
    except VerificationFailure as failure:
        if result_path is not None:
            _write_json(
                result_path,
                {
                    "status": "failed",
                    "summary": failure.message[:MAX_SUMMARY_CHARS],
                    "metrics": {},
                    "failure": {
                        "code": failure.code,
                        "message": failure.message[:MAX_SUMMARY_CHARS],
                        "retryable": False,
                    },
                },
            )
        print(f"{failure.code}: {failure.message}", file=sys.stderr)
        return 1
    except Exception as error:
        if result_path is not None:
            _write_json(
                result_path,
                {
                    "status": "failed",
                    "summary": f"verifier raised {type(error).__name__}",
                    "metrics": {},
                    "failure": {
                        "code": "verifier_internal_error",
                        "message": type(error).__name__,
                        "retryable": False,
                    },
                },
            )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
