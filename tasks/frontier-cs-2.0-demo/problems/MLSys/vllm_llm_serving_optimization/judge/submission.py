"""Validate an extracted native payload without performing or inventing a score."""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
from pathlib import Path
from typing import Any

from patch_policy import validate_patch


class SubmissionInfrastructureError(RuntimeError):
    """Pinned clean-source precondition failed independently of candidate input."""


def inspect_submission(patch_path: Path, clean_source: Path) -> dict[str, Any]:
    """Check original patch policy and application against a separate clean tree.

    The platform archive adapter owns tar extraction. Its fixed entrypoint is
    payload/solution.patch; this function consumes that already extracted file.
    """
    try:
        mode = patch_path.lstat().st_mode
    except FileNotFoundError:
        return {"valid_submission": False, "message": "solution patch does not exist"}
    if not stat.S_ISREG(mode):
        return {"valid_submission": False, "message": "solution patch must be a regular file"}
    valid, message, metrics = validate_patch(patch_path)
    if not valid:
        return {"valid_submission": False, "message": message, "patch_metrics": metrics}
    if not (clean_source / ".git").is_dir():
        raise SubmissionInfrastructureError("the independent clean vLLM checkout is unavailable")
    try:
        status = subprocess.run(
            ["git", "-C", str(clean_source), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SubmissionInfrastructureError("the independent source check failed") from error
    if status.stdout:
        raise SubmissionInfrastructureError("the independent vLLM checkout is not clean")
    patch = patch_path.read_bytes()
    if patch.decode("utf-8", errors="replace").strip():
        checked = subprocess.run(
            ["git", "-C", str(clean_source), "apply", "--check", str(patch_path.resolve())],
            capture_output=True, timeout=30,
        )
        if checked.returncode:
            return {"valid_submission": False, "message": "patch does not apply cleanly to the pinned vLLM source", "patch_metrics": metrics}
    return {
        "valid_submission": True,
        "message": "submission policy and clean application checks passed; no evaluation performed",
        "submission_sha256": "sha256:" + hashlib.sha256(patch).hexdigest(),
        "patch_metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patch", type=Path)
    parser.add_argument("--clean-source", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_submission(args.patch, args.clean_source)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["valid_submission"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
