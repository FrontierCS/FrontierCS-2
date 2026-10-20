"""Local-only fallback for running this task outside the Frontier judge."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PatchResult:
    ok: bool
    message: str
    workdir: Path | None = None
    metrics: dict = field(default_factory=dict)


def apply_submission_patch(solution_path: str, task_dir: Path, config: dict) -> PatchResult:
    try:
        patch_path = Path(solution_path)
        data = patch_path.read_bytes()
        submission = config.get("submission", {})
        if len(data) > int(submission.get("max_patch_bytes", 131072)):
            return PatchResult(False, "patch exceeds the size limit")
        if b"\x00" in data or b"GIT binary patch" in data:
            return PatchResult(False, "binary patches are not accepted")
        text = data.decode("utf-8")
        paths = re.findall(r"^diff --git a/(\S+) b/(\S+)$", text, re.MULTILINE)
        if not paths:
            return PatchResult(False, "submission is not a non-empty git patch")
        changed = {new for old, new in paths if old == new}
        if len(changed) != len(paths):
            return PatchResult(False, "renames and path changes are not accepted")
        allowed = set(submission.get("allow_paths", []))
        if not changed.issubset(allowed):
            return PatchResult(False, "patch changes a file outside the editable surface")
        work_root = Path(task_dir) / ".local-work"
        work_root.mkdir(exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="patched-", dir=work_root))
        shutil.copytree(Path(task_dir) / "harbor" / "app", workdir, dirs_exist_ok=True)
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch_path.resolve())],
            cwd=workdir,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        if proc.returncode != 0:
            shutil.rmtree(workdir, ignore_errors=True)
            return PatchResult(False, "patch does not apply to the pristine substrate")
        return PatchResult(True, "patch applied", workdir, {"changed_files": sorted(changed)})
    except Exception:
        return PatchResult(False, "patch validation failed")
