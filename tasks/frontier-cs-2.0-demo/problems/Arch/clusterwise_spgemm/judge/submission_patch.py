"""Apply and validate a patch submission.

Copied next to `evaluator.py` in every patch task, because the evaluator is loaded
by path in four places — the judge image, the local docker shim, the task's own
`test_evaluator.py`, and the structural gate's smoke run — and a sibling is the
only location that resolves in all four.

The checks here are the union of what five hand-written evaluators each
implemented separately (vllm, rocksdb, duckdb, nanowm, generals_io_bot). They are
about the *shape* of the diff: size, binary content, where it is allowed to touch.

`allow_paths` is scope control, not a security boundary. It stops a submission
sprawling across a repository and makes review cheap; it does nothing about what a
patch does *inside* a file it is allowed to touch. Forbidden tokens, structural
markers, language restrictions and anything else task-specific stay in the
evaluator, after this returns.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MAX_PATCH_BYTES = 256 * 1024
DEFAULT_MAX_CHANGED_FILES = 50
# `git diff` emits these for content it cannot express as text. Applying one is
# how a submission smuggles a compiled artifact past a source-level review.
BINARY_MARKERS = ("GIT binary patch", "literal 0", "delta 0")
BINARY_FILES_RE = re.compile(r"^Binary files .* differ$", re.M)
HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@(?:.*)$")


class InvalidPatch(ValueError):
    """Submission producer refusal before a patch reaches validation."""


@dataclass
class PatchResult:
    ok: bool
    message: str = ""
    workdir: Path | None = None
    metrics: dict = field(default_factory=dict)
    cleanup_root: Path | None = None


def _path(text: str, prefix: str = "") -> str | None:
    """Decode Git's C-quoted UTF-8 paths without shell or Unicode escape rules."""
    if text.startswith('"'):
        if not text.endswith('"'):
            raise ValueError("unterminated quoted patch path")
        raw = bytearray()
        i = 1
        escapes = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12,
                   "r": 13, '"': 34, "\\": 92}
        while i < len(text) - 1:
            char = text[i]
            if char == '"':
                raise ValueError("unescaped quote in patch path")
            if char != "\\":
                raw.extend(char.encode("utf-8"))
                i += 1
                continue
            i += 1
            if i >= len(text) - 1:
                raise ValueError("incomplete escape in patch path")
            if text[i] in escapes:
                raw.append(escapes[text[i]])
                i += 1
            elif re.fullmatch(r"[0-3][0-7]{2}", text[i:i + 3]):
                raw.append(int(text[i:i + 3], 8))
                i += 3
            else:
                raise ValueError("unsupported escape in patch path")
        text = raw.decode("utf-8")
    if text == "/dev/null" and prefix:
        return None
    if prefix:
        if not text.startswith(prefix):
            raise ValueError("patch path lacks the expected a/ or b/ prefix")
        text = text[len(prefix):]
    if (not text or text.startswith("/") or "\\" in text
            or any(part in ("", ".", "..", ".git") for part in text.split("/"))
            or any(ord(char) < 32 or ord(char) == 127 for char in text)):
        raise ValueError("unsafe patch path")
    return text


def _patch_paths(patch_text: str) -> list[str]:
    """Check every file header, skipping counted hunk bodies (which may mimic headers)."""
    changed: list[str] = []
    seen: set[str] = set()
    section: dict[str, str] = {}
    remaining = [0, 0]
    had_hunk = False

    def finish() -> None:
        if not section:
            return
        modes = [key for key in section if key.endswith("mode")]
        if any(section[key] not in {"100644", "100755"} for key in modes):
            raise ValueError("only regular source-file modes are supported")
        if ("old mode" in section) != ("new mode" in section):
            raise ValueError("incomplete mode change")
        if (("new file mode" in section and "deleted file mode" in section)
                or "old mode" in section and any(key in section for key in
                   ("new file mode", "deleted file mode"))):
            raise ValueError("conflicting file operation headers")
        if "index" in section:
            match = re.fullmatch(r"[0-9a-f]+\.\.[0-9a-f]+(?: (100644|100755))?", section["index"])
            if match is None:
                raise ValueError("invalid index header")
            index_mode = match.group(1)
            if index_mode and any(section[key] != index_mode for key in modes):
                raise ValueError("index and file mode headers disagree")
        for key in ("similarity index", "dissimilarity index"):
            if key in section and not re.fullmatch(r"(?:100|[0-9]{1,2})%", section[key]):
                raise ValueError("invalid similarity header")
        extended = [kind for kind in ("rename", "copy")
                    if f"{kind} from" in section or f"{kind} to" in section]
        if len(extended) > 1:
            raise ValueError("mixed rename and copy headers")
        kind = extended[0] if extended else None
        old = new = None
        if kind:
            if f"{kind} from" not in section or f"{kind} to" not in section:
                raise ValueError("incomplete rename or copy headers")
            old = _path(section[f"{kind} from"])
            new = _path(section[f"{kind} to"])
        has_pair = "---" in section or "+++" in section
        if has_pair:
            if "---" not in section or "+++" not in section:
                raise ValueError("incomplete old/new file headers")
            pair = (_path(section["---"], "a/"), _path(section["+++"], "b/"))
            if kind and pair != (old, new):
                raise ValueError("rename/copy and file headers disagree")
            old, new = pair
        if "diff" in section:
            # Resolve unquoted spaces from the other headers or equal-length
            # old/new names. Trying every space would be quadratic on a long path.
            text = section["diff"]
            if text.startswith('"'):
                quoted = re.match(r'"(?:[^"\\]|\\.)*" ', text)
                if quoted is None:
                    raise ValueError("invalid quoted diff --git path")
                split = quoted.end() - 1
            elif old is not None or new is not None:
                split = len("a/" + (old if old is not None else new))
            else:
                split = (len(text) - 1) // 2
            if text[split:split + 1] != " ":
                raise ValueError("invalid diff --git separator")
            diff_old, diff_new = _path(text[:split], "a/"), _path(text[split + 1:], "b/")
            if (diff_old is None or diff_new is None
                    or not kind and diff_old != diff_new
                    or old is not None and old != diff_old
                    or new is not None and new != diff_new):
                raise ValueError("inconsistent or ambiguous diff --git paths")
            if not has_pair:
                old, new = diff_old, diff_new
            if "new file mode" in section:
                if kind or has_pair and old is not None:
                    raise ValueError("inconsistent new-file headers")
                old = None
            if "deleted file mode" in section:
                if kind or has_pair and new is not None:
                    raise ValueError("inconsistent deleted-file headers")
                new = None
        elif kind or any(key.endswith("mode") for key in section):
            raise ValueError("extended headers require diff --git")
        if old is None and new is None:
            raise ValueError("patch changes no file")
        if not kind and old is not None and new is not None and old != new:
            raise ValueError("old/new paths differ without rename/copy headers")
        if not had_hunk and not (kind or "new file mode" in section
                                 or "deleted file mode" in section
                                 or "old mode" in section and "new mode" in section):
            raise ValueError("patch has no hunks or file operation")
        # A copy reads its old path but changes only the destination. A rename
        # removes its source too; both paths consume scope and the file budget.
        for name in ((new,) if kind == "copy" else (old, new)):
            if name is not None and name not in seen:
                seen.add(name)
                changed.append(name)

    for line in patch_text.split("\n"):
        if remaining != [0, 0]:
            if line == "\\ No newline at end of file":
                continue
            if line.startswith(" "):
                remaining[0] -= 1
                remaining[1] -= 1
            elif line.startswith("-"):
                remaining[0] -= 1
            elif line.startswith("+"):
                remaining[1] -= 1
            else:
                raise ValueError("invalid or truncated patch hunk")
            if min(remaining) < 0:
                raise ValueError("patch hunk exceeds declared line counts")
            continue
        if line.startswith("diff --git ") or line.startswith("--- ") and had_hunk:
            finish()
            section = {}
            had_hunk = False
        if line.startswith("diff --git "):
            section["diff"] = line[len("diff --git "):]
        elif line.startswith("@@"):
            match = HUNK_RE.fullmatch(line)
            if match is None or "---" not in section or "+++" not in section:
                raise ValueError("invalid hunk header or missing file headers")
            remaining = [int(value) if value is not None else 1 for value in match.groups()]
            had_hunk = True
        elif not line or line == "\\ No newline at end of file":
            continue
        elif line == "GIT binary patch" or BINARY_FILES_RE.fullmatch(line):
            raise ValueError("binary content; submit source changes only")
        else:
            header = next((name for name in ("---", "+++", "rename from", "rename to",
                          "copy from", "copy to", "old mode", "new mode", "new file mode",
                          "deleted file mode", "index", "similarity index", "dissimilarity index")
                           if line.startswith(name + " ")), None)
            if header is None:
                raise ValueError("unsupported patch header or content outside a hunk")
            if had_hunk or header in section:
                raise ValueError("duplicate or misplaced patch header")
            value = line[len(header) + 1:]
            if header in ("---", "+++"):
                value = value.split("\t", 1)[0]  # Traditional unified-diff timestamps.
            section[header] = value
    if remaining != [0, 0]:
        raise ValueError("truncated patch hunk")
    finish()
    return changed


def changed_files(patch_text: str) -> list[str]:
    """All changed paths, or no inventory for a malformed patch."""
    try:
        return _patch_paths(patch_text)
    except (ValueError, UnicodeError):
        return []


def _within(path: str, allowed: list[str]) -> bool:
    for prefix in allowed:
        prefix = prefix.strip().lstrip("./").rstrip("/")
        if not prefix:
            continue
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def validate_patch(patch_text: str, config: dict) -> tuple[bool, str, dict]:
    """Shape checks, before anything is written to disk. (ok, message, metrics)."""
    submission = (config.get("submission") or {}) if isinstance(config, dict) else {}
    max_bytes = int(submission.get("max_patch_bytes") or DEFAULT_MAX_PATCH_BYTES)
    max_files = int(submission.get("max_changed_files") or DEFAULT_MAX_CHANGED_FILES)
    allow_paths = [str(p) for p in (submission.get("allow_paths") or [])]

    raw = patch_text.encode("utf-8", "replace")
    files: list[str] = []
    metrics = {
        "patch_bytes": len(raw),
        "patch_sha256": hashlib.sha256(raw).hexdigest(),
        "changed_files": len(files),
        "valid_patch": 0,
    }

    if not patch_text.strip():
        if submission.get("allow_empty"):
            metrics["valid_patch"] = 1
            return True, "empty patch (allowed)", metrics
        return False, "submission is empty", metrics
    if len(raw) > max_bytes:
        return False, f"patch is {len(raw)} bytes, over the {max_bytes} limit", metrics
    if "\x00" in patch_text:
        return False, "patch contains binary content; submit source changes only", metrics
    try:
        files = _patch_paths(patch_text)
    except (ValueError, UnicodeError) as exc:
        return False, f"invalid patch headers: {exc}", metrics
    metrics["changed_files"] = len(files)
    if not files:
        return False, "patch changes no files", metrics
    if len(files) > max_files:
        return False, f"patch changes {len(files)} files, over the {max_files} limit", metrics
    for name in files:
        if name.startswith("/") or ".." in Path(name).parts:
            return False, f"patch writes outside the tree: {name}", metrics
    if allow_paths:
        outside = [f for f in files if not _within(f, allow_paths)]
        if outside:
            return False, (f"patch touches {', '.join(outside[:5])}, outside the editable "
                           f"surface ({', '.join(allow_paths)})"), metrics

    metrics["valid_patch"] = 1
    return True, "", metrics


def _make_owner_writable(root: Path) -> None:
    """Make only the disposable copy writable without following its symlinks.

    FrontierCS2 deliberately publishes the pristine substrate as 0555/0444.
    ``copytree`` preserves those modes, even though the new files are owned by
    the verifier UID, so both ``git apply`` and an in-tree build otherwise fail.
    The source tree remains immutable; permission is restored only on this
    per-evaluation copy.
    """
    for directory, names, files in os.walk(root, followlinks=False):
        for path in [Path(directory), *(Path(directory) / name for name in names + files)]:
            if path.is_symlink():
                continue
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode):
                os.chmod(
                    path,
                    stat.S_IMODE(metadata.st_mode) | stat.S_IWUSR,
                    follow_symlinks=False,
                )


def apply_submission_patch(patch_path: str | Path, task_dir: str | Path,
                           config: dict, *, pristine: str | Path | None = None,
                           scratch_parent: str | Path | None = None,
                           deadline: float | None = None) -> PatchResult:
    """Validate the diff and apply it to a fresh copy of the pristine tree.

    The caller measures `result.workdir` and disposes `result.cleanup_root`.
    A rejection is a
    *scored zero*, not an error: return `0.0, 0.0, result.message, result.metrics`
    from the evaluator, which is the convention every task already follows.
    """
    patch_path = Path(patch_path)
    task_dir = Path(task_dir)
    try:
        # Universal newline conversion changes CRLF hunk context and reference bytes.
        with patch_path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
            patch_text = stream.read()
    except OSError as exc:
        return PatchResult(False, f"could not read the submission: {exc}")

    ok, message, metrics = validate_patch(patch_text, config)
    if not ok:
        return PatchResult(False, message, metrics=metrics)

    submission = (config.get("submission") or {}) if isinstance(config, dict) else {}
    root = Path(pristine) if pristine else task_dir / "harbor" / "app"
    repo_root = str(submission.get("repo_root") or "/app").strip("/")
    if repo_root and repo_root != "app":
        # `repo_root: /app/duckdb` means the tree lives one level down, the way the
        # hand-written patch tasks lay theirs out.
        root = root / Path(repo_root).relative_to("app") if repo_root.startswith("app/") \
            else root / repo_root
    if not root.is_dir():
        return PatchResult(False, f"no pristine tree at {root}", metrics=metrics)

    # The evaluator owns a bounded scratch directory and deadline.  Keep the
    # disposable copy inside that directory so patch application cannot escape
    # the verifier's workspace quota.
    workdir = Path(tempfile.mkdtemp(
        prefix="submission-",
        dir=str(scratch_parent) if scratch_parent is not None else None,
    ))
    try:
        tree = workdir / "tree"
        shutil.copytree(root, tree, symlinks=True)
        _make_owner_writable(tree)
        staged = workdir / "submission.patch"
        staged.write_bytes(patch_text.encode("utf-8"))

        # A task-local TMPDIR may itself live below the FrontierSmith checkout.  In
        # that case `git -C tree apply ...` otherwise discovers the checkout's
        # parent `.git`, interprets patch paths relative to that unrelated worktree,
        # and reports success while printing only "Skipped patch".  Stop discovery
        # at the disposable tree's parent so `git apply` treats `tree` as its root,
        # exactly as it does for the normal system `/tmp` path.
        git_env = os.environ.copy()
        git_env["GIT_CEILING_DIRECTORIES"] = str(tree.parent.resolve())

        for args in (["apply", "--check", "--whitespace=nowarn", str(staged)],
                     ["apply", "--whitespace=nowarn", str(staged)]):
            timeout = 300.0
            if deadline is not None:
                timeout = max(1.0, min(timeout, deadline - time.monotonic()))
            proc = subprocess.run(["git", "-C", str(tree), *args],
                                  capture_output=True, text=True,
                                  timeout=timeout, env=git_env)
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout).strip().splitlines()
                shutil.rmtree(workdir, ignore_errors=True)
                return PatchResult(
                    False,
                    f"patch does not apply: {detail[-1][:200] if detail else 'git apply failed'}",
                    metrics=metrics)
        return PatchResult(True, "", workdir=tree, metrics=metrics, cleanup_root=workdir)
    except BaseException:
        # A failed copy can leave read-only directories before permission repair.
        try:
            _make_owner_writable(workdir)
        except OSError:
            pass
        shutil.rmtree(workdir, ignore_errors=True)
        raise
