"""The edit-scope check every task uses, on the Agent's side and the judge's.

A submission is a unified diff of the task's starter tree (`repo/`). Whether it stays inside
the task's editable surface is decided from the tree the patch actually produces, never from
the patch text: the patch is applied, strictly and once, to a scratch git copy of the pristine
tree, and the changed files are read back from git. The judge then measures a fresh copy of the
pristine tree with only the allowed changed files overlaid, so nothing outside the surface can
reach the measurement even if a check here were wrong.

Every rejection is a scored zero with one of CATEGORIES and a short fixed sentence, for every
task. The Agent's make_submission.sh runs this module with the task's scope rules before it
writes a patch; the content rules (forbidden_added, forbidden_content) are judge-only and never
ship to the Agent.

The shared verifier calls check(); a task never imports this module. Task-specific checks that
rules cannot express go in the judge hook `check_tree(tree, pristine, changes)`.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# The public rejection categories, the same for every task, and the sentence the Agent sees.
CATEGORIES = {
    "empty": "The patch changes nothing.",
    "too_large": "The patch is larger than the task allows.",
    "malformed": "The patch does not apply to the starter tree.",
    "forbidden_file_type": "The patch adds or changes a file of a kind the task does not allow "
                           "(binary, symlink, executable, submodule, or a mode change).",
    "out_of_scope": "The patch changes something outside the task's editable surface: a file "
                    "it may not change, add or delete, or a part of an editable file that must "
                    "stay as it is.",
    "too_many_files": "The patch changes more files than the task allows.",
    "content_policy": "The patch uses content the task forbids in the editable files.",
}

DEFAULT_MAX_CHANGED_FILES = 50
DEFAULT_MAX_PATCH_BYTES = 256 * 1024
GIT_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class Rules:
    """A task's editable surface: task.toml [submission], defaults in task-defaults.toml.

    Paths are repo-relative POSIX patterns: `dir/` is everything below dir, `*` matches within
    one path segment, `**` across segments; anything else is one exact file.
    """

    allowed_paths: tuple[str, ...]
    deny_paths: tuple[str, ...] = ()
    max_changed_files: int = DEFAULT_MAX_CHANGED_FILES
    max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES
    max_file_bytes: int | None = None
    allowed_suffixes: tuple[str, ...] = ()      # e.g. (".py", ".pyi"); empty = any
    allow_empty: bool = False                   # a zero-byte patch submits the starter
    allow_new_files: bool = True
    allow_delete: bool = False
    allow_rename: bool = False                  # a rename is one change, allowed when both
                                                # paths are in scope (not a delete + an add)
    allow_executable: bool = False
    forbidden_added: tuple[str, ...] = ()       # regexes over added lines (judge only)
    forbidden_content: tuple[str, ...] = ()     # regexes over whole changed files (judge only)
    ignore_case: bool = False

    @classmethod
    def from_mapping(cls, values: dict) -> "Rules":
        known = {name for name in cls.__dataclass_fields__}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown edit-scope rules {unknown}")
        converted = {key: tuple(value) if isinstance(value, list) else value
                     for key, value in values.items()}
        return cls(**converted)


@dataclass(frozen=True)
class Change:
    status: str                  # A, M, D, or R (a rename, only under allow_rename)
    path: str                    # the new path of a rename
    added: tuple[str, ...] = ()  # the added lines, without their "+"
    old_path: str = ""           # the old path of a rename


@dataclass
class Result:
    ok: bool
    category: str = ""
    message: str = ""
    paths: tuple[str, ...] = ()  # the offending paths of a rejection
    changes: tuple[Change, ...] = ()
    tree: Path | None = None     # pristine + allowed changed files: what the judge measures
    metrics: dict = field(default_factory=dict)
    _scratch: Path | None = None

    def dispose(self) -> None:
        if self._scratch is not None:
            _remove(self._scratch)
            self._scratch = None


def _pattern(pattern: str) -> re.Pattern:
    pattern = pattern.strip()
    if pattern.startswith("./"):
        pattern = pattern[2:]
    if pattern.endswith("/"):
        pattern += "**"
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(out + r"\Z")


def matches(path: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(_pattern(p).match(path) for p in patterns if p.strip())


def check(patch: bytes, pristine: str | Path, rules: Rules, *, build_tree: bool = True,
          content_rules: bool = True) -> Result:
    """Apply `patch` to a scratch copy of `pristine` and decide whether it stays in scope.

    On success `result.tree` (when `build_tree`) is a fresh copy of `pristine` with only the
    allowed changed files overlaid; the caller measures it and calls `result.dispose()`. The
    Agent-side pre-check passes `content_rules=False` (it never holds them anyway).
    """
    metrics = {"patch_bytes": len(patch), "patch_sha256": hashlib.sha256(patch).hexdigest(),
               "changed_files": 0}

    def reject(category: str, paths: list[str] | tuple[str, ...] = ()) -> Result:
        metrics["edit_scope"] = category
        shown = f" ({', '.join(paths[:5])}{', ...' if len(paths) > 5 else ''})" if paths else ""
        return Result(False, category, CATEGORIES[category] + shown, tuple(paths), metrics=metrics)

    pristine = Path(pristine)
    if not pristine.is_dir():
        raise FileNotFoundError(f"no pristine tree at {pristine}")
    if not patch:
        if not rules.allow_empty:
            return reject("empty")
        scratch, tree = _fresh_copy(pristine) if build_tree else (None, None)
        return Result(True, tree=tree, metrics=metrics, _scratch=scratch)
    if len(patch) > rules.max_patch_bytes:
        return reject("too_large")
    if not patch.strip():
        return reject("malformed")
    if b"\x00" in patch or b"GIT binary patch" in patch or b"\nBinary files " in patch:
        return reject("forbidden_file_type")

    scratch = Path(tempfile.mkdtemp(prefix="edit-scope-"))
    try:
        base = scratch / "applied"
        shutil.copytree(pristine, base, symlinks=True)
        _make_owner_writable(base)
        env = _git_env(scratch)
        _git(base, env, "init", "--quiet")
        _git(base, env, "add", "--all", "--force")
        _git(base, env, "commit", "--quiet", "--allow-empty", "--no-verify", "-m", "pristine")
        staged = scratch / "submission.patch"
        staged.write_bytes(patch)
        # One strict application: no fuzz, no 3-way, no reject files, no fallbacks.
        if _git(base, env, "apply", "--index", "--whitespace=nowarn", str(staged),
                check=False).returncode != 0:
            _remove(scratch)
            return reject("malformed")

        changes, odd, new_executables = _changes(base, env, renames=rules.allow_rename)
        metrics["changed_files"] = len(changes)
        if odd:
            _remove(scratch)
            return reject("forbidden_file_type", odd)
        if not changes:
            _remove(scratch)
            return reject("empty")
        def editable(path: str, suffix: bool) -> bool:
            return (matches(path, rules.allowed_paths) and not matches(path, rules.deny_paths)
                    and (not suffix or not rules.allowed_suffixes
                         or path.endswith(tuple(rules.allowed_suffixes))))

        # A rename (allow_rename only) is in scope when its old and its new path both are.
        outside = [p for c in changes for p in (
                       [p for p in (c.old_path, c.path) if not editable(p, True)]
                       if c.status == "R" else
                       [c.path] if (not editable(c.path, c.status != "D")
                                    or (c.status == "D" and not rules.allow_delete)
                                    or (c.status == "A" and not rules.allow_new_files)) else [])]
        if outside:
            _remove(scratch)
            return reject("out_of_scope", outside)
        executable = [] if rules.allow_executable else new_executables
        if executable:
            _remove(scratch)
            return reject("forbidden_file_type", executable)
        if len(changes) > rules.max_changed_files:
            _remove(scratch)
            return reject("too_many_files")
        if rules.max_file_bytes is not None:
            big = [c.path for c in changes if c.status != "D"
                   and (base / c.path).stat().st_size > rules.max_file_bytes]
            if big:
                _remove(scratch)
                return reject("too_large", big)
        if content_rules and (rules.forbidden_added or rules.forbidden_content):
            flags = re.IGNORECASE if rules.ignore_case else 0
            added = [re.compile(p, flags) for p in rules.forbidden_added]
            whole = [re.compile(p, flags) for p in rules.forbidden_content]
            hits = []
            for change in changes:
                if change.status == "D":
                    continue
                text = (base / change.path).read_text(encoding="utf-8", errors="replace")
                if (any(r.search(line) for r in added for line in change.added)
                        or any(r.search(text) for r in whole)):
                    hits.append(change.path)
            if hits:
                _remove(scratch)
                # The paths, never the pattern: forbidden patterns stay private to the judge.
                return reject("content_policy", hits)

        tree = None
        if build_tree:
            tree = scratch / "tree"
            shutil.copytree(pristine, tree, symlinks=True)
            _make_owner_writable(tree)
            for change in changes:
                target = tree / change.path
                if change.status == "R":
                    (tree / change.old_path).unlink(missing_ok=True)
                if change.status == "D":
                    target.unlink(missing_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(base / change.path, target, follow_symlinks=False)
        return Result(True, changes=tuple(changes), tree=tree, metrics=metrics, _scratch=scratch)
    except BaseException:
        _remove(scratch)
        raise


def _changes(base: Path, env: dict, *, renames: bool = False
             ) -> tuple[list[Change], list[str], list[str]]:
    """Every file the applied patch changed, with its added lines; the paths whose change is not
    a plain regular-file edit; and new executable files. A rename is a delete and an add unless
    `renames` (allow_rename): then git's rename detection on the applied tree makes it one R."""
    detection = "--find-renames" if renames else "--no-renames"
    raw = _git(base, env, "diff", "--cached", "--raw", detection, "-z", "HEAD").stdout
    fields = raw.split("\0")
    entries, odd, new_executables = [], [], []
    i = 0
    while i < len(fields) - 1:
        meta, path = fields[i], fields[i + 1]
        i += 2
        if not meta.startswith(":"):
            continue
        old_mode, new_mode, _, _, status = meta[1:].split()
        status = status[0]
        old_path = ""
        if status in "RC":
            old_path, path = path, fields[i]
            i += 1
        if status == "C":  # never produced without --find-copies; refuse rather than guess
            odd.append(path)
            continue
        entries.append((status, path, old_path))
        modes = {m for m in (old_mode, new_mode) if m != "000000"}
        if not modes <= {"100644", "100755"} or (status in "MR" and old_mode != new_mode):
            odd.append(path)
        elif status == "A" and new_mode == "100755":
            new_executables.append(path)
    # `git apply --index` stages what it writes; anything else it left behind is suspect.
    odd.extend(p for p in _git(base, env, "ls-files", "--others", "-z").stdout.split("\0") if p)
    added: dict[str, list[str]] = {}
    current = None
    diff = _git(base, env, "diff", "--cached", detection, "-U0", "--no-color", "HEAD").stdout
    for line in diff.splitlines():
        if line.startswith("+++ "):
            name = line[4:]
            current = name[2:] if name.startswith("b/") else None
            if current is not None:
                added.setdefault(current, [])
        elif line.startswith("+") and current is not None:
            added[current].append(line[1:])
    changes = [Change(status, path, tuple(added.get(path, ())), old_path)
               for status, path, old_path in entries]
    return changes, odd, new_executables


def _fresh_copy(pristine: Path) -> tuple[Path, Path]:
    scratch = Path(tempfile.mkdtemp(prefix="edit-scope-"))
    tree = scratch / "tree"
    shutil.copytree(pristine, tree, symlinks=True)
    _make_owner_writable(tree)
    return scratch, tree


def _git_env(scratch: Path) -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(
        # Never discover an enclosing repository (a TMPDIR below a checkout), never read the
        # user's or system's git config, and commit with a fixed identity.
        GIT_CEILING_DIRECTORIES=str(scratch.resolve()), GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull, HOME=str(scratch),
        GIT_AUTHOR_NAME="frontier", GIT_AUTHOR_EMAIL="frontier@invalid",
        GIT_COMMITTER_NAME="frontier", GIT_COMMITTER_EMAIL="frontier@invalid",
        GIT_AUTHOR_DATE="1970-01-01T00:00:00Z", GIT_COMMITTER_DATE="1970-01-01T00:00:00Z",
    )
    return env


def _git(cwd: Path, env: dict, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    command = ["git", "-c", "core.fileMode=true", "-c", "core.symlinks=true",
               "-c", "core.autocrlf=false", "-c", "core.quotePath=false", *args]
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True,
                            errors="surrogateescape", timeout=GIT_TIMEOUT_SECONDS)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {(result.stderr or result.stdout).strip()[:300]}")
    return result


def _make_owner_writable(root: Path) -> None:
    """The pristine tree is published 0555/0444; only this disposable copy becomes writable."""
    for directory, names, files in os.walk(root, followlinks=False):
        for path in [Path(directory), *(Path(directory) / name for name in names + files)]:
            if path.is_symlink():
                continue
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode) or stat.S_ISREG(mode):
                os.chmod(path, stat.S_IMODE(mode) | stat.S_IWUSR, follow_symlinks=False)


def _remove(root: Path) -> None:
    try:
        _make_owner_writable(root)
    except OSError:
        pass
    shutil.rmtree(root, ignore_errors=True)
