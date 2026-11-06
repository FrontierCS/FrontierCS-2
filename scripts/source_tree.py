"""A task's (or a shared image's) source tree is its git-tracked files, never what else lies on
disk: an ignored judge/__pycache__ or a stray scratch file must not reach the source identity
(the image tag) or a docker build context (#757, decision 54).

Inside a git work tree the source is `git ls-files`, and any untracked or ignored file under the
directory fails the build, so the identity a worktree computes is the one a clean checkout
computes. A tree outside any git work tree (an exported or restored copy) has no ignore rules;
every file in it is source, and so is every file of a copy staged below an ignored directory
(.generated/)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class UntrackedFiles(ValueError):
    """The directory holds files git does not track (untracked or ignored)."""


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *arguments], capture_output=True)


def in_work_tree(root: Path) -> bool:
    """Whether `root` is a checked-out source tree: inside a git work tree and not itself
    ignored (a copy staged below an ignored directory such as .generated/ is an export)."""
    result = _git(root, "rev-parse", "--is-inside-work-tree")
    if result.returncode != 0 or result.stdout.strip() != b"true":
        return False
    return _git(root, "check-ignore", "-q", "--", ".").returncode != 0


def check_clean(root: Path, paths: list[str] | None = None) -> None:
    """Fail when `root` (in a git work tree), or only `paths` below it, holds untracked or
    ignored files."""
    status = _git(root, "status", "--porcelain=v1", "-z", "--ignored=matching",
                  "--untracked-files=all", "--", *(paths or ["."]))
    if status.returncode != 0:
        raise ValueError(f"git status failed in {root}: {status.stderr.decode().strip()}")
    stray = [entry[3:].decode() for entry in status.stdout.split(b"\0")
             if entry[:3] in (b"?? ", b"!! ")]
    if stray:
        listed = ", ".join(stray[:10]) + (f" (+{len(stray) - 10} more)" if len(stray) > 10 else "")
        raise UntrackedFiles(
            f"{root} holds files git does not track, which would change the source identity or "
            f"reach the image: {listed}. Remove them (git clean -dfx -- <dir>) or commit them.")


def tracked_files(root: Path, paths: list[str] | None = None) -> list[str] | None:
    """The source files under `root` (or only at or below `paths`), relative to it: git's
    tracked files after check_clean, or None when `root` is not in a git work tree (every file
    on disk is source)."""
    if not in_work_tree(root):
        return None
    check_clean(root, paths)
    listed = _git(root, "ls-files", "-z", "--cached", "--", *(paths or ["."]))
    if listed.returncode != 0:
        raise ValueError(f"git ls-files failed in {root}: {listed.stderr.decode().strip()}")
    return sorted(os.fsdecode(path) for path in listed.stdout.split(b"\0") if path)


def stage(root: Path, target: Path, *, link: bool = False,
          paths: list[str] | None = None) -> None:
    """Copy `root`'s source files (optionally only those at or below `paths`) into the new
    directory `target`, preserving modes; `link` hard-links them instead (same filesystem, read
    only use). Outside a git work tree the whole tree is copied."""
    files = tracked_files(root, paths)
    if files is None:
        if paths:
            raise ValueError(f"{root} is not in a git work tree")
        shutil.copytree(root, target, symlinks=True)
        return
    target.mkdir(parents=True)
    for relative in files:
        source, destination = root / relative, target / relative
        if not os.path.lexists(source):
            raise ValueError(f"tracked file missing from the work tree: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            destination.symlink_to(os.readlink(source))
        elif link:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        else:
            shutil.copy2(source, destination)
