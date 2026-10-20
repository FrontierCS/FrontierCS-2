"""Produce text-source patches and validate all Git paths before application."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile


from submission_patch import InvalidPatch, _path, validate_patch


def _allowed(name: str, paths: list[str]) -> bool:
    return not paths or any(name == path.rstrip("/") or
                            (path.endswith("/") and name.startswith(path)) for path in paths)


def _residue(name: str) -> bool:
    return any(part in {"__pycache__", ".pytest_cache", ".git"} or part.endswith(".pyc")
               for part in name.split("/"))


def make_submission(source: Path, workspace: Path, policy: dict) -> Path:
    source, workspace = source.resolve(), workspace.resolve()
    if workspace not in source.parents or not (source / ".git").is_dir():
        raise InvalidPatch("prepare a Git repository below the workspace first")
    allowed = policy["allowed_paths"]
    for path in allowed:
        _path(path.rstrip("/"))
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null",
               GIT_LITERAL_PATHSPECS="1", GIT_NO_REPLACE_OBJECTS="1")
    git = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.quotePath=true", "-C", str(source)]

    def run(*args, output=subprocess.PIPE):
        return subprocess.run([*git, *args], env=env, stdout=output, stderr=subprocess.PIPE,
                              check=True, timeout=30).stdout

    names = set(run("ls-tree", "-r", "--name-only", "-z", "HEAD").split(b"\0"))
    names.update(run("ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0"))
    selected = []
    for raw in sorted(names - {b""}):
        name = _path(raw.decode("utf-8"))
        if not _allowed(name, allowed) or _residue(name):
            continue
        path = source / name
        if path.is_symlink() or source not in path.resolve().parents:
            raise InvalidPatch("source path follows a link outside the repository")
        if path.exists() and not stat.S_ISREG(path.stat().st_mode):
            raise InvalidPatch("only regular source files can be packaged")
        selected.append(raw)
    with tempfile.TemporaryDirectory(prefix="submission-", dir=workspace) as temporary:
        temporary = Path(temporary)
        env["GIT_INDEX_FILE"] = str(temporary / "index")
        run("read-tree", "HEAD")
        paths = temporary / "paths"
        paths.write_bytes(b"\0".join(selected) + (b"\0" if selected else b""))
        if selected:
            run("add", "--force", "-A", "--pathspec-from-file=" + str(paths), "--pathspec-file-nul")
        patch = temporary / "solution.patch"
        with patch.open("wb") as stream:
            run("diff", "--cached", "--no-renames", "--no-ext-diff", "--no-textconv", "--no-color",
                "--src-prefix=a/", "--dst-prefix=b/", "HEAD", "--", output=stream)
        if patch.stat().st_size > policy["max_bytes"]:
            raise InvalidPatch("patch exceeds its byte limit")
        body = patch.read_bytes()
        valid, message, _ = validate_patch(body.decode("utf-8"), {"submission": {
            "allow_paths": allowed, "max_patch_bytes": policy["max_bytes"],
            "max_changed_files": policy["max_files"],
        }})
        if not valid:
            raise InvalidPatch(message)
        member = _path(policy["member"])
        if "/" in member:
            raise InvalidPatch("patch entrypoint must be one file")
        payload = workspace / "payload"
        if payload.is_symlink():
            raise InvalidPatch("payload directory cannot be a link")
        payload.mkdir(mode=0o700, exist_ok=True)
        patch.chmod(0o600)
        destination = payload / member
        os.replace(patch, destination)
    return destination


def main() -> int:
    try:
        policy = json.loads((Path(__file__).resolve().parent / "submission-policy.json").read_text())
        print(make_submission(Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/repo"),
                              Path("/workspace"), policy))
        return 0
    except (InvalidPatch, OSError, ValueError, subprocess.SubprocessError) as error:
        if isinstance(error, InvalidPatch) and str(error) == "submission is empty":
            print("submission patch is empty", file=sys.stderr)
            return 2
        print("submission refused: " + (str(error) if isinstance(error, InvalidPatch)
                                      else type(error).__name__), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
