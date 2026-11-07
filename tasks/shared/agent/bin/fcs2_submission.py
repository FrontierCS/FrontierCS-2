"""make_submission.sh: write /workspace/payload/solution.patch from the Agent's working repo.

The patch is `git diff` of the working repo (made by prepare_workspace.sh) against the starter
commit prepare_workspace.sh recorded as refs/frontier/baseline, so changes the Agent committed
count as well as uncommitted ones. It is checked here with the same shared edit_scope module and
the same scope rules the judge applies (submission-policy.json, generated from task.toml), so a
patch that leaves the task's editable files is refused now instead of costing a train call.

What goes into the patch: every change to a starter file (edits and deletions anywhere, so an
out-of-scope edit is reported, not dropped), and new files only below the allowed paths (build
outputs and caches elsewhere in the repo stay out). It runs on the shared base's own Python,
whatever Python the task installs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edit_scope  # noqa: E402  # shipped beside this file by the shared Agent base

STARTER = Path("/frontier/task/repo")
WORKSPACE = Path("/workspace")
BASELINE_REF = "refs/frontier/baseline"


class Refused(Exception):
    pass


def make_submission(source: Path, policy: dict, *, starter: Path = STARTER,
                    workspace: Path = WORKSPACE) -> Path:
    source, workspace = source.resolve(), workspace.resolve()
    if workspace not in source.parents or not (source / ".git").is_dir():
        raise Refused("run /frontier/task/bin/prepare_workspace.sh first")
    rules = edit_scope.Rules.from_mapping(policy["edit_scope"])
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_LITERAL_PATHSPECS="1")
    git = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.quotePath=false",
           "-c", "core.autocrlf=false", "-C", str(source)]

    def run(*args: str, output=subprocess.PIPE) -> bytes:
        return subprocess.run([*git, *args], env=env, stdout=output, stderr=subprocess.PIPE,
                              check=True, timeout=60).stdout

    try:
        base = run("rev-parse", "--verify", "--quiet", BASELINE_REF + "^{commit}").decode().strip()
    except subprocess.CalledProcessError:
        raise Refused(f"{BASELINE_REF} is missing: run /frontier/task/bin/prepare_workspace.sh"
                      " to make a fresh workspace") from None
    with tempfile.TemporaryDirectory(prefix="submission-", dir=workspace) as temporary:
        temporary = Path(temporary)
        env["GIT_INDEX_FILE"] = str(temporary / "index")
        run("read-tree", base)
        run("add", "--update")  # every edit and deletion of a starter file
        new = [name for name in run("ls-files", "--others", "--exclude-standard", "-z")
               .decode("utf-8", "surrogateescape").split("\0")
               if name and edit_scope.matches(name, rules.allowed_paths)]
        if new:
            listing = temporary / "new"
            listing.write_bytes("\0".join(new).encode("utf-8", "surrogateescape") + b"\0")
            run("add", "--pathspec-from-file=" + str(listing), "--pathspec-file-nul")
        patch = temporary / "solution.patch"
        with patch.open("wb") as stream:
            # A rename is a delete and an add unless the task allows renames (allow_rename).
            run("diff", "--cached", "--binary",
                "--find-renames" if rules.allow_rename else "--no-renames", "--no-ext-diff",
                "--no-textconv", "--no-color", "--src-prefix=a/", "--dst-prefix=b/", base,
                output=stream)
        body = patch.read_bytes()
        result = edit_scope.check(body, starter, rules, build_tree=False, content_rules=False)
        if not result.ok:
            raise Refused(result.message)
        payload = workspace / "payload"
        if payload.is_symlink():
            raise Refused("/workspace/payload must be a directory, not a link")
        payload.mkdir(mode=0o700, exist_ok=True)
        destination = payload / policy["member"]
        patch.chmod(0o600)
        os.replace(patch, destination)
    return destination


def main() -> int:
    policy = json.loads((Path(__file__).resolve().parent / "submission-policy.json").read_text())
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/repo")
    try:
        print(make_submission(source, policy))
        return 0
    except Refused as error:
        print(f"submission refused: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"submission failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
