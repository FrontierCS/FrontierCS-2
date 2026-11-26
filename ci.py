#!/usr/bin/env python3
"""The checks every change must pass: `./ci.py`, from anywhere.

It runs `./pumpkin`, which installs the release pinned in `.pumpkinversion` on first use, then
`pumpkin fmt --check` on every `.star` file. It checks every task with Codex and the reference
agent, then both deployment smoke workflows with their defaults. Each check evaluates its
declarations, checks image recipes and lints the loaded files. It stops at the first failure.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUMPKIN = ROOT / "pumpkin"

# Codex's default tool mode needs a model whose entry `lib/codex-catalog` carries.
CODEX_ARGS = ["agent=codex", "model=gpt-6.1-terra"]


class Failure(Exception):
    """A check that failed, with what to tell the person running it."""


def run(command, **options):
    """Runs `command`, which prints to this script's output, and fails unless it exits 0."""
    result = subprocess.run(command, check=False, **options)
    if result.returncode != 0:
        raise Failure(
            f"`{' '.join(map(str, command))}` exited with {result.returncode}"
        )


def tasks():
    """Every task's entry file, relative to the root: each directory of `tasks/` is a task."""
    directories = sorted(path for path in (ROOT / "tasks").iterdir() if path.is_dir())
    missing = [path.name for path in directories if not (path / "task.star").is_file()]
    if missing or not directories:
        raise Failure(
            f"tasks/ has no task.star in {missing or 'any directory'}: give each task one"
        )
    return [f"tasks/{path.name}/task.star" for path in directories]


def check_format():
    """`pumpkin fmt --check` on every `.star` file below the root; `pumpkin fmt` formats them."""
    print("==> pumpkin fmt --check", flush=True)
    run([PUMPKIN, "fmt", "--check"], cwd=ROOT)


def check_entries(entries, kind, args=()):
    """Check the entries together so Pumpkin reports every problem in the batch."""
    options = [f"--arg={arg}" for arg in args]
    details = ", " + " ".join(options) if options else ""
    print(f"==> pumpkin check, {len(entries)} {kind}{details}", flush=True)
    result = subprocess.run(
        [PUMPKIN, "check", *entries, *options], check=False, cwd=ROOT
    )
    if result.returncode != 0:
        scope = args[0] if args else kind
        raise Failure("pumpkin check found problems, as above, with " + scope)


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    try:
        run([PUMPKIN, "--version"])
        check_format()
        entries = tasks()
        check_entries(entries, "tasks", CODEX_ARGS)
        check_entries(entries, "tasks", ["agent=reference"])
        check_entries(
            ["smoke/lease/task.star", "smoke/routes/task.star"], "smoke workflows"
        )
    except Failure as failure:
        print(f"ci.py: {failure}", file=sys.stderr)
        return 1
    print("ci.py: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
