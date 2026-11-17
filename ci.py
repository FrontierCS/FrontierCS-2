#!/usr/bin/env python3
"""The checks every change must pass: `./ci.py`, from anywhere.

It runs `./pumpkin`, which builds the Pumpkin that `.pumpkinversion` names on first use, then
`pumpkin fmt --check` on every `.star` file, then `pumpkin check` on every task as Codex's agent,
which evaluates the task, checks the recipes of the images it declares and lints the `.star` files
it loads. It stops at the first check that fails.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# The args every task is checked with. A harness agent declares every image a task has, which the
# reference agent does not, and every task loads every `.star` file of `lib/` whatever the agent;
# Codex's default `tools`, `functions`, needs a model whose catalog entry Pumpkin carries.
CHECK_ARGS = ["agent=codex", "model=gpt-6.1-terra"]


class Failure(Exception):
    """A check that failed, with what to tell the person running it."""


def run(command, **options):
    """Runs `command`, which prints to this script's output, and fails unless it exits 0."""
    result = subprocess.run(command, check=False, **options)
    if result.returncode != 0:
        raise Failure(
            f"`{' '.join(map(str, command))}` exited with {result.returncode}"
        )


def bootstrap(repository):
    """The command that runs `./pumpkin`, which builds Pumpkin on first use: run before the checks,
    so that the build is not taken for the first check's."""
    pumpkin = [str(ROOT / "pumpkin")]
    if repository:
        pumpkin.append(f"--pumpkin-repository={repository}")
    run([*pumpkin, "--version"])
    return pumpkin


def tasks():
    """Every task's entry file, relative to the root: each directory of `tasks/` is a task."""
    directories = sorted(path for path in (ROOT / "tasks").iterdir() if path.is_dir())
    missing = [path.name for path in directories if not (path / "task.star").is_file()]
    if missing or not directories:
        raise Failure(
            f"tasks/ has no task.star in {missing or 'any directory'}: give each task one"
        )
    return [f"tasks/{path.name}/task.star" for path in directories]


def check_format(pumpkin):
    """`pumpkin fmt --check` on every `.star` file below the root; `pumpkin fmt` formats them."""
    print("==> pumpkin fmt --check", flush=True)
    run([*pumpkin, "fmt", "--check"], cwd=ROOT)


def check_tasks(pumpkin):
    """`pumpkin check` on every task in one call, which says every problem, then the command that
    checks again the tasks that have one."""
    entries = tasks()
    options = [f"--arg={arg}" for arg in CHECK_ARGS]
    print(f"==> pumpkin check, {len(entries)} tasks, {' '.join(options)}", flush=True)
    result = subprocess.run(
        [*pumpkin, "check", *entries, *options], check=False, cwd=ROOT
    )
    if result.returncode != 0:
        raise Failure("pumpkin check found problems, as above, with " + CHECK_ARGS[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--pumpkin-repository",
        metavar="URL",
        help="Clone Pumpkin from URL, to try a ref that isn't pushed yet.",
    )
    args = parser.parse_args()
    try:
        pumpkin = bootstrap(args.pumpkin_repository)
        check_format(pumpkin)
        check_tasks(pumpkin)
    except Failure as failure:
        print(f"ci.py: {failure}", file=sys.stderr)
        return 1
    print("ci.py: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
