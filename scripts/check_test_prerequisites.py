#!/usr/bin/env python3
"""Fail early when identity/schema tests cannot run with real inputs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
# Install before reading prerequisites or importing tests.
import deployment_test_guard  # noqa: F401  # isort: skip

import glob
import importlib
import shlex

ROOT = Path(__file__).resolve().parents[1]
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
# test_native_deploy validates both task-config and candidate-config schemas.
REQUIRED_MODULES = {
    "jsonschema": "tests/test_native_deploy.py schema validation",
    "zstandard": "tests/test_measured_baseline.py actual platform evidence decoding",
}


def lfs_files(root: Path) -> list[Path]:
    """Expand the explicit filter=lfs rules in the root .gitattributes."""
    paths = set()
    for line in (root / ".gitattributes").read_text(encoding="utf-8").splitlines():
        fields = shlex.split(line, comments=True)
        if not fields or "filter=lfs" not in fields[1:]:
            continue
        pattern = fields[0]
        # Like root .gitattributes, an unanchored basename matches at any depth.
        if "/" not in pattern:
            pattern = "**/" + pattern
        pattern = pattern.removeprefix("/")
        for name in glob.iglob(
            pattern, root_dir=root, recursive=True, include_hidden=True
        ):
            path = root / name
            if path.is_relative_to(root / "tasks") and path.is_file():
                paths.add(path)
    return sorted(paths)


def check_prerequisites(root: Path = ROOT) -> list[str]:
    errors = []
    try:
        pointers = []
        for path in lfs_files(root):
            with path.open("rb") as stream:
                if stream.read(len(LFS_POINTER_PREFIX)).startswith(LFS_POINTER_PREFIX):
                    pointers.append(str(path.relative_to(root)))
        if pointers:
            errors.append(
                "Git LFS objects are missing (pointer files):\n  "
                + "\n  ".join(pointers)
                + "\nInstall git-lfs, run `git lfs install` and `git lfs pull`, then rerun `make test`."
                + " Identity tests require the hydrated file bytes and must not be skipped."
            )
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot check root .gitattributes LFS inputs: {exc}")
    for name, reason in REQUIRED_MODULES.items():
        try:
            importlib.import_module(name)
        except ImportError:
            errors.append(
                f"Python module {name!r} is required by {reason}. "
                f"Install it for the test interpreter with `{sys.executable} -m pip install {name}`, "
                "then rerun `make test` (or set PYTHON to an interpreter with it installed)."
            )
    return errors


def main() -> int:
    errors = check_prerequisites()
    if errors:
        print("Test prerequisites: FAILED\n" + "\n".join(errors), file=sys.stderr)
        return 1
    print(
        "Test prerequisites: PASS (LFS file bytes and required Python modules available)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
