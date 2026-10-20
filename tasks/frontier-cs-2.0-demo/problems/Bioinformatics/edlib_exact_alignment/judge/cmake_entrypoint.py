"""Validate an existing native build against CMake's generated target model.

This is build-output evidence only. It does not prove that an evaluator reaches
the build or invokes its result, and must not replace the static dispatch gate.
Call request_codemodel before configure, build normally, then resolve the binary.
No task source is executed by this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class CMakeEntrypointError(ValueError):
    pass


def request_codemodel(build_dir: Path) -> None:
    query = Path(build_dir) / ".cmake/api/v1/query/codemodel-v2"
    query.parent.mkdir(parents=True, exist_ok=True)
    query.touch()


def _file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise CMakeEntrypointError("expected a relative file path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise CMakeEntrypointError(f"missing or out-of-tree file: {relative}")
    return path


def _json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise CMakeEntrypointError(f"invalid CMake reply: {path.name}") from exc
    if not isinstance(data, dict):
        raise CMakeEntrypointError("CMake reply must be an object")
    return data


def resolve_cmake_entrypoint(
    source_dir: Path,
    build_dir: Path,
    entrypoint: str,
    target: str,
    *,
    configuration: str = "Release",
) -> Path:
    """Return the actual executable only when the generated model binds its source.

    Requires an already completed build. Configuring alone is insufficient when
    the target artifact does not exist. The caller owns build freshness, runtime
    admission, invocation and source provenance; this function certifies none of
    those independently.
    """
    source_dir, build_dir = Path(source_dir).resolve(), Path(build_dir).resolve()
    source = _file(source_dir, entrypoint)
    reply_dir = build_dir / ".cmake/api/v1/reply"
    indices = sorted(reply_dir.glob("index-*.json"))
    if not indices:
        raise CMakeEntrypointError("no CMake File API reply; request codemodel before configure")
    try:
        index = _json(indices[-1])
        reply = index["reply"]["codemodel-v2"]
        if reply["kind"] != "codemodel" or reply["version"]["major"] != 2:
            raise CMakeEntrypointError("unsupported CMake codemodel version")
        model = _json(_file(reply_dir, reply["jsonFile"]))
        if Path(model["paths"]["source"]).resolve() != source_dir or \
                Path(model["paths"]["build"]).resolve() != build_dir:
            raise CMakeEntrypointError("CMake reply belongs to a different source/build tree")
        configs = [c for c in model["configurations"] if c["name"] == configuration]
        if len(configs) != 1:
            raise CMakeEntrypointError("requested CMake configuration is absent or ambiguous")
        targets = [t for t in configs[0]["targets"] if t["name"] == target]
        if len(targets) != 1:
            raise CMakeEntrypointError("requested CMake target is absent or ambiguous")
        detail = _json(_file(reply_dir, targets[0]["jsonFile"]))
        if detail["name"] != target or detail["id"] != targets[0]["id"] or \
                detail["type"] != "EXECUTABLE":
            raise CMakeEntrypointError("target is not the requested native executable")
        # compileGroupIndex distinguishes compiled translation units from headers
        # and other IDE-only sources; generated sources cannot establish provenance.
        compiled = [s for s in detail["sources"] if "compileGroupIndex" in s
                    and not s.get("isGenerated", False)]
        if not any(_file(source_dir, s["path"]) == source for s in compiled):
            raise CMakeEntrypointError("entrypoint is not compiled into this executable target")
        artifacts = detail.get("artifacts", [])
        if len(artifacts) != 1:
            raise CMakeEntrypointError("expected exactly one native executable artifact")
        executable = _file(build_dir, artifacts[0]["path"])
        if not os.access(executable, os.X_OK):
            raise CMakeEntrypointError("generated executable lacks execute permission")
        return executable
    except (KeyError, TypeError, IndexError) as exc:
        raise CMakeEntrypointError("incomplete CMake codemodel reply") from exc
