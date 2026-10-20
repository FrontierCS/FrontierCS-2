"""Exact public model-file selection and verified offline image staging."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path


def relative_path(value: object) -> str:
    if (not isinstance(value, str) or not value or value.startswith("/")
            or "\\" in value or ":" in value
            or any(not char.isprintable() for char in value)
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("model asset paths and patterns must be canonical relative paths")
    return value


def patterns(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty list of relative file patterns")
    result = [relative_path(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{field} contains duplicate patterns")
    return result


def select_paths(names: list[str], model: dict) -> list[str]:
    selected = set(names)
    for field in ("allow_patterns", "ignore_patterns"):
        if field not in model:
            continue
        matched: set[str] = set()
        for pattern in patterns(model[field], field):
            found = {name for name in names if fnmatch.fnmatchcase(name, pattern)}
            if not found:
                raise ValueError(f"{field} pattern {pattern!r} matches no snapshot file")
            matched.update(found)
        if field == "allow_patterns":
            selected &= matched
        else:
            selected -= matched
    if not selected:
        raise ValueError("model file selection is empty")
    return sorted(selected)


def file_identity(item: dict) -> tuple[str, int, str, str]:
    name = relative_path(item.get("path"))
    size = item.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"model file {name!r} has an invalid byte count")
    sha256 = item.get("sha256")
    blob_id = item.get("git_blob_id")
    if isinstance(sha256, str) and re.fullmatch(r"[0-9a-f]{64}", sha256):
        return name, size, "sha256", sha256
    if isinstance(blob_id, str) and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", blob_id):
        return name, size, "git", blob_id
    raise ValueError(f"model file {name!r} has no immutable content hash")


def stage_files(*, repo_id: str, revision: str, target: str, files: list[dict],
                previous_files: list[str]) -> None:
    """Download literal filenames, verify bytes, then admit only those files.

    Per-file downloads avoid interpreting a resolved filename as a glob again.
    Temporary hub metadata/cache disappears within this same image layer.
    """
    if not files:
        raise ValueError("model staging requires a nonempty file list")
    identities = [file_identity(item) for item in files]
    names = [item[0] for item in identities]
    previous = {relative_path(name) for name in previous_files}
    if len(set(names)) != len(names) or previous & set(names):
        raise ValueError("model staging repeats a filename")
    root = Path(target)
    if root.is_symlink():
        raise ValueError("model staging target is a symlink")
    root.mkdir(parents=True, exist_ok=True)
    present: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("model staging target contains a symlink")
        if path.is_file():
            present.add(path.relative_to(root).as_posix())
    if present != previous:
        raise ValueError("model staging target differs from the preceding verified layers")

    from huggingface_hub import hf_hub_download

    with tempfile.TemporaryDirectory(prefix=".hf-stage-", dir=root.parent) as temporary:
        staging = Path(temporary) / "files"
        for name, size, kind, digest in identities:
            downloaded = Path(hf_hub_download(
                repo_id=repo_id, revision=revision, filename=name, token=False,
                local_dir=str(staging), cache_dir=str(Path(temporary) / "cache")))
            expected = staging / name
            if (downloaded != expected or downloaded.is_symlink()
                    or not downloaded.is_file()
                    or not downloaded.resolve().is_relative_to(staging.resolve())):
                raise ValueError(f"model download returned an unsafe file for {name!r}")
            if downloaded.stat().st_size != size:
                raise ValueError(f"model file {name!r} differs from its locked byte count")
            checksum = hashlib.sha1() if kind == "git" and len(digest) == 40 else hashlib.sha256()
            if kind == "git":
                checksum.update(f"blob {size}\0".encode("ascii"))
            with downloaded.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    checksum.update(chunk)
            if checksum.hexdigest() != digest:
                raise ValueError(f"model file {name!r} differs from its locked content hash")
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), destination)
            os.chmod(destination, 0o444)
