"""Resolve hash-bound native JSON selectors or opaque, uncompressed TAR datasets.

Dataset payloads are never converted or extracted. Both modes must use the same
declared format so publication and Driver paths cannot disagree.
"""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path


def dataset_extension(policy: dict) -> str:
    metadata = policy.get("metadata", {}).get("datasets", {})
    extensions = []
    for mode in ("train", "eval"):
        row = metadata.get(mode)
        if not isinstance(row, dict):
            raise ValueError(f"missing dataset declaration: {mode}")
        path = row.get("path", f"datasets/{mode}.json")
        choices = {f"datasets/{mode}.json": "json", f"datasets/{mode}.tar": "tar"}
        if not isinstance(path, str) or path not in choices:
            raise ValueError(f"unsupported dataset path: {mode}")
        extension = choices[path]
        media_type = {"json": "application/json", "tar": "application/x-tar"}[extension]
        if row.get("media_type", media_type) != media_type:
            raise ValueError(f"dataset media_type mismatch: {mode}")
        if extension == "tar" and (
            row.get("media_type") != media_type
            or row.get("source") != "task"
            or type(row.get("bytes")) is not int
            or row["bytes"] <= 0
        ):
            raise ValueError(f"TAR dataset needs exact source/media_type/bytes: {mode}")
        extensions.append(extension)
    if extensions[0] != extensions[1]:
        raise ValueError("train and eval dataset formats differ")
    return extensions[0]


def dataset_path(root: Path, mode: str, extension: str) -> Path:
    if mode not in {"train", "eval"} or extension not in {"json", "tar"}:
        raise ValueError("unsupported dataset mode or extension")
    path = root / "datasets" / f"{mode}.{extension}"
    if path.is_symlink() or not path.is_file() or path.parent.is_symlink():
        raise ValueError(f"missing regular dataset: {mode}")
    return path


def dataset_hashes(root: Path, policy: dict) -> tuple[str, dict[str, str]]:
    extension = dataset_extension(policy)
    hashes = {}
    for mode in ("train", "eval"):
        path = dataset_path(root, mode, extension)
        row = policy["metadata"]["datasets"][mode]
        if "bytes" in row and (
            type(row["bytes"]) is not int or row["bytes"] != path.stat().st_size
        ):
            raise ValueError(f"dataset byte count mismatch: {mode}")
        with path.open("rb") as body:
            digest = "sha256:" + hashlib.file_digest(body, "sha256").hexdigest()
        if row.get("version") != digest:
            raise ValueError(f"dataset hash mismatch: {mode}")
        hashes[f"{mode}_sha256"] = digest
    return extension, hashes


def check_tar_dataset(path: Path) -> None:
    """Check archive structure without extracting or interpreting scientific data."""
    seen = set()
    files = 0
    try:
        # r: deliberately refuses compressed bytes mislabeled application/x-tar.
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                name = member.name.rstrip("/") if member.isdir() else member.name
                if (
                    not name
                    or "\\" in name
                    or any(part in {"", ".", ".."} for part in name.split("/"))
                    or name in seen
                    or not (member.isfile() or member.isdir())
                    or member.sparse is not None
                    or member.offset_data + member.size > path.stat().st_size
                ):
                    raise ValueError("unsafe or duplicate TAR dataset member")
                seen.add(name)
                files += member.isfile()
    except (tarfile.TarError, OSError) as error:
        raise ValueError("invalid uncompressed TAR dataset") from error
    if not files:
        raise ValueError("TAR dataset contains no files")
