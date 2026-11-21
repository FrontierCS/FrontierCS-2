"""Checks the models against the integrity manifest and unpacks PaddleOCR's archives.

    check_models.py ROOT

ROOT is the models' directory. Every file of ROOT/integrity-manifest.json must be there with its
size and SHA-256; each archive of ROOT/ocr is then unpacked into ROOT/ocr, as the source's recipe
does.
"""

import hashlib
import json
import pathlib
import sys
import tarfile

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "integrity-manifest.json").read_text())
assert manifest["complete"] and len(manifest["files"]) == 34
for item in manifest["files"]:
    path = root / item["path"]
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    assert path.stat().st_size == item["bytes"] and digest.hexdigest() == item["sha256"], item["path"]
for path in (root / "ocr").glob("*.tar"):
    with tarfile.open(path) as archive:
        archive.extractall(root / "ocr", filter="data")
