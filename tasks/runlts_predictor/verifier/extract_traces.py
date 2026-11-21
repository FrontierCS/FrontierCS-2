"""Extracts the CBP 2025 training archives and checks the traces against the workload manifest.

    extract_traces.py ARCHIVES MANIFEST TARGET

ARCHIVES holds `<category>.tar.xz` for each category; each archive's regular files go to TARGET at
their paths in the archive. Every workload of MANIFEST must then be there with its SHA-256 and
size, and TARGET must hold no other trace.
"""

import hashlib
import json
import lzma
import pathlib
import sys
import tarfile

archives, manifest, target = (pathlib.Path(argument) for argument in sys.argv[1:4])
for archive in sorted(archives.glob("*.tar.xz")):
    with lzma.open(archive, "rb") as stream, tarfile.open(fileobj=stream, mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            path = pathlib.PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"{archive.name}: unsafe member {member.name}")
            destination = target / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, destination.open("xb") as output:
                while chunk := source.read(1 << 24):
                    output.write(chunk)

workloads = json.loads(manifest.read_text(encoding="utf-8"))["workloads"]
for workload in workloads:
    path = target / workload["path"]
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 24):
            digest.update(chunk)
    if digest.hexdigest() != workload["sha256"] or path.stat().st_size != workload["compressed_bytes"]:
        raise SystemExit(f"{workload['path']} differs from the manifest")
found = sorted(str(path.relative_to(target)) for path in target.rglob("*_trace.gz"))
if found != sorted(workload["path"] for workload in workloads):
    raise SystemExit("the archives' traces are not the manifest's")
