"""Capture installed dependency identities and retained package notices at build time."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess


def main():
    out = Path("/opt/frontier/build-evidence")
    out.mkdir(parents=True, exist_ok=True)
    apt = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\n"], text=True)
    (out / "apt-packages.tsv").write_text(apt)
    packages = []
    for dist in importlib.metadata.distributions():
        notices, missing = [], []
        for entry in dist.files or []:
            if any(key in str(entry).lower() for key in ("license", "licence", "notice", "copyright")):
                path = Path(dist.locate_file(entry))
                if path.is_file():
                    notices.append({"path": str(path), "bytes": path.stat().st_size,
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                else:
                    missing.append(str(path))
        packages.append({"name": dist.metadata["Name"], "version": dist.version,
                         "license": dist.metadata.get("License"),
                         "license_expression": dist.metadata.get("License-Expression"),
                         "license_files": dist.metadata.get_all("License-File", []),
                         "notices": notices, "missing_notice_files": missing})
    (out / "python-packages.json").write_text(json.dumps({
        "python": platform.python_version(), "packages": sorted(packages, key=lambda d: d["name"].lower())},
        indent=2) + "\n")
    notices = {}
    for root in (Path("/usr/share/doc"), Path("/opt/nvidia"), Path("/usr/local/cuda")):
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and any(key in path.name.lower() for key in ("copyright", "license", "notice")):
                    notices[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "system-notices.json").write_text(json.dumps(notices, indent=2) + "\n")


if __name__ == "__main__":
    main()
