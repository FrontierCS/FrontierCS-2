"""Verifier image build hook. Refuses incomplete acquisition locks before any I/O."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse
from urllib.request import urlopen
import venv

TASK = Path(__file__).resolve().parent
PREFIX = Path("/opt/diffusion-policy/python")
ASSET = Path("/opt/diffusion-policy/assets/pusht-transformer.ckpt")
MAX_ACQUISITION_BYTES = 2 * 1024**3
ALLOWED_HOSTS = {"files.pythonhosted.org", "download.pytorch.org", "diffusion-policy.cs.columbia.edu"}
BUILD_DIRECTORY = Path("/tmp/diffusion-runtime-build")


@contextmanager
def acquisition_directory():
    """Keep pip's recorded wheel URLs stable across independent image builds."""
    BUILD_DIRECTORY.mkdir(mode=0o700, exist_ok=False)
    try:
        yield BUILD_DIRECTORY
    finally:
        shutil.rmtree(BUILD_DIRECTORY)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_lock(lock):
    require(lock.get("status") == "frozen", "Runtime acquisition lock is not frozen")
    require(re.fullmatch(r"python@sha256:[0-9a-f]{64}", lock.get("base_image", "")) is not None,
            "A verified Python base leaf digest is required")
    require(lock.get("platform") == "linux/amd64" and lock.get("python_version") == "3.10.20",
            "The supported measurement runtime is CPython 3.10.20 on linux/amd64")
    require(lock.get("base_manifest_media_type") in {
        "application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"},
        "The Python base must be a concrete image manifest")
    expected = {}
    for line in (TASK / "measurement-requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            expected[name] = version
    wheels = lock.get("wheels")
    require(isinstance(wheels, list) and len(wheels) == len(expected), "Incomplete wheel closure")
    require({item.get("name"): item.get("version") for item in wheels} == expected, "Wheel versions differ")
    filenames = set()
    for item in wheels + [lock.get("checkpoint", {})]:
        name = item.get("filename")
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", name),
                "Acquisition artifact must have a flat filename")
        require(name not in filenames, "Duplicate acquisition filename")
        filenames.add(name)
        require(isinstance(item.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
                "Every acquisition artifact requires a proven SHA256")
        require(type(item.get("bytes")) is int and 0 < item["bytes"] <= 512 * 1024**2,
                "Every artifact requires a bounded exact size")
        if item.get("context_file") is not None:
            path = item["context_file"]
            require(isinstance(path, str) and path == "runtime-wheels/" + name and item.get("url") is None,
                    "Locally qualified wheel must be a flat task-owned build input")
        else:
            parsed = urlparse(item.get("url") or "")
            require(parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS
                    and parsed.username is None and parsed.password is None and not parsed.fragment,
                    "Artifact URL is outside the reviewed public acquisition hosts")
    require(all(item["filename"].endswith(".whl") for item in wheels), "Only qualified wheels may be installed")
    require(sum(item["bytes"] for item in wheels + [lock["checkpoint"]]) <= MAX_ACQUISITION_BYTES,
            "Acquisition closure exceeds its bound")
    require(lock["checkpoint"]["sha256"] == "fa302d345467ef5927d60f3fd7cdd222e1ee8fda52216ca64323a08e2ce700a8"
            and lock["checkpoint"]["bytes"] == 143687633, "Checkpoint provenance changed")
    return wheels


def download(item, directory):
    target = directory / item["filename"]
    digest = hashlib.sha256()
    total, deadline = 0, time.monotonic() + 600
    local = item.get("context_file")
    if local:
        source_path = TASK / local
        require(source_path.is_file() and not source_path.is_symlink()
                and source_path.resolve().is_relative_to(TASK), "Qualified local wheel is missing or linked")
        source = source_path.open("rb")
    else:
        source = urlopen(item["url"], timeout=30)
    with source, target.open("xb") as out:
        if not local:
            require(urlparse(source.url).hostname in ALLOWED_HOSTS, "Unexpected artifact redirect")
        while block := source.read(1024 * 1024):
            total += len(block)
            require(total <= item["bytes"] and time.monotonic() < deadline, "Artifact size or time bound exceeded")
            digest.update(block)
            out.write(block)
    require(total == item["bytes"] and digest.hexdigest() == item["sha256"], "Acquired artifact does not match its pin")
    return target


def prepare(lock):
    wheels = validate_lock(lock)
    import yaml
    config = yaml.safe_load((TASK / "config.yaml").read_text())
    require(config["runtime"]["docker"]["image"] == lock["base_image"], "The Docker base and artifact lock disagree")
    require(sys.version_info[:3] == (3, 10, 20), "The build interpreter differs from the frozen Python version")
    require(not PREFIX.exists() and not ASSET.exists(), "Runtime targets must be fresh")
    env = dict(os.environ, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    with acquisition_directory() as directory:
        artifacts = {item["name"]: download(item, directory) for item in wheels}
        checkpoint = download(lock["checkpoint"], directory)
        venv.EnvBuilder(with_pip=True).create(PREFIX)
        python = str(PREFIX / "bin/python")
        base = [python, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile"]
        bootstrap = [artifacts[name] for name in ("pip", "setuptools", "wheel")]
        subprocess.run(base + [str(path) for path in bootstrap], check=True, env=env, timeout=300)
        subprocess.run(base + [str(path) for path in artifacts.values()], check=True, env=env, timeout=600)
        subprocess.run([python, "-m", "pip", "check"], check=True, env=env, timeout=30)
        identity = subprocess.run([python, "-I", str(TASK / "runtime_probe.py")],
                                  capture_output=True, text=True, check=True, env=env, timeout=30)
        observed = json.loads(identity.stdout)
        require(dict(observed["distributions"]) == {item["name"]: item["version"] for item in wheels},
                "Installed runtime differs from its exact wheel closure")
        ASSET.parent.mkdir(parents=True, exist_ok=True)
        ASSET.write_bytes(checkpoint.read_bytes())
        ASSET.chmod(0o444)
        provenance = PREFIX.parent / "runtime-build-provenance.json"
        provenance.write_text(json.dumps({"lock": lock, "identity": observed, "inference": False}, indent=2) + "\n")
        provenance.chmod(0o444)


if __name__ == "__main__":
    prepare(json.loads((TASK / "runtime-inputs.json").read_text()))
