"""Select the private dataset sealed for this hardware class from the composite dataset.

The train/eval dataset object is a JSON envelope holding a tar of up to three hardware-keyed
private datasets plus a manifest. The one whose sealed hardware identity admits the actual
machine (hardware_admission.admits) is written to scratch and used; the selection logic is
unchanged from the release that ran it as an inline verifier command (it moved into the judge
with the task.toml migration).
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import tarfile

MAX_BYTES = 4 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def strict_json(body):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonfinite JSON")
    return json.loads(body, object_pairs_hook=pairs, parse_constant=constant)


def digest(body):
    return hashlib.sha256(body).hexdigest()


def hardware_key(identity):
    return digest(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode())


def actual_identity():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    return {"architecture": platform.machine(), "kernel": platform.release(),
            "cpu_models": sorted({line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")}),
            "affinity": sorted(os.sched_getaffinity(0))}


def select_dataset(path, mode, identity, scratch, read_dataset):
    """Admit by the sealed hardware class; inspect both mode-specific archives before selection."""
    require(mode in {"train", "eval"}, "Unknown native verification mode")
    path = Path(path)
    require(path.is_file() and not path.is_symlink()
            and 0 < path.stat().st_size <= MAX_BYTES, "Invalid composite dataset")
    selector = strict_json(path.read_bytes())
    require(isinstance(selector, dict)
            and set(selector) == {"schema", "problem_id", "mode", "archive_sha256", "archive_base64"}
            and selector["schema"] == "frontier.diffusion-json-archive/v1"
            and selector["problem_id"] == "diffusion_policy_cpu"
            and selector["mode"] == mode
            and isinstance(selector["archive_base64"], str)
            and isinstance(selector["archive_sha256"], str),
            "Invalid JSON archive selector")
    archive_body = base64.b64decode(selector["archive_base64"], validate=True)
    require(0 < len(archive_body) <= MAX_BYTES
            and "sha256:" + digest(archive_body) == selector["archive_sha256"],
            "Composite archive binding changed")
    bodies = {}
    with tarfile.open(fileobj=io.BytesIO(archive_body), mode="r:") as archive:
        total = 0
        for member in archive:
            require(member.isfile() and not member.pax_headers and member.sparse is None
                    and member.name not in bodies and len(bodies) < 4,
                    "Unexpected composite member")
            require(member.name == "manifest.json" or (
                len(member.name) == 68 and member.name.endswith(".tar")
                and all(c in "0123456789abcdef" for c in member.name[:-4])),
                "Invalid composite member name")
            total += member.size
            require(0 < member.size <= 2 * 1024**2 and total <= MAX_BYTES,
                    "Composite member exceeds bound")
            body = archive.extractfile(member).read(2 * 1024**2 + 1)
            require(len(body) == member.size, "Truncated composite member")
            bodies[member.name] = body
    require("manifest.json" in bodies, "Missing composite manifest")
    manifest = strict_json(bodies["manifest.json"])
    require(set(manifest) == {"schema", "mode", "source_sha256", "datasets"}
            and manifest["schema"] == "frontier.diffusion-hardware-datasets/v1"
            and manifest["mode"] == mode and isinstance(manifest["datasets"], dict)
            and 1 <= len(manifest["datasets"]) <= 3, "Invalid composite manifest")
    require(set(bodies) == {"manifest.json"} | {key + ".tar" for key in manifest["datasets"]},
            "Composite membership differs")
    from hardware_admission import admits  # sealed judge rule
    selected = None
    for key, entry in manifest["datasets"].items():
        require(set(entry) == {"identity", "sha256", "size_bytes", "seal_sha256"}
                and hardware_key(entry["identity"]) == key, "Invalid hardware binding")
        body = bodies[key + ".tar"]
        require(digest(body) == entry["sha256"] and len(body) == entry["size_bytes"],
                "Hardware dataset bytes changed")
        target = scratch / (key + ".tar")
        with target.open("xb") as stream:
            stream.write(body)
        target.chmod(0o400)
        # Existing parser enforces private membership, descriptor hashes and mode.
        inner = read_dataset(target, "dev" if mode == "train" else "final")
        seal = strict_json(inner["seal.json"])
        require(digest(inner["seal.json"]) == entry["seal_sha256"]
                and seal["source_sha256"] == manifest["source_sha256"]
                and seal["hardware_policy"]["identity"] == entry["identity"],
                "Hardware seal differs")
        if admits(entry["identity"], identity):
            require(selected is None, "Ambiguous matching hardware")
            selected = target
    require(selected is not None, "Actual hardware has no accepted dataset")
    return selected
