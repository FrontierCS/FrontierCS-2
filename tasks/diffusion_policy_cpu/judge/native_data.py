"""Bounded private dataset staging for the existing native Verifier protocol."""

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile

from task_guards import TaskIntegrityError

MAX_DATASET_BYTES = 2 * 1024**2
MAX_MEMBER_BYTES = 256 * 1024


def require(condition, message):
    if not condition:
        raise TaskIntegrityError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate private JSON key")
            result[key] = value
        return result
    def constant(value):
        raise TaskIntegrityError("Nonfinite private JSON value")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def native_mode(environ):
    mode = environ.get("FRONTIER_VERIFICATION_MODE")
    require(mode in {"train", "eval"}, "A trusted native verification mode is required")
    expected = "final" if mode == "eval" else "dev"
    role = "final" if environ.get("FRONTIER_SUBMISSION_ROLE") == "final" else "dev"
    require(role == expected, "Native and evaluator modes disagree")
    return expected


def members_for(mode):
    require(mode in {"dev", "final"}, "Unknown private dataset mode")
    return {"seal.json", mode + ".json", mode + "-constants.json",
            mode + "-reference.json", "reference-acceptance.json"}


def read_dataset(path, mode):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= MAX_DATASET_BYTES,
            "Private dataset must be a bounded regular archive")
    required = members_for(mode) | {"context.json"}
    bodies = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(path.read_bytes()), mode="r:") as archive:
            total = 0
            for member in archive:
                require(member.name in required and member.name not in bodies and member.isfile()
                        and not member.issym() and not member.islnk()
                        and not member.pax_headers and member.sparse is None,
                        "Unexpected private dataset member")
                require(0 < member.size <= MAX_MEMBER_BYTES, "Private dataset member exceeds its bound")
                total += member.size
                require(total <= MAX_DATASET_BYTES, "Private dataset expands beyond its bound")
                stream = archive.extractfile(member)
                require(stream is not None, "Unreadable private dataset member")
                body = stream.read(MAX_MEMBER_BYTES + 1)
                require(len(body) == member.size, "Truncated private dataset member")
                strict_json(body)
                bodies[member.name] = body
    except (tarfile.TarError, OSError, ValueError) as error:
        raise TaskIntegrityError("Malformed private dataset archive") from error
    require(set(bodies) == required, "Private dataset is incomplete")
    try:
        descriptor = strict_json(bodies["context.json"])
    except (ValueError, UnicodeError) as error:
        raise TaskIntegrityError("Malformed private dataset descriptor") from error
    require(isinstance(descriptor, dict) and set(descriptor) == {"version", "mode", "purpose", "files"}
            and type(descriptor["version"]) is int and descriptor["version"] == 1
            and descriptor["mode"] == mode and descriptor["purpose"] == "accepted_evaluation"
            and isinstance(descriptor["files"], dict) and set(descriptor["files"]) == members_for(mode),
            "Private dataset descriptor does not match the trusted mode")
    for name, expected in descriptor["files"].items():
        require(hashlib.sha256(bodies[name]).hexdigest() == expected, "Private dataset content hash mismatch")
    return bodies


@contextmanager
def materialize_dataset(path, mode, parent):
    bodies = read_dataset(path, mode)
    with tempfile.TemporaryDirectory(prefix="private-context-", dir=parent) as directory:
        root = Path(directory)
        root.chmod(0o700)
        for name, body in bodies.items():
            target = root / name
            with target.open("xb") as stream:
                stream.write(body)
            target.chmod(0o400)
        yield root


def package_dataset(private_root, mode, output):
    """Author-only assembly; native staging reads only the selected archive."""
    from measurement_contract import validate_final_extension
    from private_context import file_sha256, load_context, accepted_calibrations
    root = Path(private_root)
    context = load_context(root=root, verify_runtime=False)
    validate_final_extension(context["suites"]["dev"], context["suites"]["final"])
    accepted_calibrations(context)
    bodies = {}
    for name in members_for(mode):
        target = root / name
        require(target.is_file() and not target.is_symlink(), "Private packaging input is missing")
        bodies[name] = target.read_bytes()
    # The mode-specific acceptance contains no reference metadata for the other mode.
    acceptance = json.loads(bodies["reference-acceptance.json"])
    acceptance["modes"] = {mode: acceptance["modes"][mode]}
    bodies["reference-acceptance.json"] = json.dumps(acceptance, sort_keys=True).encode()
    descriptor = {"version": 1, "mode": mode, "purpose": "accepted_evaluation",
                  "files": {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()}}
    bodies["context.json"] = json.dumps(descriptor, sort_keys=True).encode()
    output = Path(output)
    with output.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, body in sorted(bodies.items()):
                require(0 < len(body) <= MAX_MEMBER_BYTES, "Private packaging member exceeds its bound")
                member = tarfile.TarInfo(name)
                member.size, member.mode, member.mtime = len(body), 0o400, 0
                archive.addfile(member, io.BytesIO(body))
    output.chmod(0o400)
    read_dataset(output, mode)
    return file_sha256(output)
