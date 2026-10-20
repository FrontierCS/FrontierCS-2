"""Immutable definition artifacts for native image localization.

The version-history checks follow FrontierSmith's official `_put_versioned`
publication contract. Conditional creation prevents overwriting an object that
appeared after the read-only history check. No registration is implicit.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from .infra import DeployError, canonical_object_sha256


def definition_bytes(document):
    # Definition hashes exclude the newline used for ordinary JSON receipts.
    body = json.dumps(
        document["driver_config"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    if "sha256:" + hashlib.sha256(body).hexdigest() != document["definition_digest"]:
        raise DeployError("derived definition byte digest differs")
    return body


def artifact_target(document):
    uri = urlsplit(document["definition_ref"])
    body = definition_bytes(document)
    if (
        uri.scheme != "s3"
        or not uri.netloc
        or uri.query
        or uri.fragment
        or "@" in uri.netloc
        or ":" in uri.netloc
        or "%" in uri.path
        or not uri.path.startswith("/definitions/")
        or any(p in {"", ".", ".."} for p in uri.path[1:].split("/"))
        or not uri.path.endswith("/" + document["definition_digest"][7:] + ".json")
    ):
        raise DeployError(
            "derived definition requires its immutable S3 content-addressed key"
        )
    return uri.netloc, uri.path[1:], body


class AwsDefinitions:
    """Use the same AWS CLI credential chain as native-publish; never read keys."""

    def __init__(self, config):
        self.config = config

    def call(self, operation, bucket, key, *args):
        from .infra import executable, run_command

        return json.loads(
            run_command(
                [
                    executable("aws"),
                    "s3api",
                    operation,
                    "--bucket",
                    bucket,
                    "--key",
                    key,
                    "--region",
                    self.config["site"]["aws_region"],
                    "--expected-bucket-owner",
                    self.config["site"]["expected_account_id"],
                    *args,
                ],
                capture=True,
                timeout=300,
            )
        )

    def history(self, bucket, key):
        from .infra import executable, run_command

        return json.loads(
            run_command(
                [
                    executable("aws"),
                    "s3api",
                    "list-object-versions",
                    "--bucket",
                    bucket,
                    "--prefix",
                    key,
                    "--max-keys",
                    "100",
                    "--no-paginate",
                    "--region",
                    self.config["site"]["aws_region"],
                    "--expected-bucket-owner",
                    self.config["site"]["expected_account_id"],
                ],
                capture=True,
                timeout=300,
            )
        )

    def versioned(self, bucket):
        from .infra import executable, run_command

        value = json.loads(
            run_command(
                [
                    executable("aws"),
                    "s3api",
                    "get-bucket-versioning",
                    "--bucket",
                    bucket,
                    "--region",
                    self.config["site"]["aws_region"],
                    "--expected-bucket-owner",
                    self.config["site"]["expected_account_id"],
                ],
                capture=True,
                timeout=300,
            )
        )
        return value.get("Status") == "Enabled"

    def get(self, bucket, key, version):
        meta = self.call("head-object", bucket, key, "--version-id", version)
        if (
            type(meta.get("ContentLength")) is not int
            or not 0 < meta["ContentLength"] <= 4 * 1024 * 1024
        ):
            raise DeployError("definition artifact exceeds bounded read size")
        from .infra import state_directory

        root = state_directory(self.config) / "native-definition-scratch"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "definition.json"
            metadata = self.call(
                "get-object", bucket, key, "--version-id", version, str(path)
            )
            if path.stat().st_size > 4 * 1024 * 1024:
                raise DeployError("definition artifact exceeds bounded read size")
            return metadata, path.read_bytes()

    def put(self, bucket, key, body):
        from .infra import state_directory

        root = state_directory(self.config) / "native-definition-scratch"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "definition.json"
            path.write_bytes(body)
            path.chmod(0o600)
            return self.call(
                "put-object",
                bucket,
                key,
                "--body",
                str(path),
                "--content-type",
                "application/json",
                "--metadata",
                "sha256=" + hashlib.sha256(body).hexdigest(),
                "--if-none-match",
                "*",
            )


def publish_definition(document, client, *, check):
    bucket, key, body = artifact_target(document)
    if len(body) > 4 * 1024 * 1024:
        raise DeployError("definition artifact exceeds bounded write size")
    digest = document["definition_digest"]
    history = client.history(bucket, key)
    if history.get("IsTruncated") or any(
        x.get("Key") == key for x in history.get("DeleteMarkers", [])
    ):
        raise DeployError("definition version history is ambiguous or deleted")
    versions = [v for v in history.get("Versions", []) if v.get("Key") == key]
    verified = []

    def read(version):
        if not isinstance(version, str) or not version or version in {"null", "None"}:
            raise DeployError("definition requires a real S3 VersionId")
        meta, actual = client.get(bucket, key, version)
        if (
            actual != body
            or meta.get("VersionId") != version
            or meta.get("ContentLength") != len(body)
            or meta.get("ContentType") != "application/json"
            or meta.get("Metadata", {}).get("sha256") != digest[7:]
        ):
            raise DeployError("definition exact-version readback differs")

    for version in versions:
        read(version.get("VersionId"))
        verified.append((str(version.get("LastModified")), version["VersionId"]))
    created = False
    if verified:
        version = sorted(verified)[0][1]
    elif check:
        return {
            "uri": document["definition_ref"],
            "body_sha256": digest,
            "status": "MISSING",
            "uploaded": False,
        }
    else:
        # A conflicting conditional write fails; never fall back to an overwrite.
        if not client.versioned(bucket):
            raise DeployError("definition destination must have S3 versioning enabled")
        response = client.put(bucket, key, body)
        version = response.get("VersionId")
        read(version)
        created = True
    return {
        "uri": document["definition_ref"],
        "body_sha256": digest,
        "size_bytes": len(body),
        "s3_version_id": version,
        "status": "MATCH",
        "uploaded": created,
    }


def rebind_transport(original, localized, bundle, output):
    """Rebind verified custody to a ref-only localization; OCI bytes stay exact."""
    import copy

    from .infra import canonical_json
    from .native_local_publication import verify_transport_bundle

    provenance = verify_transport_bundle(original, bundle)
    # Only new immutable identity and CPU image references are permitted.
    cfg = copy.deepcopy(localized["driver_config"])
    cfg["task_version_id"] = original["driver_config"]["task_version_id"]
    for role in ("agent_cpu_base", "verifier"):
        cfg["images"][role]["reference"] = original["driver_config"]["images"][role][
            "reference"
        ]
    if cfg != original["driver_config"]:
        raise DeployError("dataset rebind includes non-transport TaskConfig changes")
    for key in ("source", "build", "datasets"):
        if localized["author_receipt"][key] != original["author_receipt"][key]:
            raise DeployError("dataset rebind changed original provenance")
    # Full original bundle verification happens before any new local output.
    output.mkdir(parents=True, exist_ok=False)
    (output / "blobs/sha256").mkdir(parents=True)
    index = json.loads((bundle / "index.json").read_bytes())
    manifest_digest = index["manifests"][0]["digest"][7:]
    manifest = json.loads((bundle / "blobs/sha256" / manifest_digest).read_bytes())
    hashes = {
        manifest_digest,
        manifest["config"]["digest"][7:],
        *[v["digest"][7:] for v in manifest["layers"]],
    }
    for digest in hashes:
        source = bundle / "blobs/sha256" / digest
        destination = output / "blobs/sha256" / digest
        before = source.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise DeployError("transport blob must be a regular file")
        # No payload copy or symlink substitution. Cross-device linking fails
        # closed; the operator must select a same-filesystem output directory.
        os.link(source, destination, follow_symlinks=False)
        after = destination.lstat()
        if (after.st_dev, after.st_ino, after.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise DeployError("transport hardlink identity differs")
    for name in ("oci-layout", "index.json"):
        (output / name).write_bytes((bundle / name).read_bytes())
    updated = {
        **provenance,
        "original_receipt": copy.deepcopy(localized["author_receipt"]),
    }
    (output / "source-document.json").write_bytes(canonical_json(localized))
    (output / "provenance.json").write_bytes(canonical_json(updated))
    verify_transport_bundle(localized, output)
    return {
        "original_document_sha256": canonical_object_sha256(original),
        "localized_document_sha256": canonical_object_sha256(localized),
        "transport_manifest_digest": provenance["transport_manifest_digest"],
        "original_bundle": str(bundle),
        "rebound_bundle": str(output),
        "dataset_content_changed": False,
        "payload_copy_bytes": 0,
        "blob_transport": "verified-regular-hardlinks",
    }


def validate_candidate_readback(expected, actual):
    if actual.get("id") != expected["driver_config"]["candidate_version_id"]:
        raise DeployError("native CandidateVersion readback identity differs")
    for key, value in expected.items():
        if key == "driver_config" and actual.get(key) is None:
            continue  # Independently verified immutable definition binds it.
        if actual.get(key) != value:
            raise DeployError("native CandidateVersion readback differs: " + key)


def command(arguments):
    from .config import read_and_validate
    from .infra import (
        MANAGEMENT_SERVICE,
        atomic_write,
        canonical_json,
        load_json,
        port_forward,
        state_directory,
    )
    from .native_image_publication import mirror, plan
    from .native_local_publication import read_definition
    from .publication import ensure_version, find_version

    config, _ = read_and_validate(arguments.config)
    original = load_json(arguments.document)
    candidate = load_json(arguments.candidate) if arguments.candidate else None
    result = plan(config, original, candidate)
    result["stage"] = arguments.stage
    if arguments.stage == "rebind":
        result["dataset_rebind"] = rebind_transport(
            original, result["document"], arguments.bundle, arguments.rebound_bundle
        )
        atomic_write(arguments.output, canonical_json(result), mode=0o600)
        return result
    rows = mirror(config, original, candidate, check=True)
    result["registry_readback"] = rows
    if not arguments.check and any(row["status"] != "MATCH" for row in rows):
        raise DeployError("all frozen CPU images must be mirrored before publication")
    client = AwsDefinitions(config)
    if arguments.stage == "definitions":
        result["definitions"] = {}
        atomic_write(arguments.output, canonical_json(result), mode=0o600)
        for name, before in [("document", original), ("candidate", candidate)]:
            expected = result[name]
            if expected is None or expected == before:
                continue
            # Verify the original immutable artifact before deriving a new one;
            # never trust an edited local config in place of its frozen binding.
            fetched = read_definition(
                {
                    "definition_ref": before["definition_ref"],
                    "definition_digest": before["definition_digest"],
                },
                config,
            )
            if fetched["driver_config"] != before["driver_config"]:
                raise DeployError(
                    "original definition artifact differs from frozen config"
                )
            old_uri, new_uri = (
                urlsplit(before["definition_ref"]),
                urlsplit(expected["definition_ref"]),
            )
            if (
                new_uri.netloc != old_uri.netloc
                or new_uri.path.rsplit("/", 1)[0] != old_uri.path.rsplit("/", 1)[0]
            ):
                raise DeployError("derived definition changes the original destination")
            result["definitions"][name] = publish_definition(
                expected, client, check=arguments.check
            )
            atomic_write(arguments.output, canonical_json(result), mode=0o600)
        result["definition_upload_performed"] = any(
            r["uploaded"] for r in result["definitions"].values()
        )
    elif arguments.stage == "candidate":
        expected = result["candidate"]
        if expected is None:
            raise DeployError(
                "Candidate registration requires the original Candidate document"
            )
        # A changed Candidate must have a published S3 artifact. Existing local
        # Candidates are checked in Management and never re-created here.
        changed = expected != candidate
        if urlsplit(candidate["definition_ref"]).scheme == "s3":
            original_artifact = read_definition(
                {
                    "definition_ref": candidate["definition_ref"],
                    "definition_digest": candidate["definition_digest"],
                },
                config,
            )
            if original_artifact["driver_config"] != candidate["driver_config"]:
                raise DeployError(
                    "original Candidate artifact differs from frozen config"
                )
        elif changed:
            raise DeployError(
                "changed native Candidate requires its original S3 artifact"
            )
        if changed:
            artifact = publish_definition(expected, client, check=True)
            if artifact["status"] != "MATCH":
                raise DeployError("localized Candidate definition is not published")
        version = expected["driver_config"]["candidate_version_id"]
        with port_forward(
            state_directory(config) / "kubeconfig", MANAGEMENT_SERVICE, 8080
        ) as port:
            base = f"http://127.0.0.1:{port}"
            actual = find_version(base, "/api/v1/candidate-versions", version)
            if actual is not None:
                if (
                    urlsplit(expected["definition_ref"]).scheme != "s3"
                    and actual.get("driver_config") is None
                ):
                    raise DeployError(
                        "non-S3 local Candidate requires full config readback"
                    )
                validate_candidate_readback(expected, actual)
            elif not changed:
                raise DeployError("unchanged local Candidate must already exist")
            if not arguments.check and actual is None:
                # Persist exact immutable intent before the potentially ambiguous POST.
                atomic_write(arguments.output, canonical_json(result), mode=0o600)
                ensure_version(base, "/api/v1/candidate-versions", version, expected)
                actual = find_version(base, "/api/v1/candidate-versions", version)
                if actual is None:
                    raise DeployError("native CandidateVersion readback missing")
                validate_candidate_readback(expected, actual)
                result["registration_performed"] = True
            result["candidate_registration"] = {
                "candidate_version_id": version,
                "status": "MATCH" if actual is not None else "MISSING",
                "readback": actual,
            }
    atomic_write(arguments.output, canonical_json(result), mode=0o600)
    return result
