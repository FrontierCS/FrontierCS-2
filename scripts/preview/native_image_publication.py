"""Digest-preserving image transport for frozen native local-k8s versions.

This deliberately does not register versions or publish definitions. It emits new
immutable documents for the normal checked publication workflow, retaining the
original source/build/dataset evidence. No task catalog or runtime map is changed.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import os
import re
from urllib.parse import urlsplit

from .infra import (
    DIGEST_RE,
    REGISTRY_SERVICE,
    REPOSITORY_ROOT,
    DeployError,
    canonical_json,
    canonical_object_sha256,
    stable_ulid,
)


def _registry(config):
    address = config["network"]["registry_cluster_ip"]
    try:
        if (
            ipaddress.ip_address(address).version != 4
            or ipaddress.ip_address(address).is_loopback
            or ipaddress.ip_address(address).is_unspecified
        ):
            raise ValueError("IPv4 required")
    except ValueError as error:
        raise DeployError(
            "native image registry must be the configured IPv4 cluster IP"
        ) from error
    return address + ":5000"


def _pinned(image):
    reference, digest = image["reference"], image["digest"]
    if not DIGEST_RE.fullmatch(digest):
        raise DeployError("native image digest is invalid")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]*", reference):
        raise DeployError("native image reference is invalid")
    if "@" in reference:
        repository, stated = reference.rsplit("@", 1)
        if stated != digest:
            raise DeployError("native reference and digest disagree")
    else:
        repository = reference
    if ":" in repository.rsplit("/", 1)[-1]:
        repository = repository.rsplit(":", 1)[0]
    if "/" not in repository or any(
        p in {"", ".", ".."} for p in repository.split("/")
    ):
        raise DeployError("native image needs an explicit registry/repository")
    return repository + "@" + digest


def _validate(document, kind):
    import jsonschema

    from .config import reject_secret_fields

    reject_secret_fields(document, "native image document")
    cfg = document["driver_config"]
    schema = REPOSITORY_ROOT / f"docs/contract/schemas/{kind}-config-v1.schema.json"
    jsonschema.validate(cfg, json.loads(schema.read_text()))
    if canonical_object_sha256(cfg) != document["definition_digest"]:
        raise DeployError("native definition digest mismatch")
    if kind == "task":
        from .native_local_publication import dataset_targets

        # Reuse the official native registration precheck, including exact
        # source S3 versions and dataset URI/task/content bindings.
        dataset_targets(document)
        receipt = document["author_receipt"]
        if (
            receipt.get("schema_version") != "frontier.task-author-release/v1"
            or not isinstance(receipt.get("source"), dict)
            or not isinstance(receipt["source"].get("reference"), str)
            or not receipt["source"]["reference"]
            or not DIGEST_RE.fullmatch(receipt["source"].get("sha256", ""))
            or not isinstance(receipt.get("build"), dict)
            or receipt["build"].get("platform")
            != {"os": "linux", "architecture": "amd64"}
            or not DIGEST_RE.fullmatch(receipt["build"].get("inputs_sha256", ""))
            or not isinstance(receipt["build"].get("tool_versions"), dict)
            or not receipt["build"]["tool_versions"]
            or not all(
                isinstance(k, str) and k and isinstance(v, str) and v
                for k, v in receipt["build"]["tool_versions"].items()
            )
        ):
            raise DeployError("native source/build author receipt is incomplete")
        from .publication import receipt_images

        # Management checks the receipt against this projection: declared
        # sidecars by name to tag/digest only, parent resources omitted.
        if (
            receipt["task_version_id"] != cfg["task_version_id"]
            or receipt["definition"]
            != {
                "ref": document["definition_ref"],
                "digest": document["definition_digest"],
            }
            or receipt["images"] != receipt_images(cfg["images"])
        ):
            raise DeployError("native author receipt binding differs")
        if set(receipt["datasets"]) != set(cfg["datasets"]):
            raise DeployError("native dataset receipt modes differ")
        for mode, dataset in cfg["datasets"].items():
            evidence = receipt["datasets"][mode]
            if (
                evidence["uri"] != dataset["uri"]
                or evidence["body_sha256"] != dataset["version"]
                or type(evidence["size_bytes"]) is not int
                or evidence["size_bytes"] <= 0
            ):
                raise DeployError("native dataset receipt binding differs")
    uri = urlsplit(document["definition_ref"])
    if (
        uri.scheme not in {"s3", "preview"}
        or not uri.netloc
        or not uri.path
        or uri.query
        or uri.fragment
        or "%" in uri.path
        or any(p in {".", ".."} for p in uri.path.split("/"))
    ):
        raise DeployError(
            "native definition must have a supported immutable artifact destination"
        )
    if kind == "task" and uri.scheme != "s3":
        raise DeployError("native Task definition requires S3 artifact publication")


def _derive(document, kind, replacements):
    result = copy.deepcopy(document)
    cfg = result["driver_config"]
    for field, reference in replacements.items():
        image = cfg["images"][field] if kind == "task" else cfg["overlay_image"]
        image["reference"] = reference
    if cfg == document["driver_config"]:
        return result
    identity_key = kind + "_version_id"
    cfg[identity_key] = (
        kind[:4]
        + "ver_"
        + stable_ulid(
            canonical_json(
                {
                    "schema": "frontier.native-image-localization/v1",
                    "original_definition_digest": document["definition_digest"],
                    "driver_config": cfg,
                }
            )
        )
    )
    digest = canonical_object_sha256(cfg)
    result["definition_digest"] = digest
    result["definition_ref"] = (
        document["definition_ref"].rsplit("/", 1)[0] + "/" + digest[7:] + ".json"
    )
    result["version"] = document["version"] + "-local-" + digest[7:19]
    if kind == "task":
        receipt = result["author_receipt"]
        receipt[identity_key] = cfg[identity_key]
        receipt["definition"] = {"ref": result["definition_ref"], "digest": digest}
        from .publication import receipt_images

        receipt["images"] = receipt_images(cfg["images"])
    _validate(result, kind)
    return result


def plan(config, document, candidate=None):
    """Pure plan: only references and their immutable binding fields may change."""
    registry = _registry(config)
    _validate(document, "task")
    task_id = document["task_id"]
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", task_id):
        raise DeployError("invalid native Task identity")
    records, replacements = [], {}

    def image_plan(image, repository):
        source = _pinned(image)
        # Retain already-local references exactly, including existing CV IDs.
        if source.split("/", 1)[0] == registry:
            destination = source
            repository = source.split("/", 1)[1].split("@", 1)[0]
        else:
            destination = registry + "/" + repository + "@" + image["digest"]
        records.append(
            {
                "source_reference": source,
                "original_reference": image["reference"],
                "digest": image["digest"],
                "repository": repository,
                "reference": destination,
            }
        )
        return (
            image["reference"] if source.split("/", 1)[0] == registry else destination
        )

    for field, role in [("agent_cpu_base", "agent"), ("verifier", "verifier")]:
        replacements[field] = image_plan(
            document["driver_config"]["images"][field],
            f"frontier/tasks/{task_id.replace('_', '-')}/{role}",
        )
    localized = _derive(document, "task", replacements)
    localized_candidate = None
    if candidate is not None:
        _validate(candidate, "candidate")
        candidate_id = candidate["candidate_id"]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", candidate_id):
            raise DeployError("invalid native Candidate identity")
        reference = image_plan(
            candidate["driver_config"]["overlay_image"],
            "frontier/candidates/native-" + candidate_id.replace("_", "-").lower(),
        )
        localized_candidate = _derive(
            candidate, "candidate", {"overlay_image": reference}
        )
    return {
        "schema": "frontier.native-image-localization/v1",
        "registry": registry,
        "original_document_sha256": canonical_object_sha256(document),
        "original_author_receipt": copy.deepcopy(document["author_receipt"]),
        "original_candidate_sha256": canonical_object_sha256(candidate)
        if candidate
        else None,
        "document": localized,
        "candidate": localized_candidate,
        "images": records,
        "registration_performed": False,
        "definition_upload_performed": False,
    }


def mirror(config, document, candidate=None, *, check):
    """Use the official registry verifier/copy path; exact leaf digests only.

    No credentials are read here. The approved copy tool uses the operator's
    existing registry credential chain. A check performs only registry GET/HEAD.
    """
    from .images import cached_registry_image
    from .infra import executable, port_forward, run_command, state_directory

    # Never accept caller-supplied mirror rows: derive destinations and exact
    # source digests again from the validated frozen documents.
    prepared = plan(config, document, candidate)
    observed = []
    with port_forward(
        state_directory(config) / "kubeconfig", REGISTRY_SERVICE, 5000
    ) as port:
        for row in prepared["images"]:
            digest = cached_registry_image(port, row["repository"], row["digest"])
            if digest is None and not check:
                tool = os.environ.get("FRONTIER_REGISTRY_COPY_TOOL")
                if not tool:
                    raise DeployError(
                        "native image copy requires FRONTIER_REGISTRY_COPY_TOOL"
                    )
                run_command(
                    [
                        executable(tool),
                        "copy",
                        "--platform",
                        "linux/amd64",
                        "--jobs",
                        "2",
                        row["source_reference"],
                        f"127.0.0.1:{port}/{row['repository']}:source-{row['digest'][7:]}",
                    ],
                    timeout=3600,
                )
                digest = cached_registry_image(port, row["repository"], row["digest"])
                if digest is None:
                    raise DeployError("native copy did not retain the frozen manifest")
            if digest is not None and digest != row["digest"]:
                raise DeployError(
                    "native copy must preserve the exact leaf manifest; index substitution is forbidden"
                )
            observed.append({**row, "status": "MATCH" if digest else "MISSING"})
    return observed


def command(arguments):
    from .config import read_and_validate
    from .infra import atomic_write, load_json

    config, _ = read_and_validate(arguments.config)
    document = load_json(arguments.document)
    candidate = load_json(arguments.candidate) if arguments.candidate else None
    result = plan(config, document, candidate)
    if not arguments.plan:
        result["registry_readback"] = mirror(
            config, document, candidate, check=arguments.check
        )
    if arguments.output:
        atomic_write(arguments.output, canonical_json(result), mode=0o600)
    return result
