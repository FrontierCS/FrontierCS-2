#!/usr/bin/env python3
"""Resolve public Docker Hub tags to linux/amd64 digests; optional Docker Hub pull credentials.

All reads finish and validate before the catalog is rewritten. Authentication
uses an anonymous pull grant unless explicit pull credentials are supplied.
Credentials and grants are held in memory and never logged.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path, PurePosixPath
from urllib import parse, request

import frontier_deploy as deploy
from preview.images import linux_amd64_children
from render_task_definition import json_text

ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


def http_bytes(url: str, headers: dict) -> tuple[bytes, object]:
    with request.urlopen(request.Request(url, headers=headers), timeout=30) as response:
        body = response.read(4 * 1024 * 1024 + 1)
        if len(body) > 4 * 1024 * 1024:
            raise ValueError("registry response exceeds 4 MiB")
        return body, response.headers


def validate_descriptor(descriptor: dict) -> None:
    if (
        not isinstance(descriptor, dict)
        or not isinstance(descriptor.get("digest"), str)
        or deploy.DIGEST_RE.fullmatch(descriptor["digest"]) is None
        or type(descriptor.get("size")) is not int
        or descriptor["size"] < 0
        or not isinstance(descriptor.get("mediaType"), str)
        or not descriptor["mediaType"]
    ):
        raise ValueError("invalid registry image descriptor")


def http_blob(url: str, headers: dict, limit: int) -> bytes:
    with request.urlopen(request.Request(url, headers=headers), timeout=60) as response:
        body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError("registry layer exceeds expected payload size bound")
    return body


def verify_layer_members(
    layers: list[bytes], expected: dict[str, bytes], *, executable=False
) -> None:
    observed = {}
    parents = {
        parent.as_posix()
        for name in expected
        for parent in PurePosixPath(name).parents
        if parent != PurePosixPath(".")
    }
    directories = {}
    for body in layers:
        with tarfile.open(fileobj=io.BytesIO(body), mode="r|*") as archive:
            for member in archive:
                name = member.name.removeprefix("./")
                if not member.name or name.startswith("/") or ".." in name.split("/"):
                    raise ValueError("unsafe registry layer member")
                if member.isdir():
                    name = name.rstrip("/") or "."
                if member.isdir():
                    if name not in parents | {"."} or name in observed:
                        raise ValueError(f"unexpected registry layer directory: {name}")
                    if executable and member.mode & 0o555 != 0o555:
                        raise ValueError(
                            f"registry layer directory requires a+rx: {name}"
                        )
                    if name in directories and directories[name] != member.mode:
                        raise ValueError(
                            f"conflicting registry layer directory: {name}"
                        )
                    directories[name] = member.mode
                    continue
                if not member.isfile() or name not in expected or name in observed:
                    raise ValueError(f"unexpected registry layer member: {name}")
                if (
                    member.size != len(expected[name])
                    or executable
                    and member.mode & 0o777 != 0o555
                ):
                    raise ValueError(f"registry layer member size/mode differs: {name}")
                stream = archive.extractfile(member)
                assert stream is not None
                observed[name] = stream.read(member.size + 1)
    if observed != expected:
        raise ValueError("registry layer bodies differ from exported payloads")
    if executable and not parents.issubset(directories):
        raise ValueError("registry layer missing reference parent directories")


def resolve_digest(
    repository: str,
    tag: str,
    *,
    single_layer: bool = False,
    expected_members: dict[str, bytes] | None = None,
    executable=False,
) -> str:
    if not repository.startswith("docker.io/"):
        raise ValueError("only public Docker Hub repositories are supported")
    name = repository.removeprefix("docker.io/")
    if re.fullmatch(r"[a-z0-9._-]+/[a-z0-9._-]+", name) is None:
        raise ValueError("invalid Docker Hub repository")
    query = parse.urlencode(
        {"service": "registry.docker.io", "scope": f"repository:{name}:pull"}
    )
    username = os.environ.get("FRONTIER_DOCKERHUB_USERNAME", "")
    password = os.environ.get("FRONTIER_DOCKERHUB_TOKEN", "")
    if bool(username) != bool(password):
        raise ValueError(
            "Docker Hub pull authentication requires both username and token"
        )
    auth_headers = {}
    if username:
        if ":" in username or any(c in username + password for c in "\r\n"):
            raise ValueError("invalid Docker Hub pull credentials")
        auth_headers["Authorization"] = (
            "Basic " + base64.b64encode((username + ":" + password).encode()).decode()
        )
    body, _ = http_bytes("https://auth.docker.io/token?" + query, auth_headers)
    grant = json.loads(body)["token"]
    if not isinstance(grant, str) or not grant or any(c.isspace() for c in grant):
        raise ValueError("invalid registry grant")
    headers = {"Accept": ACCEPT, "Authorization": "Bearer " + grant}

    def manifest(reference: str) -> tuple[dict, str]:
        body, response_headers = http_bytes(
            f"https://registry-1.docker.io/v2/{name}/manifests/{parse.quote(reference, safe=':')}",
            headers,
        )
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        if response_headers.get("Docker-Content-Digest") != digest:
            raise ValueError("registry manifest body and digest header differ")
        if reference.startswith("sha256:") and reference != digest:
            raise ValueError("registry child manifest differs from requested digest")
        return json.loads(body), digest

    document, digest = manifest(tag)
    if "manifests" in document:
        children = linux_amd64_children(document, null_platform_is_empty=False)
        if (
            len(children) != 1
            or deploy.DIGEST_RE.fullmatch(children[0].get("digest", "")) is None
        ):
            raise ValueError("registry index must contain one linux/amd64 child")
        document, digest = manifest(children[0]["digest"])
    if document.get("schemaVersion") != 2 or not isinstance(
        document.get("layers"), list
    ):
        raise ValueError("registry did not return an image manifest")
    if single_layer and len(document["layers"]) != 1:
        raise ValueError("dataset image must have exactly one layer")
    config = document.get("config")
    validate_descriptor(config)
    for layer in document["layers"]:
        validate_descriptor(layer)
    if config["mediaType"] not in {
        "application/vnd.oci.image.config.v1+json",
        "application/vnd.docker.container.image.v1+json",
    }:
        raise ValueError("registry image config has an unsupported media type")
    body, _ = http_bytes(
        f"https://registry-1.docker.io/v2/{name}/blobs/{config['digest']}", headers
    )
    if (
        "sha256:" + hashlib.sha256(body).hexdigest() != config["digest"]
        or len(body) != config["size"]
    ):
        raise ValueError("registry config blob differs from its descriptor")
    platform = json.loads(body)
    if (
        not isinstance(platform, dict)
        or platform.get("os") != "linux"
        or platform.get("architecture") != "amd64"
    ):
        raise ValueError("registry image config must specify linux/amd64")
    if expected_members is not None:
        layers = []
        limit = sum(len(body) for body in expected_members.values()) + 1024 * 1024
        for layer in document["layers"]:
            if layer["size"] > limit:
                raise ValueError("registry layer exceeds expected payload size bound")
            body = http_blob(
                f"https://registry-1.docker.io/v2/{name}/blobs/{layer['digest']}",
                headers,
                limit,
            )
            if (
                "sha256:" + hashlib.sha256(body).hexdigest() != layer["digest"]
                or len(body) != layer["size"]
            ):
                raise ValueError("registry layer blob differs from its descriptor")
            layers.append(body)
        verify_layer_members(layers, expected_members, executable=executable)
    return digest


def task_images(task: dict) -> list[tuple[dict, bool]]:
    images = [
        (task[key], key == "dataset_image") for key in task if key.endswith("_image")
    ]
    images.extend((image, False) for image in task.get("sidecar_images", {}).values())
    if task.get("reference") is not None:
        images.append((task["reference"], False))
    return images


def resolve_files(
    catalog_path: Path, task_name: str | None = None, *, definitions=None, root=None
) -> None:
    catalog = deploy.load_json(catalog_path)
    deploy.validate_task_catalog(catalog, definitions=definitions, root=root)
    if task_name is not None and task_name not in {
        task["name"] for task in catalog["tasks"]
    }:
        raise ValueError(f"unknown catalog task: {task_name}")
    images = [] if task_name else [(catalog["candidate"]["overlay_image"], False)]
    for task in catalog["tasks"]:
        if task_name is None or task["name"] == task_name:
            images.extend(task_images(task))
    for image, single_layer in images:
        image["digest"] = resolve_digest(
            image["repository"], image["tag"], single_layer=single_layer
        )
    deploy.validate_task_catalog(
        catalog, require_resolved=task_name is None, definitions=definitions, root=root
    )
    if any(image["digest"] == deploy.TASK_PLACEHOLDER_DIGEST for image, _ in images):
        raise ValueError("resolver returned a placeholder")
    deploy.atomic_write(catalog_path, json_text(catalog).encode())


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=deploy.TASK_CATALOG_PATH)
    parser.add_argument(
        "--task", help="resolve only this task; leave all other pins unchanged"
    )
    options = parser.parse_args(arguments)
    resolve_files(options.catalog, options.task)
    print("Resolved catalog image digests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
