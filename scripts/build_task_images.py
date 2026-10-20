#!/usr/bin/env python3
"""Build the bundled CPU and native GPU demo images locally.

This entrypoint deliberately has no AWS, Kubernetes, registry-login, or secret
handling.  A push is opt-in and relies on the operator's existing Docker
credential configuration without reading or copying it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Sequence

import render_task_definition as renderer
from native_datasets import dataset_path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPOSITORY_ROOT / "tasks" / "frontier-cs-2.0-demo"
TASK_CATALOG = REPOSITORY_ROOT / "config" / "task-catalog.json"
SPARSE_VIDEO = "sparse_video_attention_policy"
IMAGE_PREFIX_RE = re.compile(
    r"^(?:[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?/)*"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*$"
)
DIGEST_REFERENCE_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


class BuildFailure(RuntimeError):
    """A demo image could not be built with frozen inputs."""


def run(
    command: Sequence[str], *, capture: bool = False, input_text: str | None = None
) -> str:
    result = subprocess.run(
        list(command),
        cwd=REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        input=input_text,
    )
    if result.returncode != 0:
        detail = result.stderr[-2000:].strip() if capture and result.stderr else ""
        raise BuildFailure(
            f"{command[0]} failed with exit {result.returncode}"
            + (f": {detail}" if detail else "")
        )
    return result.stdout.strip() if capture else ""


def pull_digest_reference(source: str) -> str:
    run(["docker", "pull", "--platform", "linux/amd64", source])
    if DIGEST_REFERENCE_RE.fullmatch(source):
        # A local image can have multiple RepoDigests. Preserve the exact
        # requested immutable identity, not an unrelated first alias.
        return source
    reference = run(
        ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", source],
        capture=True,
    )
    if DIGEST_REFERENCE_RE.fullmatch(reference) is None:
        raise BuildFailure(f"Docker did not resolve {source} to one immutable digest")
    return reference


def gpu_ssh_base() -> str:
    """Reuse only the trusted bootstrap binary from the accepted public image."""
    try:
        catalog = json.loads(TASK_CATALOG.read_text(encoding="utf-8"))
        task = next(task for task in catalog["tasks"] if task["name"] == SPARSE_VIDEO)
        image = task["agent_gpu_image"]
        reference = image["repository"] + "@" + image["digest"]
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        raise BuildFailure(
            "cannot resolve sparse-video bootstrap from task-catalog.json"
        ) from exc
    if (
        not reference.startswith("docker.io/")
        or DIGEST_REFERENCE_RE.fullmatch(reference) is None
    ):
        raise BuildFailure(
            "GPU SSH bootstrap must be a public Docker Hub digest reference"
        )
    return reference


BASE_IMAGES = {"ubuntu:24.04", "buildpack-deps:noble", "python:3.11-slim-bookworm"}
# Exact reviewed public SDK used by MetaOpt; no general MCR registry approval.
# Registry/config/license proof and upstream PR35 pin: docs/tool-compatibility-20261001.md.
REVIEWED_PUBLIC_BASES = {
    "mcr.microsoft.com/dotnet/sdk:8.0-bookworm-slim@sha256:3ee581addaa35c8c790f5f62441b189810522bb32e43a8db9d4283b3f649cc24",
}
SSH_BASE_ARGUMENT = "${FRONTIER_GPU_SSH_BASE_IMAGE}"


def approved_base(source: str) -> bool:
    """Accept reviewed tags or digest-pinned Docker Hub names, including shorthand."""
    if source in REVIEWED_PUBLIC_BASES:
        return True
    name = source.removeprefix("docker.io/")
    repository = name.split("@", 1)[0].rsplit("/", 1)
    # A registry hostname/port must never be interpreted as a Hub namespace.
    if len(repository) == 2 and (
        "." in repository[0] or ":" in repository[0] or repository[0] == "localhost"
    ):
        return False
    if (
        re.fullmatch(
            r"(?:[a-z0-9_-]+/)?[a-z0-9._-]+(?::[A-Za-z0-9_.-]+)?"
            r"(?:@sha256:[0-9a-f]{64})?",
            name,
        )
        is None
    ):
        return False
    return name.removeprefix("library/") in BASE_IMAGES or bool(
        DIGEST_REFERENCE_RE.fullmatch(name)
    )


def dockerfile_bases(dockerfile: Path) -> list[str]:
    """Parse and validate every FROM before either caller can pull or build."""
    sources, stages = [], set()
    for line in dockerfile_instructions(dockerfile):
        if re.match(r"^\s*FROM(?:\s|$)", line, re.IGNORECASE) is None:
            continue
        match = re.fullmatch(
            r"\s*FROM\s+(?:--platform=\S+\s+)?(\S+)"
            r"(?:\s+AS\s+([a-zA-Z0-9_.-]+))?\s*",
            line,
            re.IGNORECASE,
        )
        if match is None:
            raise BuildFailure(f"invalid FROM in {dockerfile}: {line.strip()}")
        source, alias = match.groups()
        if source.lower() not in stages:
            if source not in {"scratch", SSH_BASE_ARGUMENT} and not approved_base(
                source
            ):
                raise BuildFailure(
                    f"unreviewed or private base image in {dockerfile}: {source}"
                )
            sources.append(source)
        if alias:
            stages.add(alias.lower())
    if not sources:
        raise BuildFailure(f"missing FROM in {dockerfile}")
    return sources


def dockerfile_instructions(dockerfile: Path):
    """Join instruction continuations, removing comments and heredoc bodies."""
    text = dockerfile.read_text(encoding="utf-8")
    # Parser directives precede everything else and can change the escape
    # character; this parser only understands the default backslash.
    for line in text.splitlines():
        directive = re.fullmatch(r"#\s*([A-Za-z]+)\s*=\s*(.*?)\s*", line.strip())
        if directive is None:
            break
        if directive.group(1).lower() == "escape" and directive.group(2) != "\\":
            raise BuildFailure(
                f"unsupported parser directive in {dockerfile}: {line.strip()}"
            )
    lines = iter(text.splitlines())
    instruction = ""
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        line = line.rstrip()
        if line.endswith("\\"):
            instruction += line[:-1]
            continue
        instruction += line
        yield instruction
        if re.match(r"^\s*(?:RUN|COPY|ADD)\s", instruction, re.IGNORECASE):
            # Docker recognizes whole heredoc words (with an optional file
            # descriptor). Preserve quotes/escapes until the marker is identified:
            # echo '<<EOF' and echo prefix<<EOF must not hide later instructions.
            words = re.findall(
                r"(?:[^\s'\"\\]+|'[^']*'|\"(?:\\.|[^\"\\])*\"|\\.)+", instruction
            )
            for word in words:
                match = re.fullmatch(r"\d*<<(?P<tabs>-?)(?P<word>[^<]+)", word)
                if match is None:
                    continue
                terminator = shlex.split(match.group("word"))[0]
                for body in lines:
                    if (
                        body.lstrip("\t") if match.group("tabs") else body
                    ) == terminator:
                        break
                else:
                    raise BuildFailure(
                        f"unterminated heredoc in {dockerfile}: {terminator}"
                    )
        instruction = ""
    if instruction:
        yield instruction


def sidecar_roles(definition: dict) -> list[str]:
    """Image roles behind the declared sidecars; several sidecars may share one."""
    profiles = definition.get("sidecars") or {}
    return sorted({profile.get("image", name) for name, profile in profiles.items()})


def image_names(
    prefix: str, problem: str, tag: str, sidecars: tuple[str, ...] | list[str] = ()
) -> dict[str, str]:
    names = {
        "agent": f"{prefix}/fcs2-task-{problem}-agent:{tag}",
        "verifier": f"{prefix}/fcs2-task-verifier-{problem}:{tag}",
        "agent-gpu": f"{prefix}/fcs2-task-{problem}-agent-gpu:{tag}",
        "verifier-gpu": f"{prefix}/fcs2-task-verifier-{problem}-gpu:{tag}",
        "datasets": f"{prefix}/fcs2-task-{problem}-datasets:v1",
    }
    # Declared sidecars are task-owned images too; Modal imports them beside
    # the GPU runner, so they follow the GPU image naming and source tag.
    for sidecar in sidecars:
        names[f"sidecar-{sidecar}"] = f"{prefix}/fcs2-task-{problem}-sidecar-{sidecar}:{tag}"
    return names


def freeze_bases(
    dockerfile: Path, bases: dict[str, str], ssh_base: str | None
) -> list[str]:
    arguments = []
    for source in dockerfile_bases(dockerfile):
        if source == "scratch":
            continue
        if source == "${FRONTIER_GPU_SSH_BASE_IMAGE}":
            if (
                not ssh_base
                or DIGEST_REFERENCE_RE.fullmatch(ssh_base) is None
                or not approved_base(ssh_base)
            ):
                raise BuildFailure(
                    "FRONTIER_GPU_SSH_BASE_IMAGE must be non-empty and digest pinned"
                )
            bases[source] = ssh_base
            arguments.extend(["--build-arg", f"FRONTIER_GPU_SSH_BASE_IMAGE={ssh_base}"])
            continue
        if source not in bases:
            bases[source] = pull_digest_reference(source)
        # BuildKit image contexts override FROM resolution without editing the
        # Dockerfile or the task tree whose exact bytes define the release.
        if not DIGEST_REFERENCE_RE.fullmatch(source):
            arguments.extend(
                ["--build-context", f"{source}=docker-image://{bases[source]}"]
            )
    return arguments


def stage_datasets(root: Path, context: Path, definition: dict) -> dict[str, bytes]:
    members = {}
    extension = definition["dataset_extension"]
    for mode in ("train", "eval"):
        digest = definition["datasets"][f"{mode}_sha256"].removeprefix("sha256:")
        member = f"datasets/{mode}/sha256/{digest}.{extension}"
        body = dataset_path(root, mode, extension).read_bytes()
        if hashlib.sha256(body).hexdigest() != digest:
            raise BuildFailure(f"dataset selector changed: {mode}")
        path = context / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        members[member] = body
    (context / "Dockerfile").write_text("FROM scratch\nCOPY datasets/ /datasets/\n")
    return members


def verify_dataset_image(archive_path: Path, expected: dict[str, bytes]) -> None:
    with tarfile.open(archive_path) as archive:
        manifest_file = archive.extractfile("manifest.json")
        if manifest_file is None:
            raise BuildFailure("dataset image has no save manifest")
        manifest = json.load(manifest_file)
        if len(manifest) != 1 or len(manifest[0]["Layers"]) != 1:
            raise BuildFailure("dataset image must have exactly one layer")
        layer_file = archive.extractfile(manifest[0]["Layers"][0])
        if layer_file is None:
            raise BuildFailure("dataset image layer missing")
        with tarfile.open(fileobj=layer_file) as layer:
            observed = {}
            for member in layer:
                if member.isdir():
                    continue
                name = member.name.removeprefix("./")
                if not member.isfile() or name not in expected or name in observed:
                    raise BuildFailure(f"unexpected dataset layer member: {name}")
                if member.size != len(expected[name]):
                    raise BuildFailure(f"dataset member size differs: {name}")
                stream = layer.extractfile(member)
                assert stream is not None
                observed[name] = stream.read()
            if observed != expected:
                raise BuildFailure(
                    "dataset image member bodies differ from the selectors"
                )


def build_problem(
    problem: str,
    prefix: str,
    *,
    push: bool,
    bases: dict[str, str],
    root: Path | None = None,
    definition: dict | None = None,
) -> dict:
    root = root or renderer.problem_directory(TASK_ROOT / "problems", problem)
    definition = definition or renderer.render(root, problem)
    sidecars = sidecar_roles(definition)
    names = image_names(prefix, problem, definition["image_tag"], sidecars)
    gpu = definition.get("gpu", {})
    agent_gpu = gpu.get("agent", {}).get("max_instances", 0) > 0
    verifier_gpu = gpu.get("verifier", {}).get("max_gpu_instances", 0) > 0
    gpu_dockerfiles = [
        root / role / "Dockerfile"
        for role, enabled in (("agent-gpu", agent_gpu), ("verifier-gpu", verifier_gpu))
        if enabled
    ] + [root / f"sidecar-{sidecar}" / "Dockerfile" for sidecar in sidecars]
    needs_ssh = any(
        SSH_BASE_ARGUMENT in dockerfile_bases(path) for path in gpu_dockerfiles
    )
    ssh_base = pull_digest_reference(gpu_ssh_base()) if needs_ssh else None
    built = {}
    for directory, component in (
        ("agent-base", "agent"),
        ("verifier", "verifier"),
        ("agent-gpu", "agent-gpu"),
        ("verifier-gpu", "verifier-gpu"),
        *((f"sidecar-{sidecar}", f"sidecar-{sidecar}") for sidecar in sidecars),
    ):
        if (component == "agent-gpu" and not agent_gpu) or (
            component == "verifier-gpu" and not verifier_gpu
        ):
            continue
        if not (root / directory).is_dir():
            raise BuildFailure(f"missing enabled image directory: {directory}")
        dockerfile = root / directory / "Dockerfile"
        arguments = freeze_bases(dockerfile, bases, ssh_base)
        run(
            [
                "docker",
                "build",
                "--provenance=false",
                "--platform",
                "linux/amd64",
                "-f",
                str(dockerfile),
                *arguments,
                "--tag",
                names[component],
                str(root),
            ]
        )
        built[component] = names[component]
    with tempfile.TemporaryDirectory(prefix="frontier-datasets-") as directory:
        context = Path(directory) / "context"
        context.mkdir()
        members = stage_datasets(root, context, definition)
        run(
            [
                "docker",
                "build",
                "--provenance=false",
                "--platform",
                "linux/amd64",
                "-f",
                str(context / "Dockerfile"),
                "--tag",
                names["datasets"],
                str(context),
            ]
        )
        archive = Path(directory) / "datasets.tar"
        run(["docker", "image", "save", "--output", str(archive), names["datasets"]])
        verify_dataset_image(archive, members)
        built["datasets"] = names["datasets"]
    if push:
        for image in built.values():
            run(["docker", "push", image])
    return {
        "source": definition["source"],
        "datasets": definition["datasets"],
        "images": built,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    catalog_error = None
    try:
        catalog_problems = tuple(renderer.catalog_task_names())
    except ValueError as exc:
        # Help remains available in an incomplete checkout. A build still
        # requires valid membership and fails before any build side effects.
        catalog_problems, catalog_error = (), exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        action="append",
        choices=("all", "cpu", *catalog_problems),
        default=[],
    )
    parser.add_argument(
        "--image-prefix",
        default=os.environ.get(
            "TASK_IMAGE_PREFIX", os.environ.get(
                "DEMO_IMAGE_PREFIX", os.environ.get("FRONTIER_IMAGE_NAMESPACE", "joyemang33")
            )
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / ".generated/native-build.json",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="push after verification; never performs docker login",
    )
    parser.add_argument(
        "--incremental-from",
        help="verify this previous Git commit and plan source-only OCI layers; --push publishes them",
    )
    parser.add_argument(
        "--crane",
        default=os.environ.get("FRONTIER_REGISTRY_COPY_TOOL", "crane"),
        help="crane executable for optional incremental publication",
    )
    options = parser.parse_args(arguments)
    if catalog_error is not None:
        raise BuildFailure(str(catalog_error)) from catalog_error
    if "DEMO_IMAGE_PREFIX" in os.environ and "TASK_IMAGE_PREFIX" not in os.environ:
        print(
            "deprecated: DEMO_IMAGE_PREFIX; use TASK_IMAGE_PREFIX (alias retained for one release)",
            file=sys.stderr,
        )
    if IMAGE_PREFIX_RE.fullmatch(options.image_prefix) is None:
        raise BuildFailure("--image-prefix is not a bounded Docker repository prefix")
    if options.manifest.resolve().is_relative_to((TASK_ROOT / "problems").resolve()):
        raise BuildFailure("build manifest must live outside task trees")
    requested = options.problem or ["all"]
    try:
        directories = renderer.catalog_problem_directories(
            catalog_problems, TASK_ROOT / "problems"
        )
        definitions = json.loads(renderer.DEFINITIONS.read_text(encoding="utf-8"))
        for name, root in directories.items():
            if name not in definitions:
                raise ValueError(f"catalog task {name} has no rendered definition")
            if not isinstance(definitions[name], dict):
                raise ValueError(
                    f"catalog task {name} rendered definition must be an object"
                )
            policy_path = root / "policy.json"
            if policy_path.is_file():
                policy = json.loads(policy_path.read_text())
                if definitions[name].get("gpu") != policy.get("gpu"):
                    raise ValueError(
                        f"GPU definition disagrees with native policy: {name}"
                    )
            gpu = definitions[name].get("gpu", {})
            for role, quota in (
                ("agent", "max_instances"),
                ("verifier", "max_gpu_instances"),
            ):
                if (
                    gpu.get(role, {}).get(quota, 0) > 0
                    and not (root / f"{role}-gpu").is_dir()
                ):
                    raise ValueError(
                        f"GPU definition disagrees with {role}-gpu/ directory: {name}"
                    )
        cpu = tuple(
            name
            for name in catalog_problems
            if not definitions[name]
            .get("gpu", {})
            .get("agent", {})
            .get("max_instances", 0)
            and not definitions[name]
            .get("gpu", {})
            .get("verifier", {})
            .get("max_gpu_instances", 0)
        )
    except (OSError, ValueError) as exc:
        raise BuildFailure(str(exc)) from exc
    problems = (
        catalog_problems
        if "all" in requested
        else tuple(
            dict.fromkeys(
                name
                for selection in requested
                for name in (cpu if selection == "cpu" else (selection,))
            )
        )
    )
    if options.incremental_from:
        from preview import images as incremental

        commit = incremental.resolve_commit(REPOSITORY_ROOT, options.incremental_from)
        directory = options.manifest.with_suffix(".layers")
        manifest = {
            "schema_version": "frontier.preview-incremental-build/v1",
            "platform": "linux/amd64",
            "previous_commit": commit,
            "status": "planned",
            "tasks": {},
        }
        # Every selected task must validate before the first registry mutation.
        for problem in problems:
            manifest["tasks"][problem] = incremental.plan_problem(
                REPOSITORY_ROOT, problem, options.image_prefix, commit, directory
            )
        options.manifest.parent.mkdir(parents=True, exist_ok=True)
        options.manifest.write_text(renderer.json_text(manifest), encoding="utf-8")
        for problem in problems:
            if options.push:
                incremental.publish_plan(
                    REPOSITORY_ROOT, problem, manifest["tasks"][problem], options.crane
                )
                options.manifest.write_text(
                    renderer.json_text(manifest), encoding="utf-8"
                )
            print("pushed" if options.push else "planned", problem)
        manifest["status"] = "published" if options.push else "planned"
        options.manifest.write_text(renderer.json_text(manifest), encoding="utf-8")
        return 0
    manifest = {
        "schema_version": "frontier.preview-native-build/v1",
        "platform": "linux/amd64",
        "bases": {},
        "tasks": {},
    }
    for problem in problems:
        manifest["tasks"][problem] = build_problem(
            problem, options.image_prefix, push=options.push, bases=manifest["bases"]
        )
        options.manifest.parent.mkdir(parents=True, exist_ok=True)
        options.manifest.write_text(renderer.json_text(manifest), encoding="utf-8")
        print(("pushed" if options.push else "built"), problem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
