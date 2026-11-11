"""Registry resolution, immutable image builds and mirroring."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request

import render_task_definition as renderer
import task_layers
import validate_task_layout as layout
from task_selection import TaskSelection
from task_selection import requirements as requirements
from task_selection import select_tasks as select_tasks

from .infra import (
    DIGEST_RE,
    REGISTRY_SERVICE,
    REPOSITORY_ROOT,
    TASK_PLACEHOLDER_DIGEST,
    DeployError,
    atomic_write,
    canonical_json,
    executable,
    load_json,
    port_forward,
    require_mapping,
    run_command,
    runtime_source_commit,
    sha256_bytes,
    source_image_tag_matches,
    state_directory,
    validate_runtime_images,
)


def source_tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        digest.update(str(item.relative_to(path)).encode() + b"\0")
        digest.update(item.read_bytes())
    return "sha256:" + digest.hexdigest()


def validate_application_source(config: Mapping[str, Any], path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise DeployError(
            f"FrontierCS2 application source must be a real directory: {path}"
        )
    resolved = path.resolve(strict=True)
    git = executable("git")
    head = run_command(
        [git, "-C", str(resolved), "rev-parse", "HEAD"], capture=True, timeout=60
    ).strip()
    expected = config["application"]["commit"]
    if head != expected:
        raise DeployError(
            f"FrontierCS2 source HEAD differs: expected {expected}, observed {head}"
        )
    status = run_command(
        [git, "-C", str(resolved), "status", "--porcelain", "--untracked-files=all"],
        capture=True,
        timeout=60,
    )
    if status:
        raise DeployError(
            "FrontierCS2 application source must be clean before image build"
        )
    for relative in (
        "management-plane/Dockerfile",
        "driver/Dockerfile",
        "skills/add-frontier-task/SKILL.md",
        "skills/add-frontier-task/references/task-protocol.md",
        "skills/add-frontier-task/references/release-and-publish.md",
        "skills/add-frontier-task/scripts/publish_task.py",
    ):
        item = resolved / relative
        if item.is_symlink() or not item.is_file():
            raise DeployError(f"FrontierCS2 source is missing {relative}")
    return resolved


def ensure_application_source(config: Mapping[str, Any]) -> Path | None:
    if not config["application"]["require_source"]:
        print(
            "Task authoring handoff: UNAVAILABLE (application source is disabled for this site profile)"
        )
        return None
    override = os.environ.get("FRONTIERCS2_SOURCE_DIR")
    if override:
        return validate_application_source(config, Path(override))

    destination = state_directory(config) / "application-source"
    if destination.exists():
        return validate_application_source(config, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            executable("git"),
            "clone",
            "--no-checkout",
            "--filter=blob:none",
            config["application"]["repository"],
            str(destination),
        ],
        timeout=1800,
    )
    run_command(
        [
            executable("git"),
            "-C",
            str(destination),
            "fetch",
            "--depth=1",
            "origin",
            config["application"]["commit"],
        ],
        timeout=1800,
    )
    run_command(
        [
            executable("git"),
            "-C",
            str(destination),
            "checkout",
            "--detach",
            config["application"]["commit"],
        ],
        timeout=300,
    )
    return validate_application_source(config, destination)


def linux_amd64_children(
    manifest: Mapping[str, Any], *, null_platform_is_empty: bool
) -> list[dict[str, Any]]:
    """Share platform selection; each caller retains its existing trust checks.

    Tag readback historically treats null platform metadata as an empty object;
    cache verification retains its stricter object requirement.
    """

    def platform(child: Mapping[str, Any]) -> Mapping[str, Any]:
        value = child.get("platform", {})
        return (value or {}) if null_platform_is_empty else value

    return [
        child
        for child in manifest.get("manifests", [])
        if platform(child).get("os") == "linux"
        and platform(child).get("architecture") == "amd64"
    ]


def registry_digest(port: int, repository: str, tag: str) -> str:
    # Resolve the pushed tag to a *plain image manifest* digest.  Images built
    # by CI without --provenance=false arrive as OCI indexes (linux/amd64 plus
    # a buildx attestation child); the Driver's composition path and the
    # dataset seed job both pin and fetch plain manifests, so an index digest
    # must be narrowed to its linux/amd64 child (whose manifest and blobs the
    # docker push already uploaded).
    call = urllib_request.Request(
        f"http://127.0.0.1:{port}/v2/{repository}/manifests/{tag}",
        headers={
            "Accept": "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json"
        },
    )
    try:
        with urllib_request.urlopen(call, timeout=30) as response:
            digest = response.headers.get("Docker-Content-Digest", "")
            manifest = json.loads(response.read())
    except (urllib_error.URLError, ValueError) as error:
        raise DeployError(
            f"registry digest readback failed for {repository}:{tag}"
        ) from error
    media_type = manifest.get("mediaType", "")
    if "index" in media_type or "manifest.list" in media_type:
        children = linux_amd64_children(manifest, null_platform_is_empty=True)
        if len(children) != 1:
            raise DeployError(
                f"registry index for {repository}:{tag} has {len(children)} linux/amd64 children; expected exactly 1"
            )
        digest = children[0].get("digest", "")
    if not DIGEST_RE.fullmatch(digest):
        raise DeployError(
            f"registry returned an invalid digest for {repository}:{tag}: {digest}"
        )
    return digest


def build_runtime_images(
    config: Mapping[str, Any], kubeconfig: Path
) -> dict[str, dict[str, str]]:
    docker = executable("docker")
    application_source = ensure_application_source(config)
    if application_source is not None:
        print(
            "Task authoring skill: "
            f"{application_source / 'skills/add-frontier-task/SKILL.md'}"
        )
    specifications = {
        "storage_controller": {
            "context": REPOSITORY_ROOT / "runtime/storage-controller"
        },
        "agent_base": {"context": REPOSITORY_ROOT / "smoke/speed-add/agent-base"},
        "candidate_overlay": {"context": REPOSITORY_ROOT / "smoke/speed-add/candidate"},
        "verifier": {"context": REPOSITORY_ROOT / "smoke/speed-add/verifier"},
    }
    mirrors = {
        "management": config["images"]["management"],
        "driver": config["images"]["driver"],
        "litellm": config["images"]["litellm"],
        "postgres": config["images"]["postgres"],
    }
    endpoint = f"{config['network']['registry_cluster_ip']}:5000"
    records: dict[str, dict[str, str]] = {}
    with port_forward(kubeconfig, REGISTRY_SERVICE, 5000) as port:
        for name, specification in specifications.items():
            context = specification["context"]
            source_sha = source_tree_sha256(context)
            repository = f"frontier/{name.replace('_', '-')}"
            tag = "src-" + source_sha.removeprefix("sha256:")[:20]
            local_tag = f"127.0.0.1:{port}/{repository}:{tag}"
            # --provenance=false keeps BuildKit from wrapping the single-arch
            # image in an OCI index; the Driver's composition path pins and
            # fetches plain image manifests.
            command = [
                docker,
                "build",
                "--provenance=false",
                "--platform",
                "linux/amd64",
                "--pull",
            ]
            command.extend(["--tag", local_tag, str(context)])
            run_command(command, timeout=7200)
            run_command([docker, "push", local_tag], timeout=1800)
            digest = registry_digest(port, repository, tag)
            cluster_tag = f"{endpoint}/{repository}:{tag}"
            records[name] = {
                "tag": cluster_tag,
                "digest": digest,
                "reference": f"{endpoint}/{repository}@{digest}",
                "source_sha256": source_sha,
            }
        for name, source in mirrors.items():
            source_reference = source["reference"]
            source_digest = source["digest"]
            repository = f"frontier/{name}"
            tag = (
                "git-" + runtime_source_commit(config, name)[:20]
                if name in {"management", "driver"}
                else "upstream-" + source_digest.removeprefix("sha256:")[:20]
            )
            local_tag = f"127.0.0.1:{port}/{repository}:{tag}"
            run_command(
                [docker, "pull", "--platform", "linux/amd64", source_reference],
                timeout=1800,
            )
            run_command([docker, "tag", source_reference, local_tag], timeout=120)
            run_command([docker, "push", local_tag], timeout=1800)
            digest = registry_digest(port, repository, tag)
            cluster_tag = f"{endpoint}/{repository}:{tag}"
            records[name] = {
                "tag": cluster_tag,
                "digest": digest,
                "reference": f"{endpoint}/{repository}@{digest}",
                "source_sha256": source_digest,
            }
    records = validate_runtime_images(records, config)
    path = state_directory(config) / "runtime-images.json"
    atomic_write(path, canonical_json(records), mode=0o600)
    return records


def load_runtime_images(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    path = state_directory(config) / "runtime-images.json"
    return validate_runtime_images(load_json(path), config)


def cached_registry_image(port: int, repository: str, source_digest: str) -> str | None:
    """Reuse exact already-mirrored content without unpacking it on the operator.

    Verify source/index and leaf bytes, plus every runtime blob's size/digest
    metadata. Only a missing source manifest is a cache miss; corruption fails.
    """
    base = f"http://127.0.0.1:{port}/v2/{repository}"
    accept = ", ".join(
        (
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.v2+json",
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
        )
    )

    def manifest(digest: str, *, missing_ok: bool = False) -> dict[str, Any] | None:
        call = urllib_request.Request(
            f"{base}/manifests/{digest}", headers={"Accept": accept}
        )
        try:
            with urllib_request.urlopen(call, timeout=30) as response:
                body = response.read(4 * 1024 * 1024 + 1)
                observed = response.headers.get("Docker-Content-Digest")
        except urllib_error.HTTPError as error:
            error.close()
            if missing_ok and error.code == 404:
                return None
            raise DeployError("cached registry manifest is unavailable") from error
        if (
            len(body) > 4 * 1024 * 1024
            or sha256_bytes(body) != digest
            or observed != digest
        ):
            raise DeployError("cached registry manifest identity differs")
        return require_mapping(json.loads(body), "cached manifest")

    current = manifest(source_digest, missing_ok=True)
    if current is None:
        return None
    digest = source_digest
    if current.get("mediaType") in accept.split(", ")[2:]:
        children = linux_amd64_children(current, null_platform_is_empty=False)
        if len(children) != 1 or not DIGEST_RE.fullmatch(children[0].get("digest", "")):
            raise DeployError("cached registry index lacks one linux/amd64 child")
        digest = children[0]["digest"]
        current = manifest(digest)
    if current.get("mediaType") not in accept.split(", ")[:2]:
        raise DeployError("cached registry image is not a plain manifest")
    blobs = [current.get("config", {}), *current.get("layers", [])]
    if len(blobs) < 2 or len(blobs) > 1024:
        raise DeployError("cached registry image has invalid blob count")
    for blob in blobs:
        if (
            not DIGEST_RE.fullmatch(blob.get("digest", ""))
            or type(blob.get("size")) is not int
            or blob["size"] <= 0
        ):
            raise DeployError("cached registry blob descriptor is invalid")
        call = urllib_request.Request(f"{base}/blobs/{blob['digest']}", method="HEAD")
        with urllib_request.urlopen(call, timeout=30) as response:
            if response.headers.get("Docker-Content-Digest") != blob[
                "digest"
            ] or response.headers.get("Content-Length") != str(blob["size"]):
                raise DeployError("cached registry blob identity or size differs")
    return digest


def mirror_task_images(
    config: Mapping[str, Any],
    kubeconfig: Path,
    catalog: Mapping[str, Any],
    *,
    selection: TaskSelection,
) -> dict[str, dict[str, dict[str, str]]]:
    references = {
        task["name"]: task["reference"]
        for task in catalog["tasks"]
        if task["reference"] is not None
    }
    if any(image["digest"] == TASK_PLACEHOLDER_DIGEST for image in references.values()):
        raise ValueError("reference image digest is a placeholder")
    docker = executable("docker")
    endpoint = f"{config['network']['registry_cluster_ip']}:5000"
    records: dict[str, dict[str, dict[str, str]]] = {}
    with port_forward(kubeconfig, REGISTRY_SERVICE, 5000) as port:

        def mirror(
            source: Mapping[str, str], repository: str, *, plain_manifest: bool = False
        ) -> dict[str, str]:
            tag = "source-" + source["digest"].removeprefix("sha256:")[:20]
            source_reference = source["reference"]
            cached = cached_registry_image(port, repository, source["digest"])
            if cached is not None:
                if plain_manifest and cached != source["digest"]:
                    raise DeployError(
                        "Candidate overlay must pin the linux/amd64 manifest, not its index"
                    )
                print(f"verified registry reuse: {repository}@{cached}", flush=True)
                return {
                    "tag": f"{endpoint}/{repository}:{tag}",
                    "digest": cached,
                    "reference": f"{endpoint}/{repository}@{cached}",
                    "source_reference": source_reference,
                    "source_sha256": source["digest"],
                }
            local_tag = f"127.0.0.1:{port}/{repository}:{tag}"
            copy_tool = os.environ.get("FRONTIER_REGISTRY_COPY_TOOL")
            if copy_tool:
                # Stream registry blobs when the operator cannot unpack the
                # complete native task images on its local disk.
                run_command(
                    [
                        executable(copy_tool),
                        "copy",
                        "--platform",
                        "linux/amd64",
                        "--jobs",
                        "2",
                        source_reference,
                        local_tag,
                    ],
                    timeout=3600,
                )
            else:
                run_command(
                    [docker, "pull", "--platform", "linux/amd64", source_reference],
                    timeout=3600,
                )
                run_command([docker, "tag", source_reference, local_tag], timeout=120)
                run_command([docker, "push", local_tag], timeout=3600)
            digest = registry_digest(port, repository, tag)
            if plain_manifest and digest != source["digest"]:
                raise DeployError(
                    "Candidate overlay must pin the linux/amd64 manifest, not its index"
                )
            if (
                copy_tool
                and cached_registry_image(port, repository, source["digest"]) != digest
            ):
                raise DeployError(
                    "registry copy did not preserve the pinned source manifest"
                )
            cluster_tag = f"{endpoint}/{repository}:{tag}"
            return {
                "tag": cluster_tag,
                "digest": digest,
                "reference": f"{endpoint}/{repository}@{digest}",
                "source_reference": source_reference,
                "source_sha256": source["digest"],
            }

        records["candidate"] = {
            "overlay_image": mirror(
                catalog["candidate"]["overlay_image"],
                "frontier/candidates/codex",
                plain_manifest=True,
            ),
        }
        records["reference_candidates"] = {
            name: mirror(
                {**source, "reference": source["repository"] + "@" + source["digest"]},
                f"frontier/candidates/reference-{name.replace('_', '-')}",
            )
            for name, source in references.items()
            if name in selection.reference_names
        }
        for task in catalog["tasks"]:
            problem = task["name"]
            if problem not in selection.selected:
                continue
            records[problem] = {}
            # The CPU runner images of command test sandboxes are cluster images as well: their
            # Pods pull them from this registry (the Modal CPU backend imports the public tag).
            # Every task has its agent, verifier and dataset images; only the runner images are
            # per task.
            runners = [
                name
                for name in ("agent_cpu_runner_image", "verifier_cpu_runner_image")
                if name in task
            ]
            for image_name in (
                "agent_image",
                "verifier_image",
                "dataset_image",
                *runners,
            ):
                source = task[image_name]
                kind = image_name.removesuffix("_image").replace("_", "-")
                repository = f"frontier/tasks/{problem.replace('_', '-')}/{kind}"
                records[problem][image_name] = mirror(source, repository)
            # Modal pulls public GPU images directly. Never copy model-scale
            # layers into the CPU node or the private in-cluster registry.
            for image_name in ("agent_gpu_image", "verifier_gpu_image"):
                if image_name in task:
                    source = task[image_name]
                    records[problem][image_name] = {
                        **source,
                        "tag": f"{source['repository']}:{source['tag']}",
                    }
            if "sidecar_images" in task:
                # Declared sidecars are public Modal-side images as well.
                records[problem]["sidecar_images"] = {
                    sidecar: {
                        **source,
                        "tag": f"{source['repository']}:{source['tag']}",
                    }
                    for sidecar, source in task["sidecar_images"].items()
                }
    atomic_write(
        state_directory(config) / "task-images.json",
        canonical_json(records),
        mode=0o600,
    )
    return records


# Optional source-only layers for the task builder compatibility flag.
class IncrementalFailure(RuntimeError):
    pass


ROLES = {
    "agent": ("agent-base", "agent_image"),
    "verifier": ("verifier", "verifier_image"),
    "agent-gpu": ("agent-gpu", "agent_gpu_image"),
    "verifier-gpu": ("verifier-gpu", "verifier_gpu_image"),
}
ALLOWED = {
    "agent-base/frontier/task/README.md",
    "agent-base/frontier/task/repo/README.md",
    "agent-base/frontier/task/bin/README.md",
    "judge/evaluator.py",
    "judge/beat_spec.json",
    "judge/config.yaml",
    "judge/readme",
    "judge/DESIGN.md",
    "judge/NOTES.md",
    "judge/SUMMARY.md",
    "verifier/beat_spec_feedback.py",
    "policy.json",
    "upstream.lock.json",
    "task.toml",
    "README.md",
}
LFS_POINTER = re.compile(
    rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?"
)
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
MAX_LAYER_BYTES = 64 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IncrementalFailure(message)


def command(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    require(
        result.returncode == 0, f"{args[0]} failed: {result.stderr[-2000:].strip()}"
    )
    return result.stdout.strip()


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def safe_relative(name: str) -> str:
    path = PurePosixPath(name)
    require(
        bool(name)
        and not path.is_absolute()
        and all(p not in {".", "..", ""} for p in name.split("/")),
        f"unsafe archive path: {name!r}",
    )
    return name


def resolve_commit(repository: Path, revision: str) -> str:
    require(
        re.fullmatch(r"[0-9a-fA-F]{7,40}", revision) is not None,
        "--incremental-from must be a hexadecimal Git commit",
    )
    commit = command(
        ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=repository
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
        "cannot resolve previous commit",
    )
    return commit


def git_json(repository: Path, commit: str, path: str) -> dict:
    value = json.loads(command(["git", "show", f"{commit}:{path}"], cwd=repository))
    require(isinstance(value, dict), f"previous {path} must be a JSON object")
    return value


def previous_layers(repository: Path, commit: str) -> list[tuple[str, dict]]:
    """The task layers `commit` declared: (directory, layer.json body) pairs."""
    listed = command(["git", "ls-tree", "--name-only", commit], cwd=repository).split()
    return [
        (name, git_json(repository, commit, f"{name}/{task_layers.DECLARATION}"))
        for name in listed
        if command(
            [
                "git",
                "ls-tree",
                "--name-only",
                commit,
                f"{name}/{task_layers.DECLARATION}",
            ],
            cwd=repository,
        ).strip()
    ]


def previous_rows(repository: Path, commit: str, problem: str) -> tuple[dict, dict]:
    """The definition and catalog row of `problem` at `commit`, from whichever task layer
    listed it then (task_layers, read from that commit's tree)."""
    files = [("config/task-definitions.json", "config/task-catalog.json")]
    for name, body in previous_layers(repository, commit):
        files.append((f"{name}/{body['definitions']}", f"{name}/{body['catalog']}"))
    for definitions_path, catalog_path in files:
        definitions = git_json(repository, commit, definitions_path)
        if problem in definitions:
            catalog = git_json(repository, commit, catalog_path)
            task = next(task for task in catalog["tasks"] if task["name"] == problem)
            return definitions[problem], task
    raise DeployError(f"previous commit {commit} has no definition of {problem}")


def materialize_lfs(
    pointer: bytes, target: Path, current: Path, lfs_objects: Path
) -> None:
    match = LFS_POINTER.fullmatch(pointer)
    require(match is not None, "unsupported Git LFS pointer format")
    digest, size = match.group(1).decode(), int(match.group(2))
    candidates = [current, lfs_objects / digest[:2] / digest[2:4] / digest]
    source = next(
        (
            path
            for path in candidates
            if path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == size
            and sha(path) == "sha256:" + digest
        ),
        None,
    )
    require(
        source is not None,
        f"previous LFS object is missing or corrupt: sha256:{digest}",
    )
    shutil.copyfile(source, target)


def restore_previous_tree(
    repository: Path, commit: str, relative_root: str, target: Path
) -> None:
    """Stream a Git archive, resolving LFS from verified local bytes only."""
    safe_relative(relative_root)
    lfs_path = Path(
        command(["git", "rev-parse", "--git-path", "lfs/objects"], cwd=repository)
    )
    if not lfs_path.is_absolute():
        lfs_path = repository / lfs_path
    process = subprocess.Popen(
        ["git", "archive", "--format=tar", commit, relative_root],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(
        process.stdout is not None and process.stderr is not None,
        "cannot stream previous tree",
    )
    target.mkdir(parents=True)
    count = 0
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                name = safe_relative(member.name.rstrip("/"))
                if member.isdir() and (
                    name == relative_root or relative_root.startswith(name + "/")
                ):
                    continue
                require(
                    name.startswith(relative_root + "/"),
                    "Git archive escaped task root",
                )
                relative = safe_relative(name[len(relative_root) + 1 :])
                destination = target / relative
                require(
                    member.isdir() or member.isfile(),
                    "previous source contains a non-regular member",
                )
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                require(
                    not destination.exists(), "duplicate previous source archive member"
                )
                stream = archive.extractfile(member)
                require(stream is not None, "cannot read previous archive member")
                first = stream.read(min(1024, member.size))
                if first.startswith(b"version https://git-lfs.github.com/spec/"):
                    require(member.size <= 1024, "oversized Git LFS pointer")
                    materialize_lfs(first, destination, repository / name, lfs_path)
                else:
                    with destination.open("wb") as output:
                        output.write(first)
                        shutil.copyfileobj(stream, output, 1024 * 1024)
                destination.chmod(0o755 if member.mode & 0o111 else 0o644)
                count += 1
        process.stdout.close()
        error = process.stderr.read().decode(errors="replace")
        require(process.wait() == 0, f"Git archive failed: {error[-2000:]}")
        require(count > 0, "previous task tree is empty")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def changed_paths(previous: Path, current: Path) -> list[str]:
    layout.check_inventory(previous)
    renderer.check_tree(current)
    old = {
        p.relative_to(previous).as_posix(): p
        for p in previous.rglob("*")
        if p.is_file()
    }
    new = {
        p.relative_to(current).as_posix(): p for p in current.rglob("*") if p.is_file()
    }
    require(
        not set(old) - set(new),
        "incremental publication does not support removed task files",
    )
    changed = []
    for name, path in sorted(new.items()):
        if name in old and bool(path.stat().st_mode & 0o111) != bool(
            old[name].stat().st_mode & 0o111
        ):
            raise IncrementalFailure(
                f"incremental publication cannot change executable mode: {name}"
            )
        if (
            name in old
            and path.stat().st_size == old[name].stat().st_size
            and sha(path) == sha(old[name])
        ):
            continue
        supported = (
            name in ALLOWED
            or re.fullmatch(
                r"(?:judge/|tests/|judge/tests/)test_[a-zA-Z0-9_]+\.py", name
            )
            is not None
        )
        require(
            supported,
            f"unhandled incremental task change (use the normal source build): {name}",
        )
        changed.append(name)
    return changed


def mapped_files(root: Path, role: str, changed: list[str]) -> list[dict]:
    """Map supported changed inputs through the unchanged native COPY recipe."""
    recipe = (root / ROLES[role][0] / "Dockerfile").read_text()
    mappings = []
    for line in recipe.splitlines():
        if not line.startswith("COPY "):
            continue
        fields = shlex.split(line)[1:]
        if fields and fields[0].startswith("--from="):
            continue
        chmod = None
        if fields and re.fullmatch(r"--chmod=0?[0-7]{3}", fields[0]):
            chmod = fields.pop(0)
        require(
            len(fields) == 2 and not any(part.startswith("--") for part in fields),
            "unsupported native COPY syntax",
        )
        source, destination = fields
        source = layout.copy_source(root, source, role)
        require(
            destination.startswith("/")
            and ".." not in PurePosixPath(destination).parts,
            "unsupported native COPY destination",
        )
        mappings.append((source, destination.rstrip("/"), chmod))
    result = {}
    for name in changed:
        for source, destination, chmod in mappings:
            if name == source:
                target = destination
            elif name.startswith(source + "/") and (root / source).is_dir():
                target = destination + name[len(source) :]
            else:
                continue
            require(
                chmod is None,
                "incremental runtime replacement cannot override COPY --chmod",
            )
            if target.startswith("/frontier/task/"):
                mode = 0o444  # Only public README files are allowed above.
            elif target.startswith("/opt/frontier/task/"):
                mode = 0o555 if (root / name).stat().st_mode & 0o111 else 0o444
            else:
                require(
                    name == "verifier/beat_spec_feedback.py"
                    and target == "/opt/frontier/beat_spec_feedback.py",
                    f"unhandled flat runtime replacement: {name}",
                )
                mode = 0o644
            require(target not in result, f"duplicate layer target: {target}")
            result[target] = {
                "source_path": name,
                "image_path": target,
                "mode": mode,
                "uid": 0,
                "gid": 0,
                "bytes": (root / name).stat().st_size,
                "sha256": sha(root / name),
            }
    return [result[target] for target in sorted(result)]


def write_layer(root: Path, records: list[dict], path: Path) -> str:
    require(
        sum(record["bytes"] for record in records) <= MAX_LAYER_BYTES,
        "incremental source layer exceeds the 64 MiB source-only bound",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w", format=tarfile.GNU_FORMAT) as archive:
        for record in records:
            source = root / record["source_path"]
            require(
                sha(source) == record["sha256"],
                "source changed while constructing incremental layer",
            )
            member = tarfile.TarInfo(record["image_path"].lstrip("/"))
            member.size = record["bytes"]
            member.mode, member.uid, member.gid, member.mtime = record["mode"], 0, 0, 0
            with source.open("rb") as stream:
                archive.addfile(member, stream)
    return sha(path)


def previous_problem_path(repository: Path, commit: str, problem: str) -> str:
    """Find a task at the previous revision, in any of that revision's task layers, flat or
    one domain deep."""
    renderer.problem_directory(repository / task_layers.SHARED_PROBLEMS, problem)
    prefixes = ["tasks/problems/"] + [
        f"{name}/{body['problems']}/"
        for name, body in previous_layers(repository, commit)
    ]
    matches = []
    for prefix in prefixes:
        paths = command(
            ["git", "ls-tree", "-r", "--name-only", commit, "--", prefix],
            cwd=repository,
        )
        for value in paths.splitlines():
            parts = value[len(prefix) :].split("/")
            if (
                len(parts) in (2, 3)
                and parts[-2] == problem
                and parts[-1] in ("policy.json", "task.toml")
            ):
                matches.append(value.rsplit("/", 1)[0])
    require(
        len(matches) == 1, f"previous task must have one source directory: {problem}"
    )
    return matches[0]


def plan_problem(
    repository: Path, problem: str, prefix: str, commit: str, directory: Path
) -> dict:
    root = renderer.task_directory(problem, repository)
    relative = previous_problem_path(repository, commit, problem)
    initial_source = renderer.source_sha256(root)
    layout.validate_task(root)
    previous_definition, previous_task = previous_rows(repository, commit, problem)
    old_source = previous_definition["source"]
    require(
        old_source.get("identity_algorithm") == renderer.IDENTITY,
        "previous source identity is not supported",
    )
    with tempfile.TemporaryDirectory(prefix="frontier-previous-task-") as temporary:
        previous = Path(temporary) / problem
        restore_previous_tree(repository, commit, relative, previous)
        require(
            renderer.source_sha256(previous) == old_source["source_set_sha256"],
            f"previous Git/LFS source does not match published definition: {problem}",
        )
        changes = changed_paths(previous, root)
    definition = renderer.render(root, problem)
    require(
        definition["source"].get("identity_algorithm") == renderer.IDENTITY,
        "current source identity is not supported",
    )
    require(
        definition["source"]["source_set_sha256"] == initial_source,
        "task source changed during incremental planning",
    )
    require(
        definition["datasets"] == previous_definition["datasets"],
        "incremental publication cannot change datasets",
    )
    old_tag = "source-" + old_source["source_set_sha256"].removeprefix("sha256:")
    require(
        previous_definition["image_tag"] == old_tag,
        "previous source/tag interlock mismatch",
    )
    tag = definition["image_tag"]
    # CPU runner images are built FROM the task's whole image with the shared runner layer; a
    # source-only layer cannot carry them.
    require(
        not definition.get("cpu_runner_images"),
        "a task with CPU command sandboxes publishes by a full build (its runner images)",
    )
    plan = {
        "source": definition["source"],
        "datasets": definition["datasets"],
        "previous_commit": commit,
        "previous_source": old_source,
        "changed_paths": changes,
        "images": {},
        "components": {},
    }
    for role, (directory_name, catalog_key) in ROLES.items():
        if not (root / directory_name).is_dir():
            continue
        image = previous_task[catalog_key]
        require(
            image["repository"].startswith("docker.io/")
            and DIGEST.fullmatch(image["digest"]) is not None,
            "incremental base must be a public Docker Hub digest",
        )
        require(
            source_image_tag_matches(image["tag"], old_tag, image["digest"]),
            "previous catalog/source tag interlock mismatch",
        )
        stem = (
            f"fcs2-task-verifier-{problem}"
            if role.startswith("verifier")
            else f"fcs2-task-{problem}-agent"
        )
        if role.endswith("-gpu"):
            stem += "-gpu"
        repository_name = "docker.io/" + prefix.removeprefix("docker.io/") + "/" + stem
        require(
            repository_name == image["repository"],
            "incremental publication must reuse the previous public repository",
        )
        target = repository_name + ":" + tag
        records = mapped_files(root, role, changes)
        component = {
            "base": image["repository"] + "@" + image["digest"],
            "target": target,
            "files": records,
            "layer_path": None,
            "layer_diff_id": None,
        }
        if records:
            layer = directory / problem / f"{role}.tar"
            component["layer_path"] = str(layer.resolve())
            component["layer_diff_id"] = write_layer(root, records, layer)
        plan["images"][role] = target
        plan["components"][role] = component
    dataset = previous_task["dataset_image"]
    require(
        dataset["repository"].startswith("docker.io/")
        and DIGEST.fullmatch(dataset["digest"]) is not None,
        "previous dataset must be pinned to a public Docker Hub digest",
    )
    require(
        dataset["tag"] == "v1"
        or source_image_tag_matches(dataset["tag"], old_tag, dataset["digest"]),
        "unexpected dataset version",
    )
    plan["dataset_image"] = dict(dataset)
    if dataset["tag"] != "v1":
        plan["dataset_image"]["tag"] = tag
    plan["images"]["datasets"] = (
        dataset["repository"] + ":" + plan["dataset_image"]["tag"]
    )
    require(
        renderer.source_sha256(root) == initial_source,
        "task source changed while staging incremental layers",
    )
    return plan


def image_json(crane: str, operation: str, image: str, repository: Path) -> dict:
    result = json.loads(
        command([crane, operation, "--platform", "linux/amd64", image], cwd=repository)
    )
    require(isinstance(result, dict), f"invalid image {operation}")
    return result


def check_appended_image(
    base_manifest: dict,
    base_config: dict,
    manifest: dict,
    config: dict,
    diff_id: str | None,
) -> None:
    require(
        "manifests" not in base_manifest
        and isinstance(base_manifest.get("layers"), list),
        "incremental base must be a single-platform image",
    )
    require(
        base_config.get("os") == "linux" and base_config.get("architecture") == "amd64",
        "base platform mismatch",
    )
    extra = 1 if diff_id is not None else 0
    require(
        len(manifest.get("layers", [])) == len(base_manifest["layers"]) + extra
        and manifest["layers"][: len(base_manifest["layers"])]
        == base_manifest["layers"],
        "published image does not preserve the exact base layer chain",
    )
    require(
        config.get("rootfs", {}).get("diff_ids")
        == base_config.get("rootfs", {}).get("diff_ids", [])
        + ([diff_id] if extra else []),
        "published image does not append the verified source layer",
    )
    require(
        {
            key: value
            for key, value in config.get("rootfs", {}).items()
            if key != "diff_ids"
        }
        == {
            key: value
            for key, value in base_config.get("rootfs", {}).items()
            if key != "diff_ids"
        },
        "incremental publication changed root filesystem configuration",
    )
    require(
        {
            key: value
            for key, value in config.items()
            if key not in {"history", "rootfs"}
        }
        == {
            key: value
            for key, value in base_config.items()
            if key not in {"history", "rootfs"}
        },
        "incremental publication changed image configuration",
    )
    old_history = base_config.get("history", [])
    require(
        config.get("history", [])[: len(old_history)] == old_history,
        "incremental publication rewrote base image history",
    )


def publish_plan(repository: Path, problem: str, plan: dict, crane: str) -> None:
    root = renderer.task_directory(problem, repository)
    require(
        renderer.source_sha256(root) == plan["source"]["source_set_sha256"],
        "task source changed after incremental planning",
    )
    dataset = plan["dataset_image"]
    if dataset["tag"] != "v1":
        from import_task import remote_digest

        existing = remote_digest(dataset, single_layer=True)
        require(
            existing is None or existing == dataset["digest"],
            "immutable dataset tag collision",
        )
    for component in plan["components"].values():
        base, target = component["base"], component["target"]
        base_manifest = image_json(crane, "manifest", base, repository)
        base_config = image_json(crane, "config", base, repository)
        require(
            "manifests" not in base_manifest
            and base_config.get("os") == "linux"
            and base_config.get("architecture") == "amd64",
            "base must be a single linux/amd64 image",
        )
        if component["layer_path"]:
            require(
                sha(Path(component["layer_path"])) == component["layer_diff_id"],
                "incremental layer changed after planning",
            )
            command(
                [
                    crane,
                    "append",
                    "--platform",
                    "linux/amd64",
                    "--base",
                    base,
                    "--new_layer",
                    component["layer_path"],
                    "--new_tag",
                    target,
                ],
                cwd=repository,
            )
        else:
            command([crane, "tag", base, target.rsplit(":", 1)[1]], cwd=repository)
        digest = command(
            [crane, "digest", "--platform", "linux/amd64", target], cwd=repository
        )
        require(
            DIGEST.fullmatch(digest) is not None,
            "published image has no immutable digest",
        )
        immutable = target.rsplit(":", 1)[0] + "@" + digest
        manifest = image_json(crane, "manifest", immutable, repository)
        config = image_json(crane, "config", immutable, repository)
        check_appended_image(
            base_manifest, base_config, manifest, config, component["layer_diff_id"]
        )
        component["published_digest"] = digest
        component["published_layer"] = (
            manifest["layers"][-1] if component["layer_path"] else None
        )
    if dataset["tag"] != "v1":
        # Recheck after component publication in case another publisher raced us.
        existing = remote_digest(dataset, single_layer=True)
        require(
            existing is None or existing == dataset["digest"],
            "immutable dataset tag collision",
        )
        if existing is None:
            command(
                [
                    crane,
                    "tag",
                    dataset["repository"] + "@" + dataset["digest"],
                    dataset["tag"],
                ],
                cwd=repository,
            )
        require(
            remote_digest(dataset, single_layer=True) == dataset["digest"],
            "dataset tag readback mismatch",
        )
    require(
        renderer.source_sha256(root) == plan["source"]["source_set_sha256"],
        "task source changed during incremental publication",
    )
