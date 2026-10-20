#!/usr/bin/env python3
"""Import a native FrontierSmith export into Preview's publication catalog."""

from __future__ import annotations

import os
import sys

# --plan/--check and rendering existing trees must not create bytecode caches.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import copy
import json
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

try:
    import jsonschema
except ImportError:
    print(
        "install jsonschema (pip install jsonschema) to import tasks", file=sys.stderr
    )
    raise SystemExit(2) from None
from urllib.error import HTTPError
from urllib.parse import quote

import build_task_images as builder
import frontier_deploy as deploy
import render_task_definition as renderer
import resolve_catalog_digests as resolver
from native_datasets import dataset_extension, dataset_path
from preview import publication
from preview.infra import atomic_write
from preview.measured_baseline import baseline_from_measurement
from preview.publication import baseline_from_result_json, file_sha
from task_selection import requirements

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = Path("tasks/frontier-cs-2.0-demo/problems")
LAUNCHER = Path(
    "tasks/frontier-cs-2.0-demo/shared/reference-candidate/fcs2_reference_candidate.py"
)
MAX_FILE_BYTES = 512 * 1024 * 1024
PLACEHOLDER = deploy.TASK_PLACEHOLDER_DIGEST
# Docker Hub namespace for built/published task images. The default is the
# namespace that hosts the published, digest-pinned release images.
DEFAULT_IMAGE_NAMESPACE = os.environ.get("FRONTIER_IMAGE_NAMESPACE", "joyemang33")
ROLE_KEYS = {
    "agent": "agent_image",
    "verifier": "verifier_image",
    "datasets": "dataset_image",
    "agent_gpu": "agent_gpu_image",
    "verifier_gpu": "verifier_gpu_image",
    "reference": "reference",
}


def safe_member(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
    ):
        raise ValueError("unsafe relative member path")
    return value


def lfs_tracked(root: Path, path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "check-attr", "-z", "filter", "--", str(path)],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.split(b"\0")[-2:] == [b"lfs", b""]


def validate_export(root: Path) -> dict:
    # Check before resolve(): resolving would hide a symlink supplied as the root.
    renderer.check_tree(root)
    if not root.is_dir() or re.fullmatch(r"[a-z][a-z0-9_]*", root.name) is None:
        raise ValueError("export directory must have a valid problem name")
    for member in (
        "policy.json",
        "upstream.lock.json",
        "agent-base/Dockerfile",
        "verifier/Dockerfile",
        "verifier/verify.py",
    ):
        if not (root / member).is_file():
            raise ValueError(f"missing required export member: {member}")
    for path in root.rglob("*"):
        safe_member(path.relative_to(root).as_posix())
        if path.is_dir() and not any(path.iterdir()):
            raise ValueError(
                "empty directory cannot survive Git checkout: "
                + path.relative_to(root).as_posix()
            )
        if path.is_file():
            with path.open("rb") as stream:
                if stream.read(128).startswith(
                    b"version https://git-lfs.github.com/spec/v1"
                ):
                    raise ValueError(
                        f"unresolved LFS pointer: {path.relative_to(root)}"
                    )
            if path.stat().st_size > MAX_FILE_BYTES and not lfs_tracked(root, path):
                raise ValueError(
                    f"file exceeds 512 MiB and is not LFS-tracked: {path.relative_to(root)}"
                )
    policy = deploy.require_mapping(deploy.load_json(root / "policy.json"), "policy")
    deploy.reject_secret_fields(policy, "policy")
    if (
        policy.get("schema_version") != renderer.SCHEMA
        or policy.get("problem_id") != root.name
    ):
        raise ValueError(
            "policy schema or problem_id differs from export directory name"
        )
    lock = deploy.require_mapping(
        deploy.load_json(root / "upstream.lock.json"), "upstream.lock"
    )
    for row in lock.get("files", []):
        safe_member(row.get("path"))
    if lock.get("problem_id", root.name) != root.name:
        raise ValueError(
            "upstream.lock.json problem_id differs from export directory name"
        )
    for row in policy.get("metadata", {}).get("datasets", {}).values():
        if isinstance(row, dict) and "path" in row:
            safe_member(row["path"])
    safe_member(policy["submission"]["entrypoint"])
    requirements(policy)
    for role, quota in (("agent", "max_instances"), ("verifier", "max_gpu_instances")):
        if (
            policy.get("gpu", {}).get(role, {}).get(quota, 0) > 0
            and not (root / f"{role}-gpu/Dockerfile").is_file()
        ):
            raise ValueError(f"missing enabled GPU Dockerfile: {role}-gpu/Dockerfile")
    # Apply the same bounded base policy in plan/published modes too; no pulls.
    for dockerfile in (
        root / role / "Dockerfile"
        for role in ("agent-base", "verifier", "agent-gpu", "verifier-gpu")
    ):
        if not dockerfile.is_file():
            continue
        builder.dockerfile_bases(dockerfile)
    # Renderer checks the required semantic fields and exact dataset hashes.
    return renderer.render(root, root.name)


def entrypoint_member(definition: dict) -> str:
    entrypoint = safe_member(definition["submission"]["entrypoint"])
    if not entrypoint.startswith("payload/"):
        raise ValueError("reference submission entrypoint must be under payload/")
    member = entrypoint.removeprefix("payload/")
    # Dockerfile COPY destinations must be literal, without interpolation.
    if re.fullmatch(r"[a-zA-Z0-9_./-]+", member) is None:
        raise ValueError("unsupported reference entrypoint member")
    return member


def image_roles(definition: dict, reference: bool) -> list[str]:
    roles = ["agent", "verifier", "datasets"]
    gpu = definition.get("gpu", {})
    if gpu.get("agent", {}).get("max_instances", 0):
        roles.append("agent_gpu")
    if gpu.get("verifier", {}).get("max_gpu_instances", 0):
        roles.append("verifier_gpu")
    if reference:
        roles.append("reference")
    return roles


def catalog_entry(
    root: Path, definition: dict, namespace: str, payload: Path | None
) -> dict:
    sidecars = builder.sidecar_roles(definition)
    names = builder.image_names(
        "docker.io/" + namespace, root.name, definition["image_tag"], sidecars
    )
    names["reference"] = (
        f"docker.io/{namespace}/fcs2-task-{root.name}-reference:{definition['image_tag']}"
    )
    entry = {
        "name": root.name,
        "datasets": copy.deepcopy(definition["datasets"]),
        "reference": None,
        "baseline": baseline_from_result_json(root, definition, payload),
    }
    for role in image_roles(definition, payload is not None):
        repository, tag = names[role.replace("_", "-")].rsplit(":", 1)
        image = {"repository": repository, "tag": tag, "digest": PLACEHOLDER}
        if role == "reference":
            image.update(
                launcher_sha256=file_sha(REPOSITORY_ROOT / LAUNCHER),
                payload_sha256=file_sha(payload),
            )
        entry[ROLE_KEYS[role]] = image
    if sidecars:
        entry["sidecar_images"] = {}
        for sidecar in sidecars:
            repository, tag = names[f"sidecar-{sidecar}"].rsplit(":", 1)
            entry["sidecar_images"][sidecar] = {
                "repository": repository,
                "tag": tag,
                "digest": PLACEHOLDER,
            }
    return entry


def published_manifest(path: Path, name: str, source: str, entry: dict) -> dict:
    raw = deploy.load_json(path)
    deploy.require_exact_keys(
        raw,
        {"schema_version", "problem_id", "source_sha256", "images"},
        "published images",
    )
    if (
        raw["schema_version"] != "frontier.task-release-images/v1"
        or raw["problem_id"] != name
    ):
        raise ValueError("published-images schema or problem_id differs")
    if raw["source_sha256"] != source:
        raise ValueError(
            "published-images source_sha256 differs from preview-source-identity/v1"
        )
    expected = {role for role, key in ROLE_KEYS.items() if entry.get(key) is not None}
    sidecar_roles = {
        f"sidecar-{sidecar}": sidecar for sidecar in entry.get("sidecar_images", {})
    }
    deploy.require_exact_keys(
        deploy.require_mapping(raw["images"], "images"),
        expected | set(sidecar_roles),
        "published image roles",
    )
    result = copy.deepcopy(entry)
    for role in sorted(expected | set(sidecar_roles)):
        image = deploy.require_mapping(raw["images"][role], f"images.{role}")
        if role in sidecar_roles:
            current = entry["sidecar_images"][sidecar_roles[role]]
            target = result["sidecar_images"][sidecar_roles[role]]
        else:
            current = target = None
        tag = (current or entry[ROLE_KEYS[role]])["tag"]
        # Changed selectors get a new source-bound tag; legacy v1 is immutable.
        if role == "datasets" and image.get("tag") != "v1":
            tag = entry["agent_image"]["tag"]
        deploy.validate_image(image, f"images.{role}", expected_tag=tag)
        # Adoption can change publisher namespace but keeps the catalog naming contract.
        suffix = (current or entry[ROLE_KEYS[role]])["repository"].split("/", 2)[2]
        if (
            re.fullmatch(
                r"docker.io/[a-z0-9]+(?:[._-][a-z0-9]+)*/" + re.escape(suffix),
                image["repository"],
            )
            is None
        ):
            raise ValueError(f"published repository does not match role: {role}")
        if image["digest"] == PLACEHOLDER:
            raise ValueError("published images cannot contain placeholders")
        (target if target is not None else result[ROLE_KEYS[role]]).update(image)
    return result


def verify_images(entry: dict, root: Path, payload: Path | None) -> None:
    extension = dataset_extension(deploy.load_json(root / "policy.json"))
    for image, single_layer in resolver.task_images(entry):
        if image["digest"] == PLACEHOLDER:
            raise ValueError("image digest is a placeholder")
        expected = None
        executable = False
        if single_layer:
            expected = {
                f"datasets/{mode}/sha256/{entry['datasets'][f'{mode}_sha256'].removeprefix('sha256:')}.{extension}": dataset_path(
                    root, mode, extension
                ).read_bytes()
                for mode in ("train", "eval")
            }
        elif image is entry.get("reference"):
            if payload is None:
                raise ValueError(
                    "reference image verification requires its payload file"
                )
            definition = deploy.load_json(root / "policy.json")
            expected = {
                "opt/frontier/reference-candidate": (
                    REPOSITORY_ROOT / LAUNCHER
                ).read_bytes(),
                "opt/frontier/reference-payload/"
                + entrypoint_member(definition): payload.read_bytes(),
            }
            executable = True
        resolved = resolver.resolve_digest(
            image["repository"],
            image["digest"],
            single_layer=single_layer,
            expected_members=expected,
            executable=executable,
        )
        tagged = resolver.resolve_digest(
            image["repository"], image["tag"], single_layer=single_layer
        )
        if resolved != image["digest"] or tagged != resolved:
            raise ValueError(
                "published digest/tag does not bind the same linux/amd64 leaf"
            )


def docker(command: list[str], *, capture=False) -> str:
    return builder.run(["docker", *command], capture=capture)


def build_reference_overlay(image: dict, payload: Path, member: str) -> None:
    with tempfile.TemporaryDirectory(prefix="frontier-reference-") as directory:
        context = Path(directory)
        shutil.copyfile(REPOSITORY_ROOT / LAUNCHER, context / "launcher")
        shutil.copyfile(payload, context / "payload")
        # The launcher adds payload/ when packaging. Keeping this root matches
        # the frozen launcher and all existing reference CandidateVersions.
        (context / "Dockerfile").write_text(
            "FROM scratch\n"
            "COPY --chmod=0555 launcher /opt/frontier/reference-candidate\n"
            f"COPY --chmod=0555 payload /opt/frontier/reference-payload/{member}\n"
        )
        docker(
            [
                "build",
                "--provenance=false",
                "--platform",
                "linux/amd64",
                "--tag",
                image["repository"] + ":" + image["tag"],
                str(context),
            ]
        )


def remote_digest(image: dict, *, single_layer=False) -> str | None:
    try:
        return resolver.resolve_digest(
            image["repository"], image["tag"], single_layer=single_layer
        )
    except HTTPError as error:
        # Only this tag's own manifest 404 proves absence. An index may exist
        # while its child manifest is unavailable; that state is unknown.
        name = image["repository"].removeprefix("docker.io/")
        requested = f"https://registry-1.docker.io/v2/{name}/manifests/{quote(image['tag'], safe=':')}"
        if error.code == 404 and error.url == requested:
            return None
        raise


def local_digest(image: dict) -> str | None:
    # Docker config IDs are not registry manifest digests. A freshly built
    # image often has no RepoDigests: do not guess or overwrite an existing tag.
    values = (
        json.loads(
            docker(
                [
                    "image",
                    "inspect",
                    "--format",
                    "{{json .RepoDigests}}",
                    image["repository"] + ":" + image["tag"],
                ],
                capture=True,
            )
        )
        or []
    )
    aliases = (image["repository"], image["repository"].removeprefix("docker.io/"))
    digests = {
        value.rsplit("@", 1)[1]
        for value in values
        if any(value.startswith(alias + "@") for alias in aliases)
    }
    return next(iter(digests)) if len(digests) == 1 else None


def publish_images(entry: dict) -> None:
    images = resolver.task_images(entry)
    # Check the entire set before any push, including already-present datasets:v1.
    absent = []
    for image, single_layer in images:
        remote = remote_digest(image, single_layer=single_layer)
        if remote is None:
            absent.append(image)
        elif remote != local_digest(image):
            raise ValueError(
                f"refusing to push existing tag with different or unknown local digest: {image['repository']}:{image['tag']}"
            )
    for image in absent:
        # Recheck immediately before push; the registry has no conditional PUT.
        if (
            remote_digest(image, single_layer=image is entry["dataset_image"])
            is not None
        ):
            raise ValueError("tag appeared before push; rerun and verify its digest")
        docker(["push", image["repository"] + ":" + image["tag"]])


def tree_differences(source: Path, target: Path) -> list[str]:
    renderer.check_tree(target)

    def inventory(root):
        return {
            p.relative_to(root).as_posix(): (
                "directory" if p.is_dir() else file_sha(p),
                bool(p.stat().st_mode & 0o111),
            )
            for p in root.rglob("*")
        }

    before, after = inventory(target), inventory(source)
    return [
        f"{'added' if p not in before else 'removed' if p not in after else 'changed'}: {p}"
        for p in sorted(before.keys() | after.keys())
        if before.get(p) != after.get(p)
    ]


def row_spans(text: str) -> dict[str, tuple[int, int]]:
    decoder, spans = json.JSONDecoder(), {}
    pos = text.index("{") + 1
    while True:
        while text[pos].isspace() or text[pos] == ",":
            pos += 1
        if text[pos] == "}":
            return spans
        key, pos = decoder.raw_decode(text, pos)
        while text[pos].isspace() or text[pos] == ":":
            pos += 1
        start = pos
        _, pos = decoder.raw_decode(text, pos)
        spans[key] = (start, pos)


def definitions_text(original: str, name: str, definition: dict) -> str:
    spans = row_spans(original)
    value = json.dumps(
        definition, indent=2, ensure_ascii=False, sort_keys=True
    ).replace("\n", "\n  ")
    if name in spans:
        start, end = spans[name]
        result = original[:start] + value + original[end:]
    else:
        pos = original.rfind("}")
        prefix = original[:pos].rstrip()
        result = (
            prefix
            + ("," if spans else "")
            + "\n  "
            + json.dumps(name)
            + ": "
            + value
            + "\n"
            + original[pos:]
        )
    after = row_spans(result)
    for key, (start, end) in spans.items():
        if key != name:
            left, right = after[key]
            if original[start:end] != result[left:right]:
                raise ValueError(f"unrelated definition row changed: {key}")
    return result


def verify_definitions(
    catalog: dict, definitions: dict, *, root: Path | None = None
) -> None:
    root = root or REPOSITORY_ROOT
    # Same membership, renderer and order as render_task_definition.py --all,
    # with no rewrite of unrelated rows or their whitespace.
    rendered = {
        task["name"]: renderer.render(
            renderer.problem_directory(root / PROBLEMS, task["name"]).resolve(),
            task["name"],
        )
        for task in sorted(catalog["tasks"], key=lambda task: task["name"])
    }
    if rendered != definitions:
        raise ValueError("render --all differs from the updated task definitions")


def validate_task_schema(entry: dict, definitions: dict) -> None:
    schema = deploy.load_json(
        REPOSITORY_ROOT / "docs/contract/schemas/task-config-v1.schema.json"
    )
    images = {
        key: {**value, "reference": value["repository"] + "@" + value["digest"]}
        for key, value in entry.items()
        if key.endswith("_image")
    }
    # Reuse the frozen deployment projection with this import's definitions.
    document = publication.task_version_document(
        entry, images, definitions=definitions
    )["driver_config"]
    try:
        jsonschema.validate(document, schema)
    except jsonschema.ValidationError as error:
        raise ValueError(f"task-config schema: {error.message}") from error


def validate_replacement_catalog(
    catalog: dict, definitions: dict, name: str, entry: dict
) -> None:
    """Reconcile only the requested row after a source-first merge."""
    replacement = copy.deepcopy(catalog)
    rows = deploy.require_mapping(replacement, "task catalog").get("tasks")
    if not isinstance(rows, list):
        raise ValueError("task catalog.tasks must be an array")
    matches = [
        index
        for index, row in enumerate(rows)
        if isinstance(row, dict) and row.get("name") == name
    ]
    if len(matches) > 1:
        raise ValueError(f"task catalog contains duplicate problem: {name}")
    if matches:
        previous = rows[matches[0]]
        deploy.require_exact_keys(previous, set(entry), "replacement catalog row")
        for key, value in previous.items():
            if key.endswith("_image"):
                deploy.validate_image(value, f"replacement catalog row.{key}")
        for key, image in (previous.get("sidecar_images") or {}).items():
            deploy.validate_image(
                image, f"replacement catalog row.sidecar_images.{key}"
            )
        deploy.require_mapping(previous["datasets"], "replacement catalog row.datasets")
        deploy.require_mapping(previous["baseline"], "replacement catalog row.baseline")
        if previous["reference"] is not None:
            deploy.require_mapping(
                previous["reference"], "replacement catalog row.reference"
            )
        validate_task_schema(previous, definitions)
        del rows[matches[0]]
    unaffected = {key: value for key, value in definitions.items() if key != name}
    if rows or unaffected:
        deploy.validate_task_catalog(
            replacement, definitions=unaffected, root=REPOSITORY_ROOT
        )
    else:
        # A one-task catalog still has a publication header to validate.
        deploy.require_exact_keys(
            replacement, {"schema_version", "candidate", "tasks"}, "task catalog"
        )
        if replacement["schema_version"] != publication.SUPPORTED_TASK_CATALOG_SCHEMA:
            raise ValueError("unsupported task catalog schema_version")
        candidate = deploy.require_mapping(
            replacement["candidate"], "task catalog.candidate"
        )
        deploy.require_exact_keys(
            candidate, {"name", "overlay_image"}, "task catalog.candidate"
        )
        image = deploy.validate_image(
            candidate["overlay_image"], "task catalog.candidate.overlay_image"
        )
        if (
            candidate["name"] != "codex"
            or image["repository"] != "docker.io/joyemang33/fcs2-codex-overlay"
            or not publication.TASK_IMAGE_TAG_RE.fullmatch(image["tag"])
        ):
            raise ValueError("invalid task catalog candidate")


def validate_staged(root: Path, name: str, identity: str) -> None:
    catalog = deploy.load_json(root / "config/task-catalog.json")
    definitions = deploy.load_json(root / "config/task-definitions.json")
    if (
        renderer.source_sha256(renderer.problem_directory(root / PROBLEMS, name))
        != identity
    ):
        raise ValueError("copied tree identity differs from source")
    verify_definitions(catalog, definitions, root=root)
    deploy.validate_task_catalog(catalog, definitions=definitions, root=root)
    for task in catalog["tasks"]:
        validate_task_schema(task, definitions)


def install_staged(staged: Path, workspace: Path, name: str, record: dict) -> None:
    target = renderer.problem_directory(REPOSITORY_ROOT / PROBLEMS, name)
    previous = workspace / "previous"
    older_previous = staged / "older-previous"
    receipt = REPOSITORY_ROOT / ".generated/import" / (name + ".json")
    contents = {
        REPOSITORY_ROOT / "config" / filename: (
            staged / "config" / filename
        ).read_bytes()
        for filename in ("task-definitions.json", "task-catalog.json")
    }
    contents[receipt] = renderer.json_text(record).encode()
    originals = {
        path: path.read_bytes() if path.exists() else None for path in contents
    }
    moved_previous = moved_target = installed = False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if previous.exists():
            previous.rename(older_previous)
            moved_previous = True
        if target.exists():
            target.rename(previous)
            moved_target = True
        renderer.problem_directory(staged / PROBLEMS, name).rename(target)
        installed = True
        for path, body in contents.items():
            atomic_write(path, body)
        run_checks()
    except BaseException as error:
        # Restore in installation order, including files whose write was never
        # reached. A persistent failure must not prevent the other restorations.
        rollback = []
        if installed:
            rollback.append(("remove installed tree", lambda: shutil.rmtree(target)))
        if moved_target:
            rollback.append(("restore previous tree", lambda: previous.rename(target)))
        if moved_previous:
            rollback.append(
                ("restore older backup", lambda: older_previous.rename(previous))
            )
        for path, body in originals.items():
            if body is None:

                def restore(path=path):
                    return path.unlink(missing_ok=True)
            else:

                def restore(path=path, body=body):
                    return atomic_write(path, body)

            rollback.append((f"restore {path.relative_to(REPOSITORY_ROOT)}", restore))
        errors = []
        for step, restore in rollback:
            try:
                restore()
            except BaseException as rollback_error:
                errors.append(
                    f"{step}: {type(rollback_error).__name__}: {rollback_error}"
                )
        if errors:
            raise deploy.DeployError(
                f"import failed: {type(error).__name__}: {error}; "
                "rollback errors: " + "; ".join(errors)
            ) from error
        raise


def run_checks() -> None:
    environment = os.environ.copy()
    # Staged copies reuse this interpreter; keep an explicit caller override.
    # Escape Make's dollar expansion before the shell reads the quoted path.
    environment.setdefault("PYTHON", shlex.quote(sys.executable).replace("$", "$$"))
    for command in (
        ["make", "validate", "CONFIG=config/site.example.json"],
        ["make", "test"],
    ):
        subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def changed_files(paths: list[Path]) -> None:
    print("Changed files:")
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    print(child.relative_to(REPOSITORY_ROOT))
        else:
            print(path.relative_to(REPOSITORY_ROOT))


def install_batch(staged: Path, names: list[str], records: dict) -> None:
    """Install the complete validated batch, restoring every member on failure."""
    contents = {
        REPOSITORY_ROOT / "config" / filename: (
            staged / "config" / filename
        ).read_bytes()
        for filename in ("task-definitions.json", "task-catalog.json")
    }
    for name in names:
        contents[REPOSITORY_ROOT / ".generated/import" / (name + ".json")] = (
            renderer.json_text(records[name]).encode()
        )
    originals = {
        path: path.read_bytes() if path.exists() else None for path in contents
    }
    moved = []
    try:
        for name in names:
            target = renderer.problem_directory(REPOSITORY_ROOT / PROBLEMS, name)
            previous = staged / "previous" / name
            previous.parent.mkdir(parents=True, exist_ok=True)
            target.rename(previous)
            moved.append((target, previous))
            renderer.problem_directory(staged / PROBLEMS, name).rename(target)
        for path, body in contents.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(path, body)
        run_checks()
    except BaseException as error:
        errors = []
        for target, previous in reversed(moved):
            try:
                if target.exists():
                    shutil.rmtree(target)
                previous.rename(target)
            except BaseException as rollback_error:
                errors.append(f"restore {target}: {rollback_error}")
        for path, body in originals.items():
            try:
                if body is None:
                    path.unlink(missing_ok=True)
                else:
                    atomic_write(path, body)
            except BaseException as rollback_error:
                errors.append(f"restore {path}: {rollback_error}")
        if errors:
            raise deploy.DeployError(
                f"batch import failed: {error}; rollback errors: {'; '.join(errors)}"
            ) from error
        raise


def batch_main(arguments) -> int:
    parser = argparse.ArgumentParser(
        description="Atomically replace source-first task releases"
    )
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--replace", action="store_true", required=True)
    parser.add_argument("--plan", action="store_true")
    options = parser.parse_args(arguments)
    request = deploy.load_json(options.batch)
    deploy.require_exact_keys(request, {"schema_version", "exports"}, "batch import")
    if request["schema_version"] != "frontier.task-import-batch/v1":
        raise ValueError("unsupported batch import schema")
    members = request["exports"]
    if not isinstance(members, list) or not 2 <= len(members) <= 16:
        raise ValueError("batch requires between two and sixteen explicit exports")
    catalog = deploy.load_json(REPOSITORY_ROOT / "config/task-catalog.json")
    definitions_path = REPOSITORY_ROOT / "config/task-definitions.json"
    definitions = deploy.load_json(definitions_path)
    definition_body = definitions_path.read_text()
    names, prepared, records = [], {}, {}
    for member in members:
        deploy.require_exact_keys(
            member, {"export_dir", "published_images", "no_reference"}, "batch export"
        )
        if type(member["no_reference"]) is not bool:
            raise ValueError("no_reference must be boolean")
        for field in ("export_dir", "published_images"):
            if not isinstance(member[field], str) or not member[field]:
                raise ValueError(f"{field} must be a nonempty path")
        source = Path(member["export_dir"]).absolute()
        manifest = Path(member["published_images"]).absolute()
        definition = validate_export(source)
        name = source.name
        if name in prepared:
            raise ValueError("duplicate batch task: " + name)
        matches = [i for i, task in enumerate(catalog["tasks"]) if task["name"] == name]
        if len(matches) != 1:
            raise ValueError(
                "batch replacement requires one existing catalog task: " + name
            )
        target = renderer.problem_directory(REPOSITORY_ROOT / PROBLEMS, name)
        if source.resolve() == target.resolve() or manifest.resolve().is_relative_to(
            source.resolve()
        ):
            raise ValueError(
                "batch source/manifest conflicts with destination/export tree"
            )
        payload = None if member["no_reference"] else source / "judge/reference.patch"
        if payload is not None:
            if not payload.is_file():
                raise ValueError("batch reference payload is missing")
            entrypoint_member(definition)
        entry = published_manifest(
            manifest,
            name,
            definition["source"]["source_set_sha256"],
            catalog_entry(source, definition, DEFAULT_IMAGE_NAMESPACE, payload),
        )
        definitions[name] = definition
        definition_body = definitions_text(definition_body, name, definition)
        catalog["tasks"][matches[0]] = entry
        intent = {
            "source_sha256": definition["source"]["source_set_sha256"],
            "images": "published",
            "publish": False,
            "image_namespace": DEFAULT_IMAGE_NAMESPACE,
            "no_reference": member["no_reference"],
            "baseline_comparable": None,
            "payload_sha256": file_sha(payload) if payload else None,
            "published_images_sha256": file_sha(manifest),
        }
        records[name] = {
            "schema_version": "frontier.preview-task-import/v1",
            "intent": intent,
            "entry": entry,
        }
        prepared[name] = (source, payload, entry)
        names.append(name)
    destinations = [
        renderer.problem_directory(REPOSITORY_ROOT / PROBLEMS, name).resolve()
        for name in names
    ]
    for source, _, _ in prepared.values():
        for destination in destinations:
            if source.resolve().is_relative_to(
                destination
            ) or destination.is_relative_to(source.resolve()):
                raise ValueError("batch export overlaps a destination tree")
    workspace = REPOSITORY_ROOT / ".generated/import/batch"
    if not options.plan:
        workspace.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="staged-", dir=workspace if not options.plan else None
    ) as directory:
        staged = Path(directory)
        for task in catalog["tasks"]:
            name = task["name"]
            target = renderer.problem_directory(REPOSITORY_ROOT / PROBLEMS, name)
            staged_task = staged / target.relative_to(REPOSITORY_ROOT)
            staged_task.parent.mkdir(parents=True, exist_ok=True)
            if name in prepared:
                shutil.copytree(prepared[name][0], staged_task)
            else:
                staged_task.symlink_to(target)
        (staged / LAUNCHER).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY_ROOT / LAUNCHER, staged / LAUNCHER)
        (staged / "config").mkdir()
        (staged / "config/task-definitions.json").write_text(definition_body)
        (staged / "config/task-catalog.json").write_text(renderer.json_text(catalog))
        for name in names:
            source, payload, entry = prepared[name]
            validate_staged(staged, name, records[name]["intent"]["source_sha256"])
            staged_task = renderer.problem_directory(staged / PROBLEMS, name)
            verify_images(
                entry,
                staged_task,
                staged_task / "judge/reference.patch" if payload else None,
            )
        if options.plan:
            print("Validated complete batch without installation: " + ", ".join(names))
            return 0
        install_batch(staged, names, records)
    changed_files([REPOSITORY_ROOT / "config/task-catalog.json", definitions_path])
    return 0


def main(arguments=None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if any(value == "--batch" or value.startswith("--batch=") for value in arguments):
        return batch_main(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--published-images", type=Path)
    parser.add_argument(
        "--images", choices=("auto", "published", "build"), default="auto"
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--image-namespace", default=DEFAULT_IMAGE_NAMESPACE)
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument(
        "--measured-baseline",
        type=Path,
        metavar="FILE",
        help="qualify retained independent native reference evidence outside the export",
    )
    baseline.add_argument(
        "--baseline-comparable",
        metavar="NOTE",
        help="assert result.json measured this export revision, with an operator note",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--replace", action="store_true")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--no-reference", action="store_true")
    reference.add_argument("--reference-payload", type=Path)
    options = parser.parse_args(arguments)
    if (
        options.baseline_comparable is not None
        and not options.baseline_comparable.strip()
    ):
        raise ValueError("--baseline-comparable requires a nonempty operator note")
    if re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", options.image_namespace) is None:
        raise ValueError("image namespace must be a bounded Docker Hub namespace")
    image_mode = (
        ("published" if options.published_images else "build")
        if options.images == "auto"
        else options.images
    )
    if image_mode == "published" and not options.published_images:
        raise ValueError("--images published requires --published-images")
    if image_mode == "build" and options.published_images:
        raise ValueError("--images build cannot use --published-images")
    if options.publish and image_mode != "build":
        raise ValueError("--publish is only supported for build mode")
    source = options.export_dir.absolute()
    definition = validate_export(source)
    name, identity = source.name, definition["source"]["source_set_sha256"]
    target = renderer.problem_directory(REPOSITORY_ROOT / PROBLEMS, name)
    catalog_path = REPOSITORY_ROOT / "config/task-catalog.json"
    definitions_path = REPOSITORY_ROOT / "config/task-definitions.json"
    receipt_path = REPOSITORY_ROOT / ".generated/import" / (name + ".json")
    catalog = deploy.load_json(catalog_path)
    definitions = deploy.load_json(definitions_path)
    catalog_validation_error = None
    try:
        deploy.validate_task_catalog(
            catalog, definitions=definitions, root=REPOSITORY_ROOT
        )
    except deploy.DeployError as error:
        if not options.replace or options.check:
            raise
        catalog_validation_error = error
    existing = next((task for task in catalog["tasks"] if task["name"] == name), None)
    if source.resolve() == target.resolve() and not options.check:
        raise ValueError(
            "export must be outside its destination tree (use --check for an existing import)"
        )
    if options.published_images and options.published_images.resolve().is_relative_to(
        source.resolve()
    ):
        raise ValueError("published manifest must live outside the export tree")
    payload = (
        None
        if options.no_reference
        else options.reference_payload or (source / "judge/reference.patch")
    )
    if options.reference_payload:
        renderer.check_tree(options.reference_payload)
        if not options.reference_payload.is_file():
            raise ValueError("reference payload must be a regular file")
        with options.reference_payload.open("rb") as stream:
            if stream.read(128).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ):
                raise ValueError("reference payload is an unresolved LFS pointer")
        if (
            options.reference_payload.stat().st_size > MAX_FILE_BYTES
            and not lfs_tracked(
                options.reference_payload.parent, options.reference_payload
            )
        ):
            raise ValueError("reference payload exceeds 512 MiB and is not LFS-tracked")
        patch = source / "judge/reference.patch"
        if patch.is_file() and file_sha(patch) != file_sha(options.reference_payload):
            raise ValueError("explicit payload conflicts with judge/reference.patch")
    if payload is not None and not payload.is_file():
        payload = None
    if payload is not None:
        entrypoint_member(definition)
    entry = catalog_entry(source, definition, options.image_namespace, payload)
    if options.baseline_comparable is not None:
        if entry["baseline"]["provenance"] != "historical-authoring":
            raise ValueError(
                "--baseline-comparable requires judge/result.json and complete baseline binding files"
            )
        entry["baseline"].update(
            provenance="historical-authoring",
            comparable=True,
            comparability_note=options.baseline_comparable,
        )
    if image_mode == "published":
        entry = published_manifest(options.published_images, name, identity, entry)
    if options.measured_baseline is not None:
        if image_mode != "published" or payload is None:
            raise ValueError(
                "--measured-baseline requires published images and a reference payload"
            )
        entry["baseline"] = baseline_from_measurement(
            options.measured_baseline,
            source,
            definition,
            entry,
            payload,
            REPOSITORY_ROOT / LAUNCHER,
            REPOSITORY_ROOT,
        )
    validate_task_schema(entry, {**definitions, name: definition})
    if catalog_validation_error is not None:
        validate_replacement_catalog(catalog, definitions, name, entry)
    intent = {
        "source_sha256": identity,
        "images": image_mode,
        "publish": options.publish,
        "image_namespace": options.image_namespace,
        "no_reference": options.no_reference,
        "baseline_comparable": options.baseline_comparable,
        "payload_sha256": file_sha(payload) if payload else None,
        "published_images_sha256": file_sha(options.published_images)
        if options.published_images
        else None,
    }
    if options.measured_baseline is not None:
        intent["measured_baseline_sha256"] = entry["baseline"]["measurement"][
            "packet_sha256"
        ]
    differences = (
        tree_differences(source, target)
        if target.exists()
        else ["destination tree missing"]
    )
    if options.check:
        if existing is None or differences or definitions.get(name) != definition:
            raise ValueError(
                "imported tree/definition identity differs: " + ", ".join(differences)
            )
        # Validate the descriptor hashes independently of the original import mode.
        for key in ("datasets", "baseline", "reference"):
            if key == "reference":
                expected = entry[key]
                actual = existing[key]
                if (
                    (expected is None) != (actual is None)
                    or expected
                    and any(
                        expected[k] != actual[k]
                        for k in ("launcher_sha256", "payload_sha256")
                    )
                ):
                    raise ValueError("catalog reference differs from import inputs")
            elif key == "datasets" and existing[key] != entry[key]:
                raise ValueError("catalog datasets differ")
            elif key == "baseline":
                annotations = (
                    set()
                    if options.measured_baseline
                    else {"provenance", "comparable", "comparability_note"}
                )
                expected = {k: v for k, v in entry[key].items() if k not in annotations}
                actual = {
                    k: v for k, v in existing[key].items() if k not in annotations
                }
                if deploy.canonical_json(actual) != deploy.canonical_json(expected):
                    raise ValueError(
                        "catalog baseline differs from freshly derived import inputs"
                    )
        if options.measured_baseline is not None:
            for key in ROLE_KEYS.values():
                if key != "datasets" and existing.get(key) != entry.get(key):
                    raise ValueError(
                        "catalog images differ from measured import inputs"
                    )
        verify_images(existing, source, payload)
        print(f"{name}: check passed")
        return 0
    receipt = deploy.load_json(receipt_path) if receipt_path.exists() else None
    if existing is not None or target.exists():
        identical = (
            not differences
            and definitions.get(name) == definition
            and receipt is not None
            and receipt.get("intent") == intent
            and receipt.get("entry") == existing
        )
        if identical:
            print(f"{name}: already imported, no changes")
            if not options.plan:
                run_checks()
            return 0
        if not options.replace:
            raise ValueError(
                f"{name} already exists; use --replace; differences: "
                + ", ".join(
                    differences or ["publication options/catalog/receipt differ"]
                )
            )
    actions = [
        f"copy verbatim: {source} -> {target}",
        f"definition row: {name}",
        f"catalog v2 entry: {json.dumps(entry, sort_keys=True)}",
        f"baseline source: {'judge/result.json' if (source / 'judge/result.json').exists() else 'unverified (no result)'}",
    ]
    if options.measured_baseline is not None:
        actions[3] = (
            f"baseline source: independent measurement packet {options.measured_baseline}"
        )
    actions += [
        f"{'verify' if image_mode == 'published' else 'build'}: {image['repository']}:{image['tag']}"
        for image, _ in resolver.task_images(entry)
    ]
    if options.publish:
        actions += [
            "verify dataset bodies; refuse conflicting remote tags; push; resolve only task "
            + name
        ]
    actions += [
        f"checkpoint: {receipt_path}",
        "render --all comparison; make validate; make test",
    ]
    if options.plan:
        print("\n".join(actions))
        return 0
    definition_body = definitions_text(definitions_path.read_text(), name, definition)
    definitions[name] = definition
    if existing is None:
        catalog["tasks"].append(entry)
    else:
        catalog["tasks"][catalog["tasks"].index(existing)] = entry
    workspace = REPOSITORY_ROOT / ".generated/import" / name
    workspace.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="staged-", dir=workspace) as directory:
        staged = Path(directory)
        staged_tree = staged / target.relative_to(REPOSITORY_ROOT)
        staged_tree.parent.mkdir(parents=True)
        shutil.copytree(source, staged_tree)
        # Unchanged tasks remain read-only inputs; all changed files are copies.
        for task in catalog["tasks"]:
            if task["name"] != name:
                peer = renderer.problem_directory(
                    REPOSITORY_ROOT / PROBLEMS, task["name"]
                )
                staged_peer = staged / peer.relative_to(REPOSITORY_ROOT)
                staged_peer.parent.mkdir(parents=True, exist_ok=True)
                staged_peer.symlink_to(peer)
        (staged / LAUNCHER).parent.mkdir(parents=True)
        shutil.copyfile(REPOSITORY_ROOT / LAUNCHER, staged / LAUNCHER)
        (staged / "config").mkdir()
        (staged / "config/task-definitions.json").write_text(definition_body)
        staged_catalog = staged / "config/task-catalog.json"
        staged_catalog.write_text(renderer.json_text(catalog))
        staged_payload = None
        if payload is not None:
            if payload.resolve().is_relative_to(source.resolve()):
                staged_payload = staged_tree / payload.resolve().relative_to(
                    source.resolve()
                )
            else:
                staged_payload = staged / "reference-payload"
                shutil.copyfile(payload, staged_payload)
            if file_sha(staged_payload) != entry["reference"]["payload_sha256"]:
                raise ValueError("copied reference payload differs from source")
        validate_staged(staged, name, identity)
        build_receipt = None
        if image_mode == "published":
            verify_images(entry, staged_tree, staged_payload)
        else:
            bases = {}
            build_receipt = builder.build_problem(
                name,
                options.image_namespace,
                push=False,
                bases=bases,
                root=staged_tree,
                definition=definition,
            )
            build_receipt = {**build_receipt, "bases": bases}
            if staged_payload is not None:
                build_reference_overlay(
                    entry["reference"], staged_payload, entrypoint_member(definition)
                )
            if options.publish:
                publish_images(entry)
                resolver.resolve_files(
                    staged_catalog, name, definitions=definitions, root=staged
                )
                catalog = deploy.load_json(staged_catalog)
                entry = next(task for task in catalog["tasks"] if task["name"] == name)
                verify_images(entry, staged_tree, staged_payload)
        # Validate the final bytes that will be installed, including resolved pins.
        validate_staged(staged, name, identity)
        record = {
            "schema_version": "frontier.preview-task-import/v1",
            "intent": intent,
            "entry": entry,
        }
        if build_receipt is not None:
            record["build"] = build_receipt
        install_staged(staged, workspace, name, record)
    changed_files([target, definitions_path, catalog_path, receipt_path])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ValueError,
        OSError,
        deploy.DeployError,
        builder.BuildFailure,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(f"import-task: {error}") from None
