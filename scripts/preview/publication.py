"""Task publication, immutable version documents and seed receipts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
from pathlib import Path
from typing import Any, Mapping, Sequence

import render_task_definition as renderer
from task_selection import TaskSelection, requirements, select_tasks

from .images import (
    load_runtime_images,
    mirror_task_images,
)
from .infra import (
    DIGEST_RE,
    MANAGEMENT_SERVICE,
    PLATFORM_NAMESPACE,
    REPOSITORY_ROOT,
    SUPPORTED_TASK_CATALOG_SCHEMA,
    TASK_CATALOG_PATH,
    TASK_DEFINITIONS_PATH,
    TASK_IMAGE_TAG_RE,
    TASK_PLACEHOLDER_DIGEST,
    DeployError,
    api_json,
    atomic_write,
    canonical_json,
    canonical_object_sha256,
    executable,
    k8s_list,
    kubectl_apply,
    kubernetes_environment,
    load_json,
    platform_labels,
    port_forward,
    require_exact_keys,
    require_int,
    require_mapping,
    require_string,
    run_command,
    runtime_admission_preflight,
    source_image_tag_matches,
    stable_ulid,
    state_directory,
    validate_image,
)


def load_task_definitions() -> dict[str, dict[str, Any]]:
    """Load the reviewed definitions when used, never while importing modules."""
    return json.loads(TASK_DEFINITIONS_PATH.read_text())


def __getattr__(name: str) -> Any:
    if name == "TASK_DEFINITIONS":
        return load_task_definitions()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def validate_task_catalog(
    raw: Any,
    *,
    require_resolved: bool = False,
    definitions: dict | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    definitions = definitions if definitions is not None else load_task_definitions()
    root = root or REPOSITORY_ROOT
    catalog = require_mapping(raw, "task catalog")
    require_exact_keys(
        catalog, {"schema_version", "candidate", "tasks"}, "task catalog"
    )
    if catalog["schema_version"] != SUPPORTED_TASK_CATALOG_SCHEMA:
        raise DeployError(
            f"task catalog schema_version must be {SUPPORTED_TASK_CATALOG_SCHEMA}"
        )
    candidate = require_mapping(catalog["candidate"], "task catalog.candidate")
    require_exact_keys(candidate, {"name", "overlay_image"}, "task catalog.candidate")
    if candidate["name"] != "codex":
        raise DeployError("task catalog.candidate.name must equal codex")
    overlay = validate_image(
        candidate["overlay_image"], "task catalog.candidate.overlay_image"
    )
    if overlay["repository"] != "docker.io/joyemang33/fcs2-codex-overlay":
        raise DeployError(
            "candidate overlay repository must equal docker.io/joyemang33/fcs2-codex-overlay"
        )
    if not TASK_IMAGE_TAG_RE.fullmatch(overlay["tag"]):
        raise DeployError("candidate overlay tag must equal sha-<40 lowercase hex>")
    if not isinstance(catalog["tasks"], list) or not catalog["tasks"]:
        raise DeployError("task catalog.tasks must be a non-empty array")
    tasks, unresolved, seen = [], [], set()
    if overlay["digest"] == TASK_PLACEHOLDER_DIGEST:
        unresolved.append("candidate.overlay_image.digest")
    for index, value in enumerate(catalog["tasks"]):
        path = f"task catalog.tasks[{index}]"
        task = require_mapping(value, path)
        name = require_string(task.get("name"), f"{path}.name")
        if name not in definitions:
            raise DeployError(f"{path}.name is not in the fixed task definitions")
        if name in seen:
            raise DeployError(f"task catalog contains duplicate problem: {name}")
        seen.add(name)
        policy = definitions[name]
        requirements(policy)
        repositories = {
            "agent_image": f"fcs2-task-{name}-agent",
            "verifier_image": f"fcs2-task-verifier-{name}",
            "dataset_image": f"fcs2-task-{name}-datasets",
        }
        gpu = policy.get("gpu", {})
        if gpu.get("agent", {}).get("max_instances", 0):
            repositories["agent_gpu_image"] = f"fcs2-task-{name}-agent-gpu"
        if gpu.get("verifier", {}).get("max_gpu_instances", 0):
            repositories["verifier_gpu_image"] = f"fcs2-task-verifier-{name}-gpu"
        sidecars = policy.get("sidecars") or {}
        require_exact_keys(
            task,
            {"name", "datasets", "reference", "baseline", *repositories}
            | ({"sidecar_images"} if sidecars else set()),
            path,
        )
        images = {}
        expected_tag = policy["image_tag"]
        if not re.fullmatch(r"(?:sha-[0-9a-f]{40}|source-[0-9a-f]{64})", expected_tag):
            raise DeployError(f"task definition {name} has no immutable source tag")
        if expected_tag.startswith("source-") and policy["source"].get(
            "source_set_sha256"
        ) != "sha256:" + expected_tag.removeprefix("source-"):
            raise DeployError(
                f"task definition {name} source tag must bind source_set_sha256"
            )
        for kind, repository in repositories.items():
            image = validate_image(task[kind], f"{path}.{kind}")
            if (
                re.fullmatch(
                    r"docker.io/[a-z0-9]+(?:[._-][a-z0-9]+)*/" + re.escape(repository),
                    image["repository"],
                )
                is None
            ):
                raise DeployError(f"{path}.{kind}.repository must equal {repository}")
            tags = {"v1", expected_tag} if kind == "dataset_image" else {expected_tag}
            if image["tag"] not in tags and not source_image_tag_matches(
                image["tag"], expected_tag, image["digest"]
            ):
                raise DeployError(
                    f"{path}.{kind}.tag must equal reviewed source tag {expected_tag}"
                    + (" or legacy v1" if kind == "dataset_image" else "")
                )
            if image["digest"] == TASK_PLACEHOLDER_DIGEST:
                unresolved.append(f"{name}.{kind}.digest")
            images[kind] = image
        if sidecars:
            # One task-owned image per declared sidecar, published like the GPU
            # images under the reviewed source tag.
            declared = require_mapping(task["sidecar_images"], f"{path}.sidecar_images")
            roles = {profile.get("image", sidecar) for sidecar, profile in sidecars.items()}
            require_exact_keys(declared, roles, f"{path}.sidecar_images")
            images["sidecar_images"] = {}
            for sidecar in sorted(declared):
                kind = f"sidecar_images.{sidecar}"
                image = validate_image(declared[sidecar], f"{path}.{kind}")
                repository = f"fcs2-task-{name}-sidecar-{sidecar}"
                if (
                    re.fullmatch(
                        r"docker.io/[a-z0-9]+(?:[._-][a-z0-9]+)*/" + re.escape(repository),
                        image["repository"],
                    )
                    is None
                ):
                    raise DeployError(f"{path}.{kind}.repository must equal {repository}")
                if image["tag"] != expected_tag and not source_image_tag_matches(
                    image["tag"], expected_tag, image["digest"]
                ):
                    raise DeployError(
                        f"{path}.{kind}.tag must equal reviewed source tag {expected_tag}"
                    )
                if image["digest"] == TASK_PLACEHOLDER_DIGEST:
                    unresolved.append(f"{name}.{kind}.digest")
                images["sidecar_images"][sidecar] = image
        datasets = require_mapping(task["datasets"], f"{path}.datasets")
        require_exact_keys(
            datasets, {"train_sha256", "eval_sha256"}, f"{path}.datasets"
        )
        for key, value in datasets.items():
            if not isinstance(value, str) or not DIGEST_RE.fullmatch(value):
                raise DeployError(
                    f"{path}.datasets.{key} must be a lowercase sha256 digest"
                )
            if value != policy["datasets"][key]:
                raise DeployError(
                    f"{path}.datasets.{key} differs from the reviewed task definition"
                )
        if datasets["train_sha256"] == datasets["eval_sha256"]:
            raise DeployError(f"{path} train and eval dataset objects must be distinct")
        try:
            validate_reference(name, task["reference"], policy, root=root)
            validate_baseline(
                name, task["baseline"], task["reference"], policy, root=root
            )
        except (OSError, ValueError) as error:
            raise DeployError(str(error)) from error
        if (
            task["reference"] is not None
            and task["reference"]["digest"] == TASK_PLACEHOLDER_DIGEST
        ):
            unresolved.append(f"{name}.reference.digest")
        tasks.append(
            {
                "name": name,
                **images,
                "datasets": dict(datasets),
                "reference": task["reference"],
                "baseline": task["baseline"],
            }
        )
    missing = sorted(set(definitions) - seen)
    if missing:
        raise DeployError(
            f"task catalog must contain every fixed task exactly once; missing={missing}"
        )
    if require_resolved and unresolved:
        raise DeployError(
            "task catalog contains placeholder image digests; resolve before seeding: "
            + ", ".join(unresolved)
        )
    return {
        "schema_version": SUPPORTED_TASK_CATALOG_SCHEMA,
        "candidate": {"name": "codex", "overlay_image": overlay},
        "tasks": tasks,
        "unresolved_digests": unresolved,
    }


def load_task_selection(
    config: Mapping[str, Any],
    catalog_path: Path | None = None,
    *,
    require_resolved: bool = False,
) -> tuple[dict[str, Any], TaskSelection]:
    TASK_DEFINITIONS = load_task_definitions()
    raw = load_json(catalog_path if catalog_path is not None else TASK_CATALOG_PATH)
    catalog = validate_task_catalog(raw, require_resolved=require_resolved)
    selection = select_tasks(config, raw, TASK_DEFINITIONS)
    catalog["tasks"] = [
        task for task in catalog["tasks"] if task["name"] in selection.selected
    ]
    return catalog, selection


def validate_seed_receipt(
    seed: Mapping[str, Any], selection: TaskSelection, site: str
) -> None:
    if seed.get("site") != site:
        raise DeployError("seed receipt belongs to a different site")
    schema = seed.get("schema_version")
    if schema == "frontier.deployment-task-seed-receipt/v1":
        expected = {"catalog_sha256": selection.effective_catalog_sha256}
        if (
            "model_domains" in seed
            and sorted(set(seed["model_domains"])) != selection.model_domains
        ):
            raise DeployError(
                "seed receipt model_domains differ from the site selection"
            )
    elif schema == "frontier.deployment-task-seed-receipt/v2":
        expected = {
            key: getattr(selection, key)
            for key in (
                "catalog_sha256",
                "effective_catalog_sha256",
                "selection_digest",
                "selected",
                "excluded",
                "model_domains",
            )
        }
        tasks = seed.get("tasks")
        if not isinstance(tasks, dict) or set(tasks) != set(selection.selected):
            raise DeployError("seed receipt tasks differ from the site selection")
        for name, task in tasks.items():
            require_exact_keys(
                require_mapping(task, f"seed tasks.{name}"),
                {"task_version_id", "definition_digest"},
                f"seed tasks.{name}",
            )
        if seed.get("task_version_ids") != [
            tasks[name]["task_version_id"] for name in selection.selected
        ]:
            raise DeployError("seed receipt ordered task IDs differ from its tasks")
    else:
        raise DeployError("unsupported task seed receipt schema")
    for key, value in expected.items():
        if seed.get(key) != value:
            raise DeployError(
                f"seed receipt {key}={seed.get(key)} does not bind current selection {key}={value}"
            )


def validate_seed_task(
    seed: Mapping[str, Any], name: str, document: Mapping[str, Any]
) -> None:
    version_id = document["driver_config"]["task_version_id"]
    if version_id not in seed.get("task_version_ids", []):
        raise DeployError(f"task is absent from the seed receipt: {name}")
    if seed["schema_version"] == "frontier.deployment-task-seed-receipt/v2" and seed[
        "tasks"
    ].get(name) != {
        "task_version_id": version_id,
        "definition_digest": document["definition_digest"],
    }:
        raise DeployError(f"seed receipt task binding differs: {name}")


def task_dataset_uri(
    problem: str,
    mode: str,
    sha256: str,
    *,
    definitions: Mapping[str, Any] | None = None,
) -> str:
    TASK_DEFINITIONS = load_task_definitions() if definitions is None else definitions
    if problem not in TASK_DEFINITIONS or mode not in {"train", "eval"}:
        raise DeployError("invalid demo dataset identity")
    if not DIGEST_RE.fullmatch(sha256):
        raise DeployError("demo dataset identity must be a lowercase sha256 digest")
    return (
        "s3://frontier-local-task-data/datasets/frontier-cs-2.0-demo/"
        f"{problem}/{mode}/sha256/{sha256.removeprefix('sha256:')}.{TASK_DEFINITIONS[problem]['dataset_extension']}"
    )


def task_dataset_relative_path(problem: str, mode: str, sha256: str) -> str:
    return task_dataset_uri(problem, mode, sha256).split("/", 3)[3]


def task_dataset_seed_job_name(
    task: Mapping[str, Any], mirrored_dataset_digest: str
) -> str:
    problem_slug = task["name"].replace("_", "-")
    seed = canonical_json(
        {
            "dataset_image": mirrored_dataset_digest,
            "datasets": task["datasets"],
        }
    )
    suffix = hashlib.sha256(seed).hexdigest()[:16]
    return f"frontier-seed-{problem_slug[:27]}-{suffix}"


def render_task_dataset_seed_manifest(
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    mirrored_images: Mapping[str, Mapping[str, str]],
    runner_image: str,
    *,
    job_name: str | None = None,
    native_targets: list[dict[str, str]] | None = None,
) -> bytes:
    TASK_DEFINITIONS = {} if native_targets is not None else load_task_definitions()
    problem = task["name"]
    if native_targets is None and problem not in TASK_DEFINITIONS:
        raise DeployError("cannot render a dataset seed Job for an unknown problem")
    dataset_image = require_mapping(
        mirrored_images.get("dataset_image"), "mirrored dataset image"
    )
    digest = require_string(
        dataset_image.get("digest"), "mirrored dataset image.digest"
    )
    reference = require_string(
        dataset_image.get("reference"), "mirrored dataset image.reference"
    )
    if not DIGEST_RE.fullmatch(digest) or not reference.endswith("@" + digest):
        raise DeployError("mirrored dataset image is not digest pinned")
    labels = {
        **platform_labels(config, "frontier-task-seed"),
        "frontier.dev/task": problem.replace("_", "-")[:63],
    }
    scratch_prefix = "/data/.seed-scratch-"
    if job_name is not None:
        if len(job_name) > 63 or not re.fullmatch(
            r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", job_name
        ):
            raise DeployError(
                "dataset seed Job name must be a DNS label of at most 63 characters"
            )
        scratch_prefix += job_name + "-"
    else:
        job_name = task_dataset_seed_job_name(task, digest)
    network = config["network"]
    registry = f"http://{network['registry_cluster_ip']}:5000"
    slug = problem.replace("_", "-")
    repo = f"frontier/tasks/{slug}/dataset"
    if native_targets is not None:
        from .native_local_publication import validate_seed_targets

        targets = validate_seed_targets(native_targets)
    else:
        targets = []
        for mode in ("train", "eval"):
            sha256 = task["datasets"][f"{mode}_sha256"]
            digest_hex = sha256.removeprefix("sha256:")
            member = f"datasets/{mode}/sha256/{digest_hex}.{TASK_DEFINITIONS[problem]['dataset_extension']}"
            target = "/data/" + task_dataset_relative_path(problem, mode, sha256)
            targets.append({"member": member, "target": target, "sha256": digest_hex})
    seed_settings = json.dumps(
        {
            "registry": registry,
            "repository": repo,
            "manifest_digest": digest,
            "targets": targets,
        }
    )
    # The dataset image is FROM scratch (one layer holding dataset bodies, no
    # shell), so the seed container pulls that layer blob straight from the
    # in-cluster registry and extracts the content-addressed files onto the
    # seed PVC, verifying every SHA-256 before an atomic rename.
    script = "\n".join(
        [
            "import hashlib, json, os, shutil, tarfile, urllib.request",
            f"settings = json.loads({seed_settings!r})",
            "base = settings['registry'] + '/v2/' + settings['repository']",
            "accept = 'application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'",
            "request = urllib.request.Request(base + '/manifests/' + settings['manifest_digest'], headers={'Accept': accept})",
            "manifest = json.load(urllib.request.urlopen(request, timeout=60))",
            "layers = manifest.get('layers') or []",
            "assert len(layers) == 1, f'dataset image must have exactly one layer, got {len(layers)}'",
            f"scratch = {scratch_prefix!r} + str(os.getpid())",
            "os.makedirs(scratch, exist_ok=True)",
            "blob_path = scratch + '/layer.blob'",
            "with urllib.request.urlopen(base + '/blobs/' + layers[0]['digest'], timeout=1800) as source, open(blob_path, 'wb') as sink:",
            "    shutil.copyfileobj(source, sink, 1024 * 1024)",
            "archive = tarfile.open(blob_path, mode='r:*')",
            "for entry in settings['targets']:",
            "    target = entry['target']",
            "    if os.path.exists(target):",
            "        digest = hashlib.sha256()",
            "        with open(target, 'rb') as existing:",
            "            for chunk in iter(lambda: existing.read(1024 * 1024), b''):",
            "                digest.update(chunk)",
            "        assert digest.hexdigest() == entry['sha256'], 'existing file differs: ' + target",
            "        continue",
            "    member = archive.extractfile(entry['member'])",
            "    assert member is not None, 'missing ' + entry['member']",
            "    os.makedirs(os.path.dirname(target), exist_ok=True)",
            "    temporary = scratch + '/' + entry['sha256'] + '.tmp'",
            "    digest = hashlib.sha256()",
            "    with open(temporary, 'wb') as handle:",
            "        for chunk in iter(lambda: member.read(1024 * 1024), b''):",
            "            digest.update(chunk)",
            "            handle.write(chunk)",
            "    assert digest.hexdigest() == entry['sha256'], 'sha mismatch for ' + entry['member']",
            "    os.chmod(temporary, 0o444)",
            "    os.replace(temporary, target)",
            "archive.close()",
            "shutil.rmtree(scratch)",
            "print(json.dumps({'dataset_sizes': {entry['sha256']: os.stat(entry['target']).st_size for entry in settings['targets']}}))",
        ]
    )
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": PLATFORM_NAMESPACE,
            "labels": labels,
        },
        "spec": {
            "activeDeadlineSeconds": 1800,
            "backoffLimit": 3,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "automountServiceAccountToken": False,
                    "restartPolicy": "OnFailure",
                    "nodeSelector": {"frontier.dev/pool": "system"},
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 65532,
                        "runAsGroup": 65532,
                        "fsGroup": 65532,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "containers": [
                        {
                            "name": "seed",
                            # The dataset image has no shell. This trusted runtime
                            # downloads its single layer and copies verified bodies.
                            "image": runner_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/usr/local/bin/python3", "-c", script],
                            "resources": {
                                "requests": {"cpu": "10m", "memory": "32Mi"},
                                "limits": {"cpu": "500m", "memory": "512Mi"},
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "volumeMounts": [{"name": "data", "mountPath": "/data"}],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "data",
                            "persistentVolumeClaim": {
                                "claimName": "frontier-object-seed"
                            },
                        }
                    ],
                },
            },
        },
    }
    return k8s_list([job])


def task_version_document(
    task: Mapping[str, Any],
    mirrored_images: Mapping[str, Mapping[str, str]],
    *,
    model_domains: Sequence[str] = ("openrouter.ai",),
    definitions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    TASK_DEFINITIONS = load_task_definitions() if definitions is None else definitions
    problem = task["name"]
    policy = TASK_DEFINITIONS[problem]
    policy_schema_version = require_string(
        policy.get("policy_schema_version"), "policy_schema_version"
    )
    network = {
        **policy.get("network", {}),
        "allowed_domains": sorted(
            set(policy.get("network", {}).get("allowed_domains", []))
            | set(model_domains)
        ),
    }
    platform = {"os": "linux", "architecture": "amd64"}
    image_identities: dict[str, dict[str, str]] = {}
    for image_name in ("agent_image", "verifier_image"):
        mirrored = require_mapping(
            mirrored_images.get(image_name), f"mirrored {image_name}"
        )
        digest = require_string(mirrored.get("digest"), f"mirrored {image_name}.digest")
        if not DIGEST_RE.fullmatch(digest):
            raise DeployError(f"mirrored {image_name}.digest is invalid")
        # The Driver pulls through the in-cluster registry (its egress policy
        # does not reach public registries), so the TaskConfig references the
        # digest-pinned in-cluster mirror rather than the public repository.
        reference = require_string(
            mirrored.get("reference"), f"mirrored {image_name}.reference"
        )
        if not reference.endswith("@" + digest):
            raise DeployError(f"mirrored {image_name}.reference is not digest pinned")
        image_identities[image_name] = {
            "reference": reference,
            "digest": digest,
        }
    if "workspace_bytes" in policy["verifier"]:
        require_int(
            policy["verifier"]["workspace_bytes"],
            "verifier.workspace_bytes",
            1,
            1 << 40,
        )
    version_seed = canonical_json(
        {
            "publication_generation": 2,
            "problem": problem,
            "policy": policy,
            "network": network,
            "agent_image": image_identities["agent_image"],
            "verifier_image": image_identities["verifier_image"],
            "gpu_images": {
                name: task[name]
                for name in ("agent_gpu_image", "verifier_gpu_image")
                if name in task
            },
            # Only tasks declaring sidecars carry this key, so every existing
            # task version keeps the identity it was seeded with.
            **(
                {"sidecar_images": task["sidecar_images"]}
                if task.get("sidecar_images")
                else {}
            ),
            "datasets": task["datasets"],
        }
    )
    task_version_id = "taskver_" + stable_ulid(version_seed)
    quota = policy["quota"]
    verifier = policy["verifier"]
    gpu = policy.get("gpu", {})
    train_sha256 = task["datasets"]["train_sha256"]
    eval_sha256 = task["datasets"]["eval_sha256"]
    task_config = {
        "schema_version": "frontier.task-config/v1",
        "task_version_id": task_version_id,
        "images": {
            "agent_cpu_base": {**image_identities["agent_image"], "platform": platform},
            "verifier": image_identities["verifier_image"],
        },
        "agent": policy["agent"],
        "quota": {
            "wall_time_seconds": quota["wall_time_seconds"],
            "agent_gpu": gpu.get(
                "agent",
                {
                    "allowed_types": [],
                    "max_instances": 0,
                    "max_gpu_seconds": 0,
                    "max_instance_seconds": 0,
                },
            ),
            "verifier": {
                "max_train_calls": quota["max_train_calls"],
                "max_train_wall_seconds": quota["max_train_wall_seconds"],
                "max_gpu_instances": gpu.get("verifier", {}).get(
                    "max_gpu_instances", 0
                ),
                "max_gpu_seconds": gpu.get("verifier", {}).get("max_gpu_seconds", 0),
                "eval_timeout_seconds": quota["eval_timeout_seconds"],
            },
        },
        "submission": {
            "max_bytes": policy["submission"]["max_bytes"],
            "media_type": "application/x-tar",
            "required_sha256": True,
            "entrypoint": policy["submission"]["entrypoint"],
        },
        "datasets": {
            "train": {
                "uri": task_dataset_uri(
                    problem, "train", train_sha256, definitions=TASK_DEFINITIONS
                ),
                "version": train_sha256,
            },
            "eval": {
                "uri": task_dataset_uri(
                    problem, "eval", eval_sha256, definitions=TASK_DEFINITIONS
                ),
                "version": eval_sha256,
            },
        },
        "verifier": {
            "command": verifier["command"],
            "result_path": "/frontier/output/result.json",
            "evidence_path": verifier.get(
                "evidence_path", "/frontier/output/evidence.tar"
            ),
            "cpu_millis": verifier["cpu_millis"],
            "memory_bytes": verifier["memory_bytes"],
            "max_evidence_bytes": verifier["max_evidence_bytes"],
        },
        "network": {
            "allowed_domains": ["openrouter.ai"],
            "audit": {
                "max_model_sessions": 256,
                "max_ssh_sessions": 32,
                "max_model_request_bytes": 1048576,
                "max_model_response_bytes": 8388608,
                "max_ssh_bytes_per_direction": 8388608,
                "max_bundle_bytes": 67108864,
            },
        },
    }
    if "workspace_bytes" in verifier:
        task_config["verifier"]["workspace_bytes"] = verifier["workspace_bytes"]
    if "agent_gpu_image" in task:
        image = task["agent_gpu_image"]
        task_config["images"]["agent_gpu"] = {
            "tag": f"{image['repository']}:{image['tag']}",
            "digest": image["digest"],
        }
    if "verifier_gpu_image" in task:
        image = task["verifier_gpu_image"]
        task_config["images"]["verifier"].update(
            {
                "gpu_tag": f"{image['repository']}:{image['tag']}",
                "gpu_digest": image["digest"],
            }
        )
    if "sidecar_images" in task:
        # Declared same-host sidecars: the Driver starts these task-owned
        # images beside the GPU runner of each listed phase (Core
        # TaskConfig.images.gpu_sidecars). Modal pulls them publicly like the
        # GPU images, so they are not mirrored into the cluster registry.
        task_config["images"]["gpu_sidecars"] = {}
        for sidecar, profile in sorted(policy["sidecars"].items()):
            image = task["sidecar_images"][profile.get("image", sidecar)]
            rendered = {
                "image": {
                    "tag": f"{image['repository']}:{image['tag']}",
                    "digest": image["digest"],
                },
                "args": list(profile["args"]),
                "workdir": profile["workdir"],
                "phases": list(profile["phases"]),
            }
            if profile.get("attestation_key"):
                rendered["attestation_key"] = True
            task_config["images"]["gpu_sidecars"][sidecar] = rendered
    if "gpu_parent_resources" in policy:
        task_config["images"]["gpu_parent_resources"] = json.loads(
            json.dumps(policy["gpu_parent_resources"])
        )
    if "network" in policy:
        task_config["network"] = network
    else:
        task_config["network"]["allowed_domains"] = network["allowed_domains"]
    if "additional_keys" in policy:
        # Task-model keys (FrontierCS2 TaskConfig.additional_keys). Omitted when
        # absent so every existing task keeps its definition digest.
        task_config["additional_keys"] = json.loads(
            json.dumps(renderer.validate_additional_keys(policy["additional_keys"]))
        )
    definition_digest = canonical_object_sha256(task_config)
    source = policy["source"]
    return {
        "task_id": policy["task_id"],
        "display_name": policy["display_name"],
        "family": policy["family"],
        # Content-addressed like the candidate version: a config change (e.g.
        # image references) yields a distinct immutable (task_id, version).
        "version": f"{policy['version']}-cfg-{definition_digest.removeprefix('sha256:')[:12]}",
        "definition_ref": (
            f"s3://frontier-local-task-data/definitions/frontier-cs-2.0-demo/{problem}/"
            f"{definition_digest.removeprefix('sha256:')}.json"
        ),
        "definition_digest": definition_digest,
        "driver_config": task_config,
        "metadata": {
            **policy.get("metadata", {}),
            "frontier_cs_2_0": {
                "policy_schema_version": policy_schema_version,
                "problem_id": problem,
                "adapter_version": policy["version"],
                **source,
            },
        },
    }


def candidate_version_document(
    candidate: Mapping[str, Any],
    mirrored_overlay: Mapping[str, str] | None = None,
    *,
    execution: Mapping[str, Any] | None = None,
    identity: Mapping[str, Any] | None = None,
    deployment_registry: str | None = None,
) -> dict[str, Any]:
    overlay = candidate["overlay_image"]
    if deployment_registry is not None:
        if (
            not deployment_registry
            or "/" in deployment_registry
            or deployment_registry.split(":", 1)[0] in {"localhost", "127.0.0.1"}
        ):
            raise DeployError(
                "deployment registry must be a Driver-reachable registry authority"
            )
        if mirrored_overlay is None:
            raise DeployError("local deployment requires a candidate overlay mirror")
        reference = require_string(
            mirrored_overlay.get("reference"), "mirrored overlay reference"
        )
        if reference.split("/", 1)[0] != deployment_registry:
            raise DeployError(
                "candidate overlay mirror differs from the deployment registry"
            )
    if mirrored_overlay is not None:
        digest = require_string(
            mirrored_overlay.get("digest"), "mirrored overlay digest"
        )
        reference = require_string(
            mirrored_overlay.get("reference"), "mirrored overlay reference"
        )
        if not DIGEST_RE.fullmatch(digest) or not reference.endswith("@" + digest):
            raise DeployError("mirrored candidate overlay is not digest pinned")
        # As with task images, the Driver composes from the in-cluster mirror.
        overlay = {
            **overlay,
            "repository": reference.split("@", 1)[0],
            "tag": None,
            "digest": digest,
            "reference": reference,
        }
    overlay_reference = (
        overlay.get("reference") or f"{overlay['repository']}:{overlay['tag']}"
    )
    overlay_version_base = (
        overlay.get("tag") or overlay["digest"].removeprefix("sha256:")[:12]
    )
    execution = (
        execution
        if execution is not None
        else {
            "entrypoint": ["/opt/frontier/candidate/launch"],
            "args": [],
            "working_dir": "/workspace",
            "user": "1000:1000",
            "env": {},
        }
    )
    # The seeded Codex candidate connects to its model provider (OpenRouter by
    # default) directly from the Agent (site NetworkPolicy grants the egress), so
    # no platform model API — direct-proxy mode would demand a
    # Driver-mounted provider secret and the dispatch would never form.
    model_api = {"allowed_domains": [], "secret_refs": []}
    # Seed the immutable identity from every field that shapes the config, so
    # a template change (e.g. the model-API posture) yields a new version
    # rather than colliding with an existing immutable record.
    seed_inputs = {
        "name": candidate["name"],
        "overlay_image": {
            "reference": overlay_reference,
            "digest": overlay["digest"],
        },
        "execution": execution,
        "model_api": model_api,
    }
    if identity is not None:
        require_exact_keys(
            identity,
            {"candidate_id", "display_name", "version", "metadata"},
            "candidate identity",
        )
        seed_inputs["identity"] = identity
    candidate_seed = canonical_json(seed_inputs)
    candidate_version_id = "candver_" + stable_ulid(candidate_seed)
    candidate_config = {
        "schema_version": "frontier.candidate-config/v1",
        "candidate_version_id": candidate_version_id,
        "overlay_image": {
            "reference": overlay_reference,
            "digest": overlay["digest"],
            "format": "frontier.oci-overlay/v1",
            "platform": {"os": "linux", "architecture": "amd64"},
        },
        "execution": execution,
        "model_api": model_api,
    }
    definition_digest = canonical_object_sha256(candidate_config)
    return {
        "candidate_id": identity["candidate_id"]
        if identity is not None
        else "codex-preview",
        "display_name": identity["display_name"]
        if identity is not None
        else "Codex Preview",
        # The version is content-addressed so a candidate-config change (overlay
        # digest or template) yields a distinct immutable (candidate_id, version)
        # rather than colliding with an existing record.
        "version": f"{identity['version'] if identity is not None else overlay_version_base}-cfg-{definition_digest.removeprefix('sha256:')[:12]}",
        "definition_ref": (
            (
                f"preview://{identity['candidate_id']}/definitions/"
                if identity is not None
                else "preview://codex/definitions/"
            )
            + definition_digest.removeprefix("sha256:")
            + ".json"
        ),
        "definition_digest": definition_digest,
        "driver_config": candidate_config,
        "metadata": identity["metadata"]
        if identity is not None
        else {
            "runtime": "generic-codex-overlay",
            "model_provider": "openrouter",
        },
    }


def seed_tasks(
    config: Mapping[str, Any],
    kubeconfig: Path,
    catalog_path: Path = TASK_CATALOG_PATH,
    *,
    check: bool = False,
) -> list[dict[str, Any]]:
    TASK_DEFINITIONS = load_task_definitions()
    catalog, selection = load_task_selection(
        config, catalog_path, require_resolved=True
    )
    if check:
        return check_seed_tasks(config, kubeconfig, catalog, selection)
    mirrored = mirror_task_images(config, kubeconfig, catalog, selection=selection)
    runner_image = load_runtime_images(config)["storage_controller"]["reference"]
    environment = kubernetes_environment(kubeconfig)
    kubectl = executable("kubectl")
    published: list[dict[str, Any]] = []
    task_bindings = {}
    with port_forward(kubeconfig, MANAGEMENT_SERVICE, 8080) as port:
        base = f"http://127.0.0.1:{port}"
        candidate_document = candidate_version_document(
            catalog["candidate"],
            mirrored["candidate"]["overlay_image"],
            deployment_registry=f"{config['network']['registry_cluster_ip']}:5000",
        )
        candidate_version_id = candidate_document["driver_config"][
            "candidate_version_id"
        ]
        created_candidate = ensure_version(
            base,
            "/api/v1/candidate-versions",
            candidate_version_id,
            candidate_document,
        )
        if created_candidate.get("id") != candidate_version_id:
            raise DeployError(
                "Management returned the wrong Codex CandidateVersion identity"
            )
        print(f"candidate seed: codex -> {candidate_version_id}")
        for task in catalog["tasks"]:
            problem = task["name"]
            content_name = task_dataset_seed_job_name(
                task, mirrored[problem]["dataset_image"]["digest"]
            )
            prefix, content_hash = content_name.rsplit("-", 1)
            job_name = f"{prefix[:29]}-{content_hash}-{secrets.token_hex(8)}"
            manifest_path = state_directory(config) / "task-seeds" / f"{job_name}.json"
            atomic_write(
                manifest_path,
                render_task_dataset_seed_manifest(
                    config, task, mirrored[problem], runner_image, job_name=job_name
                ),
                mode=0o600,
            )
            # Each invocation revalidates the PVC bytes in a fresh Job. Create
            # refuses even an accidental name collision; never modify an old Job.
            run_command(
                [kubectl, "-n", PLATFORM_NAMESPACE, "create", "-f", str(manifest_path)],
                environment=environment,
                timeout=180,
            )
            run_command(
                [
                    kubectl,
                    "-n",
                    PLATFORM_NAMESPACE,
                    "wait",
                    "--for=condition=complete",
                    f"job/{job_name}",
                    "--timeout=30m",
                ],
                environment=environment,
                timeout=1860,
            )
            document = task_version_document(
                task,
                mirrored[problem],
                model_domains=selection.model_domains,
                definitions=selection.definitions,
            )
            seed_log = run_command(
                [kubectl, "-n", PLATFORM_NAMESPACE, "logs", f"job/{job_name}"],
                environment=environment,
                capture=True,
                timeout=120,
            )
            sizes = json.loads(seed_log.strip())["dataset_sizes"]
            document["author_receipt"] = local_author_receipt(
                document,
                sizes,
                TASK_DEFINITIONS[problem]["source"],
            )
            version_id = document["driver_config"]["task_version_id"]
            created = ensure_version(
                base, "/api/v1/task-versions", version_id, document
            )
            if created.get("id") != version_id:
                raise DeployError(
                    f"Management returned the wrong TaskVersion identity for {problem}"
                )
            published.append(created)
            task_bindings[problem] = {
                "task_version_id": version_id,
                "definition_digest": document["definition_digest"],
            }
            print(f"task seed: {problem} -> {version_id}")
    receipt = {
        "schema_version": "frontier.deployment-task-seed-receipt/v2",
        "site": config["site"]["name"],
        **{
            key: getattr(selection, key)
            for key in (
                "catalog_sha256",
                "effective_catalog_sha256",
                "selection_digest",
                "selected",
                "excluded",
                "model_domains",
            )
        },
        "tasks": task_bindings,
        "candidate_version_id": candidate_version_id,
        "task_version_ids": [item["id"] for item in published],
    }
    atomic_write(
        state_directory(config) / "task-seed-receipt.json",
        canonical_json(receipt),
        mode=0o600,
    )
    return published


def check_seed_tasks(
    config: Mapping[str, Any],
    kubeconfig: Path,
    catalog: Mapping[str, Any],
    selection: TaskSelection,
) -> list[dict[str, Any]]:
    """Verify existing identities using saved mirrors; never mirror, seed or write state."""
    mirrored = load_json(state_directory(config) / "task-images.json")
    endpoint = f"{config['network']['registry_cluster_ip']}:5000"
    for name, records in mirrored.items():
        for kind, record in require_mapping(records, f"saved mirrors {name}").items():
            if kind in {"agent_gpu_image", "verifier_gpu_image", "sidecar_images"}:
                # mirror_task_images leaves these public Modal images unmirrored.
                continue
            record = require_mapping(record, f"saved mirror {name}/{kind}")
            if name == "candidate":
                repository = "frontier/candidates/codex"
            elif name == "reference_candidates":
                repository = f"frontier/candidates/reference-{kind.replace('_', '-')}"
            else:
                image_kind = kind.removesuffix("_image").replace("_", "-")
                repository = f"frontier/tasks/{name.replace('_', '-')}/{image_kind}"
            digest = record.get("digest")
            if (
                not isinstance(digest, str)
                or not DIGEST_RE.fullmatch(digest)
                or record.get("reference") != f"{endpoint}/{repository}@{digest}"
            ):
                raise DeployError(
                    f"saved mirror destination differs from this site's registry: {name}/{kind}"
                )
    for name, sources, records in [
        (
            "candidate",
            {"overlay_image": catalog["candidate"]["overlay_image"]},
            mirrored.get("candidate", {}),
        ),
        *(
            (
                task["name"],
                {
                    key: task[key]
                    for key in ("agent_image", "verifier_image", "dataset_image")
                },
                mirrored.get(task["name"], {}),
            )
            for task in catalog["tasks"]
        ),
    ]:
        for kind, source in sources.items():
            record = require_mapping(records.get(kind), f"saved mirror {name}/{kind}")
            if (
                record.get("source_sha256") != source["digest"]
                or record.get("source_reference") != source["reference"]
            ):
                raise DeployError(
                    f"saved mirror differs from current catalog: {name}/{kind}"
                )
    candidate = candidate_version_document(
        catalog["candidate"], mirrored["candidate"]["overlay_image"]
    )
    documents = [
        ("codex", "/api/v1/candidate-versions", "candidate_version_id", candidate)
    ]
    documents.extend(
        (
            task["name"],
            "/api/v1/task-versions",
            "task_version_id",
            task_version_document(
                task,
                mirrored[task["name"]],
                model_domains=selection.model_domains,
                definitions=selection.definitions,
            ),
        )
        for task in catalog["tasks"]
    )
    results = []
    with port_forward(kubeconfig, MANAGEMENT_SERVICE, 8080) as port:
        for name, path, id_key, document in documents:
            version_id = document["driver_config"][id_key]
            existing = find_version(f"http://127.0.0.1:{port}", path, version_id)
            if existing is None:
                status = "MISSING"
            else:
                status = (
                    "MATCH"
                    if (
                        existing.get("id") == version_id
                        and existing.get("definition_digest")
                        == document["definition_digest"]
                    )
                    else "DIFFERS"
                )
            results.append(
                {
                    "id": version_id,
                    "definition_digest": document["definition_digest"],
                    "status": status,
                }
            )
            print(f"seed check: {name} -> {version_id}: {status}")
        runtime_admission_preflight(
            config, f"http://127.0.0.1:{port}", informational=True
        )
    if any(result["status"] != "MATCH" for result in results):
        raise DeployError("seed check failed: existing versions are missing or differ")
    return results


def receipt_images(images: Mapping[str, Any]) -> dict[str, Any]:
    """Project TaskConfig images onto the author-receipt image shape.

    Author receipts bind published images only: each declared sidecar by name
    to its exact tag/digest; commands, phases and parent resources are
    definition-digest material. Management checks the receipt against exactly
    this projection (Core dataset_authorization), so native documents carry it
    too.
    """
    result = json.loads(json.dumps(images))
    result["agent_cpu_base"].pop("platform", None)
    result.pop("gpu_parent_resources", None)
    if "gpu_sidecars" in result:
        result["gpu_sidecars"] = {
            name: profile["image"] for name, profile in result["gpu_sidecars"].items()
        }
    return result


def local_author_receipt(
    document: Mapping[str, Any],
    sizes: Mapping[str, int],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind reviewed source, exact config/images and independently read EFS bytes.

    local-sha256 is explicitly a local-k8s content version, never an S3 VersionId.
    Management reopens and verifies the read-only files before accepting it.
    """
    config = document["driver_config"]
    images = receipt_images(config["images"])
    datasets = {}
    for mode, dataset in config["datasets"].items():
        size = sizes.get(dataset["version"].removeprefix("sha256:"))
        if type(size) is not int or size <= 0:
            raise DeployError(f"missing verified dataset size: {mode}")
        datasets[mode] = {
            "uri": dataset["uri"],
            "body_sha256": dataset["version"],
            "s3_version_id": "local-" + dataset["version"],
            "size_bytes": size,
        }
    return {
        "schema_version": "frontier.task-author-release/v1",
        "task_version_id": config["task_version_id"],
        "source": {
            "reference": "preview-source-manifest:" + canonical_object_sha256(source),
            "sha256": source.get("source_set_sha256", canonical_object_sha256(source)),
        },
        "build": {
            "platform": {"os": "linux", "architecture": "amd64"},
            "inputs_sha256": canonical_object_sha256(
                {"source": source, "images": images}
            ),
            "tool_versions": {"frontier-preview-publisher": "2"},
        },
        "definition": {
            "ref": document["definition_ref"],
            "digest": document["definition_digest"],
        },
        "images": images,
        "datasets": datasets,
    }


def find_version(base: str, endpoint: str, version_id: str) -> dict[str, Any] | None:
    existing = api_json(base, "GET", endpoint)
    if not isinstance(existing, list):
        raise DeployError(f"Management version list is malformed: {endpoint}")
    matches = [item for item in existing if item.get("id") == version_id]
    if len(matches) > 1:
        raise DeployError(f"Management returned duplicate version {version_id}")
    return matches[0] if matches else None


def ensure_version(
    base: str, endpoint: str, version_id: str, body: Mapping[str, Any]
) -> dict[str, Any]:
    existing = find_version(base, endpoint, version_id)
    if existing is not None:
        for key, value in body.items():
            if key == "author_receipt":
                # The publication-only receipt is not part of the public Task view.
                continue
            server_value = existing.get(key)
            if key == "driver_config" and server_value is None:
                # Management's list view omits the full config document;
                # definition_digest is its canonical content hash and is
                # compared like every other field, so equality still holds.
                continue
            if server_value != value:
                raise DeployError(f"existing immutable version differs: {version_id}")
        return existing
    try:
        created = api_json(base, "POST", endpoint, body)
    except DeployError as error:
        # A concurrent or prior identical publish can win the race; treat an
        # already-exists conflict as success when the stored record matches.
        if "409" not in str(error):
            raise
        refreshed = api_json(base, "GET", endpoint)
        remote = [
            item
            for item in refreshed
            if isinstance(refreshed, list) and item.get("id") == version_id
        ]
        if len(remote) == 1 and remote[0].get("definition_digest") == body.get(
            "definition_digest"
        ):
            return remote[0]
        raise
    if not isinstance(created, dict) or created.get("id") != version_id:
        raise DeployError(
            f"Management did not return expected version {version_id}: {endpoint}"
        )
    return created


# Reference catalog descriptors and authoring baselines.
MANIFEST = REPOSITORY_ROOT / "config/task-catalog.json"
LAUNCHER = (
    REPOSITORY_ROOT
    / "tasks/frontier-cs-2.0-demo/shared/reference-candidate/fcs2_reference_candidate.py"
)
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
PLACEHOLDER = "sha256:" + "0" * 64
RULE = {"name": "cross-site-agreement/v1", "abs": 0.01, "rel": 0.05, "replays": 1}


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def validate_reference(name, image, definition, *, root=None, require_resolved=False):
    root = root or REPOSITORY_ROOT
    if image is None:
        return
    if not isinstance(image, dict) or set(image) != {
        "repository",
        "tag",
        "digest",
        "launcher_sha256",
        "payload_sha256",
    }:
        raise ValueError(f"invalid reference manifest fields: {name}")
    for key in ("digest", "launcher_sha256", "payload_sha256"):
        if not isinstance(image[key], str) or DIGEST.fullmatch(image[key]) is None:
            raise ValueError(f"invalid reference {key}: {name}")
    launcher = (
        root
        / "tasks/frontier-cs-2.0-demo/shared/reference-candidate/fcs2_reference_candidate.py"
    )
    patch = (
        renderer.problem_directory(root / "tasks/frontier-cs-2.0-demo/problems", name)
        / "judge/reference.patch"
    )
    if (
        not isinstance(image["repository"], str)
        or not re.fullmatch(
            r"docker.io/[a-z0-9]+(?:[._-][a-z0-9]+)*/fcs2-task-"
            + re.escape(name)
            + r"-reference",
            image["repository"],
        )
        or not source_image_tag_matches(
            image["tag"], definition["image_tag"], image["digest"]
        )
        or image["launcher_sha256"] != file_sha(launcher)
        or (patch.exists() and image["payload_sha256"] != file_sha(patch))
    ):
        raise ValueError(f"reference manifest differs from native release: {name}")
    if require_resolved and image["digest"] == PLACEHOLDER:
        raise ValueError(f"reference image digest is a placeholder: {name}")


def validate_baseline(name, baseline, image, definition, *, root=None):
    root = root or REPOSITORY_ROOT
    if (
        not isinstance(baseline, dict)
        or baseline.get("provenance")
        not in ("measured", "historical-authoring", "unverified")
        or type(baseline.get("comparable")) is not bool
    ):
        raise ValueError(
            f"baselines must be classified with valid provenance and boolean comparable: {name}"
        )
    if "comparability_note" in baseline and (
        not isinstance(baseline["comparability_note"], str)
        or not baseline["comparability_note"].strip()
    ):
        raise ValueError(f"invalid comparability_note: {name}")
    if baseline["provenance"] == "unverified" and baseline["comparable"]:
        raise ValueError(f"unverified baseline cannot be comparable: {name}")
    if not baseline["comparable"] and "historical_mean" not in baseline:
        if baseline["provenance"] != "unverified":
            raise ValueError(f"baseline without scores must be unverified: {name}")
        return
    if baseline.get("agreement_rule") != RULE:
        raise ValueError(f"unsupported agreement rule: {name}")
    for key in ("historical_mean", "historical_spread"):
        if type(baseline.get(key)) not in (int, float) or not math.isfinite(
            baseline[key]
        ):
            raise ValueError(f"invalid baseline {key}: {name}")
    if (
        baseline.get("source_set_sha256") != definition["source"]["source_set_sha256"]
        or any(
            baseline.get("datasets", {}).get(mode)
            != definition["datasets"][f"{mode}_sha256"]
            for mode in ("train", "eval")
        )
        or (
            image is not None
            and baseline.get("reference_patch_sha256") != image["payload_sha256"]
        )
    ):
        raise ValueError(f"baseline does not bind the native release: {name}")
    task_root = renderer.problem_directory(
        root / "tasks/frontier-cs-2.0-demo/problems", name
    )
    patch = task_root / "judge/reference.patch"
    if patch.is_file() and baseline.get("reference_patch_sha256") != file_sha(patch):
        raise ValueError(
            f"baseline reference_patch_sha256 differs from the bundled task: {name}"
        )
    adapter = (
        "verifier/fcs2_verifier.py"
        if (task_root / "verifier/fcs2_verifier.py").is_file()
        else "verifier/verify.py"
    )
    for key, path in (
        ("evaluator_sha256", "judge/evaluator.py"),
        ("verifier_adapter_sha256", adapter),
    ):
        if key not in baseline and baseline["provenance"] == "unverified":
            continue
        if baseline.get(key) != file_sha(task_root / path):
            raise ValueError(f"baseline {key} differs from the bundled task: {name}")


def load_manifest(
    path: Path | None = None,
    *,
    require_resolved=False,
    catalog=None,
    definitions=None,
    root=None,
):
    raw = catalog if catalog is not None else load_json(path or MANIFEST)
    if raw.get("schema_version") != "frontier.deployment-task-catalog/v2":
        raise ValueError("reference manifest requires catalog v2")
    definitions = (
        definitions
        if definitions is not None
        else load_json((root or REPOSITORY_ROOT) / "config/task-definitions.json")
    )
    result, seen = {}, set()
    if not isinstance(raw.get("tasks"), list) or not raw["tasks"]:
        raise ValueError("reference catalog requires a nonempty task list")
    for task in raw["tasks"]:
        name, image = task.get("name"), task.get("reference")
        if (
            not isinstance(name, str)
            or name not in definitions
            or name in seen
            or "reference" not in task
        ):
            raise ValueError(
                "reference catalog task is unknown, duplicated or missing reference"
            )
        seen.add(name)
        validate_reference(
            name, image, definitions[name], root=root, require_resolved=require_resolved
        )
        if image is not None:
            result[name] = image
    return result


def load_baselines(path: Path | None = None):
    catalog = load_json(path or MANIFEST)
    if catalog.get("schema_version") != "frontier.deployment-task-catalog/v2":
        raise ValueError("baselines require catalog v2")
    return {task["name"]: task["baseline"] for task in catalog["tasks"]}


def baseline_from_result_json(
    root: Path, definition: dict, payload: Path | None = None
) -> dict:
    result_path = root / "judge/result.json"
    if not result_path.is_file():
        return {
            "provenance": "unverified",
            "comparable": False,
            "comparability_note": "no judge/result.json in the export",
        }
    result = load_json(result_path)
    if not isinstance(result, dict):
        raise ValueError("judge/result.json must be an object")
    if (
        result.get("schema") != "frontiersmith/result@1"
        or result.get("task_id") != root.name
    ):
        raise ValueError(
            "judge/result.json schema or task_id does not match the export"
        )
    final = result.get("final")
    reference = final.get("reference") if isinstance(final, dict) else None
    if not isinstance(reference, dict) or not isinstance(reference.get("spread"), dict):
        raise ValueError(
            "judge/result.json requires final.reference and spread objects"
        )
    scores, spread = reference.get("scores"), reference["spread"]
    if (
        reference.get("ok") is not True
        or not isinstance(scores, list)
        or not scores
        or any(
            type(score) not in (int, float) or not math.isfinite(score)
            for score in scores
        )
        or type(reference.get("samples")) is not int
        or reference["samples"] != len(scores)
        or type(spread.get("n")) is not int
        or spread["n"] != len(scores)
    ):
        raise ValueError("judge/result.json needs successful finite reference samples")
    for key, expected in (
        ("mean", sum(scores) / len(scores)),
        ("spread", max(scores) - min(scores)),
    ):
        value = spread.get(key)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(f"judge/result.json inconsistent reference {key}")
    relative = spread.get("rel")
    expected_relative = spread["spread"] / max(abs(spread["mean"]), 1e-12)
    if (
        type(relative) not in (int, float)
        or not math.isfinite(relative)
        or not math.isclose(relative, expected_relative, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError("judge/result.json inconsistent reference rel")
    adapter = root / "verifier/fcs2_verifier.py"
    if not adapter.is_file():
        adapter = root / "verifier/verify.py"
    evaluator = root / "judge/evaluator.py"
    payload = payload or (root / "judge/reference.patch")
    baseline = {
        "agreement_rule": dict(RULE),
        "historical_scores": scores,
        "historical_mean": spread["mean"],
        "historical_spread": spread["spread"],
        "datasets": {
            mode: definition["datasets"][f"{mode}_sha256"] for mode in ("train", "eval")
        },
        "verifier_adapter_sha256": file_sha(adapter),
        "source_set_sha256": definition["source"]["source_set_sha256"],
        "result_json_source": "judge/result.json",
        "provenance": "historical-authoring",
        "comparable": False,
        "comparability_note": "result.json does not attest that these scores measured the exported input identity",
    }
    missing = []
    for key, path in (
        ("evaluator_sha256", evaluator),
        ("reference_patch_sha256", payload),
    ):
        if path.is_file():
            baseline[key] = file_sha(path)
        else:
            missing.append(path.relative_to(root).as_posix())
    if missing:
        baseline.update(
            provenance="unverified",
            comparability_note="missing baseline binding files: " + ", ".join(missing),
        )
    return baseline
