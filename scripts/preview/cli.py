"""Guarded fresh-account installer for the executable FrontierCS2 Preview profile.

The site file is deliberately non-secret. AWS CLI, Terraform, Docker, and
kubectl receive credentials through their ordinary local credential chains.
"""

from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from task_selection import TaskSelection, requirements
from task_selection import select_tasks as select_tasks

from .acceptance import (
    run_gpu_smoke,
    run_smoke,
    runtime_admission_preflight,
    verify_runtime,
)
from .api_routes import (
    api_policy_patch,
    verify_api_route_reader_rbac,
    verify_api_routes,
)
from .config import (
    read_and_validate,
    render_bundle,
    state_directory,
    validate_verifier_node_allowlists,
    validate_verifier_placement_required_task_ids,
)
from .unchanged_management import (
    unchanged_native_management, require_driver_only_patches, require_unchanged_observer,
)
from .gateway_preservation import read_gateway_snapshot
from .runtime_reference_compat import (
    preserve_reviewed_public_images,
    require_reviewed_route_only_patches,
    reviewed_retention_transition,
)
from .images import (
    build_runtime_images,
    cached_registry_image,
    load_runtime_images,
    source_tree_sha256,
)
from .infra import (
    DIGEST_RE,
    MANAGEMENT_SERVICE,
    PLATFORM_NAMESPACE,
    REGISTRY_SERVICE,
    REPOSITORY_ROOT,
    TASK_CATALOG_PATH,
    DeployError,
    api_cidrs,
    api_json,
    atomic_write,
    aws_identity,
    canonical_json,
    canonical_object_sha256,
    command_environment,
    dataset_seed_names,
    deploy_bootstrap,
    deploy_litellm,
    ensure_admission_control_maps,
    executable,
    gateway_artifact_retention_init,
    kubectl_apply,
    kubernetes_environment,
    load_json,
    management_api_routes_rbac,
    management_diagnostic_source_mount,
    management_diagnostics_rbac,
    management_result_resources,
    modal_observer_rbac,
    platform_labels,
    platform_outputs,
    port_forward,
    render_runtime_manifest,
    run_command,
    runtime_source_commit,
    state_bucket_exists,
    terraform_apply_plan,
    terraform_init_and_validate,
    terraform_plan,
    update_kubeconfig,
)
from .publication import (
    load_task_selection,
    seed_tasks,
)


def modal_credentials(
    config: Mapping[str, Any], selection: TaskSelection | None = None
) -> dict[str, str]:
    if selection is None:
        _, selection = load_task_selection(config)
    if not any(
        requirements(definition)["modal_gpu"]
        for definition in selection.definitions.values()
    ):
        return {}
    data = {}
    for variable, key in (
        ("MODAL_TOKEN_ID", "token-id"),
        ("MODAL_TOKEN_SECRET", "token-secret"),
    ):
        value = os.environ.get(variable, "").strip()
        if not value or "\n" in value or "\r" in value:
            raise DeployError(
                f"GPU Preview requires {variable} in the operator environment"
            )
        data[key] = base64.b64encode(value.encode()).decode()
    return data


def deploy_modal_credentials(config: Mapping[str, Any], kubeconfig: Path) -> None:
    data = modal_credentials(config)
    if not data:
        return
    name = config["modal"]["credentials_secret"]
    command = [executable("kubectl"), "-n", PLATFORM_NAMESPACE]
    environment = command_environment(kubernetes_environment(kubeconfig))
    # Capture and suppress API error bodies: they may echo a submitted Secret.
    existing = subprocess.run(
        command + ["get", "secret", name, "--ignore-not-found", "-o", "json"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if existing.returncode:
        raise DeployError("Could not inspect the named platform Modal Secret")
    if existing.stdout.strip():
        if json.loads(existing.stdout).get("data") != data:
            raise DeployError(
                "Platform Modal Secret differs; rotate credentials after active Runs finish"
            )
        return
    document = {
        "apiVersion": "v1",
        "kind": "Secret",
        "type": "Opaque",
        "metadata": {
            "name": name,
            "namespace": PLATFORM_NAMESPACE,
            "labels": platform_labels(config, "frontier-modal-credentials"),
        },
        "data": data,
    }
    created = subprocess.run(
        command + ["create", "-f", "-"],
        env=environment,
        input=canonical_json(document),
        capture_output=True,
        timeout=60,
        check=False,
    )
    if created.returncode:
        raise DeployError("Could not create the named platform Modal Secret")
    readback = subprocess.run(
        command + ["get", "secret", name, "-o", "json"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if readback.returncode or json.loads(readback.stdout).get("data") != data:
        raise DeployError("Platform Modal Secret readback failed")


def require_idle_runtime(base: str) -> None:
    # Filter server-side so a paginated history cannot hide a live Run.
    for state in ("QUEUED", "STARTING", "RUNNING", "FINALIZING"):
        rows = api_json(base, "GET", f"/api/v1/runs?state={state}&limit=1")
        if not isinstance(rows, list) or rows:
            raise DeployError(f"runtime upgrade requires zero {state} Runs")


def runtime_upgrade_patches(
    configmap: Mapping[str, Any],
    deployment: Mapping[str, Any],
    images: Mapping[str, Any],
    *,
    litellm_cidr: str | None = None,
    gateway_artifact_retention: bool = False,
    kubernetes_api_cidrs: Sequence[str] | None = None,
    cluster_name: str | None = None,
    region: str | None = None,
    large_verifier_results: bool = False,
    verifier_node_allowlists: list | None = None,
    verifier_placement_required_task_ids: list | None = None,
    reviewed_public_repair: bool = False,
) -> tuple[list, list, dict]:
    """Update immutable runtime pins and the optional Preview gateway wiring."""
    data = dict(configmap["data"])
    if data.get("MANAGEMENT_DRIVER_BACKENDS") != "local-k8s":
        raise DeployError(
            "runtime upgrade requires the existing local-k8s Preview profile"
        )
    containers = deployment["spec"]["template"]["spec"]["containers"]
    indices = [i for i, c in enumerate(containers) if c["name"] == "management"]
    if len(indices) != 1:
        raise DeployError("runtime upgrade requires one Management container")
    data["MANAGEMENT_DRIVER_IMAGE"] = images["driver"]["reference"]
    cm_patch = [
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": configmap["metadata"]["resourceVersion"],
        },
        {
            "op": "replace",
            "path": "/data/MANAGEMENT_DRIVER_IMAGE",
            "value": data["MANAGEMENT_DRIVER_IMAGE"],
        },
    ]
    if verifier_node_allowlists is not None:
        validate_verifier_node_allowlists(verifier_node_allowlists)
        key = "MANAGEMENT_VERIFIER_NODE_ALLOWLISTS"
        if any(entry.get("name") == key for entry in containers[indices[0]].get("env", [])):
            raise DeployError("verifier placement rejects direct environment overrides")
        if verifier_node_allowlists:
            value = json.dumps(verifier_node_allowlists, sort_keys=True, separators=(",", ":"))
            data[key] = value
            cm_patch.append({"op": "add", "path": "/data/" + key, "value": value})
        elif key in data:
            del data[key]
            cm_patch.append({"op": "remove", "path": "/data/" + key})
    if verifier_placement_required_task_ids is not None:
        validate_verifier_placement_required_task_ids(verifier_placement_required_task_ids)
        key = "MANAGEMENT_VERIFIER_PLACEMENT_REQUIRED_TASK_IDS"
        if any(entry.get("name") == key for entry in containers[indices[0]].get("env", [])):
            raise DeployError("required verifier placement rejects direct environment overrides")
        if verifier_placement_required_task_ids:
            value = json.dumps(sorted(verifier_placement_required_task_ids), separators=(",", ":"))
            data[key] = value
            cm_patch.append({"op": "add", "path": "/data/" + key, "value": value})
        elif key in data:
            del data[key]
            cm_patch.append({"op": "remove", "path": "/data/" + key})
    if cluster_name is not None or region is not None:
        if (not isinstance(cluster_name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', cluster_name)
                or not isinstance(region, str) or not re.fullmatch(r'[a-z]{2}(?:-[a-z]+)+-[0-9]', region)
                or data.get('MANAGEMENT_DRIVER_MODE') != 'kubernetes'):
            raise DeployError('Infrastructure observation requires a cluster, region and kubernetes mode')
        settings = {'MANAGEMENT_EKS_STATUS_CLUSTER': cluster_name,
                    'MANAGEMENT_EKS_STATUS_REGION': region}
        legacy = ('MANAGEMENT_EKS_STATUS_PATH', 'MANAGEMENT_EKS_STATUS_NODEGROUPS',
                  'MANAGEMENT_EKS_STATUS_MAX_AGE_SECONDS')
        if any(entry.get('name') in {*settings, *legacy}
               for entry in containers[indices[0]].get('env', [])):
            raise DeployError('Infrastructure observation rejects direct environment overrides')
        for key in legacy:
            if key in data:
                del data[key]
                cm_patch.append({'op': 'remove', 'path': '/data/' + key})
        for key, value in settings.items():
            data[key] = value
            cm_patch.append({'op': 'add', 'path': '/data/' + key, 'value': value})
    if kubernetes_api_cidrs is not None:
        if not kubernetes_api_cidrs or len(kubernetes_api_cidrs) > 64:
            raise DeployError(
                "Kubernetes API routes require 1-64 exact IPv4 destinations"
            )
        for cidr in kubernetes_api_cidrs:
            network = ipaddress.ip_network(cidr, strict=True)
            if network.version != 4 or network.prefixlen != 32:
                raise DeployError("Kubernetes API routes must be exact IPv4 /32s")
        data["MANAGEMENT_KUBERNETES_API_CIDRS"] = ",".join(
            sorted(set(kubernetes_api_cidrs))
        )
        cm_patch.append(
            {
                "op": "add",
                "path": "/data/MANAGEMENT_KUBERNETES_API_CIDRS",
                "value": data["MANAGEMENT_KUBERNETES_API_CIDRS"],
            }
        )
    if litellm_cidr is not None:
        network = ipaddress.ip_network(litellm_cidr, strict=True)
        if (
            network.version != 4
            or network.prefixlen != 32
            or not network.network_address.is_private
        ):
            raise DeployError("LiteLLM destination must be one private IPv4 /32")
        data["MANAGEMENT_LITELLM_SERVICE_CIDR"] = litellm_cidr
    if "MANAGEMENT_LITELLM_SERVICE_CIDR" in data:
        data["MANAGEMENT_LITELLM_KEY_WORKER_IMAGE"] = images["management"]["reference"]
        for key in (
            "MANAGEMENT_LITELLM_SERVICE_CIDR",
            "MANAGEMENT_LITELLM_KEY_WORKER_IMAGE",
        ):
            cm_patch.append({"op": "add", "path": "/data/" + key, "value": data[key]})
    retention_init = None
    if gateway_artifact_retention:
        pod = deployment["spec"]["template"]["spec"]
        management = containers[indices[0]]
        volumes = {v["name"]: v for v in pod.get("volumes", [])}
        mounts = {
            v["name"]: v
            for v in management.get("volumeMounts", [])
            if v.get("mountPath") in ("/var/lib/frontier", "/var/lib/frontier-local")
        }
        if (
            volumes.get("data", {}).get("persistentVolumeClaim", {}).get("claimName")
            != "frontier-management-data"
            or mounts.get("data", {}).get("mountPath") != "/var/lib/frontier"
            or mounts.get("data", {}).get("readOnly", False)
            or mounts.get("objects", {}).get("mountPath") != "/var/lib/frontier-local"
            or mounts.get("objects", {}).get("readOnly") is not True
            or volumes.get("objects", {})
            .get("persistentVolumeClaim", {})
            .get("claimName")
            in (None, "frontier-management-data")
        ):
            raise DeployError(
                "gateway retention requires distinct Management data and read-only object mounts"
            )
        security = pod.get("securityContext", {})
        if (
            security.get("runAsUser") != 65532
            or security.get("runAsGroup") != 65532
            or security.get("fsGroup") != 65532
        ):
            raise DeployError(
                "gateway retention requires the existing Management UID/GID65532"
            )
        settings = {
            "MANAGEMENT_GATEWAY_ARTIFACT_SOURCE_ROOT": "/var/lib/frontier-local/artifacts",
            "MANAGEMENT_GATEWAY_ARTIFACT_RETAINED_ROOT": "/var/lib/frontier/gateway-terminal-retained",
        }
        if any(entry.get("name") in settings for entry in management.get("env", [])):
            raise DeployError("gateway retention rejects direct environment overrides")
        for key, value in settings.items():
            if key in data and data[key] != value:
                raise DeployError("gateway retention existing root differs")
            data[key] = value
            cm_patch.append({"op": "add", "path": "/data/" + key, "value": value})
        expected_old = gateway_artifact_retention_init(management["image"])
        retention_init = []
        observed = 0
        for existing in pod.get("initContainers", []):
            if existing.get("name") == expected_old["name"]:
                observed += 1
                normalized = dict(existing)
                # Kubernetes defaults these fields on readback; the initializer
                # has no termination-message override or output contract.
                for key, value in (
                    ("terminationMessagePath", "/dev/termination-log"),
                    ("terminationMessagePolicy", "File"),
                ):
                    if normalized.get(key) == value:
                        normalized.pop(key)
                compatible = reviewed_public_repair and reviewed_retention_transition(
                    normalized, expected_old, images["management"]["reference"]
                )
                if observed > 1 or (normalized != expected_old and not compatible):
                    raise DeployError("gateway retention existing init differs")
                retention_init.append(
                    gateway_artifact_retention_init(images["management"]["reference"])
                )
            else:
                retention_init.append(existing)
        if not observed:
            retention_init.append(
                gateway_artifact_retention_init(images["management"]["reference"])
            )
    annotations = dict(
        deployment["spec"]["template"]["metadata"].get("annotations", {})
    )
    annotations["frontier.dev/config-sha256"] = canonical_object_sha256(data)
    deploy_patch = [
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": deployment["metadata"]["resourceVersion"],
        },
        {
            "op": "replace",
            "path": f"/spec/template/spec/containers/{indices[0]}/image",
            "value": images["management"]["reference"],
        },
        {
            "op": "add",
            "path": "/spec/template/metadata/annotations",
            "value": annotations,
        },
    ]
    if large_verifier_results:
        pod = deployment["spec"]["template"]["spec"]
        temporary = [(index, volume) for index, volume in enumerate(pod.get("volumes", []))
                     if volume.get("name") == "tmp"]
        mounts = [mount for mount in containers[indices[0]].get("volumeMounts", [])
                  if mount.get("name") == "tmp"]
        if (len(temporary) != 1 or len(mounts) != 1
                or mounts[0].get("mountPath") != "/tmp" or mounts[0].get("readOnly", False)
                or "emptyDir" not in temporary[0][1]
                or temporary[0][1]["emptyDir"].get("medium", "") != ""):
            raise DeployError("large verifier results require one disk-backed Management /tmp")
        temporary_index, _ = temporary[0]
        deploy_patch[-1:-1] = [
            {"op": "add", "path": f"/spec/template/spec/containers/{indices[0]}/resources",
             "value": management_result_resources()},
            {"op": "add", "path": f"/spec/template/spec/volumes/{temporary_index}/emptyDir/sizeLimit",
             "value": management_result_resources()["limits"]["ephemeral-storage"]},
        ]
    if retention_init is not None:
        deploy_patch.insert(
            -1,
            {
                "op": "add",
                "path": "/spec/template/spec/initContainers",
                "value": retention_init,
            },
        )
        volume, mount = management_diagnostic_source_mount(
            volumes["objects"]["persistentVolumeClaim"]["claimName"]
        )
        # Kubelet tracks one CSI volume per PVC; both paths must share its name.
        legacy_volume = {**volume, "name": "diagnostic-source"}
        legacy_mount = {**mount, "name": "diagnostic-source"}
        aliases = [v for v in pod["volumes"] if v["name"] == "diagnostic-source"]
        if aliases and aliases != [legacy_volume]:
            raise DeployError("diagnostic source existing mount or volume differs")
        expected_claim = volume["persistentVolumeClaim"]["claimName"]
        for existing in pod["volumes"]:
            if (
                existing["name"] not in ("objects", "diagnostic-source")
                and existing.get("persistentVolumeClaim", {}).get("claimName")
                == expected_claim
            ):
                raise DeployError(
                    "diagnostic source has an unexpected duplicate PVC alias"
                )
        for container in [*pod["containers"], *pod.get("initContainers", [])]:
            for existing in container.get("volumeMounts", []):
                if existing["name"] == "diagnostic-source" and (
                    container is not management or existing != legacy_mount
                ):
                    raise DeployError(
                        "diagnostic source legacy alias has another consumer"
                    )
                if existing["name"] == "objects" and not existing.get(
                    "readOnly", False
                ):
                    if container is not management or existing != mount:
                        raise DeployError(
                            "diagnostic source would make another object mount writable"
                        )
        artifact_mounts = [
            m
            for m in management["volumeMounts"]
            if m.get("mountPath") == mount["mountPath"]
        ]
        if artifact_mounts not in ([], [mount], [legacy_mount]):
            raise DeployError(
                "diagnostic source mount path already occupied or differs"
            )
        new_volumes = [
            volume if v["name"] == "objects" else v
            for v in pod["volumes"]
            if v["name"] != "diagnostic-source"
        ]
        new_mounts = [
            mount if m.get("mountPath") == mount["mountPath"] else m
            for m in management["volumeMounts"]
        ]
        if not artifact_mounts:
            new_mounts.append(mount)
        for path, value in (
            ("/spec/template/spec/volumes", new_volumes),
            (
                f"/spec/template/spec/containers/{indices[0]}/volumeMounts",
                new_mounts,
            ),
        ):
            deploy_patch.insert(
                -1,
                {"op": "add", "path": path, "value": value},
            )
    if reviewed_public_repair:
        require_reviewed_route_only_patches(configmap, deployment, data, deploy_patch)
    return cm_patch, deploy_patch, data


def modal_observer_upgrade_manifest(
    config: Mapping[str, Any],
    configmap: Mapping[str, Any],
    deployment: Mapping[str, Any],
) -> bytes | None:
    items = modal_observer_rbac(config)
    if not items:
        return None
    data = configmap["data"]
    pod = deployment["spec"]["template"]["spec"]
    if (
        data.get("MANAGEMENT_NAMESPACE") != PLATFORM_NAMESPACE
        or data.get("MANAGEMENT_MODAL_CREDENTIALS_SECRET")
        != config["modal"]["credentials_secret"]
        or pod.get("serviceAccountName") != "frontier-management"
    ):
        raise DeployError(
            "Modal observer existing namespace, Secret or ServiceAccount differs from site configuration"
        )
    return canonical_json({"apiVersion": "v1", "kind": "List", "items": items})


def infrastructure_role_patch(role: Mapping[str, Any]) -> tuple[list, list]:
    """Preserve existing authority while adding read-only node observation."""
    rules = list(role['rules'])
    rule = {'apiGroups': [''], 'resources': ['nodes'], 'verbs': ['get', 'list']}
    if rule not in rules:
        rules.append(rule)
    return [
        {'op': 'test', 'path': '/metadata/uid', 'value': role['metadata']['uid']},
        {'op': 'test', 'path': '/metadata/resourceVersion', 'value': role['metadata']['resourceVersion']},
        {'op': 'replace', 'path': '/rules', 'value': rules},
    ], rules


def upgrade_runtime(config: Mapping[str, Any], output: Path, kubeconfig: Path) -> None:
    """Upgrade an idle deployed site without re-running bootstrap or seeding."""
    load_task_selection(config, require_resolved=True)
    gateway_before = read_gateway_snapshot(kubeconfig)
    images = load_runtime_images(config)
    copy_tool = executable(os.environ.get("FRONTIER_REGISTRY_COPY_TOOL", "crane"))
    environment = kubernetes_environment(kubeconfig)
    kubectl = [executable("kubectl"), "-n", PLATFORM_NAMESPACE]

    def get(kind: str, name: str) -> dict:
        return json.loads(
            run_command(
                kubectl + ["get", kind, name, "-o", "json"],
                environment=environment,
                capture=True,
                timeout=60,
            )
        )

    with port_forward(kubeconfig, MANAGEMENT_SERVICE, 8080) as port:
        require_idle_runtime(f"http://127.0.0.1:{port}")
    cm = get("configmap", "frontier-management-config")
    deployment = get("deployment", "frontier-management")
    unchanged_management = unchanged_native_management(config, cm, deployment, images)
    reviewed_public_repair = (False if unchanged_management else
        preserve_reviewed_public_images(config, cm, deployment, images))
    if not reviewed_public_repair:
        with port_forward(kubeconfig, REGISTRY_SERVICE, 5000) as port:
            if unchanged_management:
                if cached_registry_image(port, unchanged_management, images["management"]["digest"]) != images["management"]["digest"]:
                    raise DeployError("unchanged Management manifest is unavailable or differs")
            for name in (("driver",) if unchanged_management else ("management", "driver")):
                source = config["images"][name]
                repository = f"frontier/{name}"
                tag = "git-" + runtime_source_commit(config, name)[:20]
                if cached_registry_image(port, repository, source["digest"]) is None:
                    run_command(
                        [
                            copy_tool,
                            "copy",
                            "--platform",
                            "linux/amd64",
                            "--jobs",
                            "2",
                            source["reference"],
                            f"127.0.0.1:{port}/{repository}:{tag}",
                        ],
                        timeout=1800,
                    )
                if (
                    cached_registry_image(port, repository, source["digest"])
                    != source["digest"]
                ):
                    raise DeployError("runtime mirror did not preserve the source manifest")
                endpoint = f"{config['network']['registry_cluster_ip']}:5000/{repository}"
                images[name] = {
                    "tag": endpoint + ":" + tag,
                    "digest": source["digest"],
                    "reference": endpoint + "@" + source["digest"],
                    "source_sha256": source["digest"],
                }
    receipt_dir = output / ("runtime-upgrade-" + config["application"]["commit"][:12])
    receipt_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(receipt_dir, 0o700)
    cidrs = api_cidrs(config, kubeconfig, receipt_path=receipt_dir / "api-routes.json")
    outputs = platform_outputs(config)
    role = get('clusterrole', 'frontier-management-preview')
    role_patch, expected_rules = infrastructure_role_patch(role)
    observer_manifest = modal_observer_upgrade_manifest(config, cm, deployment)
    if (
        deployment["spec"]["template"]["spec"].get("serviceAccountName")
        != "frontier-management"
    ):
        raise DeployError("diagnostics upgrade requires the Management ServiceAccount")
    cm_patch, deploy_patch, expected_data = runtime_upgrade_patches(
        cm,
        deployment,
        images,
        litellm_cidr=config["network"]["litellm_cluster_ip"] + "/32",
        gateway_artifact_retention=True,
        kubernetes_api_cidrs=cidrs,
        cluster_name=outputs['cluster_name'],
        region=config['site']['aws_region'],
        large_verifier_results=True,
        verifier_node_allowlists=config["execution"].get("verifier_node_allowlists", []),
        verifier_placement_required_task_ids=config["execution"].get("verifier_placement_required_task_ids", []),
        reviewed_public_repair=reviewed_public_repair,
    )
    preserved_observer = []
    if unchanged_management:
        require_driver_only_patches(cm, deployment, expected_data, deploy_patch)
        if expected_rules != role["rules"]:
            raise DeployError("unchanged Management upgrade has extra infrastructure changes")
        if observer_manifest is not None:
            for expected in json.loads(observer_manifest)["items"]:
                live = get(expected["kind"], expected["metadata"]["name"])
                require_unchanged_observer(expected, live)
                preserved_observer.append((expected, live))
            observer_manifest = None  # Already exact: leave these resources untouched.
    atomic_write(
        receipt_dir / "gateway-before.json", canonical_json(gateway_before), mode=0o600
    )
    diagnostics_path = receipt_dir / "management-diagnostics-rbac.json"
    atomic_write(
        diagnostics_path,
        canonical_json(
            {
                "apiVersion": "v1",
                "kind": "List",
                "items": management_diagnostics_rbac(config),
            }
        ),
        mode=0o600,
    )
    diagnostics_command = kubectl + [
        "apply",
        "--server-side",
        "--field-manager=frontier-deploy",
        "-f",
        str(diagnostics_path),
    ]
    run_command(
        diagnostics_command + ["--dry-run=server"], environment=environment, timeout=60
    )
    api_rbac_path = receipt_dir / "management-api-routes-rbac.json"
    atomic_write(
        api_rbac_path,
        canonical_json(
            {"apiVersion": "v1", "kind": "List", "items": management_api_routes_rbac()}
        ),
        mode=0o600,
    )
    api_rbac_command = [
        executable("kubectl"),
        "apply",
        "--server-side",
        "--field-manager=frontier-deploy",
        "-f",
        str(api_rbac_path),
    ]
    run_command(
        api_rbac_command + ["--dry-run=server"], environment=environment, timeout=60
    )
    observer_command = None
    if observer_manifest is not None:
        observer_path = receipt_dir / "modal-observer-rbac.json"
        atomic_write(observer_path, observer_manifest, mode=0o600)
        observer_command = kubectl + [
            "apply",
            "--server-side",
            "--field-manager=frontier-deploy",
            "-f",
            str(observer_path),
        ]
        run_command(
            observer_command + ["--dry-run=server"], environment=environment, timeout=60
        )
    for name, body in (
        ("configmap-before", cm),
        ("deployment-before", deployment),
        ("configmap-patch", cm_patch),
        ("deployment-patch", deploy_patch),
        ('clusterrole-before', role),
        ('clusterrole-patch', role_patch),
    ):
        atomic_write(receipt_dir / (name + ".json"), canonical_json(body), mode=0o600)
    commands = []
    policy_specs = {}
    old_cidrs = cm["data"].get("MANAGEMENT_KUBERNETES_API_CIDRS", "").split(",")
    for name in (
        "frontier-management-boundary",
        "frontier-storage-controller-boundary",
    ):
        before = get("networkpolicy", name)
        patch, policy_specs[name] = api_policy_patch(before, cidrs, old_cidrs)
        atomic_write(
            receipt_dir / (name + "-before.json"), canonical_json(before), mode=0o600
        )
        path = receipt_dir / (name + "-patch.json")
        atomic_write(path, canonical_json(patch), mode=0o600)
        command = kubectl + [
            "patch",
            "networkpolicy",
            name,
            "--type=json",
            "--patch-file",
            str(path),
        ]
        run_command(command + ["--dry-run=server"], environment=environment, timeout=60)
        commands.append(command)
    for kind, name in (
        ('clusterrole', 'frontier-management-preview'),
        ("configmap", "frontier-management-config"),
        ("deployment", "frontier-management"),
    ):
        command = kubectl + [
            "patch",
            kind,
            name,
            "--type=json",
            "--patch-file",
            str(receipt_dir / (kind + "-patch.json")),
        ]
        run_command(command + ["--dry-run=server"], environment=environment, timeout=60)
        commands.append(command)
    with port_forward(kubeconfig, MANAGEMENT_SERVICE, 8080) as port:
        require_idle_runtime(f"http://127.0.0.1:{port}")
    if read_gateway_snapshot(kubeconfig) != gateway_before:
        raise DeployError("LiteLLM gateway changed before runtime upgrade")
    if api_cidrs(config, kubeconfig) != cidrs:
        raise DeployError(
            "Kubernetes API destinations changed before upgrade; no patches applied"
        )
    for expected, before in preserved_observer:
        if get(expected["kind"], expected["metadata"]["name"]) != before:
            raise DeployError("unchanged Management Modal observer changed before upgrade")
    if observer_command is not None:
        run_command(observer_command, environment=environment, timeout=60)
    run_command(diagnostics_command, environment=environment, timeout=60)
    run_command(api_rbac_command, environment=environment, timeout=60)
    verify_api_route_reader_rbac(kubeconfig)
    for command in commands:
        run_command(command, environment=environment, timeout=60)
    run_command(
        kubectl
        + ["rollout", "status", "deployment/frontier-management", "--timeout=10m"],
        environment=environment,
        timeout=660,
    )
    after_role = get('clusterrole', 'frontier-management-preview')
    if after_role['metadata']['uid'] != role['metadata']['uid'] or after_role['rules'] != expected_rules:
        raise DeployError('Infrastructure observation role changed during runtime upgrade')
    for name, expected in policy_specs.items():
        if get("networkpolicy", name)["spec"] != expected:
            raise DeployError(
                "Kubernetes API policy readback differs from reviewed patch"
            )
    verify_api_routes(config, kubeconfig)
    after_cm = get("configmap", "frontier-management-config")
    after_deployment = get("deployment", "frontier-management")
    if after_cm["data"] != expected_data:
        raise DeployError("Management ConfigMap changed during runtime upgrade")
    expected_spec = json.loads(json.dumps(deployment["spec"]))
    i = next(
        i
        for i, c in enumerate(expected_spec["template"]["spec"]["containers"])
        if c["name"] == "management"
    )
    expected_spec["template"]["spec"]["containers"][i]["image"] = images["management"][
        "reference"
    ]
    for patch in deploy_patch:
        if patch["path"] == "/spec/template/spec/initContainers":
            expected_spec["template"]["spec"]["initContainers"] = patch["value"]
        elif patch["path"] == "/spec/template/spec/volumes":
            expected_spec["template"]["spec"]["volumes"] = patch["value"]
        elif patch["path"] == f"/spec/template/spec/containers/{i}/volumeMounts":
            expected_spec["template"]["spec"]["containers"][i]["volumeMounts"] = patch[
                "value"
            ]
        elif patch["path"] == f"/spec/template/spec/containers/{i}/resources":
            expected_spec["template"]["spec"]["containers"][i]["resources"] = patch["value"]
        elif (patch["path"].startswith("/spec/template/spec/volumes/")
              and patch["path"].endswith("/emptyDir/sizeLimit")):
            volume_index = int(patch["path"].split("/")[5])
            expected_spec["template"]["spec"]["volumes"][volume_index]["emptyDir"]["sizeLimit"] = patch["value"]
    expected_spec["template"]["metadata"]["annotations"] = deploy_patch[-1]["value"]
    actual_spec = json.loads(json.dumps(after_deployment["spec"]))
    for init in actual_spec["template"]["spec"].get("initContainers", []):
        if init.get("name") == "gateway-artifact-retention":
            for key, value in (
                ("terminationMessagePath", "/dev/termination-log"),
                ("terminationMessagePolicy", "File"),
            ):
                if init.get(key) == value:
                    init.pop(key)
    if actual_spec != expected_spec:
        raise DeployError(
            "Management Deployment changed outside the runtime image upgrade"
        )
    with port_forward(kubeconfig, MANAGEMENT_SERVICE, 8080) as port:
        api_json(f"http://127.0.0.1:{port}", "GET", "/healthz")
        require_idle_runtime(f"http://127.0.0.1:{port}")
    gateway_after = read_gateway_snapshot(kubeconfig)
    atomic_write(
        receipt_dir / "gateway-after.json", canonical_json(gateway_after), mode=0o600
    )
    if gateway_after != gateway_before:
        raise DeployError("LiteLLM gateway changed during runtime upgrade")
    atomic_write(
        state_directory(config) / "runtime-images.json",
        canonical_json(images),
        mode=0o600,
    )
    atomic_write(
        receipt_dir / "accepted.json",
        canonical_json(
            {
                "application_commit": config["application"]["commit"],
                "runtime_source_commits": {
                    role: runtime_source_commit(config, role)
                    for role in ("management", "driver")
                },
                "images": {k: images[k] for k in ("management", "driver")},
                "management_generation": after_deployment["metadata"]["generation"],
                "readback": "PASS",
                "kubernetes_api_cidrs": cidrs,
                "api_routes_receipt_sha256": canonical_object_sha256(
                    load_json(receipt_dir / "api-routes.json")
                ),
            }
        ),
        mode=0o600,
    )


def storage_controller_upgrade_patch(
    deployment: Mapping[str, Any], image: str, cidr: str
) -> tuple[list, dict]:
    network = ipaddress.ip_network(cidr, strict=True)
    if network.version != 4 or network.prefixlen != 32:
        raise DeployError("local registry access requires an exact IPv4 /32")
    expected = json.loads(json.dumps(deployment["spec"]))
    containers = expected["template"]["spec"]["containers"]
    indices = [
        i for i, container in enumerate(containers) if container["name"] == "controller"
    ]
    if len(indices) != 1:
        raise DeployError("expected one storage-controller container")
    index = indices[0]
    container = containers[index]
    container["image"] = image
    environment = [
        entry
        for entry in container.get("env", [])
        if entry["name"] != "FRONTIER_REGISTRY_SERVICE_CIDR"
    ]
    environment.append(
        {"name": "FRONTIER_REGISTRY_SERVICE_CIDR", "value": str(network)}
    )
    container["env"] = environment
    return [
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": deployment["metadata"]["resourceVersion"],
        },
        {
            "op": "replace",
            "path": f"/spec/template/spec/containers/{index}/image",
            "value": image,
        },
        {
            "op": "add",
            "path": f"/spec/template/spec/containers/{index}/env",
            "value": environment,
        },
    ], expected


def upgrade_local_network(
    config: Mapping[str, Any], output: Path, kubeconfig: Path, receipt_path: Path
) -> None:
    """Upgrade only Preview's controller; preserve every live Run and runtime pin."""
    receipt = load_json(receipt_path)
    commit = receipt["source_commit"]
    if (
        not re.fullmatch(r"[0-9a-f]{40}", commit)
        or receipt["tag"] != "sha-" + commit
        or not receipt["repository"].startswith("docker.io/")
        or not DIGEST_RE.fullmatch(receipt["digest"])
    ):
        raise DeployError(
            "controller image receipt lacks immutable public source identity"
        )
    source = REPOSITORY_ROOT / "runtime/storage-controller"
    if receipt["source_sha256"] != source_tree_sha256(source):
        raise DeployError("controller image receipt differs from checked-out source")
    changed = run_command(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "diff",
            commit,
            "--",
            "runtime/storage-controller",
        ],
        capture=True,
        timeout=30,
    )
    if changed.strip():
        raise DeployError("controller image tag does not bind its source commit")
    repository = "frontier/storage-controller"
    with port_forward(kubeconfig, REGISTRY_SERVICE, 5000) as port:
        if cached_registry_image(port, repository, receipt["digest"]) is None:
            copy_tool = executable(
                os.environ.get("FRONTIER_REGISTRY_COPY_TOOL", "crane")
            )
            run_command(
                [
                    copy_tool,
                    "copy",
                    "--platform",
                    "linux/amd64",
                    "--jobs",
                    "2",
                    receipt["repository"] + "@" + receipt["digest"],
                    f"127.0.0.1:{port}/{repository}:" + receipt["tag"],
                ],
                timeout=1800,
            )
        if (
            cached_registry_image(port, repository, receipt["digest"])
            != receipt["digest"]
        ):
            raise DeployError(
                "controller registry mirror did not preserve the manifest"
            )
    image = f"{config['network']['registry_cluster_ip']}:5000/{repository}@{receipt['digest']}"
    environment = kubernetes_environment(kubeconfig)
    kubectl = [executable("kubectl"), "-n", PLATFORM_NAMESPACE]

    def get() -> dict:
        return json.loads(
            run_command(
                kubectl
                + ["get", "deployment", "frontier-storage-controller", "-o", "json"],
                environment=environment,
                capture=True,
                timeout=60,
            )
        )

    before = get()
    patch, expected = storage_controller_upgrade_patch(
        before, image, config["network"]["registry_cluster_ip"] + "/32"
    )
    directory = output / ("local-network-upgrade-" + commit[:12])
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write(
        directory / "deployment-before.json", canonical_json(before), mode=0o600
    )
    atomic_write(directory / "deployment-patch.json", canonical_json(patch), mode=0o600)
    command = kubectl + [
        "patch",
        "deployment",
        "frontier-storage-controller",
        "--type=json",
        "--patch-file",
        str(directory / "deployment-patch.json"),
    ]
    run_command(command + ["--dry-run=server"], environment=environment, timeout=60)
    run_command(command, environment=environment, timeout=60)
    run_command(
        kubectl
        + [
            "rollout",
            "status",
            "deployment/frontier-storage-controller",
            "--timeout=5m",
        ],
        environment=environment,
        timeout=330,
    )
    after = get()
    if after["spec"] != expected or after.get("status", {}).get("readyReplicas") != 1:
        raise DeployError("controller readback differs from the exact reviewed upgrade")
    images = load_runtime_images(config)
    images["storage_controller"] = {
        "reference": image,
        "digest": receipt["digest"],
        "tag": image.split("@")[0] + ":" + receipt["tag"],
        "source_sha256": receipt["source_sha256"],
    }
    atomic_write(
        state_directory(config) / "runtime-images.json",
        canonical_json(images),
        mode=0o600,
    )
    atomic_write(
        directory / "accepted.json",
        canonical_json(
            {
                "status": "PASS",
                "image": image,
                "source_commit": commit,
                "generation": after["metadata"]["generation"],
                "registry_service_cidr": config["network"]["registry_cluster_ip"]
                + "/32",
            }
        ),
        mode=0o600,
    )


def deploy_runtime(
    config: Mapping[str, Any],
    output: Path,
    outputs: Mapping[str, Any],
    kubeconfig: Path,
    images: Mapping[str, Any],
) -> None:
    deploy_modal_credentials(config, kubeconfig)
    cidrs = api_cidrs(config, kubeconfig, receipt_path=output / "api-routes.json")
    simulator_path = output / "preview-admission-bootstrap.json"
    atomic_write(
        simulator_path,
        render_runtime_manifest(
            config,
            outputs,
            images,
            cidrs,
            simulator_bootstrap=True,
        ),
    )
    environment = kubernetes_environment(kubeconfig)
    _, seed_job_name = dataset_seed_names()
    # Job templates are immutable; a prior apply's completed seed Job would
    # reject this apply whenever the freshly built controller image digest
    # differs. The Job is disposable and reseeding is idempotent, so drop it.
    run_command(
        [
            executable("kubectl"),
            "-n",
            PLATFORM_NAMESPACE,
            "delete",
            "job",
            seed_job_name,
            "--ignore-not-found",
        ],
        environment=environment,
        timeout=120,
    )
    print("admission control maps: skipped for simulator bootstrap")
    kubectl_apply(simulator_path, kubeconfig)
    run_command(
        [
            executable("kubectl"),
            "-n",
            PLATFORM_NAMESPACE,
            "wait",
            "--for=condition=complete",
            f"job/{seed_job_name}",
            "--timeout=10m",
        ],
        environment=environment,
        timeout=660,
    )
    for deployment in ("frontier-storage-controller", "frontier-management"):
        run_command(
            [
                executable("kubectl"),
                "-n",
                PLATFORM_NAMESPACE,
                "rollout",
                "status",
                f"deployment/{deployment}",
                "--timeout=10m",
            ],
            environment=environment,
            timeout=660,
        )

    # The current Management release intentionally opens non-production
    # admission only in simulator mode. Reuse the same SQLite PVC for one
    # migration/open bootstrap, then restart into the real local-k8s launcher.
    if api_cidrs(config, kubeconfig) != cidrs:
        raise DeployError(
            "Kubernetes API destinations changed during bootstrap; rerun deploy"
        )
    path = output / "executable-preview.json"
    atomic_write(path, render_runtime_manifest(config, outputs, images, cidrs))
    kubectl_apply(path, kubeconfig)
    verify_api_route_reader_rbac(kubeconfig)
    ensure_admission_control_maps(kubeconfig)
    run_command(
        [
            executable("kubectl"),
            "-n",
            PLATFORM_NAMESPACE,
            "rollout",
            "restart",
            "deployment/frontier-management",
        ],
        environment=environment,
        timeout=120,
    )
    run_command(
        [
            executable("kubectl"),
            "-n",
            PLATFORM_NAMESPACE,
            "rollout",
            "status",
            "deployment/frontier-management",
            "--timeout=10m",
        ],
        environment=environment,
        timeout=660,
    )
    with port_forward(kubeconfig, MANAGEMENT_SERVICE, 8080) as port:
        runtime_admission_preflight(config, f"http://127.0.0.1:{port}")


def plan_site(config_path: Path) -> None:
    config, output = render_bundle(config_path)
    aws_identity(config)
    executable("terraform")
    plans = state_directory(config) / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    terraform_init_and_validate(config, "bootstrap", output / "bootstrap-backend.hcl")
    terraform_plan(
        config,
        "bootstrap",
        output / "bootstrap.auto.tfvars.json",
        plans / "bootstrap.tfplan",
    )
    if state_bucket_exists(config):
        terraform_init_and_validate(config, "platform", output / "backend.hcl")
        terraform_plan(
            config,
            "platform",
            output / "platform.auto.tfvars.json",
            plans / "platform.tfplan",
        )
    else:
        terraform_init_and_validate(
            config, "platform", output / "backend.hcl", backend=False
        )
        print(
            "Platform Terraform validated; create the state bucket before the remote-state platform plan."
        )


def deploy_existing_platform(
    config: Mapping[str, Any], output: Path, *, run_acceptance: bool
) -> None:
    _, selection = load_task_selection(config, require_resolved=True)
    modal_credentials(config, selection)
    outputs = platform_outputs(config)
    kubeconfig = update_kubeconfig(config, outputs)
    deploy_bootstrap(config, output, outputs, kubeconfig)
    images = build_runtime_images(config, kubeconfig)
    deploy_litellm(config, output, outputs, kubeconfig, images)
    deploy_runtime(config, output, outputs, kubeconfig, images)
    if run_acceptance:
        run_smoke(config, kubeconfig, images)
    if TASK_CATALOG_PATH.is_file():
        seed_tasks(config, kubeconfig, TASK_CATALOG_PATH)
    verify_runtime(config, outputs, kubeconfig, require_smoke=run_acceptance)


def apply_site(config_path: Path, confirm_account: str, confirm_site: str) -> None:
    config, output = render_bundle(config_path)
    if (
        confirm_account != config["site"]["expected_account_id"]
        or confirm_site != config["site"]["name"]
    ):
        raise DeployError(
            "apply requires exact --confirm-account and --confirm-site values"
        )
    _, selection = load_task_selection(config, require_resolved=True)
    modal_credentials(config, selection)
    aws_identity(config)
    for name in ("terraform", "kubectl", "docker"):
        executable(name)
    plans = state_directory(config) / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    terraform_init_and_validate(config, "bootstrap", output / "bootstrap-backend.hcl")
    bootstrap_plan = plans / "bootstrap.tfplan"
    terraform_plan(
        config, "bootstrap", output / "bootstrap.auto.tfvars.json", bootstrap_plan
    )
    terraform_apply_plan(config, "bootstrap", bootstrap_plan)
    terraform_init_and_validate(config, "platform", output / "backend.hcl")
    platform_plan = plans / "platform.tfplan"
    terraform_plan(
        config, "platform", output / "platform.auto.tfvars.json", platform_plan
    )
    terraform_apply_plan(config, "platform", platform_plan)
    deploy_existing_platform(config, output, run_acceptance=True)


def validate_command(config_path: Path) -> None:
    config, digest = read_and_validate(config_path)
    catalog_status = "absent"
    if TASK_CATALOG_PATH.is_file():
        catalog, selection = load_task_selection(config)
        print("selected tasks: " + ", ".join(selection.selected))
        for name, reason in selection.excluded.items():
            print(f"excluded task: {name}: {reason}")
        print(f"effective catalog: {selection.effective_catalog_sha256}")
        unresolved = len(catalog["unresolved_digests"])
        catalog_status = (
            f"valid ({len(catalog['tasks'])} tasks, {unresolved} placeholder digests)"
        )
    print(
        f"config: PASS ({digest})\nsite: {config['site']['name']}\n"
        f"account: {config['site']['expected_account_id']}\nregion: {config['site']['aws_region']}\n"
        f"task catalog: {catalog_status}\n"
        "profile: executable-preview (fixed task catalog; scripted CPU smoke; optional Modal GPU)"
    )


def preflight_command(config_path: Path) -> None:
    config, _ = read_and_validate(config_path)
    modal_credentials(config)
    for name in ("terraform", "aws", "kubectl", "docker"):
        print(f"{name}: {executable(name)}")
    copy_tool = os.environ.get("FRONTIER_REGISTRY_COPY_TOOL", "crane")
    print(
        f"registry copy tool (optional; runtime upgrades): {shutil.which(copy_tool) or 'missing: ' + copy_tool}"
    )
    aws_identity(config)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="frontier_deploy.py", description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in (
        "validate",
        "render",
        "preflight",
        "plan",
        "deploy",
        "upgrade-runtime",
        "build-images",
        "smoke",
        "verify",
    ):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
    seed_parser = commands.add_parser("seed-tasks")
    seed_parser.add_argument("--config", required=True, type=Path)
    seed_parser.add_argument("--catalog", type=Path, default=TASK_CATALOG_PATH)
    local_network = commands.add_parser("upgrade-local-network")
    local_network.add_argument("--config", required=True, type=Path)
    local_network.add_argument("--image-receipt", required=True, type=Path)
    seed_parser.add_argument(
        "--check",
        action="store_true",
        help="GET existing versions and compare digests; do not seed or write state",
    )
    native = commands.add_parser("native-publish", help="Publish a frozen native Task on local-k8s without catalog import")
    native.add_argument("--config", required=True, type=Path)
    stages = native.add_subparsers(dest="stage", required=True)
    for stage in ("plan", "prepare", "render-seed", "render-inspector", "finalize", "register"):
        command = stages.add_parser(stage)
        command.add_argument("--document", required=True, type=Path)
        if stage != "plan":
            command.add_argument("--output", required=stage != "register", type=Path)
        if stage in ("render-seed", "render-inspector", "finalize", "register"):
            command.add_argument("--bundle", required=True, type=Path)
        if stage in ("finalize", "register"):
            mount = command.add_mutually_exclusive_group(required=True)
            mount.add_argument("--mounted-root", type=Path, help="Actual read-only object PVC mount")
            mount.add_argument("--pvc-pod", help="Trusted inspector pod with the read-only object PVC")
        if stage == "register":
            action = command.add_mutually_exclusive_group(required=True)
            action.add_argument("--check", action="store_true")
            action.add_argument("--apply", action="store_true")
    native_images = commands.add_parser("native-localize-images", help="Mirror frozen native images and derive immutable local versions")
    native_images.add_argument("--config", required=True, type=Path)
    native_images.add_argument("--document", required=True, type=Path)
    native_images.add_argument("--candidate", type=Path)
    native_images.add_argument("--output", required=True, type=Path)
    action = native_images.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true", help="Offline plan only")
    action.add_argument("--check", action="store_true", help="Read-only registry verification")
    action.add_argument("--apply", action="store_true", help="Copy exact image manifests; no registration or definition upload")
    native_artifacts = commands.add_parser("native-local-artifacts", help="Publish immutable native image-localized definitions and Candidate records")
    native_artifacts.add_argument("--config", required=True, type=Path)
    artifact_stages = native_artifacts.add_subparsers(dest="stage", required=True)
    for stage in ("rebind", "definitions", "candidate"):
        artifact = artifact_stages.add_parser(stage)
        artifact.add_argument("--document", required=True, type=Path)
        artifact.add_argument("--candidate", type=Path)
        artifact.add_argument("--output", required=True, type=Path)
        if stage == "rebind":
            artifact.add_argument("--bundle", required=True, type=Path)
            artifact.add_argument("--rebound-bundle", required=True, type=Path)
        else:
            artifact_action = artifact.add_mutually_exclusive_group(required=True)
            artifact_action.add_argument("--check", action="store_true")
            artifact_action.add_argument("--apply", action="store_true")
    gpu_parser = commands.add_parser("gpu-smoke")
    gpu_parser.add_argument("--config", type=Path, required=True)
    gpu_parser.add_argument("--patch", type=Path, required=True)
    gpu_parser.add_argument("--patch-sha256", required=True)
    gpu_parser.add_argument("--attempt", required=True)
    apply_parser = commands.add_parser("apply")
    apply_parser.add_argument("--config", required=True, type=Path)
    apply_parser.add_argument("--confirm-account", required=True)
    apply_parser.add_argument("--confirm-site", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        config_path = arguments.config.resolve(strict=True)
        if arguments.command == "native-local-artifacts":
            from .native_definition_publication import command
            print(json.dumps(command(arguments), indent=2, sort_keys=True))
            return 0
        if arguments.command == "native-localize-images":
            from .native_image_publication import command
            result = command(arguments)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if arguments.command == "native-publish":
            from .native_local_publication import command
            if arguments.stage == "register" and arguments.apply and arguments.output is None:
                raise DeployError("native-publish register --apply requires --output")
            print(json.dumps(command(arguments), sort_keys=True))
        elif arguments.command == "validate":
            validate_command(config_path)
        elif arguments.command == "render":
            config, output = render_bundle(config_path)
            print(
                f"rendered: {output}\nprofile: {config['management']['profile']} "
                "(fixed task catalog; scripted CPU smoke; optional Modal GPU)"
            )
        elif arguments.command == "preflight":
            preflight_command(config_path)
        elif arguments.command == "plan":
            plan_site(config_path)
        elif arguments.command == "apply":
            apply_site(config_path, arguments.confirm_account, arguments.confirm_site)
        elif arguments.command == "seed-tasks" and arguments.check:
            config, _ = read_and_validate(config_path)
            catalog_path = arguments.catalog.resolve(strict=True)
            # Reuse the saved kubeconfig: update_kubeconfig/platform_outputs write .state.
            seed_tasks(
                config, state_directory(config) / "kubeconfig", catalog_path, check=True
            )
        else:
            catalog_path = (
                arguments.catalog.resolve(strict=True)
                if arguments.command == "seed-tasks"
                else TASK_CATALOG_PATH
            )
            config, output = render_bundle(config_path, catalog_path=catalog_path)
            if arguments.command == "seed-tasks":
                catalog_path = arguments.catalog.resolve(strict=True)
                load_task_selection(config, catalog_path, require_resolved=True)
            elif arguments.command == "deploy" and TASK_CATALOG_PATH.is_file():
                _, selection = load_task_selection(config, require_resolved=True)
                modal_credentials(config, selection)
            aws_identity(config)
            outputs = platform_outputs(config)
            kubeconfig = update_kubeconfig(config, outputs)
            if arguments.command == "deploy":
                deploy_existing_platform(config, output, run_acceptance=True)
            elif arguments.command == "upgrade-runtime":
                upgrade_runtime(config, output, kubeconfig)
            elif arguments.command == "upgrade-local-network":
                upgrade_local_network(
                    config,
                    output,
                    kubeconfig,
                    arguments.image_receipt.resolve(strict=True),
                )
            elif arguments.command == "build-images":
                deploy_bootstrap(config, output, outputs, kubeconfig)
                build_runtime_images(config, kubeconfig)
            elif arguments.command == "smoke":
                verify_api_routes(config, kubeconfig)
                run_smoke(config, kubeconfig, load_runtime_images(config))
            elif arguments.command == "gpu-smoke":
                verify_api_routes(config, kubeconfig)
                run_gpu_smoke(
                    config,
                    kubeconfig,
                    arguments.patch,
                    arguments.patch_sha256,
                    arguments.attempt,
                )
            elif arguments.command == "seed-tasks":
                seed_tasks(config, kubeconfig, catalog_path)
            elif arguments.command == "verify":
                verify_api_routes(config, kubeconfig)
                verify_runtime(config, outputs, kubeconfig)
            else:
                raise DeployError(f"unsupported command: {arguments.command}")
        return 0
    except (DeployError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
