"""Process, filesystem, Terraform, Kubernetes and runtime manifest primitives."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASK_CATALOG_PATH = REPOSITORY_ROOT / "config/task-catalog.json"
SUPPORTED_SCHEMA = "frontier.deployment-site/v2"
SUPPORTED_TASK_CATALOG_SCHEMA = "frontier.deployment-task-catalog/v2"
SUPPORTED_KUBERNETES_VERSION = "1.34"
APPLICATION_REPOSITORY = "https://github.com/FrontierCS/FrontierCS-2.git"
SITE_RE = re.compile(r"^[a-z][a-z0-9-]{1,22}[a-z0-9]$")
REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z]+)+-[0-9]$")
ACCOUNT_RE = re.compile(r"^[0-9]{12}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
INSTANCE_TYPE_RE = re.compile(r"^[a-z][a-z0-9]*[0-9][a-z0-9.]*$")
S3_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
IAM_ROLE_ID_RE = re.compile(r"^[A-Z0-9]{16,128}$")
PLATFORM_NAMESPACE = "frontier-system"
MANAGEMENT_SERVICE = "frontier-management"
REGISTRY_SERVICE = "frontier-registry"
LITELLM_NAMESPACE = "frontier-litellm"
LITELLM_SERVICE = "frontier-litellm"
LITELLM_SECRET = "frontier-litellm-secrets"
ADMISSION_COMMAND_MAP = "frontier-management-production-command"
ADMISSION_RECEIPT_MAP = "frontier-management-production-receipt"
# Verbatim platform control data; canonical_json would append an invalid newline.
ADMISSION_BOOTSTRAP_COMMAND = '{"attempt_ids":[],"command_nonce":"bootstrap-fenced","expected_epoch":0,"issued_at_unix_millis":1,"operation":"FENCE","schema_version":"frontier.management-production-command/v3"}'
ADMISSION_BOOTSTRAP_RECEIPT = '{"command_nonce":"bootstrap-fenced","command_sha256":"sha256:0000000000000000000000000000000000000000000000000000000000000000","error_code":"controller_not_observed","observed_at_unix_millis":1,"operation":"FENCE","receipt_revision":"bootstrap:0","schema_version":"frontier.management-production-command-receipt/v1","status":"PENDING"}'
LOCAL_OBJECT_PVC = "frontier-local-object-store"
LOGICAL_AGENT_BASE = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/frontier/speed-add-agent-base:v1"
)
LOGICAL_CANDIDATE_OVERLAY = "123456789012.dkr.ecr.us-east-1.amazonaws.com/frontier/speed-add-scripted-overlay:v1"
LOGICAL_VERIFIER = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/frontier/speed-add-verifier:v1"
)
LOGICAL_COMPOSED_REPOSITORY = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/frontier/speed-add-composed"
)
TRAIN_URI = "s3://frontier-local-task-data/tasks/speed-add/v1/train.json"
EVAL_URI = "s3://frontier-local-task-data/tasks/speed-add/v1/eval.json"
RUNTIME_IMAGE_NAMES = (
    "management",
    "driver",
    "storage_controller",
    "agent_base",
    "candidate_overlay",
    "verifier",
    "litellm",
    "postgres",
)
TERMINAL_EVALUATIONS = {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"}
TERMINAL_RUNS = {"COMPLETED", "FAILED", "TIMED_OUT", "BUDGET_EXHAUSTED", "CANCELLED"}
ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
TASK_PLACEHOLDER_DIGEST = "sha256:" + "0" * 64
TASK_IMAGE_TAG_RE = re.compile(r"^sha-[0-9a-f]{40}$")
# The pinned Codex overlay attests only this GPU helper contract.
CODEX_AGENT_GPU_INTERFACE = "frontier.gpu-helper/v2"
MODEL_DOMAIN_PATTERN = r"(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+"
TASK_DEFINITIONS_PATH = REPOSITORY_ROOT / "config/task-definitions.json"
FORBIDDEN_CONFIG_KEYS = {
    "access_key",
    "access_key_id",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "secret_access_key",
    "session_token",
    "password",
    "private_key",
    "github_token",
    "registry_token",
    "openrouter_api_key",
    "modal_token_id",
    "modal_token_secret",
}


class DeployError(RuntimeError):
    pass


class ManagementHTTPError(DeployError):
    def __init__(self, status: int, path: str, detail: str):
        self.status = status
        super().__init__(f"Management HTTP {status} for {path}: {detail}")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def canonical_object_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def stable_ulid(seed: bytes) -> str:
    # Encode 128 deterministic bits in the canonical 26-character Crockford
    # alphabet. A 128-bit value always keeps the first character in 0..7.
    value = int.from_bytes(hashlib.sha256(seed).digest()[:16], "big")
    encoded = ["0"] * 26
    for index in range(25, -1, -1):
        encoded[index] = ULID_ALPHABET[value & 31]
        value >>= 5
    return "".join(encoded)


def dataset_seed_names() -> tuple[str, str]:
    train = (REPOSITORY_ROOT / "smoke/speed-add/datasets/train.json").read_bytes()
    evaluation = (REPOSITORY_ROOT / "smoke/speed-add/datasets/eval.json").read_bytes()
    suffix = hashlib.sha256(train + b"\0" + evaluation).hexdigest()[:16]
    return f"frontier-speed-add-datasets-{suffix}", f"frontier-seed-speed-add-{suffix}"


def duplicate_safe_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeployError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise DeployError(f"JSON input must be a regular, non-symlink file: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=duplicate_safe_object
        )
    except DeployError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DeployError(f"failed to read JSON {path}: {error}") from error


def require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeployError(f"{path} must be an object")
    return value


def require_exact_keys(
    value: Mapping[str, Any], expected: Iterable[str], path: str
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        raise DeployError(f"{path} keys differ; missing={missing}, extra={extra}")


def require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise DeployError(f"{path} must be a non-empty, whitespace-trimmed string")
    return value


def require_int(value: Any, path: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise DeployError(f"{path} must be an integer in {minimum}..={maximum}")
    return value


def runtime_source_commit(config: Mapping[str, Any], role: str) -> str:
    """Use the validated per-role source for staged runtime releases."""
    return config["application"].get("runtime_source_commits", {}).get(
        role, config["application"]["commit"]
    )


def source_image_tag_matches(tag: str, source_tag: str, digest: str) -> bool:
    """Keep the source interlock while addressing a corrected build by its content."""
    if not all(isinstance(value, str) for value in (tag, source_tag, digest)):
        return False
    if tag == source_tag:
        return True
    return bool(
        re.fullmatch(r"source-[0-9a-f]{64}", source_tag)
        and DIGEST_RE.fullmatch(digest)
        and digest != TASK_PLACEHOLDER_DIGEST
        and tag == source_tag + "-image-" + digest.removeprefix("sha256:")[:32]
    )


def validate_image(
    raw: Any, path: str, *, expected_tag: str | None = None
) -> dict[str, str]:
    image = require_mapping(raw, path)
    require_exact_keys(image, {"repository", "tag", "digest"}, path)
    repository = require_string(image["repository"], f"{path}.repository")
    if (
        any(character.isspace() for character in repository)
        or "@" in repository
        or repository.endswith("/")
    ):
        raise DeployError(f"{path}.repository is not a canonical image repository")
    tag = require_string(image["tag"], f"{path}.tag")
    if tag == "latest" or any(character.isspace() for character in tag):
        raise DeployError(f"{path}.tag must be immutable and cannot be latest")
    digest = require_string(image["digest"], f"{path}.digest")
    if not DIGEST_RE.fullmatch(digest):
        raise DeployError(f"{path}.digest must be a lowercase sha256 digest")
    if expected_tag is not None and not source_image_tag_matches(
        tag, expected_tag, digest
    ):
        raise DeployError(f"{path}.tag must equal {expected_tag}")
    return {
        "repository": repository,
        "tag": tag,
        "digest": digest,
        "reference": f"{repository}@{digest}",
    }


def atomic_write(path: Path, content: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_litellm_secret_document(value: Any) -> dict[str, Any]:
    document = require_mapping(value, "LiteLLM Secret")
    require_exact_keys(
        document,
        {"apiVersion", "kind", "metadata", "type", "stringData"},
        "LiteLLM Secret",
    )
    metadata = require_mapping(document["metadata"], "LiteLLM Secret.metadata")
    require_exact_keys(
        metadata, {"name", "namespace", "labels"}, "LiteLLM Secret.metadata"
    )
    if (
        document["apiVersion"] != "v1"
        or document["kind"] != "Secret"
        or document["type"] != "Opaque"
        or metadata["name"] != LITELLM_SECRET
        or metadata["namespace"] != LITELLM_NAMESPACE
    ):
        raise DeployError("local LiteLLM Secret identity differs")
    data = require_mapping(document["stringData"], "LiteLLM Secret.stringData")
    require_exact_keys(
        data,
        {
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_DB",
            "DATABASE_URL",
            "LITELLM_MASTER_KEY",
            "LITELLM_SALT_KEY",
        },
        "LiteLLM Secret.stringData",
    )
    if data["POSTGRES_USER"] != "litellm" or data["POSTGRES_DB"] != "litellm":
        raise DeployError("local LiteLLM database identity differs")
    password = require_string(data["POSTGRES_PASSWORD"], "LiteLLM POSTGRES_PASSWORD")
    master = require_string(data["LITELLM_MASTER_KEY"], "LiteLLM LITELLM_MASTER_KEY")
    salt = require_string(data["LITELLM_SALT_KEY"], "LiteLLM LITELLM_SALT_KEY")
    if not re.fullmatch(r"[A-Za-z0-9_-]{40,128}", password):
        raise DeployError("local LiteLLM database credential is malformed")
    if not re.fullmatch(r"sk-[A-Za-z0-9_-]{40,128}", master):
        raise DeployError("local LiteLLM master key is malformed")
    if not re.fullmatch(r"[A-Za-z0-9_-]{40,128}", salt):
        raise DeployError("local LiteLLM salt key is malformed")
    expected_url = f"postgresql://litellm:{password}@frontier-litellm-postgres.{LITELLM_NAMESPACE}.svc:5432/litellm"
    if data["DATABASE_URL"] != expected_url:
        raise DeployError(
            "local LiteLLM DATABASE_URL differs from the generated database identity"
        )
    return document


def ensure_litellm_secret_files(
    config: Mapping[str, Any], directory: Path | None = None
) -> tuple[Path, Path]:
    private = state_directory(config) if directory is None else directory
    private.mkdir(parents=True, exist_ok=True)
    if private.is_symlink() or not private.is_dir():
        raise DeployError(
            f"LiteLLM private state path must be a real directory: {private}"
        )
    os.chmod(private, 0o700)
    secret_path = private / "litellm-secrets.json"
    admin_path = private / "litellm-admin.env"
    if secret_path.exists():
        if (secret_path.stat().st_mode & 0o777) != 0o600:
            raise DeployError(f"LiteLLM Secret state must be mode 0600: {secret_path}")
        document = validate_litellm_secret_document(load_json(secret_path))
    else:
        if admin_path.exists():
            raise DeployError(
                "LiteLLM admin state exists without its authoritative Secret state"
            )
        password = secrets.token_urlsafe(36)
        master = "sk-" + secrets.token_urlsafe(36)
        salt = secrets.token_urlsafe(36)
        document = validate_litellm_secret_document(
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": LITELLM_SECRET,
                    "namespace": LITELLM_NAMESPACE,
                    "labels": {
                        "app.kubernetes.io/name": "frontier-litellm",
                        "app.kubernetes.io/part-of": "frontiercs2",
                        "frontier.dev/owner": "frontier-deploy",
                        "frontier.dev/site": config["site"]["name"],
                    },
                },
                "type": "Opaque",
                "stringData": {
                    "POSTGRES_USER": "litellm",
                    "POSTGRES_PASSWORD": password,
                    "POSTGRES_DB": "litellm",
                    "DATABASE_URL": f"postgresql://litellm:{password}@frontier-litellm-postgres.{LITELLM_NAMESPACE}.svc:5432/litellm",
                    "LITELLM_MASTER_KEY": master,
                    "LITELLM_SALT_KEY": salt,
                },
            }
        )
        atomic_write(secret_path, canonical_json(document), mode=0o600)
    master = document["stringData"]["LITELLM_MASTER_KEY"]
    expected_admin = f"UI_USERNAME=admin\nUI_PASSWORD={master}\n".encode()
    if admin_path.exists():
        if (
            admin_path.is_symlink()
            or not admin_path.is_file()
            or (admin_path.stat().st_mode & 0o777) != 0o600
        ):
            raise DeployError(
                f"LiteLLM admin state must be a regular mode-0600 file: {admin_path}"
            )
        if admin_path.read_bytes() != expected_admin:
            raise DeployError(
                "LiteLLM admin state conflicts with its authoritative Secret state"
            )
    else:
        atomic_write(admin_path, expected_admin, mode=0o600)
    return secret_path, admin_path


def terraform_output_values(encoded: Any) -> dict[str, Any]:
    raw = require_mapping(encoded, "Terraform outputs")
    result: dict[str, Any] = {}
    for key, record in raw.items():
        record = require_mapping(record, f"Terraform outputs.{key}")
        if "value" not in record:
            raise DeployError(f"Terraform output has no value: {key}")
        result[key] = record["value"]
    return result


def validate_outputs(config: Mapping[str, Any], outputs: Mapping[str, Any]) -> None:
    required = {
        "account_id",
        "aws_region",
        "cluster_name",
        "cluster_endpoint",
        "vpc_cidr",
        "service_ipv4_cidr",
        "platform_kms_key_arn",
        "artifact_bucket",
        "system_node_group_name",
        "cpu_node_group_name",
        "management_artifact_reader_role_arn",
        "management_artifact_reader_role_id",
        "efs_file_system_id",
        "efs_object_store_access_point_id",
        "efs_registry_access_point_id",
    }
    missing = sorted(required - set(outputs))
    if missing:
        raise DeployError(f"Terraform outputs are missing: {missing}")
    if outputs["account_id"] != config["site"]["expected_account_id"]:
        raise DeployError(
            "Terraform output account differs from site.expected_account_id"
        )
    if outputs["aws_region"] != config["site"]["aws_region"]:
        raise DeployError("Terraform output region differs from site.aws_region")
    if outputs["service_ipv4_cidr"] != config["network"]["service_ipv4_cidr"]:
        raise DeployError("Terraform Service CIDR differs from the frozen site config")
    expected_reader_arn = (
        f"arn:aws:iam::{config['site']['expected_account_id']}:role/"
        f"frontiercs2-{config['site']['name']}-management-artifact-reader"
    )
    if outputs["management_artifact_reader_role_arn"] != expected_reader_arn:
        raise DeployError(
            "Management artifact reader ARN differs from the exact site identity"
        )
    if not IAM_ROLE_ID_RE.fullmatch(str(outputs["management_artifact_reader_role_id"])):
        raise DeployError("Management artifact reader RoleId is invalid")
    for field, prefix in (
        ("efs_file_system_id", "fs-"),
        ("efs_object_store_access_point_id", "fsap-"),
        ("efs_registry_access_point_id", "fsap-"),
    ):
        if not isinstance(outputs[field], str) or not outputs[field].startswith(prefix):
            raise DeployError(f"Terraform output {field} is invalid")


def k8s_list(items: Sequence[Mapping[str, Any]]) -> bytes:
    return canonical_json({"apiVersion": "v1", "kind": "List", "items": list(items)})


def platform_labels(config: Mapping[str, Any], component: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/part-of": "frontiercs2",
        "app.kubernetes.io/name": component,
        "frontier.dev/site": config["site"]["name"],
        "frontier.dev/owner": "frontier-deploy",
    }


def render_bootstrap_manifest(
    config: Mapping[str, Any], outputs: Mapping[str, Any]
) -> bytes:
    validate_outputs(config, outputs)
    network, execution = config["network"], config["execution"]
    registry_labels = platform_labels(config, "frontier-registry")
    registry_image = config["images"]["registry"]["reference"]
    registry_handle = (
        f"{outputs['efs_file_system_id']}::{outputs['efs_registry_access_point_id']}"
    )
    namespace = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": PLATFORM_NAMESPACE,
            "labels": {
                **platform_labels(config, "frontier-system"),
                "pod-security.kubernetes.io/enforce": "restricted",
                "pod-security.kubernetes.io/audit": "restricted",
                "pod-security.kubernetes.io/warn": "restricted",
            },
        },
    }
    registry_pv = {
        "apiVersion": "v1",
        "kind": "PersistentVolume",
        "metadata": {"name": "frontier-registry", "labels": registry_labels},
        "spec": {
            "capacity": {"storage": f"{execution['registry_storage_gib']}Gi"},
            "volumeMode": "Filesystem",
            "accessModes": ["ReadWriteMany"],
            "persistentVolumeReclaimPolicy": "Retain",
            "storageClassName": "",
            "mountOptions": ["tls"],
            "csi": {"driver": "efs.csi.aws.com", "volumeHandle": registry_handle},
        },
    }
    registry_pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": "frontier-registry",
            "namespace": PLATFORM_NAMESPACE,
            "labels": registry_labels,
        },
        "spec": {
            "accessModes": ["ReadWriteMany"],
            "volumeMode": "Filesystem",
            "storageClassName": "",
            "volumeName": "frontier-registry",
            "resources": {
                "requests": {"storage": f"{execution['registry_storage_gib']}Gi"}
            },
        },
    }
    registry_deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": REGISTRY_SERVICE,
            "namespace": PLATFORM_NAMESPACE,
            "labels": registry_labels,
        },
        "spec": {
            "replicas": 1,
            "strategy": {"type": "Recreate"},
            "selector": {
                "matchLabels": {"app.kubernetes.io/name": "frontier-registry"}
            },
            "template": {
                "metadata": {
                    "labels": {
                        **registry_labels,
                        "frontier.dev/network-role": "registry",
                    }
                },
                "spec": {
                    "automountServiceAccountToken": False,
                    "nodeSelector": {"frontier.dev/pool": "system"},
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 65532,
                        "runAsGroup": 65532,
                        "fsGroup": 65532,
                        "fsGroupChangePolicy": "OnRootMismatch",
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "containers": [
                        {
                            "name": "registry",
                            "image": registry_image,
                            "imagePullPolicy": "IfNotPresent",
                            "ports": [{"name": "registry", "containerPort": 5000}],
                            "env": [
                                {
                                    "name": "REGISTRY_STORAGE_DELETE_ENABLED",
                                    "value": "true",
                                }
                            ],
                            "readinessProbe": {
                                "httpGet": {"path": "/v2/", "port": "registry"},
                                "periodSeconds": 3,
                                "failureThreshold": 40,
                            },
                            "resources": {
                                "requests": {"cpu": "50m", "memory": "64Mi"},
                                "limits": {"cpu": "500m", "memory": "512Mi"},
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "volumeMounts": [
                                {"name": "data", "mountPath": "/var/lib/registry"},
                                {"name": "tmp", "mountPath": "/tmp"},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "data",
                            "persistentVolumeClaim": {"claimName": "frontier-registry"},
                        },
                        {"name": "tmp", "emptyDir": {"sizeLimit": "32Mi"}},
                    ],
                },
            },
        },
    }
    registry_service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": REGISTRY_SERVICE,
            "namespace": PLATFORM_NAMESPACE,
            "labels": registry_labels,
        },
        "spec": {
            "type": "ClusterIP",
            "clusterIP": network["registry_cluster_ip"],
            "selector": {"app.kubernetes.io/name": "frontier-registry"},
            "ports": [{"name": "registry", "port": 5000, "targetPort": "registry"}],
        },
    }
    registry_policy = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": "frontier-registry-boundary",
            "namespace": PLATFORM_NAMESPACE,
            "labels": registry_labels,
        },
        "spec": {
            "podSelector": {"matchLabels": {"frontier.dev/network-role": "registry"}},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [
                {
                    "from": [{"podSelector": {
                        "matchLabels": platform_labels(config, "frontier-task-seed")
                    }}],
                    "ports": [{"protocol": "TCP", "port": 5000}],
                },
                {
                    "from": [{"ipBlock": {"cidr": outputs["vpc_cidr"]}}],
                    "ports": [{"protocol": "TCP", "port": 5000}],
                },
                {
                    "from": [
                        {
                            "namespaceSelector": {
                                "matchLabels": {"frontier.dev/owner": "management"}
                            },
                            "podSelector": {
                                "matchLabels": {"frontier.dev/network-role": "driver"}
                            },
                        }
                    ],
                    "ports": [{"protocol": "TCP", "port": 5000}],
                },
            ],
            "egress": [],
        },
    }
    seed_policy = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": "frontier-task-seed-registry-egress",
            "namespace": PLATFORM_NAMESPACE,
            "labels": platform_labels(config, "frontier-task-seed"),
        },
        "spec": {
            "podSelector": {"matchLabels": platform_labels(config, "frontier-task-seed")},
            "policyTypes": ["Egress"],
            "egress": [{
                "to": [
                    {"podSelector": {"matchLabels": {
                        "frontier.dev/network-role": "registry",
                    }}},
                    {"ipBlock": {"cidr": f"{network['registry_cluster_ip']}/32"}},
                ],
                "ports": [{"protocol": "TCP", "port": 5000}],
            }],
        },
    }
    return k8s_list(
        [
            namespace,
            registry_pv,
            registry_pvc,
            registry_deployment,
            registry_service,
            registry_policy,
            seed_policy,
        ]
    )


def validate_runtime_images(
    images: Any, config: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    raw = require_mapping(images, "runtime images")
    require_exact_keys(raw, set(RUNTIME_IMAGE_NAMES), "runtime images")
    from .runtime_reference_compat import reviewed_public_runtime_records

    reviewed_public = reviewed_public_runtime_records(config, raw)
    endpoint = f"{config['network']['registry_cluster_ip']}:5000/"
    result: dict[str, dict[str, str]] = {}
    for name in RUNTIME_IMAGE_NAMES:
        record = require_mapping(raw[name], f"runtime images.{name}")
        require_exact_keys(
            record,
            {"tag", "digest", "reference", "source_sha256"},
            f"runtime images.{name}",
        )
        tag = require_string(record["tag"], f"runtime images.{name}.tag")
        digest = require_string(record["digest"], f"runtime images.{name}.digest")
        reference = require_string(
            record["reference"], f"runtime images.{name}.reference"
        )
        source_sha256 = require_string(
            record["source_sha256"], f"runtime images.{name}.source_sha256"
        )
        # Tags always have one explicit final tag. The registry host itself has
        # a port, so remove only the final colon-delimited component.
        repository = tag.rsplit(":", 1)[0]
        if (
            not (tag.startswith(endpoint) or (reviewed_public and name in ("management", "driver")))
            or not DIGEST_RE.fullmatch(digest)
            or reference != f"{repository}@{digest}"
        ):
            raise DeployError(
                f"runtime images.{name} is not pinned below the site registry"
            )
        if not DIGEST_RE.fullmatch(source_sha256):
            raise DeployError(f"runtime images.{name}.source_sha256 is invalid")
        result[name] = dict(record)
    return result


def api_cidrs(
    config: Mapping[str, Any], kubeconfig: Path, *, receipt_path: Path | None = None
) -> list[str]:
    # Resolve Service DNAT targets inside the cluster, never deployment-host DNS.
    from .api_routes import discover_api_routes

    return discover_api_routes(config, kubeconfig, receipt_path=receipt_path)["cidrs"]


def admission_control_rbac() -> list[dict[str, Any]]:
    """Static namespace-local grants from the pinned production manifests."""
    operator = "frontier-production-operator"
    inventory = "frontier-management-iam-inventory-reader"

    def metadata(name: str, app: str) -> dict[str, Any]:
        return {
            "name": name,
            "namespace": PLATFORM_NAMESPACE,
            "labels": {
                "app.kubernetes.io/name": app,
                "app.kubernetes.io/part-of": "frontiercs2",
            },
        }

    def binding(name: str, service_account: str) -> dict[str, Any]:
        return {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": metadata(name, service_account),
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": name,
            },
            "subjects": [
                {
                    "apiGroup": "",
                    "kind": "ServiceAccount",
                    "name": service_account,
                    "namespace": PLATFORM_NAMESPACE,
                }
            ],
        }

    return [
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": metadata(operator, operator),
            "automountServiceAccountToken": False,
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": metadata(operator, operator),
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["configmaps"],
                    "resourceNames": [ADMISSION_COMMAND_MAP],
                    "verbs": ["get", "patch"],
                },
                {
                    "apiGroups": [""],
                    "resources": ["configmaps"],
                    "resourceNames": [ADMISSION_RECEIPT_MAP],
                    "verbs": ["get"],
                },
            ],
        },
        binding(operator, operator),
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": metadata(inventory, MANAGEMENT_SERVICE),
            "rules": [
                {"apiGroups": [""], "resources": ["configmaps"], "verbs": ["list"]},
                {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["list"]},
                {
                    "apiGroups": ["rbac.authorization.k8s.io"],
                    "resources": ["roles", "rolebindings"],
                    "verbs": ["list"],
                },
            ],
        },
        binding(inventory, MANAGEMENT_SERVICE),
    ]


def management_result_resources() -> dict:
    return {
        "requests": {"cpu": "100m", "memory": "4Gi", "ephemeral-storage": "34Gi"},
        "limits": {"cpu": "1", "memory": "4Gi", "ephemeral-storage": "34Gi"},
    }


def gateway_artifact_retention_init(management_image: str) -> dict:
    """Provision only the private Management data-volume retention leaf."""
    return {
        "name": "gateway-artifact-retention",
        "image": management_image,
        "imagePullPolicy": "IfNotPresent",
        "command": [
            "sh",
            "-ec",
            'umask 077\nretained=/var/lib/frontier/gateway-terminal-retained\nif [ ! -e "$retained" ] && [ ! -L "$retained" ]; then\n    mkdir -m 0700 "$retained"\nfi\ntest ! -L "$retained"\ntest -d "$retained"\ntest "$(stat -c %u "$retained")" = 65532\ntest "$(stat -c %a "$retained")" = 700\n',
        ],
        "resources": {
            "requests": {"cpu": "10m", "memory": "16Mi"},
            "limits": {"cpu": "100m", "memory": "64Mi"},
        },
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": True,
            "capabilities": {"drop": ["ALL"]},
        },
        "volumeMounts": [{"name": "data", "mountPath": "/var/lib/frontier"}],
    }


def management_diagnostic_source_mount(
    object_claim: str = "frontier-object-seed",
) -> tuple[dict, dict]:
    """Trusted local diagnostic GC needs artifact writes, not dataset writes."""
    return (
        {
            "name": "objects",
            "persistentVolumeClaim": {"claimName": object_claim},
        },
        {
            "name": "objects",
            "mountPath": "/var/lib/frontier-local/artifacts",
            "subPath": "artifacts",
        },
    )


def management_api_routes_rbac() -> list[dict[str, Any]]:
    """Read only the control-plane Service destinations for native dispatch gating."""
    name = "frontier-management-api-routes"
    metadata = {"name": name, "namespace": "default"}
    return [
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": dict(metadata),
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["services", "endpoints"],
                    "resourceNames": ["kubernetes"],
                    "verbs": ["get"],
                },
                {
                    "apiGroups": ["discovery.k8s.io"],
                    "resources": ["endpointslices"],
                    "verbs": ["list"],
                },
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": dict(metadata),
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": name,
            },
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": "frontier-management",
                    "namespace": PLATFORM_NAMESPACE,
                }
            ],
        },
    ]


def management_diagnostics_rbac(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Add trusted diagnostic reads without replacing existing site RBAC."""
    name = "frontier-management-diagnostics"
    metadata = {"name": name, "labels": platform_labels(config, "frontier-management")}
    return [
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": metadata,
            "rules": [
                {"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"]},
                {"apiGroups": [""], "resources": ["events"], "verbs": ["list"]},
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": metadata,
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": name,
            },
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": "frontier-management",
                    "namespace": PLATFORM_NAMESPACE,
                }
            ],
        },
    ]


def modal_observer_rbac(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Exact site credential read for trusted Management; no runtime objects."""
    if not config["features"]["modal_gpu"]:
        return []
    labels = platform_labels(config, "frontier-management")
    modal = config["modal"]
    # The native Modal observer reads only the site's configured credential.
    observer_role = "frontier-management-modal-observer"
    return [
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {
                "name": observer_role,
                "namespace": PLATFORM_NAMESPACE,
                "labels": labels,
            },
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["secrets"],
                    "resourceNames": [modal["credentials_secret"]],
                    "verbs": ["get"],
                }
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {
                "name": observer_role,
                "namespace": PLATFORM_NAMESPACE,
                "labels": labels,
            },
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": observer_role,
            },
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": "frontier-management",
                    "namespace": PLATFORM_NAMESPACE,
                }
            ],
        },
    ]


def render_runtime_manifest(
    config: Mapping[str, Any],
    outputs: Mapping[str, Any],
    runtime_images: Any,
    kubernetes_api_cidrs: Sequence[str],
    *,
    simulator_bootstrap: bool = False,
) -> bytes:
    validate_outputs(config, outputs)
    images = validate_runtime_images(runtime_images, config)
    if not kubernetes_api_cidrs or any(
        ipaddress.ip_network(value).prefixlen != 32 for value in kubernetes_api_cidrs
    ):
        raise DeployError("Kubernetes API CIDRs must be a non-empty exact /32 set")
    labels = platform_labels(config, "frontier-management")
    controller_labels = platform_labels(config, "frontier-storage-controller")
    execution, network = config["execution"], config["network"]
    service_network = ipaddress.ip_network(network["service_ipv4_cidr"])
    dns_cidr = f"{service_network.network_address + 10}/32"
    object_handle = f"{outputs['efs_file_system_id']}::{outputs['efs_object_store_access_point_id']}"
    registry = f"{network['registry_cluster_ip']}:5000"
    management_image = images["management"]["reference"]
    driver_image = images["driver"]["reference"]
    pull_secret = config["management"]["registry_pull_secret_name"]
    image_pull_secrets = [] if pull_secret is None else [{"name": pull_secret}]
    train_bytes = (REPOSITORY_ROOT / "smoke/speed-add/datasets/train.json").read_bytes()
    eval_bytes = (REPOSITORY_ROOT / "smoke/speed-add/datasets/eval.json").read_bytes()
    dataset_config_name, seed_job_name = dataset_seed_names()
    # Management derives the exact dataset and image maps for each dispatch.
    # These static entries are overrides for the scripted speed-add fixtures;
    # seeded catalog tasks use their content-derived local paths and
    # digest-pinned in-cluster image references.
    dataset_map = json.dumps(
        {TRAIN_URI: "datasets/train.json", EVAL_URI: "datasets/eval.json"},
        separators=(",", ":"),
    )
    registry_map = json.dumps(
        {
            LOGICAL_AGENT_BASE: images["agent_base"]["tag"],
            LOGICAL_CANDIDATE_OVERLAY: images["candidate_overlay"]["tag"],
            LOGICAL_VERIFIER: images["verifier"]["tag"],
            LOGICAL_COMPOSED_REPOSITORY: f"{registry}/frontier/speed-add-composed",
        },
        separators=(",", ":"),
    )
    items: list[dict[str, Any]] = []
    items.append(
        {
            "apiVersion": "storage.k8s.io/v1",
            "kind": "StorageClass",
            "metadata": {"name": "frontier-gp3", "labels": labels},
            "provisioner": "ebs.csi.aws.com",
            "reclaimPolicy": "Retain",
            "allowVolumeExpansion": True,
            "volumeBindingMode": "WaitForFirstConsumer",
            "parameters": {
                "type": "gp3",
                "encrypted": "true",
                "kmsKeyId": outputs["platform_kms_key_arn"],
                "csi.storage.k8s.io/fstype": "ext4",
            },
        }
    )
    items.extend(
        [
            {
                "apiVersion": "v1",
                "kind": "ServiceAccount",
                "metadata": {
                    "name": "frontier-management",
                    "namespace": PLATFORM_NAMESPACE,
                    "labels": labels,
                    "annotations": {
                        "eks.amazonaws.com/role-arn": outputs[
                            "management_artifact_reader_role_arn"
                        ],
                        "eks.amazonaws.com/sts-regional-endpoints": "true",
                    },
                },
                "automountServiceAccountToken": True,
            },
            {
                "apiVersion": "v1",
                "kind": "ServiceAccount",
                "metadata": {
                    "name": "frontier-storage-controller",
                    "namespace": PLATFORM_NAMESPACE,
                    "labels": controller_labels,
                },
                "automountServiceAccountToken": True,
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "ClusterRole",
                "metadata": {"name": "frontier-management-preview", "labels": labels},
                "rules": [
                    {
                        "apiGroups": [""],
                        "resources": ["namespaces"],
                        "verbs": ["create", "delete", "get", "list", "patch"],
                    },
                    {
                        "apiGroups": [""],
                        "resources": ["pods"],
                        "verbs": ["delete", "get", "list"],
                    },
                    {
                        "apiGroups": [""],
                        "resources": [
                            "configmaps",
                            "limitranges",
                            "resourcequotas",
                            "serviceaccounts",
                        ],
                        "verbs": ["create", "delete", "get", "patch"],
                    },
                    {
                        "apiGroups": ["batch"],
                        "resources": ["jobs"],
                        "verbs": ["create", "delete", "get", "list", "patch"],
                    },
                    {
                        "apiGroups": ["networking.k8s.io"],
                        "resources": ["networkpolicies"],
                        "verbs": ["create", "delete", "get", "patch"],
                    },
                    {
                        "apiGroups": ["rbac.authorization.k8s.io"],
                        "resources": ["roles", "rolebindings"],
                        "verbs": [
                            "bind",
                            "create",
                            "delete",
                            "escalate",
                            "get",
                            "patch",
                        ],
                    },
                    {
                        "apiGroups": ["authentication.k8s.io"],
                        "resources": ["tokenreviews"],
                        "verbs": ["create"],
                    },
                ],
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "ClusterRoleBinding",
                "metadata": {"name": "frontier-management-preview", "labels": labels},
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "ClusterRole",
                    "name": "frontier-management-preview",
                },
                "subjects": [
                    {
                        "kind": "ServiceAccount",
                        "name": "frontier-management",
                        "namespace": PLATFORM_NAMESPACE,
                    }
                ],
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "ClusterRole",
                "metadata": {
                    "name": "frontier-storage-controller",
                    "labels": controller_labels,
                },
                "rules": [
                    {
                        "apiGroups": [""],
                        "resources": ["namespaces"],
                        "verbs": ["get", "list"],
                    },
                    {
                        "apiGroups": [""],
                        "resources": ["persistentvolumes"],
                        "verbs": ["create", "delete", "get", "list"],
                    },
                    {
                        "apiGroups": [""],
                        "resources": ["persistentvolumeclaims"],
                        "verbs": ["create", "get"],
                    },
                    {
                        "apiGroups": ["networking.k8s.io"],
                        "resources": ["networkpolicies"],
                        "verbs": ["create", "delete", "get", "update"],
                    },
                ],
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "ClusterRoleBinding",
                "metadata": {
                    "name": "frontier-storage-controller",
                    "labels": controller_labels,
                },
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "ClusterRole",
                    "name": "frontier-storage-controller",
                },
                "subjects": [
                    {
                        "kind": "ServiceAccount",
                        "name": "frontier-storage-controller",
                        "namespace": PLATFORM_NAMESPACE,
                    }
                ],
            },
        ]
    )
    items.extend(
        [
            {
                "apiVersion": "v1",
                "kind": "PersistentVolume",
                "metadata": {
                    "name": "frontier-object-seed",
                    "labels": controller_labels,
                },
                "spec": {
                    "capacity": {"storage": f"{execution['object_store_gib']}Gi"},
                    "volumeMode": "Filesystem",
                    "accessModes": ["ReadWriteMany"],
                    "persistentVolumeReclaimPolicy": "Retain",
                    "storageClassName": "",
                    "mountOptions": ["tls"],
                    "csi": {"driver": "efs.csi.aws.com", "volumeHandle": object_handle},
                },
            },
            {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {
                    "name": "frontier-object-seed",
                    "namespace": PLATFORM_NAMESPACE,
                    "labels": controller_labels,
                },
                "spec": {
                    "accessModes": ["ReadWriteMany"],
                    "volumeMode": "Filesystem",
                    "storageClassName": "",
                    "volumeName": "frontier-object-seed",
                    "resources": {
                        "requests": {"storage": f"{execution['object_store_gib']}Gi"}
                    },
                },
            },
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": dataset_config_name,
                    "namespace": PLATFORM_NAMESPACE,
                    "labels": controller_labels,
                },
                "immutable": True,
                "data": {
                    "train.json": train_bytes.decode(),
                    "eval.json": eval_bytes.decode(),
                },
            },
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "name": seed_job_name,
                    "namespace": PLATFORM_NAMESPACE,
                    "labels": controller_labels,
                },
                "spec": {
                    "backoffLimit": 3,
                    "template": {
                        "metadata": {"labels": controller_labels},
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
                                    "image": images["storage_controller"]["reference"],
                                    "imagePullPolicy": "IfNotPresent",
                                    "command": [
                                        "/usr/local/bin/python3",
                                        "-c",
                                        "import os,shutil; os.makedirs('/data/datasets',exist_ok=True); shutil.copyfile('/seed/train.json','/data/datasets/train.json'); shutil.copyfile('/seed/eval.json','/data/datasets/eval.json')",
                                    ],
                                    "resources": {
                                        "requests": {"cpu": "10m", "memory": "32Mi"},
                                        "limits": {"cpu": "100m", "memory": "128Mi"},
                                    },
                                    "securityContext": {
                                        "allowPrivilegeEscalation": False,
                                        "readOnlyRootFilesystem": True,
                                        "capabilities": {"drop": ["ALL"]},
                                    },
                                    "volumeMounts": [
                                        {"name": "data", "mountPath": "/data"},
                                        {
                                            "name": "seed",
                                            "mountPath": "/seed",
                                            "readOnly": True,
                                        },
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "data",
                                    "persistentVolumeClaim": {
                                        "claimName": "frontier-object-seed"
                                    },
                                },
                                {
                                    "name": "seed",
                                    "configMap": {"name": dataset_config_name},
                                },
                            ],
                        },
                    },
                },
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": "frontier-storage-controller",
                    "namespace": PLATFORM_NAMESPACE,
                    "labels": controller_labels,
                },
                "spec": {
                    "replicas": 1,
                    "selector": {
                        "matchLabels": {
                            "app.kubernetes.io/name": "frontier-storage-controller"
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                **controller_labels,
                                "frontier.dev/network-role": "storage-controller",
                            }
                        },
                        "spec": {
                            "serviceAccountName": "frontier-storage-controller",
                            "automountServiceAccountToken": True,
                            "nodeSelector": {"frontier.dev/pool": "system"},
                            "securityContext": {
                                "runAsNonRoot": True,
                                "runAsUser": 65532,
                                "runAsGroup": 65532,
                                "seccompProfile": {"type": "RuntimeDefault"},
                            },
                            "containers": [
                                {
                                    "name": "controller",
                                    "image": images["storage_controller"]["reference"],
                                    "imagePullPolicy": "IfNotPresent",
                                    "env": [
                                        {
                                            "name": "FRONTIER_EFS_OBJECT_VOLUME_HANDLE",
                                            "value": object_handle,
                                        },
                                        {
                                            "name": "FRONTIER_OBJECT_STORE_GIB",
                                            "value": str(execution["object_store_gib"]),
                                        },
                                        {
                                            "name": "FRONTIER_RECONCILE_SECONDS",
                                            "value": "1",
                                        },
                                        {
                                            "name": "FRONTIER_AGENT_EXTERNAL_EGRESS",
                                            "value": "true"
                                            if execution["agent_external_egress"]
                                            else "false",
                                        },
                                        {
                                            "name": "FRONTIER_DNS_CIDR",
                                            "value": dns_cidr,
                                        },
                                        {
                                            "name": "FRONTIER_REGISTRY_SERVICE_CIDR",
                                            "value": network["registry_cluster_ip"]
                                            + "/32",
                                        },
                                    ],
                                    "resources": {
                                        "requests": {"cpu": "25m", "memory": "64Mi"},
                                        "limits": {"cpu": "250m", "memory": "256Mi"},
                                    },
                                    "securityContext": {
                                        "allowPrivilegeEscalation": False,
                                        "readOnlyRootFilesystem": True,
                                        "capabilities": {"drop": ["ALL"]},
                                    },
                                }
                            ],
                        },
                    },
                },
            },
        ]
    )
    if not simulator_bootstrap:
        items.extend(admission_control_rbac())
    config_data = {
        "MANAGEMENT_BIND_ADDR": "0.0.0.0:8080",
        "MANAGEMENT_DATABASE_URL": "sqlite:///var/lib/frontier/management.sqlite3",
        "MANAGEMENT_DRIVER_MODE": "simulator" if simulator_bootstrap else "kubernetes",
        "MANAGEMENT_FAKE_DRIVER": "false",
        "MANAGEMENT_DRIVER_BACKENDS": "local-k8s",
        "MANAGEMENT_DRIVER_IMAGE": driver_image,
        "MANAGEMENT_DRIVER_CONTROLLER_POLL_MILLIS": "15000",
        "MANAGEMENT_MAX_ACTIVE_DRIVERS": str(execution["max_active_drivers"]),
        "MANAGEMENT_NAMESPACE": PLATFORM_NAMESPACE,
        "MANAGEMENT_INTERNAL_BASE_URL": f"http://{MANAGEMENT_SERVICE}.{PLATFORM_NAMESPACE}.svc:8080",
        "MANAGEMENT_SERVICE_CIDR": f"{network['management_cluster_ip']}/32",
        "MANAGEMENT_KUBERNETES_API_CIDRS": ",".join(kubernetes_api_cidrs),
        # The local-k8s Driver profile writes artifacts to the shared EFS
        # object store, so Management serves those bytes itself instead of
        # presigning the (empty) real S3 bucket. The region still shapes the
        # frozen Driver dispatch document.
        "MANAGEMENT_ARTIFACT_REGION": config["site"]["aws_region"],
        "MANAGEMENT_ARTIFACT_BUCKET": outputs["artifact_bucket"],
        "MANAGEMENT_ARTIFACT_LOCAL_ROOT": "/var/lib/frontier-local",
        "MANAGEMENT_DATASET_LOCAL_ROOT": "/var/lib/frontier-local",
        "MANAGEMENT_REGISTRY_REGION": config["site"]["aws_region"],
        "MANAGEMENT_COMPOSED_IMAGE_REPOSITORY": LOGICAL_COMPOSED_REPOSITORY,
        "MANAGEMENT_MODAL_ENVIRONMENT": "preview-disabled",
        "MANAGEMENT_MODAL_APP_NAME": "preview-disabled",
        "MANAGEMENT_MODAL_ECR_PULL_SECRET_REF": "preview-disabled",
        "MANAGEMENT_DRIVER_LOCAL_OBJECT_STORE_PVC": LOCAL_OBJECT_PVC,
        "MANAGEMENT_DRIVER_LOCAL_DATASET_MAP_JSON": dataset_map,
        "MANAGEMENT_DRIVER_LOCAL_REGISTRY_MAP_JSON": registry_map,
        "MANAGEMENT_DRIVER_LOCAL_REGISTRY_PORT": "5000",
        "MANAGEMENT_SEED_DEMO": "false",
        "MANAGEMENT_CORS_ORIGINS": "",
        "RUST_LOG": "frontier_management_api=info,tower_http=info",
    }
    allowlists = config["execution"].get("verifier_node_allowlists", [])
    if allowlists:
        config_data["MANAGEMENT_VERIFIER_NODE_ALLOWLISTS"] = json.dumps(
            allowlists, sort_keys=True, separators=(",", ":")
        )
    required_tasks = config["execution"].get("verifier_placement_required_task_ids", [])
    if required_tasks:
        config_data["MANAGEMENT_VERIFIER_PLACEMENT_REQUIRED_TASK_IDS"] = json.dumps(
            sorted(required_tasks), separators=(",", ":")
        )
    if not simulator_bootstrap:
        config_data['MANAGEMENT_EKS_STATUS_CLUSTER'] = outputs['cluster_name']
        config_data['MANAGEMENT_EKS_STATUS_REGION'] = config['site']['aws_region']
        management_role = next(item for item in items if item['kind'] == 'ClusterRole'
                               and item['metadata']['name'] == 'frontier-management-preview')
        management_role['rules'].append(
            {'apiGroups': [''], 'resources': ['nodes'], 'verbs': ['get', 'list']})
        from .manual_capacity import capacity_settings, observer_rbac
        capacity = capacity_settings(config)
        if capacity is not None:
            if capacity['cluster_arn'].rsplit('/', 1)[-1] != outputs['cluster_name']:
                raise DeployError('manual capacity cluster differs from Terraform outputs')
            if capacity['c6i']['node_group_arn'].split('/')[2] != outputs['cpu_node_group_name']:
                raise DeployError('manual c6i pool must be the exact Terraform CPU nodegroup')
            if any(capacity[role]['node_group_arn'].split('/')[2] == outputs['system_node_group_name'] for role in ('m6i', 'c6i')):
                raise DeployError('manual capacity cannot target the system nodegroup')
            config_data['MANAGEMENT_PREVIEW_CAPACITY'] = json.dumps(capacity, separators=(',', ':'))
            items.extend(observer_rbac())
        items.extend(management_diagnostics_rbac(config))
        items.extend(management_api_routes_rbac())
        # The EFS parent stays read-only; an artifacts-only overlay permits
        # trusted diagnostic GC. Retained snapshots live on the private EBS PVC.
        config_data["MANAGEMENT_GATEWAY_ARTIFACT_SOURCE_ROOT"] = (
            "/var/lib/frontier-local/artifacts"
        )
        config_data["MANAGEMENT_GATEWAY_ARTIFACT_RETAINED_ROOT"] = (
            "/var/lib/frontier/gateway-terminal-retained"
        )
        config_data["MANAGEMENT_LITELLM_SERVICE_CIDR"] = (
            network["litellm_cluster_ip"] + "/32"
        )
        config_data["MANAGEMENT_LITELLM_KEY_WORKER_IMAGE"] = management_image
    if config["features"]["modal_gpu"]:
        modal = config["modal"]
        addresses = outputs.get("nat_gateway_public_ips", [])
        if not addresses or any(
            ipaddress.ip_address(value).version != 4
            or not ipaddress.ip_address(value).is_global
            for value in addresses
        ):
            raise DeployError("Modal GPU requires the site's public NAT gateway IPs")
        config_data.update(
            {
                "MANAGEMENT_MODAL_ENVIRONMENT": modal["environment"],
                "MANAGEMENT_MODAL_APP_NAME": modal["app_name"],
                "MANAGEMENT_MODAL_ECR_PULL_SECRET_REF": modal.get("ecr_pull_secret_ref", ""),
                "MANAGEMENT_MODAL_CREDENTIALS_SECRET": modal["credentials_secret"],
                "MANAGEMENT_MODAL_INBOUND_CIDRS": ",".join(
                    f"{value}/32" for value in sorted(set(addresses))
                ),
            }
        )
        items.extend(modal_observer_rbac(config))
        controller = next(
            item
            for item in items
            if item["kind"] == "Deployment"
            and item["metadata"]["name"] == "frontier-storage-controller"
        )
        pod = controller["spec"]["template"]["spec"]
        pod["securityContext"]["fsGroup"] = 65532
        container = pod["containers"][0]
        container["env"].extend(
            [
                {"name": "FRONTIER_MODAL_GPU", "value": "true"},
                {
                    "name": "FRONTIER_MODAL_SECRET_NAME",
                    "value": modal["credentials_secret"],
                },
            ]
        )
        container["volumeMounts"] = [
            {
                "name": "modal",
                "mountPath": "/var/run/secrets/frontier-modal",
                "readOnly": True,
            }
        ]
        pod["volumes"] = [
            {
                "name": "modal",
                "secret": {
                    "secretName": modal["credentials_secret"],
                    "defaultMode": 288,
                },
            }
        ]
        role = next(
            item
            for item in items
            if item["kind"] == "ClusterRole"
            and item["metadata"]["name"] == "frontier-storage-controller"
        )
        role["rules"].extend(
            [
                {
                    "apiGroups": ["batch"],
                    "resources": ["jobs"],
                    "verbs": ["get", "list"],
                },
                {
                    "apiGroups": [""],
                    "resources": ["secrets"],
                    "resourceNames": [modal["credentials_secret"]],
                    "verbs": ["get"],
                },
                {"apiGroups": [""], "resources": ["secrets"], "verbs": ["create"]},
            ]
        )
    items.extend(
        [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": "frontier-management-config",
                    "namespace": PLATFORM_NAMESPACE,
                    "labels": labels,
                },
                "data": config_data,
            },
            {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {
                    "name": "frontier-management-data",
                    "namespace": PLATFORM_NAMESPACE,
                    "labels": labels,
                },
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "storageClassName": "frontier-gp3",
                    "resources": {
                        "requests": {
                            "storage": f"{config['management']['storage_gib']}Gi"
                        }
                    },
                },
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": "frontier-management",
                    "namespace": PLATFORM_NAMESPACE,
                    "labels": {
                        **labels,
                        "frontier.dev/deployment-profile": "executable-preview",
                    },
                },
                # The config-sha256 annotation makes a ConfigMap-only change roll
                # the Deployment; envFrom alone would keep stale pods running.
                "spec": {
                    "replicas": 1,
                    "strategy": {"type": "Recreate"},
                    "selector": {
                        "matchLabels": {"app.kubernetes.io/name": "frontier-management"}
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                **labels,
                                "frontier.dev/network-role": "management",
                            },
                            "annotations": {
                                "frontier.dev/config-sha256": canonical_object_sha256(
                                    config_data
                                )
                            },
                        },
                        "spec": {
                            "serviceAccountName": "frontier-management",
                            "automountServiceAccountToken": True,
                            "enableServiceLinks": False,
                            "terminationGracePeriodSeconds": 30,
                            "imagePullSecrets": image_pull_secrets,
                            "nodeSelector": {"frontier.dev/pool": "system"},
                            "securityContext": {
                                "runAsNonRoot": True,
                                "runAsUser": 65532,
                                "runAsGroup": 65532,
                                "fsGroup": 65532,
                                "fsGroupChangePolicy": "OnRootMismatch",
                                "seccompProfile": {"type": "RuntimeDefault"},
                            },
                            "containers": [
                                {
                                    "name": "management",
                                    "image": management_image,
                                    "imagePullPolicy": "IfNotPresent",
                                    "envFrom": [
                                        {
                                            "configMapRef": {
                                                "name": "frontier-management-config"
                                            }
                                        }
                                    ],
                                    "ports": [{"name": "http", "containerPort": 8080}],
                                    "readinessProbe": {
                                        "httpGet": {"path": "/healthz", "port": "http"},
                                        "periodSeconds": 5,
                                        "failureThreshold": 24,
                                    },
                                    "livenessProbe": {
                                        "httpGet": {"path": "/healthz", "port": "http"},
                                        "initialDelaySeconds": 20,
                                        "periodSeconds": 10,
                                    },
                                    "resources": management_result_resources(),
                                    "securityContext": {
                                        "allowPrivilegeEscalation": False,
                                        "readOnlyRootFilesystem": True,
                                        "capabilities": {"drop": ["ALL"]},
                                    },
                                    "volumeMounts": [
                                        {
                                            "name": "data",
                                            "mountPath": "/var/lib/frontier",
                                        },
                                        {
                                            "name": "objects",
                                            "mountPath": "/var/lib/frontier-local",
                                            "readOnly": True,
                                        },
                                        {"name": "tmp", "mountPath": "/tmp"},
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "data",
                                    "persistentVolumeClaim": {
                                        "claimName": "frontier-management-data"
                                    },
                                },
                                {
                                    "name": "objects",
                                    "persistentVolumeClaim": {
                                        "claimName": "frontier-object-seed",
                                        "readOnly": True,
                                    },
                                },
                                {"name": "tmp", "emptyDir": {"sizeLimit": "34Gi"}},
                            ],
                        },
                    },
                },
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": MANAGEMENT_SERVICE,
                    "namespace": PLATFORM_NAMESPACE,
                    "labels": labels,
                },
                "spec": {
                    "type": "ClusterIP",
                    "clusterIP": network["management_cluster_ip"],
                    "selector": {"app.kubernetes.io/name": "frontier-management"},
                    "ports": [{"name": "http", "port": 8080, "targetPort": "http"}],
                },
            },
            {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {
                    "name": "frontier-management-boundary",
                    "namespace": PLATFORM_NAMESPACE,
                    "labels": labels,
                },
                "spec": {
                    "podSelector": {
                        "matchLabels": {"frontier.dev/network-role": "management"}
                    },
                    "policyTypes": ["Ingress", "Egress"],
                    "ingress": [
                        {
                            "from": [{"ipBlock": {"cidr": outputs["vpc_cidr"]}}],
                            "ports": [{"protocol": "TCP", "port": 8080}],
                        },
                        {
                            "from": [
                                {
                                    "namespaceSelector": {
                                        "matchLabels": {
                                            "frontier.dev/owner": "management"
                                        }
                                    },
                                    "podSelector": {
                                        "matchLabels": {
                                            "frontier.dev/network-role": "driver"
                                        }
                                    },
                                }
                            ],
                            "ports": [{"protocol": "TCP", "port": 8080}],
                        },
                    ],
                    "egress": [
                        {
                            "to": [
                                {
                                    "namespaceSelector": {
                                        "matchLabels": {
                                            "kubernetes.io/metadata.name": "kube-system"
                                        }
                                    },
                                    "podSelector": {
                                        "matchLabels": {"k8s-app": "kube-dns"}
                                    },
                                }
                            ],
                            "ports": [
                                {"protocol": "UDP", "port": 53},
                                {"protocol": "TCP", "port": 53},
                            ],
                        },
                        {
                            "to": [
                                {"ipBlock": {"cidr": cidr}}
                                for cidr in kubernetes_api_cidrs
                            ],
                            "ports": [{"protocol": "TCP", "port": 443}],
                        },
                        {
                            "to": [
                                {
                                    "ipBlock": {
                                        "cidr": "0.0.0.0/0",
                                        "except": [
                                            "10.0.0.0/8",
                                            "100.64.0.0/10",
                                            "127.0.0.0/8",
                                            "169.254.0.0/16",
                                            "172.16.0.0/12",
                                            "192.168.0.0/16",
                                        ],
                                    }
                                }
                            ],
                            "ports": [{"protocol": "TCP", "port": 443}],
                        },
                    ],
                },
            },
            {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {
                    "name": "frontier-storage-controller-boundary",
                    "namespace": PLATFORM_NAMESPACE,
                    "labels": controller_labels,
                },
                "spec": {
                    "podSelector": {
                        "matchLabels": {
                            "frontier.dev/network-role": "storage-controller"
                        }
                    },
                    "policyTypes": ["Ingress", "Egress"],
                    "ingress": [],
                    "egress": [
                        {
                            "to": [
                                {"ipBlock": {"cidr": cidr}}
                                for cidr in kubernetes_api_cidrs
                            ],
                            "ports": [{"protocol": "TCP", "port": 443}],
                        }
                    ],
                },
            },
        ]
    )
    if not simulator_bootstrap:
        management = next(
            item
            for item in items
            if item["kind"] == "Deployment"
            and item["metadata"]["name"] == "frontier-management"
        )
        # The same pinned runtime image provisions only its private EBS directory.
        # Existing symlinks or unexpected owner/mode fail without rewriting them.
        management["spec"]["template"]["spec"]["initContainers"] = [
            gateway_artifact_retention_init(management_image)
        ]
        volume, mount = management_diagnostic_source_mount()
        pod = management["spec"]["template"]["spec"]
        pod["volumes"] = [
            volume if v["name"] == "objects" else v for v in pod["volumes"]
        ]
        pod["containers"][0]["volumeMounts"].append(mount)
    return k8s_list(items)


def render_litellm_namespace(config: Mapping[str, Any]) -> bytes:
    labels = {
        "app.kubernetes.io/name": "frontier-litellm",
        "app.kubernetes.io/part-of": "frontiercs2",
        "frontier.dev/owner": "frontier-deploy",
        "frontier.dev/site": config["site"]["name"],
        "pod-security.kubernetes.io/enforce": "restricted",
        "pod-security.kubernetes.io/audit": "restricted",
        "pod-security.kubernetes.io/warn": "restricted",
    }
    return k8s_list(
        [
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {"name": LITELLM_NAMESPACE, "labels": labels},
            }
        ]
    )


def render_litellm_manifest(
    config: Mapping[str, Any], outputs: Mapping[str, Any], runtime_images: Any
) -> bytes:
    validate_outputs(config, outputs)
    images = validate_runtime_images(runtime_images, config)
    network = config["network"]
    service_network = ipaddress.ip_network(network["service_ipv4_cidr"])
    dns_cidr = f"{service_network.network_address + 10}/32"
    labels = {
        "app.kubernetes.io/name": "frontier-litellm",
        "app.kubernetes.io/part-of": "frontiercs2",
        "frontier.dev/owner": "frontier-deploy",
        "frontier.dev/site": config["site"]["name"],
    }
    gateway_labels = {
        **labels,
        "app.kubernetes.io/component": "gateway",
        "frontier.dev/network-role": "litellm",
    }
    postgres_labels = {
        **labels,
        "app.kubernetes.io/component": "postgres",
        "frontier.dev/network-role": "litellm-postgres",
    }
    restricted = {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
    }
    proxy_config = {
        "model_list": [],
        "router_settings": {
            # Retry transient upstream errors inside the gateway, preserving the
            # Agent's session and clock. Other errors keep the zero-retry default.
            "num_retries": 0,
            "retry_after": 5,
            "retry_policy": {
                "AuthenticationErrorRetries": 16,
                "RateLimitErrorRetries": 16,
                "InternalServerErrorRetries": 16,
                "TimeoutErrorRetries": 16,
            },
            "fallbacks": [],
            "context_window_fallbacks": [],
        },
        "general_settings": {
            "master_key": "os.environ/LITELLM_MASTER_KEY",
            "database_url": "os.environ/DATABASE_URL",
            "store_model_in_db": True,
            "ui_access_mode": "admin_only",
            "allow_client_side_credentials": False,
            "reject_clientside_metadata_tags": True,
            "forward_client_headers_to_llm_api": False,
        },
        "litellm_settings": {
            "callbacks": [
                "preview_audit.preview_audit",
                "frontier_hooks.frontier_callbacks.kimi_responses_tool_order_hook",
            ],
            "global_disable_no_log_param": True,
            "redact_user_api_key_info": True,
        },
    }
    gateway_config_data = {
        "config.yaml": json.dumps(proxy_config, sort_keys=True, indent=2) + "\n",
        "preview_audit.py": (
            REPOSITORY_ROOT / "runtime/litellm/preview_audit.py"
        ).read_text(),
        **{
            name: (
                REPOSITORY_ROOT / "runtime/litellm/frontier_hooks" / name
            ).read_text()
            for name in (
                "__init__.py",
                "frontier_callbacks.py",
                "responses_tool_order.py",
            )
        },
    }
    namespace = json.loads(render_litellm_namespace(config))["items"][0]
    items: list[dict[str, Any]] = [
        namespace,
        {
            "apiVersion": "storage.k8s.io/v1",
            "kind": "StorageClass",
            "metadata": {"name": "frontier-litellm-gp3", "labels": labels},
            "provisioner": "ebs.csi.aws.com",
            "reclaimPolicy": "Retain",
            "allowVolumeExpansion": True,
            "volumeBindingMode": "WaitForFirstConsumer",
            "parameters": {
                "type": "gp3",
                "encrypted": "true",
                "kmsKeyId": outputs["platform_kms_key_arn"],
                "csi.storage.k8s.io/fstype": "ext4",
            },
        },
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": "frontier-litellm",
                "namespace": LITELLM_NAMESPACE,
                "labels": labels,
            },
            "automountServiceAccountToken": False,
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "frontier-litellm-config",
                "namespace": LITELLM_NAMESPACE,
                "labels": gateway_labels,
            },
            "data": gateway_config_data,
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "frontier-litellm-postgres",
                "namespace": LITELLM_NAMESPACE,
                "labels": postgres_labels,
            },
            "spec": {
                "clusterIP": "None",
                "selector": {
                    "app.kubernetes.io/name": "frontier-litellm",
                    "app.kubernetes.io/component": "postgres",
                },
                "ports": [{"name": "postgres", "port": 5432, "targetPort": "postgres"}],
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": LITELLM_SERVICE,
                "namespace": LITELLM_NAMESPACE,
                "labels": gateway_labels,
            },
            "spec": {
                "type": "ClusterIP",
                "clusterIP": network["litellm_cluster_ip"],
                "selector": {
                    "app.kubernetes.io/name": "frontier-litellm",
                    "app.kubernetes.io/component": "gateway",
                },
                "ports": [{"name": "http", "port": 4000, "targetPort": "http"}],
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": "frontier-litellm-postgres",
                "namespace": LITELLM_NAMESPACE,
                "labels": postgres_labels,
            },
            "spec": {
                "serviceName": "frontier-litellm-postgres",
                "replicas": 1,
                "minReadySeconds": 5,
                "persistentVolumeClaimRetentionPolicy": {
                    "whenDeleted": "Retain",
                    "whenScaled": "Retain",
                },
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": "frontier-litellm",
                        "app.kubernetes.io/component": "postgres",
                    }
                },
                "template": {
                    "metadata": {"labels": postgres_labels},
                    "spec": {
                        "automountServiceAccountToken": False,
                        "nodeSelector": {"frontier.dev/pool": "system"},
                        "terminationGracePeriodSeconds": 30,
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 999,
                            "runAsGroup": 999,
                            "fsGroup": 999,
                            "fsGroupChangePolicy": "OnRootMismatch",
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "containers": [
                            {
                                "name": "postgres",
                                "image": images["postgres"]["reference"],
                                "imagePullPolicy": "IfNotPresent",
                                "ports": [{"name": "postgres", "containerPort": 5432}],
                                "env": [
                                    {
                                        "name": name,
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": LITELLM_SECRET,
                                                "key": name,
                                            }
                                        },
                                    }
                                    for name in (
                                        "POSTGRES_USER",
                                        "POSTGRES_PASSWORD",
                                        "POSTGRES_DB",
                                    )
                                ]
                                + [
                                    {
                                        "name": "PGDATA",
                                        "value": "/var/lib/postgresql/data/pgdata",
                                    }
                                ],
                                "startupProbe": {
                                    "exec": {
                                        "command": [
                                            "sh",
                                            "-ec",
                                            'exec pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"',
                                        ]
                                    },
                                    "periodSeconds": 2,
                                    "timeoutSeconds": 2,
                                    "failureThreshold": 60,
                                },
                                "readinessProbe": {
                                    "exec": {
                                        "command": [
                                            "sh",
                                            "-ec",
                                            'exec pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"',
                                        ]
                                    },
                                    "periodSeconds": 5,
                                    "timeoutSeconds": 2,
                                    "failureThreshold": 3,
                                },
                                "livenessProbe": {
                                    "exec": {
                                        "command": [
                                            "sh",
                                            "-ec",
                                            'exec pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"',
                                        ]
                                    },
                                    "periodSeconds": 10,
                                    "timeoutSeconds": 3,
                                    "failureThreshold": 6,
                                },
                                "resources": {
                                    "requests": {"cpu": "100m", "memory": "256Mi"},
                                    "limits": {"cpu": "1", "memory": "2Gi"},
                                },
                                "securityContext": restricted,
                                "volumeMounts": [
                                    {
                                        "name": "data",
                                        "mountPath": "/var/lib/postgresql/data",
                                    },
                                    {"name": "run", "mountPath": "/var/run/postgresql"},
                                    {"name": "tmp", "mountPath": "/tmp"},
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "run", "emptyDir": {"sizeLimit": "16Mi"}},
                            {"name": "tmp", "emptyDir": {"sizeLimit": "64Mi"}},
                        ],
                    },
                },
                "volumeClaimTemplates": [
                    {
                        "metadata": {"name": "data", "labels": postgres_labels},
                        "spec": {
                            "accessModes": ["ReadWriteOnce"],
                            "storageClassName": "frontier-litellm-gp3",
                            "volumeMode": "Filesystem",
                            "resources": {
                                "requests": {
                                    "storage": f"{config['litellm']['postgres_storage_gib']}Gi"
                                }
                            },
                        },
                    }
                ],
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": LITELLM_SERVICE,
                "namespace": LITELLM_NAMESPACE,
                "labels": gateway_labels,
            },
            "spec": {
                "replicas": 1,
                "minReadySeconds": 10,
                "progressDeadlineSeconds": 900,
                "revisionHistoryLimit": 2,
                "strategy": {"type": "Recreate"},
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": "frontier-litellm",
                        "app.kubernetes.io/component": "gateway",
                    }
                },
                "template": {
                    "metadata": {
                        "labels": gateway_labels,
                        "annotations": {
                            "frontier.dev/config-sha256": canonical_object_sha256(
                                gateway_config_data
                            )
                        },
                    },
                    "spec": {
                        "serviceAccountName": "frontier-litellm",
                        "automountServiceAccountToken": False,
                        "nodeSelector": {"frontier.dev/pool": "system"},
                        "terminationGracePeriodSeconds": 60,
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "runAsGroup": 1000,
                            "fsGroup": 1000,
                            "fsGroupChangePolicy": "OnRootMismatch",
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "containers": [
                            {
                                "name": "litellm",
                                "image": images["litellm"]["reference"],
                                "imagePullPolicy": "IfNotPresent",
                                "command": ["litellm"],
                                "args": [
                                    "--config",
                                    "/etc/litellm/config.yaml",
                                    "--host",
                                    "0.0.0.0",
                                    "--port",
                                    "4000",
                                    "--num_workers",
                                    "1",
                                ],
                                "ports": [{"name": "http", "containerPort": 4000}],
                                "env": [
                                    {
                                        "name": name,
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": LITELLM_SECRET,
                                                "key": name,
                                            }
                                        },
                                    }
                                    for name in (
                                        "DATABASE_URL",
                                        "LITELLM_MASTER_KEY",
                                        "LITELLM_SALT_KEY",
                                    )
                                ]
                                + [
                                    {"name": "UI_USERNAME", "value": "admin"},
                                    {
                                        "name": "UI_PASSWORD",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": LITELLM_SECRET,
                                                "key": "LITELLM_MASTER_KEY",
                                            }
                                        },
                                    },
                                    {"name": "DISABLE_SCHEMA_UPDATE", "value": "false"},
                                    {"name": "STORE_MODEL_IN_DB", "value": "True"},
                                    {"name": "UI_ACCESS_MODE", "value": "admin_only"},
                                    {
                                        "name": "LITELLM_MIGRATION_DIR",
                                        "value": "/tmp/prisma/migrations",
                                    },
                                    {
                                        "name": "XDG_CACHE_HOME",
                                        "value": "/tmp/litellm-cache",
                                    },
                                    {"name": "PYTHONPATH", "value": "/etc/litellm"},
                                    {
                                        "name": "FRONTIER_MODEL_AUDIT_ROOT",
                                        "value": f"/var/lib/frontier-local/artifacts/{outputs['artifact_bucket']}",
                                    },
                                ],
                                "startupProbe": {
                                    "exec": {
                                        "command": [
                                            "python",
                                            "-c",
                                            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:4000/health/readiness', timeout=2).read(1)",
                                        ]
                                    },
                                    "periodSeconds": 5,
                                    "timeoutSeconds": 3,
                                    "failureThreshold": 120,
                                },
                                "readinessProbe": {
                                    "exec": {
                                        "command": [
                                            "python",
                                            "-c",
                                            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:4000/health/readiness', timeout=2).read(1)",
                                        ]
                                    },
                                    "periodSeconds": 5,
                                    "timeoutSeconds": 3,
                                    "failureThreshold": 3,
                                },
                                "livenessProbe": {
                                    "exec": {
                                        "command": [
                                            "python",
                                            "-c",
                                            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:4000/health/liveliness', timeout=2).read(1)",
                                        ]
                                    },
                                    "periodSeconds": 15,
                                    "timeoutSeconds": 5,
                                    "failureThreshold": 6,
                                },
                                "resources": {
                                    "requests": {"cpu": "250m", "memory": "1Gi"},
                                    "limits": {"cpu": "2", "memory": "4Gi"},
                                },
                                "securityContext": restricted,
                                "volumeMounts": [
                                    {
                                        "name": "config",
                                        "mountPath": "/etc/litellm",
                                        "readOnly": True,
                                    },
                                    {"name": "tmp", "mountPath": "/tmp"},
                                    {
                                        "name": "objects",
                                        "mountPath": "/var/lib/frontier-local",
                                    },
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "config",
                                "configMap": {
                                    "name": "frontier-litellm-config",
                                    "items": [
                                        {"key": name, "path": name}
                                        for name in ("config.yaml", "preview_audit.py")
                                    ]
                                    + [
                                        {"key": name, "path": "frontier_hooks/" + name}
                                        for name in (
                                            "__init__.py",
                                            "frontier_callbacks.py",
                                            "responses_tool_order.py",
                                        )
                                    ],
                                },
                            },
                            {
                                "name": "objects",
                                "persistentVolumeClaim": {
                                    "claimName": "frontier-litellm-objects"
                                },
                            },
                            {"name": "tmp", "emptyDir": {"sizeLimit": "2Gi"}},
                        ],
                    },
                },
            },
        },
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "frontier-litellm-default-deny",
                "namespace": LITELLM_NAMESPACE,
                "labels": labels,
            },
            "spec": {"podSelector": {}, "policyTypes": ["Ingress", "Egress"]},
        },
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "frontier-litellm-gateway",
                "namespace": LITELLM_NAMESPACE,
                "labels": gateway_labels,
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": "frontier-litellm",
                        "app.kubernetes.io/component": "gateway",
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [{"ipBlock": {"cidr": outputs["vpc_cidr"]}}],
                        "ports": [{"protocol": "TCP", "port": 4000}],
                    },
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {"frontier.dev/owner": "management"}
                                },
                                "podSelector": {
                                    "matchLabels": {
                                        "frontier.dev/network-role": "driver"
                                    }
                                },
                            }
                        ],
                        "ports": [{"protocol": "TCP", "port": 4000}],
                    },
                ],
                "egress": [
                    {
                        "to": [{"ipBlock": {"cidr": dns_cidr}}],
                        "ports": [
                            {"protocol": "UDP", "port": 53},
                            {"protocol": "TCP", "port": 53},
                        ],
                    },
                    {
                        "to": [
                            {
                                "podSelector": {
                                    "matchLabels": {
                                        "app.kubernetes.io/name": "frontier-litellm",
                                        "app.kubernetes.io/component": "postgres",
                                    }
                                }
                            }
                        ],
                        "ports": [{"protocol": "TCP", "port": 5432}],
                    },
                    {
                        "to": [
                            {
                                "ipBlock": {
                                    "cidr": "0.0.0.0/0",
                                    "except": [
                                        "10.0.0.0/8",
                                        "100.64.0.0/10",
                                        "127.0.0.0/8",
                                        "169.254.0.0/16",
                                        "172.16.0.0/12",
                                        "192.168.0.0/16",
                                    ],
                                }
                            }
                        ],
                        "ports": [{"protocol": "TCP", "port": 443}],
                    },
                ],
            },
        },
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "frontier-litellm-postgres",
                "namespace": LITELLM_NAMESPACE,
                "labels": postgres_labels,
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": "frontier-litellm",
                        "app.kubernetes.io/component": "postgres",
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {
                                "podSelector": {
                                    "matchLabels": {
                                        "app.kubernetes.io/name": "frontier-litellm",
                                        "app.kubernetes.io/component": "gateway",
                                    }
                                }
                            }
                        ],
                        "ports": [{"protocol": "TCP", "port": 5432}],
                    }
                ],
                "egress": [],
            },
        },
    ]
    items.extend(litellm_worker_resources(config, outputs, labels))
    return k8s_list(items)


def litellm_worker_resources(config, outputs, labels):
    namespace = LITELLM_NAMESPACE
    name = "frontier-litellm-key-worker"
    role = "frontier-key-worker-serving-ready-reader"

    def metadata(value):
        return {"name": value, "namespace": namespace, "labels": labels}

    return [
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": "frontier-key-worker-namespace-reader", "labels": labels},
            "rules": [{"apiGroups": [""], "resources": ["namespaces"], "verbs": ["get"]}],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": "frontier-key-worker-namespace-reader", "labels": labels},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": "frontier-key-worker-namespace-reader",
            },
            "subjects": [{"kind": "ServiceAccount", "name": name, "namespace": namespace}],
        },
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": metadata(name),
            "automountServiceAccountToken": False,
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": metadata(role),
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["configmaps"],
                    "resourceNames": ["frontier-litellm-serving-ready"],
                    "verbs": ["get"],
                }
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": metadata(role),
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": role,
            },
            "subjects": [
                {"kind": "ServiceAccount", "name": name, "namespace": namespace}
            ],
        },
        {
            "apiVersion": "v1",
            "kind": "PersistentVolume",
            "metadata": {"name": "frontier-litellm-objects", "labels": labels},
            "spec": {
                "capacity": {"storage": f"{config['execution']['object_store_gib']}Gi"},
                "volumeMode": "Filesystem",
                "accessModes": ["ReadWriteMany"],
                "persistentVolumeReclaimPolicy": "Retain",
                "storageClassName": "",
                "mountOptions": ["tls"],
                "csi": {
                    "driver": "efs.csi.aws.com",
                    "volumeHandle": f"{outputs['efs_file_system_id']}::{outputs['efs_object_store_access_point_id']}",
                },
            },
        },
        {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": metadata("frontier-litellm-objects"),
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "volumeMode": "Filesystem",
                "storageClassName": "",
                "volumeName": "frontier-litellm-objects",
                "resources": {
                    "requests": {
                        "storage": f"{config['execution']['object_store_gib']}Gi"
                    }
                },
            },
        },
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": metadata("frontier-litellm-key-worker"),
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "frontier.dev/resource-kind": "model-gateway-key-worker"
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "egress": [
                    {
                        "to": [{"ipBlock": {"cidr": cidr}}],
                        "ports": [{"protocol": "TCP", "port": 443}],
                    }
                    for cidr in [
                        outputs["vpc_cidr"],
                        str(
                            ipaddress.ip_network(
                                config["network"]["service_ipv4_cidr"]
                            ).network_address
                            + 1
                        )
                        + "/32",
                    ]
                ]
                + [
                    {
                        "to": [
                            {
                                "ipBlock": {
                                    "cidr": str(
                                        ipaddress.ip_network(
                                            config["network"]["service_ipv4_cidr"]
                                        ).network_address
                                        + 10
                                    )
                                    + "/32"
                                }
                            }
                        ],
                        "ports": [
                            {"protocol": "UDP", "port": 53},
                            {"protocol": "TCP", "port": 53},
                        ],
                    },
                    {
                        "to": [
                            {
                                "podSelector": {
                                    "matchLabels": {
                                        "frontier.dev/network-role": "litellm"
                                    }
                                }
                            }
                        ],
                        "ports": [{"protocol": "TCP", "port": 4000}],
                    },
                    {
                        # AWS policy enforcement can observe the Service VIP before DNAT.
                        "to": [{"ipBlock": {"cidr": config["network"]["litellm_cluster_ip"] + "/32"}}],
                        "ports": [{"protocol": "TCP", "port": 4000}],
                    },
                ],
            },
        },
    ]


def executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise DeployError(f"required executable is missing: {name}")
    return path


def command_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    environment["AWS_PAGER"] = ""
    environment["CHECKPOINT_DISABLE"] = "1"
    if extra:
        environment.update(extra)
    return environment


def run_command(
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    capture: bool = False,
    timeout: int = 3600,
    input_text: str | None = None,
) -> str:
    display = " ".join(command)
    print(f"+ {display}", flush=True)
    completed = subprocess.run(
        list(command),
        cwd=REPOSITORY_ROOT,
        env=command_environment(environment),
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        timeout=timeout,
        check=False,
        input=input_text,
    )
    if completed.returncode != 0:
        detail = (
            (completed.stderr or completed.stdout or "").strip()[-4096:]
            if capture
            else ""
        )
        raise DeployError(
            f"command failed with exit {completed.returncode}: {display}"
            + (f": {detail}" if detail else "")
        )
    return completed.stdout if capture else ""


def aws_identity(config: Mapping[str, Any]) -> dict[str, str]:
    encoded = run_command(
        [executable("aws"), "sts", "get-caller-identity", "--output", "json"],
        capture=True,
        timeout=60,
    )
    try:
        identity = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise DeployError("AWS caller identity returned malformed JSON") from error
    account, arn = identity.get("Account"), identity.get("Arn")
    if account != config["site"]["expected_account_id"]:
        raise DeployError(
            f"AWS account mismatch: expected {config['site']['expected_account_id']}, observed {account}"
        )
    if not isinstance(arn, str) or not arn.startswith("arn:"):
        raise DeployError("AWS caller identity has no canonical ARN")
    print(f"AWS account: {account}\nAWS principal: {arn}")
    return {"account": account, "arn": arn}


def terraform_environment(config: Mapping[str, Any], component: str) -> dict[str, str]:
    data = state_directory(config) / "terraform-data" / component
    data.mkdir(parents=True, exist_ok=True)
    return {"TF_DATA_DIR": str(data), "TF_IN_AUTOMATION": "1"}


def terraform_init_and_validate(
    config: Mapping[str, Any],
    component: str,
    backend_config: Path,
    *,
    backend: bool = True,
) -> None:
    terraform, root = executable("terraform"), REPOSITORY_ROOT / "terraform" / component
    command = [
        terraform,
        f"-chdir={root}",
        "init",
        "-input=false",
        "-reconfigure",
        f"-backend-config={backend_config}" if backend else "-backend=false",
    ]
    run_command(command, environment=terraform_environment(config, component))
    run_command(
        [terraform, f"-chdir={root}", "validate"],
        environment=terraform_environment(config, component),
    )


def terraform_plan(
    config: Mapping[str, Any], component: str, tfvars: Path, plan_path: Path
) -> None:
    root = REPOSITORY_ROOT / "terraform" / component
    run_command(
        [
            executable("terraform"),
            f"-chdir={root}",
            "plan",
            "-input=false",
            f"-var-file={tfvars}",
            f"-out={plan_path}",
        ],
        environment=terraform_environment(config, component),
    )


def terraform_apply_plan(
    config: Mapping[str, Any], component: str, plan_path: Path
) -> None:
    root = REPOSITORY_ROOT / "terraform" / component
    run_command(
        [
            executable("terraform"),
            f"-chdir={root}",
            "apply",
            "-input=false",
            str(plan_path),
        ],
        environment=terraform_environment(config, component),
        timeout=7200,
    )


def state_bucket_exists(config: Mapping[str, Any]) -> bool:
    completed = subprocess.run(
        [
            executable("aws"),
            "s3api",
            "head-bucket",
            "--bucket",
            config["terraform"]["state_bucket"],
        ],
        cwd=REPOSITORY_ROOT,
        env=command_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    return completed.returncode == 0


def platform_outputs(config: Mapping[str, Any]) -> dict[str, Any]:
    root = REPOSITORY_ROOT / "terraform/platform"
    encoded = run_command(
        [executable("terraform"), f"-chdir={root}", "output", "-json"],
        environment=terraform_environment(config, "platform"),
        capture=True,
    )
    try:
        outputs = terraform_output_values(json.loads(encoded))
    except json.JSONDecodeError as error:
        raise DeployError("Terraform output returned malformed JSON") from error
    validate_outputs(config, outputs)
    return outputs


def update_kubeconfig(config: Mapping[str, Any], outputs: Mapping[str, Any]) -> Path:
    kubeconfig = state_directory(config) / "kubeconfig"
    kubeconfig.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            executable("aws"),
            "eks",
            "update-kubeconfig",
            "--name",
            str(outputs["cluster_name"]),
            "--region",
            str(outputs["aws_region"]),
            "--kubeconfig",
            str(kubeconfig),
            "--alias",
            f"frontiercs2-{config['site']['name']}",
        ],
        timeout=120,
    )
    os.chmod(kubeconfig, 0o600)
    return kubeconfig


def kubernetes_environment(kubeconfig: Path) -> dict[str, str]:
    return {"KUBECONFIG": str(kubeconfig)}


def kubectl_apply(manifest_path: Path, kubeconfig: Path) -> None:
    run_command(
        [
            executable("kubectl"),
            "apply",
            "--server-side",
            "--field-manager=frontier-deploy",
            "-f",
            str(manifest_path),
        ],
        environment=kubernetes_environment(kubeconfig),
        timeout=600,
    )


def deploy_bootstrap(
    config: Mapping[str, Any],
    output: Path,
    outputs: Mapping[str, Any],
    kubeconfig: Path,
) -> None:
    path = output / "cluster-bootstrap.json"
    atomic_write(path, render_bootstrap_manifest(config, outputs))
    kubectl_apply(path, kubeconfig)
    run_command(
        [
            executable("kubectl"),
            "-n",
            PLATFORM_NAMESPACE,
            "rollout",
            "status",
            f"deployment/{REGISTRY_SERVICE}",
            "--timeout=10m",
        ],
        environment=kubernetes_environment(kubeconfig),
        timeout=660,
    )


def litellm_serving_receipt(service, postgres, pvc, admin, deployment):
    """Bind the same immutable serving identity consumed by staging key workers."""
    if (
        postgres.get("status", {}).get("readyReplicas") != 1
        or deployment.get("status", {}).get("readyReplicas") != 1
        or pvc.get("status", {}).get("phase") != "Bound"
    ):
        raise DeployError("LiteLLM gateway, PostgreSQL and storage must be Ready")
    identities = [value["metadata"]["uid"] for value in (service, postgres, pvc, admin)]
    generation = canonical_object_sha256(identities).removeprefix("sha256:")
    nonce = generation[:32]
    data = {
        "schema": "frontier.litellm-serving-ready/v1",
        "admin_identity_uid": admin["metadata"]["uid"],
        "db_identity_uid": postgres["metadata"]["uid"],
        "pvc_uid": pvc["metadata"]["uid"],
        "service_uid": service["metadata"]["uid"],
        "service_ip": service["spec"]["clusterIP"],
        "gateway_ready": "true",
        "postgres_ready": "true",
        "identity_generation": generation,
        "lease_nonce": nonce,
        "lease_revision": generation,
        # The staging protocol retains this field but no longer expires the
        # serving identity. Per-attempt virtual keys retain their own expiry.
        "lease_expires_at": "1",
    }
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "immutable": True,
        "metadata": {
            "name": "frontier-litellm-serving-ready",
            "namespace": LITELLM_NAMESPACE,
            "labels": {
                "app.kubernetes.io/name": "frontier-litellm",
                "app.kubernetes.io/component": "serving-readiness",
                "app.kubernetes.io/part-of": "frontiercs2",
                "frontier.dev/owner": "platform",
            },
            "annotations": {
                "frontier.dev/lease-nonce": nonce,
                "frontier.dev/identity-generation": generation,
            },
        },
        "data": data,
    }


def publish_litellm_serving_receipt(output, kubeconfig):
    command = [executable("kubectl"), "-n", LITELLM_NAMESPACE]
    environment = kubernetes_environment(kubeconfig)

    def read(kind, name):
        return json.loads(
            run_command(
                command + ["get", kind, name, "-o", "json"],
                environment=environment,
                capture=True,
                timeout=60,
            )
        )

    # Request only metadata for the admin Secret; never capture credential bytes.
    admin_uid = run_command(
        command
        + ["get", "secret", "litellm-master-key", "-o", "jsonpath={.metadata.uid}"],
        environment=environment,
        capture=True,
        timeout=60,
    ).strip()
    admin = {"metadata": {"uid": admin_uid}}
    receipt = litellm_serving_receipt(
        read("service", LITELLM_SERVICE),
        read("statefulset", "frontier-litellm-postgres"),
        read("pvc", "data-frontier-litellm-postgres-0"),
        admin,
        read("deployment", LITELLM_SERVICE),
    )
    path = output / "litellm-serving-ready.json"
    atomic_write(path, canonical_json(receipt))
    # An existing receipt is immutable. Identity replacement requires an idle
    # operator window and explicit reconciliation, never silent adoption.
    kubectl_apply(path, kubeconfig)


def deploy_litellm(
    config: Mapping[str, Any],
    output: Path,
    outputs: Mapping[str, Any],
    kubeconfig: Path,
    images: Mapping[str, Any],
) -> None:
    namespace_path = output / "litellm-namespace.json"
    atomic_write(namespace_path, render_litellm_namespace(config))
    kubectl_apply(namespace_path, kubeconfig)
    secret_path, admin_path = ensure_litellm_secret_files(config)
    kubectl_apply(secret_path, kubeconfig)
    secret = load_json(secret_path)
    worker_secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "type": "Opaque",
        "metadata": {"name": "litellm-master-key", "namespace": LITELLM_NAMESPACE},
        "stringData": {"master-key": secret["stringData"]["LITELLM_MASTER_KEY"]},
    }
    worker_secret_path = state_directory(config) / "litellm-worker-secret.json"
    atomic_write(worker_secret_path, canonical_json(worker_secret), mode=0o600)
    kubectl_apply(worker_secret_path, kubeconfig)
    path = output / "litellm.json"
    atomic_write(path, render_litellm_manifest(config, outputs, images))
    kubectl_apply(path, kubeconfig)
    environment = kubernetes_environment(kubeconfig)
    run_command(
        [
            executable("kubectl"),
            "-n",
            LITELLM_NAMESPACE,
            "rollout",
            "status",
            "statefulset/frontier-litellm-postgres",
            "--timeout=15m",
        ],
        environment=environment,
        timeout=960,
    )
    run_command(
        [
            executable("kubectl"),
            "-n",
            LITELLM_NAMESPACE,
            "rollout",
            "status",
            f"deployment/{LITELLM_SERVICE}",
            "--timeout=15m",
        ],
        environment=environment,
        timeout=960,
    )
    publish_litellm_serving_receipt(output, kubeconfig)
    print(f"LiteLLM admin login state: {admin_path} (mode 0600; value not printed)")


def free_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def port_forward(
    kubeconfig: Path,
    service: str,
    remote_port: int,
    *,
    namespace: str = PLATFORM_NAMESPACE,
) -> Iterator[int]:
    port = free_local_port()
    log = tempfile.TemporaryFile()
    command = [
        executable("kubectl"),
        "-n",
        namespace,
        "port-forward",
        "--address",
        "127.0.0.1",
        f"service/{service}",
        f"{port}:{remote_port}",
    ]
    print("+ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        env=command_environment(kubernetes_environment(kubeconfig)),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log.seek(0)
                raise DeployError(
                    f"kubectl port-forward exited early: {log.read().decode(errors='replace')[-2000:]}"
                )
            try:
                with urllib_request.urlopen(f"http://127.0.0.1:{port}/", timeout=1):
                    pass
                break
            except urllib_error.HTTPError:
                # The TCP tunnel is ready even when the selected service does
                # not implement GET / (the registry normally returns 404).
                break
            except (urllib_error.URLError, TimeoutError):
                time.sleep(0.25)
        else:
            raise DeployError(f"port-forward did not become ready for {service}")
        yield port
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log.close()


def admission_control_maps() -> list[dict[str, Any]]:
    return [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": PLATFORM_NAMESPACE,
                "labels": {
                    "app.kubernetes.io/name": MANAGEMENT_SERVICE,
                    "app.kubernetes.io/component": f"production-admission-{kind}",
                    "app.kubernetes.io/part-of": "frontiercs2",
                    "frontier.dev/owner": owner,
                },
            },
            "data": {f"{kind}.json": literal},
        }
        for name, kind, owner, literal in (
            (
                ADMISSION_COMMAND_MAP,
                "command",
                "production-operator",
                ADMISSION_BOOTSTRAP_COMMAND,
            ),
            (
                ADMISSION_RECEIPT_MAP,
                "receipt",
                "management",
                ADMISSION_BOOTSTRAP_RECEIPT,
            ),
        )
    ]


def ensure_admission_control_maps(kubeconfig: Path) -> None:
    """Create absent maps without taking SSA ownership of mutable control data."""
    environment = kubernetes_environment(kubeconfig)
    kubectl = [executable("kubectl"), "-n", PLATFORM_NAMESPACE]
    for document in admission_control_maps():
        name = document["metadata"]["name"]
        # --ignore-not-found suppresses only 404; RBAC/transport errors fail.
        existing = run_command(
            kubectl + ["get", "configmap", name, "--ignore-not-found", "-o", "json"],
            environment=environment,
            capture=True,
            timeout=120,
        )
        if existing.strip():
            continue
        with tempfile.TemporaryDirectory(prefix="frontier-admission-") as directory:
            path = Path(directory) / f"{name}.json"
            atomic_write(path, canonical_json(document))
            try:
                run_command(
                    kubectl + ["create", "-f", "-"],
                    environment=environment,
                    capture=True,
                    timeout=120,
                    input_text=path.read_text(),
                )
            except DeployError as error:
                if "Error from server (AlreadyExists)" not in str(error):
                    raise


def api_json(
    base: str,
    method: str,
    path: str,
    body: Any | None = None,
    headers: Mapping[str, str] | None = None,
) -> Any:
    payload, _ = api_json_with_headers(base, method, path, body, headers)
    return payload


def api_json_with_headers(
    base: str,
    method: str,
    path: str,
    body: Any | None = None,
    headers: Mapping[str, str] | None = None,
) -> tuple[Any, dict[str, str]]:
    encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request_headers = {"Accept": "application/json"}
    if encoded is not None:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)
    call = urllib_request.Request(
        base + path, data=encoded, headers=request_headers, method=method
    )
    try:
        with urllib_request.urlopen(call, timeout=30) as response:
            payload = response.read(4 * 1024 * 1024 + 1)
            response_headers = {
                key.lower(): value for key, value in response.headers.items()
            }
    except urllib_error.HTTPError as error:
        try:
            try:
                detail = error.read(4096).decode(errors="replace")
            except (http.client.HTTPException, OSError) as read_error:
                detail = f"error body unreadable ({type(read_error).__name__})"
        finally:
            try:
                error.close()
            except (http.client.HTTPException, OSError) as close_error:
                raise DeployError(
                    f"Management request failed for {path}: {close_error}"
                ) from close_error
        raise ManagementHTTPError(error.code, path, detail) from error
    except (urllib_error.URLError, http.client.HTTPException, OSError) as error:
        raise DeployError(f"Management request failed for {path}: {error}") from error
    if len(payload) > 4 * 1024 * 1024:
        raise DeployError(f"Management response exceeded 4 MiB: {path}")
    try:
        return json.loads(payload), response_headers
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DeployError(f"Management returned invalid JSON for {path}") from error


def generated_directory(config: Mapping[str, Any]) -> Path:
    return REPOSITORY_ROOT / ".generated" / config["site"]["name"]


def state_directory(config: Mapping[str, Any]) -> Path:
    return REPOSITORY_ROOT / ".state" / config["site"]["name"]


def scheduler_capacity_preflight(
    base_url: str, *, informational: bool = False
) -> dict[str, Any]:
    snapshot = require_mapping(
        api_json(base_url, "GET", "/api/v1/infrastructure/scheduler-capacity"),
        "scheduler capacity",
    )
    counts = require_mapping(snapshot.get("counts"), "scheduler capacity counts")
    runs = require_mapping(counts.get("run_counts"), "scheduler run counts")
    cleanups = require_mapping(counts.get("cleanup_counts"), "scheduler cleanup counts")

    def count(values: Mapping[str, Any], key: str) -> int:
        value = values.get(key)
        if type(value) is not int or value < 0:
            raise DeployError(f"scheduler capacity {key} must be a nonnegative integer")
        return value

    cap, occupied, available, released = [
        count(snapshot, key)
        for key in (
            "configured_cap",
            "occupied_slots",
            "available_slots",
            "released_historical_cleanups",
        )
    ]
    queued = count(runs, "QUEUED")
    active = sum(count(runs, key) for key in ("STARTING", "RUNNING", "FINALIZING"))
    progressing = sum(
        count(cleanups, key) for key in ("PENDING", "APPLYING", "RUNNING")
    )
    blocked = count(cleanups, "BLOCKED")
    print(
        f"capacity: cap={cap} occupied={occupied} available={available} queued={queued} active={active} "
        f"cleanups pending/applying/running={progressing} blocked={blocked} released={released}"
    )
    if available < 1:
        if blocked < cap and not informational:
            print(
                f"capacity temporarily exhausted: {active} active Runs, {progressing} progressing cleanups, "
                f"{blocked} retained; the smoke will queue"
            )
            return snapshot
        blockers = snapshot.get("retained_cleanup_blockers")
        if not isinstance(blockers, list):
            raise DeployError(
                "scheduler capacity retained_cleanup_blockers must be a list"
            )
        details = []
        for blocker in blockers[:10]:
            blocker = require_mapping(blocker, "scheduler cleanup blocker")
            diagnostics = require_mapping(
                blocker.get("diagnostics") or {}, "scheduler blocker diagnostics"
            )
            last_error = require_mapping(
                diagnostics.get("cleanup_job_last_error") or {},
                "scheduler cleanup last error",
            )
            details.append(
                f"attempt={blocker.get('attempt_id')} run={blocker.get('run_id')} "
                f"namespace={blocker.get('namespace')} state={blocker.get('cleanup_job_state')} "
                f"error={last_error.get('known_message') or 'unknown'}"
            )
        if len(blockers) > 10:
            details.append(f"showing 10 of {len(blockers)} returned cleanup blockers")
        if snapshot.get("blockers_truncated"):
            details.append("blockers_truncated=true (server capped the blocker list)")
        advice = []
        if blocked:
            advice.append(
                f"retained BLOCKED reservations={blocked}: raise execution.max_active_drivers above the retained count; "
                "historical-GC recovery is unsupported for these records at this platform pin"
            )
        if informational and (active or progressing or not blocked):
            advice.append(
                f"temporary occupancy (active Runs={active}, progressing cleanups={progressing}): wait or raise the cap"
            )
        raise DeployError("no scheduler capacity: " + "; ".join(advice + details))
    return snapshot


def runtime_admission_preflight(
    config: Mapping[str, Any], base_url: str, *, informational: bool = False
) -> dict[str, Any]:
    """Check the live gate and capacity; seed --check reports diagnostics only."""
    try:
        _, headers = api_json_with_headers(base_url, "GET", "/api/v1/overview")
        mode = headers.get("x-management-driver-mode")
        accepting = headers.get("x-management-accepting-evaluations")
        print(
            f"admission: site={config['site']['name']} driver_mode={mode} accepting_evaluations={accepting}"
        )
        if mode != "kubernetes" or accepting != "true":
            raise DeployError(
                "runtime admission requires x-management-driver-mode: kubernetes and "
                f"x-management-accepting-evaluations: true; observed mode={mode}, accepting={accepting}"
            )
    except DeployError as error:
        if not informational:
            raise
        print(f"runtime preflight (informational): {error}")
    try:
        return scheduler_capacity_preflight(base_url, informational=informational)
    except DeployError as error:
        if not informational:
            raise
        if isinstance(error, ManagementHTTPError) and error.status in {404, 503}:
            print(
                "capacity endpoint unavailable (pre-V3 runtime or non-kubernetes mode)"
            )
        else:
            print(f"runtime preflight (informational): {error}")
        return {}
