"""Site validation and generated deployment bundles."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .infra import (
    ACCOUNT_RE,
    APPLICATION_REPOSITORY,
    COMMIT_RE,
    FORBIDDEN_CONFIG_KEYS,
    INSTANCE_TYPE_RE,
    MODEL_DOMAIN_PATTERN,
    REGION_RE,
    S3_BUCKET_RE,
    SITE_RE,
    SUPPORTED_KUBERNETES_VERSION,
    SUPPORTED_SCHEMA,
    DeployError,
    atomic_write,
    canonical_json,
    generated_directory,
    load_json,
    require_exact_keys,
    require_int,
    require_mapping,
    require_string,
    runtime_source_commit,
    sha256_bytes,
    state_directory,
    validate_image,
)
from .infra import (
    REPOSITORY_ROOT as REPOSITORY_ROOT,
)
from .manual_capacity import capacity_settings
from .publication import (
    load_task_selection,
)


def reject_secret_fields(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.strip().lower().replace("-", "_")
            if normalized in FORBIDDEN_CONFIG_KEYS:
                raise DeployError(
                    f"secret-bearing field is forbidden in site config: {path}.{key}"
                )
            reject_secret_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_secret_fields(child, f"{path}[{index}]")


def validate_node_group(raw: Any, path: str) -> dict[str, Any]:
    group = require_mapping(raw, path)
    require_exact_keys(
        group,
        {"instance_types", "min_size", "desired_size", "max_size", "root_volume_gib"},
        path,
    )
    instance_types = group["instance_types"]
    if (
        not isinstance(instance_types, list)
        or not instance_types
        or len(instance_types) > 4
        or any(
            not isinstance(item, str) or not INSTANCE_TYPE_RE.fullmatch(item)
            for item in instance_types
        )
        or len(set(instance_types)) != len(instance_types)
    ):
        raise DeployError(
            f"{path}.instance_types must be 1-4 unique canonical EC2 types"
        )
    minimum = require_int(group["min_size"], f"{path}.min_size", 1, 20)
    desired = require_int(group["desired_size"], f"{path}.desired_size", 1, 20)
    maximum = require_int(group["max_size"], f"{path}.max_size", 1, 20)
    if not minimum <= desired <= maximum:
        raise DeployError(f"{path} must satisfy min_size <= desired_size <= max_size")
    require_int(group["root_volume_gib"], f"{path}.root_volume_gib", 30, 1024)
    return group


def validate_verifier_placement_required_task_ids(ids: Any) -> None:
    if not isinstance(ids, list) or len(ids) > 64:
        raise DeployError(
            "execution.verifier_placement_required_task_ids must be an array of at most 64 logical Task IDs"
        )
    seen = set()
    for task in ids:
        if (
            not isinstance(task, str)
            or not task
            or task.strip() != task
            or len(task.encode("utf-8")) > 128
            or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in task)
            or task in seen
        ):
            raise DeployError(
                "required verifier placement Task IDs must be distinct nonempty logical IDs without surrounding whitespace or control characters, at most 128 bytes each"
            )
        seen.add(task)


def validate_verifier_node_allowlists(policies: Any) -> None:
    if not isinstance(policies, list) or len(policies) > 64:
        raise DeployError(
            "execution.verifier_node_allowlists must be an array of at most 64 entries"
        )
    tasks = set()
    for policy in policies:
        if not isinstance(policy, dict) or set(policy) != {
            "task_version_id",
            "nodes",
        }:
            raise DeployError(
                "verifier node allowlist requires task_version_id and nodes"
            )
        task = policy["task_version_id"]
        nodes = policy["nodes"]
        if (
            not isinstance(task, str)
            or re.fullmatch(r"taskver_[0-7][0-9A-HJKMNP-TV-Z]{25}", task) is None
            or task in tasks
        ):
            raise DeployError(
                "verifier node allowlist must bind a distinct valid TaskVersion"
            )
        tasks.add(task)
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= 32:
            raise DeployError(
                "verifier node allowlist requires 1..32 qualified node names"
            )
        if any(
            not isinstance(node, str)
            or len(node) > 253
            or re.fullmatch(
                r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?)*",
                node,
            )
            is None
            for node in nodes
        ) or len(set(nodes)) != len(nodes):
            raise DeployError(
                "verifier node allowlist requires distinct valid DNS node names"
            )


def validate_config(raw: Any) -> dict[str, Any]:
    config = require_mapping(raw, "config")
    reject_secret_fields(config)
    require_exact_keys(
        config,
        {
            "schema_version",
            "site",
            "terraform",
            "network",
            "compute",
            "application",
            "images",
            "management",
            "litellm",
            "execution",
            "features",
            "tags",
        }
        | ({"modal"} if "modal" in config else set()),
        "config",
    )
    if config["schema_version"] != SUPPORTED_SCHEMA:
        raise DeployError(f"schema_version must be {SUPPORTED_SCHEMA}")

    site = require_mapping(config["site"], "site")
    require_exact_keys(
        site,
        {"name", "expected_account_id", "aws_region", "kubernetes_version"},
        "site",
    )
    site_name = require_string(site["name"], "site.name")
    if not SITE_RE.fullmatch(site_name):
        raise DeployError("site.name must be a 3-24 character lowercase slug")
    account_id = require_string(site["expected_account_id"], "site.expected_account_id")
    if not ACCOUNT_RE.fullmatch(account_id):
        raise DeployError("site.expected_account_id must contain exactly 12 digits")
    region = require_string(site["aws_region"], "site.aws_region")
    if not REGION_RE.fullmatch(region):
        raise DeployError("site.aws_region is not a canonical AWS region")
    if site["kubernetes_version"] != SUPPORTED_KUBERNETES_VERSION:
        raise DeployError(
            f"site.kubernetes_version must be {SUPPORTED_KUBERNETES_VERSION}"
        )

    terraform = require_mapping(config["terraform"], "terraform")
    require_exact_keys(
        terraform,
        {"state_bucket", "state_key"}
        | ({"operator_principals"} if "operator_principals" in terraform else set()),
        "terraform",
    )
    operator_principals = terraform.get("operator_principals", [])
    if not isinstance(operator_principals, list) or any(
        not isinstance(arn, str)
        or re.fullmatch(
            r"arn:aws(?:-cn|-us-gov)?:iam::[0-9]{12}:(?:user|role)/"
            r"[A-Za-z0-9+=,.@_-]+(?:/[A-Za-z0-9+=,.@_-]+)*",
            arn,
        )
        is None
        for arn in operator_principals
    ):
        raise DeployError(
            "terraform.operator_principals must be a list of IAM user or role ARNs"
        )
    if len(set(operator_principals)) != len(operator_principals):
        raise DeployError("terraform.operator_principals contains duplicates")
    state_bucket = require_string(terraform["state_bucket"], "terraform.state_bucket")
    if (
        not S3_BUCKET_RE.fullmatch(state_bucket)
        or ".." in state_bucket
        or ".-" in state_bucket
        or "-." in state_bucket
    ):
        raise DeployError(
            "terraform.state_bucket is not a valid lowercase S3 bucket name"
        )
    try:
        ipaddress.ip_address(state_bucket)
    except ValueError:
        pass
    else:
        raise DeployError("terraform.state_bucket cannot be formatted as an IP address")
    state_key = require_string(terraform["state_key"], "terraform.state_key")
    if (
        state_key.startswith("/")
        or ".." in Path(state_key).parts
        or not state_key.endswith(".tfstate")
    ):
        raise DeployError(
            "terraform.state_key must be a relative, traversal-free .tfstate key"
        )
    artifact_bucket = f"frontiercs2-{site_name}-{account_id}-{region}-artifacts"
    if len(artifact_bucket) > 63 or not S3_BUCKET_RE.fullmatch(artifact_bucket):
        raise DeployError("site.name is too long for the derived artifact bucket")

    network = require_mapping(config["network"], "network")
    require_exact_keys(
        network,
        {
            "vpc_cidr",
            "service_ipv4_cidr",
            "management_cluster_ip",
            "registry_cluster_ip",
            "litellm_cluster_ip",
            "availability_zone_count",
            "operator_cidrs",
            "single_nat_gateway",
        },
        "network",
    )
    try:
        vpc = ipaddress.ip_network(
            require_string(network["vpc_cidr"], "network.vpc_cidr"), strict=True
        )
        service = ipaddress.ip_network(
            require_string(network["service_ipv4_cidr"], "network.service_ipv4_cidr"),
            strict=True,
        )
    except ValueError as error:
        raise DeployError(f"network CIDR is invalid: {error}") from error
    if (
        vpc.version != 4
        or vpc.prefixlen != 16
        or not vpc.subnet_of(ipaddress.ip_network("10.0.0.0/8"))
    ):
        raise DeployError("network.vpc_cidr must be an IPv4 /16 inside 10.0.0.0/8")
    if service.version != 4 or service.prefixlen != 16 or service.overlaps(vpc):
        raise DeployError("network.service_ipv4_cidr must be a disjoint IPv4 /16")
    fixed_ips: list[ipaddress.IPv4Address] = []
    for field in ("management_cluster_ip", "registry_cluster_ip", "litellm_cluster_ip"):
        try:
            address = ipaddress.ip_address(
                require_string(network[field], f"network.{field}")
            )
        except ValueError as error:
            raise DeployError(f"network.{field} is invalid") from error
        if (
            address.version != 4
            or address not in service
            or address in {service.network_address, service.broadcast_address}
        ):
            raise DeployError(
                f"network.{field} must be a usable address inside service_ipv4_cidr"
            )
        fixed_ips.append(address)
    reserved = {service.network_address + 1, service.network_address + 10}
    if len(set(fixed_ips)) != 3 or any(address in reserved for address in fixed_ips):
        raise DeployError(
            "fixed service IPs must be distinct and cannot use Kubernetes API/CoreDNS addresses"
        )
    require_int(
        network["availability_zone_count"], "network.availability_zone_count", 2, 3
    )
    operator_cidrs = network["operator_cidrs"]
    if (
        not isinstance(operator_cidrs, list)
        or not operator_cidrs
        or len(operator_cidrs) > 16
    ):
        raise DeployError(
            "network.operator_cidrs must contain 1-16 exact IPv4 /32 values"
        )
    normalized: list[str] = []
    for index, cidr in enumerate(operator_cidrs):
        try:
            parsed = ipaddress.ip_network(
                require_string(cidr, f"network.operator_cidrs[{index}]"), strict=True
            )
        except ValueError as error:
            raise DeployError(
                f"network.operator_cidrs[{index}] is invalid: {error}"
            ) from error
        if parsed.version != 4 or parsed.prefixlen != 32:
            raise DeployError(
                "network.operator_cidrs accepts only exact IPv4 /32 values"
            )
        normalized.append(str(parsed))
    if len(set(normalized)) != len(normalized):
        raise DeployError("network.operator_cidrs contains duplicates")
    if network["single_nat_gateway"] is not True:
        raise DeployError("executable Preview requires network.single_nat_gateway=true")

    compute = require_mapping(config["compute"], "compute")
    require_exact_keys(compute, {"system", "cpu"}, "compute")
    validate_node_group(compute["system"], "compute.system")
    validate_node_group(compute["cpu"], "compute.cpu")

    application = require_mapping(config["application"], "application")
    require_exact_keys(
        application, {"repository", "commit", "require_source"}
        | ({"runtime_source_commits"} if "runtime_source_commits" in application else set()),
        "application"
    )
    repository = require_string(application["repository"], "application.repository")
    if repository != APPLICATION_REPOSITORY:
        raise DeployError(f"application.repository must equal {APPLICATION_REPOSITORY}")
    source_commit = require_string(application["commit"], "application.commit")
    if not COMMIT_RE.fullmatch(source_commit):
        raise DeployError(
            "application.commit must be a 40-character lowercase Git commit"
        )
    if not isinstance(application["require_source"], bool):
        raise DeployError("application.require_source must be a boolean")
    if application["require_source"]:
        raise DeployError(
            "application.require_source=true is not supported: this release ships "
            "prebuilt Management/Driver images; application.commit is an opaque "
            "build identifier"
        )
    if "runtime_source_commits" in application:
        role_sources = require_mapping(
            application["runtime_source_commits"], "application.runtime_source_commits"
        )
        require_exact_keys(role_sources, {"management", "driver"}, "application.runtime_source_commits")
        for role, commit in role_sources.items():
            if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
                raise DeployError(f"application.runtime_source_commits.{role} must be a 40-character lowercase Git commit")
        if role_sources["management"] != source_commit:
            raise DeployError("application.runtime_source_commits.management must equal application.commit")

    images = require_mapping(config["images"], "images")
    require_exact_keys(
        images, {"registry", "management", "driver", "litellm", "postgres"}, "images"
    )
    registry_image = validate_image(images["registry"], "images.registry")
    management_image = validate_image(
        images["management"], "images.management", expected_tag=f"sha-{runtime_source_commit(config, 'management')}"
    )
    driver_image = validate_image(
        images["driver"], "images.driver", expected_tag=f"sha-{runtime_source_commit(config, 'driver')}"
    )
    litellm_image = validate_image(images["litellm"], "images.litellm")
    postgres_image = validate_image(images["postgres"], "images.postgres")

    management = require_mapping(config["management"], "management")
    require_exact_keys(
        management,
        {
            "profile",
            "storage_gib",
            "seed_demo",
            "registry_pull_secret_name",
            "exposure",
        }
        | ({"manual_capacity"} if "manual_capacity" in management else set()),
        "management",
    )
    if management["profile"] != "executable-preview":
        raise DeployError(
            "this installer supports only management.profile=executable-preview"
        )
    require_int(management["storage_gib"], "management.storage_gib", 4, 1024)
    if management["seed_demo"] is not False:
        raise DeployError(
            "management.seed_demo must be false; the immutable smoke catalog is explicit"
        )
    pull_secret = management["registry_pull_secret_name"]
    if pull_secret is not None:
        pull_secret = require_string(
            pull_secret, "management.registry_pull_secret_name"
        )
        if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?", pull_secret):
            raise DeployError(
                "management.registry_pull_secret_name must be a DNS label"
            )
    if management["exposure"] != "port-forward":
        raise DeployError("Preview supports only operator port-forward exposure")
    try:
        capacity_settings(config)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise DeployError(str(error)) from error

    litellm = require_mapping(config["litellm"], "litellm")
    require_exact_keys(
        litellm,
        {"postgres_storage_gib", "exposure", "ui_access", "store_model_in_db"},
        "litellm",
    )
    require_int(
        litellm["postgres_storage_gib"], "litellm.postgres_storage_gib", 4, 1024
    )
    if litellm["exposure"] != "port-forward" or litellm["ui_access"] != "admin-only":
        raise DeployError(
            "LiteLLM Preview supports only admin-only operator port-forward access"
        )
    if litellm["store_model_in_db"] is not True:
        raise DeployError(
            "LiteLLM UI model management requires litellm.store_model_in_db=true"
        )

    execution = require_mapping(config["execution"], "execution")
    require_exact_keys(
        execution,
        {
            "object_store_gib",
            "registry_storage_gib",
            "max_active_drivers",
            "agent_external_egress",
            "smoke",
        }
        | ({"model_domains"} if "model_domains" in execution else set())
        | (
            {"task_verifier_workspace_bytes"}
            if "task_verifier_workspace_bytes" in execution
            else set()
        )
        | (
            {"verifier_node_allowlists"}
            if "verifier_node_allowlists" in execution
            else set()
        )
        | (
            {"verifier_placement_required_task_ids"}
            if "verifier_placement_required_task_ids" in execution
            else set()
        ),
        "execution",
    )
    if "verifier_node_allowlists" in execution:
        validate_verifier_node_allowlists(execution["verifier_node_allowlists"])
    from task_selection import verifier_workspace_budgets

    verifier_workspace_budgets(config)
    if "verifier_placement_required_task_ids" in execution:
        validate_verifier_placement_required_task_ids(
            execution["verifier_placement_required_task_ids"]
        )
    if "model_domains" in execution:
        domains = execution["model_domains"]
        if (
            not isinstance(domains, list)
            or len(domains) > 32
            or any(
                not isinstance(host, str)
                or re.fullmatch(MODEL_DOMAIN_PATTERN, host) is None
                for host in domains
            )
        ):
            raise DeployError(
                "execution.model_domains must contain lowercase DNS hostnames"
            )
    require_int(execution["object_store_gib"], "execution.object_store_gib", 1, 1024)
    require_int(
        execution["registry_storage_gib"], "execution.registry_storage_gib", 1, 1024
    )
    require_int(
        execution["max_active_drivers"], "execution.max_active_drivers", 1, 1000
    )
    if not isinstance(execution["agent_external_egress"], bool):
        raise DeployError("execution.agent_external_egress must be a boolean")
    smoke = require_mapping(execution["smoke"], "execution.smoke")
    require_exact_keys(smoke, {"enabled", "timeout_seconds"}, "execution.smoke")
    if smoke["enabled"] is not True:
        raise DeployError("executable Preview requires execution.smoke.enabled=true")
    require_int(smoke["timeout_seconds"], "execution.smoke.timeout_seconds", 120, 3600)

    features = require_mapping(config["features"], "features")
    require_exact_keys(
        features,
        {"run_execution", "litellm", "modal_gpu"}
        | ({"task_model_aliases"} if "task_model_aliases" in features else set()),
        "features",
    )
    if "task_model_aliases" in features:
        # LiteLLM aliases this site serves for task additional_keys (task_selection.py).
        aliases = features["task_model_aliases"]
        if (
            not isinstance(aliases, list)
            or len(set(aliases)) != len(aliases)
            or any(
                not isinstance(alias, str)
                or re.fullmatch(r"[a-z](?:[a-z0-9._-]{0,126}[a-z0-9])?", alias) is None
                for alias in aliases
            )
        ):
            raise DeployError("features.task_model_aliases must be unique LiteLLM model aliases")
    if (
        features["run_execution"] is not True
        or features["litellm"] is not True
        or type(features["modal_gpu"]) is not bool
    ):
        raise DeployError(
            "executable Preview requires execution and LiteLLM enabled and a boolean modal_gpu"
        )
    if features["modal_gpu"] or "modal" in config:
        modal = require_mapping(config.get("modal"), "modal")
        modal_keys = {"environment", "app_name", "credentials_secret"}
        if "ecr_pull_secret_ref" in modal:
            modal_keys.add("ecr_pull_secret_ref")
            reference = modal["ecr_pull_secret_ref"]
            if reference != "":
                require_string(reference, "modal.ecr_pull_secret_ref")
                if not re.fullmatch(r"[a-z][a-z0-9-]{0,61}[a-z0-9]", reference):
                    raise DeployError(
                        "modal.ecr_pull_secret_ref must be a lowercase DNS label"
                    )
        require_exact_keys(modal, modal_keys, "modal")
        for key in ("environment", "app_name", "credentials_secret"):
            value = require_string(modal[key], f"modal.{key}")
            if not re.fullmatch(r"[a-z][a-z0-9-]{0,61}[a-z0-9]", value):
                raise DeployError(f"modal.{key} must be a lowercase DNS label")

    tags = require_mapping(config["tags"], "tags")
    if not tags or len(tags) > 50:
        raise DeployError("tags must contain 1-50 entries")
    for key, value in tags.items():
        require_string(key, "tags key")
        require_string(value, f"tags.{key}")
        if len(key) > 128 or len(value) > 256:
            raise DeployError(f"tag is too long: {key}")

    validated = json.loads(json.dumps(config))
    validated["images"]["registry"] = registry_image
    validated["images"]["management"] = management_image
    validated["images"]["driver"] = driver_image
    validated["images"]["litellm"] = litellm_image
    validated["images"]["postgres"] = postgres_image
    return validated


def read_and_validate(config_path: Path) -> tuple[dict[str, Any], str]:
    raw = load_json(config_path)
    return validate_config(raw), sha256_bytes(canonical_json(raw))


def render_bundle(
    config_path: Path, *, catalog_path: Path | None = None
) -> tuple[dict[str, Any], Path]:
    config, config_sha256 = read_and_validate(config_path)
    _, selection = load_task_selection(config, catalog_path)
    output = generated_directory(config)
    output.mkdir(parents=True, exist_ok=True)
    site, terraform, network, compute = (
        config["site"],
        config["terraform"],
        config["network"],
        config["compute"],
    )
    bootstrap_variables = {
        "aws_region": site["aws_region"],
        "expected_account_id": site["expected_account_id"],
        "site_name": site["name"],
        "state_bucket_name": terraform["state_bucket"],
        "tags": config["tags"],
    }
    platform_variables = {
        "aws_region": site["aws_region"],
        "expected_account_id": site["expected_account_id"],
        "site_name": site["name"],
        "kubernetes_version": site["kubernetes_version"],
        "vpc_cidr": network["vpc_cidr"],
        "service_ipv4_cidr": network["service_ipv4_cidr"],
        "local_registry_endpoint": f"{network['registry_cluster_ip']}:5000",
        "availability_zone_count": network["availability_zone_count"],
        "operator_cidrs": network["operator_cidrs"],
        "single_nat_gateway": network["single_nat_gateway"],
        "operator_principals": terraform.get("operator_principals", []),
        "system_node_group": compute["system"],
        "cpu_node_group": compute["cpu"],
        "tags": config["tags"],
    }
    capacity = capacity_settings(config)
    if capacity is not None:
        platform_variables["manual_capacity_nodegroup_arns"] = [
            capacity[role]["node_group_arn"] for role in ("m6i", "c6i")
        ]
    backend = (
        f"bucket       = {json.dumps(terraform['state_bucket'])}\n"
        f"key          = {json.dumps(terraform['state_key'])}\n"
        f"region       = {json.dumps(site['aws_region'])}\n"
        "encrypt      = true\nuse_lockfile = true\n"
    ).encode()
    bootstrap_backend = f"path = {json.dumps(str(state_directory(config) / 'bootstrap.tfstate'))}\n".encode()
    release = {
        "schema_version": "frontier.deployment-release/v2",
        "site": {
            key: site[key]
            for key in (
                "name",
                "expected_account_id",
                "aws_region",
                "kubernetes_version",
            )
        },
        "application": config["application"],
        "images": {
            name: config["images"][name] for name in ("registry", "litellm", "postgres")
        },
        "profile": "executable-preview",
        "run_execution": "ENABLED_CPU_SCRIPTED",
        "external_integrations": {
            "litellm_ui": "ENABLED_ADMIN_ONLY",
            "provider_models": "OPERATOR_CONFIGURATION_REQUIRED",
            "model_enabled_runs": "GATEWAY_SMOKE_REQUIRED",
            "modal_gpu": "ENABLED" if config["features"]["modal_gpu"] else "DISABLED",
        },
        "config_sha256": config_sha256,
    }
    rendered = {
        "bootstrap.auto.tfvars.json": canonical_json(bootstrap_variables),
        "platform.auto.tfvars.json": canonical_json(platform_variables),
        "backend.hcl": backend,
        "bootstrap-backend.hcl": bootstrap_backend,
        "release.json": canonical_json(release),
    }
    for name, content in rendered.items():
        atomic_write(output / name, content)
    checksums = [
        f"{hashlib.sha256(content).hexdigest()}  {name}"
        for name, content in sorted(rendered.items())
    ]
    atomic_write(output / "bundle.sha256", ("\n".join(checksums) + "\n").encode())
    atomic_write(output / "task-selection.json", canonical_json(asdict(selection)))
    return config, output
