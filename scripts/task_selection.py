"""Site task selection over the raw publication catalog, before normalization."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from preview.infra import load_json


@dataclass
class TaskSelection:
    selected: list[str]
    excluded: dict[str, str]
    effective_catalog: dict[str, Any]
    effective_catalog_sha256: str
    definitions: dict[str, Any]
    reference_names: list[str]
    model_domains: list[str]
    selection_digest: str
    catalog_sha256: str


def requirements(definition: dict) -> dict:
    from preview import infra as deploy

    result = {"modal_gpu": False}
    if "gpu" in definition:
        gpu = deploy.require_mapping(definition["gpu"], "definition.gpu")
        if not gpu or set(gpu) - {"agent", "verifier"}:
            raise deploy.DeployError(
                "definition.gpu must contain agent and/or verifier policies"
            )
        for role, fields in (
            (
                "agent",
                {
                    "allowed_types",
                    "max_instances",
                    "max_gpu_seconds",
                    "max_instance_seconds",
                },
            ),
            ("verifier", {"max_gpu_instances", "max_gpu_seconds"}),
        ):
            if role not in gpu:
                continue
            policy = deploy.require_mapping(gpu[role], f"definition.gpu.{role}")
            deploy.require_exact_keys(policy, fields, f"definition.gpu.{role}")
            for field in fields - {"allowed_types"}:
                value = policy[field]
                if type(value) is not int or value < 0:
                    raise deploy.DeployError(
                        f"definition.gpu.{role}.{field} must be a nonnegative integer"
                    )
            if role == "agent":
                types = policy["allowed_types"]
                if (
                    not isinstance(types, list)
                    or any(
                        not isinstance(item, str) or not item or item != item.strip()
                        for item in types
                    )
                    or len(set(types)) != len(types)
                ):
                    raise deploy.DeployError(
                        "definition.gpu.agent.allowed_types must be unique nonempty strings"
                    )
            instances = policy[
                "max_instances" if role == "agent" else "max_gpu_instances"
            ]
            result["modal_gpu"] |= instances > 0
    metadata = deploy.require_mapping(
        definition.get("metadata", {}), "definition.metadata"
    )
    accelerator = metadata.get("accelerator")
    accelerator = (
        {}
        if accelerator is None
        else deploy.require_mapping(accelerator, "definition.metadata.accelerator")
    )
    if "agent_gpu_interface" in accelerator:
        result["agent_gpu_interface"] = deploy.require_string(
            accelerator["agent_gpu_interface"],
            "definition.metadata.accelerator.agent_gpu_interface",
        )
    if "additional_keys" in definition:
        # Like a GPU requirement: the site must serve every task-model alias
        # (a LiteLLM route with a nonzero price, so budget_usd is enforced).
        import render_task_definition

        keys = render_task_definition.validate_additional_keys(definition["additional_keys"])
        result["task_model_aliases"] = sorted({a for k in keys for a in k["model_aliases"]})
    return result


def deployment_catalog(catalog: dict) -> dict:
    """Project publication v2 onto the frozen deployment receipt input contract.

    Reference/baseline metadata is consumed by parity, not task/Codex seeding.
    Keep the v1 schema marker here so existing receipt bytes still bind.
    """
    return {
        **catalog,
        "schema_version": "frontier.deployment-task-catalog/v1",
        "tasks": [
            {
                key: value
                for key, value in task.items()
                if key not in {"reference", "baseline"}
            }
            for task in catalog["tasks"]
        ],
    }


def select_tasks(config: dict, catalog: dict, definitions: dict) -> TaskSelection:
    from preview import infra as deploy

    selected, excluded = [], {}
    for task in catalog["tasks"]:
        name = task["name"]
        if name not in definitions:
            raise deploy.DeployError(f"task has no definition: {name}")
        needs = requirements(definitions[name])
        served = set(config["features"].get("task_model_aliases", []))
        missing_aliases = [a for a in needs.get("task_model_aliases", []) if a not in served]
        if needs["modal_gpu"] and not config["features"]["modal_gpu"]:
            excluded[name] = "requires features.modal_gpu"
        elif missing_aliases:
            excluded[name] = "requires features.task_model_aliases: " + ", ".join(missing_aliases)
        else:
            interface = needs.get("agent_gpu_interface")
            if (
                interface is not None
                and interface not in attested_agent_gpu_interfaces()
            ):
                raise deploy.DeployError(
                    f"{name}: agent_gpu_interface {interface} is not attested by a pinned Candidate overlay"
                )
            selected.append(name)
    if not selected:
        raise deploy.DeployError("site task selection is empty")
    budgets = verifier_workspace_budgets(config)
    if set(budgets) - set(selected):
        raise deploy.DeployError("verifier workspace budgets must name selected tasks")
    effective_definitions = {name: definitions[name] for name in selected}
    for name, budget in budgets.items():
        effective_definitions[name] = copy.deepcopy(definitions[name])
        effective_definitions[name]["verifier"]["workspace_bytes"] = budget
    effective = {
        key: (
            [task for task in value if task["name"] in selected]
            if key == "tasks"
            else value
        )
        for key, value in catalog.items()
    }
    catalog_hash = deploy.sha256_bytes(
        deploy.canonical_json(deployment_catalog(catalog))
    )
    effective_hash = deploy.sha256_bytes(
        deploy.canonical_json(deployment_catalog(effective))
    )
    domains = sorted(set(config["execution"].get("model_domains", ["openrouter.ai"])))
    binding = {
        "selected": selected,
        "excluded": excluded,
        "effective_catalog_sha256": effective_hash,
        "model_domains": domains,
        "catalog_sha256": catalog_hash,
    }
    if budgets:
        binding["task_verifier_workspace_bytes"] = budgets
    references = {
        task["name"] for task in catalog["tasks"] if task.get("reference") is not None
    }
    return TaskSelection(
        selected,
        excluded,
        effective,
        effective_hash,
        effective_definitions,
        [name for name in selected if name in references],
        domains,
        deploy.sha256_bytes(deploy.canonical_json(binding)),
        catalog_hash,
    )


def verifier_workspace_budgets(config: dict) -> dict[str, int]:
    from preview import infra as deploy

    budgets = deploy.require_mapping(
        config.get("execution", {}).get("task_verifier_workspace_bytes", {}),
        "execution.task_verifier_workspace_bytes",
    )
    for name, budget in budgets.items():
        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
            raise deploy.DeployError("verifier workspace budget needs a problem name")
        deploy.require_int(budget, f"verifier workspace budget.{name}", 1, 1 << 40)
    return dict(budgets)


def attested_agent_gpu_interfaces() -> set[str]:
    """Task publication can serve both Codex V2 and the separately registered Kimi V3.

    This does not change the seeded Codex identity or make it a V3 Candidate.
    Kimi's additional contract must bind its checked-in image and source receipt.
    """
    from preview import infra as deploy

    interfaces = {deploy.CODEX_AGENT_GPU_INTERFACE}
    path = deploy.REPOSITORY_ROOT / "config/kimi-candidate-image.json"
    if not path.is_file():
        return interfaces
    receipt = load_json(path)
    image = deploy.validate_image(
        {key: receipt.get(key) for key in ("repository", "tag", "digest")},
        "Kimi Candidate image",
    )
    helper = deploy.require_mapping(receipt.get("gpu_helper"), "Kimi gpu_helper")
    if (
        image["repository"] != "docker.io/joyemang33/fcs2-kimi-overlay"
        or image["digest"] == deploy.TASK_PLACEHOLDER_DIGEST
        or image["tag"] != "sha-" + str(receipt.get("source_commit"))
        or not deploy.TASK_IMAGE_TAG_RE.fullmatch(image["tag"])
        or helper.get("overlay_digest") != image["digest"]
        or helper.get("contract") != "frontier.gpu-helper/v3"
    ):
        raise deploy.DeployError(
            "Kimi GPU helper receipt does not bind the pinned V3 overlay"
        )
    files = deploy.require_mapping(helper.get("files"), "Kimi gpu_helper.files")
    selection = deploy.require_mapping(
        receipt.get("manifest_selection"), "Kimi manifest selection"
    )
    if (
        selection.get("leaf_digest") != image["digest"]
        or (
            "index_digest" in selection
            and (
                selection["index_digest"] == image["digest"]
                or not deploy.DIGEST_RE.fullmatch(str(selection["index_digest"]))
            )
        )
        or selection.get("os") != "linux"
        or selection.get("architecture") != "amd64"
        or selection.get("media_type")
        not in {
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.v2+json",
        }
        or selection.get("public_manifest_sha256") != image["digest"]
        or selection.get("internal_manifest_sha256") != image["digest"]
        or selection.get("verified") is not True
    ):
        raise deploy.DeployError(
            "Kimi Candidate must bind a verified linux/amd64 manifest"
        )
    expected_files = {
        "candidates/kimi-code/Dockerfile",
        "candidates/kimi-code/launcher.py",
        "candidates/kimi-code/submit/frontier_submit.py",
        "candidates/kimi-code/submit/SKILL.md",
        "candidates/kimi-code/gpu-helper/frontier_gpu.py",
        "candidates/kimi-code/gpu-helper/frontier_gpu_connect_proxy.py",
        "candidates/kimi-code/gpu-helper/SKILL.md",
    }
    missing = expected_files - files.keys()
    if missing:
        raise deploy.DeployError(
            "Kimi frontier.gpu-helper/v3 is not attested: "
            "source receipt is missing hashes for " + ", ".join(sorted(missing))
        )
    for name, digest in files.items():
        if not isinstance(digest, str) or not deploy.DIGEST_RE.fullmatch(digest):
            raise deploy.DeployError(
                f"Kimi frontier.gpu-helper/v3 is not attested: "
                f"invalid sha256 source hash for {name}"
            )
        source = deploy.REPOSITORY_ROOT / name
        if (
            not source.resolve().is_relative_to(
                (deploy.REPOSITORY_ROOT / "candidates/kimi-code").resolve()
            )
            or not source.is_file()
            or deploy.sha256_bytes(source.read_bytes()) != digest
        ):
            raise deploy.DeployError(
                f"Kimi frontier.gpu-helper/v3 is not attested: "
                f"source differs from its pinned receipt: {name}"
            )
    from sync_candidate_overlay import DOCKERFILE, FILES, INSTALLED, SOURCE_REPOSITORY

    source = deploy.require_mapping(receipt.get("source"), "Kimi source")
    provenance_path = "candidates/kimi-code/source.json"
    provenance = json.loads((deploy.REPOSITORY_ROOT / provenance_path).read_text())
    if (
        source.get("schema_version") != "frontier.core-candidate-source/v1"
        or source.get("repository") != SOURCE_REPOSITORY
        or source.get("commit") != receipt["source_commit"]
        or source.get("provenance") != provenance_path
        or source.get("dockerfile") != DOCKERFILE
        or provenance.get("repository") != SOURCE_REPOSITORY
        or provenance.get("platform_source_commit") != source["commit"]
        or provenance.get("dockerfile", {}).get("source") != DOCKERFILE
        or source.get("dockerfile_sha256")
        != provenance.get("dockerfile", {}).get("sha256")
        or not isinstance(source.get("dockerfile_sha256"), str)
        or not deploy.DIGEST_RE.fullmatch(source["dockerfile_sha256"])
        or provenance.get("dockerfile", {}).get("local_sha256")
        != files["candidates/kimi-code/Dockerfile"]
    ):
        raise deploy.DeployError(
            "Kimi source provenance does not bind the published image"
        )
    sources = deploy.require_mapping(provenance.get("files"), "Kimi source files")
    readback = deploy.require_mapping(
        receipt.get("registry_readback"), "Kimi registry readback"
    )
    installed = deploy.require_mapping(readback.get("files"), "Kimi installed files")
    if (
        readback.get("schema_version") != "frontier.candidate-file-readback/v1"
        or readback.get("image_digest") != image["digest"]
        or set(sources) != set(FILES)
        or set(installed) != set(FILES)
    ):
        raise deploy.DeployError(
            "Kimi registry readback does not bind the complete installed files"
        )
    for name, core_path in FILES.items():
        local_hash = files["candidates/kimi-code/" + name]
        if sources[name] != {"source": core_path, "sha256": local_hash} or installed[
            name
        ] != {
            "path": INSTALLED[name][0],
            "mode": INSTALLED[name][1],
            "sha256": local_hash,
        }:
            raise deploy.DeployError(f"Kimi source/image file proof differs: {name}")
    interfaces.add(helper["contract"])
    return interfaces
