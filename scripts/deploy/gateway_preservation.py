"""Read-only guard for an existing gateway during a Management/Driver upgrade."""

from __future__ import annotations

import json

from .infra import (
    LITELLM_NAMESPACE,
    DeployError,
    executable,
    kubernetes_environment,
    litellm_serving_receipt,
    run_command,
)


def require(condition, message):
    if not condition:
        raise DeployError("existing LiteLLM gateway: " + message)


def stable_object(value):
    """Exclude observation churn while retaining identity, spec and secret/config data."""
    metadata = value["metadata"]
    return {
        "kind": value["kind"],
        "metadata": {
            k: metadata.get(k)
            for k in ("name", "namespace", "uid", "labels", "annotations")
        },
        **{
            k: v
            for k, v in value.items()
            if k not in ("apiVersion", "kind", "metadata", "status")
        },
    }


def validate_snapshot(resources):
    try:
        return _validate_snapshot(resources)
    except (KeyError, TypeError, ValueError) as exc:
        raise DeployError(
            "existing LiteLLM gateway: incomplete Kubernetes readback"
        ) from exc


def _validate_snapshot(resources):
    by_kind = {
        kind: {o["metadata"]["name"]: o for o in items}
        for kind, items in resources.items()
    }
    for kind, name in (
        ("services", "frontier-litellm"),
        ("statefulsets", "frontier-litellm-postgres"),
        ("persistentvolumeclaims", "data-frontier-litellm-postgres-0"),
        ("secrets", "litellm-master-key"),
        ("configmaps", "frontier-litellm-serving-ready"),
    ):
        require(
            not by_kind[kind][name]["metadata"].get("deletionTimestamp"),
            "required serving identity is terminating",
        )
    service = by_kind["services"]["frontier-litellm"]
    selector = service["spec"].get("selector", {})
    require(bool(selector), "Service has no selector")
    matches = [
        d
        for d in resources["deployments"]
        if all(
            d["spec"]["template"]["metadata"].get("labels", {}).get(k) == v
            for k, v in selector.items()
        )
    ]
    require(len(matches) == 1, "Service must select exactly one gateway Deployment")
    deployment = matches[0]
    require(
        not deployment["metadata"].get("deletionTimestamp"),
        "selected Deployment is terminating",
    )
    status = deployment.get("status", {})
    require(
        status.get("observedGeneration", 0) >= deployment["metadata"]["generation"],
        "selected Deployment generation is not observed",
    )
    require(
        status.get("availableReplicas", 0) >= 1
        and status.get("updatedReplicas", 0) >= 1,
        "selected Deployment is not available",
    )
    endpoints = by_kind["endpoints"]["frontier-litellm"]
    addresses = [
        a for s in endpoints.get("subsets", []) for a in s.get("addresses", [])
    ]
    require(bool(addresses), "Service has no Ready endpoint")
    pods = {p["metadata"]["uid"]: p for p in resources["pods"]}
    replicasets = {p["metadata"]["uid"]: p for p in resources["replicasets"]}
    expected_images = {
        c["name"]: c["image"]
        for c in deployment["spec"]["template"]["spec"]["containers"]
    }
    require(
        all("@sha256:" in image for image in expected_images.values()),
        "selected images are not digest pinned",
    )
    for address in addresses:
        ref = address.get("targetRef", {})
        require(
            ref.get("kind") == "Pod" and ref.get("uid") in pods,
            "endpoint does not bind a current Pod",
        )
        pod = pods[ref["uid"]]
        require(
            not pod["metadata"].get("deletionTimestamp")
            and pod.get("status", {}).get("phase") == "Running",
            "endpoint Pod is not active",
        )
        require(
            address.get("ip") == pod.get("status", {}).get("podIP"),
            "endpoint IP differs from Pod IP",
        )
        require(
            all(
                pod["metadata"].get("labels", {}).get(k) == v
                for k, v in selector.items()
            ),
            "endpoint Pod does not match Service selector",
        )
        require(
            any(
                c["type"] == "Ready" and c["status"] == "True"
                for c in pod.get("status", {}).get("conditions", [])
            ),
            "endpoint Pod is not Ready",
        )
        owner = next(
            (
                r
                for r in pod["metadata"].get("ownerReferences", [])
                if r.get("controller")
            ),
            {},
        )
        rs = replicasets.get(owner.get("uid"), {})
        require(
            any(
                r.get("controller") and r.get("uid") == deployment["metadata"]["uid"]
                for r in rs.get("metadata", {}).get("ownerReferences", [])
            ),
            "endpoint belongs to another Deployment",
        )
        require(
            {c["name"]: c["image"] for c in pod["spec"]["containers"]}
            == expected_images,
            "endpoint image differs from selected Deployment",
        )
        statuses = {c["name"]: c for c in pod["status"].get("containerStatuses", [])}
        for name, image in expected_images.items():
            require(
                statuses.get(name, {}).get("ready") is True
                and statuses[name]
                .get("imageID", "")
                .endswith("@" + image.split("@", 1)[1]),
                "endpoint is not running the pinned image",
            )
    expected = litellm_serving_receipt(
        service,
        by_kind["statefulsets"]["frontier-litellm-postgres"],
        by_kind["persistentvolumeclaims"]["data-frontier-litellm-postgres-0"],
        by_kind["secrets"]["litellm-master-key"],
        deployment,
    )
    receipt = by_kind["configmaps"]["frontier-litellm-serving-ready"]
    require(
        receipt.get("immutable") is True and receipt.get("data") == expected["data"],
        "serving receipt differs from current identities",
    )
    # Preserve all namespace configuration, including blue-green revisions and
    # provider Secret keys that the installer does not generate itself.
    watched = (
        "services",
        "deployments",
        "statefulsets",
        "persistentvolumeclaims",
        "configmaps",
        "secrets",
    )
    return {
        kind: sorted(
            (stable_object(o) for o in resources[kind]),
            key=lambda o: o["metadata"]["name"],
        )
        for kind in watched
    }


def read_gateway_snapshot(kubeconfig):
    command = [executable("kubectl"), "-n", LITELLM_NAMESPACE]
    environment = kubernetes_environment(kubeconfig)
    resources = {}
    for kind in (
        "services",
        "deployments",
        "statefulsets",
        "persistentvolumeclaims",
        "configmaps",
        "secrets",
        "endpoints",
        "pods",
        "replicasets",
    ):
        resources[kind] = json.loads(
            run_command(
                command + ["get", kind, "-o", "json"],
                environment=environment,
                capture=True,
                timeout=60,
            )
        )["items"]
    return validate_snapshot(resources)
