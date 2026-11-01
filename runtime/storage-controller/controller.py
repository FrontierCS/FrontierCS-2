#!/usr/bin/env python3
"""Provision the exact local-object PVC expected by local-k8s Driver Jobs."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import ssl
import time
from urllib import error, parse, request
from pathlib import Path

from modal_credentials import reconcile_modal_credentials


API_HOST = os.environ.get("KUBERNETES_SERVICE_HOST", "").strip()
API_PORT = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443").strip()
if not API_HOST or not API_PORT.isdigit():
    raise RuntimeError("Kubernetes Service environment is unavailable")
API = f"https://{API_HOST}:{API_PORT}"
TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
RUN_NAMESPACE = re.compile(r"^frontier-run-[0-9a-hjkmnp-tv-z]{26}$")
RUN_ID = re.compile(r"^run_[0-9A-HJKMNP-TV-Z]{26}$")
ATTEMPT_ID = re.compile(r"^attempt_[0-9A-HJKMNP-TV-Z]{26}$")
PVC_NAME = "frontier-local-object-store"
NETWORK_POLICY_NAME = "frontier-agent-external-egress"
REGISTRY_POLICY_NAME = "frontier-driver-local-registry"
PUBLIC_HTTPS_EXCEPT_CIDRS = [
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.168.0.0/16",
]


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def required_boolean(name: str) -> bool:
    value = required(name)
    if value not in {"true", "false"}:
        raise RuntimeError(f"{name} must be true or false")
    return value == "true"


VOLUME_HANDLE = required("FRONTIER_EFS_OBJECT_VOLUME_HANDLE")
STORAGE_GIB = int(required("FRONTIER_OBJECT_STORE_GIB"))
POLL_SECONDS = float(os.environ.get("FRONTIER_RECONCILE_SECONDS", "1"))
AGENT_EXTERNAL_EGRESS = required_boolean("FRONTIER_AGENT_EXTERNAL_EGRESS")
MODAL_GPU = os.environ.get("FRONTIER_MODAL_GPU", "false")
if MODAL_GPU not in {"true", "false"}:
    raise RuntimeError("FRONTIER_MODAL_GPU must be true or false")
MODAL_SECRET_NAME = required("FRONTIER_MODAL_SECRET_NAME") if MODAL_GPU == "true" else None
DNS_CIDR = required("FRONTIER_DNS_CIDR") if AGENT_EXTERNAL_EGRESS else None
REGISTRY_CIDR = os.environ.get("FRONTIER_REGISTRY_SERVICE_CIDR", "").strip() or None
if REGISTRY_CIDR is not None:
    registry_network = ipaddress.ip_network(REGISTRY_CIDR, strict=True)
    if registry_network.version != 4 or registry_network.prefixlen != 32:
        raise RuntimeError("FRONTIER_REGISTRY_SERVICE_CIDR must be an exact IPv4 /32")
if not re.fullmatch(r"fs-[0-9a-f]+::fsap-[0-9a-f]+", VOLUME_HANDLE):
    raise RuntimeError("FRONTIER_EFS_OBJECT_VOLUME_HANDLE is invalid")
if not 1 <= STORAGE_GIB <= 1024 or not 0.25 <= POLL_SECONDS <= 60:
    raise RuntimeError("storage size or reconcile interval is out of range")
if DNS_CIDR is not None:
    try:
        dns_network = ipaddress.ip_network(DNS_CIDR, strict=True)
    except ValueError as exc:
        raise RuntimeError("FRONTIER_DNS_CIDR is invalid") from exc
    if dns_network.version != 4 or dns_network.prefixlen != 32:
        raise RuntimeError("FRONTIER_DNS_CIDR must be an exact IPv4 /32")


SSL_CONTEXT = ssl.create_default_context(cafile=CA_PATH)


def api(method: str, path: str, body: dict | None = None, *, absent_ok: bool = False):
    encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    with open(TOKEN_PATH, encoding="utf-8") as token_file:
        token = token_file.read().strip()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if encoded is not None:
        headers["Content-Type"] = "application/json"
    call = request.Request(API + path, data=encoded, headers=headers, method=method)
    try:
        with request.urlopen(call, context=SSL_CONTEXT, timeout=15) as response:
            payload = response.read(2 * 1024 * 1024 + 1)
    except error.HTTPError as exc:
        if absent_ok and exc.code == 404:
            return None
        if method == "POST" and exc.code == 409:
            return {"conflict": True}
        raise RuntimeError(f"Kubernetes API {method} {path} returned {exc.code}") from exc
    if len(payload) > 2 * 1024 * 1024:
        raise RuntimeError("Kubernetes API response exceeded 2 MiB")
    return json.loads(payload) if payload else {}


def ownership(namespace: dict) -> tuple[str, str] | None:
    metadata = namespace.get("metadata", {})
    name = metadata.get("name", "")
    labels = metadata.get("labels", {})
    run_id = labels.get("frontier.dev/run-id", "")
    attempt_id = labels.get("frontier.dev/attempt-id", "")
    if (
        not RUN_NAMESPACE.fullmatch(name)
        or labels.get("frontier.dev/owner") != "management"
        or labels.get("frontier.dev/resource-kind") != "run-namespace"
        or not RUN_ID.fullmatch(run_id)
        or not ATTEMPT_ID.fullmatch(attempt_id)
    ):
        return None
    return run_id, attempt_id


def pv_document(namespace: str, run_id: str, attempt_id: str) -> dict:
    name = "frontier-object-" + namespace.removeprefix("frontier-run-")
    labels = {
        "app.kubernetes.io/part-of": "frontiercs2",
        "frontier.dev/owner": "storage-controller",
        "frontier.dev/run-id": run_id,
        "frontier.dev/attempt-id": attempt_id,
        "frontier.dev/run-namespace": namespace,
    }
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolume",
        "metadata": {"name": name, "labels": labels},
        "spec": {
            "capacity": {"storage": f"{STORAGE_GIB}Gi"},
            "volumeMode": "Filesystem",
            "accessModes": ["ReadWriteMany"],
            "persistentVolumeReclaimPolicy": "Retain",
            "storageClassName": "",
            "mountOptions": ["tls"],
            "csi": {
                "driver": "efs.csi.aws.com",
                "volumeHandle": VOLUME_HANDLE,
            },
        },
    }


def pvc_document(namespace: str, pv: dict, run_id: str, attempt_id: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": PVC_NAME,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/part-of": "frontiercs2",
                "frontier.dev/owner": "storage-controller",
                "frontier.dev/run-id": run_id,
                "frontier.dev/attempt-id": attempt_id,
            },
        },
        "spec": {
            "accessModes": ["ReadWriteMany"],
            "volumeMode": "Filesystem",
            "storageClassName": "",
            "volumeName": pv["metadata"]["name"],
            "resources": {"requests": {"storage": f"{STORAGE_GIB}Gi"}},
        },
    }


def network_policy_document(namespace: str, dns_cidr: str) -> dict:
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": NETWORK_POLICY_NAME,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/part-of": "frontiercs2",
                "frontier.dev/owner": "storage-controller",
            },
        },
        "spec": {
            "podSelector": {
                "matchLabels": {"frontier.dev/network-role": "agent"},
            },
            "policyTypes": ["Egress"],
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
                            "ipBlock": {
                                "cidr": "0.0.0.0/0",
                                "except": list(PUBLIC_HTTPS_EXCEPT_CIDRS),
                            },
                        },
                    ],
                    "ports": [{"protocol": "TCP", "port": 443}],
                },
            ],
        },
    }


def validate_existing(actual: dict, expected: dict, label: str) -> None:
    for pointer in (("metadata", "labels"), ("spec", "storageClassName")):
        left = actual
        right = expected
        for part in pointer:
            left = left.get(part, {}) if isinstance(left, dict) else None
            right = right.get(part, {}) if isinstance(right, dict) else None
        # The API server normalizes storageClassName "" (static provisioning)
        # to an absent field on readback; treat both spellings as equal.
        if pointer == ("spec", "storageClassName"):
            left = left if isinstance(left, str) else ""
            right = right if isinstance(right, str) else ""
        if left != right:
            raise RuntimeError(f"existing {label} differs at {'/'.join(pointer)}")
    if label == "PV" and actual.get("spec", {}).get("csi") != expected["spec"]["csi"]:
        raise RuntimeError("existing PV has a different EFS volume handle")
    if label == "PVC" and actual.get("spec", {}).get("volumeName") != expected["spec"]["volumeName"]:
        raise RuntimeError("existing PVC is bound to a different PV")


def registry_policy_document(namespace: str, run_id: str, attempt_id: str, cidr: str) -> dict:
    network = ipaddress.ip_network(cidr, strict=True)
    if network.version != 4 or network.prefixlen != 32:
        raise ValueError("registry Service must be an exact IPv4 /32")
    if not RUN_NAMESPACE.fullmatch(namespace) or not RUN_ID.fullmatch(run_id) or not ATTEMPT_ID.fullmatch(attempt_id):
        raise ValueError("registry policy requires exact Run and attempt identities")
    return {
        "apiVersion": "networking.k8s.io/v1", "kind": "NetworkPolicy",
        "metadata": {"name": REGISTRY_POLICY_NAME, "namespace": namespace, "labels": {
            "app.kubernetes.io/part-of": "frontiercs2", "frontier.dev/owner": "storage-controller",
            "frontier.dev/run-id": run_id, "frontier.dev/attempt-id": attempt_id}},
        "spec": {
            "podSelector": {"matchLabels": {
                "frontier.dev/owner": "management", "frontier.dev/network-role": "driver",
                "frontier.dev/run-id": run_id, "frontier.dev/attempt-id": attempt_id}},
            "policyTypes": ["Egress"],
            "egress": [{"to": [{"ipBlock": {"cidr": str(network)}}],
                        "ports": [{"protocol": "TCP", "port": 5000}]}],
        },
    }


def validate_network_policy(actual: dict, expected: dict) -> None:
    if actual.get("metadata", {}).get("labels") != expected["metadata"]["labels"]:
        raise RuntimeError("existing Agent external-egress NetworkPolicy has different labels")
    if actual.get("spec") != expected["spec"]:
        raise RuntimeError("existing Agent external-egress NetworkPolicy has a different spec")


def create_or_replace_network_policy(namespace: str, desired: dict) -> None:
    collection_path = f"/apis/networking.k8s.io/v1/namespaces/{namespace}/networkpolicies"
    name = desired["metadata"]["name"]
    if name not in {NETWORK_POLICY_NAME, REGISTRY_POLICY_NAME}:
        raise ValueError("unsupported local NetworkPolicy")
    policy_path = f"{collection_path}/{name}"
    actual = api("GET", policy_path, absent_ok=True)
    if actual is None:
        api("POST", collection_path, desired)
        actual = api("GET", policy_path)
    if name == REGISTRY_POLICY_NAME and actual.get("metadata", {}).get("labels") != desired["metadata"]["labels"]:
        raise RuntimeError("existing registry policy has different ownership")
    if (
        actual.get("metadata", {}).get("labels") != desired["metadata"]["labels"]
        or actual.get("spec") != desired["spec"]
    ):
        resource_version = actual.get("metadata", {}).get("resourceVersion")
        if not resource_version:
            raise RuntimeError("existing Agent external-egress NetworkPolicy lacks resourceVersion")
        replacement = {
            **desired,
            "metadata": {**desired["metadata"], "resourceVersion": resource_version},
        }
        api("PUT", policy_path, replacement)
        actual = api("GET", policy_path)
    validate_network_policy(actual, desired)


def remove_agent_external_egress_policy(namespace: str) -> bool:
    policy_path = (
        f"/apis/networking.k8s.io/v1/namespaces/{namespace}"
        f"/networkpolicies/{NETWORK_POLICY_NAME}"
    )
    if api("GET", policy_path, absent_ok=True) is None:
        return False
    api("DELETE", policy_path, absent_ok=True)
    print(f"storage-controller removed {NETWORK_POLICY_NAME} from {namespace}", flush=True)
    return True


def reconcile_namespace(namespace: dict) -> None:
    if namespace.get("metadata", {}).get("deletionTimestamp"):
        return
    identity = ownership(namespace)
    if identity is None:
        return
    name = namespace["metadata"]["name"]
    run_id, attempt_id = identity
    if not AGENT_EXTERNAL_EGRESS:
        remove_agent_external_egress_policy(name)
    # EKS can enforce the pre-DNAT Service address. The upstream frozen policy
    # already allows registry Pods; add only the configured local Service /32.
    # Reconcile before making the Driver's object-store PVC available.
    if REGISTRY_CIDR:
        create_or_replace_network_policy(name, registry_policy_document(name, run_id, attempt_id, REGISTRY_CIDR))
    if MODAL_SECRET_NAME:
        reconcile_modal_credentials(api, namespace, MODAL_SECRET_NAME, Path("/var/run/secrets/frontier-modal"))
    desired_pv = pv_document(name, run_id, attempt_id)
    pv_name = desired_pv["metadata"]["name"]
    actual_pv = api("GET", f"/api/v1/persistentvolumes/{pv_name}", absent_ok=True)
    if actual_pv is None:
        api("POST", "/api/v1/persistentvolumes", desired_pv)
        actual_pv = api("GET", f"/api/v1/persistentvolumes/{pv_name}")
    validate_existing(actual_pv, desired_pv, "PV")
    desired_pvc = pvc_document(name, desired_pv, run_id, attempt_id)
    pvc_path = f"/api/v1/namespaces/{name}/persistentvolumeclaims/{PVC_NAME}"
    actual_pvc = api("GET", pvc_path, absent_ok=True)
    if actual_pvc is None:
        api("POST", f"/api/v1/namespaces/{name}/persistentvolumeclaims", desired_pvc)
        actual_pvc = api("GET", pvc_path)
    validate_existing(actual_pvc, desired_pvc, "PVC")
    if AGENT_EXTERNAL_EGRESS:
        assert DNS_CIDR is not None
        create_or_replace_network_policy(name, network_policy_document(name, DNS_CIDR))


def cleanup_orphan_pvs(active_namespaces: set[str]) -> None:
    selector = parse.quote("frontier.dev/owner=storage-controller")
    result = api("GET", f"/api/v1/persistentvolumes?labelSelector={selector}")
    for pv in result.get("items", []):
        metadata = pv.get("metadata", {})
        labels = metadata.get("labels", {})
        namespace = labels.get("frontier.dev/run-namespace", "")
        name = metadata.get("name", "")
        expected_name = "frontier-object-" + namespace.removeprefix("frontier-run-")
        claim_namespace = pv.get("spec", {}).get("claimRef", {}).get("namespace")
        if (
            not RUN_NAMESPACE.fullmatch(namespace)
            or name != expected_name
            or claim_namespace != namespace
            or pv.get("spec", {}).get("persistentVolumeReclaimPolicy") != "Retain"
            or pv.get("spec", {}).get("csi", {}).get("volumeHandle") != VOLUME_HANDLE
            or pv.get("status", {}).get("phase") not in {"Released", "Failed"}
        ):
            continue
        if namespace in active_namespaces:
            continue
        if api("GET", f"/api/v1/namespaces/{namespace}", absent_ok=True) is not None:
            continue
        api("DELETE", f"/api/v1/persistentvolumes/{name}")


def reconcile_once() -> None:
    selector = parse.quote("frontier.dev/owner=management,frontier.dev/resource-kind=run-namespace")
    result = api("GET", f"/api/v1/namespaces?labelSelector={selector}")
    for namespace in result.get("items", []):
        try:
            reconcile_namespace(namespace)
        except Exception as exc:
            # A deletion race or broken namespace must not starve other Runs.
            name = namespace.get("metadata", {}).get("name", "unknown")
            print(f"storage-controller namespace {name} reconcile failed: {exc}", flush=True)
    active = {
        namespace.get("metadata", {}).get("name", "")
        for namespace in result.get("items", [])
        if ownership(namespace) is not None
    }
    cleanup_orphan_pvs(active)


def loop() -> None:
    while True:
        try:
            reconcile_once()
        except Exception as exc:  # keep reconciling after bounded API/network failures
            print(f"storage-controller reconcile failed: {exc}", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    loop()
