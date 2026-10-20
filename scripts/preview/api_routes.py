"""Exact Kubernetes Service destinations, observed through the cluster API."""

from __future__ import annotations

import hashlib
import ipaddress
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .infra import (
    PLATFORM_NAMESPACE,
    DeployError,
    atomic_write,
    canonical_json,
    canonical_object_sha256,
    executable,
    kubernetes_environment,
    management_api_routes_rbac,
    run_command,
)


def _identity(value: Mapping[str, Any], kind: str, name: str) -> None:
    metadata = value.get("metadata", {})
    if (
        value.get("kind") != kind
        or metadata.get("namespace") != "default"
        or metadata.get("name") != name
        or not metadata.get("uid")
        or not metadata.get("resourceVersion")
    ):
        raise DeployError(
            "Kubernetes API route object identity is incomplete or mismatched"
        )


def _ip(value: str) -> str:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError as error:
        raise DeployError("Kubernetes API destination is not an IP address") from error
    if (
        ip.version != 4
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
    ):
        raise DeployError(
            "Kubernetes API destination must be a routable exact IPv4 address"
        )
    return str(ip)


def route_snapshot(
    config: Mapping[str, Any], service: dict, endpoints: dict, slices: dict
) -> dict:
    _identity(service, "Service", "kubernetes")
    _identity(endpoints, "Endpoints", "kubernetes")
    expected = str(
        ipaddress.ip_network(config["network"]["service_ipv4_cidr"]).network_address + 1
    )
    service_ip = _ip(service.get("spec", {}).get("clusterIP", ""))
    if (
        service_ip != expected
        or service.get("spec", {}).get("type", "ClusterIP") != "ClusterIP"
    ):
        raise DeployError(
            "Kubernetes API Service differs from the configured cluster network"
        )
    ports = service.get("spec", {}).get("ports", [])
    if not any(
        p.get("name") == "https"
        and p.get("port") == 443
        and p.get("targetPort", 443) == 443
        and p.get("protocol", "TCP") == "TCP"
        for p in ports
    ):
        raise DeployError("Kubernetes API Service must target TCP 443")

    def valid_ports(ports):
        return ports and all(
            p.get("name") == "https"
            and p.get("port") == 443
            and p.get("protocol", "TCP") == "TCP"
            for p in ports
        )

    endpoint_ips = set()
    for subset in endpoints.get("subsets", []):
        if not valid_ports(subset.get("ports", [])) or subset.get("notReadyAddresses"):
            raise DeployError(
                "Kubernetes API Endpoints are not ready TCP 443 destinations"
            )
        endpoint_ips.update(_ip(a.get("ip", "")) for a in subset.get("addresses", []))
    if slices.get("kind") not in ("EndpointSliceList", "List"):
        raise DeployError("Kubernetes API EndpointSlice inventory is invalid")
    slice_ips = set()
    objects = [service, endpoints]
    for item in slices.get("items", []):
        _identity(item, "EndpointSlice", item.get("metadata", {}).get("name", ""))
        if (
            item.get("metadata", {}).get("labels", {}).get("kubernetes.io/service-name")
            != "kubernetes"
            or item.get("addressType") != "IPv4"
            or not valid_ports(item.get("ports", []))
        ):
            raise DeployError("Kubernetes API EndpointSlice identity or ports differ")
        objects.append(item)
        for endpoint in item.get("endpoints", []):
            conditions = endpoint.get("conditions", {})
            if (
                conditions.get("ready") is False
                or conditions.get("terminating") is True
            ):
                raise DeployError(
                    "Kubernetes API EndpointSlice is transitioning; retry discovery"
                )
            slice_ips.update(_ip(a) for a in endpoint.get("addresses", []))
    if not endpoint_ips or endpoint_ips != slice_ips or len(endpoint_ips) > 63:
        raise DeployError(
            "Kubernetes API Endpoints and EndpointSlices disagree or are empty"
        )
    cidrs = sorted({service_ip + "/32"} | {ip + "/32" for ip in endpoint_ips})
    return {
        "schema": "frontier.kubernetes-api-routes/v1",
        "cidrs": cidrs,
        "route_sha256": canonical_object_sha256(cidrs),
        "objects": [
            {
                "kind": o["kind"],
                "name": o["metadata"]["name"],
                "uid": o["metadata"]["uid"],
                "resource_version": o["metadata"]["resourceVersion"],
            }
            for o in objects
        ],
    }


def discover_api_routes(
    config: Mapping[str, Any], kubeconfig: Path, *, receipt_path: Path | None = None
) -> dict:
    environment = kubernetes_environment(kubeconfig)

    def read():
        values = []
        for kind, args in (
            ("service", ["kubernetes"]),
            ("endpoints", ["kubernetes"]),
            ("endpointslices", ["-l", "kubernetes.io/service-name=kubernetes"]),
        ):
            values.append(
                json.loads(
                    run_command(
                        [
                            executable("kubectl"),
                            "-n",
                            "default",
                            "get",
                            kind,
                            *args,
                            "-o",
                            "json",
                        ],
                        environment=environment,
                        capture=True,
                        timeout=60,
                    )
                )
            )
        return route_snapshot(config, *values)

    before, after = read(), read()
    if before != after:
        raise DeployError(
            "Kubernetes API destinations changed during discovery; retry before deployment"
        )
    after["observed_at"] = datetime.now(timezone.utc).isoformat()
    after["source_files_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("api_routes.py", "infra.py", "cli.py")
    }
    if receipt_path is not None:
        atomic_write(receipt_path, canonical_json(after), mode=0o600)
    return after


def api_policy_patch(
    policy: dict, cidrs: list[str], old_cidrs: list[str]
) -> tuple[list, dict]:
    """CAS only the existing API rule, retaining unrelated ingress and egress."""
    if (
        not cidrs
        or len(cidrs) > 64
        or any(not c.endswith("/32") or _ip(c[:-3]) + "/32" != c for c in cidrs)
    ):
        raise DeployError("Kubernetes API policy requires exact IPv4 /32 destinations")
    spec = json.loads(json.dumps(policy["spec"]))
    matches = []
    for i, rule in enumerate(spec.get("egress", [])):
        if (
            rule.get("ports") == [{"port": 443, "protocol": "TCP"}]
            and rule.get("to")
            and all(
                set(peer) == {"ipBlock"} and set(peer["ipBlock"]) == {"cidr"}
                for peer in rule["to"]
            )
            and sorted(peer["ipBlock"]["cidr"] for peer in rule["to"])
            in (sorted(old_cidrs), sorted(cidrs))
        ):
            matches.append(i)
    if len(matches) != 1:
        raise DeployError(
            "Existing platform API policy does not match its frozen CIDRs"
        )
    peers = [{"ipBlock": {"cidr": c}} for c in cidrs]
    index = matches[0]
    spec["egress"][index]["to"] = peers
    return [
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": policy["metadata"]["resourceVersion"],
        },
        {"op": "replace", "path": f"/spec/egress/{index}/to", "value": peers},
    ], spec


def verify_api_route_reader_rbac(kubeconfig: Path) -> None:
    environment = kubernetes_environment(kubeconfig)
    for expected in management_api_routes_rbac():
        observed = json.loads(
            run_command(
                [
                    executable("kubectl"),
                    "-n",
                    "default",
                    "get",
                    expected["kind"].lower(),
                    expected["metadata"]["name"],
                    "-o",
                    "json",
                ],
                environment=environment,
                capture=True,
                timeout=60,
            )
        )
        fields = ("rules",) if expected["kind"] == "Role" else ("roleRef", "subjects")
        if (
            observed.get("kind") != expected["kind"]
            or any(
                observed.get("metadata", {}).get(key) != value
                for key, value in expected["metadata"].items()
            )
            or any(observed.get(key) != expected[key] for key in fields)
        ):
            raise DeployError(
                "Management API route-reader RBAC differs from the exact read-only grant"
            )


def verify_api_routes(config: Mapping[str, Any], kubeconfig: Path) -> dict:
    snapshot = discover_api_routes(config, kubeconfig)
    verify_api_route_reader_rbac(kubeconfig)
    environment = kubernetes_environment(kubeconfig)

    def get(kind, name):
        return json.loads(
            run_command(
                [
                    executable("kubectl"),
                    "-n",
                    PLATFORM_NAMESPACE,
                    "get",
                    kind,
                    name,
                    "-o",
                    "json",
                ],
                environment=environment,
                capture=True,
                timeout=60,
            )
        )

    cm = get("configmap", "frontier-management-config")
    frozen = cm.get("data", {}).get("MANAGEMENT_KUBERNETES_API_CIDRS", "").split(",")
    if sorted(frozen) != snapshot["cidrs"]:
        raise DeployError(
            "Kubernetes API routes changed or are incomplete; idle upgrade-runtime required before new runs"
        )
    for name in (
        "frontier-management-boundary",
        "frontier-storage-controller-boundary",
    ):
        api_policy_patch(get("networkpolicy", name), snapshot["cidrs"], frozen)
    return snapshot
