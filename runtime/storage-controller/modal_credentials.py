"""Copy mounted Modal credentials only into matching trusted Run Drivers."""
from __future__ import annotations

import base64
import re
from pathlib import Path


def reconcile_modal_credentials(api, namespace: dict, secret_name: str,
                                credential_directory: Path) -> None:
    metadata = namespace.get("metadata", {})
    name = metadata.get("name", "")
    labels = metadata.get("labels", {})
    run_id, attempt_id = labels.get("frontier.dev/run-id"), labels.get("frontier.dev/attempt-id")
    if (not re.fullmatch(r"frontier-run-[0-9a-hjkmnp-tv-z]{26}", name) or not metadata.get("uid")
            or labels.get("frontier.dev/owner") != "management"
            or labels.get("frontier.dev/resource-kind") != "run-namespace"
            or not re.fullmatch(r"run_[0-9A-HJKMNP-TV-Z]{26}", run_id or "")
            or not re.fullmatch(r"attempt_[0-9A-HJKMNP-TV-Z]{26}", attempt_id or "")
            or metadata.get("deletionTimestamp")):
        return
    jobs = api("GET", f"/apis/batch/v1/namespaces/{name}/jobs")
    eligible = False
    for job in jobs.get("items", []):
        job_metadata = job.get("metadata", {})
        job_labels = job_metadata.get("labels", {})
        if (job_labels.get("frontier.dev/owner") != "management"
                or job_labels.get("frontier.dev/run-id") != run_id
                or job_labels.get("frontier.dev/attempt-id") != attempt_id
                or job_labels.get("frontier.dev/resource-kind") not in {"driver-job", "driver-cleanup-job"}
                or not job_metadata.get("uid")):
            continue
        template = job.get("spec", {}).get("template", {})
        pod = template.get("spec", {})
        if (pod.get("serviceAccountName") != "frontier-driver"
                or template.get("metadata", {}).get("labels", {}).get("frontier.dev/network-role") != "driver"):
            continue
        containers = pod.get("containers", [])
        if len(containers) != 1:
            continue
        container = containers[0]
        if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", container.get("image", "")):
            continue
        environment = container.get("env", [])
        env = {entry.get("name"): entry for entry in environment}
        if len(env) != len(environment):
            raise RuntimeError("Driver has duplicate environment variables")
        if env.get("FRONTIER_DRIVER_BACKENDS", {}).get("value") != "local-k8s":
            continue
        if not any(key in env for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")):
            continue
        for variable, key in (("MODAL_TOKEN_ID", "token-id"), ("MODAL_TOKEN_SECRET", "token-secret")):
            if env.get(variable) != {"name": variable, "valueFrom": {
                    "secretKeyRef": {"name": secret_name, "key": key}}}:
                raise RuntimeError("Driver Modal Secret reference does not match the site")
        eligible = True
    if not eligible:
        return
    data = {}
    for key in ("token-id", "token-secret"):
        value = (credential_directory / key).read_text().strip()
        if not value or "\n" in value or "\r" in value:
            raise RuntimeError("Modal credential file is empty or invalid")
        data[key] = base64.b64encode(value.encode()).decode()
    desired = {
        "apiVersion": "v1", "kind": "Secret", "type": "Opaque", "immutable": True,
        "metadata": {
            "name": secret_name, "namespace": name,
            "labels": {"frontier.dev/owner": "storage-controller",
                       "frontier.dev/run-id": run_id, "frontier.dev/attempt-id": attempt_id},
            "ownerReferences": [{"apiVersion": "v1", "kind": "Namespace",
                                 "name": name, "uid": metadata["uid"]}],
        },
        "data": data,
    }
    collection = f"/api/v1/namespaces/{name}/secrets"
    path = f"{collection}/{secret_name}"
    actual = api("GET", path, absent_ok=True)
    if actual is None:
        api("POST", collection, desired)
        actual = api("GET", path)
    if (actual.get("immutable") is not True or actual.get("type") != "Opaque"
            or actual.get("data") != data
            or actual.get("metadata", {}).get("labels") != desired["metadata"]["labels"]
            or actual.get("metadata", {}).get("ownerReferences") != desired["metadata"]["ownerReferences"]):
        raise RuntimeError("Existing Run Modal Secret has a different immutable identity")
