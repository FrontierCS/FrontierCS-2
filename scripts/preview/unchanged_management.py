"""Preserve a verified site-native Management image during a Driver-only upgrade."""
from __future__ import annotations

import copy
from .infra import DeployError, canonical_object_sha256, runtime_source_commit


def require_native_management_record(config, record):
    """Bind this alternate repository to the requested source and exact digest."""
    source = config["images"]["management"]
    commit = runtime_source_commit(config, "management")
    repository = f"{config['network']['registry_cluster_ip']}:5000/frontier/runtime/management"
    if (source["repository"] != "docker.io/joyemang33/fcs2-management"
            or source["tag"] != "sha-" + commit
            or source["reference"] != source["repository"] + "@" + source["digest"]
            or record != {"tag": repository + ":git-" + commit[:20],
                          "digest": source["digest"],
                          "reference": repository + "@" + source["digest"],
                          "source_sha256": source["digest"]}):
        raise DeployError("unchanged Management native record/source tuple differs")
    return repository.split('/', 1)[1]


def unchanged_native_management(config, configmap, deployment, images):
    """Return a repository to verify before reuse; never infer a new state record."""
    pod = deployment["spec"]["template"]["spec"]
    containers = [c for c in pod["containers"] if c.get("name") == "management"]
    if len(containers) != 1:
        raise DeployError("runtime upgrade requires one Management container")
    live = containers[0].get("image")
    expected = config["images"]["management"]["digest"]
    if not isinstance(live, str) or live.rsplit("@", 1)[-1] != expected:
        return None  # Management really changes: retain ordinary upgrade behavior.
    native_prefix = f"{config['network']['registry_cluster_ip']}:5000/frontier/runtime/management@"
    if not live.startswith(native_prefix):
        ordinary = native_prefix.replace("/frontier/runtime/", "/frontier/") + expected
        public = config["images"]["management"]["reference"]
        if live not in (ordinary, public):
            raise DeployError("unreviewed same-digest Management reference")
        return None  # Existing ordinary/public compatibility path remains responsible.
    repository = require_native_management_record(config, images["management"])
    if live != images["management"]["reference"]:
        raise DeployError("unchanged Management live image differs from state")
    if configmap["data"].get("MANAGEMENT_LITELLM_KEY_WORKER_IMAGE") != live:
        raise DeployError("unchanged Management key-worker image differs")
    if any(e.get("name") in {"MANAGEMENT_DRIVER_IMAGE", "MANAGEMENT_LITELLM_KEY_WORKER_IMAGE"}
           for e in containers[0].get("env", [])):
        raise DeployError("unchanged Management rejects image environment overrides")
    inits = [c for c in pod.get("initContainers", []) if c.get("name") == "gateway-artifact-retention"]
    if len(inits) != 1 or inits[0].get("image") != live:
        raise DeployError("unchanged Management retention initializer image differs")
    return repository


def require_driver_only_patches(configmap, deployment, data, patches):
    """Do not turn repository preservation into permission for unrelated changes."""
    expected = dict(configmap["data"])
    expected["MANAGEMENT_DRIVER_IMAGE"] = data["MANAGEMENT_DRIVER_IMAGE"]
    if expected != data:
        raise DeployError("unchanged Management has extra ConfigMap changes")
    for patch in patches:
        if patch["op"] == "test":
            continue
        current = deployment
        for part in patch["path"].strip('/').split('/'):
            part = part.replace('~1', '/').replace('~0', '~')
            current = current[int(part)] if isinstance(current, list) else current.get(part)
            if current is None:
                break
        expected = copy.deepcopy(current)
        if patch["path"] == "/spec/template/metadata/annotations":
            expected = dict(current or {})
            expected["frontier.dev/config-sha256"] = canonical_object_sha256(data)
        elif patch["path"] == "/spec/template/spec/initContainers" and isinstance(expected, list):
            for init in expected:
                if init.get("name") == "gateway-artifact-retention":
                    for key, default in (("terminationMessagePath", "/dev/termination-log"),
                                         ("terminationMessagePolicy", "File")):
                        if init.get(key) == default:
                            init.pop(key)
        if patch["op"] not in ("add", "replace") or patch.get("value") != expected:
            raise DeployError("unchanged Management has extra Deployment changes: " + patch["path"])


def require_unchanged_observer(expected, live):
    """Accept only the desired observer authority; ignore server-owned metadata."""
    expected_body = {k: v for k, v in expected.items() if k != "metadata"}
    live_body = {k: v for k, v in live.items() if k != "metadata"}
    metadata = live.get("metadata", {})
    if (live_body != expected_body
            or any(metadata.get(k) != v for k, v in expected["metadata"].items())):
        raise DeployError("unchanged Management Modal observer differs")
