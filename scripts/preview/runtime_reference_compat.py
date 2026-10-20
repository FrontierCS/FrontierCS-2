"""Narrow compatibility for the reviewed public-reference idle repair.

This is an exact, release-bound transition, not a repository or digest allowlist.
Other releases continue through the ordinary mirrored upgrade path.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

from .infra import DeployError, canonical_object_sha256

REVIEWED_COMMIT = "78905901bccc1d51301961cafc4fb638ec6d6c8b"
REVIEWED_DIGESTS = {
    "management": "sha256:433ae9f0ce2f04240554d27f86973bfc5af4aa8d3753572bcc5c7ea10d04bf95",
    "driver": "sha256:8613f2976870dabf40eb9b344e35a4b691cd14816b3b5f808ce3b77e86ab2442",
}
REVIEWED_REFERENCES = {
    name: f"docker.io/joyemang33/fcs2-{name}@{digest}"
    for name, digest in REVIEWED_DIGESTS.items()
}
REVIEWED_OLD_INIT = (
    "172.20.0.50:5000/frontier/management@sha256:"
    "b8e439ba2e1427776b0cd4b03529c40b07cf6dfadf73fae1003b5c9f218486c9"
)


def preserve_reviewed_public_images(
    config: Mapping[str, Any], configmap: Mapping[str, Any],
    deployment: Mapping[str, Any], images: dict[str, Any],
) -> bool:
    """Select the whole reviewed tuple, or reject an unreviewed same-pin tuple.

    ``images`` initially contains the validated runtime state records. Only after
    validating every input do we replace both records with the public pins.
    """
    containers = deployment["spec"]["template"]["spec"]["containers"]
    management = [c for c in containers if c.get("name") == "management"]
    if len(management) != 1:
        raise DeployError("runtime upgrade requires one Management container")
    data = configmap["data"]
    live = {
        "management": management[0]["image"],
        "driver": data.get("MANAGEMENT_DRIVER_IMAGE"),
        "key_worker": data.get("MANAGEMENT_LITELLM_KEY_WORKER_IMAGE"),
    }
    # Digest comparison identifies a same-pin candidate; it NEVER authorizes a
    # reference. Authorization below is full-string equality for the whole tuple.
    same_pin = all(
        isinstance(live[name], str)
        and live[name].split("@")[-1] == config["images"][name]["digest"]
        for name in ("management", "driver")
    )
    if not same_pin:
        return False
    mirrored = {
        name: f"{config['network']['registry_cluster_ip']}:5000/frontier/{name}@{config['images'][name]['digest']}"
        for name in ("management", "driver")
    }
    if (live["management"] == mirrored["management"]
            and live["driver"] == mirrored["driver"]
            and live["key_worker"] in (None, mirrored["management"])):
        return False
    expected = {**REVIEWED_REFERENCES, "key_worker": REVIEWED_REFERENCES["management"]}
    if live != expected or config["application"]["commit"] != REVIEWED_COMMIT:
        raise DeployError("unreviewed same-digest runtime references")
    for name in ("management", "driver"):
        source = config["images"][name]
        if (source["repository"] != f"docker.io/joyemang33/fcs2-{name}"
                or source["digest"] != REVIEWED_DIGESTS[name]
                or source["tag"] != "sha-" + REVIEWED_COMMIT
                or source["reference"] != REVIEWED_REFERENCES[name]
                or images[name]["digest"] != REVIEWED_DIGESTS[name]):
            raise DeployError("reviewed public runtime image pin or commit differs")
    if any(e.get("name") in {
        "MANAGEMENT_DRIVER_IMAGE", "MANAGEMENT_LITELLM_KEY_WORKER_IMAGE"
    } for e in management[0].get("env", [])):
        raise DeployError("reviewed public runtime rejects image environment overrides")
    for name in ("management", "driver"):
        source = config["images"][name]
        images[name] = {
            "tag": source["repository"] + ":" + source["tag"],
            "digest": source["digest"], "reference": source["reference"],
            "source_sha256": source["digest"],
        }
    return True


def reviewed_retention_transition(existing, expected, target_reference) -> bool:
    """The only historical initializer exception changes its image, nothing else."""
    return (
        expected.get("image") == REVIEWED_REFERENCES["management"]
        and target_reference == REVIEWED_REFERENCES["management"]
        and existing == {**expected, "image": REVIEWED_OLD_INIT}
    )


def require_reviewed_route_only_patches(configmap, deployment, data, patches):
    """Fail before applying an extra effective CM or Deployment change."""
    before = dict(configmap["data"])
    after = dict(data)
    before.pop("MANAGEMENT_KUBERNETES_API_CIDRS", None)
    after.pop("MANAGEMENT_KUBERNETES_API_CIDRS", None)
    if before != after:
        raise DeployError("reviewed public repair has extra ConfigMap changes")
    for patch in patches:
        if patch["op"] == "test":
            continue
        path = patch["path"]
        current = deployment
        for part in path.strip("/").split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            current = current[int(part)] if isinstance(current, list) else current.get(part)
            if current is None:
                break
        value = patch.get("value")
        if path == "/spec/template/metadata/annotations":
            expected = dict(current or {})
            expected["frontier.dev/config-sha256"] = canonical_object_sha256(data)
        elif path == "/spec/template/spec/initContainers":
            expected = copy.deepcopy(current)
            if not isinstance(expected, list):
                raise DeployError("reviewed public repair requires an existing initializer")
            for init in expected:
                if init.get("name") == "gateway-artifact-retention":
                    init["image"] = REVIEWED_REFERENCES["management"]
                    for key, default in (("terminationMessagePath", "/dev/termination-log"),
                                         ("terminationMessagePolicy", "File")):
                        if init.get(key) == default:
                            init.pop(key)
        else:
            expected = current
        if patch["op"] not in ("add", "replace") or value != expected:
            raise DeployError("reviewed public repair has extra Deployment changes: " + path)


def reviewed_public_runtime_records(config, records) -> bool:
    """Accept persisted public state only as the complete reviewed release pair."""
    if config["application"]["commit"] != REVIEWED_COMMIT:
        return False
    for name in ("management", "driver"):
        source = config["images"][name]
        repository = f"docker.io/joyemang33/fcs2-{name}"
        if (source["repository"] != repository
                or source["tag"] != "sha-" + REVIEWED_COMMIT
                or source["digest"] != REVIEWED_DIGESTS[name]
                or source["reference"] != REVIEWED_REFERENCES[name]
                or records.get(name) != {
                    "tag": repository + ":sha-" + REVIEWED_COMMIT,
                    "digest": REVIEWED_DIGESTS[name],
                    "reference": REVIEWED_REFERENCES[name],
                    "source_sha256": REVIEWED_DIGESTS[name],
                }):
            return False
    return True
