#!/usr/bin/env python3
"""The production layout of a Verifier container: the one definition every local check uses.

A Verifier image is only checked if it runs the way the Driver runs it. Both "works locally,
breaks in production" bugs we shipped came from checks that used the image's own /frontier
(fast_kv placement, metaopt's /tmp -> /frontier symlink + rename: EXDEV only when /frontier is
a separate filesystem). This module states the layout once; the post-build smoke
(scripts/build_task_images.py run_in_pod) and the operator's k8s_verify.sh both run the Pod it
defines (`pod`) on the production Verifier nodes.

Source: FrontierCS core, driver/crates/driver-k8s/src/lib.rs `build_verifier_job` and
driver/crates/verifier-wrapper/src/main.rs (core main 3adac45, 2026-11-05):

- Pod securityContext: runAsNonRoot, runAsUser/runAsGroup/fsGroup 65532, seccomp RuntimeDefault,
  hostUsers false (a user namespace per Pod).
- Container securityContext: readOnlyRootFilesystem, allowPrivilegeEscalation false, all
  capabilities dropped.
- Mounts: /frontier is an emptyDir (empty: it hides whatever the image has at /frontier; its own
  filesystem, so a rename between it and the image root is EXDEV), sized by
  verifier.workspace_bytes (4 GiB when the task leaves it unset); /platform (read-only emptyDir) holds the wrapper. Nothing else is
  writable: /tmp is the read-only image root.
- Process: the container command is /platform/frontier-verifier-wrapper (it replaces the image
  entrypoint; working directory = the image WORKDIR). The wrapper downloads the payload to
  /frontier/input/payload.tar and the dataset to /frontier/input/dataset, creates
  /frontier/output, and runs the task's verifier.command with the Pod environment plus
  FRONTIER_PAYLOAD_PATH, FRONTIER_DATASET_PATH, FRONTIER_VERIFICATION_MODE, FRONTIER_RESULT_PATH
  and FRONTIER_EVIDENCE_PATH. No FRONTIER_VERIFIER_SCRATCH, TMPDIR or HOME is set (HOME comes
  from the image).
- Resources: limits cpu = verifier.cpu_millis, memory = verifier.memory_bytes.
- Network: the Driver's internal URL only (namespace default-deny); the smoke Pod has none.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

UID = GID = 65532
WORK = "/frontier"
PAYLOAD = f"{WORK}/input/payload.tar"
DATASET = f"{WORK}/input/dataset"
RESULT = f"{WORK}/output/result.json"
EVIDENCE = f"{WORK}/output/evidence.tar.zst"
# The placeholders a local run gives the Pod variables that name live platform objects.
SMOKE_IDENTITY = "production-layout-smoke"
# The Driver's Verifier workspace when task.verifier.workspace_bytes is unset.
DEFAULT_WORKSPACE_BYTES = 4 << 30
# Where the post-build Verifier smoke runs: the production Verifier nodes (eval16) in a
# namespace whose deny-all NetworkPolicy keeps the Pod offline.
NODE_SELECTOR = {"eks.amazonaws.com/nodegroup": "frontiercs2-preview-eval16"}
SMOKE_NAMESPACE = "task-format-verify"


def environment(verifier: dict, mode: str, image_digest: str = "") -> dict[str, str]:
    """The environment the task's verifier.command sees in production (besides the image ENV):
    the Driver's Pod variables and the five the wrapper adds. `verifier` is the task
    definition's `verifier` table."""
    if mode not in ("train", "eval"):
        raise ValueError("mode must be train or eval")
    return {
        "FRONTIER_RUN_ID": SMOKE_IDENTITY,
        "FRONTIER_ATTEMPT_ID": SMOKE_IDENTITY,
        "FRONTIER_TASK_CONFIG_DIGEST": SMOKE_IDENTITY,
        "FRONTIER_VERIFIER_IMAGE_DIGEST": image_digest or SMOKE_IDENTITY,
        "FRONTIER_VERIFICATION_ID": SMOKE_IDENTITY,
        "FRONTIER_VERIFICATION_MODE": mode,
        "FRONTIER_DRIVER_INTERNAL_URL": "http://frontier-driver.invalid:8081",
        "FRONTIER_VERIFIER_TOKEN": SMOKE_IDENTITY,
        "FRONTIER_VERIFIER_COMMAND_JSON": json.dumps(verifier["command"]),
        "FRONTIER_VERIFIER_RESULT_PATH": RESULT,
        "FRONTIER_VERIFIER_EVIDENCE_PATH": EVIDENCE,
        "FRONTIER_MAX_EVIDENCE_BYTES": str(verifier["max_evidence_bytes"]),
        "FRONTIER_DATASET_VERSION": SMOKE_IDENTITY,
        # Added by the wrapper for the task command.
        "FRONTIER_PAYLOAD_PATH": PAYLOAD,
        "FRONTIER_DATASET_PATH": DATASET,
        "FRONTIER_RESULT_PATH": RESULT,
        "FRONTIER_EVIDENCE_PATH": EVIDENCE,
    }


def stage_work(work: Path, payload: bytes, dataset: bytes) -> None:
    """Fill a fresh host directory the way the wrapper fills the empty /frontier emptyDir."""
    work.mkdir(parents=True, exist_ok=True)
    for directory in (work, work / "input", work / "output"):
        directory.mkdir(exist_ok=True)
        # The emptyDir is 0777; the wrapper (uid 65532) owns input/ and output/.
        directory.chmod(0o777)
    for path, body in ((work / "input/payload.tar", payload), (work / "input/dataset", dataset)):
        path.write_bytes(body)
        path.chmod(0o644)


def pod_manifest(
    name: str,
    image: str,
    verifier: dict,
    mode: str,
    *,
    node_selector: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
) -> dict:
    """A Pod that keeps the production Verifier layout but idles (`sleep`) so an operator can
    copy the inputs to /frontier/input and `kubectl exec` the task command: the
    Pod environment is the production one, so the exec'd command sees what the wrapper's child
    sees. The wrapper itself, its Driver traffic and the Job wrapper are left out."""
    workspace = verifier.get("workspace_bytes") or DEFAULT_WORKSPACE_BYTES
    security = {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]},
                "readOnlyRootFilesystem": True}
    spec = {
        "restartPolicy": "Never",
        "automountServiceAccountToken": False,
        "hostUsers": False,
        "securityContext": {"runAsNonRoot": True, "runAsUser": UID, "runAsGroup": GID,
                            "fsGroup": GID, "seccompProfile": {"type": "RuntimeDefault"}},
        "containers": [{
            "name": "verifier",
            "image": image,
            "command": ["sleep", "86400"],
            "env": [{"name": key, "value": value}
                    for key, value in environment(verifier, mode).items()],
            "resources": {
                "requests": {"cpu": f"{verifier['cpu_millis']}m",
                             "memory": str(verifier["memory_bytes"])},
                "limits": {"cpu": f"{verifier['cpu_millis']}m",
                           "memory": str(verifier["memory_bytes"])},
            },
            "securityContext": security,
            "volumeMounts": [{"name": "work", "mountPath": WORK}],
        }],
        "volumes": [{"name": "work", "emptyDir": {"sizeLimit": str(workspace)}}],
    }
    if node_selector:
        spec["nodeSelector"] = node_selector
    return {"apiVersion": "v1", "kind": "Pod",
            "metadata": {"name": name, "labels": labels or {}}, "spec": spec}


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    pod = sub.add_parser("pod", help="print a production-layout Verifier Pod manifest (JSON)")
    pod.add_argument("--name", required=True)
    pod.add_argument("--image", required=True)
    pod.add_argument("--mode", choices=("train", "eval"), required=True)
    pod.add_argument("--cpu-millis", type=int, required=True)
    pod.add_argument("--memory-bytes", type=int, required=True)
    pod.add_argument("--workspace-bytes", type=int)
    pod.add_argument("--max-evidence-bytes", type=int, default=64 << 20)
    pod.add_argument("--command", default='["/opt/frontier/verifier"]',
                     help="the task's verifier.command as JSON")
    pod.add_argument("--node-selector", action="append", default=[], metavar="KEY=VALUE",
                     help="default: the production Verifier nodes (NODE_SELECTOR)")
    pod.add_argument("--label", action="append", default=[], metavar="KEY=VALUE")
    options = parser.parse_args(arguments)
    verifier = {"command": json.loads(options.command), "cpu_millis": options.cpu_millis,
                "memory_bytes": options.memory_bytes,
                "max_evidence_bytes": options.max_evidence_bytes}
    if options.workspace_bytes:
        verifier["workspace_bytes"] = options.workspace_bytes

    def pairs(values):
        return dict(value.split("=", 1) for value in values)

    json.dump(pod_manifest(options.name, options.image, verifier, options.mode,
                           node_selector=pairs(options.node_selector) or NODE_SELECTOR,
                           labels=pairs(options.label)), sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
