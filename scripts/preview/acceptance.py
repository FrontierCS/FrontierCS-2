"""Runtime preflight, smoke operations and acceptance checks."""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import secrets
import subprocess
import time
from datetime import datetime, timezone
from http.client import IncompleteRead
from pathlib import Path
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import request
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit

from render_task_definition import json_text

from .config import (
    read_and_validate,
    state_directory,
)
from .infra import (
    ADMISSION_RECEIPT_MAP,
    EVAL_URI,
    LITELLM_NAMESPACE,
    LITELLM_SECRET,
    LITELLM_SERVICE,
    LOGICAL_AGENT_BASE,
    LOGICAL_CANDIDATE_OVERLAY,
    LOGICAL_VERIFIER,
    MANAGEMENT_SERVICE,
    PLATFORM_NAMESPACE,
    REGISTRY_SERVICE,
    REPOSITORY_ROOT,
    TASK_CATALOG_PATH,
    TERMINAL_EVALUATIONS,
    TERMINAL_RUNS,
    TRAIN_URI,
    DeployError,
    ManagementHTTPError,
    api_json,
    atomic_write,
    canonical_json,
    canonical_object_sha256,
    dataset_seed_names,
    executable,
    kubernetes_environment,
    load_json,
    platform_outputs,
    port_forward,
    require_mapping,
    run_command,
    runtime_admission_preflight,
    sha256_bytes,
    stable_ulid,
    validate_litellm_secret_document,
)
from .infra import (
    api_json_with_headers as api_json_with_headers,
)
from .infra import (
    scheduler_capacity_preflight as scheduler_capacity_preflight,
)
from .publication import (
    RULE,
    candidate_version_document,
    ensure_version,
    load_baselines,
    load_manifest,
    load_task_definitions,
    load_task_selection,
    local_author_receipt,
    task_version_document,
    validate_baseline,
    validate_seed_receipt,
    validate_seed_task,
)


def print_admission_receipt(kubeconfig: Path) -> None:
    """Best-effort diagnostics; live overview headers determine admission."""
    try:
        encoded = run_command(
            [
                executable("kubectl"),
                "-n",
                PLATFORM_NAMESPACE,
                "get",
                "configmap",
                ADMISSION_RECEIPT_MAP,
                "--ignore-not-found",
                "-o",
                "json",
            ],
            environment=kubernetes_environment(kubeconfig),
            capture=True,
            timeout=120,
        )
    except (DeployError, OSError, subprocess.SubprocessError, UnicodeError):
        # Subprocess errors may contain receipt data; never print their payload.
        print("admission receipt: unreadable (kubectl failed)")
        return
    if not encoded.strip():
        print("admission receipt: absent")
        return
    try:
        receipt = json.loads(json.loads(encoded)["data"]["receipt.json"])
        if not isinstance(receipt, dict):
            raise ValueError("receipt must be an object")
    except (KeyError, TypeError, ValueError, RecursionError):
        print("admission receipt: unreadable (invalid receipt map or receipt.json)")
        return
    print(
        f"admission receipt: operation={receipt.get('operation')} status={receipt.get('status')} "
        f"error_code={receipt.get('error_code')} revision={receipt.get('receipt_revision')}"
    )
    if (receipt.get("operation"), receipt.get("status"), receipt.get("error_code")) == (
        "FENCE",
        "ERROR",
        "fence_epoch_or_state_conflict",
    ):
        print(
            "admission receipt: expected on a simulator-opened OPEN/1 site; the bootstrap FENCE leaves the gate unchanged"
        )


def smoke_documents(
    images: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_id = "candver_" + stable_ulid(
        images["candidate_overlay"]["digest"].encode()
    )
    task_seed = canonical_json(
        {
            "publication_generation": 2,
            "agent": images["agent_base"]["digest"],
            "verifier": images["verifier"]["digest"],
            "train": sha256_bytes(
                (REPOSITORY_ROOT / "smoke/speed-add/datasets/train.json").read_bytes()
            ),
            "eval": sha256_bytes(
                (REPOSITORY_ROOT / "smoke/speed-add/datasets/eval.json").read_bytes()
            ),
        }
    )
    task_id = "taskver_" + stable_ulid(task_seed)
    platform = {"os": "linux", "architecture": "amd64"}
    candidate_config = {
        "schema_version": "frontier.candidate-config/v1",
        "candidate_version_id": candidate_id,
        "overlay_image": {
            "reference": LOGICAL_CANDIDATE_OVERLAY,
            "digest": images["candidate_overlay"]["digest"],
            "format": "frontier.oci-overlay/v1",
            "platform": platform,
        },
        "execution": {
            "entrypoint": ["/opt/frontier/candidate/local-e2e-start"],
            "args": [],
            "working_dir": "/workspace",
            "user": "1000:1000",
            "env": {},
        },
        "model_api": {"allowed_domains": [], "secret_refs": []},
    }
    train_sha = sha256_bytes(
        (REPOSITORY_ROOT / "smoke/speed-add/datasets/train.json").read_bytes()
    )
    eval_sha = sha256_bytes(
        (REPOSITORY_ROOT / "smoke/speed-add/datasets/eval.json").read_bytes()
    )
    task_config = {
        "schema_version": "frontier.task-config/v1",
        "task_version_id": task_id,
        "images": {
            "agent_cpu_base": {
                "reference": LOGICAL_AGENT_BASE,
                "digest": images["agent_base"]["digest"],
                "platform": platform,
            },
            "verifier": {
                "reference": LOGICAL_VERIFIER,
                "digest": images["verifier"]["digest"],
            },
        },
        "agent": {
            "cpu_millis": 300,
            "memory_bytes": 536870912,
            "workspace_bytes": 268435456,
        },
        "quota": {
            "wall_time_seconds": 600,
            "agent_gpu": {
                "allowed_types": [],
                "max_instances": 0,
                "max_gpu_seconds": 0,
                "max_instance_seconds": 0,
            },
            "verifier": {
                "max_train_calls": 1,
                "max_train_wall_seconds": 180,
                "max_gpu_instances": 0,
                "max_gpu_seconds": 0,
                "eval_timeout_seconds": 180,
            },
        },
        "submission": {
            "max_bytes": 8388608,
            "media_type": "application/x-tar",
            "required_sha256": True,
            "entrypoint": "payload/run.sh",
        },
        "datasets": {
            "train": {"uri": TRAIN_URI, "version": train_sha},
            "eval": {"uri": EVAL_URI, "version": eval_sha},
        },
        "verifier": {
            "command": ["/opt/frontier/speed-add-verifier"],
            "result_path": "/frontier/output/result.json",
            "evidence_path": "/frontier/output/evidence.tar.zst",
            "cpu_millis": 200,
            "memory_bytes": 268435456,
            "max_evidence_bytes": 8388608,
        },
        "network": {
            "allowed_domains": [],
            "audit": {
                "max_model_sessions": 256,
                "max_ssh_sessions": 32,
                "max_model_request_bytes": 1048576,
                "max_model_response_bytes": 8388608,
                "max_ssh_bytes_per_direction": 8388608,
                "max_bundle_bytes": 67108864,
            },
        },
    }
    candidate = {
        "candidate_id": "speed-add-scripted-preview",
        "display_name": "Speed Add scripted Preview",
        "version": candidate_id.removeprefix("candver_"),
        "definition_ref": "preview://speed-add/scripted-candidate",
        "definition_digest": canonical_object_sha256(candidate_config),
        "driver_config": candidate_config,
        "metadata": {"model_execution": "scripted", "purpose": "deployment-smoke"},
    }
    task = {
        "task_id": "speed-add-preview",
        "display_name": "Speed Add Preview",
        "family": "cpu",
        "version": task_id.removeprefix("taskver_"),
        "definition_ref": "preview://speed-add/task",
        "definition_digest": canonical_object_sha256(task_config),
        "driver_config": task_config,
        "metadata": {"fixture": "speed-add", "model_execution": "scripted"},
    }
    task["definition_ref"] = (
        "s3://frontier-local-task-data/definitions/speed-add/"
        + task["definition_digest"].removeprefix("sha256:")
        + ".json"
    )
    sizes = {
        sha256_bytes(body).removeprefix("sha256:"): len(body)
        for body in (
            (REPOSITORY_ROOT / f"smoke/speed-add/datasets/{mode}.json").read_bytes()
            for mode in ("train", "eval")
        )
    }
    task["author_receipt"] = local_author_receipt(
        task, sizes, {"fixture": "speed-add", "images": images}
    )
    return candidate, task


def smoke_timestamp(value: Any, field: str) -> float:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone required")
        return parsed.timestamp()
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise DeployError(f"smoke {field} must be a timestamp with timezone") from error


def smoke_operation_key(config: Mapping[str, Any], operation: Mapping[str, Any]) -> str:
    stem = (
        f"deployment-smoke-{config['site']['name']}-{operation['candidate_version_id'][-8:].lower()}-"
        f"{operation['task_version_id'][-8:].lower()}-{operation['nonce']}"
    )
    return stem if operation["retry"] == 0 else f"{stem}-retry-{operation['retry']}"


def prepare_smoke_operation(
    config: Mapping[str, Any],
    base_url: str,
    images: Mapping[str, Mapping[str, str]],
    candidate_version_id: str,
    task_version_id: str,
) -> dict[str, Any]:
    path = state_directory(config) / "smoke-operation.json"
    runtime = {
        "management_digest": images["management"]["digest"],
        "driver_digest": images["driver"]["digest"],
    }
    if path.exists():
        operation = require_mapping(load_json(path), "smoke operation")
        if operation.get("schema_version") != "frontier.deployment-smoke-operation/v1":
            raise DeployError("unsupported smoke operation schema")
        if (
            operation.get("runtime") == runtime
            and operation.get("candidate_version_id") == candidate_version_id
            and operation.get("task_version_id") == task_version_id
        ):
            if (
                not isinstance(operation.get("nonce"), str)
                or not re.fullmatch(r"[0-9a-f]{32}", operation["nonce"])
                or type(operation.get("retry")) is not int
                or not 0 <= operation["retry"] <= 3
            ):
                raise DeployError("invalid smoke operation nonce or retry")
            smoke_timestamp(operation.get("started_at"), "operation started_at")
            if operation.get("idempotency_key") != smoke_operation_key(
                config, operation
            ):
                raise DeployError(
                    "smoke operation idempotency key does not match this site/request"
                )
            # No ID means a crash may have happened before or after POST. Retry
            # the persisted key/body so either case resolves to one Evaluation.
            evaluation_id = operation.get("evaluation_id")
            if not evaluation_id:
                return operation
            try:
                evaluation = api_json(
                    base_url, "GET", f"/api/v1/evaluations/{evaluation_id}"
                )
            except ManagementHTTPError as error:
                if error.status != 404:
                    raise
            else:
                if evaluation.get("status") not in TERMINAL_EVALUATIONS:
                    return operation
    operation = {
        "schema_version": "frontier.deployment-smoke-operation/v1",
        "nonce": secrets.token_hex(16),
        "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "candidate_version_id": candidate_version_id,
        "task_version_id": task_version_id,
        "runtime": runtime,
        "retry": 0,
    }
    operation["idempotency_key"] = smoke_operation_key(config, operation)
    atomic_write(path, canonical_json(operation), mode=0o600)
    return operation


def check_smoke_replay(
    operation: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> None:
    created_at = smoke_timestamp(evaluation.get("created_at"), "Evaluation created_at")
    started_at = smoke_timestamp(operation["started_at"], "operation started_at")
    if created_at < started_at - 60:
        raise DeployError(
            f"smoke evaluation {evaluation.get('id')} is an idempotent replay created "
            f"{evaluation.get('created_at')}; rotate the smoke operation"
        )


def run_smoke(
    config: Mapping[str, Any], kubeconfig: Path, images: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    with port_forward(kubeconfig, MANAGEMENT_SERVICE, 8080) as port:
        base = f"http://127.0.0.1:{port}"
        candidate, task = smoke_documents(images)
        created_candidate = ensure_version(
            base,
            "/api/v1/candidate-versions",
            candidate["driver_config"]["candidate_version_id"],
            candidate,
        )
        created_task = ensure_version(
            base,
            "/api/v1/task-versions",
            task["driver_config"]["task_version_id"],
            task,
        )
        operation = prepare_smoke_operation(
            config, base, images, created_candidate["id"], created_task["id"]
        )
        operation_path = state_directory(config) / "smoke-operation.json"
        body = {
            "candidate_version_id": created_candidate["id"],
            "task_version_ids": [created_task["id"]],
            "max_parallel_runs": 1,
            "priority": 0,
            "labels": {
                "environment": config["site"]["name"],
                "fixture": "speed-add",
                "model_execution": "scripted",
                "smoke_nonce": operation["nonce"],
            },
        }
        # Persist each retry before POST, and its returned identity before
        # polling. The existing bounded FAILED/CANCELLED retry policy remains.
        evaluation_id = run_id = None
        for attempt in range(operation["retry"], 4):
            if attempt != operation["retry"]:
                operation["retry"] = attempt
                operation["idempotency_key"] = smoke_operation_key(config, operation)
                operation.pop("evaluation_id", None)
                operation.pop("evaluation_created_at", None)
                atomic_write(operation_path, canonical_json(operation), mode=0o600)
            evaluation = api_json(
                base,
                "POST",
                "/api/v1/evaluations",
                body,
                headers={"Idempotency-Key": operation["idempotency_key"]},
            )
            items = evaluation.get("items", []) if isinstance(evaluation, dict) else []
            if len(items) != 1 or not isinstance(items[0].get("run_id"), str):
                raise DeployError("Management did not create exactly one smoke Run")
            evaluation_id, run_id = evaluation["id"], items[0]["run_id"]
            operation.update(
                evaluation_id=evaluation_id,
                evaluation_created_at=evaluation.get("created_at"),
            )
            atomic_write(operation_path, canonical_json(operation), mode=0o600)
            check_smoke_replay(operation, evaluation)
            run = api_json(base, "GET", f"/api/v1/runs/{run_id}")
            if run.get("state") not in {"FAILED", "CANCELLED"}:
                break
        deadline = time.monotonic() + config["execution"]["smoke"]["timeout_seconds"]
        while time.monotonic() < deadline:
            evaluation = api_json(base, "GET", f"/api/v1/evaluations/{evaluation_id}")
            run = api_json(base, "GET", f"/api/v1/runs/{run_id}")
            print(
                f"smoke evaluation={evaluation.get('status')} run={run.get('state')} stage={run.get('current_stage')}",
                flush=True,
            )
            if (
                evaluation.get("status") in TERMINAL_EVALUATIONS
                and run.get("state") in TERMINAL_RUNS
            ):
                break
            time.sleep(2)
        else:
            raise DeployError(
                "smoke Evaluation did not reach terminal state before timeout"
            )
        if (
            evaluation.get("status") != "COMPLETED"
            or run.get("state") != "COMPLETED"
            or not isinstance(run.get("score"), (int, float))
        ):
            raise DeployError(
                f"smoke Run failed closed: evaluation={evaluation.get('status')} run={run.get('state')}"
            )
        receipt = {
            "schema_version": "frontier.deployment-smoke/v1",
            "site": config["site"]["name"],
            "evaluation_id": evaluation_id,
            "run_id": run_id,
            "raw_score": run["score"],
            "model_execution": "scripted",
            "completed_at_unix": int(time.time()),
            "runtime_images": images,
        }
        receipt.update(
            nonce=operation["nonce"],
            evaluation_created_at=operation["evaluation_created_at"],
            runtime=operation["runtime"],
        )
        atomic_write(
            state_directory(config) / "smoke.json", canonical_json(receipt), mode=0o600
        )
        print(
            f"Executable Preview smoke: PASS (evaluation={evaluation_id}, run={run_id}, score={run['score']})"
        )
        return receipt


def gpu_smoke_candidate(
    candidate: Mapping[str, Any], mirrored_overlay: Mapping[str, str], patch: bytes
) -> dict[str, Any]:
    if not patch or len(patch) > 262144:
        raise DeployError(
            "GPU smoke requires a nonempty native patch no larger than 262144 bytes"
        )
    script = "\n".join(
        [
            "import base64, json, subprocess",
            "from pathlib import Path",
            "def run(args): subprocess.run(args, check=True)",
            f"Path('/workspace/replay.patch').write_bytes(base64.b64decode({base64.b64encode(patch).decode()!r}))",
            "run(['/frontier/task/prepare_workspace.sh'])",
            "run(['git', '-C', '/workspace/repo', 'apply', '/workspace/replay.patch'])",
            "run(['git', '-C', '/workspace/repo', 'add', '--intent-to-add', '.'])",
            "run(['/frontier/task/frontier-gpu', 'build'])",
            "run(['/frontier/task/make_submission.sh'])",
            "run(['tar', '--sort=name', '--mtime=@0', '--owner=0', '--group=0', '-C', '/workspace', '-cf', '/workspace/replay.tar', 'payload'])",
            "train = subprocess.check_output(['/opt/frontier/candidate/bin/frontier-submit', 'train', '/workspace/replay.tar'])",
            "receipt = json.loads(train)",
            "assert receipt['state'] == 'passed', 'train did not pass'",
            "Path('/workspace/gpu-smoke-train.json').write_bytes(train)",
            "print(train.decode(), flush=True)",
            "run(['/opt/frontier/candidate/bin/frontier-submit', 'final', '/workspace/replay.tar'])",
        ]
    )
    document = candidate_version_document(candidate, mirrored_overlay)
    driver = document["driver_config"]
    driver["execution"]["entrypoint"] = ["/usr/bin/python3", "-c"]
    driver["execution"]["args"] = [script]
    driver["execution"]["env"]["FRONTIER_GPU_TYPE"] = "H100"
    driver["execution"]["env"]["FRONTIER_GPU_SECONDS"] = "3600"
    driver["candidate_version_id"] = "candver_" + stable_ulid(canonical_json(driver))
    digest = canonical_object_sha256(driver)
    document.update(
        {
            "candidate_id": "sparse-video-replay-preview",
            "display_name": "Sparse Video infrastructure replay",
            "version": digest.removeprefix("sha256:")[:24],
            "definition_digest": digest,
            "definition_ref": "preview://sparse-video-replay/" + digest,
            "metadata": {
                "model_execution": "archived-submission-replay",
                "purpose": "gpu-deployment-acceptance",
                "patch_sha256": sha256_bytes(patch),
            },
        }
    )
    return document


def run_gpu_smoke(
    config: Mapping[str, Any],
    kubeconfig: Path,
    patch_path: Path,
    patch_sha256: str,
    attempt: str,
) -> dict[str, Any]:
    if not config["features"]["modal_gpu"] or not re.fullmatch(
        r"[a-z0-9-]{1,40}", attempt
    ):
        raise DeployError(
            "GPU smoke requires Modal enabled and a bounded lowercase attempt label"
        )
    patch = patch_path.read_bytes()
    if sha256_bytes(patch) != patch_sha256:
        raise DeployError("GPU smoke patch differs from the operator-reviewed SHA-256")
    catalog, selection = load_task_selection(config, require_resolved=True)
    seed = load_json(state_directory(config) / "task-seed-receipt.json")
    validate_seed_receipt(seed, selection, config["site"]["name"])
    task = next(
        (
            item
            for item in catalog["tasks"]
            if item["name"] == "sparse_video_attention_policy"
        ),
        None,
    )
    if task is None:
        raise DeployError("Sparse video is absent from the fixed catalog")
    endpoint = f"{config['network']['registry_cluster_ip']}:5000"
    mirrored = {
        name: {
            "reference": f"{endpoint}/frontier/tasks/sparse-video-attention-policy/{name.removesuffix('_image')}@{task[name]['digest']}",
            "digest": task[name]["digest"],
        }
        for name in ("agent_image", "verifier_image")
    }
    task_document = task_version_document(
        task,
        mirrored,
        model_domains=selection.model_domains,
        definitions=selection.definitions,
    )
    task_id = task_document["driver_config"]["task_version_id"]
    validate_seed_task(seed, task["name"], task_document)
    overlay = catalog["candidate"]["overlay_image"]
    candidate = gpu_smoke_candidate(
        catalog["candidate"],
        {
            "reference": f"{endpoint}/frontier/candidates/codex@{overlay['digest']}",
            "digest": overlay["digest"],
        },
        patch,
    )
    receipt_path = state_directory(config) / f"gpu-smoke-{attempt}.json"
    if receipt_path.exists():
        previous = load_json(receipt_path)
        if (
            previous.get("candidate_version_id")
            != candidate["driver_config"]["candidate_version_id"]
            or previous.get("task_version_id") != task_id
            or previous.get("patch_sha256") != patch_sha256
        ):
            raise DeployError(
                "GPU attempt label already binds different inputs; use a new attempt label"
            )
    with port_forward(kubeconfig, MANAGEMENT_SERVICE, 8080) as port:
        base = f"http://127.0.0.1:{port}"
        published_task = api_json(base, "GET", f"/api/v1/task-versions/{task_id}")
        if (
            published_task.get("definition_digest")
            != task_document["definition_digest"]
        ):
            raise DeployError("GPU task definition changed")
        created = ensure_version(
            base,
            "/api/v1/candidate-versions",
            candidate["driver_config"]["candidate_version_id"],
            candidate,
        )
        evaluation = api_json(
            base,
            "POST",
            "/api/v1/evaluations",
            {
                "candidate_version_id": created["id"],
                "task_version_ids": [task_id],
                "max_parallel_runs": 1,
                "priority": 0,
                "labels": {
                    "environment": config["site"]["name"],
                    "purpose": "gpu-deployment-acceptance",
                    "model_execution": "archived-submission-replay",
                },
            },
            headers={
                "Idempotency-Key": f"preview-gpu-{attempt}-{created['id'][-8:].lower()}-{task_id[-8:].lower()}"
            },
        )
        if len(evaluation.get("items", [])) != 1:
            raise DeployError("GPU acceptance did not create exactly one Run")
        run_id = evaluation["items"][0]["run_id"]
        receipt = {
            "schema_version": "frontier.gpu-deployment-smoke/v1",
            "site": config["site"]["name"],
            "evaluation_id": evaluation["id"],
            "run_id": run_id,
            "task_version_id": task_id,
            "candidate_version_id": created["id"],
            "patch_sha256": patch_sha256,
            "model_execution": "archived-submission-replay",
            "status": "RUNNING",
        }
        atomic_write(receipt_path, canonical_json(receipt), mode=0o600)
        quota = task_document["driver_config"]["quota"]
        deadline = (
            time.monotonic()
            + quota["wall_time_seconds"]
            + quota["verifier"]["eval_timeout_seconds"]
            + 600
        )
        while time.monotonic() < deadline:
            run = api_json(base, "GET", f"/api/v1/runs/{run_id}")
            print(
                f"GPU acceptance run={run_id} state={run.get('state')} stage={run.get('current_stage')}",
                flush=True,
            )
            if run.get("state") in TERMINAL_RUNS:
                break
            time.sleep(15)
        else:
            raise DeployError(
                f"GPU Run {run_id} exceeded its outer observation deadline; inspect cleanup before retry"
            )
        evaluation = api_json(base, "GET", f"/api/v1/evaluations/{evaluation['id']}")
        receipt.update(
            {
                "status": run.get("state"),
                "raw_score": run.get("score"),
                "evaluation_status": evaluation.get("status"),
            }
        )
        atomic_write(receipt_path, canonical_json(receipt), mode=0o600)
        if (
            run.get("state") != "COMPLETED"
            or evaluation.get("status") != "COMPLETED"
            or type(run.get("score")) not in (int, float)
            or not math.isfinite(run["score"])
        ):
            raise DeployError(
                f"GPU acceptance failed closed: {run_id}; inspect Run evidence before retry"
            )
        print(
            f"GPU scoring completed: {run_id}, score={run['score']}; artifact and exact cleanup checks remain required"
        )
        return receipt


def verify_litellm(config: Mapping[str, Any], kubeconfig: Path) -> None:
    environment, kubectl = kubernetes_environment(kubeconfig), executable("kubectl")
    deployment = json.loads(
        run_command(
            [
                kubectl,
                "-n",
                LITELLM_NAMESPACE,
                "get",
                "deployment",
                LITELLM_SERVICE,
                "-o",
                "json",
            ],
            environment=environment,
            capture=True,
            timeout=120,
        )
    )
    if deployment.get("status", {}).get("availableReplicas") != 1:
        raise DeployError("LiteLLM must have exactly one available gateway replica")
    postgres = json.loads(
        run_command(
            [
                kubectl,
                "-n",
                LITELLM_NAMESPACE,
                "get",
                "statefulset",
                "frontier-litellm-postgres",
                "-o",
                "json",
            ],
            environment=environment,
            capture=True,
            timeout=120,
        )
    )
    if postgres.get("status", {}).get("readyReplicas") != 1:
        raise DeployError("LiteLLM PostgreSQL must have exactly one ready replica")
    service = json.loads(
        run_command(
            [
                kubectl,
                "-n",
                LITELLM_NAMESPACE,
                "get",
                "service",
                LITELLM_SERVICE,
                "-o",
                "json",
            ],
            environment=environment,
            capture=True,
            timeout=120,
        )
    )
    if (
        service.get("spec", {}).get("clusterIP")
        != config["network"]["litellm_cluster_ip"]
    ):
        raise DeployError("LiteLLM Service does not retain the frozen site ClusterIP")
    observed_secret = run_command(
        [
            kubectl,
            "-n",
            LITELLM_NAMESPACE,
            "get",
            "secret",
            LITELLM_SECRET,
            "-o",
            "name",
        ],
        environment=environment,
        capture=True,
        timeout=120,
    ).strip()
    if observed_secret != f"secret/{LITELLM_SECRET}":
        raise DeployError("LiteLLM platform Secret is absent")
    with port_forward(
        kubeconfig, LITELLM_SERVICE, 4000, namespace=LITELLM_NAMESPACE
    ) as port:
        with urllib_request.urlopen(
            f"http://127.0.0.1:{port}/health/readiness", timeout=15
        ) as response:
            health = response.read().decode(errors="replace").strip()
        if not health:
            raise DeployError("LiteLLM readiness endpoint returned an empty response")
        try:
            with urllib_request.urlopen(
                f"http://127.0.0.1:{port}/ui/", timeout=15
            ) as response:
                body = response.read(1024 * 1024)
        except (urllib_error.URLError, TimeoutError) as error:
            raise DeployError("LiteLLM admin UI endpoint is unavailable") from error
        if response.status != 200 or not body.lstrip()[:15].lower().startswith(
            b"<!doctype html"
        ):
            raise DeployError("LiteLLM admin UI did not return its login page")
    secret_path = state_directory(config) / "litellm-secrets.json"
    admin_path = state_directory(config) / "litellm-admin.env"
    for path in (secret_path, admin_path):
        if (
            path.is_symlink()
            or not path.is_file()
            or (path.stat().st_mode & 0o777) != 0o600
        ):
            raise DeployError(
                f"LiteLLM local recovery state must be a regular mode-0600 file: {path}"
            )
    secret_document = validate_litellm_secret_document(load_json(secret_path))
    expected_admin = f"UI_USERNAME=admin\nUI_PASSWORD={secret_document['stringData']['LITELLM_MASTER_KEY']}\n".encode()
    if admin_path.read_bytes() != expected_admin:
        raise DeployError(
            "LiteLLM local admin recovery state conflicts with its Secret"
        )


def verify_runtime(
    config: Mapping[str, Any],
    outputs: Mapping[str, Any] | None = None,
    kubeconfig: Path | None = None,
    *,
    require_smoke: bool = True,
) -> None:
    load_task_selection(config)
    if outputs is None:
        outputs = platform_outputs(config)
    if kubeconfig is None:
        kubeconfig = state_directory(config) / "kubeconfig"
    if not kubeconfig.is_file():
        raise DeployError(f"site kubeconfig is absent: {kubeconfig}")
    environment, kubectl = kubernetes_environment(kubeconfig), executable("kubectl")
    for deployment in (
        REGISTRY_SERVICE,
        "frontier-storage-controller",
        "frontier-management",
    ):
        encoded = run_command(
            [
                kubectl,
                "-n",
                PLATFORM_NAMESPACE,
                "get",
                "deployment",
                deployment,
                "-o",
                "json",
            ],
            environment=environment,
            capture=True,
            timeout=120,
        )
        observed = json.loads(encoded)
        if observed.get("status", {}).get("availableReplicas") != 1:
            raise DeployError(f"{deployment} must have exactly one available replica")
    # The Management NetworkPolicy does not admit apiserver-proxy sources, so
    # health is probed the same way every other check reaches the service.
    with port_forward(kubeconfig, MANAGEMENT_SERVICE, 8080) as port:
        with urllib_request.urlopen(
            f"http://127.0.0.1:{port}/healthz", timeout=15
        ) as response:
            health = response.read().decode(errors="replace").strip()
        print_admission_receipt(kubeconfig)
        runtime_admission_preflight(config, f"http://127.0.0.1:{port}")
    if not health:
        raise DeployError("Management /healthz returned an empty response")
    verify_litellm(config, kubeconfig)
    _, seed_job_name = dataset_seed_names()
    seed = json.loads(
        run_command(
            [
                kubectl,
                "-n",
                PLATFORM_NAMESPACE,
                "get",
                "job",
                seed_job_name,
                "-o",
                "json",
            ],
            environment=environment,
            capture=True,
            timeout=120,
        )
    )
    if seed.get("status", {}).get("succeeded", 0) < 1:
        raise DeployError("speed-add dataset seed Job has not succeeded")
    if require_smoke:
        receipt_path = state_directory(config) / "smoke.json"
        receipt = load_json(receipt_path)
        if (
            receipt.get("site") != config["site"]["name"]
            or receipt.get("model_execution") != "scripted"
        ):
            raise DeployError("smoke receipt does not match this site")
    print("Infrastructure readiness: PASS")
    print("Management readiness: PASS")
    print(
        "CPU scripted Run readiness: PASS"
        if require_smoke
        else "CPU scripted Run readiness: NOT CHECKED"
    )
    print("LiteLLM gateway/UI readiness: PASS")
    print(
        "LiteLLM provider/model readiness: OPERATOR CONFIGURATION REQUIRED IN ADMIN UI"
    )
    print("LiteLLM model-enabled Run readiness: NOT IN LOCAL-K8S PROFILE")
    print(
        "Modal GPU: "
        + (
            "configured; live GPU acceptance is separate"
            if config["features"]["modal_gpu"]
            else "DISABLED"
        )
    )
    print(
        f"Access: KUBECONFIG={kubeconfig} kubectl -n {PLATFORM_NAMESPACE} port-forward service/{MANAGEMENT_SERVICE} 8080:8080"
    )
    print(
        f"LiteLLM UI: KUBECONFIG={kubeconfig} kubectl -n {LITELLM_NAMESPACE} port-forward service/{LITELLM_SERVICE} 4000:4000"
    )


# Reference execution and resumable parity acceptance.
# Agent Pod readiness precedes the Driver's RUNNING transition. A cached image
# can start the frozen launcher before mutations are allowed; gate only startup
# in CandidateConfig so existing reference image and payload identities survive.
REFERENCE_READINESS_GUARD = r"""import http.client
import os
import ssl
import sys
import time
from urllib.parse import urlsplit

def fail(message):
    print("[frontier-reference] " + message, file=sys.stderr, flush=True)
    raise SystemExit(1)

try:
    base = urlsplit(os.environ.get("FRONTIER_DRIVER_API_URL", "").strip())
    if (base.scheme not in {"http", "https"} or not base.hostname
            or base.username or base.password or base.query or base.fragment):
        raise ValueError
    host, port = base.hostname, base.port
except ValueError:
    fail("Driver readiness URL is missing or invalid")
if len(sys.argv) < 2:
    fail("Driver readiness guard requires a launcher")

connection_type = http.client.HTTPSConnection if base.scheme == "https" else http.client.HTTPConnection
reported_wait = False
while True:
    connection = None
    status = None
    try:
        connection = connection_type(host, port, timeout=5)
        connection.request("GET", base.path.rstrip("/") + "/readyz")
        with connection.getresponse() as response:
            status = response.status
    except ssl.SSLCertVerificationError:
        fail("Driver readiness TLS validation failed")
    except (OSError, http.client.RemoteDisconnected):
        pass
    except http.client.HTTPException:
        fail("Driver readiness response is invalid")
    except (ValueError, UnicodeError):
        fail("Driver readiness request is invalid")
    finally:
        if connection is not None:
            connection.close()
    if status == 204:
        print("[frontier-reference] Driver ready (204); starting launcher", flush=True)
        break
    if status is not None and status != 503:
        fail("Driver readiness returned unexpected HTTP " + str(status))
    if status == 503 and not reported_wait:
        print("[frontier-reference] Driver not ready (503); waiting", flush=True)
        reported_wait = True
    # The Driver's authored Run budget owns the total deadline, including startup.
    time.sleep(1)
try:
    os.execv(sys.argv[1], sys.argv[1:])
except OSError:
    fail("Reference launcher could not start")
"""
REFERENCE_EXECUTION = {
    "entrypoint": [
        "python3",
        "-c",
        REFERENCE_READINESS_GUARD,
        "/opt/frontier/reference-candidate",
    ],
    "args": [],
    "working_dir": "/workspace",
    "user": "10001:10001",
    "env": {},
}


TERMINAL = {"PASS", "FAIL", "TIMEOUT", "NOT_COMPARABLE"}
CLASSIFICATION_KEYS = ("provenance", "comparable", "comparability_note")
ERROR_HISTORY_LIMIT = 10
CANCEL_ATTEMPTS = 3
RESPONSE_LIMIT = 4 * 1024 * 1024
EXECUTION = REFERENCE_EXECUTION


def candidate_document(name: str, image: dict, mirror: dict) -> dict:
    if (
        mirror.get("source_sha256") != image["digest"]
        or mirror.get("source_reference") != image["repository"] + "@" + image["digest"]
    ):
        raise ValueError(f"reference mirror does not bind the public image: {name}")
    return candidate_version_document(
        {"name": name, "overlay_image": image},
        mirror,
        execution=EXECUTION,
        identity={
            "candidate_id": "reference-" + name.replace("_", "-"),
            "display_name": "Reference: " + name,
            "version": image["tag"],
            "metadata": {
                "purpose": "reference-parity",
                "launcher_sha256": image["launcher_sha256"],
                "payload_sha256": image["payload_sha256"],
            },
        },
    )


def successful_train(events: list, run_id: str) -> str | None:
    for event in events:
        payload = event.get("payload", {})
        if (
            event.get("run_id") == run_id
            and event.get("source") == "DRIVER"
            and event.get("kind") == "VERIFIER_TRAIN_COMPLETED"
            and payload.get("status") == "success"
            and isinstance(payload.get("verification_id"), str)
        ):
            return payload["verification_id"]
    return None


def verdict(
    evaluation: dict, run: dict, baseline: dict, train_id: str | None
) -> list[str]:
    failures = []
    if evaluation.get("status") != "COMPLETED":
        failures.append("Evaluation did not complete")
    if run.get("state") != "COMPLETED" or run.get("outcome") != "SUCCEEDED":
        failures.append("Run did not complete with SUCCEEDED")
    score = run.get("score")
    if type(score) not in (int, float) or not math.isfinite(score):
        failures.append("Run score is not a finite number")
    elif baseline.get("comparable", True) and abs(
        score - baseline["historical_mean"]
    ) > 0.01 + 0.05 * abs(baseline["historical_mean"]):
        failures.append("Score is outside cross-site-agreement/v1 tolerance")
    if train_id is None:
        failures.append("No successful train verification was recorded")
    return failures


def save_receipt(path: Path, receipt: dict) -> None:
    atomic_write(path, json_text(receipt).encode())


class ManagementError(Exception):
    def __init__(self, kind: str, recoverable: bool = True):
        super().__init__(kind)
        self.recoverable = recoverable


class ObservationDeadline(Exception):
    pass


def management_json(
    base: str,
    method: str,
    path: str,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: float = 30,
) -> tuple[object, str | None]:
    """Read bounded JSON plus the event cursor, without retaining error bodies."""
    encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request_headers = {"Accept": "application/json", **(headers or {})}
    if encoded is not None:
        request_headers["Content-Type"] = "application/json"
    call = request.Request(
        base + path, data=encoded, headers=request_headers, method=method
    )
    try:
        with request.urlopen(call, timeout=timeout) as response:
            payload = response.read(RESPONSE_LIMIT + 1)
            cursor = response.headers.get("X-Next-Cursor")
    except HTTPError as error:
        error.close()
        raise ManagementError(
            f"http_{error.code}", error.code in {408, 429} or 500 <= error.code < 600
        ) from None
    except (TimeoutError, ConnectionError, URLError, IncompleteRead):
        raise ManagementError("connection_or_timeout") from None
    if len(payload) > RESPONSE_LIMIT:
        raise ManagementError("response_too_large")
    try:
        return json.loads(payload), cursor
    except (ValueError, UnicodeDecodeError):
        raise ManagementError("invalid_json", recoverable=False) from None


def record_error(
    record: dict, category: str, operation: str, error: ManagementError
) -> None:
    record[category + "_error_count"] = record.get(category + "_error_count", 0) + 1
    history = record.setdefault(category + "_errors", [])
    history.append({"operation": operation, "kind": str(error)})
    del history[:-ERROR_HISTORY_LIMIT]


def observation_timeout(record: dict) -> float:
    remaining = record["deadline_unix"] - time.time()
    if remaining <= 0:
        raise ObservationDeadline
    return min(30, remaining)


def cancel_expired(
    base: str, record: dict, receipt: dict, path: Path, poll_seconds: float
) -> dict:
    # A failed cancellation is resumable and must not prevent the next task.
    record.update(
        status="CANCEL_PENDING",
        cancellation_requested=False,
        failure_summaries=[
            "Observation deadline exceeded; cancellation pending; resume this receipt to retry"
        ],
    )
    key = "parity-cancel-" + stable_ulid(record["evaluation_id"].encode())
    for attempt in range(CANCEL_ATTEMPTS):
        record["cancellation_attempts"] = record.get("cancellation_attempts", 0) + 1
        save_receipt(path, receipt)
        try:
            management_json(
                base,
                "POST",
                f"/api/v1/evaluations/{record['evaluation_id']}/cancel",
                headers={"Idempotency-Key": key},
            )
        except ManagementError as error:
            record_error(record, "cancellation", "cancel", error)
            save_receipt(path, receipt)
            if not error.recoverable:
                return record
            if attempt + 1 < CANCEL_ATTEMPTS:
                time.sleep(poll_seconds)
        else:
            record.update(
                status="TIMEOUT",
                cancellation_requested=True,
                failure_summaries=[
                    "Observation deadline exceeded; cancellation requested"
                ],
            )
            save_receipt(path, receipt)
            return record
    return record


def replay_task(
    base: str,
    name: str,
    document: dict,
    task_document: dict,
    baseline: dict,
    image: dict,
    receipt: dict,
    receipt_path: Path,
    attempt: str,
    poll_seconds: float,
) -> dict:
    candidate_id = document["driver_config"]["candidate_version_id"]
    task_id = task_document["driver_config"]["task_version_id"]
    binding = {
        "task": name,
        "task_version_id": task_id,
        "candidate_version_id": candidate_id,
        "attempt": attempt,
        "candidate_definition_digest": document["definition_digest"],
        "task_definition_digest": task_document["definition_digest"],
        "image_digests": {
            "reference_public": image["digest"],
            "reference_mirror": document["driver_config"]["overlay_image"]["digest"],
            **{
                key: value["digest"]
                for key, value in task_document["driver_config"]["images"].items()
            },
        },
        "launcher_sha256": image["launcher_sha256"],
        "payload_sha256": image["payload_sha256"],
        "source_set_sha256": baseline.get("source_set_sha256"),
        "datasets": baseline.get("datasets"),
        "evaluator_sha256": baseline.get("evaluator_sha256"),
        "verifier_adapter_sha256": baseline.get("verifier_adapter_sha256"),
        "historical_mean": baseline.get("historical_mean"),
        "historical_spread": baseline.get("historical_spread"),
        "agreement_rule": RULE,
    }
    previous = receipt["tasks"].get(name)
    classification = {
        key: baseline[key] for key in CLASSIFICATION_KEYS if key in baseline
    }
    if previous is not None:
        previous_classification = {
            key: previous[key] for key in CLASSIFICATION_KEYS if key in previous
        }
        if any(previous.get(key) != value for key, value in binding.items()) or (
            previous_classification and previous_classification != classification
        ):
            raise ValueError(
                f"receipt binds different inputs for {name}; use a new receipt and attempt"
            )
        if previous.get("status") in TERMINAL:
            return previous
    else:
        receipt["tasks"][name] = {
            **binding,
            "status": "PENDING",
            "evaluation_id": None,
            "run_id": None,
            "state": None,
            "outcome": None,
            "score": None,
            "failure_summaries": [],
            "train_verification_id": None,
        }
    record = receipt["tasks"][name]
    # Old terminal receipts retain their original verdict; old unfinished
    # receipts acquire classification when observation resumes.
    record.update(classification)
    save_receipt(receipt_path, receipt)
    if record["evaluation_id"] is None:
        # A deadline alone is not evidence of a POST. Only SUBMITTING permits
        # recovery of an uncertain response using the same idempotency identity.
        recovering = record.get("submission_state") == "SUBMITTING"
        if (
            not recovering
            and "deadline_unix" in record
            and time.time() >= record["deadline_unix"]
        ):
            raise ValueError(f"unsubmitted parity task deadline expired: {name}")
        remote = api_json(base, "GET", f"/api/v1/task-versions/{task_id}")
        if remote.get("definition_digest") != task_document["definition_digest"]:
            raise ValueError(f"seeded task definition differs in Management: {name}")
        ensure_version(base, "/api/v1/candidate-versions", candidate_id, document)
        if "deadline_unix" not in record:
            quota = task_document["driver_config"]["quota"]
            record["deadline_unix"] = (
                time.time()
                + quota["wall_time_seconds"]
                + quota["verifier"]["eval_timeout_seconds"]
                + 900
            )
            record["submission_state"] = "UNSUBMITTED"
            save_receipt(receipt_path, receipt)
        if not recovering and time.time() >= record["deadline_unix"]:
            raise ValueError(f"unsubmitted parity task deadline expired: {name}")
        key = f"parity-{name}-{candidate_id}-{attempt}"
        record["submission_state"] = "SUBMITTING"
        save_receipt(receipt_path, receipt)
        try:
            evaluation, _ = management_json(
                base,
                "POST",
                "/api/v1/evaluations",
                {
                    "candidate_version_id": candidate_id,
                    "task_version_ids": [task_id],
                    "max_parallel_runs": 1,
                    "priority": 0,
                    "labels": {"purpose": "reference-parity"},
                },
                headers={"Idempotency-Key": key},
            )
        except ManagementError as error:
            record_error(record, "submission", "submit", error)
            save_receipt(receipt_path, receipt)
            raise
        items = evaluation.get("items", [])
        if len(items) != 1 or not isinstance(items[0].get("run_id"), str):
            raise ValueError("Management did not create exactly one parity Run")
        record.update(
            evaluation_id=evaluation["id"],
            run_id=items[0]["run_id"],
            status="RUNNING",
            submission_state="SUBMITTED",
        )
        save_receipt(receipt_path, receipt)
    while True:
        operation = "evaluation"
        try:
            evaluation, _ = management_json(
                base,
                "GET",
                f"/api/v1/evaluations/{record['evaluation_id']}",
                timeout=observation_timeout(record),
            )
            operation = "run"
            run, _ = management_json(
                base,
                "GET",
                f"/api/v1/runs/{record['run_id']}",
                timeout=observation_timeout(record),
            )
            if (
                run.get("evaluation_id") != record["evaluation_id"]
                or run.get("task_version_id") != task_id
                or run.get("candidate_version_id") != candidate_id
            ):
                raise ValueError(
                    "Management Run does not belong to this parity attempt"
                )
            score = run.get("score")
            record.update(
                evaluation_state=evaluation.get("status"),
                state=run.get("state"),
                outcome=run.get("outcome"),
                score=score
                if type(score) in (int, float) and math.isfinite(score)
                else None,
            )
            completed = (
                evaluation.get("status") in TERMINAL_EVALUATIONS
                and run.get("state") in TERMINAL_RUNS
            )
            operation = "events"
            cursor = None
            while not record["train_verification_id"]:
                query = {"limit": 500}
                if cursor:
                    query["cursor"] = cursor
                events, next_cursor = management_json(
                    base,
                    "GET",
                    f"/api/v1/runs/{record['run_id']}/events?{urlencode(query)}",
                    timeout=observation_timeout(record),
                )
                record["train_verification_id"] = successful_train(
                    events, record["run_id"]
                )
                save_receipt(receipt_path, receipt)
                if not completed or not next_cursor:
                    break
                if next_cursor == cursor:
                    raise ManagementError("event_cursor_did_not_advance")
                cursor = next_cursor
            observation_timeout(record)
            if completed:
                record["failure_summaries"] = verdict(
                    evaluation, run, baseline, record["train_verification_id"]
                )
                record["status"] = (
                    "FAIL"
                    if record["failure_summaries"]
                    else "NOT_COMPARABLE"
                    if baseline.get("comparable") is False
                    else "PASS"
                )
        except ObservationDeadline:
            return cancel_expired(base, record, receipt, receipt_path, poll_seconds)
        except ManagementError as error:
            record_error(record, "observation", operation, error)
            save_receipt(receipt_path, receipt)
            if not error.recoverable:
                raise
        save_receipt(receipt_path, receipt)
        if record["status"] in TERMINAL:
            return record
        time.sleep(max(0, min(poll_seconds, record["deadline_unix"] - time.time())))


def reference_parity_main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay native reference overlays against seeded Preview tasks, resumably."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", action="append")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--management-url", required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "config/site.json"
    )
    parser.add_argument("--catalog", type=Path, default=TASK_CATALOG_PATH)
    parser.add_argument(
        "--seed-receipt",
        type=Path,
        default=REPOSITORY_ROOT / ".state/preview/task-seed-receipt.json",
    )
    parser.add_argument(
        "--mirrors", type=Path, help="default: task-images.json beside the seed receipt"
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--attempt", default="1")
    parser.add_argument("--poll-seconds", type=float, default=15)
    options = parser.parse_args(arguments)
    if re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", options.attempt) is None:
        parser.error("--attempt must be a bounded alphanumeric label")
    if not math.isfinite(options.poll_seconds) or not 0 < options.poll_seconds <= 60:
        parser.error("--poll-seconds must be between 0 and 60")
    base = options.management_url.rstrip("/")
    address = urlsplit(base)
    if (
        address.scheme not in {"http", "https"}
        or not address.hostname
        or address.username is not None
        or address.password is not None
        or address.query
        or address.fragment
    ):
        parser.error(
            "--management-url must be an HTTP(S) endpoint without credentials, query or fragment"
        )
    config, _ = read_and_validate(options.config)
    catalog, task_selection = load_task_selection(
        config, options.catalog, require_resolved=True
    )
    names = (
        task_selection.reference_names
        if options.all
        else list(dict.fromkeys(options.task))
    )
    for name in names:
        if name in task_selection.excluded:
            raise ValueError(
                f"task {name} is excluded: {task_selection.excluded[name]}"
            )
    references = load_manifest(options.catalog, require_resolved=True)
    if set(names) - set(task_selection.reference_names) or set(names) - set(references):
        parser.error("only selected catalog reference tasks may be replayed")
    baselines = load_baselines(options.catalog)
    seed = load_json(options.seed_receipt)
    try:
        validate_seed_receipt(seed, task_selection, config["site"]["name"])
    except DeployError as error:
        raise ValueError(
            f"seed receipt {options.seed_receipt} does not bind catalog {options.catalog}: {error}"
        ) from error
    mirrors = load_json(
        options.mirrors or options.seed_receipt.with_name("task-images.json")
    )
    receipt = (
        load_json(options.receipt)
        if options.receipt.exists()
        else {
            "schema_version": "frontier.reference-parity/v1",
            "management_url": base,
            "tasks": {},
        }
    )
    if (
        receipt.get("schema_version") != "frontier.reference-parity/v1"
        or receipt.get("management_url") != base
    ):
        raise ValueError("receipt belongs to a different Management endpoint or schema")
    results = []
    for name in names:
        validate_baseline(
            name, baselines[name], references[name], load_task_definitions()[name]
        )
        task = next(task for task in catalog["tasks"] if task["name"] == name)
        for kind in ("agent_image", "verifier_image", "dataset_image"):
            if mirrors[name][kind].get("source_sha256") != task[kind]["digest"]:
                raise ValueError(f"stale task mirror: {name}/{kind}")
        task_document = task_version_document(
            task,
            mirrors[name],
            model_domains=task_selection.model_domains,
            definitions=task_selection.definitions,
        )
        validate_seed_task(seed, name, task_document)
        document = candidate_document(
            name, references[name], mirrors["reference_candidates"][name]
        )
        record = replay_task(
            base,
            name,
            document,
            task_document,
            baselines[name],
            references[name],
            receipt,
            options.receipt,
            options.attempt,
            options.poll_seconds,
        )
        unclassified = not any(key in record for key in CLASSIFICATION_KEYS)
        results.append(record["status"] == "PASS" and not unclassified)
        hint = (
            " (unclassified legacy receipt — rerun with a new --receipt/--attempt)"
            if unclassified
            else ""
        )
        print(f"{name}: {record['status']}{hint}")
    return 0 if all(results) else 1
