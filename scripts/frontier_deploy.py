#!/usr/bin/env python3
"""Guarded fresh-account installer for the executable FrontierCS2 Preview profile.

The site file is deliberately non-secret. AWS CLI, Terraform, Docker, and
kubectl receive credentials through their ordinary local credential chains.
"""

import argparse
import base64
import hashlib
import http.client
import ipaddress
import json
import math
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from preview.acceptance import (
    check_smoke_replay,
    gpu_smoke_candidate,
    prepare_smoke_operation,
    print_admission_receipt,
    run_gpu_smoke,
    run_smoke,
    runtime_admission_preflight,
    scheduler_capacity_preflight,
    smoke_documents,
    smoke_operation_key,
    smoke_timestamp,
    verify_litellm,
    verify_runtime,
)
from preview.cli import (
    apply_site,
    deploy_existing_platform,
    deploy_modal_credentials,
    deploy_runtime,
    main,
    modal_credentials,
    parser,
    plan_site,
    preflight_command,
    require_idle_runtime,
    runtime_upgrade_patches,
    storage_controller_upgrade_patch,
    upgrade_local_network,
    upgrade_runtime,
    validate_command,
)
from preview.config import (
    generated_directory,
    read_and_validate,
    reject_secret_fields,
    render_bundle,
    state_directory,
    validate_config,
    validate_node_group,
)
from preview.images import (
    build_runtime_images,
    cached_registry_image,
    ensure_application_source,
    load_runtime_images,
    mirror_task_images,
    registry_digest,
    source_tree_sha256,
    validate_application_source,
)
from preview.infra import (
    ACCOUNT_RE,
    ADMISSION_BOOTSTRAP_COMMAND,
    ADMISSION_BOOTSTRAP_RECEIPT,
    ADMISSION_COMMAND_MAP,
    ADMISSION_RECEIPT_MAP,
    APPLICATION_REPOSITORY,
    CODEX_AGENT_GPU_INTERFACE,
    COMMIT_RE,
    DIGEST_RE,
    EVAL_URI,
    FORBIDDEN_CONFIG_KEYS,
    IAM_ROLE_ID_RE,
    INSTANCE_TYPE_RE,
    LITELLM_NAMESPACE,
    LITELLM_SECRET,
    LITELLM_SERVICE,
    LOCAL_OBJECT_PVC,
    LOGICAL_AGENT_BASE,
    LOGICAL_CANDIDATE_OVERLAY,
    LOGICAL_COMPOSED_REPOSITORY,
    LOGICAL_VERIFIER,
    MANAGEMENT_SERVICE,
    MODEL_DOMAIN_PATTERN,
    PLATFORM_NAMESPACE,
    REGION_RE,
    REGISTRY_SERVICE,
    REPOSITORY_ROOT,
    RUNTIME_IMAGE_NAMES,
    S3_BUCKET_RE,
    SITE_RE,
    SUPPORTED_KUBERNETES_VERSION,
    SUPPORTED_SCHEMA,
    SUPPORTED_TASK_CATALOG_SCHEMA,
    TASK_CATALOG_PATH,
    TASK_DEFINITIONS_PATH,
    TASK_IMAGE_TAG_RE,
    TASK_PLACEHOLDER_DIGEST,
    TERMINAL_EVALUATIONS,
    TERMINAL_RUNS,
    TRAIN_URI,
    ULID_ALPHABET,
    DeployError,
    ManagementHTTPError,
    admission_control_maps,
    admission_control_rbac,
    api_cidrs,
    api_json,
    api_json_with_headers,
    atomic_write,
    aws_identity,
    canonical_json,
    canonical_object_sha256,
    command_environment,
    dataset_seed_names,
    deploy_bootstrap,
    deploy_litellm,
    duplicate_safe_object,
    ensure_admission_control_maps,
    ensure_litellm_secret_files,
    executable,
    free_local_port,
    k8s_list,
    kubectl_apply,
    kubernetes_environment,
    load_json,
    platform_labels,
    platform_outputs,
    port_forward,
    render_bootstrap_manifest,
    render_litellm_manifest,
    render_litellm_namespace,
    render_runtime_manifest,
    require_exact_keys,
    require_int,
    require_mapping,
    require_string,
    run_command,
    sha256_bytes,
    stable_ulid,
    state_bucket_exists,
    terraform_apply_plan,
    terraform_environment,
    terraform_init_and_validate,
    terraform_output_values,
    terraform_plan,
    update_kubeconfig,
    validate_image,
    validate_litellm_secret_document,
    validate_outputs,
    validate_runtime_images,
)
from preview.publication import (
    candidate_version_document,
    check_seed_tasks,
    ensure_version,
    find_version,
    load_task_definitions,
    load_task_selection,
    local_author_receipt,
    render_task_dataset_seed_manifest,
    seed_tasks,
    task_dataset_relative_path,
    task_dataset_seed_job_name,
    task_dataset_uri,
    task_version_document,
    validate_seed_receipt,
    validate_seed_task,
    validate_task_catalog,
)
from task_selection import TaskSelection, requirements, select_tasks


def __getattr__(name: str):
    if name == "TASK_DEFINITIONS":
        from preview import publication

        return publication.load_task_definitions()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    raise SystemExit(main())
