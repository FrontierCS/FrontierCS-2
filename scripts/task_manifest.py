"""One task's definition and file layout, from `task.toml` or a legacy `policy.json`.

`task.toml` (schema `frontier.task/v1`) is the only definition of a migrated task. Tools
never parse it themselves: they call `load_policy` for the policy view they always used, and
`layout` for the paths that differ between the two formats. A task directory holds exactly one
of the two files; support for `policy.json` goes once every task is migrated.

The policy view keeps the legacy shape so that the rendered task definition, and therefore the
platform's TaskConfig, are produced by the same code for both formats. Values that the judge or
the Agent image need at build time (`config.yaml`, `submission-policy.json`, the verifier's
entrypoint) are generated from `task.toml` by `generated_files`, never committed.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHARED = REPO / "tasks" / "shared"
TASK_SCHEMA = "frontier.task/v1"
POLICY_SCHEMA = "frontier.frontiersmith-task-policy/v1"
MANIFEST = "task.toml"
LEGACY = "policy.json"
GENERATED = ".frontier-generated"

# role -> (legacy Dockerfile, native Dockerfile), both relative to the task root.
DOCKERFILES = {
    "agent-base": ("agent-base/Dockerfile", "docker/agent.Dockerfile"),
    "verifier": ("verifier/Dockerfile", "docker/verifier.Dockerfile"),
    "agent-gpu": ("agent-gpu/Dockerfile", "docker/agent-gpu.Dockerfile"),
    "verifier-gpu": ("verifier-gpu/Dockerfile", "docker/verifier-gpu.Dockerfile"),
}


class ManifestError(ValueError):
    """A task definition that is missing, ambiguous or malformed."""


# ---------------------------------------------------------------- schema
# Each table lists its keys: True = required, False = optional. Unknown keys are refused.
# A task.toml holds only what differs per task; tasks/shared/task-defaults.toml holds the
# explicit defaults of the optional keys, and the rest is derived (see that file's header).
TOP = {
    "schema": True, "id": True, "display_name": True, "version": True,
    "agent": True, "verifier": True, "quota": True, "submission": True, "network": False,
    "judge": False, "gpu": False, "accelerator": False, "gpu_parent_resources": False,
    "additional_keys": False, "pumpkin": False, "paper": False, "feedback": False,
    "test_sandboxes": False,
}
TABLES = {
    "agent": {"cpu_millis": True, "memory_bytes": True, "workspace_bytes": True},
    "quota": {"wall_time_seconds": True, "max_train_calls": True,
              "max_train_wall_seconds": True, "eval_timeout_seconds": True},
    # Keys of edit_scope.Rules plus the platform's entrypoint and archive cap.
    "submission": {"max_bytes": True, "allowed_paths": True, "entrypoint": False,
                   "max_patch_bytes": False,
                   "max_changed_files": False, "deny_paths": False, "allowed_suffixes": False,
                   "max_file_bytes": False, "allow_empty": False, "allow_new_files": False,
                   "allow_delete": False, "allow_rename": False, "allow_executable": False, "forbidden_added": False,
                   "forbidden_content": False, "ignore_case": False},
    "verifier": {"cpu_millis": True, "memory_bytes": True, "max_evidence_bytes": False,
                 "max_metrics_bytes": False, "command": False, "workspace_bytes": False},
    "network": {"allowed_domains": False, "audit": False},
    "judge": {"config": False},
    "pumpkin": {"slot": False, "trust": False, "result": False},
    # What the task is based on, for readers (the public task pages, tasks/metadata/<id>.yaml);
    # the platform and the judge never read it.
    "paper": {"title": True, "venue": True, "url": True, "summary": True},
}
# [gpu.verifier]: the platform's Driver quota (max_gpu_instances, max_gpu_seconds) and the shared
# GPU adapter's lease (fcs2_verifier_gpu.py). The adapter keys go to verifier.json only, never
# into the platform definition. The placement is derived, never written: "runner" (the whole
# verification on one runner; lease_seconds defaults to quota.eval_timeout_seconds in both
# modes), or "control" for a task that declares a command test sandbox (below): its judge runs
# on the CPU control image and it sets no lease_seconds (each sandbox declares its own).
GPU_VERIFIER_KEYS = {"max_gpu_instances": True, "max_gpu_seconds": True, "lease_seconds": False}
GPU_ADAPTER_KEYS = ("placement", "lease_seconds")
RUNNER_PLACEMENT = "runner"
CONTROL_PLACEMENT = "control"
MAX_LEASE_SECONDS = 7 * 24 * 60 * 60
AUDIT_KEYS = {"max_bundle_bytes": False, "max_model_request_bytes": False,
              "max_model_response_bytes": False, "max_model_sessions": False,
              "max_ssh_bytes_per_direction": False, "max_ssh_sessions": False,
              "ssh_session_budgets": False}
# Test sandboxes (the FCS paper's term): containers the Driver starts beside a GPU runner of
# the task's Verifier or Agent; the platform calls each one a "sidecar" (policy/definition key
# `sidecars`). [test_sandboxes.<name>] declares one sandbox or a template of many:
#   trust      = "untrusted" (default) | "trusted"; only a trusted sandbox may hold judge
#                secrets such as the operation's attestation key
#   dockerfile = its image recipe, docker/test-sandbox*.Dockerfile
#   workdir    = its working directory
#   one sandbox:  args = [...], phases = [...] (agent/train/eval), attestation_key (trusted only);
#                 sidecar and image `<name>`
#   a template:   instances = { dataset_key = "...", manifest = "judge/....json" } and
#                 [test_sandboxes.<name>.roles.<role>] args = [...] with {instance}: one
#                 sandbox per instance id that datasets/{train,eval}.json select in their list
#                 `dataset_key`, running in the modes whose dataset selects it. The manifest's
#                 "test_sandboxes": {<name>: {<id>: {index, base, build_args?}}} gives each its
#                 slot, its digest-pinned base image (build arg TEST_SANDBOX_BASE, used in a
#                 FROM) and extra TEST_SANDBOX_* build args; TEST_SANDBOX_INSTANCE is the id.
#                 Sidecars `<name>-<index>-<role>` on image `<name>-<index>`.
# Images are published as fcs2-task-<id>-sidecar-<image>. resolve() expands every table into
# the platform's `sidecars` mapping (task["sidecars"], derived) and the recipe behind each image
# (task["sidecar_images"]).
#
# access = "service" (the default) is the kind above. access = "command" is the smallest
# extension (see tasks/template/TEMPLATE.md): a machine of its own that the
# judge, now on the CPU control image, leases per run through the shared adapter's
# run_sandboxes(), uploads only what it passes into `workdir`, runs commands there (deadline
# enforced from the control side) and gets declared files back. Keys:
#   access, trust, dockerfile, workdir   as above (trust "untrusted" = candidate-only machine)
#   resources     = the machine's shape: { gpus = 1..=64, gpu_type = "<Driver GPU type>" } is
#                   a GPU runner; { cpu_millis, memory_bytes } without gpus is a CPU machine
#                   (a dedicated pod; its provisioning path is the next step, refused for now)
#   max_concurrent = machines of this sandbox at once
#   lease_seconds = { train = N, eval = M }: the lease requested per machine, per mode
# `dockerfile` names the runner image: docker/agent-gpu.Dockerfile (the task's Agent GPU image,
# the candidate's own environment) or, trusted only, docker/verifier-gpu.Dockerfile (an image
# that carries the judge's harness). An untrusted one's recipe copies nothing from judge/.
# The adapter reads them from verifier.json `gpu.sandboxes`; they are never platform sidecars.
TEST_SANDBOX_KEYS = {"access": False, "trust": False, "dockerfile": True, "workdir": True,
                     "args": False, "phases": False, "attestation_key": False,
                     "instances": False, "roles": False}
TEST_SANDBOX_ACCESS = ("service", "command")
COMMAND_SANDBOX_KEYS = {"access": True, "trust": False, "dockerfile": True, "workdir": True,
                        "resources": True, "max_concurrent": True, "lease_seconds": True}
# A command sandbox's recipe -> (the Driver runner image, its [gpu.<role>] table, trusted only).
# Each is an image the platform already publishes for the task (TaskConfig images.agent_gpu and
# images.verifier gpu_tag); a docker/test-sandbox*.Dockerfile runner image needs a TaskConfig
# image entry of its own in core first.
COMMAND_RUNNER_IMAGES = {"docker/agent-gpu.Dockerfile": ("agent_gpu", "agent", False),
                         "docker/verifier-gpu.Dockerfile": ("verifier_gpu", "verifier", True)}
GPU_TYPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
MAX_RUNNER_GPUS = 64  # the Driver's gpu_count bound per runner
TEST_SANDBOX_SINGLE = {"args", "phases"}
TEST_SANDBOX_TEMPLATE = {"instances", "roles"}
TEST_SANDBOX_TRUST = ("untrusted", "trusted")
TEST_SANDBOX_ROLE_KEYS = {"args": True}
TEST_SANDBOX_INSTANCES_KEYS = {"dataset_key": True, "manifest": True}
TEST_SANDBOX_DOCKERFILE = re.compile(r"docker/test-sandbox(?:-[a-z0-9-]+)?\.Dockerfile")
TEST_SANDBOX_PLACEHOLDER = "{instance}"
TEST_SANDBOX_BUILD_ARG = re.compile(r"TEST_SANDBOX_[A-Z0-9_]+")
SIDECAR_PHASES = ("agent", "train", "eval")
SIDECAR_LABEL = re.compile(r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?")
DIGEST_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
DEFAULTS_FILE = SHARED / "task-defaults.toml"
# The category folders of FrontierCS-2 main (tasks/frontier-cs-2.0-demo/problems/<Category>/).
CATEGORIES = frozenset({
    "Arch", "Bioinformatics", "Crypto", "Database", "HPC", "MLSys", "Machine Learning",
    "Network", "Programming Language", "Robotics", "Security", "Software Engineering", "Theory",
})
TEMPLATE_ROOT = REPO / "tasks" / "template"


def _edit_scope():
    """tasks/shared/lib/edit_scope.py, loaded once by path (it is not a package)."""
    import importlib.util
    import sys
    name = "frontier_shared_edit_scope"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SHARED / "lib/edit_scope.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # No __pycache__ in tasks/shared: an ignored file there fails the base builds.
        writes, sys.dont_write_bytecode = sys.dont_write_bytecode, True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = writes
    return sys.modules[name]


def _check_keys(where: str, table: object, spec: dict) -> dict:
    if not isinstance(table, dict):
        raise ManifestError(f"{where} must be a table")
    unknown = sorted(set(table) - set(spec))
    if unknown:
        raise ManifestError(f"{where}: unknown keys {unknown}; expected some of {sorted(spec)}")
    missing = sorted(key for key, required in spec.items() if required and key not in table)
    if missing:
        raise ManifestError(f"{where}: missing keys {missing}")
    return table


def _positive_int(where: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ManifestError(f"{where} must be a positive integer")
    return value


def defaults() -> dict:
    """tasks/shared/task-defaults.toml."""
    return tomllib.loads(DEFAULTS_FILE.read_text(encoding="utf-8"))


def _merge(base: dict, extra: dict) -> dict:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def category(root: Path) -> str:
    """The task's category: its folder under problems/, one of FrontierCS-2 main's folders.
    The template stands in Theory/ like a real task."""
    name = Path(root).resolve().parent.name
    if name not in CATEGORIES:
        raise ManifestError(f"{root}: task folder must be inside a category folder "
                            f"({', '.join(sorted(CATEGORIES))}), not {name!r}")
    return name


# [feedback]: metric name -> the one sentence the Agent's README shows for it, in display order.
# These evaluator metrics, and only these, reach the Agent in a train result (besides
# score_unbounded, beat_spec_progress and rejected, which every task returns).
FEEDBACK_NAME = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
FEEDBACK_RESERVED = frozenset({"score_unbounded", "beat_spec_progress",
                               "beat_spec_workload_passes", "rejected"})


def _check_feedback(feedback: object) -> dict:
    if not isinstance(feedback, dict):
        raise ManifestError("feedback must be a table of metric = \"meaning\"")
    for name, meaning in feedback.items():
        if FEEDBACK_NAME.fullmatch(name) is None:
            raise ManifestError(f"feedback.{name}: names are lower_snake_case")
        if (name in FEEDBACK_RESERVED or "reference" in name or "margin" in name
                or name.startswith("beats")):
            raise ManifestError(f"feedback.{name}: a reserved or private metric name")
        if not isinstance(meaning, str) or not meaning.strip():
            raise ManifestError(f"feedback.{name} must be a non-empty sentence")
    return feedback


def visible_metrics(task: dict) -> list[str]:
    """The evaluator metrics a train result returns to the Agent: the [feedback] names."""
    return list(task["feedback"])


def datasets(root: Path) -> dict:
    """The two dataset objects: datasets/<mode>.json selectors or datasets/<mode>.tar."""
    import hashlib

    found = {}
    for mode in ("train", "eval"):
        paths = [Path(root) / "datasets" / f"{mode}.{ext}" for ext in ("json", "tar")]
        present = [path for path in paths if path.is_file()]
        if len(present) != 1:
            raise ManifestError(f"{root}: needs exactly one of datasets/{mode}.json, .tar")
        path = present[0]
        body = path.read_bytes()
        entry = {"path": f"datasets/{path.name}", "bytes": len(body),
                 "version": "sha256:" + hashlib.sha256(body).hexdigest()}
        if path.suffix == ".tar":
            entry.update(media_type="application/x-tar", source="task")
        found[mode] = entry
    if found["train"]["version"] == found["eval"]["version"]:
        raise ManifestError(f"{root}: train and eval datasets must differ")
    return found


def read_task(root: Path) -> dict:
    """Parse and check `root/task.toml`; return it merged over the shared defaults, with the
    derived values added."""
    return resolve(root)[0]


def _leaves(table: dict, prefix: tuple = ()) -> dict[tuple, object]:
    found = {}
    for key, value in table.items():
        if isinstance(value, dict) and value:
            found.update(_leaves(value, (*prefix, key)))
        else:
            found[(*prefix, key)] = value
    return found


DERIVED = {
    "category": "derived: the task's folder",
    "datasets": "derived: datasets/ files",
}
# Expanded from [test_sandboxes] (and the datasets); `show` summarizes them.
EXPANDED = ("sidecars", "sidecar_images")


def resolve(root: Path) -> tuple[dict, dict[tuple, str]]:
    """The effective task and where each value comes from: task.toml, a default in
    tasks/shared/task-defaults.toml, or derived from the task's files."""
    root = Path(root)
    path = root / MANIFEST
    try:
        task = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ManifestError(f"{path}: {error}") from error
    _check_keys(MANIFEST, task, TOP)
    if task["schema"] != TASK_SCHEMA:
        raise ManifestError(f"{MANIFEST}: schema must be {TASK_SCHEMA!r}")
    if task["id"] != root.name:
        raise ManifestError(f"{MANIFEST}: id {task['id']!r} does not match directory {root.name!r}")
    for key in ("display_name", "version"):
        if not isinstance(task[key], str) or not task[key]:
            raise ManifestError(f"{MANIFEST}: {key} must be a non-empty string")
    for name, spec in TABLES.items():
        if name in task:
            _check_keys(name, task[name], spec)
    if "audit" in task.get("network", {}):
        _check_keys("network.audit", task["network"]["audit"], AUDIT_KEYS)
    gpu_verifier = task.get("gpu", {}).get("verifier") if isinstance(task.get("gpu"), dict) else None
    if gpu_verifier is not None:
        _check_keys("gpu.verifier", gpu_verifier, GPU_VERIFIER_KEYS)
        for key in ("max_gpu_instances", "max_gpu_seconds"):
            _positive_int(f"gpu.verifier.{key}", gpu_verifier[key])
        if "lease_seconds" in gpu_verifier:
            lease = _check_keys("gpu.verifier.lease_seconds", gpu_verifier["lease_seconds"],
                                {"train": True, "eval": True})
            for mode, value in lease.items():
                if type(value) is not int or not 1 <= value <= MAX_LEASE_SECONDS:
                    raise ManifestError(f"gpu.verifier.lease_seconds.{mode} must be an integer "
                                        f"in 1..={MAX_LEASE_SECONDS}")
    written = set(_leaves(task))
    task = _merge(defaults(), task)
    for name in ("agent", "quota"):
        for key, value in task[name].items():
            _positive_int(f"{name}.{key}", value)
    submission = task["submission"]
    # The patch cap defaults to the archive cap (task-defaults.toml says so); a task whose Agent
    # was told a smaller patch limit sets it.
    submission.setdefault("max_patch_bytes", submission["max_bytes"])
    for key in ("max_bytes", "max_changed_files", "max_patch_bytes"):
        _positive_int(f"submission.{key}", submission[key])
    if submission["max_patch_bytes"] > submission["max_bytes"]:
        raise ManifestError("submission.max_patch_bytes may not exceed submission.max_bytes "
                            "(the patch travels inside the archive)")
    try:  # the shared module is the one definition of what these rules mean
        _edit_scope().Rules.from_mapping(edit_scope_rules(task))
        for pattern in (*submission["forbidden_added"], *submission["forbidden_content"]):
            re.compile(pattern)
    except (TypeError, ValueError, re.error) as error:
        raise ManifestError(f"submission: {error}") from error
    if (not isinstance(submission["allowed_paths"], list) or not submission["allowed_paths"]
            or not all(isinstance(p, str) and p for p in submission["allowed_paths"])):
        raise ManifestError("submission.allowed_paths must be a non-empty list of paths")
    if submission["entrypoint"] != "payload/solution.patch":
        raise ManifestError("submission.entrypoint must be payload/solution.patch")
    _positive_int("verifier.max_evidence_bytes", task["verifier"]["max_evidence_bytes"])
    _positive_int("verifier.max_metrics_bytes", task["verifier"]["max_metrics_bytes"])
    for key, value in task.get("paper", {}).items():
        if not isinstance(value, str) or not value.strip():
            raise ManifestError(f"paper.{key} must be a non-empty string")
    if "paper" in task and not task["paper"]["url"].startswith("https://"):
        raise ManifestError("paper.url must be an https:// link")
    pumpkin = task["pumpkin"]
    if pumpkin["slot"] not in {"code", "data"}:
        raise ManifestError("pumpkin.slot must be 'code' or 'data'")
    if pumpkin["trust"] not in {"trusted", "untrusted"}:
        raise ManifestError("pumpkin.trust must be 'trusted' or 'untrusted'")
    _check_keys("pumpkin.result", pumpkin["result"], {"status": True, "score": True})
    _check_feedback(task["feedback"])
    commands = _command_sandboxes(task)
    if commands:
        if gpu_verifier is None:
            raise ManifestError("a GPU command test sandbox needs [gpu.verifier] (its runners "
                                "are the Verifier's GPU quota)")
        if "lease_seconds" in gpu_verifier:
            raise ManifestError("a task with a command test sandbox runs its judge on the "
                                "control image: [gpu.verifier] sets no lease_seconds (each "
                                "sandbox declares its own lease_seconds)")
        needed = sum(table["max_concurrent"] * table["resources"].get("gpus", 0)
                     for table in commands.values())
        if needed > gpu_verifier["max_gpu_instances"]:
            raise ManifestError(f"the command test sandboxes hold up to {needed} GPUs at once, "
                                f"more than [gpu.verifier] max_gpu_instances "
                                f"({gpu_verifier['max_gpu_instances']})")
        task["gpu"]["verifier"]["placement"] = CONTROL_PLACEMENT
    elif gpu_verifier is not None:
        timeout = task["quota"]["eval_timeout_seconds"]
        task["gpu"]["verifier"]["placement"] = RUNNER_PLACEMENT
        task["gpu"]["verifier"].setdefault("lease_seconds", {"train": timeout, "eval": timeout})
    task["category"] = category(root)
    task["datasets"] = datasets(root)
    expanded, images = _sidecars(root, task)
    if expanded:
        task["sidecars"], task["sidecar_images"] = expanded, images
    origins = {}
    for key in _leaves(task):
        if key[0] in EXPANDED:
            continue
        if key[0] in DERIVED:
            origins[key] = DERIVED[key[0]]
        elif key == ("submission", "max_patch_bytes") and key not in written:
            origins[key] = "default: submission.max_bytes"
        elif key == ("gpu", "verifier", "placement") and commands:
            origins[key] = 'derived: [test_sandboxes] access = "command"'
        else:
            origins[key] = "task.toml" if key in written else "default"
    return task, origins


def _string_list(where: str, value: object) -> list[str]:
    if (not isinstance(value, list) or not value
            or not all(isinstance(item, str) and item for item in value)):
        raise ManifestError(f"{where} must be a non-empty list of non-empty strings")
    return value


def _phases(where: str, value: object) -> list[str]:
    phases = _string_list(where, value)
    if len(set(phases)) != len(phases) or not set(phases) <= set(SIDECAR_PHASES):
        raise ManifestError(f"{where} must be unique values of {list(SIDECAR_PHASES)}")
    return phases


def _label(where: str, value: object) -> str:
    if not isinstance(value, str) or SIDECAR_LABEL.fullmatch(value) is None:
        raise ManifestError(f"{where} must be a lowercase DNS label")
    return value


def _selected(root: Path, mode: str, key: str) -> list[str]:
    path = Path(root) / "datasets" / f"{mode}.json"
    try:
        selector = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ManifestError(f"test_sandboxes: needs the selector {path}: {error}") from error
    ids = selector.get(key) if isinstance(selector, dict) else None
    if (not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids)
            or len(set(ids)) != len(ids)):
        raise ManifestError(f"datasets/{mode}.json: {key} must be a list of distinct ids")
    return ids


def _instances(root: Path, name: str, spec: object) -> list[tuple[str, str, str, dict, list]]:
    """(index, id, base, build args, phases) of every instance of a template, by index."""
    where = f"test_sandboxes.{name}.instances"
    _check_keys(where, spec, TEST_SANDBOX_INSTANCES_KEYS)
    key, manifest_path = spec["dataset_key"], spec["manifest"]
    if not isinstance(key, str) or not key:
        raise ManifestError(f"{where}.dataset_key must name the selectors' list of ids")
    if (not isinstance(manifest_path, str) or not manifest_path.startswith("judge/")
            or ".." in Path(manifest_path).parts):
        raise ManifestError(f"{where}.manifest must be a JSON file below judge/")
    selected = {mode: _selected(root, mode, key) for mode in ("train", "eval")}
    try:
        manifest = json.loads((Path(root) / manifest_path).read_text(encoding="utf-8"))
        instances = manifest["test_sandboxes"][name]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ManifestError(f"{manifest_path}: needs \"test_sandboxes\": {{{name!r}: "
                            f"{{id: {{index, base, build_args?}}}}}} ({error!r})") from error
    if not isinstance(instances, dict):
        raise ManifestError(f"{manifest_path}: test_sandboxes.{name} must map ids to entries")
    wanted = set(selected["train"]) | set(selected["eval"])
    if set(instances) != wanted:
        raise ManifestError(f"{manifest_path}: test_sandboxes.{name} must hold exactly the ids "
                            f"the datasets select; differs by {sorted(set(instances) ^ wanted)[:5]}")
    rows, indexes = [], set()
    for identity, entry in instances.items():
        at = f"{manifest_path}: test_sandboxes.{name}.{identity}"
        if any(c.isspace() for c in identity):
            raise ManifestError(f"{at}: an instance id has no whitespace")
        _check_keys(at, entry, {"index": True, "base": True, "build_args": False})
        index, base = entry["index"], entry["base"]
        if not isinstance(index, str):
            raise ManifestError(f"{at}: index must be a string")
        _label(f"{at}: index", f"{name}-{index}")
        if index in indexes:
            raise ManifestError(f"{at}: index {index!r} is repeated")
        indexes.add(index)
        if not isinstance(base, str) or DIGEST_IMAGE.fullmatch(base) is None:
            raise ManifestError(f"{at}: base must be an image pinned by @sha256 digest")
        extra = entry.get("build_args", {})
        if (not isinstance(extra, dict)
                or not all(TEST_SANDBOX_BUILD_ARG.fullmatch(k) and isinstance(v, str)
                           for k, v in extra.items())
                or {"TEST_SANDBOX_BASE", "TEST_SANDBOX_INSTANCE"} & set(extra)):
            raise ManifestError(f"{at}: build_args are TEST_SANDBOX_* = string (not BASE or "
                                "INSTANCE, which the build sets)")
        phases = [mode for mode in ("train", "eval") if identity in selected[mode]]
        rows.append((index, identity, base, extra, phases))
    return sorted(rows)


def _test_sandbox(root: Path, name: str, table: object) -> tuple[dict, dict]:
    """(sidecar name -> profile, image -> (Dockerfile, build args)) of one table."""
    where = f"test_sandboxes.{name}"
    _check_keys(where, table, TEST_SANDBOX_KEYS)
    _label(where, name)
    trust = table.get("trust", "untrusted")
    if trust not in TEST_SANDBOX_TRUST:
        raise ManifestError(f"{where}.trust must be one of {list(TEST_SANDBOX_TRUST)}")
    if "attestation_key" in table:
        if type(table["attestation_key"]) is not bool:
            raise ManifestError(f"{where}.attestation_key must be a boolean")
        if table["attestation_key"] and trust != "trusted":
            raise ManifestError(f"{where}: only a trust = \"trusted\" test sandbox may hold the "
                                "attestation key")
    dockerfile = table["dockerfile"]
    if not isinstance(dockerfile, str) or TEST_SANDBOX_DOCKERFILE.fullmatch(dockerfile) is None:
        raise ManifestError(f"{where}.dockerfile must be docker/test-sandbox[-<name>].Dockerfile")
    workdir = table["workdir"]
    if not isinstance(workdir, str) or not workdir.startswith("/"):
        raise ManifestError(f"{where}.workdir must be an absolute path")
    keys = set(table)
    if keys & TEST_SANDBOX_SINGLE and keys & TEST_SANDBOX_TEMPLATE or not (
            TEST_SANDBOX_SINGLE <= keys or TEST_SANDBOX_TEMPLATE <= keys):
        raise ManifestError(f"{where}: one sandbox needs args and phases; a template needs "
                            "instances and roles; never both")
    if TEST_SANDBOX_SINGLE <= keys:
        profile = {"args": list(_string_list(f"{where}.args", table["args"]))}
        if "attestation_key" in table:
            profile["attestation_key"] = table["attestation_key"]
        profile.update(image=name, phases=list(_phases(f"{where}.phases", table["phases"])),
                       workdir=workdir)
        return {name: profile}, {name: (dockerfile, {})}
    if "attestation_key" in table:
        raise ManifestError(f"{where}: a template's sandboxes never hold the attestation key")
    roles = table["roles"]
    if not isinstance(roles, dict) or not roles:
        raise ManifestError(f"{where}.roles must hold [{where}.roles.<role>] tables")
    for role, spec in roles.items():
        _label(f"{where}.roles.{role}", role)
        _check_keys(f"{where}.roles.{role}", spec, TEST_SANDBOX_ROLE_KEYS)
        args = _string_list(f"{where}.roles.{role}.args", spec["args"])
        if not any(TEST_SANDBOX_PLACEHOLDER in arg for arg in args):
            raise ManifestError(f"{where}.roles.{role}.args must pass {TEST_SANDBOX_PLACEHOLDER}")
    profiles, images = {}, {}
    for index, identity, base, extra, phases in _instances(root, name, table["instances"]):
        image = f"{name}-{index}"
        images[image] = (dockerfile, {"TEST_SANDBOX_BASE": base,
                                      "TEST_SANDBOX_INSTANCE": identity, **extra})
        for role, spec in roles.items():
            profiles[f"{image}-{role}"] = {
                "args": [arg.replace(TEST_SANDBOX_PLACEHOLDER, identity) for arg in spec["args"]],
                "image": image, "phases": list(phases), "workdir": workdir}
    return profiles, images


def _access(name: str, table: object) -> str:
    access = table.get("access", "service") if isinstance(table, dict) else "service"
    if access not in TEST_SANDBOX_ACCESS:
        raise ManifestError(f"test_sandboxes.{name}.access must be one of {list(TEST_SANDBOX_ACCESS)}")
    return access


def _sandbox_tables(task: dict) -> dict:
    tables = task.get("test_sandboxes", {})
    if not isinstance(tables, dict):
        raise ManifestError("test_sandboxes must be a table of [test_sandboxes.<name>] tables")
    return tables


def command_sandboxes(task: dict) -> dict:
    """The resolved task's access = "command" test sandboxes, by name (already validated)."""
    return {name: table for name, table in _sandbox_tables(task).items()
            if _access(name, table) == "command"}


def _command_sandboxes(task: dict) -> dict:
    """Validate every access = "command" table of the merged task; return them by name."""
    found = command_sandboxes(task)
    timeout = task["quota"]["eval_timeout_seconds"]
    for name, table in found.items():
        where = f"test_sandboxes.{name}"
        _check_keys(where, table, COMMAND_SANDBOX_KEYS)
        _label(where, name)
        trust = table.get("trust", "untrusted")
        if trust not in TEST_SANDBOX_TRUST:
            raise ManifestError(f"{where}.trust must be one of {list(TEST_SANDBOX_TRUST)}")
        resources = _check_keys(f"{where}.resources", table["resources"],
                                {"gpus": False, "gpu_type": False, "cpu_millis": False,
                                 "memory_bytes": False})
        if "gpus" not in resources:
            for key in ("cpu_millis", "memory_bytes"):
                _positive_int(f"{where}.resources.{key}", resources.get(key))
            raise ManifestError(f"{where}: a CPU command sandbox (resources without gpus) needs "
                                "the platform's CPU sandbox provisioning, the next step of "
                                "tasks/template/TEMPLATE.md; only GPU runners exist")
        if set(resources) != {"gpus", "gpu_type"}:
            raise ManifestError(f"{where}.resources of a GPU runner are gpus and gpu_type (the "
                                "host's CPU and memory are [gpu_parent_resources])")
        dockerfile = table["dockerfile"]
        if dockerfile not in COMMAND_RUNNER_IMAGES:
            raise ManifestError(f"{where}.dockerfile must be one of {sorted(COMMAND_RUNNER_IMAGES)} "
                                "(a runner image the task publishes)")
        _image, role, trusted_only = COMMAND_RUNNER_IMAGES[dockerfile]
        if trusted_only and trust != "trusted":
            raise ManifestError(f"{where}: {dockerfile} carries the judge; only a "
                                'trust = "trusted" command sandbox may run on it')
        if role not in task.get("gpu", {}):
            raise ManifestError(f"{where}.dockerfile = {dockerfile!r} needs [gpu.{role}], which "
                                "publishes that image")
        workdir = table["workdir"]
        if (not isinstance(workdir, str) or not workdir.startswith("/workspace/")
                or ".." in Path(workdir).parts or any(c.isspace() for c in workdir)):
            raise ManifestError(f"{where}.workdir must be a directory below /workspace/ (the "
                                "runner user's writable home)")
        if type(resources["gpus"]) is not int or not 1 <= resources["gpus"] <= MAX_RUNNER_GPUS:
            raise ManifestError(f"{where}.resources.gpus must be an integer in 1..={MAX_RUNNER_GPUS}")
        if not isinstance(resources["gpu_type"], str) or GPU_TYPE.fullmatch(resources["gpu_type"]) is None:
            raise ManifestError(f"{where}.resources.gpu_type must be a Driver GPU type such as "
                                '"A100-80GB"')
        _positive_int(f"{where}.max_concurrent", table["max_concurrent"])
        lease = _check_keys(f"{where}.lease_seconds", table["lease_seconds"],
                            {"train": True, "eval": True})
        for mode, value in lease.items():
            if type(value) is not int or not 1 <= value <= timeout:
                raise ManifestError(f"{where}.lease_seconds.{mode} must be an integer in "
                                    f"1..={timeout} (quota.eval_timeout_seconds)")
    return found


def _sidecars(root: Path, task: dict) -> tuple[dict, dict]:
    """The platform's `sidecars` mapping and the recipe behind each image, in task.toml order
    (a template's instances by index, then its roles in task.toml order). Only the
    access = "service" tables are sidecars."""
    tables = {name: table for name, table in _sandbox_tables(task).items()
              if _access(name, table) == "service"}
    found, images = {}, {}
    for name, table in tables.items():
        profiles, recipes = _test_sandbox(root, name, table)
        clash = sorted((set(found) & set(profiles)) | (set(images) & set(recipes)))
        if clash:
            raise ManifestError(f"test sandboxes share the names {clash}")
        found.update(profiles)
        images.update(recipes)
    return found, images


# ---------------------------------------------------------------- policy view
JUDGE_ONLY_VERIFIER_KEYS = frozenset({"max_metrics_bytes"})


def policy_from_task(task: dict) -> dict:
    """The legacy policy dict that `task.toml` stands for."""
    submission = task["submission"]
    policy = {
        "schema_version": POLICY_SCHEMA,
        "problem_id": task["id"],
        "display_name": task["display_name"],
        "family": task["category"],
        "version": task["version"],
        "agent": dict(task["agent"]),
        "quota": dict(task["quota"]),
        "submission": {"entrypoint": submission["entrypoint"], "max_bytes": submission["max_bytes"]},
        # max_metrics_bytes is the shared verifier's own cap (config.yaml), not a platform field.
        "verifier": {k: v for k, v in task["verifier"].items() if k not in JUDGE_ONLY_VERIFIER_KEYS},
        "network": task["network"],
        "metadata": {
            "accelerator": task.get("accelerator"),
            "datasets": task["datasets"],
            "evaluation": {
                "authored_max_dev_submissions": task["quota"]["max_train_calls"],
                "solver_visible_metrics": visible_metrics(task),
            },
            "runtime": {
                "allowed_paths": list(submission["allowed_paths"]),
                "submission_kind": "patch",
            },
        },
    }
    for key in ("gpu", "gpu_parent_resources", "additional_keys"):
        if key in task:
            policy[key] = task[key]
    if "sidecars" in task:
        policy["sidecars"] = task["sidecars"]
    if "verifier" in task.get("gpu", {}):  # the adapter's keys are the judge's, not the platform's
        verifier = {k: v for k, v in task["gpu"]["verifier"].items() if k not in GPU_ADAPTER_KEYS}
        policy["gpu"] = {**task["gpu"], "verifier": verifier}
    return policy


def manifest_path(root: Path) -> Path:
    """The one definition file of the task at `root`."""
    native, legacy = root / MANIFEST, root / LEGACY
    if native.exists() and legacy.exists():
        raise ManifestError(f"{root}: holds both {MANIFEST} and {LEGACY}; keep one")
    if native.exists():
        return native
    if legacy.exists():
        return legacy
    raise ManifestError(f"{root}: no {MANIFEST} or {LEGACY}")


def is_task_dir(path: Path) -> bool:
    return (path / MANIFEST).exists() or (path / LEGACY).exists()


def is_native(root: Path) -> bool:
    """Whether `root` is a task.toml task (`manifest_path` refuses a directory holding both)."""
    return (root / MANIFEST).is_file()


def load_policy(root: Path) -> dict:
    """The task's policy, from whichever definition file it holds."""
    path = manifest_path(root)
    if path.name == MANIFEST:
        return policy_from_task(read_task(root))
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- layout
@dataclass(frozen=True)
class Layout:
    """Where a task's parts live; the two formats differ only in these paths."""

    root: Path
    native: bool

    @property
    def manifest(self) -> Path:
        return self.root / (MANIFEST if self.native else LEGACY)

    @property
    def public(self) -> Path:
        """The legacy Agent-visible tree; a native task has none in its source."""
        return self.root / "agent-base/frontier/task"

    @property
    def statement(self) -> Path:
        return self.root / "README.md" if self.native else self.public / "README.md"

    @property
    def repo(self) -> Path:
        return self.root / "repo" if self.native else self.public / "repo"

    @property
    def starter(self) -> Path:
        return self.root / "starter" if self.native else self.public / "starter"

    @property
    def agent_bin(self) -> Path:
        """Where the Agent tools come from: a legacy task's bin/, else the shared tools (a
        task.toml task's own bin/ overrides them file by file, see `agent_tool`)."""
        return SHARED / "agent/bin" if self.native else self.public / "bin"

    def agent_tool(self, name: str) -> Path:
        """One Agent tool as the image holds it at /frontier/task/bin/<name>: a task.toml
        task's own bin/<name> if it has one, else the shared one."""
        own = self.root / "bin" / name
        return own if self.native and own.is_file() else self.agent_bin / name

    @property
    def verifier_entry(self) -> Path:
        """The Verifier entry: the shared one, or a task.toml task's own verifier/verify.py
        (a standalone Verifier, run through a non-default verifier.command)."""
        own = self.root / "verifier/verify.py"
        if not self.native or own.is_file():
            return own
        return SHARED / "verifier/verify.py"

    @property
    def beat_spec_helper(self) -> Path:
        if self.native:
            return SHARED / "verifier/beat_spec_feedback.py"
        return self.root / "verifier/beat_spec_feedback.py"

    @property
    def lock(self) -> Path | None:
        return None if self.native else self.root / "upstream.lock.json"

    def dockerfile(self, role: str) -> Path:
        if role.startswith("sidecar-"):
            if not self.native:
                return self.root / role / "Dockerfile"
            return self.root / self._sidecar_image(role)[0]
        legacy, native = DOCKERFILES[role]
        return self.root / (native if self.native else legacy)

    def build_args(self, role: str) -> dict[str, str]:
        """The docker build args of one image role (only test sandboxes have any)."""
        if self.native and role.startswith("sidecar-"):
            return dict(self._sidecar_image(role)[1])
        return {}

    def _sidecar_image(self, role: str) -> tuple[str, dict[str, str]]:
        images = read_task(self.root).get("sidecar_images", {})
        name = role.removeprefix("sidecar-")
        if name not in images:
            raise ManifestError(f"{self.root.name}: no service or test sandbox image {name!r}")
        return images[name]

    def has_role(self, role: str) -> bool:
        return self.dockerfile(role).is_file() if self.native else (self.root / role).is_dir()


def layout(root: Path) -> Layout:
    return Layout(root, is_native(root))


# ---------------------------------------------------------------- generated files
# edit_scope.Rules keys the Agent's pre-check gets; the content rules stay with the judge.
AGENT_SCOPE_KEYS = ("allowed_paths", "deny_paths", "max_changed_files", "max_file_bytes",
                    "allowed_suffixes", "allow_empty", "allow_new_files", "allow_delete",
                    "allow_rename", "allow_executable")


def edit_scope_rules(task: dict, *, agent: bool = False) -> dict:
    """The task's edit_scope.Rules as a mapping (the patch cap is the archive cap)."""
    submission = task["submission"]
    keys = AGENT_SCOPE_KEYS if agent else (*AGENT_SCOPE_KEYS, "forbidden_added",
                                           "forbidden_content", "ignore_case")
    rules = {key: submission[key] for key in keys if key in submission}
    rules["max_patch_bytes"] = submission["max_patch_bytes"]
    return rules


def judge_config(task: dict) -> dict:
    """`config.yaml` for the judge (JSON is valid YAML): the edit-scope rules the shared
    verifier applies, the run limits, and any task-specific [judge.config] keys."""
    config = {
        "edit_scope": edit_scope_rules(task),
        "runtime": {"timeout_seconds": task["quota"]["eval_timeout_seconds"]},
        "evaluation": {
            "max_dev_submissions": task["quota"]["max_train_calls"],
            "max_train_wall_seconds": task["quota"]["max_train_wall_seconds"],
            "answering_timeout_seconds": task["quota"]["wall_time_seconds"],
            "solver_visible_metrics": visible_metrics(task),
            "max_metrics_bytes": task["verifier"]["max_metrics_bytes"],
        },
    }
    return _merge(config, task.get("judge", {}).get("config", {}))


def submission_policy(task: dict) -> dict:
    """`bin/submission-policy.json`: what the shared Agent submission tool checks before it
    writes a patch (the scope rules, never the judge's content rules)."""
    return {"edit_scope": edit_scope_rules(task, agent=True),
            "member": Path(task["submission"]["entrypoint"]).name}


def verifier_parameters(task: dict) -> dict:
    """`/opt/frontier/verifier.json`, which the shared verifier entry reads; for a Verifier-GPU
    task also `gpu`, which the shared GPU adapter reads: placement and runner cap, plus the
    per-mode lease ("runner") or the command test sandboxes ("control")."""
    parameters = {"entrypoint": task["submission"]["entrypoint"]}
    if "verifier" in task.get("gpu", {}):
        verifier = task["gpu"]["verifier"]
        gpu = {"placement": verifier["placement"],
               "max_gpu_instances": verifier["max_gpu_instances"]}
        if verifier["placement"] == CONTROL_PLACEMENT:
            gpu["sandboxes"] = {
                name: {"image": COMMAND_RUNNER_IMAGES[table["dockerfile"]][0],
                       "workdir": table["workdir"], "resources": dict(table["resources"]),
                       "max_concurrent": table["max_concurrent"],
                       "lease_seconds": dict(table["lease_seconds"])}
                for name, table in command_sandboxes(task).items()}
        else:
            gpu["lease_seconds"] = dict(verifier["lease_seconds"])
        parameters["gpu"] = gpu
    return parameters


def _dump(document: dict) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        return "{ " + ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items()) + " }"
    if value is None:
        return '""'
    raise TypeError(f"no TOML form for {type(value).__name__}")


def effective_toml(root: Path) -> str:
    """The complete effective definition of a task.toml task, every value annotated with its
    origin: like `helm template` or `git config --show-origin`. Generated, never committed."""
    task, origins = resolve(root)
    lines = [f"# Effective definition of {task['id']}, generated from task.toml by",
             "# scripts/task_manifest.py; do not edit. Each value names its origin:",
             "#   task.toml | default (tasks/shared/task-defaults.toml) | derived: <source>", ""]
    tables: dict[tuple, list[tuple[str, object, str]]] = {}
    for key, value in origins.items():
        leaf = _leaves(task).get(key)
        tables.setdefault(key[:-1], []).append((key[-1], leaf, value))
    width = max(len(f"{k} = {_toml_value(v)}") for rows in tables.values() for k, v, _ in rows)
    for table in sorted(tables, key=lambda t: (len(t) > 0, t)):
        if table:
            lines += ["", "[" + ".".join(json.dumps(p) if " " in p else p for p in table) + "]"]
        for key, value, origin in tables[table]:
            text = f"{key} = {_toml_value(value)}"
            lines.append(f"{text.ljust(width)}  # {origin}")
    if "sidecars" in task:
        roles = sorted({profile["image"] for profile in task["sidecars"].values()})
        lines += ["", f"# sidecars: {len(task['sidecars'])} (platform policy `sidecars`) on "
                  f"{len(roles)} images, derived: [test_sandboxes], datasets/"]
    return "\n".join(lines) + "\n"


def agent_readme(root: Path, task: dict | None = None) -> str:
    """/frontier/task/README.md: the task's own README.md (where to edit, its own tools) followed
    by the shared platform text (tasks/shared/agent/README.platform.md) with this task's limits
    filled in, so every task states the same platform rules with its actual numbers."""
    task = task or read_task(root)
    quota = task["quota"]
    platform = (SHARED / "agent/README.platform.md").read_text(encoding="utf-8").format(
        size_limits=(f"at most {task['submission']['max_bytes']:,} bytes"
                     if task["submission"]["max_patch_bytes"] == task["submission"]["max_bytes"]
                     else f"the patch at most {task['submission']['max_patch_bytes']:,} bytes, "
                          f"the tar at most {task['submission']['max_bytes']:,} bytes"),
        max_changed_files=(f"{task['submission']['max_changed_files']} file"
                           + ("" if task["submission"]["max_changed_files"] == 1 else "s")),
        file_rules="".join(
            f" It may not {what}." for what, allowed in (
                ("delete files", task["submission"]["allow_delete"]),
                ("add new files", task["submission"]["allow_new_files"]))
            if not allowed) + (
            " It may rename a file when both the old and the new path are editable."
            if task["submission"].get("allow_rename") else ""),
        train_calls=quota["max_train_calls"],
        train_wall_seconds=quota["max_train_wall_seconds"],
        eval_timeout_seconds=quota["eval_timeout_seconds"],
        wall_time_seconds=quota["wall_time_seconds"],
        cpus=f"{task['agent']['cpu_millis'] / 1000:g}",
        verifier_cpus=f"{task['verifier']['cpu_millis'] / 1000:g}",
        verifier_memory_gib=f"{task['verifier']['memory_bytes'] / 2**30:g}",
        memory_gib=f"{task['agent']['memory_bytes'] / 2**30:g}",
        workspace_gib=f"{task['agent']['workspace_bytes'] / 2**30:g}",
        feedback="\n".join(f"- `{name}`: {meaning}" for name, meaning in task["feedback"].items())
        or "- No task-specific metrics.",
    )
    own = (Path(root) / "README.md").read_text(encoding="utf-8").rstrip()
    text = own + "\n\n" + platform
    gpu = task.get("gpu", {}).get("agent")
    # The shared GPU half, only for tasks whose Agent has a GPU budget: an all-zero [gpu.agent]
    # (allowed_types only, which Management needs when the Verifier has a GPU) adds nothing.
    if gpu and gpu.get("max_instances", 0) > 0:
        audit = task["network"]["audit"]
        budgets = audit.get("ssh_session_budgets", {})
        text += "\n" + (SHARED / "agent/README.gpu.md").read_text(encoding="utf-8").format(
            gpu_type=" or ".join(gpu["allowed_types"]),
            max_instances=gpu["max_instances"],
            instance_seconds=gpu["max_instance_seconds"],
            gpu_seconds=gpu["max_gpu_seconds"],
            agent_ssh=budgets.get("agent", audit["max_ssh_sessions"]),
            train_ssh=budgets.get("verifier_train", 0),
            final_ssh=budgets.get("verifier_eval", 0),
            ssh_bytes=f"{audit['max_ssh_bytes_per_direction']:,}",
        )
    return text


def generated_files(root: Path) -> dict[str, bytes]:
    """Files the build adds under `.frontier-generated/` of a native task's build context."""
    task = read_task(root)
    return {
        f"{GENERATED}/effective-task.toml": effective_toml(root).encode(),
        f"{GENERATED}/judge/config.yaml": _dump(judge_config(task)),
        f"{GENERATED}/agent/submission-policy.json": _dump(submission_policy(task)),
        f"{GENERATED}/agent/README.md": agent_readme(root, task).encode(),
        f"{GENERATED}/verifier/verifier.json": _dump(verifier_parameters(task)),
    }


# ---------------------------------------------------------------- shared bases
def bases() -> dict:
    """`tasks/shared/bases.json`: the one current set of shared images, role -> digest-pinned
    reference (agent and verifier bases to build FROM, gpu-tools to COPY --from). Every task
    uses the current set; a new base is a new set for all tasks at once."""
    return json.loads((SHARED / "bases.json").read_text(encoding="utf-8"))


BASE_ROLES = ("agent", "verifier", "gpu-tools")


def base_reference(role: str) -> str:
    if role not in BASE_ROLES:
        raise ManifestError(f"no shared base role {role!r}")
    try:
        return bases()[role]
    except KeyError as error:
        raise ManifestError(f"tasks/shared/bases.json has no {role!r}") from error


# A task.toml recipe names a shared base by build argument (`ARG VERIFIER_BASE`, then
# `FROM ${VERIFIER_BASE}`); the build injects bases.json, so a bump is one bases.json change.
SHARED_BASE_ARGUMENTS = {"AGENT_BASE": "agent", "VERIFIER_BASE": "verifier",
                         "GPU_TOOLS": "gpu-tools"}
_BASE_ARGUMENT_FROM = re.compile(r"^\s*FROM\s+(?:--platform=\S+\s+)?\$\{([A-Z_]+)\}",
                                 re.IGNORECASE | re.MULTILINE)


def resolved_bases(root: Path) -> dict[str, str]:
    """role -> bases.json reference, for every shared base a task.toml task's recipes name by
    build argument. The recipe text no longer pins these, so the source identity binds them."""
    roles = set()
    for recipe in sorted((Path(root) / "docker").glob("*.Dockerfile")):
        for name in _BASE_ARGUMENT_FROM.findall(recipe.read_text(encoding="utf-8")):
            if name in SHARED_BASE_ARGUMENTS:
                roles.add(SHARED_BASE_ARGUMENTS[name])
    return {role: base_reference(role) for role in sorted(roles)}


_BUNDLES: dict[Path, Path] = {}


def judge_bundle(root: Path) -> Path:
    """The judge directory as the Verifier image holds it at /opt/frontier/task, for tests and
    tools that import a task's judge. A legacy task's is judge/ itself; a task.toml task's is
    a per-process copy of judge/ with the generated config.yaml and the starter at pristine/."""
    root = Path(root).resolve()
    if not is_native(root):
        return root / "judge"
    if root not in _BUNDLES:
        import shutil
        import tempfile
        bundle = Path(tempfile.mkdtemp(prefix=f"judge-{root.name}-")) / "task"
        shutil.copytree(root / "judge", bundle, symlinks=True)
        shutil.copytree(root / "repo", bundle / "pristine", symlinks=True)
        (bundle / "config.yaml").write_bytes(generated_files(root)[f"{GENERATED}/judge/config.yaml"])
        _BUNDLES[root] = bundle
    return _BUNDLES[root]


def main(arguments: list[str] | None = None) -> int:
    """`task_manifest.py show TASK_DIR`: print the effective definition with origins."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("command", choices=["show"])
    parser.add_argument("task_dir", type=Path)
    options = parser.parse_args(arguments)
    try:
        sys.stdout.write(effective_toml(options.task_dir.resolve()))
    except ManifestError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
