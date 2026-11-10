#!/usr/bin/env python3
"""Render Preview definitions from native task trees, without network or Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from native_datasets import dataset_hashes
import source_tree
import task_layers
import task_manifest
from task_layers import json_text
from task_manifest import load_policy

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "frontier.frontiersmith-task-policy/v1"
IDENTITY = "preview-source-identity/v1"
SPARSE = "sparse_video_attention_policy"
# Hard-linked staging of a task's tracked files for its identity (same filesystem as the tree).
STAGING = ROOT / ".generated" / "source-identity"


KEY_NAME_RE = re.compile(r"[a-z][a-z0-9-]{0,62}")
MODEL_ALIAS_RE = re.compile(r"[a-z](?:[a-z0-9._-]{0,126}[a-z0-9])?")
KEY_ENVS = {"agent", "verifier-train", "verifier-eval"}
MAX_BUDGET_MICRO_USD = (1 << 64) - 1


def validate_additional_keys(value: object) -> list[dict]:
    """Check task-model key declarations against FrontierCS2 TaskConfig additional_keys.

    Mirrors docs/contract/task-additional-keys.md in the platform: unique names,
    non-empty unique aliases, positive USD budgets with at most six decimals, and
    non-empty unique envs. A malformed declaration is an error; the field is never
    dropped, because a task published without its keys cannot run.
    """
    if not isinstance(value, list) or not value:
        raise ValueError("additional_keys must be a non-empty list when present")
    names = set()
    for index, entry in enumerate(value):
        where = f"additional_keys[{index}]"
        if not isinstance(entry, dict) or set(entry) != {"name", "model_aliases", "budget_usd", "envs"}:
            raise ValueError(f"{where} must have exactly name, model_aliases, budget_usd, envs")
        name = entry["name"]
        if not isinstance(name, str) or KEY_NAME_RE.fullmatch(name) is None or name in names:
            raise ValueError(f"{where}.name must be a unique key name")
        names.add(name)
        aliases = entry["model_aliases"]
        if (not isinstance(aliases, list) or not aliases or len(set(aliases)) != len(aliases)
                or any(not isinstance(a, str) or MODEL_ALIAS_RE.fullmatch(a) is None for a in aliases)):
            raise ValueError(f"{where}.model_aliases must be non-empty unique model aliases")
        budget = entry["budget_usd"]
        if isinstance(budget, bool) or not isinstance(budget, (int, float)):
            raise ValueError(f"{where}.budget_usd must be a number")
        micro = round(budget * 1_000_000)
        if not 0 < budget or abs(micro - budget * 1_000_000) > 1e-6 or micro > MAX_BUDGET_MICRO_USD:
            raise ValueError(f"{where}.budget_usd must be positive with at most six decimals")
        envs = entry["envs"]
        if (not isinstance(envs, list) or not envs or len(set(envs)) != len(envs)
                or any(e not in KEY_ENVS for e in envs)):
            raise ValueError(f"{where}.envs must be non-empty unique values of {sorted(KEY_ENVS)}")
    return value


def catalog_task_names(catalog_path: Path | None = None) -> list[str]:
    """Task membership of the catalog at `catalog_path`, or of every task layer's catalog, in
    order; ambiguous JSON and a task in two layers are errors."""
    if catalog_path is not None:
        return _catalog_file_names(catalog_path)
    names = [name for layer in task_layers.layers(ROOT)
             for name in _catalog_file_names(layer.catalog)]
    if len(set(names)) != len(names):
        raise ValueError("a task is in more than one layer's catalog")
    return names


def _catalog_file_names(catalog_path: Path) -> list[str]:

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        catalog = json.loads(
            Path(catalog_path).read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
        if not isinstance(catalog, dict) or not isinstance(catalog.get("tasks"), list):
            raise ValueError("tasks must be a list")
        if not catalog["tasks"]:
            raise ValueError("task catalog lists no tasks")
        names = []
        for task in catalog["tasks"]:
            name = task.get("name") if isinstance(task, dict) else None
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
            ):
                raise ValueError("every task must have a valid name")
            if name in names:
                raise ValueError(f"duplicate task name: {name}")
            names.append(name)
        return names
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read task catalog {catalog_path}: {exc}") from exc


def problem_directory(problems: Path, name: str) -> Path:
    """Resolve a flat or one-domain-deep task without changing its identity."""
    if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        raise ValueError("invalid problem name")
    candidates = [
        path
        for path in [problems / name, *sorted(problems.glob(f"*/{name}"))]
        if path.is_dir()
    ]
    if len(candidates) > 1:
        raise ValueError(f"ambiguous problem directories for {name}: {candidates}")
    return candidates[0] if candidates else problems / name


def task_directory(name: str, root: Path = ROOT) -> Path:
    """The directory of task `name` in `root`'s task layers (task_layers), flat or one domain
    deep; a task found in two layers is an error."""
    found = [path for layer in task_layers.layers(root) if layer.problems.is_dir()
             for path in [problem_directory(layer.problems, name)] if path.is_dir()]
    if len(found) != 1:
        raise ValueError(f"task {name} must have one directory in the task layers: {found}")
    return found[0]


def catalog_problem_directories(
    names: list[str] | tuple[str, ...], root: Path = ROOT
) -> dict[str, Path]:
    """Each catalog task's directory; a directory of every layer's problems that holds no
    catalog task is reported and ignored."""
    directories = {}
    for name in names:
        try:
            directories[name] = task_directory(name, root)
        except ValueError as error:
            raise ValueError(f"catalog task {name} has no problem directory: {error}") from error
    for layer in task_layers.layers(root):
        for path in sorted(layer.problems.iterdir()) if layer.problems.is_dir() else ():
            if (
                path.is_dir()
                and path not in directories.values()
                and not any(task.parent == path for task in directories.values())
            ):
                print(f"{path}: ignored: not in the task catalog", file=sys.stderr)
    return directories


def check_tree(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(
                f"source must contain only regular files and directories: {path}"
            )


def tree_sha256(root: Path) -> str:
    """sha256 of a canonical GNU tar of the directory as it is on disk."""
    check_tree(root)
    command = [
        "tar",
        "--format=gnu",
        "--sort=name",
        "--mtime=@0",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        # a-st: a setgid/sticky parent (directory bits survive the = forms) must not change it.
        "--mode=u=rwX,go=rX,a-st",
        "--hard-dereference",
        "--blocking-factor=20",
        "-C",
        str(root),
        "-cf",
        "-",
        ".",
    ]
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, env={**os.environ, "LC_ALL": "C", "TZ": "UTC"}
    ) as archive:
        assert archive.stdout is not None
        with archive.stdout:
            digest = hashlib.file_digest(archive.stdout, "sha256").hexdigest()
        if archive.wait() != 0:
            raise ValueError("source identity tar failed")
    return digest


def source_sha256(root: Path) -> str:
    """The task's source identity: a canonical tar of its git-tracked files (source_tree; an
    untracked or ignored file under the task fails), plus for a task.toml task its generated
    files and resolved shared bases."""
    if "TAR_OPTIONS" in os.environ:
        raise ValueError("TAR_OPTIONS must be unset")
    if source_tree.in_work_tree(root):
        STAGING.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=STAGING) as directory:
            staged = Path(directory) / "tree"
            source_tree.stage(root, staged, link=True)
            digest = tree_sha256(staged)
    else:
        digest = tree_sha256(root)
    if task_manifest.is_native(root):
        # A task.toml task's images also hold files generated from task.toml and the shared
        # defaults (tasks/shared/task-defaults.toml); bind them, so a default that changes
        # what a build produces also changes the source identity and image tag.
        combined = hashlib.sha256(f"tree {digest}\n".encode())
        for name, body in sorted(task_manifest.generated_files(root).items()):
            combined.update(f"{name} {hashlib.sha256(body).hexdigest()}\n".encode())
        # Shared bases a recipe takes by build argument are injected from bases.json at build
        # time; bind the resolved references, so a base bump moves the image tag too.
        for role, reference in task_manifest.resolved_bases(root).items():
            combined.update(f"base {role} {reference}\n".encode())
        digest = combined.hexdigest()
    return "sha256:" + digest


def render(root: Path, name: str) -> dict:
    if "TAR_OPTIONS" in os.environ:
        raise ValueError("TAR_OPTIONS must be unset")
    if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        raise ValueError("invalid problem name")
    check_tree(root)
    policy = load_policy(root)
    if policy.get("schema_version") != SCHEMA:
        raise ValueError("unsupported native policy schema_version")
    required = ("quota", "agent", "verifier", "submission", "metadata")
    if any(not isinstance(policy.get(key), dict) for key in required):
        raise ValueError("missing quota/agent/verifier/submission/metadata.datasets")
    if "workspace_bytes" in policy["verifier"]:
        budget = policy["verifier"]["workspace_bytes"]
        if type(budget) is not int or not 1 <= budget <= 1 << 40:
            raise ValueError(
                "verifier.workspace_bytes must be an integer in 1..=1099511627776"
            )
    if not isinstance(policy["metadata"].get("datasets"), dict):
        raise ValueError("missing metadata.datasets")
    extension, datasets = dataset_hashes(root, policy)
    row = {key: policy[key] for key in (*required, "family", "display_name", "version")}
    row.update(
        {
            "task_id": "frontier-cs-2.0-" + name.replace("_", "-"),
            "datasets": datasets,
            "dataset_extension": extension,
            "policy_schema_version": SCHEMA,
            "verifier": {
                **policy["verifier"],
                "evidence_path": "/frontier/output/evidence.tar.zst",
            },
        }
    )
    for key in ("network", "gpu", "sidecars", "gpu_parent_resources", "verifier_cpu",
                "cpu_runner_images"):
        if key in policy:
            row[key] = policy[key]
    if "additional_keys" in policy:
        row["additional_keys"] = validate_additional_keys(policy["additional_keys"])
    digest = source_sha256(root)
    row["source"] = {"source_set_sha256": digest, "identity_algorithm": IDENTITY}
    row["image_tag"] = "source-" + digest.removeprefix("sha256:")
    return row


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem_dir", type=Path, nargs="?")
    parser.add_argument("--name")
    parser.add_argument("--all", action="store_true")
    options = parser.parse_args(arguments)
    try:
        if options.all:
            if options.problem_dir or options.name:
                parser.error(
                    "--all cannot be combined with a problem directory or --name"
                )
            directories = catalog_problem_directories(catalog_task_names())
        else:
            if options.problem_dir is None or not options.name:
                parser.error("provide a problem directory and --name, or --all")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if options.all:
        rows = {name: render(directories[name], name) for name in sorted(directories)}
        for layer in task_layers.layers(ROOT):
            names = set(_catalog_file_names(layer.catalog))
            layer.definitions.write_text(
                json_text({name: row for name, row in rows.items() if name in names}),
                encoding="utf-8",
            )
    else:
        print(json_text(render(options.problem_dir, options.name)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
