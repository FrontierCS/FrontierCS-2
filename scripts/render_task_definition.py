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
from pathlib import Path

from native_datasets import dataset_hashes

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "tasks/frontier-cs-2.0-demo"
DEFINITIONS = ROOT / "config/task-definitions.json"
CATALOG = ROOT / "config/task-catalog.json"
SCHEMA = "frontier.frontiersmith-task-policy/v1"
IDENTITY = "preview-source-identity/v1"
SPARSE = "sparse_video_attention_policy"


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


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


def catalog_task_names(catalog_path: Path = CATALOG) -> list[str]:
    """Read task membership in catalog order, rejecting ambiguous JSON."""

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


def catalog_problem_directories(
    names: list[str] | tuple[str, ...], problems: Path
) -> dict[str, Path]:
    directories = {name: problem_directory(problems, name) for name in names}
    for name, path in directories.items():
        if not path.is_dir():
            raise ValueError(f"catalog task {name} has no problem directory: {path}")
    for path in sorted(problems.iterdir()):
        if (
            path.is_dir()
            and path not in directories.values()
            and not any(task.parent == path for task in directories.values())
        ):
            print(f"{path}: ignored: not in config/task-catalog.json", file=sys.stderr)
    return directories


def check_tree(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(
                f"source must contain only regular files and directories: {path}"
            )


def source_sha256(root: Path) -> str:
    if "TAR_OPTIONS" in os.environ:
        raise ValueError("TAR_OPTIONS must be unset")
    check_tree(root)
    command = [
        "tar",
        "--format=gnu",
        "--sort=name",
        "--mtime=@0",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "--mode=u=rwX,go=rX",
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
    return "sha256:" + digest


def render(root: Path, name: str) -> dict:
    if "TAR_OPTIONS" in os.environ:
        raise ValueError("TAR_OPTIONS must be unset")
    if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        raise ValueError("invalid problem name")
    check_tree(root)
    policy = json.loads((root / "policy.json").read_text(encoding="utf-8"))
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
    for key in ("network", "gpu", "sidecars", "gpu_parent_resources"):
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
            directories = catalog_problem_directories(
                catalog_task_names(), TASK_ROOT / "problems"
            )
        else:
            if options.problem_dir is None or not options.name:
                parser.error("provide a problem directory and --name, or --all")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if options.all:
        rows = {name: render(directories[name], name) for name in sorted(directories)}
        DEFINITIONS.write_text(json_text(rows), encoding="utf-8")
    else:
        print(json_text(render(options.problem_dir, options.name)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
