#!/usr/bin/env python3
"""Keep the per-task metadata in tasks/metadata/ consistent with what the platform enforces.

Each task has one flat file, tasks/metadata/<id>.yaml. Every line is
``key: <JSON value>``, which is valid YAML and loads without a YAML library.
The schema is flat with consistent types so the records can be published as a
dataset (for example as JSON Lines).

Hand-written fields: summary, paper_title, paper_venue, paper_url.
Every other field is derived from config/task-definitions.json (rendered from
each task's policy.json) and the task's GPU verifier image recipe. Task trees
are never edited: they are bound to the published images by source identity.

    python3 scripts/task_metadata.py write   # refresh derived fields, keep hand fields
    python3 scripts/task_metadata.py check   # fail if any file is stale or missing
    python3 scripts/task_metadata.py table   # rewrite TASKS.md from the metadata
    python3 scripts/task_metadata.py jsonl   # print all records as JSON Lines
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = ROOT / "config/task-definitions.json"
CATALOG = ROOT / "config/task-catalog.json"
PROBLEMS = ROOT / "tasks/frontier-cs-2.0-demo/problems"
METADATA = ROOT / "tasks/metadata"
TABLE = ROOT / "TASKS.md"
GIB = 1 << 30

HAND_FIELDS = ("summary", "paper_title", "paper_venue", "paper_url")
FIELDS = (
    "id",
    "display_name",
    "category",
    "summary",
    "paper_title",
    "paper_venue",
    "paper_url",
    "agent_cpu_cores",
    "agent_memory_gib",
    "agent_disk_gib",
    "agent_gpu_type",
    "agent_gpu_count",
    "agent_gpu_seconds",
    "verifier_cpu_cores",
    "verifier_memory_gib",
    "verifier_disk_gib",
    "verifier_gpu_type",
    "verifier_gpu_count",
    "verifier_gpu_seconds",
    "agent_model_api",
    "task_model_keys",
    "task_model_budget_usd",
    "network_domains",
    "agent_wall_time_seconds",
    "train_calls",
    "train_wall_seconds",
    "eval_timeout_seconds",
)
AGENT_MODEL_API = (
    "candidate"  # the Agent's model comes from the Candidate's provider route
)


def number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else round(value, 3)


def task_directory(name: str) -> Path:
    matches = sorted(PROBLEMS.glob(f"*/{name}"))
    if len(matches) != 1:
        raise SystemExit(f"{name}: expected one problem directory, found {matches}")
    return matches[0]


def verifier_gpu_type(directory: Path) -> str | None:
    """GPU type from the verifier-gpu recipe, else from the GPU requests the
    CPU verifier sends to the Driver (tasks that orchestrate their own GPUs)."""
    types = set()
    for path in sorted((directory / "verifier-gpu").glob("**/*")):
        if path.is_file() and path.stat().st_size < 1 << 20:
            text = path.read_text(encoding="utf-8", errors="ignore")
            types.update(re.findall(r'VERIFIER_GPU_TYPE="?([A-Za-z0-9-]+)', text))
    if not types:
        for path in sorted((directory / "verifier").glob("**/*.py")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            types.update(re.findall(r"""['"]gpu_type['"]\s*:\s*['"]([A-Za-z0-9-]+)['"]""", text))
    if len(types) > 1:
        raise SystemExit(
            f"{directory.name}: ambiguous verifier GPU types {sorted(types)}"
        )
    return types.pop() if types else None


def derived(name: str, definition: dict) -> dict:
    directory = task_directory(name)
    agent, verifier, quota = (
        definition["agent"],
        definition["verifier"],
        definition["quota"],
    )
    gpu = definition.get("gpu") or {}
    agent_gpu = gpu.get("agent") or {}
    verifier_gpu = gpu.get("verifier") or {}
    agent_types = agent_gpu.get("allowed_types") or []
    keys = definition.get("additional_keys") or []
    record = {
        "id": name,
        "display_name": definition["display_name"],
        "category": directory.parent.name,
        "agent_cpu_cores": number(agent["cpu_millis"] / 1000),
        "agent_memory_gib": number(agent["memory_bytes"] / GIB),
        "agent_disk_gib": number(agent["workspace_bytes"] / GIB)
        if "workspace_bytes" in agent
        else None,
        "agent_gpu_type": "/".join(agent_types)
        if agent_types and agent_gpu.get("max_instances", 0)
        else None,
        "agent_gpu_count": agent_gpu.get("max_instances", 0),
        "agent_gpu_seconds": agent_gpu.get("max_gpu_seconds", 0),
        "verifier_cpu_cores": number(verifier["cpu_millis"] / 1000),
        "verifier_memory_gib": number(verifier["memory_bytes"] / GIB),
        "verifier_disk_gib": number(verifier["workspace_bytes"] / GIB)
        if "workspace_bytes" in verifier
        else None,
        "verifier_gpu_type": verifier_gpu_type(directory)
        if verifier_gpu.get("max_gpu_instances")
        else None,
        "verifier_gpu_count": verifier_gpu.get("max_gpu_instances", 0),
        "verifier_gpu_seconds": verifier_gpu.get("max_gpu_seconds", 0),
        "agent_model_api": AGENT_MODEL_API,
        "task_model_keys": sorted(
            {alias for key in keys for alias in key["model_aliases"]}
        ),
        "task_model_budget_usd": number(sum(key["budget_usd"] for key in keys)),
        "network_domains": sorted(
            definition.get("network", {}).get("allowed_domains", [])
        ),
        "agent_wall_time_seconds": quota["wall_time_seconds"],
        "train_calls": quota["max_train_calls"],
        "train_wall_seconds": quota["max_train_wall_seconds"],
        "eval_timeout_seconds": quota["eval_timeout_seconds"],
    }
    if record["verifier_gpu_count"] and record["verifier_gpu_type"] is None:
        raise SystemExit(f"{name}: GPU verifier image declares no VERIFIER_GPU_TYPE")
    return record


def dump(record: dict) -> str:
    return "".join(
        f"{key}: {json.dumps(record[key], ensure_ascii=False)}\n" for key in FIELDS
    )


def load(path: Path) -> dict:
    record = {}
    for number_, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(": ")
        if not separator or key not in FIELDS or key in record:
            raise SystemExit(
                f"{path}:{number_}: expected a unique known 'key: value' line"
            )
        record[key] = json.loads(value)
    if tuple(record) != FIELDS:
        raise SystemExit(
            f"{path}: fields must be exactly, in order: {', '.join(FIELDS)}"
        )
    return record


def catalog_names() -> list[str]:
    return [task["name"] for task in json.loads(CATALOG.read_text())["tasks"]]


def expected_records(existing: dict[str, dict]) -> dict[str, dict]:
    definitions = json.loads(DEFINITIONS.read_text())
    records = {}
    for name in catalog_names():
        hand = {key: (existing.get(name) or {}).get(key) for key in HAND_FIELDS}
        record = derived(name, definitions[name])
        records[name] = {
            key: hand[key] if key in HAND_FIELDS else record[key] for key in FIELDS
        }
    return records


def existing_records() -> dict[str, dict]:
    return {path.stem: load(path) for path in sorted(METADATA.glob("*.yaml"))}


def short(value: float) -> str:
    """Two decimals at most, without trailing zeros (0.8333 -> 0.83, 5.0 -> 5)."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def hours(seconds: int) -> str:
    return f"{short(seconds / 3600)} h"


def gpu_cell(kind: str | None, count: int, seconds: int) -> str:
    if not count:
        return "-"
    return f"{count}x {kind}, {short(seconds / 3600)} GPU-h"


# Full domain names for TASKS.md; `category` itself stays the task's folder name.
CATEGORY_NAMES = {"Arch": "Architecture", "Crypto": "Cryptography", "Programming Language": "Programming Languages"}


def table(records: dict[str, dict]) -> str:
    lines = [
        "# FrontierCS 2 tasks",
        "",
        "Generated by `python3 scripts/task_metadata.py table` from `tasks/metadata/`.",
        "Resources are the limits the platform enforces for each run: CPU cores / memory GiB /",
        "workspace GiB, GPU type and total GPU-hours, the Agent's answering time, the number",
        "of train (development) evaluations and the final evaluation timeout.",
        "",
        "| Task | Category | Paper | Agent CPU / mem / disk | Agent GPU | Verifier CPU / mem | Verifier GPU | Agent time | Train calls | Eval timeout |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    ordered = sorted(records.values(), key=lambda r: (CATEGORY_NAMES.get(r["category"], r["category"]), r["display_name"]))
    for r in ordered:
        if r["paper_title"]:
            paper = r["paper_title"].split(":")[0]
            paper = f"[{paper}]({r['paper_url']})" if r["paper_url"] else paper
            if r["paper_venue"]:
                paper += f" ({r['paper_venue']})"
        else:
            paper = "-"
        disk = "-" if r["agent_disk_gib"] is None else f"{r['agent_disk_gib']:g}"
        lines.append(
            f"| `{r['id']}`<br>{r['display_name']} | {CATEGORY_NAMES.get(r['category'], r['category'])} | {paper} | "
            f"{r['agent_cpu_cores']:g} / {r['agent_memory_gib']:g} / {disk} | "
            f"{gpu_cell(r['agent_gpu_type'], r['agent_gpu_count'], r['agent_gpu_seconds'])} | "
            f"{r['verifier_cpu_cores']:g} / {r['verifier_memory_gib']:g} | "
            f"{gpu_cell(r['verifier_gpu_type'], r['verifier_gpu_count'], r['verifier_gpu_seconds'])} | "
            f"{hours(r['agent_wall_time_seconds'])} | {r['train_calls']} | "
            f"{short(r['eval_timeout_seconds'] / 60)} min |"
        )
    gpu = sum(1 for r in ordered if r["agent_gpu_count"] or r["verifier_gpu_count"])
    keyed = [r for r in ordered if r["task_model_keys"]]
    lines += [
        "",
        f"{len(ordered)} tasks: {len(ordered) - gpu} CPU-only, {gpu} using Modal GPUs.",
        "The Agent's model calls go through the Candidate's provider route.",
    ]
    if not keyed:
        lines.append("No task needs model keys of its own (`task_model_keys` is empty for every task).")
    for r in keyed:
        lines.append(
            f"`{r['id']}` also uses a task model: the site must serve the LiteLLM alias "
            + ", ".join(f"`{alias}`" for alias in r["task_model_keys"])
            + f" (`features.task_model_aliases`), with a total key budget of ${short(r['task_model_budget_usd'])} per run."
        )
    lines.append("")
    return "\n".join(lines)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("write", "check", "table", "jsonl"))
    options = parser.parse_args(arguments)
    existing = existing_records()
    records = expected_records(existing)
    if options.command == "write":
        METADATA.mkdir(exist_ok=True)
        for path in METADATA.glob("*.yaml"):
            if path.stem not in records:
                path.unlink()
        for name, record in records.items():
            (METADATA / f"{name}.yaml").write_text(dump(record), encoding="utf-8")
        return 0
    problems = []
    if options.command in ("check", "table"):
        for name in sorted(set(existing) | set(records)):
            if name not in records:
                problems.append(
                    f"{name}: metadata for a task that is not in the catalog"
                )
            elif name not in existing:
                problems.append(f"{name}: missing tasks/metadata/{name}.yaml")
            elif existing[name] != records[name]:
                changed = [k for k in FIELDS if existing[name][k] != records[name][k]]
                problems.append(
                    f"{name}: stale fields {changed}; run `task_metadata.py write`"
                )
            elif not existing[name]["summary"]:
                problems.append(f"{name}: summary is empty")
        if options.command == "check" and TABLE.exists() and not problems:
            if TABLE.read_text(encoding="utf-8") != table(existing):
                problems.append("TASKS.md is stale; run `task_metadata.py table`")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    if options.command == "table":
        TABLE.write_text(table(existing), encoding="utf-8")
    elif options.command == "jsonl":
        for name in records:
            print(json.dumps(existing[name], ensure_ascii=False))
    else:
        print(f"Task metadata: PASS ({len(records)} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
