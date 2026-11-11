#!/usr/bin/env python3
"""Validate Preview's public task sources and their complete source inventories.

This check uses the bundled policies, Dockerfiles and locks. It needs neither
FrontierSmith's private pipeline nor image/registry/cloud credentials.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import image_namespace
import task_layers
import task_manifest
from native_datasets import check_tar_dataset, dataset_hashes, dataset_path

DEFAULT_PROBLEMS_ROOT = Path(__file__).resolve().parents[1] / "tasks/problems"
# Reviewed public-copy exceptions, keyed by (task, image role). None are needed
# by the current catalog; the mechanism stays for imported tasks.
PUBLIC_COPY_REVIEWS: dict[tuple[str, str], Path] = {}
FEEDBACK_STATEMENT_REVIEWS = {
    "ckks_composable_noise": (
        Path(__file__).resolve().parent
        / "reference/ckks-feedback-statement-reviewed-v1.json"
    ),
}
# Every workload measures and gates every declared metric. These tasks still declare
# workloads that carry only part of the metric set and are reported as
# warnings until their own fixes land. Any other such task fails, and an entry
# must be removed once its task complies.
NONUNIFORM_WORKLOAD_METRICS_EXEMPT: dict[str, str] = {}


class LayoutFailure(ValueError):
    pass


def warn(root: Path, message: str) -> None:
    print(f"WARN {root.name}: {message}", file=sys.stderr)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LayoutFailure(message)


def regular(path: Path) -> Path:
    require(path.is_file() and not path.is_symlink(), f"missing regular file: {path}")
    return path


def relative_path(value: object) -> str:
    require(
        isinstance(value, str)
        and bool(value)
        and not value.startswith("/")
        and all(part not in {"", ".", ".."} for part in value.split("/")),
        f"unsafe source or payload path: {value!r}",
    )
    return value


def document(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(regular(path).read_text(), object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LayoutFailure(f"invalid JSON document: {path}") from error
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def digest(path: Path) -> str:
    with regular(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def copy_source(root: Path, value: str, role: str = "") -> str:
    if task_manifest.is_native(root) and value.startswith(
        task_manifest.GENERATED + "/"
    ):
        return (
            value  # written by the build from task.toml (task_manifest.generated_files)
        )
    # Docker accepts a directory's trailing slash. Do not normalize traversal,
    # repeated slashes, symlinks, or a slash appended to a regular file.
    trailing = value.endswith("/")
    source = relative_path(value[:-1] if trailing else value)
    path = root
    for component in source.split("/"):
        path = path / component
        require(not path.is_symlink(), f"symlink COPY source: {source}")
    label = f"{role} " if role else ""
    require(path.is_file() or path.is_dir(), f"missing {label}COPY source: {source}")
    require(
        not trailing or path.is_dir(), f"COPY trailing slash needs directory: {source}"
    )
    return source


def check_reviewed_public_copies(root: Path, role: str, sources: set[str]) -> None:
    """Bind exceptional public inputs to a reviewed recipe and complete bytes.

    This is an Agent visibility review, not a generic runtime-context allowlist
    or an assertion of image publication rights. The full task inventory is
    still independently checked by validate_task.
    """
    review_path = PUBLIC_COPY_REVIEWS.get((root.name, role))
    require(
        review_path is not None, f"non-public Agent COPY sources: {sorted(sources)}"
    )
    review = document(review_path)
    require(
        review.get("schema") == "preview.reviewed-public-copies/v1"
        and review.get("problem_id") == root.name
        and review.get("role") == role,
        "public COPY review identity mismatch",
    )
    bindings = review.get("bindings", {})
    require(
        set(bindings) == {"upstream.lock.json", "policy.json", f"{role}/Dockerfile"},
        "public COPY review needs lock, policy and recipe bindings",
    )
    for name, expected in bindings.items():
        require(
            digest(root / name) == expected, f"public COPY binding mismatch: {name}"
        )
    copies = review.get("copies", {})
    require(set(copies) == sources, "unreviewed public COPY source set")
    observed_files, observed_directories = set(), set()
    for source, kind in copies.items():
        path = root / copy_source(root, source)
        require(kind in {"file", "directory"}, "invalid reviewed COPY kind")
        require(
            path.is_dir() == (kind == "directory"),
            f"public COPY kind mismatch: {source}",
        )
        members = [path, *path.rglob("*")] if path.is_dir() else [path]
        for member in members:
            require(
                not member.is_symlink() and (member.is_file() or member.is_dir()),
                f"non-regular public COPY member: {member}",
            )
            target = observed_files if member.is_file() else observed_directories
            target.add(member.relative_to(root).as_posix())
    records = review.get("files", {})
    require(set(records) == observed_files, "public COPY file inventory mismatch")
    require(
        set(review.get("directories", [])) == observed_directories,
        "public COPY directory inventory mismatch",
    )
    for name, record in records.items():
        path = root / relative_path(name)
        require(
            path.stat().st_size == record.get("bytes")
            # Match Git/source-identity executable semantics, not the author's
            # umask (0664 and a fresh checkout's 0644 describe the same source).
            and bool(path.stat().st_mode & 0o111) == record.get("executable")
            and digest(path) == record.get("sha256"),
            f"reviewed public COPY bytes or executable bit mismatch: {name}",
        )


def check_dockerfile(root: Path, role: str) -> None:
    paths = task_manifest.layout(root)
    if paths.native:
        check_native_dockerfile(root, role)
        return
    recipe = regular(root / role / "Dockerfile").read_text()
    if role == "agent-base":
        require(
            "agent-base/frontier/task" in recipe,
            "Agent must copy its native public task tree",
        )
    exceptional = set()
    for line in recipe.splitlines():
        if not line.startswith("COPY "):
            continue
        fields = shlex.split(line)[1:]
        if any(field.startswith("--from=") for field in fields):
            continue
        fields = [field for field in fields if not field.startswith("--")]
        require(len(fields) == 2, f"use one task-root source per COPY in {role}")
        source = copy_source(root, fields[0], role)
        if role.startswith("agent"):
            public = (
                source == "agent-base/frontier/task"
                or source.startswith("agent-base/frontier/task/")
                or source == "agent-gpu/task_runtime.py"
            )
            if not public:
                exceptional.add(source)
    if exceptional:
        check_reviewed_public_copies(root, role, exceptional)


NATIVE_BASE_ROLE = {"agent-base": "agent", "verifier": "verifier"}
NATIVE_AGENT_SOURCES = ("README.md", "repo", "starter", "bin")
NATIVE_GENERATED_AGENT = f"{task_manifest.GENERATED}/agent/"
# COPY --from=<registry>/<repository>@sha256:<64 hex>: task-private judge data or toolchains,
# pinned by digest; Verifier recipes only.
PINNED_COPY_ROLES = frozenset({"verifier", "verifier-gpu"})
PINNED_IMAGE = re.compile(
    r"(?:[a-z0-9-]+\.)+[a-z0-9-]+(?::[0-9]+)?(?:/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)+"
    r"@sha256:[0-9a-f]{64}"
)


def check_no_platform_numbers(root: Path) -> None:
    """A task.toml task's statement and own README never restate platform numbers: the generated
    /frontier/task/README.md states them from task.toml, so a copy elsewhere could only drift."""
    task = task_manifest.read_task(root)
    quota, submission = task["quota"], task["submission"]

    def forms(value: int) -> str:
        return rf"(?:{value}|{value:,})"

    patterns = {
        "answering window": rf"\b{forms(quota['wall_time_seconds'])}\s*(?:-\s*)?s(?:ec(?:ond)?s?)?\b",
        "train calls": rf"\b{quota['max_train_calls']}\s+(?:accepted\s+)?train\s+calls?\b",
        "train wall time": rf"\b{forms(quota['max_train_wall_seconds'])}\s*(?:-\s*)?s(?:ec(?:ond)?s?)?\b",
        "evaluation timeout": rf"\b{forms(quota['eval_timeout_seconds'])}\s*(?:-\s*)?s(?:ec(?:ond)?s?)?\b",
        "submission size": rf"\b{forms(submission['max_bytes'])}\s*(?:-\s*)?bytes?\b",
    }
    for name in ("README.md", "repo/README.md"):
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for what, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            require(
                match is None,
                f"{name} restates the {what} ({match.group(0) if match else ''}); the generated "
                "/frontier/task/README.md states platform numbers, so refer to it instead",
            )


# The statement template (repo/README.md of a task.toml task): one title, then these sections.
STATEMENT_SECTIONS = (
    "Task",
    "What you may change",
    "Interface",
    "Correctness",
    "Workload",
    "Scoring",
)
STATEMENT_OPTIONAL_SECTIONS = ("Running locally", "Environment", "Notes")
STATEMENT_MAX_WORDS = 1600
STATEMENT_WARN_WORDS = 1200
# The shared generated README states the objective and scoring rules; a task README never does.
TASK_README_FORBIDDEN_SECTION = re.compile(r"scor|objective|priorit", re.IGNORECASE)


def _markdown_headings(text: str) -> list[tuple[int, str]]:
    """(level, title) of every ATX heading outside fenced code blocks."""
    headings, fence = [], None
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = None if fence == marker else (fence or marker)
            continue
        if fence is None:
            match = re.match(r"(#{1,6})\s+(.*?)\s*#*\s*$", line)
            if match:
                headings.append((len(match.group(1)), match.group(2)))
    return headings


def check_statement_structure(root: Path) -> None:
    """A task.toml task's statement follows the statement template: exactly one title, the
    required sections in order, then only the optional ones in order; at most
    STATEMENT_MAX_WORDS words. Its task README has no scoring, objective or priority section."""
    paths = task_manifest.layout(root)
    text = regular(paths.repo / "README.md").read_text(encoding="utf-8")
    headings = _markdown_headings(text)
    require(
        sum(1 for level, _ in headings if level == 1) == 1,
        "repo/README.md must have exactly one '# ' title",
    )
    sections = [title for level, title in headings if level == 2]
    required = list(STATEMENT_SECTIONS)
    extra = sections[len(required) :]
    optional = [name for name in STATEMENT_OPTIONAL_SECTIONS if name in extra]
    require(
        sections[: len(required)] == required and extra == optional,
        "repo/README.md sections must be, in order: "
        + ", ".join(required)
        + ", then optionally "
        + ", ".join(STATEMENT_OPTIONAL_SECTIONS)
        + f"; found {sections}",
    )
    scoring = text.split("\n## Scoring", 1)[1].split("\n## ", 1)[0]
    require(
        re.search(r"(?:^|\n\s*\n)\*\*Beats the reference\*\*", scoring) is not None,
        "repo/README.md's Scoring section needs a paragraph starting with "
        "'**Beats the reference**' (the generated README's Objective points to it)",
    )
    words = len(text.split())
    require(
        words <= STATEMENT_MAX_WORDS,
        f"repo/README.md has {words} words; the limit is {STATEMENT_MAX_WORDS}",
    )
    if words > STATEMENT_WARN_WORDS:
        warn(
            root,
            f"repo/README.md has {words} words (target at most {STATEMENT_WARN_WORDS})",
        )
    own = regular(root / "README.md").read_text(encoding="utf-8")
    for _, title in _markdown_headings(own):
        require(
            TASK_README_FORBIDDEN_SECTION.search(title) is None,
            f"README.md has a {title!r} section; scoring and the objective are stated by "
            "the statement and the generated README",
        )


def base_argument(name: str) -> str:
    return "${" + name + "}"


ROLE_ARGUMENT = {
    role: name for name, role in task_manifest.SHARED_BASE_ARGUMENTS.items()
}


def check_shared_base_arguments(path: Path, recipe: str) -> dict[str, str]:
    """A recipe names the shared bases only by build argument (`ARG VERIFIER_BASE`, then
    `FROM ${VERIFIER_BASE}`); the build injects tasks/shared/bases.json. Returns stage alias ->
    shared role for the `FROM ${...} AS <stage>` stages."""
    code = "\n".join(
        line for line in recipe.splitlines() if not line.lstrip().startswith("#")
    )
    literal = SHARED_IMAGE_REFERENCE.search(code)
    require(
        literal is None,
        f"{path.name} names {literal.group(0) if literal else ''} by reference; take a shared "
        "base by build argument (ARG AGENT_BASE / VERIFIER_BASE / GPU_TOOLS, then "
        "FROM ${NAME}), which the build injects from tasks/shared/bases.json",
    )
    lines = code.splitlines()
    first = next(
        (i for i, line in enumerate(lines) if line.startswith("FROM ")), len(lines)
    )
    declared = {
        line.split()[1]
        for line in lines[:first]
        if line.startswith("ARG ") and len(line.split()) == 2
    }
    stages = {}
    for line in lines:
        if not line.startswith("FROM "):
            continue
        words = line.split()
        name = next(
            (
                n
                for n in task_manifest.SHARED_BASE_ARGUMENTS
                if words[1] == base_argument(n)
            ),
            None,
        )
        if name is None:
            continue
        require(
            name in declared,
            f"{path.name} uses ${{{name}}} without a global `ARG {name}` (no default) "
            "before its first FROM",
        )
        if len(words) == 4 and words[2] == "AS":
            stages[words[3]] = task_manifest.SHARED_BASE_ARGUMENTS[name]
    return stages


def check_native_dockerfile(root: Path, role: str) -> None:
    """A task.toml task's recipe takes the shared bases by build argument (its final stage FROM
    its role's base) and the Agent image copies only the statement, the starter repo and
    generated Agent files."""
    path = task_manifest.layout(root).dockerfile(role)
    recipe = regular(path).read_text()
    task_manifest.read_task(root)
    froms = [
        line.split()[1] for line in recipe.splitlines() if line.startswith("FROM ")
    ]
    require(bool(froms), f"{path.name} has no FROM")
    shared_stages = check_shared_base_arguments(path, recipe)
    shared = {base_argument(name) for name in task_manifest.SHARED_BASE_ARGUMENTS}
    if role in NATIVE_BASE_ROLE:
        expected = base_argument(ROLE_ARGUMENT[NATIVE_BASE_ROLE[role]])
        # The image is the last stage; earlier stages only name other shared bases.
        require(
            froms[-1] == expected,
            f"{path.name} must build its final stage FROM {expected}",
        )
        require(
            all(ref in shared for ref in froms[:-1]),
            f"{path.name}: an earlier stage may only be FROM a shared base build argument",
        )
    # The other stages a task.toml recipe may copy from: the shared gpu-tools stage, and for a
    # Verifier recipe the shared verifier base stage (the shared verifier modules).
    allowed_roles = {"gpu-tools"} | (
        {"verifier"} if role.startswith("verifier") else set()
    )
    tools_stages = {
        stage for stage, base in shared_stages.items() if base in allowed_roles
    }
    own_stages = {
        line.split()[3]
        for line in recipe.splitlines()
        if line.startswith("FROM ")
        and len(line.split()) == 4
        and line.split()[2] == "AS"
    }
    for line in recipe.splitlines():
        if not line.startswith("COPY "):
            continue
        stage = next(
            (
                f.removeprefix("--from=")
                for f in shlex.split(line)
                if f.startswith("--from=")
            ),
            None,
        )
        if stage is not None:
            # A Verifier recipe may also copy task-private judge data or toolchains from an
            # image named by full repository and digest (never a tag); an Agent recipe never.
            # An Agent GPU recipe may carry over public Agent-side data from the task's own
            # published Agent images (never a Verifier image), pinned by digest.
            own_agent = re.compile(
                re.escape(image_namespace.repository(f"fcs2-task-{root.name}"))
                + r"-agent(?:-gpu)?@sha256:[0-9a-f]{64}"
            )
            # A service or test-sandbox recipe (judge-side only) may also copy from its own
            # earlier stages.
            require(
                stage in tools_stages
                or (role.startswith("sidecar-") and stage in own_stages)
                or (
                    (role in PINNED_COPY_ROLES or role.startswith("sidecar-"))
                    and PINNED_IMAGE.fullmatch(stage) is not None
                )
                or (role == "agent-gpu" and own_agent.fullmatch(stage) is not None),
                f"{path.name} copies from {stage!r}; only a shared base stage "
                "(FROM ${GPU_TOOLS}"
                + ("" if role.startswith("agent") else " or ${VERIFIER_BASE}")
                + " AS <stage>) is allowed"
                + (
                    ""
                    if role.startswith("agent")
                    else ", or a <registry>/<repository>@sha256:<digest> image"
                ),
            )
            continue
        fields = [f for f in shlex.split(line)[1:] if not f.startswith("--")]
        require(len(fields) == 2, f"use one task-root source per COPY in {path.name}")
        source = copy_source(root, fields[0], role)
        if role.startswith("agent"):
            require(
                source.split("/")[0] in NATIVE_AGENT_SOURCES
                or source.startswith(NATIVE_GENERATED_AGENT)
                or (role == "agent-gpu" and source == "gpu/task_runtime.py"),
                f"Agent image copies a non-public source: {source}",
            )


# A reference to one of the shared images in a recipe (FROM, COPY --from), with tag and digest.
SHARED_IMAGE_REFERENCE = re.compile(
    r"\S*/frontier-(?:agent-base|verifier-base|gpu-tools)(?:[:@]\S*)?"
)


def check_gpu_adapter_pins(root: Path) -> None:
    """A task.toml that sets `[gpu.verifier] lease_seconds`, or declares a command test sandbox,
    needs the shared verifier and adapter that implement them (an older fcs2_verifier.py does not
    know the control placement and silently ships the whole verification to one runner, or drops
    the Driver bearer run_sandboxes() leases machines with): its control Verifier takes the
    adapter from a `${GPU_TOOLS}` stage and, for a Verifier-GPU task, its GPU Verifier the
    verifier modules from a `${VERIFIER_BASE}` stage, both the tasks/shared/bases.json releases.
    The same holds for GPU runners and CPU machines: the judge runs on the control image."""
    raw = tomllib.loads((root / task_manifest.MANIFEST).read_text(encoding="utf-8"))
    settings = raw.get("gpu", {}).get("verifier", {})
    uses = []
    if "lease_seconds" in settings:
        uses.append("[gpu.verifier] lease_seconds")
    if task_manifest.command_sandboxes(raw):
        uses.append('[test_sandboxes] access = "command"')
    if not uses:
        return
    feature = " and ".join(uses)
    roles = (("verifier", "GPU_TOOLS"),) + (
        (("verifier-gpu", "VERIFIER_BASE"),) if "verifier" in raw.get("gpu", {}) else ()
    )
    for role, name in roles:
        path = task_manifest.layout(root).dockerfile(role)
        stages = check_shared_base_arguments(path, regular(path).read_text())
        require(
            task_manifest.SHARED_BASE_ARGUMENTS[name] in stages.values(),
            f"{path.name} needs a `FROM ${{{name}}} AS <stage>` stage: task.toml sets "
            f"{feature}, which the tasks/shared/bases.json shared verifier and GPU adapter "
            "implement",
        )


def check_bases() -> None:
    """tasks/shared/bases.json: each role's tag is derived from its sources (src-<tree hash>,
    scripts/build_base_images.py), pinned by digest; a source change without a new release, or
    a hand-picked tag, fails here."""
    import build_base_images

    recorded = task_manifest.bases()
    require(
        set(recorded) == {*task_manifest.BASE_ROLES, "schema_version"},
        "tasks/shared/bases.json must name exactly agent, verifier and gpu-tools",
    )
    for role in task_manifest.BASE_ROLES:
        tag = build_base_images.source_tag(role)
        expected = (
            rf"docker\.io/[a-z0-9_-]+/{re.escape(build_base_images.IMAGE_NAMES[role])}:"
            rf"{re.escape(tag)}@sha256:[0-9a-f]{{64}}"
        )
        require(
            re.fullmatch(expected, recorded[role]) is not None,
            f"tasks/shared/bases.json {role} is {recorded[role]}, but its sources are {tag}: "
            f"build and push the release (scripts/build_base_images.py --roles {role} --push)",
        )


def check_inventory(root: Path) -> None:
    lock_path = root / "upstream.lock.json"
    lock = document(lock_path)
    require(lock.get("problem_id") == root.name, "source lock problem_id mismatch")
    records = lock.get("files")
    require(
        isinstance(records, list) and lock.get("file_count") == len(records),
        "source lock needs an exact file inventory",
    )
    observed = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        require(
            not {"__pycache__", ".pytest_cache"}.intersection(relative.parts)
            and path.suffix not in {".pyc", ".pyo"},
            f"generated Python cache is not task source: {relative}",
        )
        require(
            not path.is_symlink() and (path.is_file() or path.is_dir()),
            f"source must contain regular files and directories: {path}",
        )
        if path.is_file() and path != lock_path:
            observed[path.relative_to(root).as_posix()] = path
    seen = set()
    for record in records:
        require(isinstance(record, dict), "source lock record must be an object")
        name = relative_path(record.get("path"))
        require(
            name not in seen and name in observed,
            f"unexpected or duplicate source lock member: {name}",
        )
        seen.add(name)
        path = observed[name]
        require(
            record.get("bytes") == path.stat().st_size
            and record.get("sha256") == digest(path),
            f"source lock mismatch: {name}",
        )
    require(seen == set(observed), "source lock does not cover the complete task tree")


def check_feedback_statement(root: Path, statement: str, visible: list[str]) -> None:
    for name in (
        "beat_spec_progress",
        "passed",
        "failed",
        "total",
        "pass_percent",
        "fail_percent",
    ):
        require(
            name in statement,
            f"public statement does not describe beat-spec feedback {name}",
        )
    missing = [name for name in visible if name not in statement]
    if not missing:
        return
    review_path = FEEDBACK_STATEMENT_REVIEWS.get(root.name)
    require(
        review_path is not None,
        f"public statement does not describe beat-spec feedback {missing}",
    )
    review = document(review_path)
    require(
        review.get("schema") == "preview.reviewed-feedback-statement/v1"
        and review.get("problem_id") == root.name,
        "feedback statement review identity mismatch",
    )
    bindings = review.get("bindings", {})
    require(
        set(bindings)
        == {
            "upstream.lock.json",
            "policy.json",
            "judge/beat_spec.json",
            "agent-base/frontier/task/README.md",
            "agent-base/frontier/task/repo/README.md",
        },
        "feedback statement review needs exact source bindings",
    )
    for name, expected in bindings.items():
        require(
            digest(root / name) == expected,
            f"feedback statement binding mismatch: {name}",
        )
    descriptions = review.get("fields", {})
    require(set(descriptions) == set(visible), "unreviewed feedback field set")
    sentence = review.get("sentence")
    require(
        isinstance(sentence, str)
        and bool(sentence)
        and sentence in statement
        and all(
            isinstance(text, str) and bool(text) and text in sentence
            for text in descriptions.values()
        ),
        "public statement omits reviewed absolute development summaries",
    )


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def check_beat_spec_schema(spec: dict, evaluation: dict) -> tuple[list[str], list[str]]:
    """Check the portable schema used by FrontierSmith's beat-spec design pass.

    Absence is deliberately an error for Preview releases. The private pipeline
    tolerates historical tasks without a spec; a newly published layout cannot.
    """
    for field in ("unit", "beats_reference", "statement_sentences"):
        require(_nonempty(spec.get(field)), f"beat_spec.json needs non-empty {field}")
    names_by_field = {}
    for field in ("metrics", "solver_feedback"):
        entries = spec.get(field)
        require(isinstance(entries, list), f"beat_spec.json {field} must be a list")
        require(
            field != "metrics" or bool(entries),
            "beat_spec.json must name at least one metric",
        )
        names = []
        for entry in entries:
            require(
                isinstance(entry, dict),
                f"beat_spec.json {field} entries must be objects",
            )
            name = entry.get("name")
            require(
                isinstance(name, str)
                and re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", name) is not None
                and name not in names,
                f"beat_spec.json {field} has an invalid or duplicate metric name",
            )
            names.append(name)
            require(_nonempty(entry.get("means")), f"beat_spec.json {name} needs means")
            if field == "metrics":
                for key in ("source", "clears_when", "tolerance"):
                    require(
                        _nonempty(entry.get(key)), f"beat_spec.json {name} needs {key}"
                    )
                require(
                    entry.get("better") in {"higher", "lower"},
                    f"beat_spec.json {name} has invalid better direction",
                )
            else:
                answer = entry.get("reveals_reference")
                require(
                    _nonempty(answer)
                    and re.match(r"no\b\W+\S", answer.strip(), re.IGNORECASE)
                    is not None,
                    f"beat_spec.json {name} needs a no-with-reason privacy explanation",
                )
                require(
                    "reference" not in name
                    and "margin" not in name
                    and not name.startswith("beats")
                    and name not in {"beat_spec_workload_passes", "beat_spec_progress"},
                    f"beat_spec.json solver_feedback names a private or reserved field: {name}",
                )
        names_by_field[field] = names

    withheld = spec.get("withheld")
    require(
        isinstance(withheld, list)
        and all(_nonempty(name) for name in withheld)
        and len(set(withheld)) == len(withheld),
        "beat_spec.json withheld must be unique non-empty strings",
    )
    visible = names_by_field["solver_feedback"]
    require(
        not set(visible).intersection(withheld),
        "beat_spec.json exposes a withheld field",
    )
    declared = evaluation.get("solver_visible_metrics")
    require(
        isinstance(declared, list)
        and all(isinstance(name, str) for name in declared)
        and len(set(declared)) == len(declared)
        and set(declared) == set(visible),
        "policy solver_visible_metrics must exactly match beat_spec.json solver_feedback",
    )

    if "beats_reference_aggregation" in spec:
        aggregation = spec["beats_reference_aggregation"]
        require(
            isinstance(aggregation, dict),
            "beat_spec.json aggregation must be an object when declared",
        )
        kind = aggregation.get("kind")
        if kind == "conjunction":
            require(
                set(aggregation) == {"kind"},
                "conjunction aggregation contains unknown fields",
            )
        else:
            require(
                kind == "score_majority_coverage",
                "unsupported beats_reference aggregation kind",
            )
            fields = {
                "kind",
                "minimum_score_margin",
                "minimum_comparison_pass_fraction_exclusive",
                "require_each_workload",
                "require_each_metric",
            }
            require(
                set(aggregation) == fields,
                "score_majority_coverage aggregation requires exactly its declared fields",
            )
            margin = aggregation["minimum_score_margin"]
            fraction = aggregation["minimum_comparison_pass_fraction_exclusive"]
            require(
                _finite_number(margin) and margin >= 0,
                "aggregation minimum_score_margin must be finite and non-negative",
            )
            require(
                _finite_number(fraction) and 0 <= fraction < 1,
                "aggregation pass fraction must be finite and in [0, 1)",
            )
            require(
                all(
                    type(aggregation[key]) is bool
                    for key in ("require_each_workload", "require_each_metric")
                ),
                "aggregation coverage flags must be boolean",
            )
    return names_by_field["metrics"], visible


def declared_workload_metrics(spec: dict, names: list[str]) -> dict[str, set[str]]:
    """Map each workload row or group the beat spec declares to its metrics.

    Rows are listed under any top-level key ending in "workloads". A row carries
    its own "metrics" when given, else the spec's matching "<prefix>metrics"
    list (for example final_only_workloads with final_only_metrics), else every
    declared metric.
    """
    declared = {}
    groups = spec.get("workload_groups")
    if groups is not None:
        require(isinstance(groups, dict) and bool(groups), "invalid workload groups")
        for group, record in groups.items():
            require(
                isinstance(record, dict) and isinstance(record.get("metrics"), list),
                "invalid workload group metrics",
            )
            declared[f"workload_groups.{group}"] = set(record["metrics"])
    for key, rows in spec.items():
        if not key.endswith("workloads"):
            continue
        require(isinstance(rows, list), f"beat_spec.json {key} must be a list")
        prefix = key[: -len("workloads")]
        shared = spec.get(f"{prefix}metrics") if prefix else None
        for index, row in enumerate(rows):
            if isinstance(row, str):
                row = {"name": row}
            require(
                isinstance(row, dict) and _nonempty(row.get("name")),
                f"beat_spec.json {key}[{index}] needs a workload name",
            )
            carried = row.get("metrics", shared)
            if carried is None:
                carried = names
            require(
                isinstance(carried, (list, dict))
                and all(isinstance(name, str) for name in carried),
                f"beat_spec.json {key}[{index}] has invalid metrics",
            )
            declared[f"{key}[{row['name']}]"] = set(carried)
    return declared


def check_uniform_workload_metrics(root: Path, spec: dict, names: list[str]) -> None:
    """Every workload the beat spec declares measures every declared metric."""
    expected = set(names)
    problems = []
    if "workload_groups" in spec:
        problems.append("workload_groups is retired")
    for row, carried in declared_workload_metrics(spec, names).items():
        if carried != expected:
            detail = []
            if expected - carried:
                detail.append(f"lacks {sorted(expected - carried)}")
            if carried - expected:
                detail.append(f"names undeclared {sorted(carried - expected)}")
            problems.append(f"{row} {' and '.join(detail)}")
    exemption = NONUNIFORM_WORKLOAD_METRICS_EXEMPT.get(root.name)
    if not problems:
        require(
            exemption is None,
            f"remove the stale exemption for {root.name}: "
            "every declared workload now measures every metric",
        )
        return
    shown = "; ".join(problems[:3])
    if len(problems) > 3:
        shown += f"; and {len(problems) - 3} more"
    message = f"every workload must measure every declared metric: {shown}"
    require(exemption is not None, message)
    warn(root, f"{message} (exempt: {exemption})")


def check_beat_spec_feedback(
    root: Path, spec: dict, names: list[str], visible: list[str]
) -> None:
    """Exercise the shipped, dependency-free feedback boundary without a workload.

    Compile in memory to keep validation from adding __pycache__ to a locked
    source tree. This does not import or execute the private evaluator.
    """
    helper = regular(task_manifest.layout(root).beat_spec_helper)
    namespace = {"__name__": "preview_feedback_check", "__file__": str(helper)}
    try:
        exec(compile(helper.read_text(), str(helper), "exec"), namespace)
        public_metrics = namespace["public_metrics"]
        values = {name: [1.0, 2.0] for name in visible}
        values.update(
            {
                "beat_spec_workload_passes": [
                    {
                        "workload": f"private-layout-probe-{index}",
                        "metrics": {name: True for name in names},
                    }
                    for index in range(2)
                ],
                "beats_reference": 1,
                "reference_score": 50.0,
                "margin": spec.get("beats_reference_aggregation", {}).get(
                    "minimum_score_margin", 0.0
                ),
            }
        )
        kwargs = {
            "beat_spec_path": root / "judge/beat_spec.json",
            "configured_visible": visible,
        }
        totals = {name: 2 for name in names}
        groups = spec.get("workload_groups")
        if groups is not None:
            # Retired layout, probed only for an exempt task whose shipped
            # helper still requires grouped rows.
            require(
                root.name in NONUNIFORM_WORKLOAD_METRICS_EXEMPT,
                "beat_spec.json workload_groups is retired: every workload "
                "must measure every declared metric",
            )
            require(
                isinstance(groups, dict) and bool(groups), "invalid workload groups"
            )
            rows = []
            totals = {}
            for group_name, group in groups.items():
                require(isinstance(group, dict), "invalid workload group")
                counts = group.get("counts")
                require(isinstance(counts, dict), "invalid workload group counts")
                count = counts.get("train")
                metrics = group.get("metrics")
                require(
                    type(count) is int and 0 < count <= 100000,
                    "invalid or excessive grouped feedback probe count",
                )
                require(
                    isinstance(metrics, list) and bool(metrics),
                    "invalid workload group metrics",
                )
                require(
                    len(rows) + count <= 100000,
                    "grouped feedback probe exceeds bounded layout budget",
                )
                for name in metrics:
                    require(
                        name in names and name not in totals,
                        "workload groups must partition declared metrics",
                    )
                    totals[name] = count
                first = len(rows)
                rows.extend(
                    {
                        "workload": f"private-layout-probe-{first + index}",
                        "group": group_name,
                        "metrics": {name: True for name in metrics},
                    }
                    for index in range(count)
                )
            require(set(totals) == set(names), "workload groups omit metrics")
            values["beat_spec_workload_passes"] = rows
            values["beat_spec_suite_role"] = "train"
        actual = public_metrics(values, **kwargs)
        expected = {name: values[name] for name in visible}
        expected["beat_spec_progress"] = {
            name: {
                "passed": totals[name],
                "failed": 0,
                "total": totals[name],
                "pass_percent": 100.0,
                "fail_percent": 0.0,
            }
            for name in names
        }
        require(
            actual == expected,
            "beat-spec helper exposes private fields or returns incorrect progress",
        )

        invalid_payloads = []
        inconsistent = copy.deepcopy(values)
        inconsistent["beats_reference"] = 0
        invalid_payloads.append((inconsistent, "inconsistent beats_reference"))
        duplicate = copy.deepcopy(values)
        duplicate["beat_spec_workload_passes"].append(
            copy.deepcopy(duplicate["beat_spec_workload_passes"][0])
        )
        invalid_payloads.append((duplicate, "duplicate workload"))
        wrong_metric = copy.deepcopy(values)
        wrong_metric["beat_spec_workload_passes"][0]["metrics"][
            "undeclared_probe_metric"
        ] = True
        invalid_payloads.append((wrong_metric, "undeclared row metric"))
        if root.name not in NONUNIFORM_WORKLOAD_METRICS_EXEMPT:
            partial = copy.deepcopy(values)
            partial["beat_spec_workload_passes"][0]["metrics"].pop(names[0])
            invalid_payloads.append((partial, "workload row missing a declared metric"))
        if visible:
            identity = copy.deepcopy(values)
            identity[visible[0]] = {"nested": [{"private-layout-probe-0": 1.0}]}
            invalid_payloads.append(
                (identity, "private workload identity in visible metrics")
            )
        if groups is not None:
            missing_role = copy.deepcopy(values)
            missing_role.pop("beat_spec_suite_role")
            invalid_payloads.append((missing_role, "missing grouped suite role"))
            missing_row = copy.deepcopy(values)
            missing_row["beat_spec_workload_passes"].pop()
            invalid_payloads.append((missing_row, "missing grouped workload"))
            wrong_group = copy.deepcopy(values)
            wrong_group["beat_spec_workload_passes"][0]["group"] = "undeclared-group"
            invalid_payloads.append((wrong_group, "unknown workload group"))
        for payload, reason in invalid_payloads:
            try:
                public_metrics(payload, **kwargs)
            except ValueError:
                continue
            raise LayoutFailure(f"beat-spec helper does not reject {reason}")
    except LayoutFailure:
        raise
    except Exception as error:
        raise LayoutFailure(
            f"beat-spec feedback helper contract failed: {error}"
        ) from error


def _evaluator_scope_nodes(statements: list[ast.AST]) -> list[ast.AST]:
    """Do not borrow bindings or literals from an unrelated nested scope."""
    pending = list(statements)
    nodes = []
    boundaries = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    while pending:
        node = pending.pop()
        nodes.append(node)
        if not isinstance(node, boundaries):
            pending.extend(ast.iter_child_nodes(node))
    return nodes


def _evaluator_bindings(nodes: list[ast.AST]) -> dict[str, list[ast.AST]]:
    bindings = {}
    for node in nodes:
        names = []
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names = [node.id]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [
                alias.asname
                or (
                    alias.name.split(".")[0]
                    if isinstance(node, ast.Import)
                    else alias.name
                )
                for alias in node.names
            ]
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names = (
                node.names
            )  # Dynamic outer writes cannot establish a helper binding.
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names = [node.name]
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names = [node.name]
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names = [node.rest]
        elif isinstance(node, ast.arg):
            names = [node.arg]
        for name in names:
            bindings.setdefault(name, []).append(node)
    return bindings


def referenced_local_evaluator_nodes(evaluator: Path) -> list[ast.AST]:
    """Inspect unambiguously bound local helpers without executing evaluators.

    This structural check cannot prove Python control flow or measurement
    semantics. Rebound, nested-only and otherwise unknown imports contribute
    no fields; the runtime feedback and independent measurement gates remain.
    """
    directory = evaluator.parent.resolve()
    pending = [(evaluator, "evaluate")]
    seen = set()
    nodes = []
    while pending:
        path, symbol = pending.pop()
        key = (path, symbol)
        if key in seen:
            continue
        seen.add(key)
        require(len(seen) <= 256, "evaluator helper graph exceeds structural bound")
        require(
            path.resolve().parent == directory and not path.is_symlink(),
            "evaluator helper must be a regular local judge file",
        )
        tree = ast.parse(regular(path).read_text(), filename=str(path))
        module_bindings = _evaluator_bindings(_evaluator_scope_nodes(tree.body))
        definitions = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name) and isinstance(
                            name.ctx, ast.Store
                        ):
                            definitions[name.id] = node
        selected = definitions.get(symbol)
        if selected is None or len(module_bindings.get(symbol, [])) != 1:
            continue
        is_function = isinstance(selected, (ast.FunctionDef, ast.AsyncFunctionDef))
        inspected = _evaluator_scope_nodes(selected.body if is_function else [selected])
        nodes.extend(inspected)
        referenced = {
            node.id
            for node in inspected
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        local_bindings = (
            _evaluator_bindings(inspected + list(ast.walk(selected.args)))
            if is_function
            else {}
        )
        for name in referenced:
            bindings = local_bindings if name in local_bindings else module_bindings
            candidates = bindings.get(name, [])
            if len(candidates) != 1 or "*" in bindings:
                continue  # Ambiguous writes/imports cannot supply metric evidence.
            binding = candidates[0]
            if bindings is module_bindings and name in definitions:
                pending.append((path, name))
            elif isinstance(binding, ast.ImportFrom) and binding.level in (0, 1):
                if not binding.module or not binding.module.isidentifier():
                    continue
                imported = directory / (binding.module + ".py")
                if not imported.exists():
                    continue  # External dependencies are not inspected.
                for alias in binding.names:
                    if (alias.asname or alias.name) == name:
                        pending.append((imported, alias.name))
    return nodes


# Metric names the feedback probe gives the shared helper for a task.toml task: its
# rows, not a spec file, define the metric names, so any fixed pair exercises it.
FEEDBACK_PROBE_METRICS = ("probe_time", "probe_bytes")


def check_task_feedback(root: Path, evaluation: dict) -> None:
    """A task.toml task's scoring contract: no judge/beat_spec.json; the evaluator's
    docstring is the scoring note and it emits the private beat-spec fields; what train returns
    to the Agent is task.toml [feedback], stated in the generated README; and the shared helper
    returns exactly those values plus aggregate progress."""
    require(
        not (root / "judge/beat_spec.json").exists(),
        "judge/beat_spec.json is removed for task.toml tasks: the scoring note is the "
        "docstring of judge/evaluator.py and the Agent's train feedback is task.toml [feedback]",
    )
    visible = evaluation.get("solver_visible_metrics")
    require(
        isinstance(visible, list)
        and all(isinstance(n, str) for n in visible)
        and len(set(visible)) == len(visible),
        "invalid solver_visible_metrics",
    )
    # The judge entry: judge/evaluator.py, or a standalone Verifier's verifier/verify.py (it
    # projects its own results, so it emits beat_spec_progress instead of the workload rows).
    standalone = (
        task_manifest.read_task(root)["verifier"]["command"]
        != task_manifest.defaults()["verifier"]["command"]
    )
    entry = "verifier/verify.py" if standalone else "judge/evaluator.py"
    evaluator = regular(root / entry)
    try:
        tree = ast.parse(evaluator.read_text(), filename=str(evaluator))
    except SyntaxError as error:
        raise LayoutFailure(f"{entry} is not valid Python") from error
    require(
        _nonempty(ast.get_docstring(tree)),
        f"{entry} needs a module docstring: the task's scoring note",
    )
    inspected = list(ast.walk(tree))
    if not standalone:
        inspected += referenced_local_evaluator_nodes(evaluator)
    output_keys = set()
    for node in inspected:
        keys = (
            node.keys
            if isinstance(node, ast.Dict)
            else [node.slice]
            if isinstance(node, ast.Subscript)
            else []
        )
        output_keys.update(
            key.value
            for key in keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dict"
        ):
            output_keys.update(keyword.arg for keyword in node.keywords if keyword.arg)
    private = (
        ("beat_spec_progress", "beats_reference")
        if standalone
        else ("beat_spec_workload_passes", "beats_reference")
    )
    for name in private:
        require(
            name in output_keys, f"{entry} does not emit private beat-spec field {name}"
        )
    literals = {
        node.value
        for node in inspected
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for name in visible:
        require(
            name in literals,
            f"{entry} does not name task.toml [feedback] metric {name}",
        )
    # The generated README lists the shared train fields and every [feedback] metric; the
    # task's own README and statement need not name them.
    readme = task_manifest.agent_readme(root)
    for name in ("beat_spec_progress", *visible):
        require(
            f"`{name}`" in readme,
            f"the generated README does not name train metric {name}",
        )
    check_shared_feedback(task_manifest.layout(root).beat_spec_helper, visible)


def check_shared_feedback(helper: Path, visible: list[str]) -> None:
    """Exercise the shared feedback helper on evaluator-shaped metrics, compiled in memory so
    validation adds no __pycache__ and never imports the task's evaluator."""
    namespace = {"__name__": "preview_feedback_check", "__file__": str(helper)}
    try:
        exec(compile(regular(helper).read_text(), str(helper), "exec"), namespace)
        public_metrics, private_metrics = (
            namespace["public_metrics"],
            namespace["private_metrics"],
        )
        names = list(FEEDBACK_PROBE_METRICS)
        values = {name: [1.0, 2.0] for name in visible}
        values.update(
            {
                "beat_spec_workload_passes": [
                    {
                        "workload": f"private-layout-probe-{index}",
                        "metrics": {
                            name: index == 0 or name == names[0] for name in names
                        },
                    }
                    for index in range(2)
                ],
                "beats_reference": 0,
                "reference_score": 50.0,
                "margin": -1.0,
                "private_probe_detail": 3.0,
            }
        )
        expected = {name: values[name] for name in visible}
        expected["beat_spec_progress"] = {
            names[0]: {
                "passed": 2,
                "failed": 0,
                "total": 2,
                "pass_percent": 100.0,
                "fail_percent": 0.0,
            },
            names[1]: {
                "passed": 1,
                "failed": 1,
                "total": 2,
                "pass_percent": 50.0,
                "fail_percent": 50.0,
            },
        }
        require(
            public_metrics(values, visible=visible) == expected,
            "shared feedback helper exposes private fields or returns incorrect progress",
        )
        require(
            public_metrics(values, visible=[*visible, "reference_score", "margin"])
            == expected,
            "shared feedback helper returns a private metric",
        )
        private = private_metrics(values)
        require(
            private
            == {
                "beat_spec_progress": expected["beat_spec_progress"],
                "beats_reference": 0,
                "reference_score": 50.0,
                "margin": -1.0,
            },
            "shared feedback helper returns incorrect private metrics",
        )
        invalid = []
        for reason, change in (
            ("an invalid beats_reference", lambda v: v.update(beats_reference=2)),
            (
                "a duplicate workload",
                lambda v: v["beat_spec_workload_passes"].append(
                    copy.deepcopy(v["beat_spec_workload_passes"][0])
                ),
            ),
            (
                "a row with an extra metric",
                lambda v: v["beat_spec_workload_passes"][1]["metrics"].update(
                    undeclared_probe_metric=True
                ),
            ),
            (
                "a row missing a metric",
                lambda v: v["beat_spec_workload_passes"][1]["metrics"].pop(names[0]),
            ),
            (
                "a non-boolean decision",
                lambda v: v["beat_spec_workload_passes"][0]["metrics"].update(
                    {names[0]: 1}
                ),
            ),
            ("no workload rows", lambda v: v.update(beat_spec_workload_passes=[])),
        ):
            payload = copy.deepcopy(values)
            change(payload)
            invalid.append((payload, reason))
        if visible:
            identity = copy.deepcopy(values)
            identity[visible[0]] = {"nested": [{"private-layout-probe-0": 1.0}]}
            invalid.append(
                (identity, "a private workload identity in a visible metric")
            )
        for payload, reason in invalid:
            try:
                public_metrics(payload, visible=visible)
            except ValueError:
                continue
            raise LayoutFailure(f"shared feedback helper does not reject {reason}")
    except LayoutFailure:
        raise
    except Exception as error:
        raise LayoutFailure(
            f"shared feedback helper contract failed: {error}"
        ) from error


def check_native_contract(root: Path, evaluation: dict) -> None:
    """A policy.json task without judge/beat_spec.json declares its own contract: a
    native_contract.py at its root whose check_native_contract(root, evaluation) raises
    ValueError when the task breaks it."""
    contract = root / "native_contract.py"
    require(
        contract.is_file() and not contract.is_symlink(),
        "a policy.json task without judge/beat_spec.json needs native_contract.py",
    )
    spec = importlib.util.spec_from_file_location(
        f"native_contract_{root.name}", contract
    )
    module = importlib.util.module_from_spec(spec)
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write
    try:
        module.check_native_contract(root, evaluation)
    except ValueError as error:
        raise LayoutFailure(str(error)) from error


def check_beat_spec(root: Path, evaluation: dict) -> None:
    if task_manifest.is_native(root):
        check_task_feedback(root, evaluation)
        return
    if not (root / "judge/beat_spec.json").exists():
        check_native_contract(root, evaluation)
        return
    spec = document(root / "judge/beat_spec.json")
    names, visible = check_beat_spec_schema(spec, evaluation)
    check_uniform_workload_metrics(root, spec, names)
    evaluator = regular(root / "judge/evaluator.py")
    try:
        tree = ast.parse(evaluator.read_text(), filename=str(evaluator))
    except SyntaxError as error:
        raise LayoutFailure("beat-spec evaluator is not valid Python") from error
    # Literal keys are stronger evidence than substring searches in comments;
    # this remains a structural check, not a proof of measurement semantics.
    inspected = list(ast.walk(tree))
    inspected.extend(referenced_local_evaluator_nodes(evaluator))
    literals = {
        node.value
        for node in inspected
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for name in names:
        require(name in literals, f"evaluator does not name beat-spec metric {name}")
    output_keys = set()
    for node in inspected:
        keys = (
            node.keys
            if isinstance(node, ast.Dict)
            else [node.slice]
            if isinstance(node, ast.Subscript)
            else []
        )
        output_keys.update(
            key.value
            for key in keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dict"
        ):
            output_keys.update(keyword.arg for keyword in node.keywords if keyword.arg)
    for name in ("beat_spec_workload_passes", "beats_reference"):
        require(
            name in output_keys,
            f"evaluator does not emit private beat-spec field {name}",
        )
    paths = task_manifest.layout(root)
    statement = (
        task_manifest.agent_readme(root)
        if paths.native
        else regular(paths.statement).read_text()
    )
    if (paths.repo / "README.md").is_file():
        statement += "\n" + regular(paths.repo / "README.md").read_text()
    check_feedback_statement(root, statement, visible)
    check_beat_spec_feedback(root, spec, names, visible)


def partitioned_audit_bundle_bytes(policy: dict) -> int:
    """Fund the full declared envelope, as FrontierSmith fcs2.audit_budget does.

    This exceeds upstream's one-exchange admission minimum. It retains every
    declared submission/result and model/SSH session, fixed gateway reserves,
    and an 8 MiB cushion rounded upward to a whole MiB.
    """
    mib = 1024 * 1024
    audit = policy["network"]["audit"]
    calls = policy["quota"]["max_train_calls"]
    evidence = policy["verifier"]["max_evidence_bytes"]
    submission = policy["submission"]["max_bytes"]
    require(
        type(calls) is int
        and calls >= 0
        and type(evidence) is int
        and evidence > 0
        and type(submission) is int
        and submission > 0,
        "invalid retained audit payload budgets",
    )
    require(
        (2 * calls + 2) * 4096 <= 256 * mib,
        "retained audit spool count exceeds the platform envelope",
    )
    floor = (
        (24 + 128 + 4) * mib
        + (calls + 1) * (evidence + submission + 2 * 64 * 1024)
        + audit["max_model_sessions"]
        * (audit["max_model_request_bytes"] + audit["max_model_response_bytes"])
    )
    gpu = policy.get("gpu", {})
    if gpu.get("agent", {}).get("max_instances", 0) or gpu.get("verifier", {}).get(
        "max_gpu_instances", 0
    ):
        floor += audit["max_ssh_sessions"] * (
            2 * audit["max_ssh_bytes_per_direction"] + 24 * 1024
        )
    return max(200 * mib, ((floor + 8 * mib + mib - 1) // mib) * mib)


def native_audit_minimum_bytes(policy: dict) -> int:
    """Match the native admission floor for an independently bounded Run.

    Session and stream limits are ceilings, not promises to fill every stream.
    Driver accounts all retained capture against max_bundle_bytes in real time
    and fails closed on exhaustion. Keep the configured gateway reserve inside
    that same bundle: 4 GiB snapshot for bundles >= 8 GiB, otherwise 128 MiB,
    plus the standard 4 MiB disposition reserve.
    """
    mib = 1024 * 1024
    audit = policy["network"]["audit"]
    calls = policy["quota"]["max_train_calls"]
    evidence = policy["verifier"]["max_evidence_bytes"]
    submission = policy["submission"]["max_bytes"]
    require(
        type(calls) is int
        and calls >= 0
        and type(evidence) is int
        and evidence > 0
        and type(submission) is int
        and submission > 0,
        "invalid retained audit payload budgets",
    )
    require(
        (2 * calls + 2) * 4096 <= 256 * mib,
        "retained audit spool count exceeds the platform envelope",
    )
    snapshot = 4096 * mib if audit["max_bundle_bytes"] >= 8192 * mib else 128 * mib
    floor = (
        evidence
        + (24 + 4) * mib
        + snapshot
        + audit["max_model_request_bytes"]
        + audit["max_model_response_bytes"]
    )
    gpu = policy.get("gpu", {})
    if gpu.get("agent", {}).get("max_instances", 0) or gpu.get("verifier", {}).get(
        "max_gpu_instances", 0
    ):
        floor += 2 * audit["max_ssh_bytes_per_direction"] + 24 * 1024
    return floor


def check_ssh_session_budgets(policy: dict) -> None:
    audit = policy["network"].get("audit", {})
    require(isinstance(audit, dict), "network.audit must be an object")
    accelerator = policy["metadata"].get("accelerator") or {}
    require(
        isinstance(accelerator, dict),
        "accelerator metadata must be an object when present",
    )
    interface = accelerator.get("agent_gpu_interface")
    if "ssh_session_budgets" not in audit:
        require(
            interface != "frontier.gpu-helper/v3",
            "Candidate helper V3 task requires partitioned SSH budgets",
        )
        return  # Preserve the explicit legacy shared policy on historical tasks.
    pools = audit["ssh_session_budgets"]
    require(
        isinstance(pools, dict)
        and set(pools) == {"agent", "verifier_train", "verifier_eval"},
        "SSH budgets require exactly Agent, Verifier train and Verifier eval pools",
    )
    require(
        all(type(value) is int and 0 <= value <= 2048 for value in pools.values()),
        "SSH pool counts must be non-negative integers at most 2048",
    )
    for field in ("max_model_sessions", "max_ssh_sessions"):
        require(
            type(audit.get(field)) is int and 1 <= audit[field] <= 2048,
            f"invalid audit {field}",
        )
    require(
        sum(pools.values()) == audit["max_ssh_sessions"],
        "SSH pools must sum to max_ssh_sessions",
    )
    require(
        audit["max_model_sessions"] + audit["max_ssh_sessions"] <= 4096,
        "combined model and SSH session count exceeds 4096",
    )
    for field in (
        "max_model_request_bytes",
        "max_model_response_bytes",
        "max_ssh_bytes_per_direction",
    ):
        limit_mib = 512 if field == "max_ssh_bytes_per_direction" else 64
        require(
            type(audit.get(field)) is int
            and 1 <= audit[field] <= limit_mib * 1024 * 1024,
            f"audit {field} must fit its {limit_mib} MiB stream bound",
        )
    gpu = policy.get("gpu", {})
    if gpu.get("agent", {}).get("max_instances", 0):
        require(pools["agent"] > 0, "Agent GPU requires a positive Agent SSH pool")
        require(
            interface == "frontier.gpu-helper/v3",
            "partitioned Agent GPU task requires Candidate helper V3",
        )
    if gpu.get("verifier", {}).get("max_gpu_instances", 0):
        require(
            pools["verifier_eval"] > 0,
            "Verifier GPU requires a reserved final SSH pool",
        )
        require(
            not policy["quota"]["max_train_calls"] or pools["verifier_train"] > 0,
            "Verifier GPU requires a train SSH pool",
        )
    bundle = audit.get("max_bundle_bytes")
    require(
        type(bundle) is int
        and native_audit_minimum_bytes(policy) <= bundle <= 32 * 1024**3,
        "partitioned audit bundle must fund the native exchange and gateway reserve within the 32 GiB cap",
    )


SIDECAR_NAME_RE = re.compile(r"(?!main$)[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?")
SIDECAR_PHASES = {"agent": "agent", "train": "verifier", "eval": "verifier"}
MAX_SIDECARS = 250  # Modal's concurrent-sidecar cap per Sandbox (measured 2026-10-12)
MAX_SIDECAR_ARG_BYTES = 32 * 1024
# Modal admits sidecars against the parent reservation at this per-container
# floor; the provider default reservation therefore allows four sidecars.
SIDECAR_MIN_CPU_MILLIS = 32
SIDECAR_MIN_MEMORY_BYTES = 32 * 1024 * 1024
DEFAULT_RESERVATION_SIDECARS = 4


def check_native_sidecar_recipes(root: Path) -> None:
    """A task.toml task's sidecar recipes are exactly the docker/test-sandbox*.Dockerfile files
    its [test_sandboxes.<name>] tables name; a template's recipe builds a stage FROM the
    per-instance build arg ${TEST_SANDBOX_BASE}."""
    task = task_manifest.read_task(root)
    commands = task_manifest.command_sandboxes(task)
    tables = {
        name: table
        for name, table in task.get("test_sandboxes", {}).items()
        if name not in commands
    }
    declared = {table["dockerfile"] for table in tables.values()}
    present = {
        f"docker/{path.name}"
        for path in (root / "docker").glob("test-sandbox*.Dockerfile")
    }
    require(
        present == declared,
        f"test-sandbox recipes and task.toml disagree: {sorted(present ^ declared)}",
    )
    require(
        not any(path.is_dir() for path in root.glob("sidecar-*")),
        "a task.toml task keeps its test-sandbox recipes in docker/, not sidecar-*/",
    )
    for name, table in tables.items():
        if "instances" not in table:
            continue
        lines = regular(root / table["dockerfile"]).read_text().splitlines()
        froms = [line.split() for line in lines if line.startswith("FROM ")]
        require(
            bool(froms)
            and "${TEST_SANDBOX_BASE}" in [f[1] for f in froms]
            and "ARG TEST_SANDBOX_BASE" in lines
            and "ARG TEST_SANDBOX_INSTANCE" in lines,
            f"{table['dockerfile']} ([test_sandboxes.{name}]) must declare ARG "
            "TEST_SANDBOX_BASE and ARG TEST_SANDBOX_INSTANCE and build a stage FROM "
            "${TEST_SANDBOX_BASE}",
        )
    # An untrusted command sandbox runs candidate code on a runner of its image: that recipe
    # copies nothing from judge/ (the judge, the reference and the hidden suites stay on the
    # control image).
    for name, table in commands.items():
        if table.get("trust", "untrusted") == "trusted":
            continue
        recipe = root / table["dockerfile"]
        copies = [
            line
            for line in regular(recipe).read_text().splitlines()
            if re.match(r"\s*(COPY|ADD)\b", line, re.IGNORECASE)
            and re.search(r"(^|[\s=,/])judge(/|\s|$)", line)
        ]
        require(
            not copies,
            f"{recipe.name} runs [test_sandboxes.{name}] candidate commands "
            f"and must copy nothing from judge/: {copies[:2]}",
        )


def check_sidecars(root: Path, policy: dict) -> None:
    """Declared same-host sidecars: fixed images built from `sidecar-<role>/Dockerfile`.

    Each declaration mirrors `TaskConfig.images.gpu_sidecars`: the Driver starts
    the containers beside the GPU runner of the matching phase, so every phase
    needs its GPU role enabled, and more than the provider-default count needs
    an explicit `gpu_parent_resources` allocation for that parent. A profile's
    optional `image` names the role image it runs (default: its own name), so
    several sidecars can share one published image.
    """
    gpu = policy.get("gpu", {})
    enabled = {
        "agent": bool(gpu.get("agent", {}).get("max_instances")),
        "verifier": bool(gpu.get("verifier", {}).get("max_gpu_instances")),
    }
    sidecars = policy.get("sidecars", {})
    require(isinstance(sidecars, dict), "sidecars must be an object")
    require(len(sidecars) <= MAX_SIDECARS, f"at most {MAX_SIDECARS} sidecars")
    per_parent = {"agent": 0, "verifier": 0}
    roles = set()
    for name, profile in sidecars.items():
        require(
            isinstance(name, str) and SIDECAR_NAME_RE.fullmatch(name) is not None,
            f"invalid sidecar name: {name!r}",
        )
        require(isinstance(profile, dict), f"sidecar {name} must be an object")
        require(
            {"args", "workdir", "phases"} <= set(profile)
            and set(profile)
            <= {"args", "workdir", "phases", "attestation_key", "image"},
            f"sidecar {name} needs args, workdir, phases and optional attestation_key/image",
        )
        role = profile.get("image", name)
        require(
            isinstance(role, str) and SIDECAR_NAME_RE.fullmatch(role) is not None,
            f"sidecar {name} image role must be a lowercase DNS label",
        )
        roles.add(role)
        args = profile["args"]
        require(
            isinstance(args, list)
            and 0 < len(args) <= 64
            and all(
                isinstance(arg, str)
                and "\0" not in arg
                and len(arg) <= MAX_SIDECAR_ARG_BYTES
                for arg in args
            )
            and args[0],
            f"sidecar {name} args must be a fixed non-empty command",
        )
        workdir = profile["workdir"]
        require(
            isinstance(workdir, str)
            and workdir.startswith("/")
            and "\0" not in workdir
            and len(workdir) <= MAX_SIDECAR_ARG_BYTES,
            f"sidecar {name} workdir must be absolute",
        )
        phases = profile["phases"]
        require(
            isinstance(phases, list)
            and phases
            and len(set(phases)) == len(phases)
            and set(phases) <= set(SIDECAR_PHASES),
            f"sidecar {name} phases must be unique agent/train/eval values",
        )
        require(
            type(profile.get("attestation_key", False)) is bool,
            f"sidecar {name} attestation_key must be boolean",
        )
        for parent in {SIDECAR_PHASES[phase] for phase in phases}:
            require(
                enabled[parent],
                f"sidecar {name} needs the {parent} GPU role enabled",
            )
            per_parent[parent] += 1
    for role in sorted(roles):
        check_dockerfile(root, f"sidecar-{role}")
    if task_manifest.is_native(root):
        check_native_sidecar_recipes(root)
    else:
        declared = {f"sidecar-{role}" for role in roles}
        present = {path.name for path in root.glob("sidecar-*") if path.is_dir()}
        require(
            present == declared,
            f"sidecar image directories and declarations disagree: {sorted(present ^ declared)}",
        )
    resources = policy.get("gpu_parent_resources", {})
    require(isinstance(resources, dict), "gpu_parent_resources must be an object")
    require(
        set(resources) <= {"agent", "verifier"}
        and ("gpu_parent_resources" not in policy or resources),
        "gpu_parent_resources declares agent and/or verifier",
    )
    for parent, allocation in resources.items():
        require(enabled[parent], f"gpu_parent_resources.{parent} needs that GPU role")
        require(
            isinstance(allocation, dict)
            and set(allocation) == {"cpu_millis", "memory_bytes"}
            and type(allocation["cpu_millis"]) is int
            and 1 <= allocation["cpu_millis"] <= 64_000
            and type(allocation["memory_bytes"]) is int
            and 1 <= allocation["memory_bytes"] <= 512 * 1024**3
            and allocation["memory_bytes"] % (1024 * 1024) == 0,
            f"gpu_parent_resources.{parent} needs cpu_millis and whole-MiB memory_bytes",
        )
    for parent, count in per_parent.items():
        if count > DEFAULT_RESERVATION_SIDECARS:
            allocation = resources.get(parent)
            require(
                allocation is not None
                and allocation["cpu_millis"] >= SIDECAR_MIN_CPU_MILLIS * count
                and allocation["memory_bytes"] >= SIDECAR_MIN_MEMORY_BYTES * count,
                f"{count} {parent} sidecars need gpu_parent_resources.{parent} of at least "
                f"{SIDECAR_MIN_CPU_MILLIS * count} mCPU and {SIDECAR_MIN_MEMORY_BYTES * count} bytes",
            )


def validate_task(root: Path) -> str:
    try:
        paths = task_manifest.layout(root)
        policy = (
            task_manifest.load_policy(root)
            if paths.native
            else document(root / "policy.json")
        )
    except task_manifest.ManifestError as error:
        raise LayoutFailure(str(error)) from error
    require(
        policy.get("schema_version") == "frontier.frontiersmith-task-policy/v1",
        "unsupported policy schema",
    )
    require(policy.get("problem_id") == root.name, "policy problem_id mismatch")
    for key in ("display_name", "family", "version"):
        require(
            isinstance(policy.get(key), str) and bool(policy[key]),
            f"policy needs {key}",
        )
    for key in ("agent", "quota", "submission", "verifier", "network", "metadata"):
        require(isinstance(policy.get(key), dict), f"policy needs {key}")
    evaluation = policy["metadata"].get("evaluation", {})
    require(isinstance(evaluation, dict), "policy needs evaluation metadata")
    calls = evaluation.get("authored_max_dev_submissions")
    require(
        type(calls) is int
        and calls > 0
        and type(policy["quota"].get("max_train_calls")) is int
        and calls == policy["quota"]["max_train_calls"],
        "quota must match task-authored max_dev_submissions",
    )
    entrypoint = relative_path(policy["submission"].get("entrypoint"))
    task = paths.public
    readme = (
        task_manifest.agent_readme(root)
        if paths.native
        else regular(paths.statement).read_text()
    )
    for text in (
        "frontier-submit train",
        "frontier-submit final",
        entrypoint,
    ):
        require(text in readme, f"Agent README does not name {text!r}")
    require(
        f"{calls} train call" in readme or f"最多{calls}次 train" in readme,
        f"Agent README does not name the {calls}-call train limit",
    )
    legacy_paths = [
        root / "public",
        root / "agent-base/README.md",
        root / "agent-base/bin",
    ]
    if paths.native:  # a task.toml task keeps none of the legacy per-task copies
        check_no_platform_numbers(root)
        check_statement_structure(root)
        legacy_paths += [
            root / "agent-base",
            root / "upstream.lock.json",
            root / "judge/config.yaml",
        ]
        # No forks of shared interfaces: the submission tools, the edit-scope check and the
        # verifier adapter exist once, in tasks/shared. A task differs only through task.toml
        # and its judge hooks (judge/evaluator.py: evaluate, optional check_tree).
        # The one exception is a standalone Verifier (verifier/verify.py) that a task.toml
        # selects with its own [verifier] command: a judge the shared verifier cannot express
        # (vllm's Driver-orchestrated GPU phases and signed test-sandbox evidence).
        standalone = (
            task_manifest.read_task(root)["verifier"]["command"]
            != task_manifest.defaults()["verifier"]["command"]
        )
        require(
            not standalone or (root / "verifier/verify.py").is_file(),
            "a task.toml [verifier] command other than the shared one needs the task's "
            "standalone Verifier at verifier/verify.py",
        )
        for own in (
            "bin",
            "verifier",
            "starter",
            "judge/submission_patch.py",
            "judge/edit_scope.py",
            "judge/fcs2_verifier.py",
            "judge/harbor",
            "judge/isolate.py",
            "judge/candidate_sandbox.py",
        ):
            if own == "verifier" and standalone:
                continue
            require(
                not (root / own).exists(),
                f"{own} forks shared code or layout; use tasks/shared and task.toml instead",
            )
    for legacy in legacy_paths:
        require(not legacy.exists(), f"legacy public source path: {legacy}")
    submission_kind = policy["metadata"].get("runtime", {}).get("submission_kind")
    if paths.repo.is_dir():
        for helper in ("prepare_workspace.sh", "make_submission.sh"):
            regular(paths.agent_tool(helper))
        profile = "patch-workspace"
    elif submission_kind == "patch":
        # A single-file (or few-file) answer submitted as a diff against a
        # published starter the Verifier reconstructs from. The starter holds the
        # answer files; the entrypoint is the patch itself.
        for helper in ("prepare_workspace.sh", "make_submission.sh"):
            regular(paths.agent_tool(helper))
        require(
            str(entrypoint) == "payload/solution.patch",
            "a patch submission must use the payload/solution.patch entrypoint",
        )
        allowed = policy["metadata"]["runtime"].get("allowed_paths")
        require(
            isinstance(allowed, list) and bool(allowed),
            "a patch submission must declare its allowed answer paths",
        )
        for answer in allowed:
            require(
                isinstance(answer, str) and bool(answer),
                "allowed answer paths must be non-empty strings",
            )
            if not answer.endswith("/"):
                regular(paths.starter / relative_path(answer))
        profile = "patch-starter"
    else:
        regular(paths.starter / entrypoint)
        profile = "direct-payload"
    command = policy["verifier"].get("command")
    require(
        isinstance(command, list)
        and bool(command)
        and all(isinstance(part, str) and part for part in command),
        "explicit Verifier command required",
    )
    regular(paths.verifier_entry)
    try:
        extension, hashes = dataset_hashes(root, policy)
        for mode in ("train", "eval"):
            path = dataset_path(root, mode, extension)
            if extension == "json":
                selector = document(path)
                require(
                    selector.get("problem_id") == root.name
                    and selector.get("mode") == mode,
                    f"dataset identity mismatch: {mode}",
                )
            else:
                check_tar_dataset(path)
    except ValueError as error:
        raise LayoutFailure(str(error)) from error
    require(
        hashes["train_sha256"] != hashes["eval_sha256"],
        "train and eval must be distinct source objects",
    )
    for role in ("agent-base", "verifier"):
        check_dockerfile(root, role)
    gpu = policy.get("gpu", {})
    require(isinstance(gpu, dict), "GPU policy must be an object")
    for role, field in (("agent-gpu", "agent"), ("verifier-gpu", "verifier")):
        quota = gpu.get(field, {})
        require(isinstance(quota, dict), f"invalid {role} quota")
        enabled = bool(
            quota.get("max_instances" if field == "agent" else "max_gpu_instances")
        )
        require(enabled == paths.has_role(role), f"{role} source and quota disagree")
        if enabled:
            check_dockerfile(root, role)
            if role == "verifier-gpu" and paths.native:
                check_gpu_adapter_pins(root)
            if role == "agent-gpu":
                accelerator = policy["metadata"].get("accelerator", {})
                require(
                    isinstance(accelerator, dict)
                    and accelerator.get("agent_gpu_interface")
                    in {"frontier.gpu-helper/v2", "frontier.gpu-helper/v3"},
                    "Agent GPU task must declare Candidate helper compatibility",
                )
                require(
                    not (task / "bin/frontier-gpu").exists(),
                    "legacy task-owned GPU client remains",
                )
    check_sidecars(root, policy)
    check_ssh_session_budgets(policy)
    check_beat_spec(root, evaluation)
    if not paths.native:  # provenance of a task.toml task lives in docs/task-authoring/
        check_inventory(root)
    return profile


def sandbox_relaxations(root: Path) -> list[str]:
    """The isolate layers in judge/*.py that relax the shared boundary, for review: each
    direct Layer subclass with a non-empty `relaxes`, which must be a literal tuple of strings."""
    found = []
    for path in sorted((root / "judge").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_bytes(), str(path))):
            if not isinstance(node, ast.ClassDef) or not any(
                getattr(base, "id", getattr(base, "attr", None)) == "Layer"
                for base in node.bases
            ):
                continue
            for statement in node.body:
                if not (
                    isinstance(statement, ast.Assign)
                    and any(
                        getattr(t, "id", None) == "relaxes" for t in statement.targets
                    )
                ):
                    continue
                try:
                    relaxes = ast.literal_eval(statement.value)
                except ValueError:
                    relaxes = None
                require(
                    isinstance(relaxes, tuple)
                    and all(isinstance(r, str) for r in relaxes),
                    f"judge/{path.name}: {node.name}.relaxes must be a literal tuple of "
                    "strings",
                )
                if relaxes:
                    found.append(
                        f"judge/{path.name}:{node.name} relaxes {', '.join(relaxes)}"
                    )
    return found


def check_shared_from(ref: str, root: Path = task_manifest.REPO) -> None:
    """tasks/shared must be exactly `ref`'s (task-format/landing against origin/main): shared
    infrastructure changes on main only, so a staging branch that edits or lags it builds bases
    no other branch has."""
    listed = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", ref, "--", "tasks/shared"],
        capture_output=True,
        text=True,
    )
    require(
        listed.returncode == 0,
        f"cannot compare tasks/shared with {ref}: {listed.stderr.strip()}",
    )
    differing = listed.stdout.split()
    require(
        not differing,
        f"tasks/shared differs from {ref}: {', '.join(differing)}; "
        f"merge {ref} (shared changes land on main first)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems-root",
        type=Path,
        action="append",
        help="a problems directory to check (default: every task layer's)",
    )
    parser.add_argument(
        "--shared-from",
        metavar="REF",
        help="also require tasks/shared to equal REF's (e.g. origin/main)",
    )
    options = parser.parse_args()
    failures = 0
    if options.shared_from:
        try:
            check_shared_from(options.shared_from)
        except LayoutFailure as error:
            failures += 1
            print(f"FAIL tasks/shared: {error}", file=sys.stderr)
    tasks = []
    roots = options.problems_root or [
        layer.problems for layer in task_layers.layers() if layer.problems.is_dir()
    ]
    for path in sorted(path for root in roots for path in root.iterdir()):
        if not path.is_dir():
            continue
        if not task_manifest.is_task_dir(path) and any(
            task_manifest.is_task_dir(child)
            for child in path.iterdir()
            if child.is_dir()
        ):
            tasks.extend(sorted(child for child in path.iterdir() if child.is_dir()))
        else:
            tasks.append(path)
    try:
        check_bases()
    except LayoutFailure as error:
        failures += 1
        print(f"FAIL tasks/shared/bases.json: {error}", file=sys.stderr)
    for task in tasks:
        try:
            profile = validate_task(task)
            relaxations = sandbox_relaxations(task)
        except (LayoutFailure, OSError) as error:
            failures += 1
            print(f"FAIL {task.name}: {error}", file=sys.stderr)
        else:
            native = " (task.toml)" if task_manifest.is_native(task) else ""
            print(f"PASS {task.name}: {profile}{native}")
            for relaxation in relaxations:
                print(f"RELAXES {task.name}: {relaxation}")
    print(f"{len(tasks) - failures}/{len(tasks)} task layouts valid")
    return int(bool(failures) or not tasks)


if __name__ == "__main__":
    raise SystemExit(main())
