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
import json
import math
import re
import shlex
import sys
from pathlib import Path

from native_datasets import check_tar_dataset, dataset_hashes, dataset_path

DEFAULT_PROBLEMS_ROOT = (
    Path(__file__).resolve().parents[1] / "tasks/frontier-cs-2.0-demo/problems"
)
# Reviewed public-copy exceptions, keyed by (task, image role). None are needed
# by the current catalog; the mechanism stays for imported tasks.
PUBLIC_COPY_REVIEWS: dict[tuple[str, str], Path] = {}
FEEDBACK_STATEMENT_REVIEWS = {
    "ckks_composable_noise": (
        Path(__file__).resolve().parent
        / "reference/ckks-feedback-statement-reviewed-v1.json"
    ),
}


class LayoutFailure(ValueError):
    pass


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
    require(not trailing or path.is_dir(), f"COPY trailing slash needs directory: {source}")
    return source


def check_reviewed_public_copies(root: Path, role: str, sources: set[str]) -> None:
    """Bind exceptional public inputs to a reviewed recipe and complete bytes.

    This is an Agent visibility review, not a generic runtime-context allowlist
    or an assertion of image publication rights. The full task inventory is
    still independently checked by validate_task.
    """
    review_path = PUBLIC_COPY_REVIEWS.get((root.name, role))
    require(review_path is not None, f"non-public Agent COPY sources: {sorted(sources)}")
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
        require(digest(root / name) == expected, f"public COPY binding mismatch: {name}")
    copies = review.get("copies", {})
    require(set(copies) == sources, "unreviewed public COPY source set")
    observed_files, observed_directories = set(), set()
    for source, kind in copies.items():
        path = root / copy_source(root, source)
        require(kind in {"file", "directory"}, "invalid reviewed COPY kind")
        require(path.is_dir() == (kind == "directory"), f"public COPY kind mismatch: {source}")
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
        "beat_spec_progress", "passed", "failed", "total", "pass_percent", "fail_percent",
    ):
        require(name in statement, f"public statement does not describe beat-spec feedback {name}")
    missing = [name for name in visible if name not in statement]
    if not missing:
        return
    review_path = FEEDBACK_STATEMENT_REVIEWS.get(root.name)
    require(review_path is not None, f"public statement does not describe beat-spec feedback {missing}")
    review = document(review_path)
    require(
        review.get("schema") == "preview.reviewed-feedback-statement/v1"
        and review.get("problem_id") == root.name,
        "feedback statement review identity mismatch",
    )
    bindings = review.get("bindings", {})
    require(set(bindings) == {
        "upstream.lock.json", "policy.json", "judge/beat_spec.json",
        "agent-base/frontier/task/README.md", "agent-base/frontier/task/repo/README.md",
    }, "feedback statement review needs exact source bindings")
    for name, expected in bindings.items():
        require(digest(root / name) == expected, f"feedback statement binding mismatch: {name}")
    descriptions = review.get("fields", {})
    require(set(descriptions) == set(visible), "unreviewed feedback field set")
    sentence = review.get("sentence")
    require(
        isinstance(sentence, str) and bool(sentence) and sentence in statement
        and all(isinstance(text, str) and bool(text) and text in sentence
                for text in descriptions.values()),
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


def check_beat_spec_feedback(
    root: Path, spec: dict, names: list[str], visible: list[str]
) -> None:
    """Exercise the shipped, dependency-free feedback boundary without a workload.

    Compile in memory to keep validation from adding __pycache__ to a locked
    source tree. This does not import or execute the private evaluator.
    """
    helper = regular(root / "verifier/beat_spec_feedback.py")
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
        # Grouped tasks measure separate metric subsets at different granularities.
        totals = {name: 2 for name in names}
        groups = spec.get("workload_groups")
        if groups is not None:
            require(isinstance(groups, dict) and bool(groups), "invalid workload groups")
            rows = []
            totals = {}
            for group_name, group in groups.items():
                require(isinstance(group, dict), "invalid workload group")
                counts = group.get("counts")
                require(isinstance(counts, dict), "invalid workload group counts")
                count = counts.get("train")
                metrics = group.get("metrics")
                require(type(count) is int and 0 < count <= 100000,
                        "invalid or excessive grouped feedback probe count")
                require(isinstance(metrics, list) and bool(metrics),
                        "invalid workload group metrics")
                require(len(rows) + count <= 100000,
                        "grouped feedback probe exceeds bounded layout budget")
                for name in metrics:
                    require(name in names and name not in totals,
                            "workload groups must partition declared metrics")
                    totals[name] = count
                first = len(rows)
                rows.extend({"workload": f"private-layout-probe-{first + index}",
                             "group": group_name,
                             "metrics": {name: True for name in metrics}}
                            for index in range(count))
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


def check_beat_spec(root: Path, evaluation: dict) -> None:
    from native_beat_spec import NativeBeatSpecFailure
    from native_beat_spec import check as check_native_beat_spec
    from native_beat_spec import is_native as native_beat_spec_task

    if (root / "judge/beat_spec.json").exists() and native_beat_spec_task(root):
        # A native verifier keeps its reviewed contract; its formal beat spec
        # must transcribe exactly the decisions that verifier emits.
        try:
            check_native_beat_spec(root, evaluation, check_beat_spec_schema)
        except NativeBeatSpecFailure as error:
            raise LayoutFailure(str(error)) from error
        return
    if (
        not (root / "judge/beat_spec.json").exists()
        and root.name == "join_tree_rewriting"
    ):
        from join_tree_contract import NativeContractFailure, check_native_contract

        try:
            check_native_contract(root, evaluation)
        except NativeContractFailure as error:
            raise LayoutFailure(f"reviewed native query protocol: {error}") from error
        return
    spec = document(root / "judge/beat_spec.json")
    names, visible = check_beat_spec_schema(spec, evaluation)
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
    task = root / "agent-base/frontier/task"
    statement = regular(task / "README.md").read_text()
    if (task / "repo/README.md").is_file():
        statement += "\n" + regular(task / "repo/README.md").read_text()
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
MAX_SIDECARS = 250  # Modal's concurrent-sidecar cap per Sandbox (measured)
MAX_SIDECAR_ARG_BYTES = 32 * 1024
# Modal admits sidecars against the parent reservation at this per-container
# floor; the provider default reservation therefore allows four sidecars.
SIDECAR_MIN_CPU_MILLIS = 32
SIDECAR_MIN_MEMORY_BYTES = 32 * 1024 * 1024
DEFAULT_RESERVATION_SIDECARS = 4


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
            and set(profile) <= {"args", "workdir", "phases", "attestation_key", "image"},
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
                isinstance(arg, str) and "\0" not in arg and len(arg) <= MAX_SIDECAR_ARG_BYTES
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
    declared = {f"sidecar-{role}" for role in roles}
    present = {path.name for path in root.glob("sidecar-*") if path.is_dir()}
    require(present == declared, f"sidecar image directories and declarations disagree: {sorted(present ^ declared)}")
    resources = policy.get("gpu_parent_resources", {})
    require(isinstance(resources, dict), "gpu_parent_resources must be an object")
    require(
        set(resources) <= {"agent", "verifier"} and ("gpu_parent_resources" not in policy or resources),
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
    policy = document(root / "policy.json")
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
    task = root / "agent-base/frontier/task"
    readme = regular(task / "README.md").read_text()
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
    for legacy in (
        root / "public",
        root / "agent-base/README.md",
        root / "agent-base/bin",
    ):
        require(not legacy.exists(), f"legacy public source path: {legacy}")
    submission_kind = policy["metadata"].get("runtime", {}).get("submission_kind")
    if (task / "repo").is_dir():
        for helper in ("prepare_workspace.sh", "make_submission.sh"):
            regular(task / "bin" / helper)
        profile = "patch-workspace"
    elif submission_kind == "patch":
        # A single-file (or few-file) answer submitted as a diff against a
        # published starter the Verifier reconstructs from. The starter holds the
        # answer files; the entrypoint is the patch itself.
        for helper in ("prepare_workspace.sh", "make_submission.sh"):
            regular(task / "bin" / helper)
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
                regular(task / "starter" / relative_path(answer))
        profile = "patch-starter"
    else:
        regular(task / "starter" / entrypoint)
        profile = "direct-payload"
    command = policy["verifier"].get("command")
    require(
        isinstance(command, list)
        and bool(command)
        and all(isinstance(part, str) and part for part in command),
        "explicit Verifier command required",
    )
    regular(root / "verifier/verify.py")
    try:
        extension, hashes = dataset_hashes(root, policy)
        for mode in ("train", "eval"):
            path = dataset_path(root, mode, extension)
            if extension == "json":
                selector = document(path)
                require(
                    selector.get("problem_id") == root.name and selector.get("mode") == mode,
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
        require(enabled == (root / role).is_dir(), f"{role} source and quota disagree")
        if enabled:
            check_dockerfile(root, role)
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
    check_inventory(root)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems-root", type=Path, default=DEFAULT_PROBLEMS_ROOT)
    options = parser.parse_args()
    tasks = []
    for path in sorted(options.problems_root.iterdir()):
        if not path.is_dir():
            continue
        if not (path / "policy.json").exists() and any(path.glob("*/policy.json")):
            tasks.extend(sorted(child for child in path.iterdir() if child.is_dir()))
        else:
            tasks.append(path)
    failures = 0
    for task in tasks:
        try:
            profile = validate_task(task)
        except (LayoutFailure, OSError) as error:
            failures += 1
            print(f"FAIL {task.name}: {error}", file=sys.stderr)
        else:
            print(f"PASS {task.name}: {profile}")
    print(f"{len(tasks) - failures}/{len(tasks)} task layouts valid")
    return int(bool(failures) or not tasks)


if __name__ == "__main__":
    raise SystemExit(main())
