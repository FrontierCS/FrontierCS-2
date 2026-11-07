"""Judge-side source and feedback checks; never import submitted Python."""

from __future__ import annotations

import ast
import io
import math
from pathlib import Path
import tokenize

POLICY = "diffusion_policy/policy/diffusion_transformer_lowdim_policy.py"
HELPERS = "diffusion_policy/sampler_extensions"
MAX_HELPER_FILES = 16
MAX_HELPER_BYTES = 65536
MAX_POLICY_BYTES = 131072
DEV_FIELDS = (
    "dev_mean_episode_max_reward", "dev_episode_success_rate",
    "dev_mean_predict_action_seconds", "dev_total_workload_wall_seconds",
)
METRICS = (
    "mean_episode_max_reward", "episode_success_rate",
    "mean_predict_action_seconds", "complete_workload_wall_seconds",
)


class InvalidSubmission(ValueError):
    pass


class TaskIntegrityError(RuntimeError):
    pass


class CandidateTimeout(RuntimeError):
    """The candidate's own sampler, episode process or the suite budget ran out: a scored zero."""


def method_region(source: str) -> tuple[ast.Module, ast.FunctionDef, int, int]:
    try:
        tree = ast.parse(source)
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "DiffusionTransformerLowdimPolicy"]
        if len(classes) != 1:
            raise ValueError("class")
        methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef)
                   and n.name == "conditional_sample"]
        if len(methods) != 1:
            raise ValueError("method")
        method = methods[0]
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        in_header = False
        depth = 0
        header_end = None
        for token in tokens:
            if token.start == (method.lineno, method.col_offset) and token.string == "def":
                in_header = True
            if not in_header:
                continue
            if token.type == tokenize.OP:
                if token.string in "([{":
                    depth += 1
                elif token.string in ")]}":
                    depth -= 1
                elif token.string == ":" and depth == 0:
                    header_end = token.end[0]
                    break
        if header_end is None or method.body[0].lineno <= header_end:
            raise ValueError("header")
        lines = source.splitlines(keepends=True)
        start = sum(map(len, lines[:header_end]))
        end = sum(map(len, lines[:method.end_lineno]))
        return tree, method, start, end
    except (SyntaxError, ValueError, tokenize.TokenError, RecursionError) as exc:
        raise InvalidSubmission("The required sampler interface must remain valid Python.") from exc


def verify_method_only(pristine: bytes, submitted: bytes) -> None:
    try:
        original = pristine.decode("utf-8")
        changed = submitted.decode("utf-8")
    except UnicodeError as exc:
        raise InvalidSubmission("Source files must be UTF-8 Python.") from exc
    _, a, a_start, a_end = method_region(original)
    _, b, b_start, b_end = method_region(changed)
    if (original[:a_start] != changed[:b_start]
            or original[a_end:] != changed[b_end:]
            or ast.dump(a.args) != ast.dump(b.args)):
        raise InvalidSubmission("Only the conditional_sample body may be modified.")


def check_source(root: Path, pristine: Path) -> None:
    """The source rules task.toml [submission] cannot express.

    The shared edit scope already admits only the policy file and new, non-executable `.py`
    helpers under HELPERS (no links, mode changes, deletions or dot paths) and rebuilds the
    judged tree from the pristine starter. Left here: the helper count and byte allowance, the
    policy byte allowance, the conditional_sample-body-only rule and helpers that parse.
    """
    helper_root = root / HELPERS
    helpers = sorted(p for p in helper_root.rglob("*") if p.is_file()) if helper_root.is_dir() else []
    if (len(helpers) > MAX_HELPER_FILES
            or sum(p.stat().st_size for p in helpers) > MAX_HELPER_BYTES):
        raise InvalidSubmission("The helper file or byte allowance was exceeded.")
    policy = root / POLICY
    if policy.stat().st_size > MAX_POLICY_BYTES:
        raise InvalidSubmission("The sampler source byte allowance was exceeded.")
    verify_method_only((pristine / POLICY).read_bytes(), policy.read_bytes())
    for helper in helpers:
        try:
            ast.parse(helper.read_bytes())
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise InvalidSubmission("A helper is not valid Python.") from exc


def suite_mode(environ: dict[str, str]) -> str:
    return "final" if environ.get("FRONTIER_SUBMISSION_ROLE") == "final" else "dev"


def public_feedback(mode: str, *, complete: bool, candidate: dict | None = None,
                    progress: dict | None = None) -> dict:
    """Return an allowlist projection; private rows never cross this boundary."""
    if mode not in {"dev", "final"}:
        raise ValueError("Invalid suite mode.")
    if mode == "final":
        return {}
    result = {key: None for key in DEV_FIELDS}
    if not complete:
        return result
    candidate = candidate or {}
    for key in DEV_FIELDS:
        value = candidate.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise TaskIntegrityError("Incomplete candidate metrics cannot become feedback.")
        if value < 0 or (key in DEV_FIELDS[:2] and value > 1):
            raise TaskIntegrityError("Candidate metric is outside its declared range.")
        result[key] = float(value)
    if progress is not None:
        if set(progress) != set(METRICS):
            raise TaskIntegrityError("Criterion progress must cover the declared metrics.")
        safe = {}
        totals = set()
        for name in METRICS:
            row = progress[name]
            passed, total = row.get("passed"), row.get("total")
            if (type(passed) is not int or type(total) is not int
                    or total <= 0 or not 0 <= passed <= total):
                raise TaskIntegrityError("Invalid aggregate criterion counts.")
            totals.add(total)
            safe[name] = {"passed": passed, "failed": total - passed, "total": total,
                          "pass_percent": 100.0 * passed / total,
                          "fail_percent": 100.0 * (total - passed) / total}
        if len(totals) != 1:
            raise TaskIntegrityError("Criteria must use the same complete workload set.")
        result["beat_spec_progress"] = safe
    return result

