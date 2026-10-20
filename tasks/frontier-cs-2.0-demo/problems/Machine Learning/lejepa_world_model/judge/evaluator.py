"""Evaluator for the SIGReg objective reconstruction task."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import yaml

import candidate_failure
from sandbox_runner import STATEFUL_LOSS_EXIT
from submission_patch import apply_submission_patch


TASK_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((TASK_DIR / "config.yaml").read_text(encoding="utf-8"))
REFERENCE_PATCH = TASK_DIR / "reference.patch"
SANDBOX_RUNNER = TASK_DIR / "sandbox_runner.py"
ENTRYPOINT = "experiments/run.py"
UPSTREAM_ROOT = Path(".")
SPEC_METRICS = ("r2_zh", "r2_hz", "orth_err_normalized")
# Paper Appendix H.5 bound/recovery quantities, computed by the unchanged
# upstream metrics.py on the same final evaluation set (beat spec v2).
H5_METRICS = ("epsilon", "delta", "procrustes_mse")
BEAT_METRICS = SPEC_METRICS + H5_METRICS
FINAL_SEEDS = (104729, 130363, 155921)
DEV_SEEDS = (7331, 7332, 7333)
RUN_TIMEOUT_SECONDS = 150
FORBIDDEN_CALLS = {
    "open", "eval", "exec", "compile", "__import__", "breakpoint",
    "getattr", "setattr", "delattr", "vars", "globals", "locals", "input",
    "exit", "quit", "help", "license", "credits",
}
FORBIDDEN_NAMES = FORBIDDEN_CALLS | {
    "__builtins__", "os", "sys", "subprocess", "importlib", "builtins",
    "SystemExit", "KeyboardInterrupt", "type", "object",
}
SAFE_BUILTIN_CALLS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "isinstance", "len", "list", "map", "max", "min", "next",
    "pow", "range", "reversed", "round", "set", "sorted", "sum", "super",
    "tuple", "zip", "RuntimeError", "ValueError", "TypeError",
    "NotImplementedError",
}
DANGEROUS_ATTRIBUTES = {
    "os", "sys", "subprocess", "builtins", "importlib", "serialization",
    "modules", "module", "system", "popen", "fork", "forkpty", "vfork",
    "spawn", "_exit", "execv", "execve", "execl", "execlp", "execvp",
    "open", "fdopen", "listdir", "scandir", "walk", "chdir", "getcwd",
    "environ", "getenv", "putenv", "remove", "unlink", "rename", "replace",
    "mkdir", "makedirs", "rmdir", "removedirs", "read_text", "write_text",
    "read_bytes", "write_bytes", "tofile", "dump", "load", "save",
    "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "cr_await",
    "ag_frame", "ag_code", "ag_await", "f_globals", "f_locals", "f_builtins",
    "f_back", "f_code", "f_trace", "tb_frame", "tb_next", "mro",
}
SAFE_TORCH_TOP = {
    "Tensor", "Size", "device", "dtype", "finfo", "iinfo", "pi",
    "float16", "bfloat16", "float32", "float64", "int32", "int64", "bool",
    "tensor", "as_tensor", "arange", "linspace", "logspace", "zeros", "ones",
    "empty", "full", "zeros_like", "ones_like", "empty_like", "full_like",
    "rand", "randn", "randint", "randperm", "rand_like", "randn_like", "eye",
    "cat", "concat", "stack", "vstack", "hstack", "split", "chunk", "unbind",
    "abs", "square", "pow", "sqrt", "rsqrt", "exp", "expm1", "log", "log1p",
    "sin", "cos", "tan", "tanh", "sigmoid", "erf", "erfc", "softmax",
    "logsumexp", "sum", "mean", "var", "std", "cov", "amin", "amax",
    "min", "max", "minimum", "maximum", "clamp", "where", "isfinite",
    "isnan", "isinf", "all", "any", "sort", "argsort", "topk", "unique",
    "matmul", "mm", "bmm", "mv", "dot", "inner", "outer", "einsum",
    "triu", "tril", "diag", "diagonal", "trace", "kron", "cdist", "pdist",
    "no_grad", "enable_grad", "is_grad_enabled", "manual_seed",
}
SAFE_NN = {"Module", "Parameter", "ParameterList", "ModuleList", "Identity"}
SAFE_FUNCTIONAL = {
    "normalize", "mse_loss", "l1_loss", "cosine_similarity", "softmax",
    "log_softmax", "relu", "gelu", "silu", "tanh", "sigmoid", "pad",
}
SAFE_LINALG = {
    "norm", "vector_norm", "matrix_norm", "qr", "svd", "svdvals", "solve",
    "lstsq", "inv", "pinv", "cholesky", "eig", "eigh", "eigvals", "eigvalsh",
    "det", "slogdet", "matrix_rank", "matrix_exp", "cond", "cross", "householder_product",
}
SAFE_SPECIAL = {
    "erf", "erfc", "erfcx", "expit", "gammaln", "log_ndtr", "ndtr",
    "entr", "xlog1py", "xlogy", "sinc", "softmax", "log_softmax",
}
SAFE_MATH = {
    "pi", "e", "tau", "inf", "nan", "sqrt", "exp", "expm1", "log", "log1p",
    "sin", "cos", "tan", "tanh", "erf", "erfc", "gamma", "lgamma", "floor",
    "ceil", "fabs", "isfinite", "isnan", "isinf", "prod",
}
FRAMEWORK_TYPES = {
    "torch.Tensor", "torch.Size", "torch.device", "torch.dtype",
    "torch.nn.Module", "torch.nn.Parameter", "torch.nn.ParameterList",
    "torch.nn.ModuleList", "torch.nn.Identity", "torch.autograd.Function",
}
STATEFUL_LOSS_MESSAGE = "invalid submission: a scored loss carries state between calls"
_REFERENCE_CACHE: dict[tuple[bool, bool], list[dict[str, float]]] = {}


def _zero(reason: str) -> tuple[float, float, str, dict]:
    return 0.0, 0.0, reason[:240], {}


def _suite() -> tuple[bool, bool, tuple[int, ...]]:
    final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
    test_mode = os.environ.get("FRONTIER_EVALUATOR_TEST_MODE") == "1"
    if test_mode:
        return final, True, ((91, 92, 93) if final else (81, 82, 83))
    return final, False, (FINAL_SEEDS if final else DEV_SEEDS)


def _attribute_chain(node: ast.Attribute) -> list[str] | None:
    parts = [node.attr]
    current: ast.expr = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return list(reversed(parts))


def _safe_module_chain(kind: str, parts: list[str]) -> bool:
    tail = parts[1:]
    if not tail:
        return True
    if any(part in DANGEROUS_ATTRIBUTES or part.startswith("__") for part in tail):
        return False
    if kind == "torch":
        if tail == ["nn", "functional"]:
            return True
        if len(tail) == 1:
            return tail[0] in SAFE_TORCH_TOP or tail[0] in {"nn", "linalg", "special", "autograd"}
        if tail[0] == "nn" and len(tail) == 2:
            return tail[1] in SAFE_NN
        if tail[:2] == ["nn", "functional"] and len(tail) == 3:
            return tail[2] in SAFE_FUNCTIONAL
        if tail[0] == "linalg" and len(tail) == 2:
            return tail[1] in SAFE_LINALG
        if tail[0] == "special" and len(tail) == 2:
            return tail[1] in SAFE_SPECIAL
        return tail in (["autograd", "Function"], ["autograd", "grad"])
    if kind == "nn":
        return len(tail) == 1 and tail[0] in SAFE_NN
    if kind == "F":
        return len(tail) == 1 and tail[0] in SAFE_FUNCTIONAL
    if kind == "math":
        return len(tail) == 1 and tail[0] in SAFE_MATH
    # Local helpers must expose symbols through ``from ... import symbol``.
    # Allowing attributes on an imported helper module would let it relay an
    # otherwise forbidden framework object under an untracked alias.
    return not tail


def _policy_check(tree: Path) -> tuple[bool, str]:
    root = tree / UPSTREAM_ROOT / "experiments" / "lejepa_id"
    paths = [root / "losses.py"]
    helper = root / "frontier_solution"
    if helper.exists():
        for item in sorted(helper.rglob("*")):
            if item.is_symlink():
                return False, "solution helper namespace may not contain symlinks"
            if item.is_file() and item.suffix != ".py":
                return False, "solution helper namespace accepts Python source files only"
            if item.is_file():
                paths.append(item)
    for path in paths:
        is_helper = helper in path.parents
        try:
            source = path.read_text(encoding="utf-8")
            parsed = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            return False, f"invalid Python source: {str(exc).splitlines()[0]}"
        aliases: dict[str, str] = {}
        defined: set[str] = set()
        parents = {child: parent for parent in ast.walk(parsed)
                   for child in ast.iter_child_nodes(parent)}
        for item in ast.walk(parsed):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(item.name)
            if isinstance(item, ast.arg):
                defined.add(item.arg)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                defined.add(item.id)
            if isinstance(item, (ast.Import, ast.ImportFrom)):
                for alias in item.names:
                    defined.add(alias.asname or alias.name.split(".", 1)[0])
                    if isinstance(item, ast.Import):
                        bound = alias.asname or alias.name.split(".", 1)[0]
                        aliases[bound] = {"torch": "torch", "torch.nn": "nn",
                                          "torch.nn.functional": "F", "math": "math"}.get(alias.name, "local")
                    else:
                        aliases[alias.asname or alias.name] = "local"
        for node in ast.walk(parsed):
            binding = None
            if isinstance(node, ast.alias):
                binding = node.asname or node.name.split(".", 1)[0]
            elif isinstance(node, ast.arg):
                binding = node.arg
            elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
                binding = node.name
            elif isinstance(node, ast.MatchMapping):
                binding = node.rest
            elif isinstance(node, ast.ClassDef):
                binding = node.name
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                binding = node.name
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                binding = node.id
            if binding and binding.startswith("__") and (
                not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                or binding in {"__annotations__", "__builtins__"}
            ):
                return False, "solution may not bind interpreter-managed namespace entries"
            if binding in aliases and not isinstance(node, ast.alias):
                return False, "solution may not rebind or shadow imported names"
            # Contexts cover nested unpacking, loops, comprehensions, with-targets,
            # deletion and ordinary assignment without a separate syntax allowlist.
            if isinstance(node, (ast.Attribute, ast.Subscript)) \
                    and isinstance(node.ctx, (ast.Store, ast.Del)):
                current = node.value
                while isinstance(current, (ast.Attribute, ast.Subscript)):
                    current = current.value
                if isinstance(current, ast.Name) and current.id in aliases:
                    return False, "solution may not mutate imported framework objects"
            if isinstance(node, ast.ClassDef) and node.keywords:
                return False, "solution classes may not use custom metaclasses or class keywords"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == "__mro_entries__":
                return False, "solution classes may not intercept framework base construction"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    allowed = {
                        "torch": ("torch", "torch"),
                        "torch.nn": ("nn", "nn"),
                        "torch.nn.functional": ("F", "F"),
                        "math": ("math", "math"),
                    }.get(alias.name)
                    bound = alias.asname or alias.name.split(".", 1)[0]
                    if allowed is None or bound != allowed[0]:
                        return False, "solution import is outside the explicit safe API"
                    aliases[bound] = allowed[1]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    package = ["lejepa_id", *path.parent.relative_to(root).parts]
                    if node.level > len(package):
                        return False, "solution relative import escapes its helper namespace"
                    module = ".".join(package[:len(package) - node.level + 1]
                                      + (module.split(".") if module else []))
                local = module == "lejepa_id.frontier_solution" or module.startswith("lejepa_id.frontier_solution.")
                if not local:
                    return False, "solution import is outside the explicit safe API"
                for alias in node.names:
                    if alias.name == "*" or alias.name in {"torch", "nn", "F", "math"} \
                            or (alias.asname or alias.name) in {"torch", "nn", "F", "math"} \
                            or alias.name in FORBIDDEN_NAMES or alias.name.startswith("__"):
                        return False, "solution helpers may not relay imported capabilities"
                    aliases[alias.asname or alias.name] = "local"
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id.startswith("__") or node.id in FORBIDDEN_NAMES:
                    return False, f"solution uses disallowed capability name {node.id}"
                if aliases.get(node.id) in {"torch", "nn", "F", "math"}:
                    parent = parents.get(node)
                    if not isinstance(parent, ast.Attribute) or parent.value is not node:
                        return False, "solution may not alias or pass framework modules"
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if (node.func.id not in SAFE_BUILTIN_CALLS
                        and node.func.id not in defined):
                    return False, f"solution call is outside the explicit safe API: {node.func.id}"
            elif isinstance(node, ast.Attribute):
                if (node.attr.startswith("__") and node.attr != "__init__") \
                        or node.attr in DANGEROUS_ATTRIBUTES:
                    return False, "solution uses a disallowed capability attribute"
                chain = _attribute_chain(node)
                if chain and chain[0] in aliases \
                        and not _safe_module_chain(aliases[chain[0]], chain):
                    return False, "solution accesses an attribute outside the explicit safe API"
                if chain and aliases.get(chain[0]) == "torch" and chain[1:] in (
                    ["nn"], ["linalg"], ["special"], ["autograd"], ["nn", "functional"],
                ):
                    parent = parents.get(node)
                    if not isinstance(parent, ast.Attribute) or parent.value is not node:
                        return False, "solution may not alias or pass framework submodules"
                if chain and chain[0] in aliases:
                    base = {"torch": "torch", "nn": "torch.nn"}.get(aliases[chain[0]])
                    if base and ".".join([base, *chain[1:]]) in FRAMEWORK_TYPES:
                        parent = parents.get(node)
                        safe = (isinstance(parent, ast.Call) and parent.func is node) or (
                            isinstance(parent, ast.ClassDef) and node in parent.bases)
                        safe = safe or (isinstance(parent, ast.arg) and parent.annotation is node)
                        safe = safe or (isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                                        and parent.returns is node)
                        safe = safe or (isinstance(parent, ast.AnnAssign) and parent.annotation is node)
                        if not safe:
                            return False, "framework types are limited to constructors, bases, and direct annotations"
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                return False, "solution may not use global or nonlocal capability aliases"
    return True, ""


def _seal_solver_tree(tree: Path) -> None:
    """Make submitted source readable but immutable to the sandbox UID."""
    os.chmod(tree.parent, 0o711)
    staged_patch = tree.parent / "submission.patch"
    if staged_patch.exists():
        os.chmod(staged_patch, 0o600)
    for directory, names, files in os.walk(tree, followlinks=False):
        os.chmod(directory, 0o555)
        for name in names + files:
            path = Path(directory) / name
            if not path.is_symlink():
                os.chmod(path, 0o555 if path.is_dir() else 0o444)


def _cleanup_tree(root: Path) -> None:
    # The unprivileged verifier owns the sealed directories and must unseal them.
    for directory, _names, _files in os.walk(root, followlinks=False):
        os.chmod(directory, 0o700)
    shutil.rmtree(root)


def _preflight(tree: Path) -> tuple[bool, str]:
    experiments = tree / UPSTREAM_ROOT / "experiments"
    env = _run_env(tree)
    try:
        proc = subprocess.run(
            [sys.executable, str(SANDBOX_RUNNER), "preflight", str(experiments)],
            capture_output=True, text=True, timeout=30, env=env, cwd=experiments,
        )
    except subprocess.TimeoutExpired:
        return False, "loss interface preflight timed out"
    if proc.returncode:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, f"loss interface preflight failed: {detail[-1][:160] if detail else 'error'}"
    return True, ""


def _purity_check(tree: Path) -> tuple[bool, str, bool]:
    """Run the sandboxed statelessness check; the last field marks a stateful loss."""
    experiments = tree / UPSTREAM_ROOT / "experiments"
    try:
        # The replayed calls are a subset of one training run's, so its limit applies.
        proc = subprocess.run(
            [sys.executable, str(SANDBOX_RUNNER), "purity", str(experiments)],
            capture_output=True, text=True, timeout=RUN_TIMEOUT_SECONDS,
            env=_run_env(tree), cwd=experiments,
        )
    except subprocess.TimeoutExpired:
        return False, "loss purity check timed out", False
    if proc.returncode == STATEFUL_LOSS_EXIT:
        return False, STATEFUL_LOSS_MESSAGE, True
    if proc.returncode:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, f"loss purity check failed: {detail[-1][:160] if detail else 'error'}", False
    return True, "", False


def _run_env(tree: Path) -> dict[str, str]:
    scratch = tree.parent / ".runtime-scratch"
    scratch.mkdir(mode=0o700, exist_ok=True)
    if os.geteuid() == 0:
        os.chown(scratch, 65534, 65534)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(tree / UPSTREAM_ROOT / "experiments"),
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "FRONTIER_SANDBOX_SCRATCH": str(scratch),
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "XDG_CACHE_HOME": str(scratch),
    })
    return env


def _run_one(tree: Path, seed: int, test_mode: bool) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="lejepa-run-") as tmp_name:
        tmp = Path(tmp_name)
        experiments = tree / UPSTREAM_ROOT / "experiments"
        cfg = yaml.safe_load((experiments / "configs" / "2d.yaml").read_text())
        cfg["out"] = str(tmp / "results")
        cfg["steps"] = 3 if test_mode else 1000
        cfg["log_every"] = 3 if test_mode else 250
        if test_mode:
            cfg["num_eval"] = 128
            cfg["batch_size"] = 32
            cfg["runs"]["spiral_lejepa"]["hidden"] = 16
        config_path = tmp / "workload.yaml"
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        # The confined child writes its own run artifacts; the evaluator reads
        # them after it exits, including when both processes use the same UID.
        if os.geteuid() == 0:
            os.chown(tmp, 65534, 65534)
        os.chmod(tmp, 0o700)
        os.chmod(config_path, 0o444)
        cmd = [
            sys.executable, str(SANDBOX_RUNNER), "train", str(experiments),
            "--config", str(config_path),
            "--run", "spiral_lejepa", "--seed", str(seed),
        ]
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, cwd=experiments, env=_run_env(tree),
                capture_output=True, text=True,
                timeout=30 if test_mode else RUN_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"training workload timed out after {exc.timeout:g} seconds") from None
        elapsed = time.perf_counter() - started
        if proc.returncode:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            raise RuntimeError(
                f"training workload failed: {detail[-1][:160] if detail else 'nonzero exit'}"
            )
        result_path = tmp / "results" / f"spiral_lejepa_seed={seed}.json"
        pt_path = result_path.with_suffix(".pt")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            row = {name: float(payload[name]) for name in BEAT_METRICS}
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"training produced invalid metrics: {exc}") from None
        if not all(math.isfinite(value) for value in row.values()):
            raise RuntimeError("training produced non-finite metrics")
        if any(row[name] < 0 for name in H5_METRICS):
            raise RuntimeError("training produced invalid bound metrics")
        # Do not unpickle candidate-controlled data.  A genuine torch.save
        # artifact is a nontrivial zip archive containing data.pkl.
        try:
            if pt_path.stat().st_size < 1024 or not zipfile.is_zipfile(pt_path):
                raise RuntimeError("training produced an invalid PyTorch result")
            with zipfile.ZipFile(pt_path) as archive:
                if not any(name.endswith("/data.pkl") or name == "data.pkl"
                           for name in archive.namelist()):
                    raise RuntimeError("training produced an invalid PyTorch result")
        except OSError as exc:
            raise RuntimeError(f"training did not produce its PyTorch result: {exc}") from None
        row["seconds"] = elapsed
        return row


def _run_suite(tree: Path, seeds: tuple[int, ...], test_mode: bool) -> list[dict[str, float]]:
    return [_run_one(tree, seed, test_mode) for seed in seeds]


def _reference_suite(final: bool, test_mode: bool, seeds: tuple[int, ...]) -> list[dict[str, float]]:
    key = (final, test_mode)
    if key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]
    applied = apply_submission_patch(REFERENCE_PATCH, TASK_DIR, CONFIG)
    if not applied.ok or applied.workdir is None:
        raise RuntimeError(f"internal reference patch error: {applied.message}")
    try:
        _seal_solver_tree(applied.workdir)
        rows = _run_suite(applied.workdir, seeds, test_mode)
    finally:
        _cleanup_tree(applied.workdir.parent)
    _REFERENCE_CACHE[key] = rows
    return rows


def _delta_clears(candidate: float, reference: float) -> bool:
    """Strictly smaller alignment gap, except at the shared 0 clamp floor.

    metrics.py defines delta = max(L_h - 2(1-rho) tr(Cov h), 0). When the
    reference is already at that floor a strictly smaller value is impossible,
    so exact equality at 0 clears; any positive tie or larger value fails.
    """
    return candidate < reference or (candidate == 0.0 and reference == 0.0)


def _decisions(cand: dict[str, float], ref: dict[str, float]) -> dict[str, bool]:
    return {
        "r2_zh": bool(cand["r2_zh"] > ref["r2_zh"]),
        "r2_hz": bool(cand["r2_hz"] > ref["r2_hz"]),
        "orth_err_normalized": bool(
            cand["orth_err_normalized"] < ref["orth_err_normalized"]
        ),
        "epsilon": bool(cand["epsilon"] < ref["epsilon"]),
        "delta": bool(_delta_clears(cand["delta"], ref["delta"])),
        "procrustes_mse": bool(cand["procrustes_mse"] < ref["procrustes_mse"]),
    }


def _quality_score(rows: list[dict[str, float]]) -> float:
    values = []
    for row in rows:
        r2_zh = min(1.0, max(0.0, row["r2_zh"]))
        r2_hz = min(1.0, max(0.0, row["r2_hz"]))
        orth_quality = min(1.0, max(0.0, 1.0 - row["orth_err_normalized"]))
        values.append(35.0 * r2_zh + 35.0 * r2_hz + 20.0 * orth_quality)
    return sum(values) / len(values)


def _scores(candidate: list[dict[str, float]], reference: list[dict[str, float]]) -> tuple[float, float]:
    candidate_seconds = sum(row["seconds"] for row in candidate)
    reference_seconds = sum(row["seconds"] for row in reference)
    efficiency = min(10.0, 5.0 * reference_seconds / max(candidate_seconds, 1e-9))
    candidate_score = min(100.0, max(0.0, _quality_score(candidate) + efficiency))
    reference_score = min(100.0, max(0.0, _quality_score(reference) + 5.0))
    # Upstream torch.linalg.lstsq can vary in the sixth decimal place on its
    # first CPU call. Public scores are resolved to 1e-4 points; raw metrics and
    # the strict per-workload beat decisions remain unrounded.
    return round(candidate_score, 4), round(reference_score, 4)


def prepare() -> dict[str, Any]:
    """Cache the paper reference for the active hidden suite."""
    final, test_mode, seeds = _suite()
    rows = _reference_suite(final, test_mode, seeds)
    return {"ready": True, "workloads": len(rows), "suite": "final" if final else "development"}


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    applied = None
    try:
        applied = apply_submission_patch(solution_path, TASK_DIR, CONFIG)
        if not applied.ok or applied.workdir is None:
            return _zero(applied.message or "invalid submission patch")
        ok, reason = _policy_check(applied.workdir)
        if not ok:
            return _zero(reason)
        _seal_solver_tree(applied.workdir)
        ok, reason = _preflight(applied.workdir)
        if not ok:
            return _zero(reason)
        final, test_mode, seeds = _suite()
        ok, reason, stateful = _purity_check(applied.workdir)
        if not ok:
            rejected = _zero(reason)
            if stateful and not final:
                # The verifier admits this to train feedback only as the sole
                # metric of a scored zero; a final rejection stays an empty zero.
                rejected[3]["candidate_diagnostic"] = candidate_failure.diagnostic("stateful_loss")
            return rejected

        reference = _reference_suite(final, test_mode, seeds)
        submitted_bytes = Path(solution_path).read_bytes()
        if submitted_bytes == REFERENCE_PATCH.read_bytes():
            candidate = [dict(row) for row in reference]
        else:
            candidate = _run_suite(applied.workdir, seeds, test_mode)
        score, reference_score = _scores(candidate, reference)

        audit = []
        clears: dict[str, list[bool]] = {name: [] for name in BEAT_METRICS}
        for index, (cand, ref) in enumerate(zip(candidate, reference), start=1):
            decisions = _decisions(cand, ref)
            audit.append({"workload": f"gaussian_2d_spiral_{index:02d}", "metrics": decisions})
            for name in BEAT_METRICS:
                clears[name].append(decisions[name])
        beats_reference = int(all(v for row in audit for v in row["metrics"].values()))

        metrics: dict[str, Any] = {
            "r2_zh": sum(row["r2_zh"] for row in candidate) / len(candidate),
            "r2_hz": sum(row["r2_hz"] for row in candidate) / len(candidate),
            "orth_err_normalized": sum(row["orth_err_normalized"] for row in candidate) / len(candidate),
            "beat_spec_workload_passes": audit,
            "beats_reference": beats_reference,
            "reference_score": reference_score,
            "margin": score - reference_score,
            "reference_r2_zh": [row["r2_zh"] for row in reference],
            "reference_r2_hz": [row["r2_hz"] for row in reference],
            "reference_orth_err_normalized": [row["orth_err_normalized"] for row in reference],
            "r2_zh_clear_by_workload": clears["r2_zh"],
            "r2_hz_clear_by_workload": clears["r2_hz"],
            "orth_err_normalized_clear_by_workload": clears["orth_err_normalized"],
        }
        for name in H5_METRICS:
            metrics[name] = sum(row[name] for row in candidate) / len(candidate)
            metrics[f"reference_{name}"] = [row[name] for row in reference]
            metrics[f"{name}_clear_by_workload"] = clears[name]
        return score, score, "completed authentic LeJEPA training workloads", metrics
    except Exception as exc:
        return _zero(f"evaluation failed: {type(exc).__name__}: {exc}")
    finally:
        if applied is not None and applied.workdir is not None:
            _cleanup_tree(applied.workdir.parent)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else str(REFERENCE_PATCH)
    score, unbounded, message, metrics = evaluate(path)
    print(json.dumps({
        "score": score,
        "score_unbounded": unbounded,
        "message": message,
        "metrics": metrics,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
