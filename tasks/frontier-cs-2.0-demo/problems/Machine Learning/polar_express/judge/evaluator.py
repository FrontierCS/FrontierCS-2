"""Deterministic evaluator for the removed Polar Express numerical routines."""

from __future__ import annotations

import ast
import base64
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Bound trusted QR/SVD work as well as the notebook subprocess.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np

from submission_patch import apply_submission_patch


TASK_DIR = Path(__file__).resolve().parent
SUBSTRATE = TASK_DIR / "harbor" / "app"
REFERENCE_PATCH = TASK_DIR / "reference.patch"
CONFIG: dict[str, Any] = {
    "submission": {
        "kind": "patch",
        "allow_paths": ["polar_express.py", "frontier_solution/"],
        "max_changed_files": 20,
        "max_patch_bytes": 1_048_576,
    }
}

# Both suites have the same two-cases-per-shape stratification and disjoint seeds.
# Each seed draws new orthonormal factors and new singular spectra for every member.
DEV_SEED = 847_221
FINAL_SEED = 3_914_087
WORKLOADS = (
    {"id": "square_0", "shape": [12, 12], "l": 0.010, "steps": 5, "safety": 0.010, "cushion": 0.020},
    {"id": "square_1", "shape": [12, 12], "l": 0.020, "steps": 5, "safety": 0.008, "cushion": 0.015},
    {"id": "tall_0", "shape": [18, 10], "l": 0.030, "steps": 4, "safety": 0.006, "cushion": 0.012},
    {"id": "tall_1", "shape": [18, 10], "l": 0.050, "steps": 4, "safety": 0.004, "cushion": 0.010},
    {"id": "wide_0", "shape": [10, 18], "l": 0.080, "steps": 3, "safety": 0.003, "cushion": 0.008},
    {"id": "wide_1", "shape": [10, 18], "l": 0.120, "steps": 3, "safety": 0.002, "cushion": 0.006},
)

# Beat-spec v3 paper-regime stratum (Section 4.1 / H.1 of the source paper):
# larger matrices, the Muon iteration counts (Figure 5), and singular spectra
# drawn per member from hidden families, so neither the shape nor the step
# count identifies the spectrum. The scalar record keeps the paper's offline
# defaults (l = 1e-3, safety factor 1.01, Implementation 2 cushion). These six
# members are graded by the beat spec only; the continuous score keeps its
# original six-workload definition.
PAPER_CUSHION = 0.02407327424182761
PAPER_WORKLOADS = (
    {"id": "paper_square_0", "shape": [256, 256], "l": 0.001, "steps": 5, "safety": 0.01, "cushion": PAPER_CUSHION},
    {"id": "paper_square_1", "shape": [256, 256], "l": 0.001, "steps": 5, "safety": 0.01, "cushion": PAPER_CUSHION},
    {"id": "paper_wide_0", "shape": [128, 512], "l": 0.001, "steps": 6, "safety": 0.01, "cushion": PAPER_CUSHION},
    {"id": "paper_wide_1", "shape": [128, 512], "l": 0.001, "steps": 6, "safety": 0.01, "cushion": PAPER_CUSHION},
    {"id": "paper_tall_0", "shape": [512, 128], "l": 0.001, "steps": 5, "safety": 0.01, "cushion": PAPER_CUSHION},
    {"id": "paper_tall_1", "shape": [512, 128], "l": 0.001, "steps": 5, "safety": 0.01, "cushion": PAPER_CUSHION},
)
SUITE = WORKLOADS + PAPER_WORKLOADS
SCORED_IDS = tuple(spec["id"] for spec in WORKLOADS)
PAPER_IDS = tuple(spec["id"] for spec in PAPER_WORKLOADS)
# Independent generator streams: the paper stratum, and the spectra of the six
# original members (their orthonormal factors keep the main stream).
PAPER_STREAM = 20261007
ORIGINAL_SPECTRUM_STREAM = 20261031
# Figure 10 ignores singular values below gamma * sigma_max.
TOP_SUBSPACE_GAMMA = 1e-3
BEAT_METRICS = (
    "worst_case_scalar_error",
    "relative_polar_factor_error",
    "spectral_polar_error",
    "polar_cosine_similarity",
    "top_subspace_polar_error",
)

_REFERENCE_CACHE: dict[tuple[int, bool], dict[str, list[dict[str, Any]]]] = {}

FORBIDDEN_DECOMPOSITIONS = {
    "svd", "svd_lowrank", "pca_lowrank", "eig", "eigh", "eigvals",
    "eigvalsh", "pinv", "polar",
}

ALLOWED_IMPORT_MODULES = {
    "itertools", "math", "numpy", "numpy.linalg", "numpy.polynomial",
    "numpy.polynomial.polynomial", "torch",
}
FORBIDDEN_DYNAMIC_NAMES = {
    "__builtins__", "__loader__", "__spec__",
    "__import__", "breakpoint", "compile", "eval", "exec", "getattr", "globals",
    "help", "locals", "setattr", "vars",
}
# The upstream torch.compile decorator is valid; bare compile stays forbidden.
FORBIDDEN_ATTRIBUTES = FORBIDDEN_DECOMPOSITIONS | (FORBIDDEN_DYNAMIC_NAMES - {"compile"}) | {
    "__array__", "__array_interface__", "__array_struct__", "__builtins__",
    "__globals__", "__subclasses__", "_getframe", "ctypes", "item", "numpy",
    "cholesky", "dot", "einsum", "from_dlpack", "inner", "inv", "lstsq",
    "lu", "matrix_power", "multi_dot", "outer", "qr", "storage",
    "tensordot", "tolist", "untyped_storage", "vdot",
    "inspect", "currentframe", "getattr_static", "f_back", "f_globals",
    "f_locals", "f_builtins", "gi_frame", "gi_code", "cr_frame", "cr_code",
    "ag_frame", "ag_code",
    "sys", "os", "builtins", "importlib", "subprocess", "pathlib", "attrgetter", "methodcaller",
}

MATMUL_GUARD_MODULE = "_frontier_matmul_guard"
MATMUL_GUARD_NAME = "_frontier_checked_matmul"
MATMUL_GUARD_SOURCE = '''import os
import sys
import torch
from types import BuiltinMethodType

_torch = torch
_tensor = torch.Tensor
_matmul = torch.matmul
_tensor_matmul = torch.Tensor.matmul
_getattr = getattr
_root = os.path.realpath(os.path.dirname(__file__))

def _protected(path):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    resolved = os.path.realpath(os.fsdecode(path))
    return resolved == _root or resolved.startswith(_root + os.sep)

def _audit(event, args):
    if event == "open":
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+"))
        writing = writing or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing and _protected(path):
            raise PermissionError("candidate execution source is read-only")
    mutations = {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.truncate", "os.utime"}
    if event in mutations and args and _protected(args[0]):
        raise PermissionError("candidate execution source is read-only")
    if event in {"os.rename", "os.link", "os.symlink"} and any(_protected(p) for p in args[:2]):
        raise PermissionError("candidate execution source is read-only")
    if event in {"subprocess.Popen", "os.system", "os.fork", "os.forkpty", "os.posix_spawn", "os.exec", "os.spawn"}:
        raise PermissionError("candidate child processes are outside the numerical operation model")

# Protect the instrumented code, not arbitrary host files or a second OS boundary.
sys.addaudithook(_audit)

def checked_matmul(receiver):
    if receiver is _torch:
        return _matmul
    if receiver is _tensor:
        return _tensor_matmul
    if isinstance(receiver, _tensor):
        return checked_import(_getattr(receiver, "matmul"))
    # Helper modules may re-export the same genuine callable.
    return checked_import(_getattr(receiver, "matmul"))

def checked_import(value):
    if value is _matmul or value is _tensor_matmul:
        return value
    if (type(value) is BuiltinMethodType and isinstance(value.__self__, _tensor)
            and value == _tensor_matmul.__get__(value.__self__, _tensor)):
        return value
    raise ValueError("matmul import requires a genuine PyTorch callable")
'''


def _allowed_import(module: str) -> bool:
    return (module in ALLOWED_IMPORT_MODULES or module == "frontier_solution"
            or module.startswith("frontier_solution."))


def _policy_check(repo: Path) -> None:
    """Check the declared numerical API restrictions before candidate execution."""
    paths = [repo / "polar_express.py"]
    extension = repo / "frontier_solution"
    if extension.exists():
        paths.extend(extension.rglob("*.py"))
    for path in paths:
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"Python source is not parseable: {path.name}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(not _allowed_import(alias.name) for alias in node.names):
                    raise ValueError("Import is outside the numerical execution capability set")
            elif isinstance(node, ast.ImportFrom):
                if node.level or not _allowed_import(node.module or ""):
                    raise ValueError("Import is outside the numerical execution capability set")
                if any(alias.name in FORBIDDEN_ATTRIBUTES or alias.name.startswith("_")
                       or alias.name == "*" for alias in node.names):
                    raise ValueError("Private, reflection, decomposition or wildcard imports are outside the operation model")
                if (any(alias.name == "matmul" for alias in node.names)
                        and node.module != "torch"
                        and not (node.module or "").startswith("frontier_solution")):
                    raise ValueError("NumPy matmul imports are outside the operation model")
            elif isinstance(node, ast.Name) and node.id in FORBIDDEN_DYNAMIC_NAMES:
                raise ValueError(f"Dynamic lookup or execution is outside the operation model: {node.id}")
            elif isinstance(node, ast.MatchClass):
                if node.patterns or any(name in FORBIDDEN_ATTRIBUTES or name == "matmul"
                                        or name.startswith("_") for name in node.kwd_attrs):
                    raise ValueError("Class-pattern attribute lookup is outside the operation model")
            elif isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_ATTRIBUTES or node.attr.startswith("_"):
                    raise ValueError(f"Reflection, tensor export, or decomposition is outside the operation model: {node.attr}")
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    raise ValueError("Import-time monkeypatching is outside the operation model")
            # Reserve the injected runtime binding in every Python binding form.
            for _, value in ast.iter_fields(node):
                names = [value] if isinstance(value, str) else value if isinstance(value, list) else []
                if any(isinstance(name, str) and name.startswith("_frontier_") for name in names):
                    raise ValueError("Verifier runtime names are reserved")


def _prepare_runtime_tree(repo: Path) -> None:
    """Guard attribute lookup by actual receiver identity, including aliases."""
    class MatmulLookup(ast.NodeTransformer):
        def visit_Attribute(self, node):
            node = self.generic_visit(node)
            if node.attr == "matmul" and isinstance(node.ctx, ast.Load):
                return ast.copy_location(ast.Call(
                    func=ast.Name(id=MATMUL_GUARD_NAME, ctx=ast.Load()),
                    args=[node.value], keywords=[]), node)
            return node

        def visit_ImportFrom(self, node):
            checks = []
            for alias in node.names:
                if alias.name == "matmul":
                    name = alias.asname or alias.name
                    checks.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                        value=ast.Call(func=ast.Name(id="_frontier_checked_import", ctx=ast.Load()),
                                       args=[ast.Name(id=name, ctx=ast.Load())], keywords=[])))
            return [node, *checks]

    paths = [repo / "polar_express.py"]
    extension = repo / "frontier_solution"
    if extension.exists():
        paths.extend(extension.rglob("*.py"))
    for path in paths:
        tree = MatmulLookup().visit(ast.parse(path.read_text(), filename=str(path)))
        start = 0
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            start = 1
        while (start < len(tree.body) and isinstance(tree.body[start], ast.ImportFrom)
               and tree.body[start].module == "__future__"):
            start += 1
        tree.body.insert(start, ast.ImportFrom(module=MATMUL_GUARD_MODULE,
            names=[ast.alias(name="checked_matmul", asname=MATMUL_GUARD_NAME),
                   ast.alias(name="checked_import", asname="_frontier_checked_import")], level=0))
        path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    (repo / (MATMUL_GUARD_MODULE + ".py")).write_text(MATMUL_GUARD_SOURCE)

RUNNER = r'''
import base64, json, math, os, sys
from pathlib import Path

payload = json.loads(sys.stdin.readline())
safe_dumps = json.dumps
repo = Path(payload["repo"])
sys.path.insert(0, str(repo))
os.chdir(repo)

# Invoke the redacted authentic entrypoint. Test mode still executes its import
# and first real composition/plot cell while keeping evaluator unit tests quick.
import nbformat
from nbclient import NotebookClient
nb = nbformat.read(repo / "plotsforpolar.ipynb", as_version=4)
if payload.get("fast"):
    nb.cells = nb.cells[:5]
NotebookClient(nb, timeout=60, kernel_name="python3", allow_errors=False).execute(cwd=str(repo))

import torch
from torch.utils._python_dispatch import TorchDispatchMode

torch.set_num_threads(1)
specs = payload["workloads"]
records = []
matrices = []

for encoded, shape in zip(payload["matrices"], (s["shape"] for s in specs)):
    raw = base64.b64decode(encoded, validate=True)
    matrices.append(torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(*shape).clone())

# Candidate code never needs to export a scored Tensor to another backend.
# Install these denials before importing it, so saving an import-time alias does
# not evade the boundary.  Keep only a private serializer for the runner itself.
safe_tolist = torch.Tensor.tolist
def deny_tensor_export(self, *args, **kwargs):
    raise ValueError("exporting scored tensors is outside the operation model")
for export_name in ("numpy", "tolist", "item", "__array__", "__dlpack__"):
    if hasattr(torch.Tensor, export_name):
        setattr(torch.Tensor, export_name, deny_tensor_export)

import polar_express as pe

def odd_eval(coeffs, x):
    # Scalar accuracy is computed here without importing a second matrix backend.
    y = [0.0] * len(x)
    for i, c in enumerate(coeffs):
        power = 2 * i + 1
        y = [old + float(c) * value ** power for old, value in zip(y, x)]
    return y

class MatrixOpGuard(TorchDispatchMode):
    """Count dense matrix products and reject decomposition dispatches."""
    def __init__(self):
        self.products = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        forbidden = (
            "linalg_svd", "linalg_eig", "linalg_qr", "linalg_pinv",
            "linalg_cholesky", "linalg_lu", "_local_scalar_dense",
        )
        if any(token in name for token in forbidden):
            raise ValueError("direct matrix decompositions are not permitted")
        products = ("aten.bmm.", "aten.mm.", "aten.matmul.", "aten.addmm.")
        allowed = products + (
            "aten._to_copy.", "aten.expand.", "aten.view.", "aten._unsafe_view.",
            "aten.transpose.", "aten.permute.", "aten.t.",
            "aten.linalg_vector_norm.", "aten.add.", "aten.sub.", "aten.rsub.", "aten.mul.", "aten.div.",
        )
        if not any(name.startswith(prefix) for prefix in allowed):
            raise ValueError("PyTorch operation is outside the three-product polynomial model: " + name)
        if any(name.startswith(prefix) for prefix in products):
            self.products += 1
        return func(*args, **(kwargs or {}))

for spec in specs:
    l = float(spec["l"])
    q = pe.optimal_quintic(l, 1.0)
    if len(q) != 3 or not all(math.isfinite(float(v)) for v in q):
        raise ValueError("optimal_quintic must return three finite coefficients")
    seq = pe.optimal_composition(
        l, int(spec["steps"]), degree=5,
        safety_factor_eps=float(spec["safety"]), cushion=float(spec["cushion"]),
    )
    if len(seq) != int(spec["steps"]):
        raise ValueError("optimal_composition returned the wrong number of steps")
    x = [l + (1.0 - l) * i / 999 for i in range(1000)]
    for coeffs in seq:
        a = list(coeffs)
        if len(a) != 3 or not all(math.isfinite(float(v)) for v in a):
            raise ValueError("each quintic step must contain three finite coefficients")
        x = odd_eval(a, x)
        if not all(math.isfinite(v) for v in x):
            raise ValueError("polynomial composition produced non-finite values")
    scalar_error = max(abs(v - 1.0) for v in x)
    records.append({"id": spec["id"], "coefficients": [[float(v) for v in step] for step in seq]})

# The two members of each shape group are executed as one batch; each output
# element remains a separate beat-spec workload.
for start in range(0, len(specs), 2):
    batch = torch.stack(matrices[start:start + 2])
    before = batch.clone()
    steps = int(specs[start]["steps"])
    guard = MatrixOpGuard()
    with torch.no_grad(), guard:
        out = pe.PolarExpress(batch, steps=steps)
    if guard.products > 3 * steps:
        raise ValueError("PolarExpress exceeded three matrix products per requested step")
    if not isinstance(out, torch.Tensor) or tuple(out.shape) != tuple(batch.shape):
        raise ValueError("PolarExpress returned the wrong type or shape")
    if not torch.equal(batch, before):
        raise ValueError("PolarExpress mutated its input")
    if not bool(torch.isfinite(out).all()):
        raise ValueError("PolarExpress returned non-finite values")
    for j in range(2):
        packed = out[j].detach().cpu().to(torch.float32).contiguous().view(torch.uint8).flatten()
        raw = safe_tolist(packed)
        records[start + j]["output"] = base64.b64encode(bytes(raw)).decode("ascii")

print("FCS_RESULT=" + safe_dumps(records, separators=(",", ":")))
'''


def _invalid(reason: str) -> tuple[float, float, str, dict]:
    return 0.0, 0.0, reason[:240], {}


def _paper_spectrum(rng: np.random.Generator, k: int) -> np.ndarray:
    """One hidden singular spectrum for a paper-regime member.

    Families follow the paper's test matrices: log-spaced synthetic spectra
    (Figure 3, here sigma_min/sigma_max between 10^-4.5 and 10^-2.5), power-law
    decay, and low effective rank with a small tail (the gradient-matrix regime
    discussed in Section 3.3). An arbitrary global scale is applied because the
    method must normalise its input.
    """
    family = int(rng.integers(3))
    if family == 0:
        singular = np.geomspace(10.0 ** -float(rng.uniform(2.5, 4.5)), 1.0, k)
    elif family == 1:
        singular = np.arange(1, k + 1, dtype=np.float64) ** -float(rng.uniform(0.6, 1.6))
    else:
        rank = max(1, k // 8)
        tail = float(rng.uniform(2.0, 3.0))
        singular = np.concatenate([
            np.geomspace(0.1, 1.0, rank),
            np.geomspace(10.0 ** -(tail + 2.0), 10.0 ** -tail, k - rank),
        ])
    return singular * 10.0 ** float(rng.uniform(-2.0, 2.0))


def _original_spectrum(rng: np.random.Generator, l: float, k: int) -> np.ndarray:
    """Singular values of one original member, drawn per seed.

    Log-spaced as in the artifact's experiment, with sigma_min/sigma_max within a
    factor of two of the record's scalar l and an arbitrary global scale, so the
    shape, step count and norm do not identify the spectrum.
    """
    ratio = l * 2.0 ** float(rng.uniform(-1.0, 1.0))
    return np.geomspace(ratio, 1.0, k) * 10.0 ** float(rng.uniform(-2.0, 2.0))


def _oracle(matrix: np.ndarray) -> dict[str, np.ndarray]:
    u, singular, vh = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
    keep = singular >= TOP_SUBSPACE_GAMMA * singular[0]
    return {"polar": u @ vh, "u1": u[:, keep], "v1": vh[keep].T}


def _suite(seed: int) -> tuple[list[str], list[dict[str, np.ndarray]]]:
    """Create candidate inputs and private float64 oracles in the judge process."""
    rng = np.random.default_rng(seed)
    spectrum_rng = np.random.default_rng([seed, ORIGINAL_SPECTRUM_STREAM])
    encoded: list[str] = []
    oracles: list[dict[str, np.ndarray]] = []
    for spec in WORKLOADS:
        m, n = map(int, spec["shape"])
        k = min(m, n)
        left, _ = np.linalg.qr(rng.standard_normal((m, k)))
        right, _ = np.linalg.qr(rng.standard_normal((n, k)))
        singular = _original_spectrum(spectrum_rng, float(spec["l"]), k)
        matrix = ((left * singular) @ right.T).astype(np.float32)
        encoded.append(base64.b64encode(matrix.tobytes()).decode("ascii"))
        oracles.append(_oracle(matrix))
    paper_rng = np.random.default_rng([seed, PAPER_STREAM])
    for spec in PAPER_WORKLOADS:
        m, n = map(int, spec["shape"])
        k = min(m, n)
        left, _ = np.linalg.qr(paper_rng.standard_normal((m, k)))
        right, _ = np.linalg.qr(paper_rng.standard_normal((n, k)))
        singular = _paper_spectrum(paper_rng, k)
        matrix = ((left * singular) @ right.T).astype(np.float32)
        encoded.append(base64.b64encode(matrix.tobytes()).decode("ascii"))
        oracles.append(_oracle(matrix))
    return encoded, oracles


def _matrix_errors(output: np.ndarray, oracle: dict[str, np.ndarray]) -> dict[str, float]:
    """Paper matrix metrics for one returned member (float64, judge side)."""
    polar = oracle["polar"]
    difference = output - polar
    output_norm = float(np.linalg.norm(output, ord="fro"))
    polar_norm = float(np.linalg.norm(polar, ord="fro"))
    top_left, top_right = oracle["u1"], oracle["v1"]
    truncated = top_left @ top_right.T
    projected = top_left @ (top_left.T @ output @ top_right) @ top_right.T
    return {
        # Appendix H.1, Figure 8: relative Frobenius error.
        "polar": float(np.linalg.norm(difference, ord="fro") / polar_norm),
        # Section 4.1, Figure 3: spectral-norm error.
        "spectral": float(np.linalg.norm(difference, ord=2)),
        # Appendix H.1, Figure 8: Frobenius cosine similarity.
        "cosine": float(np.sum(output * polar) / (output_norm * polar_norm)) if output_norm > 0.0 else 0.0,
        # Appendix H.1, Figure 10: error on the top singular subspace.
        "top": float(np.linalg.norm(projected - truncated, ord="fro") / np.linalg.norm(truncated, ord="fro")),
    }


def _run_tree(repo: Path, seed: int, *, authentic: bool = True) -> list[dict[str, Any]]:
    _policy_check(repo)
    # Only the disposable execution copy is instrumented; submission bytes stay intact.
    with tempfile.TemporaryDirectory(prefix="polar-runtime-") as runtime:
        prepared = Path(runtime) / "repo"
        shutil.copytree(repo, prepared, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        _prepare_runtime_tree(prepared)
        return _run_prepared_tree(prepared, seed, authentic=authentic)


def _run_prepared_tree(repo: Path, seed: int, *, authentic: bool = True) -> list[dict[str, Any]]:
    matrices, oracles = _suite(seed)
    payload = {
        "repo": str(repo),
        "workloads": SUITE,
        "matrices": matrices,
        "fast": bool(os.environ.get("FCS_EVALUATOR_TEST_FAST")) or not authentic,
    }
    env = os.environ.copy()
    env.update({
        "MPLBACKEND": "Agg",
        "TORCH_COMPILE_DISABLE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        # Keep kernelspec discovery independent of a writable user home and
        # make the kernelspec's `python` resolve to this evaluator interpreter.
        "JUPYTER_PATH": str(Path(sys.prefix) / "share" / "jupyter"),
        "JUPYTER_DATA_DIR": str(repo / ".frontier_jupyter"),
        "PATH": str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
    })
    proc = subprocess.run(
        [sys.executable, "-c", RUNNER],
        input=json.dumps(payload) + "\n",
        text=True,
        capture_output=True,
        env=env,
        timeout=100,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise ValueError(detail[-1] if detail else "numerical runner failed")
    lines = [line for line in proc.stdout.splitlines() if line.startswith("FCS_RESULT=")]
    if len(lines) != 1:
        raise ValueError("numerical runner returned no result")
    rows = json.loads(lines[0].split("=", 1)[1])
    if not isinstance(rows, list) or len(rows) != len(SUITE):
        raise ValueError("numerical runner returned incomplete results")
    measured = []
    for spec, row, oracle in zip(SUITE, rows, oracles):
        if row.get("id") != spec["id"]:
            raise ValueError("numerical runner returned mismatched results")
        coefficients = row.get("coefficients")
        if not isinstance(coefficients, list) or len(coefficients) != int(spec["steps"]):
            raise ValueError("numerical runner returned invalid coefficients")
        x = np.linspace(float(spec["l"]), 1.0, 1000, dtype=np.float64)
        for coeffs in coefficients:
            a = np.asarray(coeffs, dtype=np.float64).reshape(-1)
            if a.size != 3 or not np.isfinite(a).all():
                raise ValueError("numerical runner returned invalid coefficients")
            x = sum(float(c) * x ** (2 * i + 1) for i, c in enumerate(a))
        scalar = float(np.max(np.abs(x - 1.0)))
        shape = tuple(map(int, spec["shape"]))
        raw = base64.b64decode(row.get("output", ""), validate=True)
        output = np.frombuffer(raw, dtype=np.float32)
        if output.size != math.prod(shape):
            raise ValueError("numerical runner returned an invalid tensor payload")
        output = output.reshape(shape).astype(np.float64)
        if not (math.isfinite(scalar) and np.isfinite(output).all()):
            raise ValueError("numerical runner returned non-finite results")
        measured.append({"id": spec["id"], "scalar": scalar, **_matrix_errors(output, oracle)})
    return measured


def _score(rows: list[dict[str, Any]]) -> float:
    """Original continuous score: the six original workloads only."""
    scalar_quality = []
    polar_quality = []
    rows = [row for row in rows if row["id"] in SCORED_IDS]
    for row in rows:
        scalar = max(float(row["scalar"]), 1e-15)
        polar = max(float(row["polar"]), 1e-15)
        scalar_quality.append(min(1.0, max(0.0, -math.log10(scalar) / 12.0)))
        polar_quality.append(min(1.0, max(0.0, -math.log10(polar) / 8.0)))
    return 100.0 * (sum(scalar_quality) / len(rows) + sum(polar_quality) / len(rows)) / 2.0


# Implementation 1 casts to bfloat16 "for speed" on a GPU and applies one fixed
# schedule (l = 1e-3, ten steps) to every matrix. The bar is the stricter, per
# workload and metric, of four valid runs of the same upstream source on the same
# inputs. Each variant edits only the disposable reference copy:
#   published     unchanged (bfloat16 online stage, fixed schedule);
#   matched       the speed cast replaced by the input's float32 precision;
#   per_workload  float32, and each member's matrix schedule is
#                 optimal_composition(l, steps, 5, safety, cushion) of its record;
#   unguarded     as per_workload, with the schedule built without the safety
#                 factor and cushion and the 1.01 normalisation margin removed.
# The scalar sequences returned to the runner are the record's own
# optimal_composition output in every variant.
PUBLISHED_CAST = "X = G.bfloat16()  # for speed"
MATCHED_CAST = "X = G.float()  # precision-matched float32 online stage"
FIXED_SCHEDULE = "coeffs_list = optimal_composition(l=1e-3, num_iters=10, degree=5, safety_factor_eps=1e-2, cushion=0.02)"
FIXED_STEPS = """    hs = coeffs_list[:steps] + list( 
        repeat(coeffs_list[-1], steps - len(coeffs_list)))"""
PUBLISHED_NORMALISATION = "X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)"
UNGUARDED_NORMALISATION = "X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)"
PER_WORKLOAD_SCHEDULE = """

# Judge-side reference variant: the runner calls optimal_composition once per
# record, in suite order, before the matrix calls; each pair of calls supplies
# the per-member schedules of the next PolarExpress batch.
workload_schedules = []
batch_schedules = []
batch_calls = [0]
upstream_composition = optimal_composition

def optimal_composition(l, num_iters, degree=5, safety_factor_eps=0, cushion=0):
    seq = upstream_composition(l, num_iters, degree, safety_factor_eps, cushion)
    matrix_seq = SCHEDULE_ARGUMENTS
    workload_schedules.append([[float(c) for c in step] for step in matrix_seq])
    if len(workload_schedules) % 2 == 0:
        first, second = workload_schedules[-2], workload_schedules[-1]
        batch_schedules.append([tuple(torch.tensor([first[t][i], second[t][i]], dtype=torch.float32).view(2, 1, 1)
                                      for i in range(3)) for t in range(min(len(first), len(second)))])
    return seq
"""
PER_WORKLOAD_STEPS = """    hs = batch_schedules[batch_calls[0]]
    batch_calls[0] = batch_calls[0] + 1"""
REFERENCE_VARIANTS = ("published", "matched", "per_workload", "unguarded")
LOWER_IS_BETTER = ("scalar", "polar", "spectral", "top")


class ReferenceFailure(RuntimeError):
    """The paired reference did not produce a valid measurement.

    This is an evaluator error, never a candidate result: a failed reference
    cannot set a bar and the candidate is not scored against it.
    """
HIGHER_IS_BETTER = ("cosine",)


def _variant_source(source: str, variant: str) -> str:
    edits = [(PUBLISHED_CAST, MATCHED_CAST)]
    if variant in ("per_workload", "unguarded"):
        arguments = ("seq" if variant == "per_workload"
                     else "upstream_composition(l, num_iters, degree, 0, 0)")
        edits += [(FIXED_SCHEDULE, FIXED_SCHEDULE + PER_WORKLOAD_SCHEDULE.replace("SCHEDULE_ARGUMENTS", arguments)),
                  (FIXED_STEPS, PER_WORKLOAD_STEPS)]
    if variant == "unguarded":
        edits.append((PUBLISHED_NORMALISATION, UNGUARDED_NORMALISATION))
    for old, new in edits:
        if source.count(old) != 1:
            raise RuntimeError("reference variant edit is not the reviewed upstream line")
        source = source.replace(old, new)
    return source


def _reference_tree(variant: str = "published") -> Path:
    # Reference replay needs the same writable disposable copy as a submission.
    if variant not in REFERENCE_VARIANTS:
        raise ValueError("unknown reference variant")
    applied = apply_submission_patch(REFERENCE_PATCH, TASK_DIR, CONFIG)
    if not applied.ok:
        raise RuntimeError("reference patch could not be applied: " + applied.message)
    repo = Path(applied.workdir)
    if variant != "published":
        module = repo / "polar_express.py"
        try:
            module.write_text(_variant_source(module.read_text(), variant))
        except Exception:
            shutil.rmtree(repo.parent, ignore_errors=True)
            raise
    return repo


def _measure_reference(seed: int, variant: str) -> list[dict[str, Any]]:
    repo = _reference_tree(variant)
    try:
        return _run_tree(repo, seed, authentic=False)
    finally:
        shutil.rmtree(repo.parent, ignore_errors=True)


def _stricter_bar(*runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per workload and metric, the best value over the valid reference runs."""
    bar = []
    for rows in zip(*runs, strict=True):
        if len({row["id"] for row in rows}) != 1:
            raise RuntimeError("reference runs disagree on the graded suite")
        row = {"id": rows[0]["id"]}
        for key in LOWER_IS_BETTER:
            row[key] = min(float(r[key]) for r in rows)
        for key in HIGHER_IS_BETTER:
            row[key] = max(float(r[key]) for r in rows)
        bar.append(row)
    return bar


def _reference_rows(seed: int) -> dict[str, list[dict[str, Any]]]:
    """Measure the reference variants once per deterministic suite.

    ``per_workload`` defines the reference continuous score; ``bar`` is the
    beat-spec comparator, the stricter of all four variants per workload and
    metric. Every run must complete; a failed reference is an evaluator error.
    """
    fast = bool(os.environ.get("FCS_EVALUATOR_TEST_FAST"))
    key = (seed, fast)
    if key not in _REFERENCE_CACHE:
        try:
            runs = {variant: _measure_reference(seed, variant) for variant in REFERENCE_VARIANTS}
            bar = _stricter_bar(*runs.values())
        except Exception as error:
            raise ReferenceFailure(f"reference measurement failed: {type(error).__name__}") from error
        _REFERENCE_CACHE[key] = {"runs": runs, "bar": bar}
    # JSON round-tripping gives callers an isolated copy of this tiny structure.
    return json.loads(json.dumps(_REFERENCE_CACHE[key]))


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    candidate_root: Path | None = None
    try:
        applied = apply_submission_patch(solution_path, TASK_DIR, CONFIG)
        if not applied.ok:
            return _invalid(applied.message)
        candidate_repo = Path(applied.workdir)
        candidate_root = candidate_repo.parent
        _policy_check(candidate_repo)
        final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
        seed = FINAL_SEED if final else DEV_SEED

        candidate = _run_tree(candidate_repo, seed, authentic=True)
        reference_runs = _reference_rows(seed)
        reference = reference_runs["bar"]

        score = _score(candidate)
        reference_score = _score(reference_runs["runs"]["per_workload"])
        rows = []
        for cand, ref in zip(candidate, reference):
            if cand["id"] != ref["id"]:
                raise RuntimeError("reference rows do not match the graded suite")
            rows.append({
                "workload": str(cand["id"]),
                "metrics": {
                    "worst_case_scalar_error": bool(float(cand["scalar"]) < float(ref["scalar"])),
                    "relative_polar_factor_error": bool(float(cand["polar"]) < float(ref["polar"])),
                    "spectral_polar_error": bool(float(cand["spectral"]) < float(ref["spectral"])),
                    "polar_cosine_similarity": bool(float(cand["cosine"]) > float(ref["cosine"])),
                    "top_subspace_polar_error": bool(float(cand["top"]) < float(ref["top"])),
                },
            })
        beats = int(all(v for row in rows for v in row["metrics"].values()))
        original = [r for r in candidate if r["id"] in SCORED_IDS]
        paper = [r for r in candidate if r["id"] in PAPER_IDS]
        metrics = {
            "beat_spec_workload_passes": rows,
            "beats_reference": beats,
            "reference_score": reference_score,
            "margin": score - reference_score,
            "worst_case_scalar_error": sum(float(r["scalar"]) for r in original) / len(original),
            "relative_polar_factor_error": sum(float(r["polar"]) for r in original) / len(original),
            "paper_regime_relative_polar_factor_error": sum(float(r["polar"]) for r in paper) / len(paper),
            "spectral_polar_error": sum(float(r["spectral"]) for r in candidate) / len(candidate),
            "polar_cosine_similarity": sum(float(r["cosine"]) for r in candidate) / len(candidate),
            "top_subspace_polar_error": sum(float(r["top"]) for r in candidate) / len(candidate),
        }
        return score, score, "Scored six deterministic interval-matrix workloads; graded twelve.", metrics
    except ReferenceFailure:
        raise
    except subprocess.TimeoutExpired:
        return _invalid("Submission timed out during notebook or numerical execution.")
    except Exception as exc:
        return _invalid(f"Invalid submission: {type(exc).__name__}: {exc}")
    finally:
        if candidate_root is not None:
            shutil.rmtree(candidate_root, ignore_errors=True)


def _main() -> int:
    if len(sys.argv) != 2:
        print("usage: evaluator.py SOLUTION_PATCH", file=sys.stderr)
        return 2
    result = evaluate(sys.argv[1])
    print(json.dumps({"score": result[0], "score_unbounded": result[1], "message": result[2], "metrics": result[3]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
