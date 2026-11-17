"""Judge of polygon (Polygon counterexample search) for the shared verifier: evaluate(tree).

Score over the 19 non-hard workloads W (16 pairs, 3 disambiguation records) of the suite: r_w = 1
if MariaDB validates the returned database, u_w = 2 min(b_w, max(0, t_w)) / b_w with b_w the
case budget (a failed case: r_w = 0, u_w = 2); T = 0.05 ceil(sum u_w / 0.05);
E = exp(-T / (|W| 0.25)); raw_score = score_unbounded = clip(100 (sum r_w + E) / (|W| + 1), 0, 100).
check_tree: screened added-line tokens and forbidden imports (content_policy), scaffold outside
the edit markers (out_of_scope); the exact reference is exempt. A build failure scores 0.

Where it runs (task.toml [test_sandboxes.search], a trusted CPU command sandbox): this judge runs
on the control image and runs no submitted code. One sandbox machine gets the judge's search
harness (search_harness.py, search_child.py, the judge's copy of the fixed adapter), both trees
and the measurement plan; it runs every side of every case, one at a time (reference first on
even cases, candidate first on odd ones), each search in a fresh shared isolate instance with
no network and no judge file (AF_UNIX only for the search's own IPC; no MariaDB or other service
runs on that machine), timed by the harness outside the instance
(the start reading taken just before the submitted modules are imported, to the result's
arrival). The counterexample comes back as data and is validated here, on a MariaDB started
fresh for it, with the pristine adapter's validation (MySQLTester). A reference that fails to
refute any workload raises (ReferenceFailure), as do a validator or a sandbox failure.

Output cap: the harness's records come back in one collect of at most SEARCH_OUTPUT_BYTES, so
every search's result line is capped at the fixed adapter's RESULT_MAX_BYTES (the statement
states it). A longer one is never truncated: that search is a failed case (r_w = 0, u_w = 2, the
rule for every failed case) and the summary names it; a reference over the cap raises.

Beat-spec metrics (beat_spec_workload_passes, all 21 workloads incl. the two hard pairs):
- refuted_pair: passes when r_cand >= r_ref.
- wall_time_seconds, lower is better: non-hard t_c <= t_r + max(0.03, 0.20 t_r); hard pairs
  t_c < 0.5 t_r.
beats_reference = 1 iff every workload passes both.

The shared verifier has already applied the patch (edit_scope, task.toml [submission]);
`evaluate(tree)` gets the pristine starter with only the allowed files overlaid.
"""

from __future__ import annotations

import ast
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
from fcs2_verifier_gpu import VerifierGpuError  # the shared command-sandbox adapter (control image)


AUTHENTIC_ENTRYPOINT = "evaluation/leetcode/online.py"
RUNNER = "frontier_runner.py"
REFERENCE_PATCH = TASK_DIR / "reference.patch"
CASE_TIMEOUT_SECONDS = 2
SUBPROCESS_TIMEOUT_SECONDS = 14
SUBPROCESS_MARGIN_SECONDS = 12
ORIGINAL_STRATUM = "original"
HARD_STRATUM = "paper_hard"
HARD_TIME_RATIO = 0.5
HARD_SUBPROCESS_MARGIN_SECONDS = 14
REFERENCE_BUDGET_EXTRA_SECONDS = 10
TIME_SCORE_QUANTUM_SECONDS = 0.05
TIME_SCORE_SCALE_SECONDS = 0.25
TIME_COMPARISON_ABS_TOLERANCE = 0.03
TIME_COMPARISON_REL_TOLERANCE = 0.20


def _pristine_root() -> Path:
    """The starter tree (repo/), copied to pristine/ by the Verifier image."""
    return TASK_DIR / "pristine"


def _reference_tree() -> Path:
    """A fresh copy of the pristine starter with reference.patch applied (trusted, judge-owned).
    The caller removes its parent."""
    scratch = Path(tempfile.mkdtemp(prefix="polygon-reference-"))
    tree = scratch / "tree"
    shutil.copytree(_pristine_root(), tree, symlinks=True)
    for path in (tree, *tree.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(scratch.resolve()))
    subprocess.run(["git", "apply", "--whitespace=nowarn", str(REFERENCE_PATCH)], cwd=tree,
                   check=True, capture_output=True, timeout=60, env=env)
    return tree

EDIT_REGIONS = {
    "sql_testing/environment.py": (
        (
            "# FRONTIER_EDIT_BEGIN: counterexample discovery",
            "# FRONTIER_EDIT_END: counterexample discovery",
        ),
        (
            "# FRONTIER_EDIT_BEGIN: query disambiguation",
            "# FRONTIER_EDIT_END: query disambiguation",
        ),
    ),
    "sql_testing/smt/formula.py": (
        (
            "# FRONTIER_EDIT_BEGIN: exhaustive baseline",
            "# FRONTIER_EDIT_END: exhaustive baseline",
        ),
        ("# FRONTIER_EDIT_BEGIN: search policy", "# FRONTIER_EDIT_END: search policy"),
        ("# FRONTIER_EDIT_BEGIN: recovery policy", "# FRONTIER_EDIT_END: recovery policy"),
        ("# FRONTIER_EDIT_BEGIN: search memory", "# FRONTIER_EDIT_END: search memory"),
        (
            "# FRONTIER_EDIT_BEGIN: precise fallback",
            "# FRONTIER_EDIT_END: precise fallback",
        ),
        (
            "# FRONTIER_EDIT_BEGIN: additional search constraints",
            "# FRONTIER_EDIT_END: additional search constraints",
        ),
    ),
    "sql_testing/smt/knowledgebase.py": (
        ("# FRONTIER_EDIT_BEGIN: search state", "# FRONTIER_EDIT_END: search state"),
        (
            "# FRONTIER_EDIT_BEGIN: search-state update",
            "# FRONTIER_EDIT_END: search-state update",
        ),
    ),
}

FORBIDDEN_ADDITIONS = (
    "__import__",
    "__main__",
    "atexit",
    "builtins",
    "compile(",
    "eval(",
    "exec(",
    "inspect",
    "open(",
    "os.",
    "pathlib",
    "pickle",
    "resource",
    "socket",
    "subprocess",
    "sys.",
    "/proc",
    "/sys",
)
# Modules no submitted file may import, in any import form or alias.
FORBIDDEN_IMPORTS = frozenset({
    "__main__", "_io", "atexit", "builtins", "ctypes", "gc", "importlib",
    "inspect", "io", "marshal", "mmap", "os", "pickle", "posix", "resource",
    "socket", "subprocess", "sys",
})


@dataclass(frozen=True)
class Workload:
    identifier: str
    kind: str
    timeout_seconds: int
    payload: dict
    stratum: str = ORIGINAL_STRATUM


@dataclass(frozen=True)
class Observation:
    refuted: bool
    time_seconds: float
    iterations: int
    backtracks: int
    timeout_seconds: int = CASE_TIMEOUT_SECONDS
    valid: bool = True


def _load_suite() -> tuple[Workload, ...]:
    final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
    name = "final.json" if final else "dev.json"
    rows = json.loads((TASK_DIR / "workloads" / name).read_text(encoding="utf-8"))
    workloads = []
    for row in rows:
        kind = str(row.get("kind") or "pair")
        timeout = int(row.get("timeout_seconds") or CASE_TIMEOUT_SECONDS)
        if timeout < 1:
            raise ValueError("workload timeout_seconds must be positive")
        if kind == "pair":
            payload = {
                "kind": kind,
                "problem_id": int(row["problem_id"]),
                "qid": int(row["qid"]),
                "bound": int(row["bound"]),
                "query": str(row["query"]),
                "timeout_seconds": timeout,
            }
        elif kind == "disambiguation":
            payload = {
                "kind": kind,
                "schema": list(row["schema"]),
                "queries": [str(query) for query in row["queries"]],
                "bound": int(row["bound"]),
                "timeout_seconds": timeout,
            }
        else:
            raise ValueError(f"unsupported workload kind: {kind}")
        workloads.append(Workload(str(row["id"]), kind, timeout, payload, str(row.get("stratum", ORIGINAL_STRATUM))))
    workloads = tuple(workloads)
    if os.environ.get("FRONTIER_FAST_TEST") == "1":
        return workloads[:1]
    return workloads


def _audit_rows(workloads: tuple[Workload, ...], decisions=None) -> list[dict]:
    decisions = decisions or {}
    rows = []
    for index, workload in enumerate(workloads):
        values = decisions.get(workload.identifier, {})
        rows.append(
            {
                "workload": f"workload-{index + 1}",
                "metrics": {
                    "refuted_pair": bool(values.get("refuted_pair", False)),
                    "wall_time_seconds": bool(values.get("wall_time_seconds", False)),
                },
            }
        )
    return rows


def score_observations(observations: list[Observation]) -> float:
    if not observations:
        return 0.0
    solved = sum(int(row.refuted) for row in observations)
    total_time = sum(
        CASE_TIMEOUT_SECONDS
        * min(1.0, max(0.0, float(row.time_seconds)) / row.timeout_seconds)
        for row in observations
    )
    quantized = math.ceil(total_time / TIME_SCORE_QUANTUM_SECONDS) * TIME_SCORE_QUANTUM_SECONDS
    efficiency = math.exp(
        -quantized / (len(observations) * TIME_SCORE_SCALE_SECONDS)
    )
    return max(0.0, min(100.0, 100.0 * (solved + efficiency) / (len(observations) + 1)))


def _metrics(
    workloads: tuple[Workload, ...],
    observations: list[Observation] | None = None,
    reference_score: float = 0.0,
    candidate_score: float = 0.0,
    decisions=None,
) -> dict:
    rows = _audit_rows(workloads, decisions)
    observations = observations or []
    return {
        "beat_spec_workload_passes": rows,
        "beats_reference": int(
            bool(rows)
            and all(value for row in rows for value in row["metrics"].values())
        ),
        "reference_score": float(reference_score),
        "margin": float(candidate_score - reference_score),
        "refuted_pairs": [int(row.refuted) for row in observations],
        "wall_time_seconds": [round(row.time_seconds, 6) for row in observations],
        "search_iterations": [int(row.iterations) for row in observations],
        "backtracks": [int(row.backtracks) for row in observations],
    }


def _invalid(reason: str, workloads: tuple[Workload, ...]):
    return 0.0, 0.0, reason[:140], _metrics(workloads)


def _added_lines_ok(changes) -> list[str]:
    """The changed paths whose added lines contain a screened token (case-insensitive)."""
    return [change.path for change in changes
            if any(token in line.lower() for line in change.added for token in FORBIDDEN_ADDITIONS)]


def _is_reference(tree: Path, changes) -> bool:
    """Whether the judged tree is byte-identical to the judge's reference on every change."""
    reference = _reference_tree()
    try:
        return bool(changes) and all(
            (reference / c.path).is_file()
            and (tree / c.path).read_bytes() == (reference / c.path).read_bytes()
            if c.status != "D" else not (reference / c.path).exists()
            for c in changes)
    finally:
        shutil.rmtree(reference.parent, ignore_errors=True)


def check_tree(tree: str, pristine: str, changes: list) -> tuple[str, list[str]] | None:
    """Shared-verifier hook: screened tokens in added lines and forbidden imports
    (content_policy), and the immutable scaffold outside the edit markers (out_of_scope)."""
    tree = Path(tree)
    if _is_reference(tree, changes):
        return None
    hits = _added_lines_ok(changes)
    if hits:
        return "content_policy", hits
    ok, _ = _scaffold_ok(tree)
    if not ok:
        return "out_of_scope", [c.path for c in changes if c.path in EDIT_REGIONS]
    ok, _ = _imports_ok(tree)
    if not ok:
        return "content_policy", [c.path for c in changes if c.status != "D"]
    return None


def _imports_ok(root: Path) -> tuple[bool, str]:
    """Reject submitted Python files that import a forbidden module.

    The substring policy only sees added lines; this check parses every
    editable file as applied, so aliases and multi-line imports are covered.
    """
    paths = [root / relative for relative in EDIT_REGIONS]
    extensions = root / "sql_testing/smt/search_extensions"
    if extensions.is_dir():
        paths.extend(sorted(extensions.rglob("*.py")))
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, SyntaxError, ValueError):
            continue  # the build check rejects unreadable or invalid source
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] in FORBIDDEN_IMPORTS for name in names):
                return False, "environment, file, process, or harness introspection is not allowed"
    return True, ""


def _mask_edit_regions(text: str, regions) -> str:
    output = []
    cursor = 0
    for begin, end in regions:
        if text.count(begin) != 1 or text.count(end) != 1:
            raise ValueError("editable-region markers must be retained")
        begin_at = text.index(begin, cursor)
        line_end = text.find("\n", begin_at)
        end_at = text.index(end, line_end)
        end_line_start = text.rfind("\n", 0, end_at) + 1
        output.append(text[cursor : line_end + 1])
        output.append("<FRONTIER_EDIT_REGION>\n")
        cursor = end_line_start
    output.append(text[cursor:])
    return "".join(output)


def _scaffold_ok(root: Path) -> tuple[bool, str]:
    pristine = _pristine_root()
    try:
        for relative, regions in EDIT_REGIONS.items():
            candidate = (root / relative).read_text(encoding="utf-8")
            original = (pristine / relative).read_text(encoding="utf-8")
            if _mask_edit_regions(candidate, regions) != _mask_edit_regions(original, regions):
                return False, f"{relative} changes immutable scaffolding"
    except (OSError, UnicodeError, ValueError):
        return False, "editable-region markers or source files are invalid"
    return True, ""


def _build(root: Path) -> None:
    if not (root / AUTHENTIC_ENTRYPOINT).is_file():
        raise ValueError("authentic benchmark entrypoint is missing")
    if not (root / RUNNER).is_file():
        raise ValueError("benchmark adapter is missing")
    proc = subprocess.run(
        ["bash", str(root / "frontier-build.sh")],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    if proc.returncode:
        raise ValueError("submitted source did not build")


CANDIDATE_SIDE = "candidate"
REFERENCE_SIDE = "reference"
SANDBOX = "search"   # task.toml [test_sandboxes.search]
PYTHON = "/opt/frontier/runtime-python"   # the sandbox image's interpreter
# Untimed preparation of a search process (third-party imports and the SQL grammar), as in the
# fixed adapter (frontier_runner.SETUP_SECONDS); the instance limit below bounds it further.
SEARCH_SETUP_SECONDS = 30.0
HARNESS_FILES = ("search_harness.py", "search_child.py")
# The one collect of the harness's out/: run_sandboxes() allows at most one SSH chunk, 7/8 of the
# Verifier's channel size (FRONTIER_MAX_SSH_BYTES_PER_DIRECTION, the shared Verifier base's
# 8 MiB), and refuses a larger max_output_bytes. 7 MiB fits every channel of 8 MiB or more; 42 entries x (RESULT_MAX_BYTES + a record of at most ~10 KiB) stays below it
# (tests/test_polygon_search_sandbox.py).
SEARCH_OUTPUT_BYTES = 7 * 1024 * 1024
VALIDATOR_START_SECONDS = 20.0
VALIDATOR_START_ATTEMPTS = 2
VALIDATOR_MAX_CONNECTIONS = 32
DIAGNOSTIC_MAX_BYTES = 120 * 1024
# A small InnoDB footprint keeps template copies and restarts cheap. The validator only
# executes the two queries on the returned counterexample, outside every timed region, so its
# storage sizing never enters a metric.
VALIDATOR_STORAGE_ARGS = (
    "--innodb-log-file-size=4M",
    "--innodb-undo-tablespaces=0",
    "--innodb-data-file-path=ibdata1:4M:autoextend",
)
VALIDATOR_SYSTEM_SCHEMAS = frozenset({"mysql", "information_schema", "performance_schema", "sys"})


class ReferenceFailure(RuntimeError):
    """The paired reference did not refute a graded pair within its budget.

    An evaluator error, never a candidate result: a failed reference cannot set
    a bar and the candidate is not scored against it. ``private_execution_evidence``
    carries the bounded per-case records to the operator-only verifier evidence.
    """

    def __init__(self, message: str, evidence: dict | None = None):
        super().__init__(message)
        if evidence is not None:
            self.private_execution_evidence = evidence


class ValidatorFailure(ReferenceFailure):
    """The judge's SQL validator could not be provisioned for a case.

    Infrastructure, not the candidate: never scored as a candidate zero.
    """


class SearchSandboxFailure(ReferenceFailure):
    """The trusted search harness on the sandbox failed, or returned no records.

    Infrastructure, not the candidate: the harness runs only judge code outside its isolate
    instances, and the lease covers every instance's limit.
    """


_ADAPTER = None


def _adapter():
    """The judge's own copy of the fixed adapter (pristine frontier_runner.py), with the
    pristine starter on sys.path for its validation inputs and MySQLTester."""
    global _ADAPTER
    if _ADAPTER is None:
        import importlib.util
        import logging

        root = str(_pristine_root())
        if root not in sys.path:
            sys.path.append(root)
        spec = importlib.util.spec_from_file_location(
            "polygon_judge_adapter", _pristine_root() / RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from evaluation.calcite.tester import MySQLTester

        # The upstream tester logs every compared result to stdout; the judge's stdout is not
        # a log. Logging never changes a validation.
        logging.getLogger("logger").disabled = True
        module.MySQLTester = MySQLTester
        _ADAPTER = module
    return _ADAPTER


def _validator_template(directory: Path) -> Path:
    """Initialize one pristine datadir; every server start uses a copy."""
    data = directory / "template"
    try:
        init = subprocess.run(
            [
                "mariadb-install-db",
                "--no-defaults",
                f"--datadir={data}",
                "--auth-root-authentication-method=normal",
                "--lower-case-table-names=1",
                *VALIDATOR_STORAGE_ARGS,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValidatorFailure("could not initialize SQL validator") from error
    if init.returncode:
        raise ValidatorFailure("could not initialize SQL validator")
    return data


class _MariaDB:
    """The judge's SQL validator, on the control image.

    No submitted code runs on this machine and nothing but the judge ever connects: the
    search ran on the sandbox, and only its returned counterexample (data) reaches here.
    ``fresh()`` starts a server on a new copy of the pristine template for each validation and
    ``stop()`` kills it after, so no counterexample's database outlives its own case.
    """

    def __init__(self, template: Path):
        self.template = template
        self.root = Path(tempfile.mkdtemp(prefix="polygon-sql-"))
        self.process: subprocess.Popen | None = None
        self.starts = 0

    @property
    def socket(self) -> str:
        return str(self.root / "mysql.sock")

    def _server_args(self, data: Path) -> list[str]:
        server = shutil.which("mariadbd") or "/usr/sbin/mariadbd"
        args = [server, "--no-defaults"]
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            # MariaDB refuses to launch as root unless this is explicit; retain the normal
            # unprivileged behavior for runs under another UID.
            args.append("--user=root")
        args.extend(
            [
                f"--datadir={data}",
                f"--socket={self.socket}",
                f"--pid-file={self.root / 'mysql.pid'}",
                "--skip-grant-tables",
                "--skip-networking",
                "--lower-case-table-names=1",
                "--innodb-buffer-pool-size=64M",
                f"--max-connections={VALIDATOR_MAX_CONNECTIONS}",
                *VALIDATOR_STORAGE_ARGS,
            ]
        )
        return args

    def _start_once(self) -> bool:
        data = self.root / "data"
        shutil.rmtree(data, ignore_errors=True)
        for leftover in ("mysql.sock", "mysql.sock.lock", "mysql.pid"):
            try:
                (self.root / leftover).unlink()
            except FileNotFoundError:
                pass
        shutil.copytree(self.template, data)
        self.starts += 1
        self.process = subprocess.Popen(
            self._server_args(data),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + VALIDATOR_START_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                return False
            ping = subprocess.run(
                ["mysqladmin", f"--socket={self.socket}", "ping", "--silent"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if ping.returncode == 0:
                return True
            time.sleep(0.02)
        return False

    def fresh(self) -> str:
        """Start a server on a new copy of the pristine template."""
        self.stop()
        for _attempt in range(VALIDATOR_START_ATTEMPTS):
            try:
                if self._start_once():
                    # The pristine state has no user database at all (the installer's empty
                    # `test` schema included).
                    self._drop_user_databases()
                    return self.socket
            except Exception:  # noqa: BLE001 - any provisioning error retries once
                pass
            self.stop()
        raise ValidatorFailure("SQL validator did not start")

    def _drop_user_databases(self) -> None:
        import mysql.connector  # the declared judge runtime

        connection = mysql.connector.connect(
            unix_socket=self.socket, user="root", connection_timeout=10
        )
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT SCHEMA_NAME FROM information_schema.SCHEMATA")
            for (name,) in cursor.fetchall():
                if str(name).lower() not in VALIDATOR_SYSTEM_SCHEMAS:
                    cursor.execute("DROP DATABASE `" + str(name).replace("`", "``") + "`")
        finally:
            cursor.close()
            connection.close()

    def stop(self) -> None:
        # The datadir is discarded, so an immediate kill is safe.
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process.wait()
            self.process = None
        shutil.rmtree(self.root / "data", ignore_errors=True)

    def close(self) -> None:
        self.stop()
        shutil.rmtree(self.root, ignore_errors=True)


class _Validator:
    """Context manager owning the judge's validator and its pristine template."""

    def __init__(self):
        self.root: Path | None = None
        self.server: _MariaDB | None = None

    def __enter__(self) -> _MariaDB:
        self.root = Path(tempfile.mkdtemp(prefix="polygon-sql-template-"))
        try:
            self.server = _MariaDB(_validator_template(self.root))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self.server

    def __exit__(self, _type, _value, _traceback):
        if self.server is not None:
            self.server.close()
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
        return False


def _reference_budget_seconds(workload: Workload) -> int:
    """The paired reference's search budget for one case.

    The reference only sets the bar; it must not be timeout-flaky. Its full
    final self-run took up to the whole 12 s on the D100 disambiguation case at
    the declared 2 CPU / 8 GiB (9.2-11.2 s in other replays), so it receives
    max(2x, +10 s) of the candidate's budget. The candidate's budget, payload,
    scoring normalization and both gates are unchanged.
    """
    return max(2 * workload.timeout_seconds, workload.timeout_seconds + REFERENCE_BUDGET_EXTRA_SECONDS)


def _limit_seconds(workload: Workload, budget: int) -> int:
    """One search instance's whole life (setup, search, result), the old runner's limit."""
    margin = HARD_SUBPROCESS_MARGIN_SECONDS if workload.stratum == HARD_STRATUM else SUBPROCESS_MARGIN_SECONDS
    return max(SUBPROCESS_TIMEOUT_SECONDS, budget + margin)


def _search_payload(workload: Workload, budget: int) -> dict:
    """What the search process reads: the fixed adapter's child payload (frontier_runner.main)."""
    payload = dict(workload.payload)
    payload["timeout_seconds"] = budget
    if workload.kind == "pair":
        payload["groundtruth"] = _adapter()._pair_inputs(payload, _pristine_root())["groundtruth"]
    return payload


def _plan(workloads: tuple[Workload, ...]) -> list[dict]:
    """Every side of every case in measurement order: the reference first on even cases, the
    candidate first on odd ones."""
    entries = []
    for index, workload in enumerate(workloads):
        order = (REFERENCE_SIDE, CANDIDATE_SIDE) if index % 2 == 0 else (CANDIDATE_SIDE, REFERENCE_SIDE)
        for side in order:
            budget = _reference_budget_seconds(workload) if side == REFERENCE_SIDE else workload.timeout_seconds
            entries.append({
                "index": index, "side": side, "budget_seconds": budget,
                "setup_seconds": SEARCH_SETUP_SECONDS,
                "limit_seconds": _limit_seconds(workload, budget),
                "result_max_bytes": _adapter().RESULT_MAX_BYTES,
                "payload": _search_payload(workload, budget),
            })
    return entries


def _search_run(candidate: Path, reference: Path, entries: list[dict], files: Path):
    """The one trusted sandbox run: the judge's harness, both trees and the plan go up."""
    from fcs2_verifier_gpu import SandboxRun, SandboxStep, sandbox_workdir

    root = sandbox_workdir(SANDBOX)
    for name in HARNESS_FILES:
        shutil.copy2(TASK_DIR / name, files / name)
    shutil.copy2(_pristine_root() / RUNNER, files / RUNNER)
    for side, tree in ((CANDIDATE_SIDE, candidate), (REFERENCE_SIDE, reference)):
        shutil.copytree(tree, files / "trees" / side, symlinks=True)
    (files / "cases.json").write_text(json.dumps(entries), encoding="utf-8")
    for path in (files, *files.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    step = SandboxStep([PYTHON, f"{root}/search_harness.py", root], None,
                       {"PYTHONDONTWRITEBYTECODE": "1"})
    return SandboxRun(SANDBOX, files, [step], ["out"], SEARCH_OUTPUT_BYTES)


def _harness_records(result, entries: list[dict]) -> Path:
    """The harness's output directory, or SearchSandboxFailure (the judge's own fault)."""
    step = result.steps[0]
    tail = (result.logs[0] or b"").decode("utf-8", errors="replace")[-500:]
    if step is None or step.exit_code != 0 or result.outputs is None:
        raise SearchSandboxFailure(f"the search sandbox run failed: {step}: {tail}")
    out = result.outputs / "out"
    missing = [n for n in range(len(entries)) if not (out / f"{n}.json").is_file()]
    if missing:
        raise SearchSandboxFailure(f"the search harness returned no record for {len(missing)} searches")
    return out


def _case_record(index: int, workload: Workload, side: str) -> dict:
    return {
        "index": index,
        "workload": workload.identifier,
        "kind": workload.kind,
        "stratum": workload.stratum,
        "timeout_seconds": workload.timeout_seconds,
        "side": side,
    }


def _observe(workload: Workload, entry: dict, harness: dict, message: bytes | None,
             validator: _MariaDB, record: dict) -> Observation:
    """One side of one case: the harness's measurement, the returned counterexample validated
    on the judge's fresh MariaDB. The fixed adapter's rules (frontier_runner.main): an
    undecided search is charged the full budget; a failed search is an invalid case."""
    budget = entry["budget_seconds"]
    record.update({"budget_seconds": budget, "reason": harness["reason"],
                   "returncode": harness["returncode"], "limit": harness["limit"],
                   "wall_seconds": harness["wall_seconds"], "stderr_tail": harness["stderr_tail"],
                   "backend": harness["backend"], "result_over_cap": harness["result_over_cap"]})
    invalid = Observation(False, float(budget), 0, 0, workload.timeout_seconds, False)
    if harness["reason"] is not None:
        return invalid
    adapter = _adapter()
    elapsed = harness["seconds"] if harness["decided"] else None
    try:
        reported = json.loads(message.decode("utf-8")) if message is not None else {}
    except ValueError:
        record["reason"] = "malformed search result"
        return invalid
    if not isinstance(reported, dict):
        reported, elapsed = {}, None
    decided = reported.get("decided") is True and elapsed is not None
    refuted = False
    if decided and reported.get("eq") is False and "cex" in reported:
        try:
            counterexample = adapter._decode(reported["cex"])
        except Exception as error:  # noqa: BLE001 - a malformed counterexample fails its case
            record["reason"] = f"malformed counterexample: {type(error).__name__}"
            return invalid
        started = time.monotonic()
        socket = validator.fresh()
        try:
            if workload.kind == "pair":
                inputs = adapter._pair_inputs(entry["payload"], _pristine_root())
                refuted = adapter._validate_pair(entry["payload"], inputs, counterexample,
                                                 adapter.MySQLTester, socket)
            else:
                refuted = adapter._validate_disambiguation(entry["payload"], counterexample,
                                                           adapter.MySQLTester, socket)
        except Exception as error:  # noqa: BLE001 - as in the fixed adapter: the case fails
            record["reason"] = f"validation failed: {type(error).__name__}"
            return invalid
        finally:
            validator.stop()
            record["validation_seconds"] = round(time.monotonic() - started, 3)
    record["reported_time_seconds"] = adapter._reported(reported.get("reported_time_seconds"))
    seconds = min(float(budget), max(0.0, elapsed)) if decided else float(budget)
    return Observation(bool(refuted), seconds, adapter._as_int(reported.get("iterations")),
                       adapter._as_int(reported.get("backtracks")), workload.timeout_seconds)


def _finish_record(record: dict, row: Observation) -> None:
    record.update(
        {
            "valid": bool(row.valid),
            "refuted": bool(row.refuted),
            "time_seconds": round(float(row.time_seconds), 6),
            "iterations": int(row.iterations),
            "backtracks": int(row.backtracks),
        }
    )


def _measure_pair(
    candidate: Path,
    reference: Path,
    workloads: tuple[Workload, ...],
    records: list[dict] | None = None,
) -> tuple[list[Observation], list[Observation]]:
    """Measure every case on both sides on the search sandbox; validate here.

    One trusted sandbox machine runs both sides' searches, one at a time in the planned order,
    each in a fresh isolate instance (no network, no judge file), timed by the harness outside it.
    Each returned counterexample is validated on the judge's MariaDB, started fresh for it on
    the control image, which the candidate's code can neither reach nor run on.
    """
    from fcs2_verifier_gpu import run_sandboxes

    records = [] if records is None else records
    entries = _plan(workloads)
    scratch = Path(tempfile.mkdtemp(prefix="polygon-search-"))
    try:
        (scratch / "files").mkdir()
        (scratch / "into").mkdir()
        (result,) = run_sandboxes([_search_run(candidate, reference, entries, scratch / "files")],
                                  into=scratch / "into")
        out = _harness_records(result, entries)
        rows: dict[tuple[int, str], Observation] = {}
        with _Validator() as validator:
            for number, entry in enumerate(entries):
                workload = workloads[entry["index"]]
                record = _case_record(entry["index"], workload, entry["side"])
                records.append(record)
                harness = json.loads((out / f"{number}.json").read_text(encoding="utf-8"))
                message_path = out / f"{number}.message"
                message = message_path.read_bytes() if message_path.is_file() else None
                row = _observe(workload, entry, harness, message, validator, record)
                _finish_record(record, row)
                rows[entry["index"], entry["side"]] = row
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return ([rows[index, CANDIDATE_SIDE] for index in range(len(workloads))],
            [rows[index, REFERENCE_SIDE] for index in range(len(workloads))])


def _private_case_evidence(reason: str, records: list[dict]) -> dict:
    """Bounded operator-only evidence: every case, stderr only for failed rows."""
    rows = []
    for record in records:
        row = dict(record)
        keep = row.get("valid") is not True or (
            row.get("side") == REFERENCE_SIDE and row.get("refuted") is not True
        )
        if not keep:
            row.pop("stderr_tail", None)
        rows.append(row)
    evidence = {
        "execution_valid": False,
        "reason": reason,
        "isolation": "searches on the trusted search sandbox (isolate, no network); "
                     "validation on the judge's fresh MariaDB per counterexample",
        "cases": rows,
    }
    # Shed stderr tails, newest first, until the evidence fits its contract.
    holders = [row for row in rows if "stderr_tail" in row]
    while holders and len(json.dumps(evidence, sort_keys=True).encode("utf-8")) > DIAGNOSTIC_MAX_BYTES:
        holders.pop().pop("stderr_tail")
        evidence["stderr_truncated"] = True
    return evidence


def _original(workloads, rows):
    """The continuous score keeps its v2 definition over the original stratum."""
    return [row for workload, row in zip(workloads, rows) if workload.stratum == ORIGINAL_STRATUM]


def _comparison_decisions(
    workloads: tuple[Workload, ...],
    candidate: list[Observation],
    reference: list[Observation],
) -> dict[str, dict[str, bool]]:
    output = {}
    for workload, cand, ref in zip(workloads, candidate, reference):
        if workload.stratum == HARD_STRATUM:
            # Strict superiority: no allowance, and a noise-safe ratio.
            faster = cand.time_seconds < HARD_TIME_RATIO * ref.time_seconds
        else:
            tolerance = max(
                TIME_COMPARISON_ABS_TOLERANCE,
                TIME_COMPARISON_REL_TOLERANCE * ref.time_seconds,
            )
            faster = cand.time_seconds <= ref.time_seconds + tolerance
        output[workload.identifier] = {
            "refuted_pair": int(cand.refuted) >= int(ref.refuted),
            "wall_time_seconds": faster,
        }
    return output


def evaluate(tree: str) -> tuple[float, float, str, dict]:
    reference_root = None
    try:
        workloads = _load_suite()
        candidate_root = Path(tree)
        try:
            reference_root = _reference_tree()
        except subprocess.CalledProcessError:
            return _invalid("reference patch is invalid", workloads)

        _build(candidate_root)
        _build(reference_root)
        records: list[dict] = []
        try:
            candidate_rows, reference_rows = _measure_pair(
                candidate_root, reference_root, workloads, records
            )
        except ValidatorFailure as failure:
            failure.private_execution_evidence = _private_case_evidence(
                str(failure)[:128], records
            )
            raise
        if not all(row.valid and row.refuted for row in reference_rows):
            raise ReferenceFailure(
                "reference execution failed",
                _private_case_evidence("reference execution failed", records),
            )

        candidate_score = score_observations(_original(workloads, candidate_rows))
        reference_score = score_observations(_original(workloads, reference_rows))
        decisions = _comparison_decisions(
            workloads, candidate_rows, reference_rows
        )
        metrics = _metrics(
            workloads,
            candidate_rows,
            reference_score,
            candidate_score,
            decisions,
        )
        solved = sum(row.refuted for row in candidate_rows)
        message = f"validated {solved}/{len(workloads)} Polygon workloads"
        over_cap = [f"workload-{record['index'] + 1}" for record in records
                    if record["side"] == CANDIDATE_SIDE and record["result_over_cap"]]
        if over_cap:
            message += (f"; failed: the search result exceeded the "
                        f"{_adapter().RESULT_MAX_BYTES}-byte output cap on {', '.join(over_cap)}")
        return candidate_score, candidate_score, message, metrics
    except (ReferenceFailure, VerifierGpuError):
        raise  # the judge's or the platform's fault: unscored, never a candidate zero
    except Exception as exc:
        try:
            workloads
        except UnboundLocalError:
            workloads = tuple()
        return _invalid(f"evaluation failed: {type(exc).__name__}", workloads)
    finally:
        if reference_root is not None:
            shutil.rmtree(reference_root.parent, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: evaluator.py TREE", file=sys.stderr)
        raise SystemExit(2)
    score, unbounded, message, metrics = evaluate(sys.argv[1])
    print(
        json.dumps(
            {
                "score": score,
                "score_unbounded": unbounded,
                "message": message,
                "metrics": metrics,
            },
            sort_keys=True,
        )
    )
