"""Evaluator for the excised Polygon under-approximation search policy."""

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
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from submission_patch import apply_submission_patch


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
CONFIG = {
    "submission": {
        "repo_root": "/app",
        "allow_empty": False,
        "allow_paths": [
            "sql_testing/environment.py",
            "sql_testing/smt/formula.py",
            "sql_testing/smt/knowledgebase.py",
            "sql_testing/smt/search_extensions/",
        ],
        "max_changed_files": 20,
        "max_patch_bytes": 262144,
    }
}


def _pristine_root() -> Path:
    """Resolve the one public source tree in source and packaged layouts."""
    packaged = TASK_DIR / "harbor" / "app"
    if packaged.is_dir():
        return packaged
    return TASK_DIR.parent / "agent-base" / "frontier" / "task" / "repo"

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
    "frontier_",
    "harbor_",
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


def _policy_ok(patch: Path) -> tuple[bool, str]:
    if patch.read_bytes() == REFERENCE_PATCH.read_bytes():
        return True, ""
    text = patch.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        lowered = line[1:].lower()
        if any(token in lowered for token in FORBIDDEN_ADDITIONS):
            return False, "environment, file, process, or harness introspection is not allowed"
    return True, ""


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
VALIDATOR_START_SECONDS = 20.0
VALIDATOR_START_ATTEMPTS = 2
VALIDATOR_MAX_CONNECTIONS = 32
DIAGNOSTIC_STDERR_BYTES = 1536
DIAGNOSTIC_MAX_BYTES = 120 * 1024
# A small InnoDB footprint keeps template copies and restarts cheap. The
# validator only executes the two queries on the found counterexample outside
# every timed region, so its storage sizing never enters a metric.
VALIDATOR_STORAGE_ARGS = (
    "--innodb-log-file-size=4M",
    "--innodb-undo-tablespaces=0",
    "--innodb-data-file-path=ibdata1:4M:autoextend",
)
VALIDATOR_SYSTEM_SCHEMAS = frozenset({"mysql", "information_schema", "performance_schema", "sys"})
# Persistent optimizer statistics follow every CREATE/DROP TABLE by design.
VALIDATOR_VOLATILE_SYSTEM_TABLES = frozenset({"innodb_table_stats", "innodb_index_stats"})


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
    """A private SQL validator could not be provisioned for a case.

    Infrastructure, not the candidate: never scored as a candidate zero.
    """


def _validator_template(directory: Path) -> Path:
    """Initialize one pristine datadir; every server (re)start uses a copy."""
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
    """One side's private SQL validator.

    The candidate and the reference each own a separate server process,
    datadir and socket created from one pristine template; only the owning
    side's worker is ever given its socket. Before and after every case of its
    side, ``reset()`` kills every client connection (with any running query,
    lock, temporary table or session variable), drops every non-system
    database, and verifies global variables and system tables against the
    pristine fingerprint; any drift or reset error restarts the server from
    the template. Between its cases the server is frozen (SIGSTOP), so it
    neither runs nor answers while the other side is measured.
    """

    def __init__(self, template: Path, side: str):
        self.template = template
        self.side = side
        self.root = Path(tempfile.mkdtemp(prefix=f"polygon-{side}-sql-"))
        self.process: subprocess.Popen | None = None
        self.starts = 0
        self.baseline: dict | None = None

    @property
    def socket(self) -> str:
        return str(self.root / "mysql.sock")

    def _server_args(self, data: Path) -> list[str]:
        server = shutil.which("mariadbd") or "/usr/sbin/mariadbd"
        args = [server, "--no-defaults"]
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            # The Harbor judge starts as root.  MariaDB refuses to launch as
            # root unless this is explicit; retain the normal unprivileged
            # behavior for local runs under another UID.
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
        """(Re)start this side's server on a new copy of the pristine template."""
        self.stop()
        for _attempt in range(VALIDATOR_START_ATTEMPTS):
            try:
                if self._start_once():
                    # The pristine state has no user database at all (the
                    # installer's empty `test` schema included).
                    self._clear()
                    self.baseline = self._fingerprint()
                    return self.socket
            except Exception:  # noqa: BLE001 - any provisioning error retries once
                pass
            self.stop()
        raise ValidatorFailure(f"{self.side} SQL validator did not start")

    def _connect(self):
        import mysql.connector  # the declared judge runtime; workers use the same client

        connection = mysql.connector.connect(
            unix_socket=self.socket, user="root", connection_timeout=10
        )
        cursor = connection.cursor()
        # A session inherits tampered globals; pin what the reset itself reads.
        cursor.execute(
            "SET SESSION lock_wait_timeout = 10, sql_select_limit = 18446744073709551615, "
            "max_join_size = 18446744073709551615, sql_mode = ''"
        )
        return connection, cursor

    def _fingerprint(self) -> dict:
        connection, cursor = self._connect()
        try:
            cursor.execute("SHOW GLOBAL VARIABLES")
            variables = {str(name): str(value) for name, value in cursor.fetchall()}
            cursor.execute(
                "SELECT TABLE_NAME FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = 'mysql' AND TABLE_TYPE = 'BASE TABLE'"
            )
            names = sorted(
                str(row[0]) for row in cursor.fetchall()
                if str(row[0]).lower() not in VALIDATOR_VOLATILE_SYSTEM_TABLES
            )
            checksums = {}
            if names:
                cursor.execute(
                    "CHECKSUM TABLE "
                    + ", ".join("mysql.`" + name.replace("`", "``") + "`" for name in names)
                )
                checksums = {str(name): str(value) for name, value in cursor.fetchall()}
            return {"variables": variables, "system_tables": checksums}
        finally:
            cursor.close()
            connection.close()

    def _clear(self) -> dict:
        """Kill every other client connection and drop every user database."""
        connection, cursor = self._connect()
        summary = {"killed_connections": 0, "dropped_databases": 0}
        try:
            deadline = time.monotonic() + 10.0
            while True:
                cursor.execute(
                    "SELECT ID FROM information_schema.PROCESSLIST "
                    "WHERE ID <> CONNECTION_ID() AND COMMAND <> 'Daemon'"
                )
                others = [int(row[0]) for row in cursor.fetchall()]
                if not others:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("client connections survived KILL")
                for identifier in others:
                    try:
                        cursor.execute(f"KILL CONNECTION {identifier}")
                        summary["killed_connections"] += 1
                    except Exception:  # noqa: BLE001 - it may exit on its own
                        pass
                time.sleep(0.02)
            cursor.execute("SELECT SCHEMA_NAME FROM information_schema.SCHEMATA")
            for (name,) in cursor.fetchall():
                if str(name).lower() in VALIDATOR_SYSTEM_SCHEMAS:
                    continue
                cursor.execute("DROP DATABASE `" + str(name).replace("`", "``") + "`")
                summary["dropped_databases"] += 1
        finally:
            cursor.close()
            connection.close()
        return summary

    def _sql_reset(self) -> dict:
        summary = self._clear()
        fingerprint = self._fingerprint()
        drift = sorted(
            f"{group}:{key}"
            for group in ("variables", "system_tables")
            for key in set(fingerprint[group]) | set(self.baseline[group])
            if fingerprint[group].get(key) != self.baseline[group].get(key)
        )
        if drift:
            summary["drift"] = drift[:16]
        return summary

    def reset(self) -> dict:
        """Return this side's server to the pristine state, restarting if needed."""
        self.thaw()
        summary: dict = {}
        if self.process is None or self.process.poll() is not None:
            summary["server_exited"] = True
        else:
            try:
                summary.update(self._sql_reset())
            except Exception as error:  # noqa: BLE001 - restart covers every failure
                summary["reset_error"] = type(error).__name__[:64]
        if summary.get("server_exited") or summary.get("reset_error") or summary.get("drift"):
            started = time.monotonic()
            self.fresh()
            summary["restarted"] = True
            summary["restart_seconds"] = round(time.monotonic() - started, 3)
        return summary

    def _signal(self, number: int) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, number)
            except ProcessLookupError:
                pass

    def freeze(self) -> None:
        self._signal(signal.SIGSTOP)

    def thaw(self) -> None:
        self._signal(signal.SIGCONT)

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        # The datadir is discarded, so an immediate kill is safe and also ends
        # every client thread and in-flight query.
        if self.process is not None:
            self._signal(signal.SIGKILL)
            self.process.wait()
            self.process = None
        shutil.rmtree(self.root / "data", ignore_errors=True)

    def close(self) -> None:
        self.stop()
        shutil.rmtree(self.root, ignore_errors=True)


class _Validators:
    """Context manager owning both sides' private validators."""

    def __init__(self):
        self.root: Path | None = None
        self.sides: dict[str, _MariaDB] = {}

    def __enter__(self) -> dict[str, _MariaDB]:
        self.root = Path(tempfile.mkdtemp(prefix="polygon-sql-template-"))
        try:
            template = _validator_template(self.root)
            for side in (CANDIDATE_SIDE, REFERENCE_SIDE):
                self.sides[side] = _MariaDB(template, side)
                self.sides[side].fresh()
                self.sides[side].freeze()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self.sides

    def __exit__(self, _type, _value, _traceback):
        for validator in self.sides.values():
            validator.close()
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
        return False


def _worker_oom_priority() -> None:
    # Keep solver pressure from selecting a SQL validator as its victim.
    descriptor = os.open("/proc/self/oom_score_adj", os.O_WRONLY)
    try:
        os.write(descriptor, b"1000")
    finally:
        os.close(descriptor)


def _kill_session(session: int) -> None:
    """Kill every process left in the case's session, whatever its process group."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rindex(")") + 2:].split()
            if int(fields[3]) == session:
                os.kill(int(entry.name), signal.SIGKILL)
        except (OSError, ValueError, IndexError):
            pass


def _stderr_tail(handle) -> str:
    try:
        handle.flush()
        size = handle.seek(0, os.SEEK_END)
        handle.seek(max(0, size - DIAGNOSTIC_STDERR_BYTES))
        raw = handle.read()
    except (OSError, ValueError):
        return ""
    # ASCII keeps the JSON size bounded; progress bars carry no diagnosis.
    return raw.decode("ascii", errors="replace").replace("\r", "\n")


def _reference_budget_seconds(workload: Workload) -> int:
    """The paired reference's search budget for one case.

    The reference only sets the bar; it must not be timeout-flaky. Its full
    final self-run took up to the whole 12 s on the D100 disambiguation case at
    the declared 2 CPU / 8 GiB (9.2-11.2 s in other replays), so it receives
    max(2x, +10 s) of the candidate's budget. The candidate's budget, payload,
    scoring normalization and both gates are unchanged.
    """
    return max(2 * workload.timeout_seconds, workload.timeout_seconds + REFERENCE_BUDGET_EXTRA_SECONDS)


def _run_case(
    root: Path,
    workload: Workload,
    socket: str,
    record: dict | None = None,
    budget_seconds: int | None = None,
) -> Observation:
    """Run one side of one case; fill ``record`` with private diagnostics.

    ``budget_seconds`` defaults to the workload's (candidate) budget; only the
    paired reference passes a larger one.
    """
    record = {} if record is None else record
    budget = workload.timeout_seconds if budget_seconds is None else int(budget_seconds)
    record.update({"returncode": None, "timed_out": False, "reason": None, "budget_seconds": budget})
    payload = dict(workload.payload)
    payload["timeout_seconds"] = budget
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["PATH"] = "/usr/bin:/usr/local/bin:/bin"
    env["FRONTIER_MYSQL_SOCKET"] = socket
    case_temp = None
    stderr_file = None
    started = time.monotonic()
    try:
        case_temp = tempfile.mkdtemp(prefix="polygon-case-")
        env["TMPDIR"] = case_temp
        stderr_file = tempfile.TemporaryFile(prefix="polygon-stderr-")
        proc = subprocess.Popen(
            [sys.executable, str(root / RUNNER)],
            cwd=root,
            env=env,
            stdin=subprocess.PIPE,
            text=True,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            start_new_session=True,
            preexec_fn=_worker_oom_priority,
        )
        stdout = ""
        try:
            stdout, _ = proc.communicate(
                json.dumps(payload),
                timeout=max(SUBPROCESS_TIMEOUT_SECONDS, budget + (HARD_SUBPROCESS_MARGIN_SECONDS if workload.stratum == HARD_STRATUM else SUBPROCESS_MARGIN_SECONDS)),
            )
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
            raise ValueError("worker timed out")
        finally:
            # A timed-out search can leave Z3 children competing with the paired reference.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _kill_session(proc.pid)
            proc.wait()
            record["returncode"] = proc.returncode
            for stream in (proc.stdin, proc.stdout):
                if stream is not None:
                    stream.close()
        lines = [line for line in stdout.splitlines() if line.startswith("{")]
        if len(lines) > 1:
            raise ValueError("worker printed more than one result")
        if proc.returncode or not lines:
            if lines:
                try:
                    error = json.loads(lines[-1]).get("error")
                    if isinstance(error, str):
                        record["worker_error"] = error[:128]
                except (ValueError, AttributeError):
                    pass
            raise ValueError("worker failed")
        result = json.loads(lines[-1])
        if result.get("ok") is not True:
            raise ValueError("worker rejected the result")
        # time_seconds is measured by the fixed adapter; the submission's own
        # duration is private telemetry only.
        record["reported_time_seconds"] = result.get("reported_time_seconds")
        elapsed = float(result["time_seconds"])
        iterations = int(result["iterations"])
        backtracks = int(result["backtracks"])
        if (
            not math.isfinite(elapsed)
            or elapsed < 0
            or elapsed > budget
            or iterations < 0
            or backtracks < 0
        ):
            raise ValueError("worker metrics are invalid")
        return Observation(
            bool(result["refuted"]), elapsed, iterations, backtracks,
            workload.timeout_seconds,
        )
    except Exception as error:
        record["reason"] = (
            str(error)[:128] if isinstance(error, ValueError) else type(error).__name__
        )
        return Observation(
            False, float(budget), 0, 0,
            workload.timeout_seconds, False,
        )
    finally:
        record["wall_seconds"] = round(time.monotonic() - started, 3)
        if stderr_file is not None:
            record["stderr_tail"] = _stderr_tail(stderr_file)
            stderr_file.close()
        if case_temp is not None:
            shutil.rmtree(case_temp, ignore_errors=True)


def _case_record(index: int, workload: Workload, side: str) -> dict:
    return {
        "index": index,
        "workload": workload.identifier,
        "kind": workload.kind,
        "stratum": workload.stratum,
        "timeout_seconds": workload.timeout_seconds,
        "side": side,
    }


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
    """Measure every case on both sides, each against its own validator.

    Run order alternates exactly as before. Each side's worker only receives
    its own side's socket; that server is reset to the pristine state (and
    verified) immediately before and after the side's case, and frozen while
    the other side runs, so neither side's database state, connections,
    queries or server settings can reach the other side's case.
    """
    records = [] if records is None else records
    candidate_rows: list[Observation] = []
    reference_rows: list[Observation] = []
    roots = {CANDIDATE_SIDE: candidate, REFERENCE_SIDE: reference}
    with _Validators() as validators:
        for index, workload in enumerate(workloads):
            order = (
                (REFERENCE_SIDE, CANDIDATE_SIDE)
                if index % 2 == 0
                else (CANDIDATE_SIDE, REFERENCE_SIDE)
            )
            rows = {}
            for side in order:
                validator = validators[side]
                record = _case_record(index, workload, side)
                records.append(record)
                record["validator_before"] = validator.reset()
                try:
                    row = _run_case(
                        roots[side], workload, validator.socket, record,
                        _reference_budget_seconds(workload) if side == REFERENCE_SIDE else None,
                    )
                finally:
                    record["validator_after"] = validator.reset()
                    record["validator_starts"] = validator.starts
                    validator.freeze()
                _finish_record(record, row)
                rows[side] = row
            candidate_rows.append(rows[CANDIDATE_SIDE])
            reference_rows.append(rows[REFERENCE_SIDE])
    return candidate_rows, reference_rows


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
        "isolation": "per-side private MariaDB, fresh datadir per case",
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


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    candidate_root = None
    reference_root = None
    try:
        workloads = _load_suite()
        patch = Path(solution_path)
        applied = apply_submission_patch(
            patch, TASK_DIR, CONFIG, pristine=_pristine_root()
        )
        if not applied.ok:
            return _invalid(applied.message, workloads)
        candidate_root = applied.workdir

        ok, reason = _policy_ok(patch)
        if not ok:
            return _invalid(reason, workloads)
        is_reference = patch.read_bytes() == REFERENCE_PATCH.read_bytes()
        if not is_reference:
            ok, reason = _scaffold_ok(candidate_root)
            if not ok:
                return _invalid(reason, workloads)
            ok, reason = _imports_ok(candidate_root)
            if not ok:
                return _invalid(reason, workloads)

        restored = apply_submission_patch(
            REFERENCE_PATCH, TASK_DIR, CONFIG, pristine=_pristine_root()
        )
        if not restored.ok:
            return _invalid("reference patch is invalid", workloads)
        reference_root = restored.workdir

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
        return candidate_score, candidate_score, message, metrics
    except ReferenceFailure:
        raise
    except Exception as exc:
        try:
            workloads
        except UnboundLocalError:
            workloads = tuple()
        return _invalid(f"evaluation failed: {type(exc).__name__}", workloads)
    finally:
        for root in (candidate_root, reference_root):
            if root is not None:
                shutil.rmtree(root.parent, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: evaluator.py SOLUTION_PATCH", file=sys.stderr)
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
