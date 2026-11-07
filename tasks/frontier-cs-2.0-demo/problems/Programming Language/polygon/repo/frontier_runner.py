#!/usr/bin/env python3
"""Small adapter around the retained LeetCode worker.

This process never imports submitted code. It loads every validation input,
runs the submitted search in a child process, times that search itself, and
validates the returned counterexample with MariaDB after the child is stopped.
"""

from __future__ import annotations

import builtins
import datetime
import decimal
import importlib
import io
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


_clock = time.monotonic  # CLOCK_MONOTONIC: one system-wide clock for both processes
_real_open = builtins.open

SEARCH_FLAG = "--search"
# Untimed preparation of the search process (third-party imports and the SQL
# grammar) must finish within this limit.
SETUP_SECONDS = 30.0
START_MAX_BYTES = 64
RESULT_MAX_BYTES = 16 * 1024 * 1024

# Third-party libraries the fixed tree imports at module level. The search
# process loads them before timing starts, identically for every submission.
_SHARED_LIBRARIES = (
    "numpy", "pandas", "z3", "multiprocess.pool", "bidict", "lark",
    "mo_parsing", "mo_sql_parsing", "mysql.connector", "prettytable", "tqdm",
    "ujson", "yaml",
)
_PLAIN_LEAVES = (
    type(None), bool, int, float, str, bytes, decimal.Decimal,
    datetime.date, datetime.datetime, datetime.time, datetime.timedelta,
)
_DEFAULT_K_OPERATORS = (
    "filter", "project", "union", "inner join", "left join",
    "right join", "full join", "product", "order by",
)


# --------------------------------------------------------------------------
# Plain counterexample values and their transport encoding.

def _plain(value, depth=0):
    """Copy a returned counterexample into plain built-in values."""
    if depth > 8:
        raise ValueError("counterexample is nested too deeply")
    kind = type(value)
    if kind in _PLAIN_LEAVES:
        return value
    if isinstance(value, dict):
        return {_plain(k, depth + 1): _plain(v, depth + 1) for k, v in dict.items(value)}
    if isinstance(value, list):
        return [_plain(v, depth + 1) for v in list.__iter__(value)]
    if isinstance(value, tuple):
        return tuple(_plain(v, depth + 1) for v in tuple.__iter__(value))
    if isinstance(value, int):
        return int.__int__(value)
    if isinstance(value, float):
        return float.__float__(value)
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, bytes):
        return bytes.__bytes__(value)
    raise TypeError("counterexample values must be plain")


def _encode(value):
    """Plain value -> JSON value; every JSON object is a type tag."""
    kind = type(value)
    if value is None or kind in (bool, int, str):
        return value
    if kind is float:
        return value if math.isfinite(value) else {"$float": repr(value)}
    if kind is decimal.Decimal:
        return {"$decimal": str(value)}
    if kind is bytes:
        return {"$bytes": value.hex()}
    if kind is datetime.datetime:
        return {"$datetime": value.isoformat()}
    if kind is datetime.date:
        return {"$date": value.isoformat()}
    if kind is datetime.time:
        return {"$time": value.isoformat()}
    if kind is datetime.timedelta:
        return {"$timedelta": [value.days, value.seconds, value.microseconds]}
    if kind is list:
        return [_encode(item) for item in value]
    if kind is tuple:
        return {"$tuple": [_encode(item) for item in value]}
    if kind is dict:
        return {"$dict": [[_encode(k), _encode(v)] for k, v in value.items()]}
    raise TypeError("counterexample values must be plain")


def _decode(value, depth=0):
    if depth > 24:
        raise ValueError("counterexample is nested too deeply")
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is list:
        return [_decode(item, depth + 1) for item in value]
    if type(value) is not dict or len(value) != 1:
        raise ValueError("malformed counterexample encoding")
    (tag, body), = value.items()
    if tag == "$float":
        return float(body)
    if tag == "$decimal":
        return decimal.Decimal(body)
    if tag == "$bytes":
        return bytes.fromhex(body)
    if tag == "$datetime":
        return datetime.datetime.fromisoformat(body)
    if tag == "$date":
        return datetime.date.fromisoformat(body)
    if tag == "$time":
        return datetime.time.fromisoformat(body)
    if tag == "$timedelta":
        days, seconds, microseconds = body
        return datetime.timedelta(days=days, seconds=seconds, microseconds=microseconds)
    if tag == "$tuple":
        return tuple(_decode(item, depth + 1) for item in body)
    if tag == "$dict":
        return {_decode(k, depth + 1): _decode(v, depth + 1) for k, v in body}
    raise ValueError("malformed counterexample encoding")


def _as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _reported(value):
    """The submission's own duration, kept as telemetry only."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


# --------------------------------------------------------------------------
# Search process: imports the submitted tree and runs one check/disambiguate.

def _message(result) -> dict:
    """The worker's view of a check result, fixed before the message is sent."""
    if result is None:
        return {"decided": False}
    eq, cex, _checking_time, total_time, ret = result
    message = {"decided": eq is not None}
    if eq is not None:
        message["eq"] = bool(eq)
        if not message["eq"]:
            message["cex"] = _encode(_plain(cex))
    telemetry = dict(ret) if isinstance(ret, dict) else {}
    message["reported_time_seconds"] = _reported(total_time)
    message["iterations"] = _as_int(telemetry.get("iters"))
    message["backtracks"] = _as_int(telemetry.get("backtracks"))
    return message


class _Result:
    """Sends the settled result once, then ends the search process at once."""

    def __init__(self, descriptor: int):
        self._descriptor = descriptor

    def send(self, result) -> None:
        try:
            line = json.dumps(_message(result), allow_nan=False) + "\n"
        except Exception:  # noqa: BLE001 - a result that cannot be settled is undecided
            line = json.dumps({"decided": False}) + "\n"
        data = line.encode("utf-8")
        while data:
            data = data[os.write(self._descriptor, data):]
        os._exit(0)


def _import_open(path, *args, **kwargs):
    # The retained entrypoint eagerly reads a 32 MB result log used only for
    # cross-tool reporting.  The benchmark below does not consume that report.
    if str(path).endswith("leetcode_RQ1_2023_03_27_23_07_03.out"):
        return io.StringIO("")
    return _real_open(path, *args, **kwargs)


def _load_tree():
    """Import the worker and the submitted modules; runs inside the timed window."""
    builtins.open = _import_open
    try:
        from evaluation.leetcode import online
    finally:
        builtins.open = _real_open
    from evaluation.calcite.tester import MySQLTester  # noqa: F401 - same imports for every submission
    from sql_testing.environment import Environment
    return online, Environment


def _warm_parser() -> None:
    """Build the SQL grammar before timing starts, identically for every submission."""
    try:
        import mo_sql_parsing

        mo_sql_parsing.parse("SELECT 1")
    except Exception:  # noqa: BLE001 - a broken parser fails later, inside the timed call
        pass


def _search_pair(payload: dict, result: _Result) -> None:
    timeout_seconds = int(payload["timeout_seconds"])
    online = _load_tree()[0]
    original_environment = online.Environment
    depth = [0]

    class BoundedEnvironment(original_environment):
        def __init__(self, *args, **kwargs):
            kwargs["time_budget"] = timeout_seconds
            super().__init__(*args, **kwargs)

        def check(self, *args, **kwargs):
            if depth[0]:
                return super().check(*args, **kwargs)
            depth[0] += 1
            try:
                settled = super().check(*args, **kwargs)
            except BaseException:  # noqa: BLE001 - a failed check is undecided
                settled = None
            result.send(settled)

    online.Environment = BoundedEnvironment
    # The search has no SQL validator; validation runs after it is stopped.
    online.DB_CONFIG = {}
    with tempfile.TemporaryDirectory(prefix="polygon-search-") as directory:
        os.chdir(directory)
        online.leetcode_worker(
            0, int(payload["problem_id"]), payload["groundtruth"],
            [(int(payload["qid"]), str(payload["query"]))], int(payload.get("bound", 2)),
        )


def _search_disambiguation(payload: dict, result: _Result) -> None:
    bound = int(payload.get("bound", 1))
    _online, Environment = _load_tree()
    environment = Environment(
        list(payload["schema"]), [], bound=bound,
        default_k={name: bound for name in _DEFAULT_K_OPERATORS},
        time_budget=int(payload["timeout_seconds"]),
    )
    try:
        settled = environment.disambiguate(
            [str(query) for query in payload["queries"]], group_range=0, use_precise_encoding=False
        )
    except BaseException:  # noqa: BLE001 - a failed call is undecided
        settled = None
    result.send(settled)


def _search_main(start_descriptor: int, result_descriptor: int) -> None:
    payload = json.loads(sys.stdin.read())
    for name in _SHARED_LIBRARIES:
        try:
            importlib.import_module(name)
        except ImportError:
            pass  # a missing library fails later, inside the timed import
    _warm_parser()
    result = _Result(result_descriptor)
    # The start time is read here, before any submitted module is imported.
    os.write(start_descriptor, f"{_clock()!r}\n".encode("ascii"))
    os.close(start_descriptor)
    if payload.get("kind") == "disambiguation":
        _search_disambiguation(payload, result)
    else:
        _search_pair(payload, result)
    result.send(None)  # the worker returned without calling check


# --------------------------------------------------------------------------
# Adapter process: inputs, timing and validation.

def _pair_inputs(payload: dict, root: Path) -> dict:
    import pandas as pd
    from sql_testing.constraint_parser import ConstraintParser

    problem_id = int(payload["problem_id"])
    benchmark = root / "benchmarks" / "leetcode"
    return {
        "groundtruth": str(pd.read_csv(benchmark / "groundtruths" / f"{problem_id}.csv")["mysql_groundtruth"][0]),
        "schema": json.loads((benchmark / "schemas" / f"{problem_id}.json").read_text(encoding="utf-8")),
        "constraints": ConstraintParser().parse_from_file(str(benchmark / "constraints" / f"{problem_id}.yml")),
    }


def _validate_pair(payload: dict, inputs: dict, cex, tester_class, socket: str) -> bool:
    with tester_class(
        {"unix_socket": socket, "user": "root"}, inputs["schema"]["Tables"], db_predix="leetcode_0"
    ) as tester:
        tester.create_all_databases([cex], inputs["constraints"])
        return bool(tester.test_pair(inputs["groundtruth"], str(payload["query"])))


def _validate_disambiguation(payload: dict, cex, tester_class, socket: str) -> bool:
    queries = [str(query) for query in payload["queries"]]
    with tester_class(
        {"unix_socket": socket, "user": "root"},
        list(payload["schema"]),
        db_predix="frontier_disambiguation",
    ) as tester:
        tester.create_all_databases([cex], [])
        tester.cursor.execute("USE frontier_disambiguation_0")
        clusters = {}
        for query in queries:
            try:
                tester.cursor.execute(query)
                key = tuple(tester.cursor.fetchall())
            except Exception:
                key = "query_not_executable"
            clusters[key] = clusters.get(key, 0) + 1
    counts = sorted(int(count) for key, count in clusters.items() if key != "query_not_executable")
    return (
        "query_not_executable" not in clusters
        and len(counts) == 2
        and counts == [len(queries) // 2, len(queries) // 2]
    )


def _measure(child_payload: dict, budget: float) -> tuple[dict | None, float | None]:
    """Run the search process; return its result message and its measured seconds.

    Time runs from the search process's start reading, taken just before it
    imports the submitted modules, to this process's arrival time of the
    complete result message. A result that has not arrived within the budget
    is undecided (``None`` seconds).
    """
    start_read, start_write = os.pipe()
    result_read, result_write = os.pipe()
    env = dict(os.environ)
    env.pop("FRONTIER_MYSQL_SOCKET", None)
    child = None
    selector = selectors.DefaultSelector()
    try:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), SEARCH_FLAG, str(start_write), str(result_write)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            pass_fds=(start_write, result_write),
            env=env,
            process_group=0,
        )
        os.close(start_write)
        os.close(result_write)
        start_write = result_write = None
        child.stdin.write(json.dumps(child_payload).encode("utf-8"))
        child.stdin.close()
        selector.register(start_read, selectors.EVENT_READ, "start")
        selector.register(result_read, selectors.EVENT_READ, "result")
        started = None
        start_line = bytearray()
        result_line = bytearray()
        setup_deadline = _clock() + SETUP_SECONDS
        while selector.get_map():
            now = _clock()
            deadline = setup_deadline if started is None else started + budget
            if now >= deadline:
                break
            events = selector.select(deadline - now)
            stamp = _clock()
            # The start message is always handled before a result read in the same wakeup.
            for key, _mask in sorted(events, key=lambda event: event[0].data != "start"):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fd)
                elif key.data == "start":
                    if started is None:
                        start_line += chunk
                        if len(start_line) > START_MAX_BYTES:
                            raise ValueError("malformed start message")
                        if b"\n" in start_line:
                            reading = float(start_line[: start_line.index(b"\n")].decode("ascii"))
                            if not math.isfinite(reading) or reading > stamp:
                                raise ValueError("malformed start message")
                            started = reading
                elif started is not None:
                    result_line += chunk
                    if len(result_line) > RESULT_MAX_BYTES:
                        raise ValueError("search result is too large")
                    if b"\n" in result_line:
                        if stamp - started > budget:
                            return None, None
                        line = bytes(result_line[: result_line.index(b"\n")])
                        return json.loads(line.decode("utf-8")), stamp - started
        if started is None:
            raise RuntimeError("search process did not start")
        if selector.get_map():
            return None, None  # the budget ran out first
        raise RuntimeError("search process exited without a result")
    finally:
        selector.close()
        for descriptor in (start_read, result_read, start_write, result_write):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def main() -> int:
    payload = json.loads(sys.stdin.read())
    socket = os.environ["FRONTIER_MYSQL_SOCKET"]
    root = Path(__file__).resolve().parent
    kind = "disambiguation" if payload.get("kind") == "disambiguation" else "pair"
    budget = int(payload.get("timeout_seconds", 10 if kind == "disambiguation" else 4))
    # Every validation input and library is loaded before submitted code runs.
    from evaluation.calcite.tester import MySQLTester

    child_payload = {key: value for key, value in payload.items() if key != "groundtruth"}
    child_payload["kind"] = kind
    child_payload["timeout_seconds"] = budget
    inputs = None
    if kind == "pair":
        inputs = _pair_inputs(payload, root)
        child_payload["groundtruth"] = inputs["groundtruth"]

    message, elapsed = _measure(child_payload, budget)
    if not isinstance(message, dict):
        message, elapsed = {}, None
    decided = message.get("decided") is True and elapsed is not None
    refuted = False
    if decided and message.get("eq") is False and "cex" in message:
        counterexample = _decode(message["cex"])
        if kind == "pair":
            refuted = _validate_pair(payload, inputs, counterexample, MySQLTester, socket)
        else:
            refuted = _validate_disambiguation(payload, counterexample, MySQLTester, socket)
    # An undecided search is charged the full case budget.
    seconds = min(float(budget), max(0.0, elapsed)) if decided else float(budget)
    print(json.dumps({
        "ok": True,
        "refuted": bool(refuted),
        "time_seconds": seconds,
        "reported_time_seconds": _reported(message.get("reported_time_seconds")),
        "iterations": _as_int(message.get("iterations")),
        "backtracks": _as_int(message.get("backtracks")),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == SEARCH_FLAG:
        _search_main(int(sys.argv[2]), int(sys.argv[3]))
        raise SystemExit(0)
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        raise SystemExit(1)
