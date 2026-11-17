"""The trusted search harness on the CPU command sandbox (task.toml [test_sandboxes.search]).

    search_harness.py ROOT

judge/evaluator.py uploads, as ROOT (the sandbox's workdir): this harness, search_child.py, the
judge's copy of the starter's adapter (frontier_runner.py), both trees (trees/candidate,
trees/reference) and cases.json, the measurement plan: one entry per side of each case in the
order to run them (reference first on even cases, candidate first on odd ones), each with the
search payload, its budget and limits and the output cap of its result line
(result_max_bytes). The harness runs every entry's search in a fresh
shared-isolate instance (search_child.py: the instance's stdout pipe is its only channel to the
harness; no network, no judge file, nothing left from the previous entry) and
times it itself, as the fixed adapter did: from the start reading the search process takes just
before it imports the submitted modules to the arrival of its complete result line here. It
writes ROOT/out/<n>.json per entry ({decided, seconds, reason, ...}) and ROOT/out/<n>.message
(the result line, at most result_max_bytes, unparsed). The counterexample is validated on the
judge's own MariaDB, never here.

Output cap: the judge collects ROOT/out in one run_sandboxes() collect, which has a fixed size
limit; the per-entry cap keeps every plan's out/ below it. A result line longer than its cap is
never truncated: the harness stops reading, kills the instance and records the case as failed
with result_over_cap true (the candidate's fault; on the reference side the judge's).

AF_UNIX sockets: the search needs them for its own IPC (the starter's search starts a
multiprocess SyncManager, which listens on a pathname socket below TMPDIR). Under Landlock an
instance can connect to any pathname socket it can name, which is harmless on this machine: no
validator or other service listens here, and only one instance runs at a time.

The harness fails (non-zero exit) only for its own faults: an IsolateError, or when
FRONTIER_VERIFIER_LOCAL_DEADLINE leaves too little time for the next entry. A search's
failure is a fact in its record.
"""

from __future__ import annotations

import json
import math
import os
import selectors
import sys
import time
from pathlib import Path

import isolate  # the shared launcher, on the image interpreter's import path (frontier.pth)

START_MAX_BYTES = 64
DIAGNOSTIC_STDERR_BYTES = 1536
STDERR_BYTES = 64 << 20
RESERVE_SECONDS = 30   # to finish writing out/ and exit before the step's deadline
PATH = "/usr/bin:/usr/local/bin:/bin"


def _stderr_tail(raw: bytes) -> str:
    raw = raw[-DIAGNOSTIC_STDERR_BYTES:]
    return raw.decode("ascii", errors="replace").replace("\r", "\n")


def over_cap_reason(result_max: int) -> str:
    return f"search result exceeds the {result_max}-byte output cap"


def _measure(instance: isolate.Instance, budget: float, setup_seconds: float,
             limit_end: float, result_max: int) -> tuple[bytes | None, float | None, str | None]:
    """(result line, measured seconds, failure reason) of one search instance.

    The fixed adapter's protocol: the first line is the start reading (time.monotonic(), one
    system-wide clock), the next the result. A result that arrives after the budget is
    undecided (None, None, None); a search that does not start, ends without a result, sends a
    malformed start or a result line over `result_max` bytes, or outlives the instance limit
    is a failure.
    """
    selector = selectors.DefaultSelector()
    descriptor = instance.stdout.fileno()
    selector.register(descriptor, selectors.EVENT_READ)
    started = None
    buffer = bytearray()
    setup_end = time.monotonic() + setup_seconds
    try:
        while True:
            now = time.monotonic()
            phase_end = setup_end if started is None else started + budget
            end = min(phase_end, limit_end)
            if now >= end:
                if end == limit_end and limit_end < phase_end:
                    return None, None, "worker timed out"
                if started is None:
                    return None, None, "search process did not start"
                return None, None, None  # the budget ran out first
            events = selector.select(end - now)
            stamp = time.monotonic()
            if not events:
                continue
            chunk = os.read(descriptor, 65536)
            if not chunk:
                if started is None:
                    return None, None, "search process did not start"
                return None, None, "search process exited without a result"
            buffer += chunk
            if started is None:
                if b"\n" not in buffer[:START_MAX_BYTES + 1]:
                    if len(buffer) > START_MAX_BYTES:
                        return None, None, "malformed start message"
                    continue
                line, _, rest = bytes(buffer).partition(b"\n")
                try:
                    reading = float(line.decode("ascii"))
                except ValueError:
                    return None, None, "malformed start message"
                if not math.isfinite(reading) or reading > stamp:
                    return None, None, "malformed start message"
                started = reading
                buffer = bytearray(rest)
            if started is not None:
                end_of_line = buffer.find(b"\n")
                if (end_of_line if end_of_line >= 0 else len(buffer)) > result_max:
                    return None, None, over_cap_reason(result_max)
                if end_of_line >= 0:
                    if stamp - started > budget:
                        return None, None, None
                    return bytes(buffer[:end_of_line]), stamp - started, None
    finally:
        selector.close()


def run_entry(root: Path, entry: dict, out: Path, number: int) -> None:
    budget = float(entry["budget_seconds"])
    launched = time.monotonic()
    instance = isolate.start(
        [sys.executable, "{in}/search_child.py", "{in}/frontier_runner.py"],
        tree=root / "trees" / entry["side"],
        inputs={"search_child.py": root / "search_child.py",
                "frontier_runner.py": root / "frontier_runner.py"},
        env={"PYTHONPATH": "{work}", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
        timeout_seconds=entry["limit_seconds"], path=PATH, output_bytes=STDERR_BYTES,
        unix_sockets=True)
    try:
        instance.stdin.write(json.dumps(entry["payload"]).encode("utf-8"))
        instance.stdin.close()
    except BrokenPipeError:
        pass  # the search ended before reading its input: no start line follows
    try:
        result_max = int(entry["result_max_bytes"])
        message, seconds, reason = _measure(instance, budget, float(entry["setup_seconds"]),
                                            launched + float(entry["limit_seconds"]), result_max)
    finally:
        result = instance.close()
    record = {"index": entry["index"], "side": entry["side"], "decided": message is not None,
              "seconds": seconds, "reason": reason,
              "result_over_cap": reason == over_cap_reason(result_max),
              "returncode": result.returncode,
              "limit": result.limit, "backend": result.backend,
              "wall_seconds": round(time.monotonic() - launched, 3),
              "stderr_tail": _stderr_tail(result.stderr)}
    if message is not None:
        (out / f"{number}.message").write_bytes(message)
    (out / f"{number}.json").write_text(json.dumps(record, sort_keys=True))


def main(argv: list[str]) -> int:
    root = Path(argv[0])
    deadline = float(os.environ["FRONTIER_VERIFIER_LOCAL_DEADLINE"])
    entries = json.loads((root / "cases.json").read_text(encoding="utf-8"))
    out = root / "out"
    out.mkdir(exist_ok=True)
    for number, entry in enumerate(entries):
        if deadline - time.monotonic() - RESERVE_SECONDS < float(entry["limit_seconds"]):
            raise SystemExit("the step's deadline leaves no time for the next search")
        run_entry(root, entry, out, number)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
