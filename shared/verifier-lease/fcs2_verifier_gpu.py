#!/usr/bin/env python3
"""FrontierCS's command test sandboxes for a judge that Pumpkin runs: the interface of the shared
GPU adapter's `run_sandboxes` (its "Command test sandboxes"), over the sessions of the judge's
role, which it drives through `sandboxctl call <session>.<op>`. A
task's judge imports it as the source's (`from fcs2_verifier_gpu import run_sandboxes, ...`), so
its code is the source's.

    run_sandboxes([SandboxRun(sandbox, files, steps, outputs=(), max_output_bytes=None)], *, into)
        -> [SandboxResult(steps, outputs, outputs_exceeded, logs, unreadable_outputs)]

Each run is a lease of the judge's session named as the sandbox (`fcs_task`'s
`test_sandboxes` gives each judge one), on a machine of the session's runner role: all the runs are
leased at once, at most the session's `instances` of one sandbox, which is the sandbox's
`max_concurrent`, and each lease lasts the session's `instance_seconds` from its machine's start,
which is the sandbox's `lease_seconds` for the judge's mode. Then, as in the source:

- `files` (a local directory) is put as the sandbox's `workdir` (verifier.json's `sandboxes`);
  nothing else reaches the machine.
- The steps run there in order, each `SandboxStep(argv, seconds, env)` under `timeout` with the
  image's environment plus `env`, `TMPDIR` below `/workspace` and
  FRONTIER_VERIFIER_LOCAL_DEADLINE, the machine's `time.monotonic()` at the step's limit. The
  last step may set seconds=None: its limit is the rest of the lease, less the source's per-step
  grace and collect reserve, fixed when it starts. A step that runs past its limit is
  StepResult(None, True, ...); one whose machine failed under it is StepResult(None, False, ...,
  lost_session=True), and the run ends with outputs None; a non-zero exit is StepResult(code,
  False, ...); a timed-out or failed step ends the run.
- The declared outputs that exist (files or directories below the workdir) come back into
  `into/<i>`, at most max_output_bytes (default and maximum the grant's `transfer_bytes`); over
  it, outputs None and outputs_exceeded True. `logs[i]` is the last 8 KiB of step i's output, its
  standard output and error together; `unreadable_outputs` lists the declared files the machine's
  user could not read, which are not returned.
- Every lease is closed, and its machine gone, before run_sandboxes returns or raises; a failing
  run stops the others. A failure of the platform (a machine that does not start, a transfer or
  a close that fails, a lease too short for the next step) raises VerifierGpuError, which the
  shared verifier reports unscored.

Unlike the source's, a tree keeps no file mode but whether a file is executable, and holds no
symbolic link (a link in `files` is refused before anything is leased), and a machine is driven
by the host, with no SSH session, so there is no transport to lose but the machine itself.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath
from typing import NamedTuple

SANDBOXCTL = "/.pumpkin/bin/sandboxctl"
PARAMETERS = Path("/opt/frontier/verifier.json")
WORKSPACE = "/workspace/"  # FrontierCS's runner user's home; every sandbox's workdir is below it
# Pumpkin's /tmp is read-only: a step's TMPDIR, and the declared outputs on their way back.
SANDBOX_TMP = "/workspace/frontier-sandbox-tmp"
SANDBOX_COLLECT = "/workspace/frontier-sandbox-collect"
# The source's bounds on a run, and its reserves around each step and for the collect.
STEP_KILL_AFTER_SECONDS = 5
STEP_TRANSPORT_GRACE_SECONDS = 60
SANDBOX_UPLOAD_SECONDS = 120
SANDBOX_COLLECT_SECONDS = 120
MAX_SANDBOX_STEPS = 16
MAX_SANDBOX_OUTPUTS = 64
MAX_LOG_TAIL_BYTES = 8192
MAX_ARGV_BYTES = 16 * 1024
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
# How long the host may take to remove a closed lease's machine (the source's delete reserve),
# and what is kept of the judge's call for its result and evidence after that.
DELETE_RESERVE_SECONDS = 660
EVIDENCE_RESERVE_SECONDS = 30
# The states of a lease that has ended for good; `closing` still has its machine.
ENDED = frozenset({"closed", "expired", "failed"})
STEP_STATUS = re.compile(r"frontier-sandbox-step ([0-9]{1,3}) ([0-9]{1,20})")
COLLECT_STATUS = re.compile(r"frontier-sandbox-collect ([0-9]{1,20})")

# One step on the machine: the image's interpreter turns the limit into the machine's monotonic
# deadline, as the source's runtime prelude does; the step runs in the workdir under `timeout`,
# its output and errors together on standard output; the last line of standard error is its
# exit status and elapsed nanoseconds.
STEP_SCRIPT = r"""limit=$1 workdir=$2 tmp=$3; shift 3
for python in /opt/frontier/runtime-python /opt/conda/bin/python3 /usr/bin/python3; do
    [ -x "$python" ] && break
done
mkdir -p "$tmp" || exit 125
deadline=$("$python" -c 'import sys, time; print(repr(time.monotonic() + float(sys.argv[1])))' "$limit") || exit 125
export FRONTIER_VERIFIER_LOCAL_DEADLINE="$deadline" TMPDIR="$tmp"
cd "$workdir" || exit 125
start=$(date +%s%N)
timeout --kill-after=5 "$limit" env -- "$@" </dev/null 2>&1
status=$?
end=$(date +%s%N)
printf 'frontier-sandbox-step %s %s\n' "$status" "$((end - start))" >&2
exit "$status"
"""

# The declared outputs that exist, copied below the collect directory as they lie in the workdir,
# with only regular files and directories kept; standard output lists the declared files (or files
# below them) the runner user cannot read, one per line, and the last line of standard error is
# the bytes of files to collect.
COLLECT_SCRIPT = r"""workdir=$1 collect=$2; shift 2
rm -rf "$collect" && mkdir -p "$collect" || exit 125
cd "$workdir" || exit 125
for output in "$@"; do
    if [ -e "$output" ] || [ -L "$output" ]; then
        find "$output" ! -type l ! -readable -print 2>/dev/null
        mkdir -p "$collect/$(dirname "$output")" && cp -R -P "$output" "$collect/$output" 2>/dev/null
    fi
done
find "$collect" ! -type f ! -type d -exec rm -f {} + 2>/dev/null
bytes=$(find "$collect" -type f -printf '%s\n' | awk '{ total += $1 } END { print total + 0 }')
printf 'frontier-sandbox-collect %s\n' "$bytes" >&2
"""


class VerifierGpuError(Exception):
    """The platform's failure, not the submission's: the shared verifier reports it unscored. Not
    a RuntimeError, so a judge's `except RuntimeError` never turns it into a score."""

    def __init__(self, message: str, *, private_evidence: dict | None = None):
        super().__init__(message)
        self.private_evidence = private_evidence


class _SandboxCancelled(VerifierGpuError):
    def __init__(self):
        super().__init__("the command sandbox run was stopped after another run failed")


class SandboxStep(NamedTuple):
    """One command of a run: argv (argv[0] absolute), its wall limit in seconds and extra
    environment variables. The last step of a run may set seconds=None: its limit is the rest of
    the lease (less the per-step grace and the collect reserve), fixed when it starts."""

    argv: list[str] | tuple[str, ...]
    seconds: int | None
    env: dict[str, str] | None = None


class SandboxRun(NamedTuple):
    """One machine of a declared command sandbox: `files` (a local directory) becomes the
    sandbox's workdir, `steps` run there in order, `outputs` (paths relative to the workdir,
    files or directories) come back. `max_output_bytes` defaults to (and may not exceed) the
    grant's `transfer_bytes`."""

    sandbox: str
    files: Path | str
    steps: list[SandboxStep] | tuple[SandboxStep, ...]
    outputs: list[str] | tuple[str, ...] = ()
    max_output_bytes: int | None = None


class StepResult(NamedTuple):
    """`exit_code` is None when the step timed out or its machine failed under it
    (`lost_session`)."""

    exit_code: int | None
    timed_out: bool
    seconds: float
    lost_session: bool = False


class SandboxResult(NamedTuple):
    """`steps[i]` is None for a step that did not run. `outputs` is the local directory holding
    the declared outputs that existed (laid out as in the workdir), None when they exceeded
    max_output_bytes (`outputs_exceeded`) or the machine failed. `logs[i]` is the last 8 KiB of
    step i's output (None if it did not run). `unreadable_outputs`: declared outputs (or files
    below them, workdir-relative) the machine's user could not read, not returned."""

    steps: list[StepResult | None]
    outputs: Path | None
    outputs_exceeded: bool
    logs: list[bytes | None]
    unreadable_outputs: list[str] = []


def _ctl(*args) -> dict:
    """The reply of `sandboxctl <args>`; a refusal or an error is the platform's failure."""
    done = subprocess.run([SANDBOXCTL, *map(str, args)], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, check=False)
    try:
        reply = json.loads(done.stdout)
    except ValueError:
        reply = None
    if not isinstance(reply, dict):
        raise VerifierGpuError(f"sandboxctl {args[0]} {args[1] if len(args) > 1 else ''} failed: "
                               f"{done.stderr.strip()[-512:]}")
    if reply.get("kind") == "refused":
        raise VerifierGpuError(f"sandboxctl {' '.join(map(str, args[:2]))} was refused: "
                               f"{str(reply.get('message'))[:512]}")
    return reply


def sandbox_parameters() -> dict:
    """verifier.json `sandboxes`: the task's command test sandboxes, by name, as FrontierCS's build
    writes them (this adapter reads each one's `workdir`)."""
    return json.loads(PARAMETERS.read_text(encoding="utf-8"))["sandboxes"]


def sandbox_workdir(name: str) -> str:
    """The declared workdir of command test sandbox `name`: where its files land on a machine."""
    sandboxes = sandbox_parameters()
    if name not in sandboxes:
        raise VerifierGpuError(f"no command test sandbox {name!r} in task.toml")
    return sandboxes[name]["workdir"]


def _relative(path) -> PurePosixPath:
    if not isinstance(path, str) or not path or "\x00" in path or "\n" in path:
        raise VerifierGpuError("a sandbox output is not a relative path")
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or relative.parts in ((), (".",)):
        raise VerifierGpuError(f"a sandbox output must be a path below the workdir: {path!r}")
    return relative


def _check_files(files: Path) -> None:
    if not files.is_dir() or files.is_symlink():
        raise VerifierGpuError("a sandbox run's files is not a directory")
    for path in files.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise VerifierGpuError(f"a sandbox upload holds {path.relative_to(files)}, which is "
                                   "not a regular file or a directory: Pumpkin moves only those")


def _sandbox_requests(runs, sandboxes: dict, grants: dict) -> list[SandboxRun]:
    """Refuse a malformed set of runs before any machine is leased, as the source does, with the
    grant's bounds for the sandbox's."""
    runs = list(runs) if isinstance(runs, (list, tuple)) else None
    if not runs:
        raise VerifierGpuError("run_sandboxes needs a non-empty list of SandboxRun")
    counts: dict[str, int] = {}
    for run in runs:
        if not isinstance(run, SandboxRun):
            raise VerifierGpuError("a sandbox run is not a SandboxRun")
        if run.sandbox not in sandboxes:
            raise VerifierGpuError(f"no command test sandbox {run.sandbox!r} in task.toml "
                                   f"(declared: {sorted(sandboxes)})")
        if run.sandbox not in grants:
            raise VerifierGpuError(f"this judge holds no session {run.sandbox!r} "
                                   f"(held: {sorted(grants)})")
        grant = grants[run.sandbox]
        counts[run.sandbox] = counts.get(run.sandbox, 0) + 1
        _check_files(Path(run.files))
        steps = list(run.steps) if isinstance(run.steps, (list, tuple)) else []
        if not 1 <= len(steps) <= MAX_SANDBOX_STEPS:
            raise VerifierGpuError(f"a sandbox run needs 1..={MAX_SANDBOX_STEPS} steps")
        budget = SANDBOX_UPLOAD_SECONDS + SANDBOX_COLLECT_SECONDS
        for number, step in enumerate(steps):
            if not isinstance(step, SandboxStep):
                raise VerifierGpuError("a sandbox step is not a SandboxStep")
            argv = step.argv
            if (not isinstance(argv, (list, tuple)) or not argv
                    or not all(isinstance(item, str) and "\x00" not in item for item in argv)
                    or not argv[0].startswith("/")
                    or sum(len(item.encode()) + 1 for item in argv) > MAX_ARGV_BYTES):
                raise VerifierGpuError("a sandbox step's argv must be a bounded list of strings "
                                       "starting with an absolute program path")
            if step.seconds is None:
                if number != len(steps) - 1:
                    raise VerifierGpuError("only the last sandbox step may take the rest of the "
                                           "grant (seconds=None)")
            elif type(step.seconds) is not int or step.seconds < 1:
                raise VerifierGpuError("a sandbox step's seconds must be a positive integer")
            env = step.env or {}
            if (not isinstance(env, dict)
                    or not all(isinstance(k, str) and ENV_NAME.fullmatch(k) and isinstance(v, str)
                               and "\x00" not in v for k, v in env.items())):
                raise VerifierGpuError("a sandbox step's env must map variable names to strings")
            budget += (step.seconds or 1) + STEP_KILL_AFTER_SECONDS + STEP_TRANSPORT_GRACE_SECONDS
        if budget > grant["instance_seconds"]:
            raise VerifierGpuError(
                f"the steps of a {run.sandbox!r} run need up to {budget} s with upload, collect "
                f"and per-step grace; its lease lasts {grant['instance_seconds']} s")
        outputs = list(run.outputs) if isinstance(run.outputs, (list, tuple)) else None
        if outputs is None or len(outputs) > MAX_SANDBOX_OUTPUTS:
            raise VerifierGpuError(f"a sandbox run declares at most {MAX_SANDBOX_OUTPUTS} outputs")
        for output in outputs:
            _relative(output)
        capture = grant["transfer_bytes"]
        if run.max_output_bytes is not None and (
                type(run.max_output_bytes) is not int or not 1 <= run.max_output_bytes <= capture):
            raise VerifierGpuError(f"a sandbox run's max_output_bytes must be in 1..={capture}")
    for name, count in counts.items():
        if count > grants[name]["instances"]:
            raise VerifierGpuError(f"{count} runs of {name!r} at once; its max_concurrent is "
                                   f"{grants[name]['instances']}")
    return runs


class _Lease:
    """One lease of the judge's, driven op by op; `wait` polls its status until `done` says."""

    def __init__(self, grant: str, cancel: threading.Event, end: float):
        self.cancel = cancel
        self.end = end  # the judge's call's end, on this machine's monotonic clock
        self.grant = grant
        self.number = self._call("open")["lease"]
        self.ran = 0  # the commands and transfers started, which number them from 0

    def _call(self, op: str, *args) -> dict:
        return _ctl("call", f"{self.grant}.{op}", *args)

    def status(self) -> dict:
        return self._call("status", self.number)

    def wait(self, done, *, until: float | None = None, cancellable: bool = True) -> dict:
        """The first status that `done` accepts, polled each second for half a minute and then
        every five; None once `until` (or the call's end) has passed."""
        until = self.end if until is None else min(until, self.end)
        started = time.monotonic()
        while True:
            if cancellable and self.cancel.is_set():
                raise _SandboxCancelled()
            reply = self.status()
            if done(reply):
                return reply
            now = time.monotonic()
            if now >= until:
                return None
            time.sleep(min(1.0 if now - started < 30 else 5.0, until - now))

    def start(self, *args) -> dict:
        """Start a command or a transfer, then wait until its end is known or the lease has
        ended under it."""
        self._call(args[0], self.number, *args[1:])
        number, self.ran = self.ran, self.ran + 1

        def done(reply):
            last = reply.get("last") or {}
            ours = last.get("command", last.get("transfer")) == number
            return reply["state"] != "running" and (
                reply["state"] != "ready" or (ours and last.get("end") is not None))
        return self.wait(done)

    def run(self, argv: list[str], timeout: int) -> dict:
        return self.start("run", "--timeout", timeout, "--", *argv)

    def transfer(self, op: str, source: str, to: str) -> dict:
        reply = self.start(op, source, to)
        last = (reply or {}).get("last") or {}
        if reply is None or reply["state"] != "ready" or last.get("kind") != "transfer":
            raise VerifierGpuError(f"the sandbox machine ended during a {op}")
        return last["end"]

    def close(self) -> bool:
        """Close the lease and wait until its machine is gone: whether that was seen."""
        try:
            self._call("close", self.number)
        except VerifierGpuError:
            pass  # it may have ended already; its status says
        until = min(time.monotonic() + DELETE_RESERVE_SECONDS, self.end)
        try:
            return self.wait(lambda reply: reply["state"] in ENDED, until=until,
                             cancellable=False) is not None
        except VerifierGpuError:
            return False


def _step_result(index: int, step: SandboxStep, lease: _Lease, workdir: str,
                 deadline: float) -> tuple[StepResult, bytes | None]:
    grace = STEP_KILL_AFTER_SECONDS + STEP_TRANSPORT_GRACE_SECONDS
    start = time.monotonic()
    limit = (step.seconds if step.seconds is not None
             else math.floor(deadline - start - grace - SANDBOX_COLLECT_SECONDS))
    if limit < 1 or start + limit + grace > deadline:
        raise VerifierGpuError("the sandbox machine's lease cannot fit the next step")
    variables = [f"{key}={value}" for key, value in sorted((step.env or {}).items())]
    argv = ["/bin/sh", "-c", STEP_SCRIPT, "frontier-sandbox-step", str(limit), workdir,
            SANDBOX_TMP, *variables, *step.argv]
    reply = lease.run(argv, limit + STEP_KILL_AFTER_SECONDS + 10)
    elapsed = time.monotonic() - start
    last = (reply or {}).get("last") or {}
    end = last.get("end") if last.get("kind") == "command" else None
    if reply is None or reply["state"] == "expired" or end == "timed_out":
        return StepResult(None, True, elapsed), None
    if end is None or end == "unstarted":
        return StepResult(None, False, elapsed, lost_session=True), None
    log = last.get("stdout", "").encode("utf-8")[-MAX_LOG_TAIL_BYTES:]
    match = STEP_STATUS.fullmatch(last.get("stderr", "").strip().rsplit("\n", 1)[-1])
    exited = end.get("exited", {})
    if match is None:
        code = exited.get("code", 128 + exited.get("signal", 0))
        return StepResult(code, False, elapsed), log
    status, seconds = int(match.group(1)), int(match.group(2)) / 1e9
    if status in (124, 137) and seconds >= limit:
        return StepResult(None, True, seconds), log
    return StepResult(status, False, seconds), log


def _collect(lease: _Lease, request: SandboxRun, workdir: str, capture: int,
             destination: Path, record: dict) -> tuple[Path | None, bool, list[str]]:
    """The declared outputs into `destination`: (outputs, exceeded, unreadable)."""
    if not request.outputs:
        destination.mkdir()
        return destination, False, []
    limit = request.max_output_bytes or capture
    argv = ["/bin/sh", "-c", COLLECT_SCRIPT, "frontier-sandbox-collect", workdir,
            SANDBOX_COLLECT, *request.outputs]
    reply = lease.run(argv, SANDBOX_COLLECT_SECONDS)
    last = (reply or {}).get("last") or {}
    match = COLLECT_STATUS.fullmatch(last.get("stderr", "").strip().rsplit("\n", 1)[-1])
    if reply is None or reply["state"] != "ready" or last.get("end") != {"exited": {"code": 0}} \
            or match is None:
        raise VerifierGpuError("the sandbox outputs could not be gathered")
    unreadable = sorted(line for line in last.get("stdout", "").splitlines() if line)
    if int(match.group(1)) > limit:
        record["collected_bytes"] = int(match.group(1))
        return None, True, unreadable
    end = lease.transfer("get", SANDBOX_COLLECT, str(destination))
    if "failed" in end:
        raise VerifierGpuError(f"the sandbox outputs could not be collected: {end['failed'][:512]}")
    record["collected_bytes"] = end["moved"]["bytes"]
    if sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()) > limit:
        shutil.rmtree(destination)
        return None, True, unreadable
    return destination, False, unreadable


def _sandbox_run(index: int, request: SandboxRun, workdir: str, capture: int, end: float,
                 destination: Path, cancel: threading.Event, record: dict) -> SandboxResult:
    lease = None
    try:
        record["phase"] = "provisioning"
        if cancel.is_set():
            raise _SandboxCancelled()
        lease = _Lease(request.sandbox, cancel, end)
        record["lease"] = lease.number
        reply = lease.wait(lambda reply: reply["state"] != "starting")
        if reply is None or reply["state"] != "ready":
            failure = (reply or {}).get("failure") or (reply or {}).get("state", "no start")
            raise VerifierGpuError(f"the sandbox machine did not start: {str(failure)[:512]}")
        deadline = min(time.monotonic() + reply["remaining"]["instance_seconds"], end)
        record["granted_seconds"] = reply["remaining"]["instance_seconds"]
        record["phase"] = "upload"
        moved = lease.transfer("put", str(Path(request.files).resolve()), workdir)
        if "failed" in moved:
            raise VerifierGpuError(f"the sandbox upload failed: {moved['failed'][:512]}")
        steps: list[StepResult | None] = [None] * len(request.steps)
        logs: list[bytes | None] = [None] * len(request.steps)
        for number, step in enumerate(request.steps):
            record["phase"] = f"step-{number}"
            steps[number], logs[number] = _step_result(number, step, lease, workdir, deadline)
            if steps[number].timed_out or steps[number].exit_code != 0:
                break
        record["steps"] = [None if r is None else "lost" if r.lost_session
                           else "timeout" if r.timed_out else r.exit_code for r in steps]
        if any(result is not None and result.lost_session for result in steps):
            record["status"] = "completed"
            return SandboxResult(steps, None, False, [None] * len(steps))
        if any(result is not None and result.timed_out for result in steps) \
                and lease.status()["state"] != "ready":
            record["status"] = "completed"
            return SandboxResult(steps, None, False, logs)
        record["phase"] = "collect"
        outputs, exceeded, unreadable = _collect(lease, request, workdir, capture, destination,
                                                 record)
        record["status"] = "completed"
        return SandboxResult(steps, outputs, exceeded, logs, unreadable)
    except _SandboxCancelled:
        record["status"], record["reason"] = "cancelled", "cancelled"
        raise
    except BaseException:
        record["status"] = "failed"
        record.setdefault("reason", record["phase"])
        cancel.set()
        raise
    finally:
        if lease is None:
            record["termination"] = "not_requested"
        else:
            confirmed = lease.close()
            record["termination"] = "confirmed" if confirmed else "unconfirmed"
            if not confirmed:
                cancel.set()


def _held_grants() -> tuple[dict, float]:
    """The judge's sessions by name, and its call's end on the monotonic clock."""
    reply = _ctl("ops")
    end = time.monotonic() + reply["remaining_seconds"] - EVIDENCE_RESERVE_SECONDS
    return {grant["name"]: grant for grant in reply["grants"] if grant.get("kind") == "session"}, end


def run_sandboxes(runs, *, into) -> list[SandboxResult]:
    """Run each SandboxRun on its own machine of its declared command test sandbox, all at once;
    the results in order. `into` is an empty local directory the judge owns: run i's outputs go to
    `into/<i>`. See the module's docstring."""
    sandboxes = sandbox_parameters()
    mode = os.environ.get("FRONTIER_VERIFICATION_MODE")
    grants, end = _held_grants()
    runs = _sandbox_requests(runs, sandboxes, grants)
    into = Path(into)
    if not into.is_dir() or any(into.iterdir()):
        raise VerifierGpuError("run_sandboxes needs an empty directory `into`")
    records = [{"run": index, "sandbox": run.sandbox, "phase": "pending", "status": "started",
                "termination": "not_requested"} for index, run in enumerate(runs)]
    cancel = threading.Event()
    results: list = [None] * len(runs)
    errors: list = []

    def one(index: int) -> None:
        run = runs[index]
        try:
            results[index] = _sandbox_run(
                index, run, sandboxes[run.sandbox]["workdir"],
                grants[run.sandbox]["transfer_bytes"], end, into / str(index), cancel,
                records[index])
        except BaseException as error:  # noqa: BLE001 - classified below
            errors.append(error)

    threads = [threading.Thread(target=one, args=(index,), name=f"frontier-sandbox-{index}",
                                daemon=True) for index in range(len(runs))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    evidence = {"schema": "frontiersmith.verifier-sandboxes/v1", "mode": mode,
                "runs": [{key: value for key, value in record.items()
                          if key in ("run", "sandbox", "lease", "phase", "status", "reason",
                                     "steps", "termination", "granted_seconds",
                                     "collected_bytes")}
                         for record in records]}
    cause = next((error for error in errors if not isinstance(error, _SandboxCancelled)),
                 errors[0] if errors else None)
    if any(record["termination"] not in ("confirmed", "not_requested") for record in records):
        raise VerifierGpuError("command sandbox machine removal was not confirmed",
                               private_evidence=evidence) from cause
    if cause is not None:
        raise VerifierGpuError(f"a command sandbox run failed: {cause}",
                               private_evidence=evidence) from cause
    return results
