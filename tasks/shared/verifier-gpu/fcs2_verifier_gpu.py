#!/usr/bin/env python3
"""Run FrontierSmith verifier work on Driver-owned Modal GPUs (shared GPU adapter, gpu-tools).

The EKS Verifier Pod remains the control plane.  It already has a verification-scoped Driver
bearer, so this adapter creates Verifier-owned runners, connects only through the Driver's
audited SSH gateway, and copies bounded results back.  The Modal credential and provider
address never enter this process or the task image.

The bearer comes from the shared verifier's sealed handle (fcs2_verifier.verifier_token(); the
shared verifier removed it from os.environ before any task code ran).  It stays in this process:
no child gets it in its environment or argv.  ssh's ProxyCommand (`_connect`, a child of ssh)
reaches the Driver CONNECT endpoint through a Unix socket that this process serves
(_ConnectBroker); this process sends the bearer and relays the upgraded stream.

Configuration (task.toml `[gpu.verifier]`, generated into /opt/frontier/verifier.json `gpu`;
the GPU type and count stay FRONTIER_VERIFIER_GPU_TYPE / FRONTIER_VERIFIER_GPU_COUNT):

    placement = "runner"   (default) the WHOLE verification runs on ONE runner with
                           FRONTIER_VERIFIER_GPU_COUNT GPUs: `run()` ships the payload and
                           dataset, the runner's verify.py applies the patch and calls
                           evaluate(tree), result.json + evidence.tar.zst come back.
    placement = "control"  (derived: the task declares a [test_sandboxes.<name>] with
                           access = "command") the shared verifier runs the judge on the CPU
                           control image; evaluate() runs commands on sandbox runners with
                           `run_sandboxes()` below.
    lease_seconds = { train = N, eval = M }
                           the runner lease requested per verification mode (default: both
                           quota.eval_timeout_seconds). The Driver may grant less.

Judge deadline: every command this adapter starts on a runner (verify.py, or a command
sandbox step) gets FRONTIER_VERIFIER_LOCAL_DEADLINE, a `time.monotonic()` value on that runner:
for verify.py the end of the runner's granted lease, for a step the end of its limit.  The judge
must finish and write its result before it (keep a margin of a few minutes); after it the
runner may stop the command and the verification is an unscored infrastructure failure, never
a score.

Errors: VerifierGpuError subclasses Exception, NOT RuntimeError, so a judge's
`except RuntimeError` for its own submission failures never turns a lost runner into a score.
Let it propagate: the shared verifier reports it as an unscored `verifier_gpu_error`.

Command test sandboxes (placement = "control"):

    run_sandboxes([SandboxRun(sandbox, files, steps, outputs=(), max_output_bytes=None)], *, into)
        -> [SandboxResult(steps, outputs, outputs_exceeded, logs)]
        One runner per run of the declared sandbox (its image, resources and per-mode
        lease_seconds from verifier.json), all leased at once, at most max_concurrent per
        sandbox. The upload is exactly `files` (a local directory), extracted as the sandbox's
        workdir; nothing else reaches the runner (the judge bundle only when a trusted
        sandbox's judge puts it in `files`). Steps
        (SandboxStep(argv, seconds, env)) run there in order with the image runtime plus `env`;
        the runner kills a step's process group at `seconds` (KILL 5 s later) and the control
        side stops the session at seconds + 65 s while the heartbeat is alive: either way the
        step is StepResult(None, timed_out=True, ...). The last step may set seconds=None: its
        limit is then the rest of the runner's grant less the per-step grace and the collect
        reserve, fixed when it starts (a trusted harness that budgets itself from
        FRONTIER_VERIFIER_LOCAL_DEADLINE, the end of its step's limit). A non-zero exit is StepResult(code,
        False, ...). A timed-out or failed step ends the run (later steps None). Then the
        declared outputs (files or directories below the workdir) that exist come back in one
        session, at most max_output_bytes (default and maximum one SSH upload chunk) into
        `into/<i>`; over the cap -> outputs None, outputs_exceeded True. Only regular files
        and directories are extracted (modes kept); logs[i] is the last 8 KiB of step i's
        output; unreadable_outputs lists declared files the runner user could not read (on a
        candidate machine the candidate's chmod), not returned. These are facts for the judge to classify; outputs of an untrusted sandbox are
        untrusted data. A step session that stalls (no heartbeat) or ends without its status is
        StepResult(None, False, ..., lost_session=True): the run ends with outputs None (a
        candidate machine's loss is the candidate's scored zero, a trusted machine's is
        infrastructure: the judge classifies). Raised as VerifierGpuError (unscored): runner
        create, grant or termination failure, upload or collect failure, a grant too short for
        the next step. Every runner is terminated
        before run_sandboxes returns or raises; a failing run stops the others.
        SSH sessions per run: upload chunks + one per step + one for the outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pwd
import re
import selectors
import signal
import shlex
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import NamedTuple
from urllib.parse import urlparse


RUNNER_ID = re.compile(r"^gpurunner_[0-7][0-9A-HJKMNP-TV-Z]{25}$")
VERIFICATION_ID = re.compile(r"^verification_[0-7][0-9A-HJKMNP-TV-Z]{25}$")
GPU_TYPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_TOKEN_BYTES = 4096
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_CONNECT_HEADERS = 8192
MAX_RESULT_BYTES = 1024 * 1024
ARCHIVE_OVERHEAD_BYTES = 4 * 1024 * 1024
# A cold immutable model image can spend up to one Driver create window being
# imported and another waiting for its GPU sandbox/readiness probe.  Keep the
# HTTP client alive across both bounded phases; the runner's metered lifetime
# starts only after provisioning succeeds.
GPU_CREATE_TIMEOUT_SECONDS = 2 * 60 * 60 + 120
# The verifier normally emits only one tar stream when it exits.  A tiny
# stderr heartbeat keeps the Driver/Modal SSH channel observable while a long
# GPU evaluator runs, and lets the Pod observe transport activity rather than a
# channel whose upstream close was not propagated.  The heartbeat never enters
# result.json or evidence.tar.zst.
REMOTE_HEARTBEAT_SECONDS = 20
REMOTE_STALL_TIMEOUT_SECONDS = 120
MAX_PRIVATE_STDERR_BYTES = 8192
MAX_UPLOAD_DIAGNOSTIC_BYTES = 2048
NSS_WRAPPER_ROOT = Path("/usr/lib")
PASSWD_DATABASE = Path("/etc/passwd")
GROUP_DATABASE = Path("/etc/group")
DEFAULT_SSH_CAPTURE_BYTES = 8 * 1024 * 1024
MAX_SSH_CAPTURE_BYTES = 64 * 1024 * 1024
MAX_UPLOAD_RETRIES = 2
# /opt/frontier/verifier.json, generated from task.toml; its `gpu` table configures this adapter.
PARAMETERS = Path("/opt/frontier/verifier.json")
PLACEMENTS = frozenset({"runner", "control"})
MAX_LEASE_SECONDS = 7 * 24 * 60 * 60
MAX_GPU_INSTANCES = 64
MAX_RUNNER_GPUS = 64  # the Driver's gpu_count bound per runner
MAX_ARGV_BYTES = 16 * 1024


class VerifierGpuError(Exception):
    """A bounded Verifier GPU lifecycle or transport failure: unscored infrastructure, never a
    submission's failure (deliberately not a RuntimeError, which judges catch as one)."""

    def __init__(self, message: str, *, private_evidence: dict | None = None):
        super().__init__(message)
        self.private_evidence = private_evidence


MAX_PHASE_EVENTS = 16
MAX_PHASE_EVIDENCE_BYTES = 4096
DELETE_RESERVE_SECONDS = 660
EVIDENCE_RESERVE_SECONDS = 30
PHASES = frozenset({"configuration", "input_archive", "provisioning", "upload",
                   "remote_execution", "output_validation", "delete", "cleanup"})


class _PhaseTimeout(VerifierGpuError):
    def __init__(self, boundary: str):
        super().__init__("the Verifier GPU control wait reached its deadline")
        self.boundary = boundary


class _OuterDeadline:
    def __init__(self, end: float, basis: str = "trusted_outer"):
        self.end = end
        self.basis = basis
        self.cleanup_end = end - EVIDENCE_RESERVE_SECONDS
        self.work_end = self.cleanup_end - DELETE_RESERVE_SECONDS

    @classmethod
    def from_environment(cls):
        # Optional future platform contract: expiry of the ALREADY RUNNING
        # verification. Absence uses the conservative returned-grant route.
        raw = os.environ.get("FRONTIER_VERIFIER_DEADLINE_UNIX_MS", "")
        if not raw:
            return None
        if not re.fullmatch(r"[0-9]{1,16}", raw):
            raise VerifierGpuError("the trusted Verifier outer deadline is absent or malformed")
        now = time.monotonic()
        remaining = int(raw) / 1000 - time.time()
        if not math.isfinite(remaining) or remaining > 7 * 24 * 3600:
            raise VerifierGpuError("the trusted Verifier outer deadline is outside its bound")
        # Convert once, so wall-clock changes cannot renew subsequent phases.
        return cls(now + remaining)

    def with_grant(self, end: float):
        basis = "returned_grant_and_outer" if self.basis == "trusted_outer" else "returned_grant_and_declared_cap"
        return _OuterDeadline(min(self.end, end), basis)

    def window(self, seconds: float, *, cleanup: bool = False):
        cutoff = self.cleanup_end if cleanup else self.work_end
        end = min(time.monotonic() + seconds, cutoff)
        boundary = ("cleanup_budget_exhausted" if cleanup else "work_budget_exhausted") \
            if end == cutoff else "phase_timeout"
        return _WaitWindow(end, boundary)


class _WaitWindow:
    def __init__(self, end: float, boundary: str):
        self.end, self.boundary = end, boundary

    def remaining(self, ceiling: float | None = None) -> float:
        left = self.end - time.monotonic()
        if left <= 0:
            raise _PhaseTimeout(self.boundary)
        return left if ceiling is None else min(left, ceiling)

    @contextmanager
    def guard(self):
        # Socket timeouts bound each read, not an entire HTTP body. The main
        # control thread also needs an absolute alarm around blocking phases.
        left = self.remaining()
        if signal.getitimer(signal.ITIMER_REAL)[0]:
            raise VerifierGpuError("the Verifier control timer is already owned")
        previous = signal.getsignal(signal.SIGALRM)

        def expired(signum, frame):
            raise _PhaseTimeout(self.boundary)

        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, left)
        try:
            yield self
            self.remaining()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


class _PhaseEvidence:
    def __init__(self, writer):
        self.writer = writer
        self.started = time.monotonic()
        self.deadline = None
        self.events = []
        self.write_failed = False
        self.remote_failure = None

    def snapshot(self) -> dict:
        value = {"schema": "frontiersmith.verifier-control-phases/v1",
                 "evaluator_progress": "unobserved", "events": self.events,
                 "evidence_write_failed": self.write_failed,
                 "deadline_basis": self.deadline.basis if self.deadline else "unavailable"}
        if self.remote_failure is not None:
            value["remote_failure"] = self.remote_failure
        # All fields are closed enums, bounded times and booleans, never logs,
        # argv, identities, HTTP bodies or exception strings.
        if len(self.events) > MAX_PHASE_EVENTS or len(json.dumps(value).encode()) > MAX_PHASE_EVIDENCE_BYTES:
            raise VerifierGpuError("the Verifier phase evidence exceeded its bound")
        return json.loads(json.dumps(value))

    def flush(self):
        try:
            if self.deadline is None:
                self.writer(self.snapshot())
            else:
                with _WaitWindow(self.deadline.end, "expired").guard():
                    self.writer(self.snapshot())
        except Exception:
            self.write_failed = True

    def outer_exhausted(self, boundary: str):
        if boundary == "phase_timeout":
            return
        status = "bound_expired" if self.deadline and time.monotonic() >= self.deadline.end else boundary
        basis = self.deadline.basis if self.deadline else "unavailable"
        phase = "local_budget" if basis == "declared_request_pending_grant" else "outer_deadline"
        if not any(event["phase"] == phase for event in self.events):
            self.events.append({"phase": phase, "status": status, "bound_basis": basis,
                                "driver_expiry_observed": False})

    @contextmanager
    def phase(self, name: str, seconds: float | None = None, *, cleanup: bool = False):
        if name not in PHASES or any(e["phase"] == name for e in self.events):
            raise VerifierGpuError("the Verifier phase transition is invalid")
        event = {"phase": name, "status": "started"}
        if name == "remote_execution":
            event["transport_activity_seen"] = False
        self.events.append(event)
        self.flush()
        started = time.monotonic()
        try:
            if self.deadline is None:
                yield None
            else:
                window = self.deadline.window(seconds if seconds is not None else 7 * 24 * 3600,
                                              cleanup=cleanup)
                with window.guard():
                    yield window
            event["status"] = "completed"
        except _PhaseTimeout as error:
            event["status"] = "timed_out"
            self.outer_exhausted(error.boundary)
            raise
        except BaseException:
            event["status"] = "failed"
            raise
        finally:
            event["elapsed_ms"] = min(604800000, max(0, int((time.monotonic() - started) * 1000)))
            self.flush()

    def transport_activity(self):
        event = self.events[-1]
        if event["phase"] != "remote_execution" or event["status"] != "started":
            raise VerifierGpuError("the Verifier transport observation is out of phase")
        # Aggregate activity once; repeated heartbeats cannot evict a failure.
        event["transport_activity_seen"] = True


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


_HTTP = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), _RejectRedirects())


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value or value != value.strip() or "\x00" in value:
        raise VerifierGpuError(f"{name} is absent or malformed")
    return value


def _driver_config() -> tuple[str, str, str]:
    """(Driver URL, verification id, bearer) of this verification's scoped Driver API."""
    driver_url = _required_env("FRONTIER_DRIVER_INTERNAL_URL").rstrip("/")
    parsed = urlparse(driver_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname \
            or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise VerifierGpuError("FRONTIER_DRIVER_INTERNAL_URL is not a closed HTTP(S) URL")
    verification_id = _required_env("FRONTIER_VERIFICATION_ID")
    if VERIFICATION_ID.fullmatch(verification_id) is None:
        raise VerifierGpuError("FRONTIER_VERIFICATION_ID is not canonical")
    return driver_url, verification_id, _driver_token()


def _config() -> tuple[str, str, str, str, int]:
    """_driver_config() plus the Verifier GPU image's runner shape (placement "runner")."""
    driver_url, verification_id, token = _driver_config()
    gpu_type = _required_env("FRONTIER_VERIFIER_GPU_TYPE")
    if GPU_TYPE.fullmatch(gpu_type) is None:
        raise VerifierGpuError("FRONTIER_VERIFIER_GPU_TYPE is malformed")
    try:
        gpu_count = int(os.environ.get("FRONTIER_VERIFIER_GPU_COUNT", "1"))
    except ValueError as error:
        raise VerifierGpuError("FRONTIER_VERIFIER_GPU_COUNT is not an integer") from error
    if not 1 <= gpu_count <= 64:
        raise VerifierGpuError("FRONTIER_VERIFIER_GPU_COUNT is outside 1..=64")
    return driver_url, verification_id, token, gpu_type, gpu_count


def _driver_token() -> str:
    """The Driver bearer from the shared verifier's sealed handle, never from os.environ."""
    try:
        from fcs2_verifier import verifier_token
    except ImportError as error:
        raise VerifierGpuError("the shared verifier's Driver credential handle is unavailable") from error
    token = verifier_token()
    if not token:
        raise VerifierGpuError("the Verifier has no Driver credential")
    if len(token.encode("utf-8")) > MAX_TOKEN_BYTES or any(c.isspace() for c in token):
        raise VerifierGpuError("the Verifier Driver credential is malformed")
    return token


def gpu_parameters() -> dict:
    """The task's verifier.json `gpu`: placement and max_gpu_instances, plus lease_seconds
    ("runner") or the command test sandboxes ("control", as task_manifest wrote them)."""
    if "FRONTIER_VERIFIER_GPU_SECONDS" in os.environ:
        raise VerifierGpuError("FRONTIER_VERIFIER_GPU_SECONDS is retired; the lease is "
                               "task.toml [gpu.verifier] lease_seconds")
    try:
        gpu = json.loads(PARAMETERS.read_text(encoding="utf-8"))["gpu"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise VerifierGpuError("the Verifier GPU parameters are missing from verifier.json") from error
    if gpu["placement"] == "control":
        return gpu
    lease = gpu.get("lease_seconds") if isinstance(gpu, dict) else None
    instances = gpu.get("max_gpu_instances") if isinstance(gpu, dict) else None
    if (not isinstance(gpu, dict) or gpu.get("placement") not in PLACEMENTS
            or not isinstance(lease, dict) or set(lease) != {"train", "eval"}
            or any(type(v) is not int or not 1 <= v <= MAX_LEASE_SECONDS for v in lease.values())
            or type(instances) is not int or not 1 <= instances <= MAX_GPU_INSTANCES):
        raise VerifierGpuError("the Verifier GPU parameters in verifier.json are malformed")
    return gpu


def placement() -> str:
    """"runner" (the whole verification on one runner), "control" (the judge runs on the control
    image and calls run_sandboxes)."""
    return gpu_parameters()["placement"]


def lease_seconds(mode: str) -> int:
    if mode not in {"train", "eval"}:
        raise VerifierGpuError("the verification mode is invalid")
    gpu = gpu_parameters()
    if gpu["placement"] == "control":
        raise VerifierGpuError("a control-placement Verifier has a lease per command test "
                               "sandbox, not one [gpu.verifier] lease")
    return gpu["lease_seconds"][mode]


def _request(method: str, driver_url: str, verification_id: str, token: str,
             suffix: str, *, body: dict | None = None, expect: int = 200,
             timeout: float = 660.0, idempotency_scope: str | None = None) -> dict:
    base = f"/internal/v1/verifications/{verification_id}/gpu-runners"
    path = base + suffix
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8") \
        if body is not None else None
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if encoded is not None:
        identity = f"frontiersmith-verifier-gpu-v1\0{verification_id}"
        if idempotency_scope is not None:  # one key per sandbox run; None is the one runner
            if not 1 <= len(idempotency_scope) <= 128 or not idempotency_scope.isascii():
                raise VerifierGpuError("the GPU create idempotency scope is malformed")
            identity += "\0" + idempotency_scope
        stable = hashlib.sha256(identity.encode()).hexdigest()
        headers.update({"Content-Type": "application/json", "Idempotency-Key": stable})
    request = urllib.request.Request(
        driver_url + path, data=encoded, headers=headers, method=method)
    try:
        with _HTTP.open(request, timeout=timeout) as response:
            if response.status != expect or response.geturl() != driver_url + path:
                raise VerifierGpuError(f"{method} Verifier GPU endpoint was not accepted")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        if error.code == 429:
            raise VerifierGpuError("the Verifier GPU quota is exhausted") from error
        raise VerifierGpuError(
            f"{method} Verifier GPU endpoint returned HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise VerifierGpuError("the Verifier could not reach its scoped Driver API") from error
    if len(raw) > MAX_RESPONSE_BYTES:
        raise VerifierGpuError("the Driver GPU response exceeded its bound")
    try:
        value = json.loads(raw or b"{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerifierGpuError("the Driver GPU response was not JSON") from error
    if not isinstance(value, dict):
        raise VerifierGpuError("the Driver GPU response was not an object")
    return value


def _create_runner(driver_url: str, verification_id: str, token: str,
                   gpu_type: str, seconds: int, gpu_count: int = 1, *,
                   window: _WaitWindow | None = None,
                   idempotency_scope: str | None = None,
                   image: str | None = None) -> dict:
    """A ready runner of the Verifier GPU image, or with `image` ("agent_gpu", a command test
    sandbox) of that TaskConfig image; the Driver echoes the image it used."""
    body = {"gpu_type": gpu_type, "gpu_count": gpu_count, "max_duration_seconds": seconds}
    if image is not None:
        body["image"] = image
    runner = _request("POST", driver_url, verification_id, token, "", expect=202, body=body,
                      timeout=window.remaining(GPU_CREATE_TIMEOUT_SECONDS) if window else GPU_CREATE_TIMEOUT_SECONDS,
                      idempotency_scope=idempotency_scope)
    runner_id = str(runner.get("runner_id") or "")
    if (RUNNER_ID.fullmatch(runner_id) is None
            or runner.get("state") != "ready"
            or runner.get("gpu_count") != gpu_count
            or (image is not None and runner.get("image") != image)):
        raise VerifierGpuError("the Driver did not return a ready canonical GPU runner")
    expected = (f"{driver_url}/internal/v1/verifications/{verification_id}/"
                f"gpu-runners/{runner_id}/connect")
    if runner.get("proxy_endpoint") != expected:
        raise VerifierGpuError("the Verifier GPU proxy endpoint did not match its owner")
    gateway = runner.get("ssh_gateway")
    if (not isinstance(gateway, dict)
            or gateway.get("schema_version") != "frontier.ssh-gateway/v1"
            or gateway.get("username") != "frontier"
            or not str(gateway.get("server_host_public_key") or "").startswith("ssh-")
            or not str(gateway.get("server_host_key_sha256") or "").startswith("SHA256:")
            or gateway.get("authentication") != "connect_bearer"
            or gateway.get("max_channels") != 1):
        raise VerifierGpuError("the Driver returned an invalid GPU SSH identity")
    return runner


def _terminate_runner(driver_url: str, verification_id: str, token: str,
                      runner_id: str, *, window: _WaitWindow | None = None) -> bool:
    _close_broker(runner_id)
    try:
        result = _request("DELETE", driver_url, verification_id, token,
                          f"/{runner_id}", timeout=window.remaining(660) if window else 660)
    except _PhaseTimeout:
        raise
    except VerifierGpuError:
        return False
    return result.get("runner_id") == runner_id and result.get("state") == "terminated"


def _runner_deadline(runner: dict, requested_seconds: int, *, anchor: float | None = None) -> float:
    """Bound local waits by the returned grant, not the requested quota.

    Called after retaining the owned identity. The Driver caps this grant by
    verification timeout minus floor(elapsed seconds). Anchor BEFORE create and
    subtract one second to cover that truncation; provisioning spends this bound.
    Omitting anchor retains the established response-receipt clock for callers
    that have not opted into phase evidence. Neither proves provider billing.
    """
    granted = runner.get("granted_seconds")
    if type(granted) is not int or not 1 <= granted <= requested_seconds:
        raise VerifierGpuError("the Driver returned an invalid GPU granted duration")
    if anchor is None:
        return time.monotonic() + granted
    return anchor + granted - 1


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise VerifierGpuError("the Verifier GPU lease is exhausted")
    return remaining


def _connect_socket(driver_url: str, verification_id: str, token: str,
                    runner_id: str) -> socket.socket:
    parsed = urlparse(driver_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        raise VerifierGpuError("the Driver URL has no host")
    connection = socket.create_connection((host, port), timeout=30)
    if parsed.scheme == "https":
        connection = ssl.create_default_context().wrap_socket(
            connection, server_hostname=host)
    path = (f"/internal/v1/verifications/{verification_id}/gpu-runners/"
            f"{runner_id}/connect")
    try:
        authority = f"[{host}]" if ":" in host else host
        connection.sendall(
            f"CONNECT {path} HTTP/1.1\r\n"
            f"Host: {authority}:{port}\r\n"
            f"Authorization: Bearer {token}\r\n\r\n".encode("ascii"))
        header = bytearray()
        while not header.endswith(b"\r\n\r\n"):
            if len(header) >= MAX_CONNECT_HEADERS:
                raise VerifierGpuError("the Driver CONNECT headers exceeded their bound")
            chunk = connection.recv(1)
            if not chunk:
                raise VerifierGpuError("the Driver closed CONNECT before upgrading")
            header.extend(chunk)
        if not bytes(header).split(b"\r\n", 1)[0].startswith(b"HTTP/1.1 200 "):
            raise VerifierGpuError("the Driver refused the Verifier GPU CONNECT")
        connection.settimeout(None)
        return connection
    except BaseException:
        connection.close()
        raise


def _relay(first: socket.socket, second: socket.socket) -> None:
    """Copy bytes both ways until both directions are closed, then close both sockets."""
    selector = selectors.DefaultSelector()
    selector.register(first, selectors.EVENT_READ, second)
    selector.register(second, selectors.EVENT_READ, first)
    open_directions = 2
    try:
        while open_directions:
            for key, _ in selector.select():
                source, target = key.fileobj, key.data
                try:
                    chunk = source.recv(65536)
                    while chunk and isinstance(source, ssl.SSLSocket) and source.pending():
                        chunk += source.recv(source.pending())
                    if chunk:
                        target.sendall(chunk)
                        continue
                except OSError:
                    pass
                selector.unregister(source)
                open_directions -= 1
                try:
                    target.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
    finally:
        selector.close()
        first.close()
        second.close()


class _ConnectBroker:
    """The Driver CONNECT tunnel for one runner, opened by THIS process for ssh's ProxyCommand.

    ssh runs the ProxyCommand (`fcs2_verifier_gpu.py _connect <runner> <socket>`) as its own
    child, and must not inherit the bearer.  The child connects to this Unix socket (in the
    0700 per-runner scratch, same uid only), this process opens the CONNECT with the bearer
    and answers one status line ("OK" or "ERR <bounded adapter message>"), then relays.  The
    broker serves only its own runner and closes when that runner is terminated."""

    def __init__(self, runner_id: str, path: Path, config: tuple):
        self.runner_id, self.path, self._config = runner_id, path, config
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._listener.bind(str(path))
            path.chmod(0o600)
            self._listener.listen(16)
        except BaseException:
            self._listener.close()
            raise
        threading.Thread(target=self._serve, name=f"gpu-connect-{runner_id}", daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                local, _ = self._listener.accept()
            except OSError:
                return  # closed
            threading.Thread(target=self._tunnel, args=(local,), daemon=True).start()

    def _tunnel(self, local: socket.socket) -> None:
        try:
            uid = struct.unpack("3i", local.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                                       struct.calcsize("3i")))[1]
            if uid != os.getuid():
                raise VerifierGpuError("the ProxyCommand peer is not this Verifier")
            driver_url, verification_id, token, _gpu_type, _gpu_count = self._config
            try:
                upstream = _connect_socket(driver_url, verification_id, token, self.runner_id)
            except OSError as error:
                raise VerifierGpuError("the Verifier could not reach the Driver CONNECT endpoint") from error
        except VerifierGpuError as error:
            try:
                local.sendall(f"ERR {str(error)[:400]}\n".encode("utf-8", errors="replace"))
            except OSError:
                pass
            local.close()
            return
        except BaseException:
            local.close()
            raise
        try:
            local.sendall(b"OK\n")
        except OSError:
            local.close()
            upstream.close()
            return
        _relay(local, upstream)

    def close(self) -> None:
        self._listener.close()
        try:
            self.path.unlink()
        except OSError:
            pass


_BROKERS: dict[str, _ConnectBroker] = {}
_BROKERS_LOCK = threading.Lock()


def _broker(runner_id: str, scratch: Path) -> _ConnectBroker:
    with _BROKERS_LOCK:
        broker = _BROKERS.get(runner_id)
        if broker is None:
            broker = _BROKERS[runner_id] = _ConnectBroker(runner_id, scratch / "connect.sock", _config())
        return broker


def _close_broker(runner_id: str) -> None:
    with _BROKERS_LOCK:
        broker = _BROKERS.pop(runner_id, None)
    if broker is not None:
        broker.close()


def _ssh_argv(runner: dict, scratch: Path) -> list[str]:
    gateway = runner["ssh_gateway"]
    known_hosts = scratch / "known_hosts"
    known_hosts.write_text(
        f"gpu-runner {str(gateway['server_host_public_key']).strip()}\n",
        encoding="utf-8")
    known_hosts.chmod(0o600)
    broker = _broker(str(runner["runner_id"]), scratch)
    proxy = (f"{sys.executable} /opt/frontier/fcs2_verifier_gpu.py _connect "
             f"{runner['runner_id']} {shlex.quote(str(broker.path))}")
    return [
        "/usr/bin/ssh", "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "HostKeyAlias=gpu-runner", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=30", "-o", f"ProxyCommand={proxy}",
        f"{gateway['username']}@gpu-runner",
    ]


def _ssh_environment(scratch: Path) -> dict[str, str]:
    """Give OpenSSH a passwd identity for the Pod's platform-assigned UID."""
    uid = os.getuid()
    gid = os.getgid()
    environment = dict(os.environ)
    # The bearer is sealed by the shared verifier; drop it anyway should a caller have set it.
    for name in ("LD_PRELOAD", "NSS_WRAPPER_PASSWD", "NSS_WRAPPER_GROUP", "FRONTIER_VERIFIER_TOKEN"):
        environment.pop(name, None)
    try:
        pwd.getpwuid(uid)
    except KeyError:
        libraries = tuple(NSS_WRAPPER_ROOT.glob("*/libnss_wrapper.so"))
        if len(libraries) != 1 or not libraries[0].is_file():
            raise VerifierGpuError(
                "the Verifier GPU SSH identity adapter is unavailable")
        try:
            passwd_body = PASSWD_DATABASE.read_text(encoding="utf-8")
            group_body = GROUP_DATABASE.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise VerifierGpuError(
                "the Verifier GPU SSH identity database is unreadable") from error
        passwd = scratch / "passwd"
        passwd.write_text(
            passwd_body + ("" if passwd_body.endswith("\n") else "\n")
            + f"frontier-runtime:x:{uid}:{gid}:Frontier verifier:{scratch}:"
              "/usr/sbin/nologin\n",
            encoding="utf-8")
        group = scratch / "group"
        group.write_text(
            group_body + ("" if group_body.endswith("\n") else "\n")
            + f"frontier-runtime:x:{gid}:\n",
            encoding="utf-8")
        passwd.chmod(0o600)
        group.chmod(0o600)
        environment.update({
            "HOME": str(scratch),
            "LD_PRELOAD": str(libraries[0]),
            "NSS_WRAPPER_PASSWD": str(passwd),
            "NSS_WRAPPER_GROUP": str(group),
        })
        return environment
    environment["HOME"] = str(scratch)
    return environment


def _input_archive(path: Path, payload: Path, dataset: Path) -> None:
    for source, label in ((payload, "payload"), (dataset, "dataset")):
        if not source.is_file() or source.is_symlink():
            raise VerifierGpuError(f"the platform {label} input is not a regular file")
    with tarfile.open(path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        archive.add(payload, arcname="input/payload.tar", recursive=False)
        archive.add(dataset, arcname="input/dataset", recursive=False)


def _upload_chunk_bytes(environment: dict[str, str]) -> int:
    try:
        capture = int(environment.get(
            "FRONTIER_MAX_SSH_BYTES_PER_DIRECTION", str(DEFAULT_SSH_CAPTURE_BYTES)))
    except ValueError as error:
        raise VerifierGpuError("the Verifier GPU SSH capture cap is invalid") from error
    if not 1 <= capture <= MAX_SSH_CAPTURE_BYTES:
        raise VerifierGpuError("the Verifier GPU SSH capture cap is outside 1..64 MiB")
    # The gateway records plaintext stdin; exec metadata has its own 24 KiB
    # allowance. Leave one eighth of each direction unused for protocol margin.
    return max(1, capture * 7 // 8)


def _upload_command(index: int, count: int, size: int, digest: str,
                    total_size: int, total_digest: str,
                    destination: str = "/workspace/frontier-verifier") -> str:
    staging = "/workspace/frontier-verifier-upload"
    part = f"{staging}/part-{index:08d}"
    command = "set -eu; "
    if index == 0:
        command += f"rm -rf {staging} {destination}; mkdir -p {staging}; "
    command += (
        f"cat > {part}.tmp; "
        f"test \"$(wc -c < {part}.tmp)\" -eq {size}; "
        f"printf '%s  %s\\n' {digest} {part}.tmp | sha256sum -c --status; "
        f"mv {part}.tmp {part}; "
    )
    if index == count - 1:
        # Nothing is extracted until every byte of the reassembled archive is
        # verified. Retain parts so a lost SSH exit status can be retried safely.
        command += (
            f"cat {staging}/part-* > {staging}/inputs.tar; "
            f"test \"$(wc -c < {staging}/inputs.tar)\" -eq {total_size}; "
            f"printf '%s  %s\\n' {total_digest} {staging}/inputs.tar "
            "| sha256sum -c --status; "
            f"rm -rf {destination}; mkdir -p {destination}; "
            f"tar -C {destination} -xf {staging}/inputs.tar; "
        )
    return command


def _write_upload_diagnostic(*, chunk: int, attempt: int, exit_code: int | None,
                             timed_out: bool, timeout_seconds: float | None,
                             stderr: bytes | str | None,
                             event: str = "attempt_failed") -> None:
    # Emit only fixed stderr signatures. Blacklist redaction cannot guarantee
    # that arbitrary SSH/proxy/remote output contains no credential values.
    tail = (stderr or b"")[-MAX_PRIVATE_STDERR_BYTES:]
    truncated = len(stderr or b"") > MAX_PRIVATE_STDERR_BYTES
    if isinstance(tail, str):
        tail = tail.encode("utf-8", errors="replace")
        truncated = truncated or len(tail) > MAX_PRIVATE_STDERR_BYTES
        tail = tail[-MAX_PRIVATE_STDERR_BYTES:]
    tail = tail.lower()
    signatures = (
        (b"driver refused the verifier gpu connect", "driver_connect_refused"),
        (b"driver closed connect before upgrading", "driver_connect_closed"),
        (b"driver connect headers exceeded", "driver_connect_headers_too_large"),
        (b"connection refused", "connection_refused"),
        (b"connection timed out", "connection_timeout"),
        (b"connection closed", "connection_closed"),
        (b"connection reset", "connection_reset"),
        (b"no route to host", "no_route_to_host"),
        (b"network is unreachable", "network_unreachable"),
        (b"could not resolve hostname", "hostname_resolution_failed"),
        (b"host key verification failed", "host_key_verification_failed"),
        (b"permission denied", "permission_denied"),
        (b"kex_exchange_identification", "ssh_key_exchange_error"),
        (b"no matching host key type", "ssh_host_key_negotiation_failed"),
        (b"broken pipe", "broken_pipe"),
        (b"no space left on device", "no_space_left"),
        (b"read-only file system", "read_only_filesystem"),
        (b"tar:", "tar_error"),
        (b"sha256sum:", "checksum_error"),
    )
    record = {
        "event": event, "chunk": chunk, "attempt": attempt,
        "exit_code": exit_code, "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "stderr_markers": [label for needle, label in signatures if needle in tail][:8],
        "stderr_present": bool(stderr),
        "stderr_text_redacted": True,
        "stderr_tail_truncated": truncated,
    }
    prefix = "frontier-verifier-gpu private upload diagnostic: "
    line = prefix + json.dumps(record, sort_keys=True)
    if len(line.encode("utf-8")) + 1 > MAX_UPLOAD_DIAGNOSTIC_BYTES:
        line = prefix + '{"event":"diagnostic_size_limit"}'
    try:
        print(line, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass  # A private logging failure must not replace the upload outcome.


def _upload_inputs(ssh: list[str], inputs: Path, seconds: float,
                   environment: dict[str, str], *,
                   runner_deadline: float | None = None,
                   window: _WaitWindow | None = None,
                   destination: str = "/workspace/frontier-verifier") -> None:
    """Upload audited, verified chunks under one deadline and retry allowance."""
    deadline = time.monotonic() + min(seconds, 120)
    if window is not None:
        deadline = min(deadline, window.end)
    if runner_deadline is not None:
        deadline = min(deadline, runner_deadline)
    chunk_bytes = _upload_chunk_bytes(environment)
    total_size = inputs.stat().st_size
    if total_size <= 0:
        raise VerifierGpuError("the Verifier GPU input archive is empty")
    total_digest = hashlib.sha256()
    with inputs.open("rb") as source:
        while body := source.read(1024 * 1024):
            total_digest.update(body)
    count = (total_size + chunk_bytes - 1) // chunk_bytes
    retries = 0
    attempts = 0
    with inputs.open("rb") as source:
        for index in range(count):
            size = min(chunk_bytes, total_size - index * chunk_bytes)
            digest = hashlib.sha256()
            # A regular temporary file bounds both memory and stdin, unlike
            # passing the remaining full archive to OpenSSH for each chunk.
            with tempfile.TemporaryFile(dir=inputs.parent) as stream:
                left = size
                while left:
                    body = source.read(min(left, 1024 * 1024))
                    if not body:
                        raise VerifierGpuError("the Verifier GPU input archive was truncated")
                    stream.write(body)
                    digest.update(body)
                    left -= len(body)
                command = _upload_command(
                    index, count, size, digest.hexdigest(), total_size,
                    total_digest.hexdigest(), destination)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        _write_upload_diagnostic(
                            chunk=index, attempt=attempts, exit_code=None,
                            timed_out=False, timeout_seconds=None, stderr=None,
                            event="deadline_exceeded")
                        raise VerifierGpuError("the Verifier GPU input upload exceeded its deadline")
                    stream.seek(0)
                    attempts += 1
                    timeout_stderr = None
                    try:
                        if window is None:
                            # Preserve the established callable/subprocess.run seam.
                            uploaded = subprocess.run(
                                ssh + [command], stdin=stream, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, timeout=min(45, remaining),
                                check=False, env=environment)
                        else:
                            uploaded = _upload_process(
                                ssh + [command], stream, environment,
                                min(45, remaining), deadline)
                    except subprocess.TimeoutExpired as error:
                        uploaded = None
                        timeout_stderr = error.stderr
                    if uploaded is not None and uploaded.returncode == 0:
                        break
                    _write_upload_diagnostic(
                        chunk=index, attempt=attempts,
                        exit_code=None if uploaded is None else uploaded.returncode,
                        timed_out=uploaded is None, timeout_seconds=min(45, remaining),
                        stderr=timeout_stderr if uploaded is None else uploaded.stderr)
                    stderr = b"" if uploaded is None else uploaded.stderr or b""
                    if b"Host key verification failed" in stderr:
                        raise VerifierGpuError("the Verifier GPU host-key check failed")
                    if b"Permission denied" in stderr:
                        raise VerifierGpuError("the Verifier GPU SSH authentication failed")
                    if uploaded is not None and uploaded.returncode != 255:
                        raise VerifierGpuError("the Verifier GPU input upload failed integrity or extraction checks")
                    if retries >= MAX_UPLOAD_RETRIES:
                        raise VerifierGpuError("the Verifier GPU input upload exhausted its transport retries")
                    retries += 1
                    time.sleep(min(2, max(0, deadline - time.monotonic())))


def _copy_outputs(archive_path: Path, result_path: Path, evidence_path: Path,
                  max_evidence_bytes: int) -> None:
    limit = MAX_RESULT_BYTES + max_evidence_bytes + ARCHIVE_OVERHEAD_BYTES
    if not archive_path.is_file() or not 0 < archive_path.stat().st_size <= limit:
        raise VerifierGpuError("the Verifier GPU output archive exceeded its bound")
    expected = {"result.json": (result_path, MAX_RESULT_BYTES),
                "evidence.tar.zst": (evidence_path, max_evidence_bytes)}
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        if {member.name for member in members} != set(expected) or len(members) != 2:
            raise VerifierGpuError("the Verifier GPU returned an unexpected output archive")
        for member in members:
            destination, maximum = expected[member.name]
            if not member.isfile() or not 0 < member.size <= maximum:
                raise VerifierGpuError(
                    f"the Verifier GPU {member.name} is not a bounded regular file")
            source = archive.extractfile(member)
            if source is None:
                raise VerifierGpuError(f"the Verifier GPU {member.name} is unreadable")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.gpu.tmp")
            with temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.replace(destination)


def _read_remote_failure(archive_path: Path, maximum: int) -> dict | None:
    """Read only a bounded failed-result schema, even if its archive is absent."""
    if not archive_path.is_file() or not 0 < archive_path.stat().st_size <= maximum:
        return None
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if (len(names) != len(set(names)) or "result.json" not in names
                    or not set(names) <= {"result.json", "evidence.tar.zst"}):
                return None
            member = archive.getmember("result.json")
            if not member.isfile() or not 0 < member.size <= MAX_RESULT_BYTES:
                return None
            source = archive.extractfile(member)
            if source is None:
                return None
            result = json.loads(source.read(MAX_RESULT_BYTES + 1))
            if (not isinstance(result, dict) or result.get("status") != "failed"
                    or "raw_score" in result or not isinstance(result.get("failure"), dict)):
                return None
            code = result["failure"].get("code")
            allowed = {"trusted_evaluation_failure", "invalid_evaluator_result", "evaluator_error",
                       "verifier_configuration_error", "verifier_internal_error", "invalid_execution_evidence"}
            if not isinstance(code, str) or code not in allowed:
                return None
            return {"code": code, "scientific_result": False,
                    "evidence_available": "evidence.tar.zst" in names}
    except (OSError, ValueError, tarfile.TarError):
        return None


_REMOTE_TRANSPORT_SUPERVISOR = r'''
"""Inactive Linux-only transport repair prototype; no provider API access."""
import os
import selectors
import signal
import subprocess
import sys
import time

GRACE_SECONDS = 2.0
REAP_SECONDS = 2.0
interrupted = 0


def cancel(signum, frame):
    global interrupted
    interrupted = signum


def write_private(data):
    while data:
        count = os.write(2, data)
        if count <= 0:
            raise OSError("private transport closed")
        data = data[count:]


def signal_group(pid, signum):
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def child_state(pid):
    return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)


def main():
    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGPIPE):
        signal.signal(signum, cancel)
    process = None
    completed = False
    custody_lost = False
    status = 125
    selector = selectors.DefaultSelector()
    try:
        # This single-threaded supervisor is the sole child waiter. Reset
        # inherited auto-reaping before spawn; WNOWAIT needs a retained child.
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        os.set_blocking(2, False)
        write_private(b"frontier-verifier-gpu-heartbeat\n")
        if interrupted:
            return 128 + interrupted
        process = subprocess.Popen(
            ["/bin/sh", "-c", sys.argv[1]], stdin=subprocess.DEVNULL,
            stderr=subprocess.PIPE, start_new_session=True)
        os.set_blocking(process.stderr.fileno(), False)
        selector.register(process.stderr, selectors.EVENT_READ)
        interval = float(sys.argv[2])
        next_heartbeat = time.monotonic() + interval
        while not interrupted:
            now = time.monotonic()
            if now >= next_heartbeat:
                write_private(b"frontier-verifier-gpu-heartbeat\n")
                next_heartbeat = now + interval
            for key, _ in selector.select(min(0.1, max(0, next_heartbeat - now))):
                data = os.read(key.fileobj.fileno(), 8192)
                if data:
                    write_private(data)
                else:
                    selector.unregister(key.fileobj)
            # Keep the group leader unreaped until the final group signal;
            # its PID cannot be reused for an unrelated group during cleanup.
            try:
                state = child_state(process.pid)
            except OSError:
                custody_lost = True
                raise
            if state is not None:
                for _ in range(16):
                    try:
                        data = os.read(process.stderr.fileno(), 8192)
                    except BlockingIOError:
                        break
                    if not data:
                        break
                    write_private(data)
                completed = True
                break
    except (OSError, ValueError):
        status = 125
    finally:
        selector.close()
        if process is not None:
            cleanup_errors = []
            for signum, stage in ((signal.SIGTERM, "term"), (signal.SIGKILL, "kill")):
                if custody_lost:
                    cleanup_errors.append("leader-custody-lost")
                    break
                if signum == signal.SIGKILL:
                    try:
                        time.sleep(GRACE_SECONDS)
                    except OSError:
                        cleanup_errors.append("grace")
                # Never signal a numeric group after ownership became unknown.
                try:
                    child_state(process.pid)
                except OSError:
                    custody_lost = True
                    cleanup_errors.append("leader-custody-lost")
                    break
                try:
                    signal_group(process.pid, signum)
                except OSError:
                    cleanup_errors.append(stage)
            try:
                # Waiting is safe even after lost custody; it sends no signal.
                result = process.wait(timeout=REAP_SECONDS)
                if completed and not cleanup_errors:
                    status = result if result >= 0 else 128 - result
            except (OSError, subprocess.TimeoutExpired):
                cleanup_errors.append("reap")
            if cleanup_errors:
                status = 125
                try:
                    write_private(("frontier-verifier-gpu cleanup incomplete: "
                                   + ",".join(cleanup_errors) + "\n").encode())
                except OSError:
                    pass
            process.stderr.close()
    return 128 + interrupted if interrupted else status


if __name__ == "__main__":
    sys.exit(main())
'''


def _remote_runtime(remaining_seconds: float) -> str:
    """Restore only immutable runtime state after sshd clears its env, then export the judge
    deadline: FRONTIER_VERIFIER_LOCAL_DEADLINE, this host's monotonic clock at lease end."""
    if not (isinstance(remaining_seconds, (int, float)) and math.isfinite(remaining_seconds)
            and 0 < remaining_seconds <= MAX_LEASE_SECONDS):
        raise VerifierGpuError("the remaining Verifier GPU lease is invalid")
    return (
        "if [ -d /opt/frontier/models ]; then "
        "export FRONTIER_MODEL_ROOT=/opt/frontier/models "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi; "
        "if [ -d /opt/frontier/models/torch ]; then "
        "export TORCH_HOME=/opt/frontier/models/torch; fi; "
        # OpenSSH starts a forced command with its own restricted PATH.  A
        # Conda-derived GPU image installs dependencies with /opt/conda/bin/pip3
        # during the build, but a bare python3 over SSH can resolve to
        # /usr/bin/python3 and see none of them.  New images freeze their build
        # interpreter at runtime-python; the Conda branch keeps already-published
        # immutable images executable while they are rolled forward.
        "if [ -x /opt/frontier/runtime-python ]; then "
        "verifier_python=/opt/frontier/runtime-python; "
        "elif [ -x /opt/conda/bin/python3 ]; then "
        "verifier_python=/opt/conda/bin/python3; "
        "else verifier_python=/usr/bin/python3; fi; "
        # Ask the selected interpreter for its installed tools directory. This
        # supports the venv-preserving launcher and existing Conda symlinks;
        # readlink would either find the launcher or escape a virtualenv.
        "verifier_python_dir=$(\"$verifier_python\" -c "
        "'import sysconfig; print(sysconfig.get_path(\"scripts\"))'); "
        "test -d \"$verifier_python_dir\"; "
        "export PATH=\"$verifier_python_dir:$PATH\"; "
        "if [ -x /usr/local/cuda/bin/nvcc ]; then "
        "export CUDA_HOME=/usr/local/cuda CUDACXX=/usr/local/cuda/bin/nvcc; "
        "export PATH=\"/usr/local/cuda/bin:$PATH\"; "
        "fi; "
        # Runtime-only CUDA images have shared libraries but no nvcc. The
        # sshd bootstrap clears image ENV, so restore the existing library
        # directory independently; never add an empty/current-directory entry.
        "if [ -d /usr/local/cuda/lib64 ]; then "
        "export LD_LIBRARY_PATH=\"/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"; "
        "fi; "
        "export MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=1; "
        "if [ -x /usr/bin/gcc-11 ]; then export CC=/usr/bin/gcc-11; "
        "elif [ -x /usr/bin/gcc ]; then export CC=/usr/bin/gcc; fi; "
        "if [ -x /usr/bin/g++-11 ]; then export CXX=/usr/bin/g++-11; "
        "elif [ -x /usr/bin/g++ ]; then export CXX=/usr/bin/g++; fi; "
        # Host monotonic clocks are not shared: convert the remaining lease on the runner.
        "FRONTIER_VERIFIER_LOCAL_DEADLINE=$(\"$verifier_python\" -c "
        "'import sys, time; print(repr(time.monotonic() + float(sys.argv[1])))' "
        f"{remaining_seconds:.6f}) || exit 125; export FRONTIER_VERIFIER_LOCAL_DEADLINE; "
    )


def _supervised(command: str) -> str:
    # The heartbeat/cleanup supervisor starts before the evaluator's selector.
    # Python/Conda images may not contain /usr/bin/python3 at all; use the same
    # immutable, qualified runtime choices here without inheriting author PATH.
    transport_python = (
        '"$(if [ -x /opt/frontier/runtime-python ]; then '
        'printf %s /opt/frontier/runtime-python; '
        'elif [ -x /opt/conda/bin/python3 ]; then '
        'printf %s /opt/conda/bin/python3; '
        'else printf %s /usr/bin/python3; fi)"'
    )
    return ("exec " + transport_python + " -c " + shlex.quote(_REMOTE_TRANSPORT_SUPERVISOR)
            + " " + shlex.quote(command) + " " + str(REMOTE_HEARTBEAT_SECONDS))


def _remote_verifier_command(mode: str, gpu_count: int = 1, *, remaining_seconds: float) -> str:
    """placement = "runner": the shared verify.py on the runner, result + evidence back."""
    if mode not in {"train", "eval"}:
        raise VerifierGpuError("the verification mode is invalid")
    if type(gpu_count) is not int or not 1 <= gpu_count <= MAX_RUNNER_GPUS:
        raise VerifierGpuError("the Verifier GPU count is invalid")
    command = (
        "mkdir -p /workspace/frontier-verifier/output; status=0; "
        + _remote_runtime(remaining_seconds) +
        f"FRONTIER_PAYLOAD_PATH=/workspace/frontier-verifier/input/payload.tar "
        f"FRONTIER_DATASET_PATH=/workspace/frontier-verifier/input/dataset "
        f"FRONTIER_VERIFICATION_MODE={mode} "
        f"FRONTIER_VERIFIER_GPU_COUNT={gpu_count} "
        "FRONTIER_RESULT_PATH=/workspace/frontier-verifier/output/result.json "
        "FRONTIER_EVIDENCE_PATH=/workspace/frontier-verifier/output/evidence.tar.zst "
        "FRONTIER_VERIFIER_SCRATCH=/workspace TMPDIR=/workspace "
        "\"$verifier_python\" /opt/frontier/verify.py "
        ">/workspace/frontier-verifier/verifier.log 2>&1 || status=$?; "
        "if [ \"$status\" -ne 0 ]; then "
        f"tail -c {MAX_PRIVATE_STDERR_BYTES} "
        "/workspace/frontier-verifier/verifier.log >&2; fi; "
        "tar -C /workspace/frontier-verifier/output -cf - "
        "result.json evidence.tar.zst || { archive_status=$?; "
        "[ \"$status\" -ne 0 ] || status=$archive_status; }; exit $status"
    )
    return _supervised(command)


def _stop_process(process: subprocess.Popen, deadline: float | None = None) -> bool:
    if process.poll() is not None:
        return True
    end = deadline if deadline is not None else time.monotonic() + 20
    for action in (process.terminate, process.kill):
        try:
            action()
        except ProcessLookupError:
            pass
        left = min(10, max(0, end - time.monotonic()))
        if left > 0:
            try:
                process.wait(timeout=left)
                return True
            except subprocess.TimeoutExpired:
                pass
    return process.poll() is not None


def _upload_process(argv, stream, environment, timeout, deadline):
    process = subprocess.Popen(argv, stdin=stream, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, env=environment)
    try:
        _, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(argv, process.returncode, stderr=stderr)
    finally:
        # subprocess.run's TimeoutExpired path can reap without a time bound.
        # Both escalation waits must instead fit the existing upload window.
        stopped = _stop_process(process, deadline)
        if process.stderr is not None:
            process.stderr.close()
        if not stopped and not isinstance(sys.exc_info()[1], _PhaseTimeout):
            raise VerifierGpuError("the Verifier upload process cleanup was not confirmed")


def _run_remote_verifier(ssh: list[str], remote: str, outputs: Path,
                         seconds: float, environment: dict[str, str], *,
                         runner_deadline: float | None = None,
                         window: _WaitWindow | None = None,
                         evidence: _PhaseEvidence | None = None) -> tuple[int, bool]:
    """Run the remote verifier and bound a lost SSH close independently.

    ``stalled`` means the child was stopped after its heartbeat disappeared.
    A complete archive alone does not confirm successful remote completion.
    """
    now = time.monotonic()
    if runner_deadline is not None and now >= runner_deadline:
        raise VerifierGpuError("the Verifier GPU grant expired before evaluator execution")
    deadline = now + seconds + 60
    if window is not None:
        deadline = min(deadline, window.end)
    if runner_deadline is not None:
        # Keep only the existing bounded local transport-close grace. This
        # does not allocate GPU lifetime or renew it after input upload.
        deadline = min(deadline, runner_deadline + 60)
    last_activity = time.monotonic()
    stalled = False
    with outputs.open("wb") as stream:
        process = subprocess.Popen(
            ssh + [remote], stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.PIPE, env=environment)
        if process.stderr is None:
            _stop_process(process, deadline)
            raise VerifierGpuError("the Verifier GPU SSH stderr pipe is unavailable")
        stderr_tail = bytearray()
        selector = None
        try:
            os.set_blocking(process.stderr.fileno(), False)
            selector = selectors.DefaultSelector()
            selector.register(process.stderr, selectors.EVENT_READ)
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    _stop_process(process, deadline)
                    if window is not None and now >= window.end:
                        raise _PhaseTimeout(window.boundary)
                    raise VerifierGpuError("the remote transport reached its GPU grant deadline" if window
                                           else "the evaluator exceeded its Verifier GPU deadline")
                remaining_stall = REMOTE_STALL_TIMEOUT_SECONDS - (now - last_activity)
                if remaining_stall <= 0:
                    stalled = True
                    _stop_process(process, deadline)
                    break
                wait = min(5.0, deadline - now, remaining_stall)
                for key, _ in selector.select(wait):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if chunk:
                        last_activity = time.monotonic()
                        if evidence is not None:
                            evidence.transport_activity()
                        stderr_tail.extend(chunk)
                        del stderr_tail[:-MAX_PRIVATE_STDERR_BYTES]
                    else:
                        selector.unregister(key.fileobj)
        finally:
            if selector is not None:
                selector.close()
            stopped = _stop_process(process, deadline)
            if evidence is not None:
                evidence.events[-1]["local_process_cleanup"] = "confirmed" if stopped else "unconfirmed"
            # Fast failures can exit before select sees their final diagnostic.
            # Bound this drain even if a descendant retains the pipe writer.
            for _ in range(16):
                try:
                    chunk = os.read(process.stderr.fileno(), 65536)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                stderr_tail.extend(chunk)
                del stderr_tail[:-MAX_PRIVATE_STDERR_BYTES]
            process.stderr.close()
            if evidence is None and stderr_tail and (process.returncode != 0 or stalled):
                diagnostic = stderr_tail.decode("utf-8", errors="replace")
                diagnostic = diagnostic.encode("utf-8")[-MAX_PRIVATE_STDERR_BYTES:].decode(
                    "utf-8", errors="ignore")
                # Compatibility callers retain the existing private stderr seam.
                # The phase-evidence wrapper never copies these bytes into logs.
                print(f"frontier-verifier-gpu private stderr tail:\n{diagnostic}",
                      file=sys.stderr, flush=True)
    return process.returncode, stalled


def run(*, payload: Path, dataset: Path, mode: str, result_path: Path,
        evidence_path: Path, record_evidence=None) -> None:
    # The updated wrapper opts into shared deadlines and bounded checkpoints.
    # Existing callers retain their grant, diagnostic and subprocess contracts.
    phase_control = record_evidence is not None
    evidence = _PhaseEvidence(record_evidence if phase_control else lambda _: None)
    runner = None
    scratch = None
    failure = None
    terminated = True
    try:
        with evidence.phase("configuration"):
            if mode not in {"train", "eval"}:
                raise VerifierGpuError("the verification mode is invalid")
            driver_url, verification_id, token, gpu_type, gpu_count = _config()
            seconds = lease_seconds(mode)
            if phase_control:
                evidence.deadline = _OuterDeadline.from_environment()
                if evidence.deadline is None:
                    # This local cap is not the already-running Driver deadline.
                    evidence.deadline = _OuterDeadline(time.monotonic() + seconds, "declared_request_pending_grant")
            try:
                max_evidence = int(_required_env("FRONTIER_MAX_EVIDENCE_BYTES"))
            except ValueError as error:
                raise VerifierGpuError("FRONTIER_MAX_EVIDENCE_BYTES is invalid") from error
            if max_evidence <= 0:
                raise VerifierGpuError("FRONTIER_MAX_EVIDENCE_BYTES must be positive")
            if phase_control and max_evidence < 8192:
                raise VerifierGpuError("the Verifier evidence allowance is too small")
            _upload_chunk_bytes(os.environ)
            if evidence.write_failed:
                raise VerifierGpuError("the initial Verifier phase evidence could not be retained")
        with evidence.phase("input_archive"):
            scratch_parent = "/frontier" if os.access("/frontier", os.W_OK) else None
            scratch = Path(tempfile.mkdtemp(prefix="frontier-verifier-gpu-", dir=scratch_parent))
            inputs, outputs = scratch / "inputs.tar", scratch / "outputs.tar"
            _input_archive(inputs, payload, dataset)
        with evidence.phase("provisioning", GPU_CREATE_TIMEOUT_SECONDS) as window:
            create_anchor = time.monotonic()
            runner = _create_runner(driver_url, verification_id, token, gpu_type,
                                    seconds, gpu_count, **({"window": window} if phase_control else {}))
            runner_deadline = _runner_deadline(runner, seconds,
                                               anchor=create_anchor if phase_control else None)
            if phase_control:
                evidence.deadline = evidence.deadline.with_grant(runner_deadline)
            granted_seconds = runner["granted_seconds"]
        with evidence.phase("upload", min(granted_seconds, 120)) as window:
            ssh = _ssh_argv(runner, scratch)
            ssh_environment = _ssh_environment(scratch)
            _upload_inputs(ssh, inputs, granted_seconds, ssh_environment,
                           runner_deadline=runner_deadline, **({"window": window} if phase_control else {}))
        with evidence.phase("remote_execution", granted_seconds + 60) as window:
            remote = _remote_verifier_command(
                mode, gpu_count, remaining_seconds=_remaining(runner_deadline))
            if phase_control:
                returncode, stalled = _run_remote_verifier(
                    ssh, remote, outputs, granted_seconds, ssh_environment,
                    runner_deadline=runner_deadline, window=window, evidence=evidence)
            else:
                returncode, stalled = _run_remote_verifier(
                    ssh, remote, outputs, granted_seconds, ssh_environment,
                    runner_deadline=runner_deadline)
            evidence.events[-1]["transport_outcome"] = (
                "stalled" if stalled else "zero_exit" if returncode == 0 else "nonzero_or_unconfirmed_exit")
            if stalled:
                raise VerifierGpuError("the Verifier GPU transport stalled; remote completion is unconfirmed")
            if returncode != 0:
                if phase_control:
                    evidence.remote_failure = _read_remote_failure(
                        outputs, MAX_RESULT_BYTES + max_evidence + ARCHIVE_OVERHEAD_BYTES)
                if not (phase_control and evidence.remote_failure
                        and evidence.remote_failure["evidence_available"]):
                    raise VerifierGpuError("the remote Verifier GPU command failed" if phase_control
                                           else "the evaluator failed on the Verifier GPU")
        with evidence.phase("output_validation"):
            _copy_outputs(outputs, result_path, evidence_path, max_evidence)
    except BaseException as error:
        failure = error
    finally:
        try:
            with evidence.phase("delete", DELETE_RESERVE_SECONDS, cleanup=True) as window:
                if runner is None:
                    # A lost create response can hide a runner. Absence is not proven.
                    evidence.events[-1]["runner_termination"] = (
                        "unconfirmed_no_identity" if any(e["phase"] == "provisioning" for e in evidence.events)
                        else "not_requested")
                else:
                    terminated = _terminate_runner(driver_url, verification_id, token,
                                                   str(runner["runner_id"]),
                                                   **({"window": window} if phase_control else {}))
                    evidence.events[-1]["runner_termination"] = "confirmed" if terminated else "unconfirmed"
                    if not terminated:
                        raise VerifierGpuError("Verifier GPU runner termination was not confirmed")
        except BaseException as error:
            if failure is None:
                failure = error
        try:
            with evidence.phase("cleanup", DELETE_RESERVE_SECONDS, cleanup=True):
                if scratch is not None:
                    shutil.rmtree(scratch, **({} if phase_control else {"ignore_errors": True}))
        except BaseException as error:
            if failure is None:
                failure = error
    if not phase_control:
        if not terminated:
            if failure is not None:
                raise VerifierGpuError(
                    "Verifier GPU execution failed and runner termination was not confirmed") from failure
            raise VerifierGpuError("Verifier GPU runner termination was not confirmed")
        if failure is not None:
            if isinstance(failure, VerifierGpuError):
                raise failure
            raise VerifierGpuError(
                f"Verifier GPU execution raised {type(failure).__name__}") from failure
        return
    if evidence.write_failed and failure is None:
        failure = VerifierGpuError("the Verifier phase evidence could not be retained")
    if failure is not None:
        # Fixed public text; the typed phase record is operator-only evidence.
        raise VerifierGpuError("Verifier GPU control failed", private_evidence=evidence.snapshot()) from failure


def _collect(function, index: int, errors: list) -> None:
    try:
        function(index)
    except BaseException as error:  # noqa: BLE001 - classified by run_sandboxes
        errors.append(error)


# ---------------------------------------------------------------- command test sandboxes
# task.toml [test_sandboxes.<name>] access = "command" (verifier.json gpu.sandboxes, placement
# "control"): the judge runs here and runs commands on runners of the sandbox's image. The full
# contract is in the module docstring ("Command test sandboxes").
WORKSPACE = "/workspace/"  # the runner user's home; every sandbox path is below it
SANDBOX_LOGS = "/workspace/frontier-sandbox-logs"
STEP_KILL_AFTER_SECONDS = 5
STEP_TRANSPORT_GRACE_SECONDS = 60
SANDBOX_UPLOAD_SECONDS = 120
SANDBOX_COLLECT_SECONDS = 120
MAX_SANDBOX_STEPS = 16
MAX_SANDBOX_OUTPUTS = 64
MAX_LOG_TAIL_BYTES = 8192
STEP_STATUS = re.compile(rb"frontier-sandbox-step ([0-9]{1,3}) ([0-9]{1,20})\n")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")


class SandboxStep(NamedTuple):
    """One command of a run: argv (argv[0] absolute), its wall limit in seconds and extra
    environment variables. It runs in the sandbox's workdir with the image runtime. The last
    step of a run may set seconds=None: its limit is the rest of the runner's grant (less the
    per-step grace and the collect reserve), fixed when it starts."""

    argv: list[str] | tuple[str, ...]
    seconds: int | None
    env: dict[str, str] | None = None


class SandboxRun(NamedTuple):
    """One runner of a declared command sandbox: `files` (a local directory) becomes the
    sandbox's workdir, `steps` run there in order, `outputs` (paths relative to the workdir,
    files or directories) come back. `max_output_bytes` defaults to (and may not exceed) one
    SSH capture chunk."""

    sandbox: str
    files: Path | str
    steps: list[SandboxStep] | tuple[SandboxStep, ...]
    outputs: list[str] | tuple[str, ...] = ()
    max_output_bytes: int | None = None


class StepResult(NamedTuple):
    """`exit_code` is None when the step timed out or its session was lost. `lost_session`: the
    step's session stalled (no heartbeat) or ended without the step's status line, so the
    machine lost the command's wrapper; on a candidate-owned machine that is the candidate's
    doing (owner 2026-11-05: a scored zero), on a trusted one infrastructure."""

    exit_code: int | None
    timed_out: bool
    seconds: float
    lost_session: bool = False


class SandboxResult(NamedTuple):
    """`steps[i]` is None for a step that did not run (an earlier one failed or timed out).
    `outputs` is the local directory holding the declared outputs that existed (laid out as
    in the workdir), None when they exceeded max_output_bytes (`outputs_exceeded`). `logs[i]`
    is the last 8 KiB of step i's stdout+stderr (None if it did not run). `unreadable_outputs`:
    declared outputs (or files below them, workdir-relative) the runner user could not read,
    not returned (on a candidate machine: the candidate's chmod)."""

    steps: list[StepResult | None]
    outputs: Path | None
    outputs_exceeded: bool
    logs: list[bytes | None]
    unreadable_outputs: list[str] = []


class _SandboxCancelled(VerifierGpuError):
    def __init__(self):
        super().__init__("the command sandbox run was stopped after another run failed")


_SANDBOX_CALLS = 0
_SANDBOX_LOCK = threading.Lock()


def _relative(path) -> PurePosixPath:
    if not isinstance(path, str) or not path or "\x00" in path or "\n" in path:
        raise VerifierGpuError("a sandbox output is not a relative path")
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or relative.parts in ((), (".",)):
        raise VerifierGpuError(f"a sandbox output must be a path below the workdir: {path!r}")
    return relative


def _sandbox_requests(runs, sandboxes: dict, mode: str, capture: int) -> list[SandboxRun]:
    """Refuse a malformed set of runs before any runner is leased."""
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
        counts[run.sandbox] = counts.get(run.sandbox, 0) + 1
        files = Path(run.files)
        if not files.is_dir() or files.is_symlink():
            raise VerifierGpuError("a sandbox run's files is not a directory")
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
        lease = sandboxes[run.sandbox]["lease_seconds"][mode]
        if budget > lease:
            raise VerifierGpuError(
                f"the steps of a {run.sandbox!r} run need up to {budget} s with upload, collect "
                f"and per-step grace; its lease_seconds.{mode} is {lease}")
        outputs = list(run.outputs) if isinstance(run.outputs, (list, tuple)) else None
        if outputs is None or len(outputs) > MAX_SANDBOX_OUTPUTS:
            raise VerifierGpuError(f"a sandbox run declares at most {MAX_SANDBOX_OUTPUTS} outputs")
        for output in outputs:
            _relative(output)
        if run.max_output_bytes is not None and (
                type(run.max_output_bytes) is not int or not 1 <= run.max_output_bytes <= capture):
            raise VerifierGpuError(f"a sandbox run's max_output_bytes must be in 1..={capture}")
    for name, count in counts.items():
        if count > sandboxes[name]["max_concurrent"]:
            raise VerifierGpuError(f"{count} runs of {name!r} at once; its max_concurrent is "
                                   f"{sandboxes[name]['max_concurrent']}")
    return runs


def _sandbox_inputs(path: Path, files: Path) -> None:
    def refuse_special(info: tarfile.TarInfo):
        if not (info.isfile() or info.isdir() or info.issym()):
            raise VerifierGpuError("a sandbox upload contains a device, FIFO or hard link")
        return info

    with tarfile.open(path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        archive.add(files, arcname=".", filter=refuse_special)


def _step_command(index: int, step: SandboxStep, workdir: str, limit: int) -> str:
    """One step: the image runtime, the step's env and argv under `timeout` (its `limit`, also
    FRONTIER_VERIFIER_LOCAL_DEADLINE) in the workdir, its output to the step log; one status
    line (exit status, elapsed ns) on stdout."""
    variables = " ".join(shlex.quote(f"{k}={v}") for k, v in sorted((step.env or {}).items()))
    command = (
        f"mkdir -p {SANDBOX_LOGS} || exit 125; "
        + _remote_runtime(limit) +
        f"cd {shlex.quote(workdir)} || exit 125; "
        "frontier_start=$(date +%s%N); "
        f"env -- {variables} timeout --kill-after={STEP_KILL_AFTER_SECONDS} {limit} "
        + " ".join(shlex.quote(item) for item in step.argv) +
        f" </dev/null >{SANDBOX_LOGS}/step-{index}.log 2>&1; frontier_status=$?; "
        "frontier_end=$(date +%s%N); "
        "printf 'frontier-sandbox-step %s %s\\n' \"$frontier_status\" "
        "\"$((frontier_end - frontier_start))\""
    )
    return _supervised(command)


def _collect_command(workdir: str, outputs: list[str], steps: int) -> str:
    """One tar on stdout: the declared outputs that exist, the step log tails, and the list of
    declared outputs (or files below them) the runner user cannot read, as paths relative to
    /workspace. On a candidate machine the candidate owns its workdir and can chmod a file it
    wrote; such a file is not archived (tar skips it) and is reported, a fact like any other
    output (find's own exit status is not one: an unreadable directory makes it non-zero
    after listing that directory)."""
    listing = f"{SANDBOX_LOGS}/collect.list"
    unreadable = f"{SANDBOX_LOGS}/unreadable.list"
    command = (f"mkdir -p {SANDBOX_LOGS} || exit 125; : > {listing} || exit 125; "
               f": > {unreadable} || exit 125; ")
    for index in range(steps):
        log, tail = f"{SANDBOX_LOGS}/step-{index}.log", f"{SANDBOX_LOGS}/tail-{index}"
        command += (f"if [ -f {log} ]; then tail -c {MAX_LOG_TAIL_BYTES} {log} > {tail} && "
                    f"printf '%s\\0' {tail.removeprefix(WORKSPACE)} >> {listing}; fi; ")
    command += f"cd {WORKSPACE} || exit 125; "
    for output in outputs:
        path = shlex.quote(f"{workdir}/{output}".removeprefix(WORKSPACE))
        command += (f"if [ -e {path} ] || [ -L {path} ]; then "
                    f"printf '%s\\0' {path} >> {listing}; "
                    f"find {path} ! -type l ! -readable -print0 >> {unreadable} 2>/dev/null; fi; ")
    command += (f"printf '%s\\0' {unreadable.removeprefix(WORKSPACE)} >> {listing}; "
                f"exec tar -C {WORKSPACE} -cf - --ignore-failed-read --warning=none --null "
                f"-T {listing}")
    return command


def _session(ssh: list[str], remote: str, stdout_path: Path, end: float,
             environment: dict[str, str], cancel: threading.Event, *,
             max_stdout: int | None = None, heartbeat: bool = True) -> tuple[int | None, str]:
    """Run one SSH session until it exits or `end`; (returncode, outcome) with outcome
    "exited", "deadline", "stalled" (no heartbeat for REMOTE_STALL_TIMEOUT_SECONDS) or
    "oversize" (stdout passed max_stdout). The local ssh is always stopped."""
    last_activity = time.monotonic()
    outcome = "exited"
    with stdout_path.open("wb") as stream:
        process = subprocess.Popen(ssh + [remote], stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.PIPE, env=environment)
        selector = selectors.DefaultSelector()
        try:
            os.set_blocking(process.stderr.fileno(), False)
            selector.register(process.stderr, selectors.EVENT_READ)
            while process.poll() is None:
                now = time.monotonic()
                if cancel.is_set():
                    raise _SandboxCancelled()
                if now >= end:
                    outcome = "deadline"
                    break
                if heartbeat and now - last_activity >= REMOTE_STALL_TIMEOUT_SECONDS:
                    outcome = "stalled"
                    break
                if max_stdout is not None and stdout_path.stat().st_size > max_stdout:
                    outcome = "oversize"
                    break
                for key, _ in selector.select(min(1.0, max(0.0, end - now))):
                    if os.read(key.fileobj.fileno(), 65536):
                        last_activity = time.monotonic()
                    else:
                        selector.unregister(key.fileobj)
        finally:
            selector.close()
            stopped = _stop_process(process, time.monotonic() + 20)
            process.stderr.close()
            if not stopped:
                raise VerifierGpuError("a sandbox SSH session could not be stopped")
    if outcome == "exited" and max_stdout is not None and stdout_path.stat().st_size > max_stdout:
        outcome = "oversize"
    return (process.returncode if outcome == "exited" else None), outcome


def _sandbox_step(ssh: list[str], environment: dict[str, str], index: int, step: SandboxStep,
                  workdir: str, deadline: float, cancel: threading.Event,
                  scratch: Path) -> StepResult:
    grace = STEP_KILL_AFTER_SECONDS + STEP_TRANSPORT_GRACE_SECONDS
    start = time.monotonic()
    limit = (step.seconds if step.seconds is not None
             else math.floor(deadline - start - grace - SANDBOX_COLLECT_SECONDS))
    if limit < 1 or start + limit + grace > deadline:
        raise VerifierGpuError("the sandbox runner's granted lease cannot fit the next step")
    status_path = scratch / f"step-{index}.status"
    returncode, outcome = _session(
        ssh, _step_command(index, step, workdir, limit), status_path,
        start + limit + grace, environment, cancel)
    elapsed = time.monotonic() - start
    if outcome == "deadline":
        # The heartbeat was alive past the step's limit: the command outlived its own timeout.
        return StepResult(None, True, elapsed)
    body = status_path.read_bytes()
    match = STEP_STATUS.fullmatch(body) if len(body) <= 64 else None
    if outcome == "stalled" or returncode != 0 or match is None:
        return StepResult(None, False, elapsed, lost_session=True)
    status, nanoseconds = int(match.group(1)), int(match.group(2))
    seconds = nanoseconds / 1e9
    if status in (124, 128 + signal.SIGKILL) and seconds >= limit:
        return StepResult(None, True, seconds)
    return StepResult(status, False, seconds)


def _extract_outputs(archive_path: Path, workdir: str, outputs: list[str], steps: int,
                     destination: Path) -> tuple[list[bytes | None], list[str]]:
    """Extract the declared outputs (regular files and directories only, modes kept) into
    `destination`; return the step log tails and the unreadable outputs (workdir-relative)."""
    root = PurePosixPath(workdir.removeprefix(WORKSPACE))
    wanted = [root / _relative(output) for output in outputs]
    tails = {PurePosixPath(SANDBOX_LOGS.removeprefix(WORKSPACE)) / f"tail-{index}": index
             for index in range(steps)}
    unreadable_list = PurePosixPath(SANDBOX_LOGS.removeprefix(WORKSPACE)) / "unreadable.list"
    unreadable: list[str] = []
    logs: list[bytes | None] = [None] * steps
    modes: list[tuple[Path, int]] = []
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts:
                    continue
                if name in tails and member.isfile() and member.size <= MAX_LOG_TAIL_BYTES:
                    logs[tails[name]] = archive.extractfile(member).read()
                    continue
                if name == unreadable_list and member.isfile():
                    body = archive.extractfile(member).read().decode("utf-8", errors="replace")
                    unreadable = sorted(str(PurePosixPath(item).relative_to(root))
                                        for item in body.split("\0") if item)
                    continue
                if not any(name == path or path in name.parents for path in wanted):
                    continue
                target = destination.joinpath(*name.relative_to(root).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    target.chmod(0o700)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("wb") as stream:
                        shutil.copyfileobj(archive.extractfile(member), stream, 1024 * 1024)
                else:
                    continue
                modes.append((target, member.mode & 0o777))
    except (OSError, tarfile.TarError) as error:
        raise VerifierGpuError("a sandbox output archive could not be read") from error
    for target, mode in reversed(modes):
        target.chmod(mode)
    return logs, unreadable


def _sandbox_run(index: int, request: SandboxRun, spec: dict, *, mode: str, call: int,
                 scratch: Path, driver: tuple, outer: _OuterDeadline | None,
                 capture: int, destination: Path, cancel: threading.Event,
                 record: dict) -> SandboxResult:
    driver_url, verification_id, token = driver
    seconds = spec["lease_seconds"][mode]
    workdir = spec["workdir"]
    runner = None
    try:
        record["phase"] = "input_archive"
        inputs = scratch / "inputs.tar"
        _sandbox_inputs(inputs, Path(request.files))
        record["phase"] = "provisioning"
        if cancel.is_set():
            raise _SandboxCancelled()
        create_end = time.monotonic() + GPU_CREATE_TIMEOUT_SECONDS
        if outer is not None:
            create_end = min(create_end, outer.work_end)
        anchor = time.monotonic()
        record["create_requested"] = True
        runner = _create_runner(
            driver_url, verification_id, token, spec["resources"]["gpu_type"], seconds,
            spec["resources"]["gpus"], window=_WaitWindow(create_end, "work_budget_exhausted"),
            idempotency_scope=f"sandbox-v1:{call}:{mode}:{index}:{request.sandbox}",
            image=spec["image"])
        deadline = _runner_deadline(runner, seconds, anchor=anchor)
        if outer is not None:
            deadline = min(deadline, outer.work_end)
        record["granted_seconds"] = runner["granted_seconds"]
        record["phase"] = "upload"
        if cancel.is_set():
            raise _SandboxCancelled()
        ssh = _ssh_argv(runner, scratch)
        environment = _ssh_environment(scratch)
        _upload_inputs(ssh, inputs, min(_remaining(deadline), SANDBOX_UPLOAD_SECONDS),
                       environment, runner_deadline=deadline, destination=workdir)
        steps: list[StepResult | None] = [None] * len(request.steps)
        for number, step in enumerate(request.steps):
            record["phase"] = f"step-{number}"
            steps[number] = _sandbox_step(ssh, environment, number, step, workdir, deadline,
                                          cancel, scratch)
            if steps[number].timed_out or steps[number].exit_code != 0:
                break
        if any(result is not None and result.lost_session for result in steps):
            # The machine lost a command's wrapper: nothing it returns is collected.
            record["steps"] = [None if r is None else "lost" if r.lost_session
                               else "timeout" if r.timed_out else r.exit_code for r in steps]
            record["status"] = "completed"
            return SandboxResult(steps, None, False, [None] * len(steps))
        record["steps"] = [None if r is None else ("timeout" if r.timed_out else r.exit_code)
                           for r in steps]
        record["phase"] = "collect"
        ran = sum(result is not None for result in steps)
        limit = request.max_output_bytes or capture
        archive = scratch / "outputs.tar"
        end = min(deadline, time.monotonic() + SANDBOX_COLLECT_SECONDS)
        returncode, outcome = _session(
            ssh, _collect_command(workdir, list(request.outputs), ran), archive, end,
            environment, cancel, max_stdout=limit + ARCHIVE_OVERHEAD_BYTES, heartbeat=False)
        if outcome == "oversize":
            record["status"] = "completed"
            return SandboxResult(steps, None, True, [None] * len(steps))
        if outcome != "exited" or returncode != 0:
            raise VerifierGpuError("the sandbox outputs could not be collected")
        destination.mkdir()
        logs, unreadable = _extract_outputs(archive, workdir, list(request.outputs), ran,
                                            destination)
        logs += [None] * (len(steps) - ran)
        if sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()) > limit:
            shutil.rmtree(destination)
            record["status"] = "completed"
            return SandboxResult(steps, None, True, logs)
        record["status"] = "completed"
        return SandboxResult(steps, destination, False, logs, unreadable)
    except _SandboxCancelled:
        record["status"], record["reason"] = "cancelled", "cancelled"
        raise
    except BaseException as error:
        record["status"] = "failed"
        record.setdefault("reason", "deadline" if isinstance(error, _PhaseTimeout) else record["phase"])
        cancel.set()
        raise
    finally:
        if runner is None:
            record["termination"] = ("unconfirmed_no_identity" if record.get("create_requested")
                                     else "not_requested")
        else:
            end = time.monotonic() + DELETE_RESERVE_SECONDS
            if outer is not None:
                end = min(end, outer.cleanup_end)
            try:
                terminated = _terminate_runner(driver_url, verification_id, token,
                                               str(runner["runner_id"]),
                                               window=_WaitWindow(end, "cleanup_budget_exhausted"))
            except VerifierGpuError:
                terminated = False
            record["termination"] = "confirmed" if terminated else "unconfirmed"
            if not terminated:
                cancel.set()


def sandbox_workdir(name: str) -> str:
    """The declared workdir of command test sandbox `name`: where its files land on a runner."""
    gpu = gpu_parameters()
    if gpu["placement"] != "control" or name not in gpu["sandboxes"]:
        raise VerifierGpuError(f"no command test sandbox {name!r} in task.toml")
    return gpu["sandboxes"][name]["workdir"]


def run_sandboxes(runs, *, into) -> list[SandboxResult]:
    """Run each SandboxRun on its own runner of its declared command test sandbox, all at once;
    the results in order. `into` is an empty local directory the judge owns: run i's outputs
    go to `into/<i>`. See "Command test sandboxes" in the module docstring."""
    global _SANDBOX_CALLS
    gpu = gpu_parameters()
    if gpu["placement"] != "control":
        raise VerifierGpuError('run_sandboxes needs a task.toml [test_sandboxes.<name>] with '
                               'access = "command"')
    mode = _required_env("FRONTIER_VERIFICATION_MODE")
    capture = _upload_chunk_bytes(os.environ)
    runs = _sandbox_requests(runs, gpu["sandboxes"], mode, capture)
    into = Path(into)
    if not into.is_dir() or any(into.iterdir()):
        raise VerifierGpuError("run_sandboxes needs an empty directory `into`")
    driver = _driver_config()
    outer = _OuterDeadline.from_environment()
    with _SANDBOX_LOCK:
        _SANDBOX_CALLS += 1
        call = _SANDBOX_CALLS
    scratch_parent = "/frontier" if os.access("/frontier", os.W_OK) else None
    scratch = Path(tempfile.mkdtemp(prefix="frontier-verifier-sandbox-", dir=scratch_parent))
    records = [{"run": index, "sandbox": run.sandbox, "phase": "pending", "status": "started",
                "termination": "not_requested"} for index, run in enumerate(runs)]
    cancel = threading.Event()
    results: list = [None] * len(runs)
    errors: list = []
    try:
        def one(index: int):
            directory = scratch / f"run-{index}"
            directory.mkdir()
            results[index] = _sandbox_run(
                index, runs[index], gpu["sandboxes"][runs[index].sandbox], mode=mode, call=call,
                scratch=directory, driver=driver, outer=outer, capture=capture,
                destination=into / str(index), cancel=cancel, record=records[index])

        threads = [threading.Thread(target=_collect, args=(one, index, errors),
                                    name=f"frontier-sandbox-{index}", daemon=True)
                   for index in range(len(runs))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    evidence = {"schema": "frontiersmith.verifier-sandboxes/v1", "mode": mode, "call": call,
                "runs": [{key: value for key, value in record.items()
                          if key in ("run", "sandbox", "phase", "status", "reason", "steps",
                                     "termination", "granted_seconds")}
                         for record in records]}
    cause = next((error for error in errors if not isinstance(error, _SandboxCancelled)),
                 errors[0] if errors else None)
    if any(record["termination"] not in ("confirmed", "not_requested") for record in records):
        raise VerifierGpuError("command sandbox runner termination was not confirmed",
                               private_evidence=evidence) from cause
    if cause is not None:
        raise VerifierGpuError("a command sandbox run failed", private_evidence=evidence) from cause
    return results


def _pump(connection: socket.socket) -> None:
    selector = selectors.DefaultSelector()
    selector.register(connection, selectors.EVENT_READ, "network")
    selector.register(sys.stdin.buffer, selectors.EVENT_READ, "stdin")
    try:
        while True:
            for key, _ in selector.select():
                if key.data == "network":
                    chunk = connection.recv(65536)
                    if not chunk:
                        return
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                else:
                    chunk = os.read(sys.stdin.buffer.fileno(), 65536)
                    if not chunk:
                        selector.unregister(sys.stdin.buffer)
                        connection.shutdown(socket.SHUT_WR)
                    else:
                        connection.sendall(chunk)
    finally:
        selector.close()


def _connect_main(runner_id: str, broker_path: str) -> int:
    """ssh's ProxyCommand: relay stdin/stdout through the Verifier's _ConnectBroker. It never
    holds the Driver bearer."""
    if RUNNER_ID.fullmatch(runner_id) is None:
        raise VerifierGpuError("the ProxyCommand runner id is invalid")
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            connection.connect(broker_path)
        except OSError as error:
            raise VerifierGpuError("the Verifier GPU connect broker is unavailable") from error
        status = bytearray()
        while not status.endswith(b"\n"):
            if len(status) >= 512:
                raise VerifierGpuError("the Verifier GPU connect broker reply exceeded its bound")
            chunk = connection.recv(1)
            if not chunk:
                raise VerifierGpuError("the Verifier GPU connect broker closed before replying")
            status.extend(chunk)
        reply = status.decode("utf-8", errors="replace").rstrip("\n")
        if reply != "OK":
            raise VerifierGpuError(reply.removeprefix("ERR ") or "the Driver CONNECT failed")
        _pump(connection)
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    try:
        if len(sys.argv) != 4 or sys.argv[1] != "_connect":
            raise VerifierGpuError("this adapter is not a public command")
        raise SystemExit(_connect_main(sys.argv[2], sys.argv[3]))
    except VerifierGpuError as error:
        print(f"frontier-verifier-gpu: {error}", file=sys.stderr)
        raise SystemExit(1)
