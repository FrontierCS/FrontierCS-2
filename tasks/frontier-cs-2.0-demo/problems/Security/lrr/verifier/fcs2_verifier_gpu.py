#!/usr/bin/env python3
"""Run one FrontierSmith verifier evaluation on Driver-owned Modal GPUs.

The EKS Verifier Pod remains the control plane.  It already has a
verification-scoped Driver bearer, so this adapter can create exactly one
Verifier-owned runner with the task-declared GPU count, connect only through
the Driver's audited SSH gateway,
copy the platform-downloaded payload and dataset to the private verifier image,
and copy the bounded result/evidence back.  The Modal credential and provider
address never enter this process or the task image.
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
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
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


class VerifierGpuError(RuntimeError):
    """A bounded Verifier GPU lifecycle or transport failure."""

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

    def snapshot(self) -> dict:
        value = {"schema": "frontiersmith.verifier-control-phases/v1",
                 "evaluator_progress": "unobserved", "events": self.events,
                 "evidence_write_failed": self.write_failed,
                 "deadline_basis": self.deadline.basis if self.deadline else "unavailable"}
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


def _config() -> tuple[str, str, str, str, int, int]:
    driver_url = _required_env("FRONTIER_DRIVER_INTERNAL_URL").rstrip("/")
    parsed = urlparse(driver_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname \
            or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise VerifierGpuError("FRONTIER_DRIVER_INTERNAL_URL is not a closed HTTP(S) URL")
    verification_id = _required_env("FRONTIER_VERIFICATION_ID")
    if VERIFICATION_ID.fullmatch(verification_id) is None:
        raise VerifierGpuError("FRONTIER_VERIFICATION_ID is not canonical")
    token = _required_env("FRONTIER_VERIFIER_TOKEN")
    if len(token.encode("utf-8")) > MAX_TOKEN_BYTES or any(c.isspace() for c in token):
        raise VerifierGpuError("FRONTIER_VERIFIER_TOKEN is malformed")
    gpu_type = _required_env("FRONTIER_VERIFIER_GPU_TYPE")
    if GPU_TYPE.fullmatch(gpu_type) is None:
        raise VerifierGpuError("FRONTIER_VERIFIER_GPU_TYPE is malformed")
    try:
        seconds = int(_required_env("FRONTIER_VERIFIER_GPU_SECONDS"))
    except ValueError as error:
        raise VerifierGpuError("FRONTIER_VERIFIER_GPU_SECONDS is not an integer") from error
    if not 1 <= seconds <= 7 * 24 * 60 * 60:
        raise VerifierGpuError("FRONTIER_VERIFIER_GPU_SECONDS is outside 1s..=7d")
    try:
        gpu_count = int(os.environ.get("FRONTIER_VERIFIER_GPU_COUNT", "1"))
    except ValueError as error:
        raise VerifierGpuError("FRONTIER_VERIFIER_GPU_COUNT is not an integer") from error
    if not 1 <= gpu_count <= 64:
        raise VerifierGpuError("FRONTIER_VERIFIER_GPU_COUNT is outside 1..=64")
    return driver_url, verification_id, token, gpu_type, seconds, gpu_count


def _request(method: str, driver_url: str, verification_id: str, token: str,
             suffix: str, *, body: dict | None = None, expect: int = 200,
             timeout: float = 660.0) -> dict:
    base = f"/internal/v1/verifications/{verification_id}/gpu-runners"
    path = base + suffix
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8") \
        if body is not None else None
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if encoded is not None:
        stable = hashlib.sha256(
            f"frontiersmith-verifier-gpu-v1\0{verification_id}".encode()).hexdigest()
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
                   window: _WaitWindow | None = None) -> dict:
    runner = _request("POST", driver_url, verification_id, token, "", expect=202,
                      body={"gpu_type": gpu_type, "gpu_count": gpu_count,
                            "max_duration_seconds": seconds},
                      timeout=window.remaining(GPU_CREATE_TIMEOUT_SECONDS) if window else GPU_CREATE_TIMEOUT_SECONDS)
    runner_id = str(runner.get("runner_id") or "")
    if (RUNNER_ID.fullmatch(runner_id) is None
            or runner.get("state") != "ready"
            or runner.get("gpu_count") != gpu_count):
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


def _ssh_argv(runner: dict, scratch: Path) -> list[str]:
    gateway = runner["ssh_gateway"]
    known_hosts = scratch / "known_hosts"
    known_hosts.write_text(
        f"gpu-runner {str(gateway['server_host_public_key']).strip()}\n",
        encoding="utf-8")
    known_hosts.chmod(0o600)
    proxy = (f"{sys.executable} /opt/frontier/fcs2_verifier_gpu.py _connect "
             f"{runner['runner_id']}")
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
    for name in ("LD_PRELOAD", "NSS_WRAPPER_PASSWD", "NSS_WRAPPER_GROUP"):
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
                    total_size: int, total_digest: str) -> str:
    staging = "/workspace/frontier-verifier-upload"
    destination = "/workspace/frontier-verifier"
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
                   window: _WaitWindow | None = None) -> None:
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
                    total_digest.hexdigest())
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


def _remote_verifier_command(mode: str, gpu_count: int = 1) -> str:
    """Restore only immutable runtime state after sshd clears its env."""
    if mode not in {"train", "eval"}:
        raise VerifierGpuError("the verification mode is invalid")
    if not 1 <= gpu_count <= 64:
        raise VerifierGpuError("the Verifier GPU count is invalid")
    command = (
        "mkdir -p /workspace/frontier-verifier/output; status=0; "
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
            driver_url, verification_id, token, gpu_type, seconds, gpu_count = _config()
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
            remote = _remote_verifier_command(mode, gpu_count)
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


def _connect_main(runner_id: str) -> int:
    driver_url, verification_id, token, _gpu_type, _seconds, _gpu_count = _config()
    if RUNNER_ID.fullmatch(runner_id) is None:
        raise VerifierGpuError("the ProxyCommand runner id is invalid")
    connection = _connect_socket(driver_url, verification_id, token, runner_id)
    try:
        _pump(connection)
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3 or sys.argv[1] != "_connect":
            raise VerifierGpuError("this adapter is not a public command")
        raise SystemExit(_connect_main(sys.argv[2]))
    except VerifierGpuError as error:
        print(f"frontier-verifier-gpu: {error}", file=sys.stderr)
        raise SystemExit(1)
