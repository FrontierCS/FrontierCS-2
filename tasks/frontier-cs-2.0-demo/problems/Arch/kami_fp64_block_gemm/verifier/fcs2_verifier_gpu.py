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
import os
import pwd
import re
import selectors
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
# GPU evaluator runs, and lets the Pod distinguish a live evaluator from a
# channel whose upstream close was not propagated.  The heartbeat never enters
# result.json or evidence.tar.zst.
REMOTE_HEARTBEAT_SECONDS = 20
REMOTE_STALL_TIMEOUT_SECONDS = 120
NSS_WRAPPER_ROOT = Path("/usr/lib")
PASSWD_DATABASE = Path("/etc/passwd")
GROUP_DATABASE = Path("/etc/group")


class VerifierGpuError(RuntimeError):
    """A bounded Verifier GPU lifecycle or transport failure."""


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
                   gpu_type: str, seconds: int, gpu_count: int = 1) -> dict:
    runner = _request("POST", driver_url, verification_id, token, "", expect=202,
                      body={"gpu_type": gpu_type, "gpu_count": gpu_count,
                            "max_duration_seconds": seconds},
                      timeout=GPU_CREATE_TIMEOUT_SECONDS)
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


def _runner_deadline(runner: dict, requested_seconds: int, *, anchor: float) -> float:
    """Bound local waits by the returned grant, not the requested quota.

    Called after retaining the owned identity. The Driver caps this grant by
    verification timeout minus floor(elapsed seconds). Anchor BEFORE create and
    subtract one second to cover that truncation; provisioning spends this bound.
    The anchor is required: response-receipt time must not renew the grant.
    This bound does not prove provider billing.
    """
    granted = runner.get("granted_seconds")
    if type(granted) is not int or not 1 <= granted <= requested_seconds:
        raise VerifierGpuError("the Driver returned an invalid GPU granted duration")
    return anchor + granted - 1


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise VerifierGpuError("the Verifier GPU deadline is exhausted")
    return remaining


def _terminate_runner(driver_url: str, verification_id: str, token: str,
                      runner_id: str) -> bool:
    try:
        result = _request("DELETE", driver_url, verification_id, token,
                          f"/{runner_id}")
    except VerifierGpuError:
        return False
    return result.get("runner_id") == runner_id and result.get("state") == "terminated"


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


def _upload_inputs(ssh: list[str], inputs: Path, seconds: int,
                   environment: dict[str, str], *, runner_deadline: float | None = None) -> None:
    """Upload exactly once; every accepted SSH operation consumes quota."""
    command = ["rm -rf /workspace/frontier-verifier && "
               "mkdir -p /workspace/frontier-verifier && "
               "tar -C /workspace/frontier-verifier -xf -"]
    try:
        with inputs.open("rb") as stream:
            uploaded = subprocess.run(
                ssh + command, stdin=stream, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, timeout=min(seconds, 120, _remaining(runner_deadline)) if runner_deadline is not None else min(seconds, 120),
                check=False, env=environment)
    except subprocess.TimeoutExpired as error:
        raise VerifierGpuError("the Verifier GPU input upload timed out") from error
    stderr = uploaded.stderr or b""
    if b"Host key verification failed" in stderr:
        raise VerifierGpuError("the Verifier GPU host-key check failed")
    if b"Permission denied" in stderr:
        raise VerifierGpuError("the Verifier GPU SSH authentication failed")
    if uploaded.returncode != 0:
        raise VerifierGpuError("the Verifier GPU input upload failed")


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


def _remote_verifier_command(mode: str, gpu_count: int = 1) -> str:
    """Restore only immutable runtime state after sshd clears its env."""
    if mode not in {"train", "eval"}:
        raise VerifierGpuError("the verification mode is invalid")
    if not 1 <= gpu_count <= 64:
        raise VerifierGpuError("the Verifier GPU count is invalid")
    return (
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
        # `runtime-python` is a stable symlink in generated GPU images. Its
        # lexical dirname is /opt/frontier, while pip-installed build tools
        # live beside the real Conda interpreter. Resolve it before PATH.
        "verifier_python_real=$(readlink -f \"$verifier_python\"); "
        "test -n \"$verifier_python_real\"; "
        "verifier_python_dir=${verifier_python_real%/*}; "
        "export PATH=\"$verifier_python_dir:$PATH\"; "
        "if [ -x /usr/local/cuda/bin/nvcc ]; then "
        "export CUDA_HOME=/usr/local/cuda CUDACXX=/usr/local/cuda/bin/nvcc; "
        "export PATH=\"/usr/local/cuda/bin:$PATH\"; "
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
        ">/workspace/frontier-verifier/verifier.log 2>&1 & "
        "verifier_pid=$!; "
        "(while kill -0 \"$verifier_pid\" 2>/dev/null; do "
        "printf 'frontier-verifier-gpu-heartbeat\\n' >&2; "
        f"sleep {REMOTE_HEARTBEAT_SECONDS}; done) & heartbeat_pid=$!; "
        "wait \"$verifier_pid\" || status=$?; "
        "kill \"$heartbeat_pid\" 2>/dev/null || true; "
        "wait \"$heartbeat_pid\" 2>/dev/null || true; "
        "tar -C /workspace/frontier-verifier/output -cf - "
        "result.json evidence.tar.zst; exit $status"
    )


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _run_remote_verifier(ssh: list[str], remote: str, outputs: Path,
                         seconds: int, environment: dict[str, str], *,
                         runner_deadline: float | None = None) -> tuple[int, bool]:
    """Run the remote verifier and bound a lost SSH close independently.

    ``stalled`` means the child was stopped after its heartbeat disappeared.
    The caller may still accept a complete, bounded output archive: Modal can
    deliver the final tar bytes and then lose the SSH channel-close message.
    """
    deadline = time.monotonic() + seconds + 60
    if runner_deadline is not None:
        deadline = min(deadline, runner_deadline)
    _remaining(deadline)
    last_progress = time.monotonic()
    stalled = False
    with outputs.open("wb") as stream:
        process = subprocess.Popen(
            ssh + [remote], stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.PIPE, env=environment)
        if process.stderr is None:
            _stop_process(process)
            raise VerifierGpuError("the Verifier GPU SSH stderr pipe is unavailable")
        selector = selectors.DefaultSelector()
        selector.register(process.stderr, selectors.EVENT_READ)
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    _stop_process(process)
                    raise VerifierGpuError("the evaluator exceeded its Verifier GPU deadline")
                remaining_stall = REMOTE_STALL_TIMEOUT_SECONDS - (now - last_progress)
                if remaining_stall <= 0:
                    stalled = True
                    _stop_process(process)
                    break
                wait = min(5.0, deadline - now, remaining_stall)
                for key, _ in selector.select(wait):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if chunk:
                        last_progress = time.monotonic()
                    else:
                        selector.unregister(key.fileobj)
        finally:
            selector.close()
            process.stderr.close()
            _stop_process(process)
    return process.returncode, stalled


def run(*, payload: Path, dataset: Path, mode: str, result_path: Path,
        evidence_path: Path) -> None:
    if mode not in {"train", "eval"}:
        raise VerifierGpuError("the verification mode is invalid")
    driver_url, verification_id, token, gpu_type, seconds, gpu_count = _config()
    try:
        max_evidence = int(_required_env("FRONTIER_MAX_EVIDENCE_BYTES"))
    except ValueError as error:
        raise VerifierGpuError("FRONTIER_MAX_EVIDENCE_BYTES is invalid") from error
    if max_evidence <= 0:
        raise VerifierGpuError("FRONTIER_MAX_EVIDENCE_BYTES must be positive")

    scratch_parent = "/frontier" if os.access("/frontier", os.W_OK) else None
    scratch = Path(tempfile.mkdtemp(prefix="frontier-verifier-gpu-", dir=scratch_parent))
    runner: dict | None = None
    runner_deadline: float | None = None
    failure: BaseException | None = None
    try:
        inputs = scratch / "inputs.tar"
        outputs = scratch / "outputs.tar"
        _input_archive(inputs, payload, dataset)
        create_anchor = time.monotonic()
        runner = _create_runner(
            driver_url, verification_id, token, gpu_type, seconds, gpu_count)
        runner_deadline = _runner_deadline(runner, seconds, anchor=create_anchor)
        _remaining(runner_deadline)
        ssh = _ssh_argv(runner, scratch)
        ssh_environment = _ssh_environment(scratch)
        _upload_inputs(ssh, inputs, _remaining(runner_deadline), ssh_environment,
                       runner_deadline=runner_deadline)

        remote = _remote_verifier_command(mode, gpu_count)
        returncode, stalled = _run_remote_verifier(
            ssh, remote, outputs, _remaining(runner_deadline), ssh_environment,
            runner_deadline=runner_deadline)
        if returncode != 0 and not stalled:
            raise VerifierGpuError("the evaluator failed on the Verifier GPU")
        try:
            _copy_outputs(outputs, result_path, evidence_path, max_evidence)
        except BaseException as error:
            if stalled:
                raise VerifierGpuError(
                    "the Verifier GPU transport stalled before returning a complete output archive"
                ) from error
            raise
    except BaseException as error:
        failure = error
    finally:
        terminated = runner is None or _terminate_runner(
            driver_url, verification_id, token, str(runner["runner_id"]))
        shutil.rmtree(scratch, ignore_errors=True)
    if not terminated:
        if failure is not None:
            raise VerifierGpuError(
                "Verifier GPU execution failed and runner termination was not confirmed") \
                from failure
        raise VerifierGpuError("Verifier GPU runner termination was not confirmed")
    if failure is not None:
        if isinstance(failure, VerifierGpuError):
            raise failure
        raise VerifierGpuError(
            f"Verifier GPU execution raised {type(failure).__name__}") from failure


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
