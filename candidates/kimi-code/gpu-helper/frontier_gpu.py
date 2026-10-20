#!/usr/bin/env python3
"""Use a Driver-managed GPU runner without exposing Modal or SSH credentials."""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.parse import urlsplit
import uuid


RUNNER_ID = re.compile(r"^gpurunner_[0-9A-HJKMNP-TV-Z]{26}$")
WORKSPACE = Path("/workspace")
SSH = Path("/usr/bin/ssh")
PROXY = Path("/opt/frontier/candidate/bin/frontier-gpu-connect-proxy")
MAX_RESPONSE = 64 * 1024 * 1024
REQUEST_ATTEMPTS = 16
MAX_TOKEN = 16 * 1024
DEFAULT_TRANSFER_LIMIT = 64 * 1024 * 1024
# Driver create waits for image import and sandbox readiness. Each provider
# phase can take one hour; keep a small response margin, not a 30s RPC timeout.
GPU_CREATE_TIMEOUT_SECONDS = 2 * 60 * 60 + 120
GPU_HELPER_CONTRACT = "frontier.gpu-helper/v3"


class GpuError(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def token() -> str:
    path = Path(os.environ.get("FRONTIER_DRIVER_TOKEN_FILE", ""))
    if not path.is_absolute():
        raise GpuError("FRONTIER_DRIVER_TOKEN_FILE must be absolute")
    try:
        volume_root = path.parent.resolve(strict=True)
        source = path.lstat()
        resolved = path.resolve(strict=True)
        resolved.relative_to(volume_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise GpuError("Driver token file is unavailable or leaves its volume") from error
    if not (stat.S_ISREG(source.st_mode) or stat.S_ISLNK(source.st_mode)):
        raise GpuError("Driver token path is not a regular file or projected symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= MAX_TOKEN:
            raise GpuError("Driver token file has an invalid size")
        value = os.read(descriptor, MAX_TOKEN + 1)
    finally:
        os.close(descriptor)
    decoded = value.decode().strip()
    if not decoded or any(character.isspace() for character in decoded):
        raise GpuError("Driver token has an invalid format")
    return decoded


class Client:
    def __init__(self) -> None:
        parsed = urlsplit(os.environ.get("FRONTIER_DRIVER_API_URL", ""))
        if parsed.scheme != "http" or not parsed.hostname or parsed.path not in {"", "/"}:
            raise GpuError("FRONTIER_DRIVER_API_URL must be a plain HTTP origin")
        self.host, self.port, self.auth = parsed.hostname, parsed.port or 80, token()

    def request(self, method: str, path: str, body: object | None = None, *, idempotency: str | None = None, timeout: float = 30) -> object:
        encoded = canonical(body) if body is not None else None
        headers = {"Authorization": f"Bearer {self.auth}"}
        if encoded is not None:
            headers.update({"Content-Type": "application/json", "Content-Length": str(len(encoded))})
        if idempotency:
            headers["Idempotency-Key"] = idempotency
        attempts = REQUEST_ATTEMPTS if method in {"GET", "DELETE"} or idempotency else 1
        for attempt in range(attempts):
            connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
            try:
                connection.request(method, path, body=encoded, headers=headers)
                response = connection.getresponse()
                data = response.read(MAX_RESPONSE + 1)
                if len(data) > MAX_RESPONSE:
                    raise GpuError("Driver response exceeded its size limit")
                if response.status in {408, 429, 500, 502, 503, 504} and attempt + 1 < attempts:
                    time.sleep(min(10, 0.25 * (2 ** attempt)))
                    continue
                if not 200 <= response.status < 300:
                    raise GpuError(f"Driver returned HTTP {response.status}")
                try:
                    return json.loads(data)
                except json.JSONDecodeError as error:
                    raise GpuError("Driver response was not JSON") from error
            except (OSError, http.client.HTTPException) as error:
                if attempt + 1 == attempts:
                    raise GpuError("Driver request failed") from error
                time.sleep(min(10, 0.25 * (2 ** attempt)))
            finally:
                connection.close()
        raise GpuError("Driver request failed")


def validate_runner_id(value: str) -> str:
    if not RUNNER_ID.fullmatch(value):
        raise GpuError("GPU runner_id is invalid")
    return value


def runner(client: Client, runner_id: str) -> dict[str, object]:
    value = client.request("GET", f"/v1/gpu-runners/{validate_runner_id(runner_id)}")
    if not isinstance(value, dict) or value.get("runner_id") != runner_id:
        raise GpuError("Driver returned an invalid GPU runner")
    return value


def ready_runner(client: Client, runner_id: str) -> dict[str, object]:
    value = runner(client, runner_id)
    gateway = value.get("ssh_gateway")
    if value.get("state") != "ready" or not isinstance(gateway, dict):
        raise GpuError("GPU runner is not ready")
    if gateway.get("username") != "frontier" or gateway.get("authentication") != "connect_bearer":
        raise GpuError("Driver returned an unsupported SSH gateway")
    return value


def fingerprint(public_key: str) -> str:
    fields = public_key.split()
    if len(fields) not in {2, 3} or fields[0] != "ssh-ed25519":
        raise GpuError("Driver SSH host key is not Ed25519")
    try:
        blob = base64.b64decode(fields[1], validate=True)
    except ValueError as error:
        raise GpuError("Driver SSH host key is malformed") from error
    prefix = len(b"ssh-ed25519").to_bytes(4, "big") + b"ssh-ed25519" + (32).to_bytes(4, "big")
    if len(blob) != len(prefix) + 32 or not blob.startswith(prefix):
        raise GpuError("Driver SSH host key has an invalid Ed25519 blob")
    return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")


def ssh_command(value: dict[str, object], remote: list[str], known_hosts: Path, *, tty: bool = False) -> list[str]:
    runner_id = str(value["runner_id"])
    gateway = value["ssh_gateway"]
    assert isinstance(gateway, dict)
    public_key = gateway.get("server_host_public_key")
    expected = gateway.get("server_host_key_sha256")
    if not isinstance(public_key, str) or fingerprint(public_key) != expected:
        raise GpuError("Driver SSH host key fingerprint mismatch")
    alias = f"frontier-gpu-{runner_id}"
    known_hosts.write_text(f"{alias} {public_key.rstrip()}\n", encoding="utf-8")
    known_hosts.chmod(0o600)
    command = [
        str(SSH), "-tt" if tty else "-T", "-o", "BatchMode=yes",
        "-o", "PreferredAuthentications=none", "-o", "PasswordAuthentication=no",
        "-o", "KbdInteractiveAuthentication=no", "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={known_hosts}", "-o", f"HostKeyAlias={alias}",
        "-o", f"ProxyCommand={PROXY} {runner_id}", "frontier@frontier-gpu",
    ]
    return command + remote


def run_ssh(value: dict[str, object], remote: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
    with tempfile.TemporaryDirectory(prefix="frontier-gpu-ssh-") as directory:
        known_hosts = Path(directory) / "known_hosts"
        command = ssh_command(value, remote, known_hosts, tty=bool(kwargs.pop("tty", False)))
        max_output = kwargs.pop("max_output_bytes", None)
        if max_output is None:
            return subprocess.run(command, check=False, **kwargs)
        output = kwargs.pop("stdout")
        with subprocess.Popen(command, stdout=subprocess.PIPE, **kwargs) as process:
            try:
                total = 0
                while chunk := process.stdout.read(min(65536, max_output + 1 - total)):
                    total += len(chunk)
                    if total > max_output:
                        raise GpuError("pull archive exceeds the transfer byte limit")
                    output.write(chunk)
                return subprocess.CompletedProcess(command, process.wait())
            finally:
                if process.poll() is None:
                    process.kill()


def workspace_path(raw: str, *, existing: bool) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise GpuError("path must be absolute")
    resolved = path.resolve(strict=existing)
    try:
        resolved.relative_to(WORKSPACE)
    except ValueError as error:
        raise GpuError("path must stay below /workspace") from error
    return resolved


def remote_path(raw: str, *, directory: bool = False) -> PurePosixPath:
    path = PurePosixPath(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise GpuError("remote path must be absolute and must not contain '..'")
    if not directory and path == PurePosixPath("/"):
        raise GpuError("remote source path must not be the filesystem root")
    return path


def transfer_limit() -> int:
    try:
        value = int(os.environ.get("FRONTIER_GPU_TRANSFER_MAX_BYTES", str(DEFAULT_TRANSFER_LIMIT)))
    except ValueError as error:
        raise GpuError("FRONTIER_GPU_TRANSFER_MAX_BYTES must be an integer") from error
    if not 1 <= value <= 1024 * 1024 * 1024:
        raise GpuError("GPU transfer limit must be between 1 byte and 1 GiB")
    return value


def build_tar(source: Path, output: Path) -> None:
    """Preserve source.basename; push's remote directory is its parent."""
    total = 0
    with tarfile.open(output, "w", format=tarfile.USTAR_FORMAT) as archive:
        entries = [source] if source.is_file() else [source, *sorted(source.rglob("*"))]
        for path in entries:
            metadata = path.lstat()
            if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)):
                raise GpuError(f"push rejects links and special files: {path}")
            total += metadata.st_size if stat.S_ISREG(metadata.st_mode) else 0
            if total > transfer_limit():
                raise GpuError("push exceeds the transfer byte limit")
            archive.add(path, arcname=path.relative_to(source.parent), recursive=False)


def validate_pull_tar(path: Path) -> None:
    total = 0
    with tarfile.open(path, "r:") as archive:
        members = archive.getmembers()
        if not members or len(members) > 100_000:
            raise GpuError("pull archive has an invalid member count")
        for member in members:
            name = PurePosixPath(member.name)
            if member.name.startswith("/") or ".." in name.parts or not (member.isdir() or member.isfile()):
                raise GpuError(f"pull archive has an unsafe member: {member.name}")
            total += member.size
            if total > transfer_limit():
                raise GpuError("pull exceeds the transfer byte limit")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=GPU_HELPER_CONTRACT)
    sub = parser.add_subparsers(dest="operation", required=True)
    create = sub.add_parser("create")
    create.add_argument("--type", required=True, dest="gpu_type")
    create.add_argument("--count", type=int, default=1)
    create.add_argument("--seconds", type=int, required=True)
    create.add_argument("--idempotency-key", default=None,
                        help="reuse this key and identical arguments after an ambiguous create reply")
    sub.add_parser("list")
    sub.add_parser("quota", help="show remaining Agent SSH sessions without opening SSH")
    for name in ("status", "delete", "ssh"):
        command = sub.add_parser(name); command.add_argument("runner_id")
    execute = sub.add_parser("exec"); execute.add_argument("runner_id"); execute.add_argument("command", nargs=argparse.REMAINDER)
    push = sub.add_parser("push"); push.add_argument("runner_id"); push.add_argument("source"); push.add_argument("remote_directory")
    pull = sub.add_parser("pull"); pull.add_argument("runner_id"); pull.add_argument("remote_path"); pull.add_argument("destination")
    args = parser.parse_args(argv)
    try:
        client = Client()
        if args.operation == "quota":
            view = client.request("GET", "/v1/run")
            quota = view.get("ssh_sessions") if isinstance(view, dict) else None
            if not isinstance(quota, dict):
                raise GpuError("Driver does not expose SSH quota; remaining sessions are unknown")
            scope = quota.get("scope")
            separate = quota.get("verifier_uses_separate_budget")
            counts = [quota.get(k) for k in ("limit", "used", "remaining")]
            if (not isinstance(scope, str) or scope not in {"agent", "legacy_shared"}
                    or type(separate) is not bool or separate != (scope == "agent")
                    or any(type(n) is not int or not 0 <= n <= 1024 for n in counts)
                    or counts[1] + counts[2] != counts[0]):
                raise GpuError("Driver returned an invalid SSH quota")
            print(json.dumps({k: quota[k] for k in
                ("scope", "limit", "used", "remaining", "verifier_uses_separate_budget")}, sort_keys=True))
            if not separate:
                print("Legacy shared SSH quota: development/final verifier SSH also consumes this quota.", file=sys.stderr)
            return 0
        if args.operation == "create":
            if args.count <= 0 or args.seconds <= 0:
                raise GpuError("GPU count and seconds must be positive")
            key = args.idempotency_key or f"frontier-gpu-{uuid.uuid4()}"
            if not 1 <= len(key) <= 255 or not key.isascii() or any(
                ord(character) < 33 or ord(character) > 126 for character in key
            ):
                raise GpuError("idempotency key must be 1-255 printable ASCII characters without spaces")
            print(f"GPU create idempotency key: {key}. If the reply is lost, use list/status; "
                  "retry only with this same key and arguments.", file=sys.stderr)
            value = client.request("POST", "/v1/gpu-runners", {"gpu_type": args.gpu_type, "gpu_count": args.count, "max_duration_seconds": args.seconds}, idempotency=key, timeout=GPU_CREATE_TIMEOUT_SECONDS)
            print(json.dumps(value, sort_keys=True)); return 0
        if args.operation == "list":
            print(json.dumps(client.request("GET", "/v1/gpu-runners"), sort_keys=True)); return 0
        if args.operation == "status":
            print(json.dumps(runner(client, args.runner_id), sort_keys=True)); return 0
        if args.operation == "delete":
            print(json.dumps(client.request("DELETE", f"/v1/gpu-runners/{validate_runner_id(args.runner_id)}"), sort_keys=True)); return 0
        value = ready_runner(client, args.runner_id)
        if args.operation == "ssh":
            return run_ssh(value, [], tty=True).returncode
        if args.operation == "exec":
            if not args.command or args.command[0] == "--": args.command = args.command[1:]
            if not args.command: raise GpuError("exec requires a command after --")
            return run_ssh(value, [shlex.join(args.command)]).returncode
        if args.operation == "push":
            source = workspace_path(args.source, existing=True)
            remote = remote_path(args.remote_directory, directory=True)
            with tempfile.NamedTemporaryFile(prefix="frontier-gpu-push-", suffix=".tar") as archive:
                build_tar(source, Path(archive.name)); archive.seek(0)
                command = f"mkdir -p -- {shlex.quote(str(remote))} && tar -xpf - -C {shlex.quote(str(remote))}"
                result = run_ssh(value, [command], stdin=archive)
            return result.returncode
        if args.operation == "pull":
            remote = remote_path(args.remote_path)
            destination = workspace_path(args.destination, existing=False)
            destination.mkdir(mode=0o700, parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix="frontier-gpu-pull-", suffix=".tar") as archive:
                parent, name = str(remote.parent), remote.name
                result = run_ssh(value, [f"tar -cpf - -C {shlex.quote(parent)} -- {shlex.quote(name)}"], stdout=archive,
                                 max_output_bytes=transfer_limit() + 10 * 1024 * 1024)
                if result.returncode: return result.returncode
                archive.flush()
                if Path(archive.name).stat().st_size > transfer_limit() + 10 * 1024 * 1024:
                    raise GpuError("pull archive exceeds the transfer byte limit")
                validate_pull_tar(Path(archive.name))
                with tarfile.open(archive.name, "r:") as incoming:
                    incoming.extractall(destination, filter="data")
            return 0
        raise AssertionError(args.operation)
    except (GpuError, OSError, UnicodeError, http.client.HTTPException, subprocess.SubprocessError, tarfile.TarError) as error:
        print(f"frontier-gpu: ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
