#!/usr/bin/env python3
"""OpenSSH ProxyCommand for a bearer-authenticated Frontier GPU CONNECT route."""

from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
import re
import selectors
import socket
import stat
import sys
from urllib.parse import urlsplit


MAX_CONNECT_HEADERS = 16 * 1024
MAX_TOKEN_BYTES = 16 * 1024
RUNNER_ID_PATTERN = re.compile(r"^gpurunner_[0-9A-HJKMNP-TV-Z]{26}$")


class ProxyError(RuntimeError):
    pass


def parse_driver_url(value: str) -> tuple[str, int]:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ProxyError("Driver API URL is malformed") from error
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or port is None
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ProxyError("Driver API URL must be an exact http origin with an explicit port")
    try:
        ipaddress.ip_address(parsed.hostname)
    except ValueError as error:
        raise ProxyError("Driver API URL host must be a literal IP address") from error
    return parsed.hostname, port


def read_token(path: Path) -> str:
    if not path.is_absolute():
        raise ProxyError("Driver token path must be absolute")
    try:
        root = path.parent.resolve(strict=True)
        source = path.lstat()
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise ProxyError("Driver token file is unavailable or leaves its volume") from error
    if not (stat.S_ISREG(source.st_mode) or stat.S_ISLNK(source.st_mode)):
        raise ProxyError("Driver token path is not a regular file or projected symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= MAX_TOKEN_BYTES:
            raise ProxyError("Driver token file has an invalid type or size")
        raw = os.read(descriptor, MAX_TOKEN_BYTES + 1)
    finally:
        os.close(descriptor)
    try:
        token = raw.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ProxyError("Driver token is not UTF-8") from error
    if not token or any(character.isspace() for character in token):
        raise ProxyError("Driver token has an invalid format")
    return token


def open_connect_socket(base_url: str, runner_id: str) -> socket.socket:
    if not RUNNER_ID_PATTERN.fullmatch(runner_id):
        raise ProxyError("GPU runner_id is invalid")
    host, port = parse_driver_url(base_url)
    token_path = Path(os.environ.get("FRONTIER_DRIVER_TOKEN_FILE", "/var/run/frontier/token"))
    bearer = read_token(token_path)
    try:
        connection = socket.create_connection((host, port), timeout=30)
    except OSError as error:
        raise ProxyError("could not connect to the Driver Agent API") from error
    try:
        connection.settimeout(30)
        authority_host = f"[{host}]" if ":" in host else host
        request = (
            f"CONNECT /v1/gpu-runners/{runner_id}/connect HTTP/1.1\r\n"
            f"Host: {authority_host}:{port}\r\n"
            f"Authorization: Bearer {bearer}\r\n\r\n"
        ).encode("ascii")
        connection.sendall(request)
        response = bytearray()
        while not response.endswith(b"\r\n\r\n"):
            if len(response) >= MAX_CONNECT_HEADERS:
                raise ProxyError("Driver CONNECT response headers exceeded the limit")
            chunk = connection.recv(1)
            if not chunk:
                raise ProxyError("Driver closed the CONNECT response early")
            response.extend(chunk)
        first_line = bytes(response).split(b"\r\n", 1)[0]
        if not first_line.startswith(b"HTTP/1.1 200 "):
            fields = first_line.split(b" ", 2)
            status = (
                fields[1].decode("ascii")
                if len(fields) >= 2 and len(fields[1]) == 3 and fields[1].isdigit()
                else "unknown"
            )
            raise ProxyError(f"Driver CONNECT was rejected with HTTP {status}")
        connection.settimeout(None)
        return connection
    except BaseException:
        connection.close()
        raise


def bridge(connection: socket.socket) -> None:
    # BufferedReader.read(n) waits for n bytes or EOF on a pipe, so even the
    # first short SSH handshake can deadlock. A daemon stdin reader also races
    # Python shutdown when the peer exits first. Pump raw descriptors in one
    # thread, with bounded queues and backpressure in both directions.
    incoming, outgoing = sys.stdin.fileno(), sys.stdout.fileno()
    limit = 256 * 1024
    to_peer, to_stdout = bytearray(), bytearray()
    stdin_eof = peer_eof = write_closed = False
    blocking = {fd: os.get_blocking(fd) for fd in (incoming, outgoing)}
    timeout = connection.gettimeout()
    # epoll cannot watch regular files (e.g. redirected stdout).
    selector_type = (selectors.SelectSelector if any(
        stat.S_ISREG(os.fstat(fd).st_mode) for fd in blocking
    ) else selectors.DefaultSelector)
    with selector_type() as selector:
        def watch(fileobj: object, events: int, label: str) -> None:
            try:
                selector.get_key(fileobj)
            except KeyError:
                if events:
                    selector.register(fileobj, events, label)
            else:
                if events:
                    selector.modify(fileobj, events, label)
                else:
                    selector.unregister(fileobj)

        try:
            for fd in blocking:
                os.set_blocking(fd, False)
            connection.setblocking(False)
            while True:
                if peer_eof and not to_stdout:
                    return
                if stdin_eof and not to_peer and not write_closed:
                    connection.shutdown(socket.SHUT_WR)
                    write_closed = True
                watch(incoming, selectors.EVENT_READ if not stdin_eof and
                      not peer_eof and len(to_peer) < limit else 0, "stdin")
                watch(outgoing, selectors.EVENT_WRITE if to_stdout else 0, "stdout")
                watch(connection, (selectors.EVENT_READ if not peer_eof and
                      len(to_stdout) < limit else 0) | (selectors.EVENT_WRITE if
                      to_peer and not peer_eof else 0), "peer")
                for key, events in selector.select():
                    try:
                        if key.data == "stdin":
                            chunk = os.read(incoming, min(65536, limit - len(to_peer)))
                            if chunk:
                                to_peer.extend(chunk)
                            else:
                                stdin_eof = True
                        elif key.data == "stdout":
                            del to_stdout[:os.write(outgoing, to_stdout)]
                        else:
                            if events & selectors.EVENT_READ:
                                chunk = connection.recv(min(65536, limit - len(to_stdout)))
                                if chunk:
                                    to_stdout.extend(chunk)
                                else:
                                    peer_eof = True
                            if events & selectors.EVENT_WRITE and not peer_eof:
                                del to_peer[:connection.send(to_peer)]
                    except BlockingIOError:
                        continue
        finally:
            connection.settimeout(timeout)
            for fd, original in blocking.items():
                os.set_blocking(fd, original)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runner_id")
    arguments = parser.parse_args()
    try:
        with open_connect_socket(os.environ.get("FRONTIER_DRIVER_API_URL", ""), arguments.runner_id) as connection:
            bridge(connection)
        return 0
    except (ProxyError, OSError) as error:
        print(f"frontier GPU CONNECT proxy: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
