#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import select
import socket
import subprocess
import sys
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("frontier_gpu_connect_proxy.py")
SPEC = importlib.util.spec_from_file_location("frontier_gpu_connect_proxy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy)
RUNNER = "gpurunner_01ARZ3NDEKTSV4RRFFQ69G5FAV"


class FakeSocket:
    def __init__(self, response: bytes):
        self.response = bytearray(response)
        self.request = bytearray()
        self.closed = False

    def settimeout(self, _timeout: object) -> None: pass
    def sendall(self, body: bytes) -> None: self.request.extend(body)
    def close(self) -> None: self.closed = True
    def recv(self, maximum: int) -> bytes:
        value = bytes(self.response[:maximum]); del self.response[:maximum]; return value
    def __enter__(self): return self
    def __exit__(self, *_args: object) -> None: self.close()


class ProxyTests(unittest.TestCase):
    def start_bridge(self):
        parent, child = socket.socketpair()
        parent.settimeout(5)
        script = (
            "import importlib.util,socket,sys; "
            "s=importlib.util.spec_from_file_location('proxy',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "m.bridge(socket.socket(fileno=int(sys.argv[2])))"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(MODULE_PATH), str(child.fileno())],
            pass_fds=(child.fileno(),), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        child.close()

        def cleanup():
            parent.close()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream and not stream.closed:
                    stream.close()
        self.addCleanup(cleanup)
        return parent, process

    def test_short_handshakes_flow_before_stdin_eof(self) -> None:
        peer, process = self.start_bridge()
        banner = b"SSH-2.0-test\r\n"
        process.stdin.write(banner)
        process.stdin.flush()
        self.assertEqual(peer.recv(1024), banner)
        peer.sendall(banner)
        self.assertTrue(select.select([process.stdout], [], [], 5)[0])
        self.assertEqual(process.stdout.read(len(banner)), banner)
        self.assertIsNone(process.poll())

    def test_stdin_eof_half_closes_but_drains_peer_output(self) -> None:
        peer, process = self.start_bridge()
        process.stdin.close()
        self.assertEqual(peer.recv(1), b"")
        payload = bytes(range(256)) * 5000
        # Drain concurrently via communicate; the peer sender is a test-only
        # non-daemon worker, not part of the runtime implementation.
        import threading
        def send():
            peer.sendall(payload)
            peer.shutdown(socket.SHUT_WR)
        thread = threading.Thread(target=send)
        thread.start()
        process.stdin = None
        output, error = process.communicate(timeout=10)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(process.returncode, 0, error)
        self.assertEqual(output, payload)

    def test_peer_eof_exits_while_stdin_is_still_open(self) -> None:
        peer, process = self.start_bridge()
        peer.sendall(b"final-output")
        peer.shutdown(socket.SHUT_WR)
        self.assertEqual(process.wait(timeout=5), 0, process.stderr.read())
        self.assertFalse(process.stdin.closed)
        self.assertEqual(process.stdout.read(), b"final-output")

    def test_large_upload_preserves_every_byte(self) -> None:
        peer, process = self.start_bridge()
        payload = bytes(range(256)) * 5000
        import threading
        def write():
            process.stdin.write(payload)
            process.stdin.close()
        thread = threading.Thread(target=write)
        thread.start()
        received = bytearray()
        while chunk := peer.recv(65536):
            received.extend(chunk)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(received, payload)
        peer.shutdown(socket.SHUT_WR)
        self.assertEqual(process.wait(timeout=5), 0)

    def test_connect_authenticates_and_preserves_upgraded_bytes(self) -> None:
        fake = FakeSocket(b"HTTP/1.1 200 OK\r\n\r\nssh-bytes")
        with mock.patch.object(proxy.socket, "create_connection", return_value=fake), mock.patch.object(proxy, "read_token", return_value="secret"):
            with proxy.open_connect_socket("http://127.0.0.1:8080", RUNNER) as connection:
                self.assertEqual(connection.recv(32), b"ssh-bytes")
        self.assertIn(f"CONNECT /v1/gpu-runners/{RUNNER}/connect HTTP/1.1".encode(), fake.request)
        self.assertIn(b"Authorization: Bearer secret\r\n", fake.request)

    def test_rejection_does_not_echo_bearer(self) -> None:
        fake = FakeSocket(b"HTTP/1.1 403 Forbidden\r\n\r\n")
        with mock.patch.object(proxy.socket, "create_connection", return_value=fake), mock.patch.object(proxy, "read_token", return_value="do-not-log"):
            with self.assertRaises(proxy.ProxyError) as captured:
                proxy.open_connect_socket("http://127.0.0.1:8080", RUNNER)
        self.assertNotIn("do-not-log", str(captured.exception))
        self.assertIn("HTTP 403", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
