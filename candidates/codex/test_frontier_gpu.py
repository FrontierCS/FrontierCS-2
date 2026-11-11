#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import io
import os
import json
import subprocess
import sys
from pathlib import Path
import tarfile
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("frontier_gpu.py")
SPEC = importlib.util.spec_from_file_location("frontier_gpu", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gpu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu)


RUNNER = "gpurunner_01ARZ3NDEKTSV4RRFFQ69G5FAV"
PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEBl6i5bIpO9XaiHpHtaWNJYcINmCqRzQ0XGtLu6dGFL"


class GpuTests(unittest.TestCase):
    def test_response_larger_than_one_mebibyte(self) -> None:
        data = json.dumps({"detail": "x" * (2 * 1024 * 1024)}).encode()
        client = gpu.Client.__new__(gpu.Client)
        client.host, client.port, client.auth = "localhost", 8080, "test-token"
        response = mock.Mock(status=200)
        response.read.side_effect = lambda n: data[:n]
        connection = mock.Mock()
        connection.getresponse.return_value = response
        with mock.patch.object(gpu.http.client, "HTTPConnection", return_value=connection):
            self.assertEqual(client.request("GET", "/v1/gpu-runners")["detail"], "x" * (2 * 1024 * 1024))

    def test_create_retries_only_with_same_idempotency_key(self) -> None:
        client = gpu.Client.__new__(gpu.Client)
        client.host, client.port, client.auth = "localhost", 8080, "test-token"
        for key, expected in (("same-key", 6), (None, 1)):
            connection = mock.Mock()
            connection.request.side_effect = [OSError("temporary")] * 5 + [None]
            connection.getresponse.return_value = mock.Mock(status=200)
            connection.getresponse.return_value.read.return_value = b'{"runner_id":"same-runner"}'
            with mock.patch.object(gpu.http.client, "HTTPConnection", return_value=connection), mock.patch.object(gpu.time, "sleep"):
                if key:
                    self.assertEqual(client.request("POST", "/v1/gpu-runners", {}, idempotency=key), {"runner_id": "same-runner"})
                else:
                    with self.assertRaises(gpu.GpuError):
                        client.request("POST", "/v1/gpu-runners", {})
            self.assertEqual(connection.request.call_count, expected)
            self.assertTrue(all(call == connection.request.call_args_list[0] for call in connection.request.call_args_list))

    def test_quota_is_read_only_and_never_opens_ssh(self) -> None:
        quota = {"scope": "agent", "limit": 32, "used": 14, "remaining": 18,
                 "verifier_uses_separate_budget": True}
        client = mock.Mock()
        client.request.return_value = {"ssh_sessions": quota}
        with mock.patch.object(gpu, "Client", return_value=client), \
                mock.patch.object(gpu, "run_ssh") as ssh, \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(gpu.main(["quota"]), 0)
            self.assertEqual(json.loads(out.getvalue()), quota)
        client.request.assert_called_once_with("GET", "/v1/run")
        ssh.assert_not_called()

    def test_quota_reports_legacy_shared_and_unknown_honestly(self) -> None:
        client = mock.Mock()
        for value, expected_code, expected_message in [
            ({}, 1, "unknown"),
            ({"ssh_sessions": {"scope": "legacy_shared", "limit": 32, "used": 32,
              "remaining": 0, "verifier_uses_separate_budget": False}}, 0, "Legacy shared"),
            ({"ssh_sessions": {"scope": "agent", "limit": 32, "used": True,
              "remaining": 31, "verifier_uses_separate_budget": True}}, 1, "invalid"),
            ({"ssh_sessions": {"scope": [], "limit": 32, "used": 14,
              "remaining": 18, "verifier_uses_separate_budget": True}}, 1, "invalid"),
        ]:
            client.request.return_value = value
            with mock.patch.object(gpu, "Client", return_value=client), \
                    mock.patch("sys.stdout", new_callable=io.StringIO), \
                    mock.patch("sys.stderr", new_callable=io.StringIO) as err:
                self.assertEqual(gpu.main(["quota"]), expected_code)
                self.assertIn(expected_message, err.getvalue())

    def test_create_waits_for_cold_provisioning_and_reuses_explicit_key(self) -> None:
        client = mock.Mock()
        client.request.return_value = {"runner_id": RUNNER}
        with mock.patch.object(gpu, "Client", return_value=client), \
                mock.patch("sys.stdout", new_callable=io.StringIO), \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(gpu.main(["create", "--type", "T4", "--seconds", "300",
                                       "--idempotency-key", "stable-create-key"]), 0)
        client.request.assert_called_once_with(
            "POST", "/v1/gpu-runners",
            {"gpu_type": "T4", "gpu_count": 1, "max_duration_seconds": 300},
            idempotency="stable-create-key", timeout=7320,
        )

    def test_ordinary_requests_keep_short_timeout(self) -> None:
        client = object.__new__(gpu.Client)
        client.host, client.port, client.auth = "127.0.0.1", 8080, "test-token"
        response = mock.Mock(status=200)
        response.read.return_value = b"{}"
        with mock.patch.object(gpu.http.client, "HTTPConnection") as connection:
            connection.return_value.getresponse.return_value = response
            client.request("GET", "/v1/gpu-runners")
        connection.assert_called_once()
        self.assertEqual(connection.call_args.args, ("127.0.0.1", 8080))
        self.assertGreater(connection.call_args.kwargs["timeout"], 29)
        self.assertLessEqual(connection.call_args.kwargs["timeout"], 30)
        connection.return_value.close.assert_called_once()

    def recovery_fixture(self, replies):
        client = gpu.Client.__new__(gpu.Client)
        client.host, client.port, client.auth = "localhost", 8080, "SECRET-TOKEN"
        connections = []
        for status, body in replies:
            connection = mock.Mock()
            response = mock.Mock(status=status)
            response.read.return_value = json.dumps(body).encode() if not isinstance(body, bytes) else body
            connection.getresponse.return_value = response
            connections.append(connection)
        return client, connections

    def test_pending_recovers_same_key_and_canonical_body(self):
        client, connections = self.recovery_fixture([
            (425, {"status": 425, "code": "idempotency_in_progress", "title": "SECRET-TOKEN"}),
            (200, {"runner_id": RUNNER}),
        ])
        with mock.patch.object(gpu.http.client, "HTTPConnection", side_effect=connections), \
                mock.patch.object(gpu.time, "sleep"), mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            result = client.request("POST", "/v1/gpu-runners", {"b": 2, "a": 1}, idempotency="stable")
        self.assertEqual(result, {"runner_id": RUNNER})
        self.assertEqual(connections[0].request.call_args, connections[1].request.call_args)
        self.assertEqual(connections[0].request.call_args.kwargs["body"], b'{"a":1,"b":2}\n')
        self.assertIn("idempotency_in_progress", err.getvalue())
        self.assertIn("1/1024", err.getvalue())
        self.assertNotIn("SECRET", err.getvalue())

    def test_only_typed_rate_429_recovers_not_quota_or_unknown(self):
        for code, status, retries in [
            ("mutation_rate_exhausted", 429, True),
            ("quota_exhausted", 429, False),
            ("gpu_type_not_allowed", 429, False),
            ("unknown_secret_code", 429, False),
            ("gpu_backend_unavailable", 502, False),
        ]:
            client, connections = self.recovery_fixture([
                (status, {"status": status, "code": code, "title": "SECRET-TOKEN"}),
                (200, {}),
            ])
            with self.subTest(code=code), mock.patch.object(gpu.http.client, "HTTPConnection", side_effect=connections) as factory, \
                    mock.patch.object(gpu.time, "sleep"), mock.patch("sys.stderr", new_callable=io.StringIO):
                if retries:
                    self.assertEqual(client.request("POST", "/v1/gpu-runners", {}, idempotency="stable"), {})
                else:
                    with self.assertRaises(gpu.GpuError) as raised:
                        client.request("POST", "/v1/gpu-runners", {}, idempotency="stable")
                    self.assertNotIn("SECRET", str(raised.exception))
                self.assertEqual(factory.call_count, 2 if retries else 1)

    def test_malformed_problem_is_terminal_and_never_echoed(self):
        for body in [b'SECRET-TOKEN', b'[]', b'{"code":"idempotency_in_progress"}',
                     b'{"status":429,"code":"idempotency_in_progress"}']:
            client, connections = self.recovery_fixture([(425, body)])
            with mock.patch.object(gpu.http.client, "HTTPConnection", side_effect=connections) as factory:
                with self.assertRaisesRegex(gpu.GpuError, "unrecognized_problem") as raised:
                    client.request("POST", "/v1/gpu-runners", {}, idempotency="stable")
            self.assertEqual(factory.call_count, 1)
            self.assertNotIn("SECRET", str(raised.exception))

    def test_read_and_delete_keep_existing_transient_http_retries(self):
        for method in ("GET", "DELETE"):
            client, connections = self.recovery_fixture([(503, b"SECRET-TOKEN"), (200, {})])
            with self.subTest(method=method), \
                    mock.patch.object(gpu.http.client, "HTTPConnection", side_effect=connections), \
                    mock.patch.object(gpu.time, "sleep"), mock.patch("sys.stderr", new_callable=io.StringIO) as err:
                self.assertEqual(client.request(method, "/v1/gpu-runners"), {})
                self.assertEqual(connections[0].request.call_args, connections[1].request.call_args)
                self.assertNotIn("SECRET", err.getvalue())

    def test_duplicate_key_conflict_is_terminal_and_safe(self):
        client, connections = self.recovery_fixture([
            (409, {"status": 409, "code": "idempotency_conflict", "title": "SECRET-TOKEN"}),
        ])
        with mock.patch.object(gpu.http.client, "HTTPConnection", side_effect=connections) as factory:
            with self.assertRaisesRegex(gpu.GpuError, "idempotency_conflict.*different arguments") as raised:
                client.request("POST", "/v1/gpu-runners", {}, idempotency="stable")
        self.assertEqual(factory.call_count, 1)
        self.assertNotIn("SECRET", str(raised.exception))

    def test_quota_detail_is_only_a_fixed_driver_public_reason(self):
        for title, expected in [("GPU instance limit reached", "GPU instance limit reached"),
                                ("GPU seconds exhausted", "GPU seconds exhausted"),
                                ([], "allowance is exhausted"),
                                ("SECRET-TOKEN", "allowance is exhausted")]:
            client, connections = self.recovery_fixture([
                (429, {"status": 429, "code": "quota_exhausted", "title": title}),
            ])
            with mock.patch.object(gpu.http.client, "HTTPConnection", side_effect=connections):
                with self.assertRaisesRegex(gpu.GpuError, expected) as raised:
                    client.request("POST", "/v1/gpu-runners", {}, idempotency="stable")
            self.assertNotIn("SECRET", str(raised.exception))

    def test_quota_facts_are_safe_terminal_and_do_not_change_the_request(self):
        quota = {"resource": "gpu_seconds", "principal": "verifier", "limit": 100,
                 "used": 90, "reserved": 10, "remaining": 0,
                 "requested_count": 1, "requested_seconds": 30, "private": "SECRET"}
        for mutation in ({}, {"used": "SECRET"}, {"used": True}, {"principal": "SECRET"},
                         {"resource": []}, {"principal": []}):
            client, connections = self.recovery_fixture([
                (429, {"status": 429, "code": "quota_exhausted", "quota_failure": {**quota, **mutation}}),
                (200, {}),
            ])
            with mock.patch.object(gpu.http.client, "HTTPConnection", side_effect=connections) as factory:
                with self.assertRaises(gpu.GpuError) as raised:
                    client.request("POST", "/v1/gpu-runners", {}, idempotency="stable")
            self.assertEqual(factory.call_count, 1)
            self.assertEqual("quota=" in str(raised.exception), not mutation)
            self.assertNotIn("SECRET", str(raised.exception))

    def test_pending_and_uncertain_cleanup_reasons_remain_distinct(self):
        for title in ['GPU create is pending; retry the same key and arguments',
                      'GPU create cleanup is uncertain; retry the same key and arguments']:
            client, connections = self.recovery_fixture([
                (425, {"status": 425, "code": "idempotency_in_progress", "title": title}),
            ])
            with mock.patch.object(gpu.http.client, "HTTPConnection", side_effect=connections), \
                    mock.patch.object(gpu, "CREATE_RECOVERY_ATTEMPTS", 1):
                with self.assertRaises(gpu.GpuError) as raised:
                    client.request('POST', '/v1/gpu-runners', {}, idempotency='stable')
            self.assertIn(title, str(raised.exception))

    def test_pending_deadline_bounds_attempt_timeouts_and_sleep(self):
        client, connections = self.recovery_fixture([(425, {"status": 425, "code": "idempotency_in_progress"})] * 8)
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        with mock.patch.object(gpu.http.client, "HTTPConnection", side_effect=connections) as factory, \
                mock.patch.object(gpu.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(gpu.time, "sleep", side_effect=sleep), mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaisesRegex(gpu.GpuError, "deadline exceeded.*idempotency_in_progress"):
                client.request("POST", "/v1/gpu-runners", {}, idempotency="stable", timeout=1)
        self.assertEqual(clock[0], 1)
        self.assertEqual([c.kwargs["timeout"] for c in factory.call_args_list], [1, .75, .25])

    def test_recovery_attempt_cap_preserves_final_typed_error(self):
        client, connections = self.recovery_fixture([
            (425, {"status": 425, "code": "idempotency_in_progress"}),
        ] * 2)
        with mock.patch.object(gpu.http.client, "HTTPConnection", side_effect=connections) as factory, \
                mock.patch.object(gpu, "CREATE_RECOVERY_ATTEMPTS", 2), \
                mock.patch.object(gpu.time, "sleep"), mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaisesRegex(gpu.GpuError, "idempotency_in_progress.*after 2 attempts"):
                client.request("POST", "/v1/gpu-runners", {}, idempotency="stable")
        self.assertEqual(factory.call_count, 2)

    def test_oversize_problem_is_bounded_and_not_echoed(self):
        client, connections = self.recovery_fixture([(425, b'SECRET' * 1000)])
        with mock.patch.object(gpu.http.client, "HTTPConnection", side_effect=connections):
            with self.assertRaisesRegex(gpu.GpuError, 'size limit') as raised:
                client.request('POST', '/v1/gpu-runners', {}, idempotency='stable')
        connections[0].getresponse.return_value.read.assert_called_once_with(gpu.MAX_PROBLEM + 1)
        self.assertNotIn('SECRET', str(raised.exception))

    def test_absolute_deadline_interrupts_trickling_headers(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                try:
                    for byte in b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}':
                        self.connection.sendall(bytes([byte]))
                        time.sleep(.03)
                except OSError:
                    pass
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = gpu.Client.__new__(gpu.Client)
        client.host, client.port, client.auth = '127.0.0.1', server.server_port, 'test'
        started = time.monotonic()
        try:
            with mock.patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaisesRegex(gpu.GpuError, 'deadline exceeded'):
                    client.request('POST', '/v1/gpu-runners', {}, idempotency='stable', timeout=.15)
            self.assertLess(time.monotonic() - started, .75)
        finally:
            server.shutdown()
            server.server_close()

    def test_absolute_deadline_interrupts_trickling_body(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200)
                self.send_header('Content-Length', '100')
                self.end_headers()
                try:
                    for _ in range(100):
                        self.wfile.write(b' ')
                        self.wfile.flush()
                        time.sleep(.03)
                except OSError:
                    pass
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        client = gpu.Client.__new__(gpu.Client)
        client.host, client.port, client.auth = '127.0.0.1', server.server_port, 'test'
        started = time.monotonic()
        try:
            with mock.patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaisesRegex(gpu.GpuError, 'deadline exceeded'):
                    client.request('POST', '/v1/gpu-runners', {}, idempotency='stable', timeout=.15)
            self.assertLess(time.monotonic() - started, .75)
        finally:
            server.shutdown()
            server.server_close()

    def test_documented_push_exec_pull_round_trip(self) -> None:
        # Exercise actual tar commands using a local fake SSH transport. No
        # Driver, network, credentials, model request or GPU is involved.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace, remote = root / "workspace", root / "remote"
            source = workspace / "gpu-work"
            source.mkdir(parents=True)
            remote.mkdir()
            (source / "test.py").write_text("public-task-code")
            def local_ssh(_value, command, **kwargs):
                kwargs.pop("max_output_bytes", None)
                return subprocess.run(command[0], shell=True, check=False, **kwargs)
            with mock.patch.object(gpu, "WORKSPACE", workspace), \
                    mock.patch.object(gpu, "Client"), \
                    mock.patch.object(gpu, "ready_runner", return_value={}), \
                    mock.patch.object(gpu, "run_ssh", side_effect=local_ssh):
                self.assertEqual(gpu.main(["push", RUNNER, str(source), str(remote)]), 0)
                check = remote / "gpu-work" / "test.py"
                self.assertEqual(check.read_text(), "public-task-code")
                self.assertEqual(gpu.main(["exec", RUNNER, "--", "test", "-f", str(check)]), 0)
                output = workspace / "gpu-output"
                self.assertEqual(gpu.main(["pull", RUNNER, str(check), str(output)]), 0)
                self.assertEqual((output / "test.py").read_text(), "public-task-code")

    def test_pull_stream_preserves_bytes_without_recording_quota(self) -> None:
        command = [sys.executable, "-c", "import os; os.write(1,b'x'*20000)"]
        with tempfile.TemporaryFile() as output:
            with mock.patch.object(gpu, "ssh_command", return_value=command):
                gpu.run_ssh({}, [], stdout=output)
            self.assertEqual(output.seek(0, 2), 20000)

    def test_runner_id_is_closed(self) -> None:
        self.assertEqual(gpu.validate_runner_id(RUNNER), RUNNER)
        for value in ("bad", "gpurunner_../../x", RUNNER.lower()):
            with self.assertRaises(gpu.GpuError):
                gpu.validate_runner_id(value)

    def test_ssh_command_pins_driver_key_and_proxy(self) -> None:
        value = {
            "runner_id": RUNNER,
            "ssh_gateway": {
                "server_host_public_key": PUBLIC_KEY,
                "server_host_key_sha256": gpu.fingerprint(PUBLIC_KEY),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            known_hosts = Path(directory) / "known_hosts"
            command = gpu.ssh_command(value, ["true"], known_hosts)
            self.assertIn("StrictHostKeyChecking=yes", command)
            self.assertIn(f"ProxyCommand={gpu.PROXY} {RUNNER}", command)
            self.assertIn(PUBLIC_KEY, known_hosts.read_text())

    def test_push_tar_rejects_links_and_is_relative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            (root / "code.py").write_text("print('ok')\n")
            archive = Path(directory) / "source.tar"
            with mock.patch.dict(os.environ, {"FRONTIER_GPU_TRANSFER_MAX_BYTES": "1"}):
                gpu.build_tar(root, archive)
            with tarfile.open(archive, "r:") as value:
                self.assertEqual([member.name for member in value.getmembers()], ["source", "source/code.py"])
            (root / "link").symlink_to("code.py")
            with self.assertRaisesRegex(gpu.GpuError, "rejects links"):
                gpu.build_tar(root, archive)

    def test_pull_archive_accepts_past_legacy_member_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "many.tar"
            with tarfile.open(archive, "w") as value:
                for index in range(100001):
                    value.addfile(tarfile.TarInfo(f"item-{index}"))
                    value.members.clear()
            gpu.validate_pull_tar(archive)

    def test_pull_tar_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.tar"
            with tarfile.open(archive, "w") as value:
                member = tarfile.TarInfo("../escape")
                member.size = 1
                value.addfile(member, io.BytesIO(b"x"))
            with self.assertRaisesRegex(gpu.GpuError, "unsafe member"):
                gpu.validate_pull_tar(archive)

    def test_remote_paths_are_not_limited_to_workspace(self) -> None:
        self.assertEqual(gpu.remote_path("/tmp/project"), gpu.PurePosixPath("/tmp/project"))
        self.assertEqual(gpu.remote_path("/opt/output", directory=True), gpu.PurePosixPath("/opt/output"))
        for value in ("relative", "/tmp/../etc"):
            with self.assertRaises(gpu.GpuError):
                gpu.remote_path(value)


if __name__ == "__main__":
    unittest.main()
