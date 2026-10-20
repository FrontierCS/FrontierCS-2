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
        connection.assert_called_once_with("127.0.0.1", 8080, timeout=30)
        connection.return_value.close.assert_called_once()

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
