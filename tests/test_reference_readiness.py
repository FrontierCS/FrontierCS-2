from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from preview import acceptance as parity
from preview import acceptance as references


class ReferenceReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.marker = Path(self.temp.name) / "launcher.json"

    def start_driver(self, statuses):
        calls = []
        remaining = list(statuses)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append((self.path, self.headers.get("Authorization")))
                status = remaining.pop(0) if remaining else 500
                if status is None:
                    # A real connection closes before any response headers.
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                body = b"private-response-body-must-not-be-logged"
                self.send_response(status)
                self.send_header(
                    "Content-Length", str(len(body) if status != 204 else 0)
                )
                if status == 302:
                    self.send_header("Location", "/secret-redirect")
                self.end_headers()
                if status != 204:
                    try:
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()

        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(cleanup)
        return f"http://127.0.0.1:{server.server_port}/driver", calls

    def run_guard(self, url):
        target = (
            "import json,os,sys; from pathlib import Path; "
            "Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'args':sys.argv[2:]}))"
        )
        env = {
            **os.environ,
            "FRONTIER_DRIVER_API_URL": url,
            "FRONTIER_DRIVER_TOKEN_FILE": "/must-not-be-read/private-token",
        }
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                references.REFERENCE_READINESS_GUARD,
                sys.executable,
                "-c",
                target,
                str(self.marker),
                "forwarded-argument",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            self.fail("readiness guard did not complete the bounded test scenario")
        output = stdout + stderr
        for forbidden in (url, "private-token", "private-response-body", "Traceback"):
            if forbidden:
                self.assertNotIn(forbidden, output)
        return process, stdout, stderr

    def test_http_503_then_204_executes_once_after_readiness(self):
        url, calls = self.start_driver([503, 503, 204])
        process, stdout, stderr = self.run_guard(url)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(calls, [("/driver/readyz", None)] * 3)
        self.assertEqual(
            json.loads(self.marker.read_text()),
            {
                "pid": process.pid,
                "args": ["forwarded-argument"],
            },
        )
        self.assertEqual(
            stdout.splitlines(),
            [
                "[frontier-reference] Driver not ready (503); waiting",
                "[frontier-reference] Driver ready (204); starting launcher",
            ],
        )
        self.assertEqual(stderr, "")

    def test_transient_connection_close_retries_before_exec(self):
        url, calls = self.start_driver([None, 204])
        process, stdout, stderr = self.run_guard(url)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads(self.marker.read_text())["pid"], process.pid)
        self.assertEqual(
            stdout.splitlines(),
            ["[frontier-reference] Driver ready (204); starting launcher"],
        )

    def test_other_http_statuses_fail_without_executing_or_following_redirects(self):
        for status in (200, 302, 401, 404, 500):
            with self.subTest(status=status):
                url, calls = self.start_driver([status, 204])
                process, stdout, stderr = self.run_guard(url)
                self.assertEqual(process.returncode, 1)
                self.assertEqual(len(calls), 1)
                self.assertFalse(self.marker.exists())
                self.assertEqual(stdout, "")
                self.assertEqual(
                    stderr.strip(),
                    f"[frontier-reference] Driver readiness returned unexpected HTTP {status}",
                )

    def test_missing_or_malformed_url_fails_without_exposing_it(self):
        for url in (
            "",
            "http://private-token@private-host",
            "http://private-host:not-a-port",
        ):
            with self.subTest(url_kind="missing" if not url else "malformed"):
                process, stdout, stderr = self.run_guard(url)
                self.assertEqual(process.returncode, 1)
                self.assertFalse(self.marker.exists())
                self.assertEqual(stdout, "")
                self.assertEqual(
                    stderr.strip(),
                    "[frontier-reference] Driver readiness URL is missing or invalid",
                )

    def test_parity_uses_shared_model_free_execution_without_private_environment(self):
        execution = references.REFERENCE_EXECUTION
        self.assertIs(parity.EXECUTION, execution)
        self.assertEqual(
            execution,
            {
                "entrypoint": [
                    "python3",
                    "-c",
                    references.REFERENCE_READINESS_GUARD,
                    "/opt/frontier/reference-candidate",
                ],
                "args": [],
                "working_dir": "/workspace",
                "user": "10001:10001",
                "env": {},
            },
        )


if __name__ == "__main__":
    unittest.main()
