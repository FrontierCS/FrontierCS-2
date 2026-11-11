"""Native failure evidence observes the executed side, exit code and cgroup."""

import deployment_test_guard  # noqa: F401 # isort: skip

import ast
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TASK = Path(__file__).resolve().parents[1] / ("tasks/problems/Arch/clusterwise_spgemm")


def helpers():
    source = TASK / "judge/trusted_runner.py"
    names = {
        "NativeProcessFailure",
        "_resource_snapshot",
        "_native_process_diagnostic",
        "remaining",
        "sample",
        "_sample",
    }
    definitions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "Path": Path,
        "shutil": shutil,
        "os": os,
        "sys": sys,
        "signal": signal,
        "subprocess": subprocess,
        "tempfile": tempfile,
        "time": time,
        "isolate": FakeSandbox,
        **dict.fromkeys(("Pinned", "AddressSpace", "OutputsDir"), lambda *args: None),
    }
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


class FakeSandbox:
    """isolate.run without the boundary: runs argv[0] and reports its exit like Result."""

    @staticmethod
    def run(argv, **kwargs):
        process = subprocess.run(
            [argv[0]], stdin=subprocess.DEVNULL, capture_output=True
        )
        return SimpleNamespace(
            returncode=process.returncode,
            limit=None,
            timed_out=False,
            command_seconds=0.01,
            outputs={"product": None},
        )


class ClusterwiseNativeDiagnosticTests(unittest.TestCase):
    def test_actual_native_exit_binds_side_binary_and_counter_observations(self):
        namespace = helpers()
        for side in ("candidate", "reference"):
            with self.subTest(side=side), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                group = root / "cgroup"
                group.mkdir()
                (group / "memory.events").write_text("oom 0\noom_kill 0\n")
                (group / "memory.peak").write_text("1234")
                (group / "memory.current").write_text("123")
                binary = root / "native"
                # The fixture increments a synthetic counter; no actual OOM is inferred.
                binary.write_text(
                    "#!/usr/bin/python3\nfrom pathlib import Path\n"
                    f"Path({str(group / 'memory.events')!r}).write_text('oom 1\\noom_kill 1\\n')\n"
                    "raise SystemExit(137)\n"
                )
                binary.chmod(0o755)
                context = SimpleNamespace(
                    cgroup=group,
                    threads=60,
                    cpus=[0],
                    memory_check=lambda *args: None,
                    environment=lambda: {},
                )
                stream = io.StringIO()
                with (
                    patch.dict(
                        namespace,
                        {
                            "require_disk_capture": lambda file: None,
                            "CONFIG": {"environment": {"memory_mb": 65536}},
                            "LIMITS": SimpleNamespace(
                                child_address_space_bytes=48 * 1024**3,
                                process_cpu_seconds=10800,
                                process_wall_seconds=180,
                                max_product_bytes=7516192768,
                            ),
                        },
                    ),
                    patch("sys.stderr", stream),
                ):
                    with self.assertRaises(namespace["NativeProcessFailure"]) as caught:
                        namespace["sample"](
                            binary,
                            root / "matrix",
                            root / "order",
                            None,
                            context,
                            time.monotonic() + 10,
                            root,
                            side=side,
                        )
                error = caught.exception
                self.assertEqual(error.process_returncode, 137)
                self.assertIn("returncode=137", str(error))
                diagnostic = error.native_diagnostic
                self.assertEqual(diagnostic["side"], side)
                self.assertEqual(
                    diagnostic["binary_sha256"],
                    "sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    diagnostic["resource_before"]["memory_events"]["oom_kill"], 0
                )
                self.assertEqual(
                    diagnostic["resource_after"]["memory_events"]["oom_kill"], 1
                )
                self.assertEqual(
                    diagnostic["limits"]["verifier_memory_bytes"], 64 * 1024**3
                )
                self.assertEqual(
                    json.loads(stream.getvalue().split(": ", 1)[1]), diagnostic
                )

    def test_missing_cgroup_counters_are_explicitly_unobserved(self):
        namespace = helpers()
        with tempfile.TemporaryDirectory() as directory:
            result = namespace["_resource_snapshot"](
                SimpleNamespace(cgroup=Path(directory))
            )
        self.assertFalse(result["observed"])
        self.assertNotIn("memory_events", result)

    def test_unbound_sample_side_is_rejected_before_starting_a_process(self):
        namespace = helpers()
        with self.assertRaisesRegex(ValueError, "Unknown trusted sample side"):
            namespace["sample"](None, None, None, None, None, 0, None, side="solver")


if __name__ == "__main__":
    unittest.main()
