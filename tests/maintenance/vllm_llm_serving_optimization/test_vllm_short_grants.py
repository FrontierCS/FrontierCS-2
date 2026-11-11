"""Run task transport regressions in an isolated import namespace."""

import deployment_test_guard  # noqa: F401

import os
import subprocess
import sys
import unittest
from pathlib import Path


class NativeShortGrantTests(unittest.TestCase):
    def test_short_grants_and_bounded_control_diagnostics(self):
        path = Path(__file__).parent / "check_short_grants.py"
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[2]))
        result = subprocess.run(
            [sys.executable, "-B", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
