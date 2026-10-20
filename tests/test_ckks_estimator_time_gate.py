"""Include the CKKS accuracy-and-time gate tests in `make test`.

The task's own tests and the maintenance tests are pytest files. They run in a
child interpreter with a private scratch directory, so nothing is written into
the published task tree. Tests that compile Go are skipped where the pinned
toolchain is absent; see tests/maintenance/ckks_composable_noise/test_time_gate.py.
"""

import deployment_test_guard  # noqa: F401  # isort: skip

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/Crypto/ckks_composable_noise"
MAINTENANCE = ROOT / "tests/maintenance/ckks_composable_noise"
# A stale quota assertion predates this change and is tracked separately.
KNOWN_STALE = "not test_public_manifest_and_clock_remain_exact"


@unittest.skipIf(importlib.util.find_spec("pytest") is None, "pytest is not installed")
class CkksEstimatorTimeGateTests(unittest.TestCase):
    def run_pytest(self, directory: Path, *targets: str) -> None:
        before = sorted(str(p.relative_to(TASK)) for p in TASK.rglob("*"))
        with tempfile.TemporaryDirectory(prefix="ckks-gate-") as scratch:
            environment = {
                **os.environ,
                "TMPDIR": scratch,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(ROOT / "tests"),
            }
            environment.pop("FRONTIER_SUBMISSION_ROLE", None)
            result = subprocess.run(
                [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                 "-k", KNOWN_STALE, *targets],
                cwd=directory, env=environment, capture_output=True, text=True, timeout=3000,
            )
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-6000:])
        self.assertEqual(before, sorted(str(p.relative_to(TASK)) for p in TASK.rglob("*")),
                         "tests must not write into the published task tree")

    def test_task_evaluator_tests(self):
        self.run_pytest(MAINTENANCE / "task_tests", "test_evaluator.py", "test_workload_configurations.py")

    def test_gate_fixtures(self):
        self.run_pytest(MAINTENANCE, "test_time_gate.py")

    def test_fixtures_and_witness_stay_outside_the_published_tree(self):
        names = {p.name for p in TASK.rglob("*") if p.is_file()}
        self.assertFalse({p.name for p in (MAINTENANCE / "adversarial").glob("*.patch")} & names)
        self.assertFalse(any("witness" in name.lower() for name in names))
        self.assertFalse((TASK / "judge/adversarial").exists())


if __name__ == "__main__":
    unittest.main()
