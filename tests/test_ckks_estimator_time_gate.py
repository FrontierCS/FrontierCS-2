"""The CKKS accuracy-and-time gate's adversarial fixtures and witness stay out of the task.

The gate's own tests are the maintenance suite tests/maintenance/ckks_composable_noise, which
tests/test_maintenance_suites.py runs.
"""

import deployment_test_guard  # noqa: F401  # isort: skip

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/problems/Crypto/ckks_composable_noise"
MAINTENANCE = ROOT / "tests/maintenance/ckks_composable_noise"


class CkksEstimatorTimeGateTests(unittest.TestCase):
    def test_fixtures_and_witness_stay_outside_the_published_tree(self):
        names = {p.name for p in TASK.rglob("*") if p.is_file()}
        self.assertFalse(
            {p.name for p in (MAINTENANCE / "adversarial").glob("*.patch")} & names
        )
        self.assertFalse(any("witness" in name.lower() for name in names))
        self.assertFalse((TASK / "judge/adversarial").exists())


if __name__ == "__main__":
    unittest.main()
