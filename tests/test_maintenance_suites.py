"""Run the per-task maintenance suites, tests/maintenance/<task>, in `make test`."""

import deployment_test_guard  # noqa: F401  # isort: skip

import unittest

from suite_runner import ROOT, assert_suites_pass

MAINTENANCE = ROOT / "tests/maintenance"


class MaintenanceSuiteTests(unittest.TestCase):
    def test_every_suite_passes_without_writing_into_its_task(self):
        suites = {}
        for suite in sorted(path for path in MAINTENANCE.iterdir() if path.is_dir()):
            (suites[suite],) = (ROOT / "tasks/problems").glob(f"*/{suite.name}")
        assert_suites_pass(self, suites)


if __name__ == "__main__":
    unittest.main()
