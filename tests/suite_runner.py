"""Run directories of per-task pytest files, each in its own process, as `make test` does for
tests/maintenance/<task> (tests/test_maintenance_suites.py).

The files import a task's judge modules under their generic names (`evaluator`, ...), so each
directory runs in its own pytest process; a bounded pool runs the processes in parallel. The
shared verifier and judges import the shared launcher (tasks/shared/lib), which the Verifier
image puts on the default import path. Each process has a private scratch directory, and no
suite may write into its task's tree.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))


def listing(task: Path) -> list[str]:
    return sorted(str(path.relative_to(task)) for path in task.rglob("*"))


def run_suite(suite: Path) -> subprocess.CompletedProcess:
    # A short scratch path: task codecs keep the file names they are given in fixed buffers.
    with tempfile.TemporaryDirectory(prefix="suite-") as scratch:
        environment = {
            **os.environ,
            "TMPDIR": scratch,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT / "tests"), str(ROOT / "tasks/shared/lib")]
            ),
        }
        environment.pop("FRONTIER_SUBMISSION_ROLE", None)
        return subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
            + [str(suite)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=3000,
        )


def assert_suites_pass(case: unittest.TestCase, suites: dict[Path, Path]) -> None:
    """SUITES maps each suite directory to its task tree; one subtest per suite."""
    case.assertTrue(suites)
    before = {suite: listing(task) for suite, task in suites.items()}
    with ThreadPoolExecutor(WORKERS) as pool:
        results = dict(zip(suites, pool.map(run_suite, suites), strict=True))
    for suite, task in suites.items():
        with case.subTest(suite=suite.name):
            result = results[suite]
            case.assertEqual(
                result.returncode, 0, (result.stdout + result.stderr)[-6000:]
            )
            case.assertEqual(
                before[suite],
                listing(task),
                "tests must not write into the published task tree",
            )
