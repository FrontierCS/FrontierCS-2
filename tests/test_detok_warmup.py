"""detok_latent_denoising: one untimed warm-up process runs before any timed
process, so the first timed measurement is not cold (issue #315)."""

import deployment_test_guard  # noqa: F401  # isort: skip

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TASK = Path(os.environ.get("FRONTIER_DETOK_EXPORT", str(
    ROOT / "tasks/frontier-cs-2.0-demo/problems/Machine Learning/detok_latent_denoising"
)))
JUDGE = TASK / "judge"


def load_evaluator():
    sys.path.insert(0, str(JUDGE))
    try:
        spec = importlib.util.spec_from_file_location("detok_evaluator", JUDGE / "evaluator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path and sys.path[0] == str(JUDGE):
            sys.path.pop(0)


class DetokWarmup(unittest.TestCase):
    def test_untimed_warmup_runs_first_and_is_discarded(self):
        evaluator = load_evaluator()
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        workdir = Path(scratch.name) / "repo"
        workdir.mkdir()
        suite = Path(scratch.name) / "suite"
        suite.mkdir()
        for i in range(3):
            (suite / f"workload-{i}").mkdir()
        workloads = sorted(p for p in suite.iterdir() if p.is_dir())

        calls = []

        def fake_run(tree, workload, output):
            calls.append({"workload": Path(workload).name, "output": Path(output).name})
            return {k: 10.0 for k in evaluator.SPEC_METRICS}

        applied = types.SimpleNamespace(ok=True, workdir=str(workdir), message="")
        with mock.patch.object(evaluator.yaml, "safe_load", return_value={}), \
                mock.patch.object(evaluator, "apply_submission_patch", return_value=applied), \
                mock.patch.object(evaluator, "_suite_root", return_value=suite), \
                mock.patch.object(evaluator.shutil, "copytree"), \
                mock.patch.object(evaluator.subprocess, "run"), \
                mock.patch.object(evaluator, "_run", side_effect=fake_run):
            evaluator.evaluate("ignored-solution-path")

        # 1 untimed warm-up + 3 candidate + 3 reference timed processes.
        self.assertEqual(len(calls), 7)
        # The warm-up is first, targets the first workload, and its output dir is
        # the discarded "warmup-output" (never one of the timed/scored outputs).
        self.assertEqual(calls[0]["output"], "warmup-output")
        self.assertEqual(calls[0]["workload"], workloads[0].name)
        self.assertEqual(sum(c["output"] == "warmup-output" for c in calls), 1)
        self.assertNotIn("warmup-output", [c["output"] for c in calls[1:]])


if __name__ == "__main__":
    unittest.main()
