"""tasks/template/Machine Learning/sandbox_template_task stays a valid, working task.

The command-sandbox template: its judge runs on the control image and leases two
[test_sandboxes.*] machines per verification through the shared GPU adapter's run_sandboxes().
These cases run the shared verifier exactly as the control image does, on the adapter test
harness of tests/test_shared_gpu_adapter.py (a fake Driver, and a fake SSH gateway that runs
each remote command locally with /workspace mapped to a per-runner directory), so the real
upload, step, timeout and collect commands run: class 1 runs the candidate's fit on its "own
machine", class 2 runs the judge's timing harness, which runs every predict through the shared
isolate launcher. Only the runner image's interpreter path is the test's own (PYTHON).
"""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_shared_gpu_adapter import AdapterCase, task_manifest
from test_template_task import payload

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import validate_task_layout  # noqa: E402

TEMPLATE = REPO / "tasks/template/Machine Learning/sandbox_template_task"
FIT = (TEMPLATE / "repo/model/fit.py").read_text()


def patch_of(text: str) -> str:
    """A patch that replaces model/fit.py with `text`."""
    return "".join(
        difflib.unified_diff(
            FIT.splitlines(keepends=True),
            text.splitlines(keepends=True),
            "a/model/fit.py",
            "b/model/fit.py",
        )
    )


REFERENCE = (TEMPLATE / "judge/reference.patch").read_text()
REFERENCE_FIT = None  # model/fit.py of the reference, built in setUpModule


def setUpModule():
    global REFERENCE_FIT
    scratch = Path(tempfile.mkdtemp(prefix="sandbox-template-ref-"))
    try:
        shutil.copytree(TEMPLATE / "repo", scratch / "repo")
        subprocess.run(
            ["git", "apply", str(TEMPLATE / "judge/reference.patch")],
            cwd=scratch / "repo",
            check=True,
            capture_output=True,
            env={**os.environ, "GIT_CEILING_DIRECTORIES": str(scratch)},
        )
        REFERENCE_FIT = (scratch / "repo/model/fit.py").read_text()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


class SandboxTemplate(AdapterCase):
    def setUp(self):
        super().setUp()
        generated = task_manifest.generated_files(TEMPLATE)
        self.parameters.write_bytes(
            generated[".frontier-generated/verifier/verifier.json"]
        )
        self.bundle = self.scratch / "task"
        shutil.copytree(task_manifest.judge_bundle(TEMPLATE), self.bundle)
        # The timing machine's interpreter imports isolate from frontier.pth in the Verifier
        # GPU image; the fake runner is this host, so the step inherits this import path.
        environment = mock.patch.dict(
            os.environ, {"PYTHONPATH": str(REPO / "tasks/shared/lib")}
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.judge_globals = {"PYTHON": sys.executable}
        original = self.verifier.load_evaluator

        def load_evaluator(task_dir):
            evaluate, check_tree = original(task_dir)
            evaluate.__wrapped__.__globals__.update(self.judge_globals)
            return evaluate, check_tree

        loader = mock.patch.object(self.verifier, "load_evaluator", load_evaluator)
        loader.start()
        self.addCleanup(loader.stop)

    def verify(self, patch: str, mode: str) -> dict:
        os.environ["FRONTIER_VERIFICATION_MODE"] = mode
        work = Path(tempfile.mkdtemp(dir=self.scratch))
        result = work / "result.json"
        self.verifier.verify(
            bundle=self.bundle,
            entrypoint="payload/solution.patch",
            payload=payload(patch, work),
            dataset=TEMPLATE / "datasets" / f"{mode}.json",
            mode=mode,
            result_path=result,
            evidence_path=work / "evidence.tar.zst",
        )
        return json.loads(result.read_text())

    def assertScoredZero(self, patch: str, reason: str, modes=("train", "eval")):
        for mode in modes:
            result = self.verify(patch, mode)
            self.assertEqual(
                (result["status"], result["raw_score"]), ("passed", 0.0), (mode, result)
            )
            self.assertIn(reason, result["summary"])
        if "eval" in modes:
            self.assertEqual(result["private_metrics"]["beats_reference"], 0)

    def test_layout_is_valid(self):
        self.assertEqual(
            validate_task_layout.validate_task(TEMPLATE), "patch-workspace"
        )

    def test_the_two_classes_are_declared_as_the_judge_uses_them(self):
        task = task_manifest.read_task(TEMPLATE)
        parameters = task_manifest.verifier_parameters(task)
        self.assertEqual(parameters["gpu"]["placement"], "control")
        self.assertEqual(
            {
                name: (s["image"], s["max_concurrent"])
                for name, s in parameters["sandboxes"].items()
            },
            {"candidate": ("agent_gpu", 2), "timing": ("verifier_gpu", 1)},
        )
        self.assertEqual(
            task_manifest.policy_from_task(task)["gpu"]["verifier"],
            {"max_gpu_instances": 3, "max_gpu_seconds": 10800},
        )
        self.assertNotIn(
            "sidecars", task
        )  # command sandboxes are never platform sidecars

    def test_reference_scores_fifty_on_three_machines_per_verification(self):
        train = self.verify(REFERENCE, "train")
        self.assertEqual((train["status"], train["raw_score"]), ("passed", 50.0), train)
        self.assertEqual(
            set(train["metrics"]), {"score_unbounded", "beat_spec_progress", "mse"}
        )
        self.assertEqual(len(train["metrics"]["mse"]), 3)
        self.assertEqual(train["metrics"]["beat_spec_progress"]["mse"]["total"], 3)
        self.assertEqual(
            sorted(c["image"] for c in self.driver.creates),
            ["agent_gpu", "agent_gpu", "verifier_gpu"],
        )
        self.assertEqual(len(self.driver.deleted), 3)
        final = self.verify(REFERENCE, "eval")
        self.assertEqual(final["raw_score"], 50.0)
        self.assertEqual(
            final["private_metrics"]["beat_spec_progress"]["mse"]["total"], 5
        )
        self.assertEqual(
            final["private_metrics"]["beats_reference"], 0
        )  # ties do not beat it

    def test_a_better_fit_scores_above_fifty_and_beats_the_reference(self):
        better = REFERENCE_FIT.replace("DEGREE = 3", "DEGREE = 9")
        final = self.verify(patch_of(better), "eval")
        self.assertGreater(final["raw_score"], 60.0, final)
        self.assertEqual(final["private_metrics"]["beats_reference"], 1)

    def test_the_starter_scores_below_fifty(self):
        result = self.verify(
            patch_of(
                FIT.replace("Starter: a straight", "Starter, as shipped: a straight")
            ),
            "train",
        )
        self.assertLess(result["raw_score"], 50.0)

    def test_a_crashing_fit_is_a_scored_zero(self):
        self.assertScoredZero(
            patch_of(FIT.replace("    n = len(xs)\n", "    raise ValueError('no')\n")),
            "the fit exited with status 1",
        )

    def test_malformed_predictions_are_a_scored_zero(self):
        self.assertScoredZero(
            patch_of(
                FIT.replace(
                    'return [params["slope"] * x + params["intercept"] for x in xs]',
                    'return [float("nan") for x in xs]',
                )
            ),
            "not one finite number per query",
        )

    def test_a_fit_past_its_limit_is_a_scored_zero(self):
        self.judge_globals["FIT_SECONDS"] = 2
        self.assertScoredZero(
            patch_of(
                FIT.replace("    n = len(xs)\n", "    while True:\n        pass\n")
            ),
            "exceeded its 2-second limit",
            modes=("train",),
        )

    def test_a_predict_that_fails_in_isolate_on_the_timing_machine_is_a_scored_zero(
        self,
    ):
        # isolate builds the instance's environment from nothing: no FRONTIER_* variable reaches
        # it, while the candidate's own machine has its step deadline.
        needs_runner = FIT.replace(
            '    return [params["slope"]',
            '    import os\n    os.environ["FRONTIER_VERIFIER_LOCAL_DEADLINE"]\n'
            '    return [params["slope"]',
        )
        self.assertScoredZero(
            patch_of(needs_runner), "on the timing machine", modes=("train",)
        )

    def test_a_failing_reference_is_not_a_score(self):
        (self.bundle / "reference.patch").chmod(0o644)
        (self.bundle / "reference.patch").write_text(
            patch_of(FIT.replace("    n = len(xs)\n", "    raise ValueError('no')\n"))
        )
        result = self.verify(REFERENCE, "train")
        self.assertEqual(result["status"], "failed", result)
        self.assertNotIn("raw_score", result)

    def test_a_lost_machine_is_an_unscored_gpu_failure(self):
        self.driver.fail_scope_suffix = 1
        result = self.verify(REFERENCE, "train")
        self.assertEqual(
            (result["status"], result["failure"]["code"]),
            ("failed", "verifier_gpu_error"),
        )


if __name__ == "__main__":
    unittest.main()
