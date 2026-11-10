"""tasks/template/MLSys/gpu_template_task stays a valid, working task.

The GPU template: its whole verification runs on one Verifier GPU runner (placement "runner"),
where the shared verifier applies the patch and its judge runs every implementation through the
shared isolate launcher with the GPU. CI has no GPU and no PyTorch, so, as for
ivf_pq_gpu_kernel_optimization (tests/test_ivf_pq_gpu_metrics_shape.py), only the GPU worker
process is replaced: a stand-in for the judge's isolate.run checks each call (the GPU, the
worker as an input, the tree) and answers as judge/worker.py would, from the real inputs the
judge wrote. Everything else (payload, edit scope, the judge's own recompute of the checked
rows, scoring, the result) is the shared verifier's real path. The control side (the shared
verifier shipping the verification to a runner) runs on the adapter test harness of
tests/test_shared_gpu_adapter.py.
"""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import array
import difflib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_shared_gpu_adapter import AdapterCase, load_verifier, task_manifest
from test_template_task import payload

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tasks/shared/lib"))
import isolate  # noqa: E402
import validate_task_layout  # noqa: E402

TEMPLATE = REPO / "tasks/template/MLSys/gpu_template_task"
STARTER = (TEMPLATE / "repo/kernel/distances.py").read_text()
# Small shapes, so the judge's double-precision recompute stays fast in CI.
SUITES = {"PUBLIC": {"a": (24, 20, 8, 11), "b": (16, 28, 4, 12), "c": (8, 12, 16, 13)},
          "HELD_OUT": {"d": (20, 20, 8, 21), "e": (12, 16, 32, 22)}}
REFERENCE = (TEMPLATE / "judge/reference.patch").read_text()


def patch_of(text: str) -> str:
    return "".join(difflib.unified_diff(STARTER.splitlines(keepends=True),
                                        text.splitlines(keepends=True),
                                        "a/kernel/distances.py", "b/kernel/distances.py"))


def stand_in(behaviour: dict):
    """The GPU worker's answer for one implementation (judge/worker.py's contract): its latency
    and the asked rows, exact, shifted or missing as `behaviour` (by kernel source) says."""
    calls = []

    def run(argv, **options):
        calls.append(options)
        assert options["devices"] == ["nvidia"], options
        assert Path(options["inputs"]["worker.py"]).name == "worker.py"
        assert options["outputs"] == ["result.json"]
        source = (Path(options["tree"]) / "kernel/distances.py").read_text()
        latency, shift, fails = next(v for k, v in behaviour.items() if k in source)
        if fails:
            return isolate.Result(returncode=1, stdout=b"", stderr=b"Traceback: boom")
        n, m, d = map(int, argv[5:8])
        rows = json.loads(argv[8])
        x = array.array("f", Path(options["inputs"]["x.bin"]).read_bytes())
        y = array.array("f", Path(options["inputs"]["y.bin"]).read_bytes())
        assert (len(x), len(y)) == (n * d, m * d)
        picked = [[sum((x[i * d + k] - y[j * d + k]) ** 2 for k in range(d)) + shift
                   for j in range(m)] for i in rows]
        body = json.dumps({"latency_ms": latency, "rows": picked}).encode()
        return isolate.Result(returncode=0, stdout=b"", stderr=b"",
                              outputs={"result.json": body}, backend="userns")

    return run, calls


# kernel source marker -> (latency_ms, error added to every entry, the run fails)
BEHAVIOUR = {"stand-in: twice as fast": (0.5, 0.0, False),
             "stand-in: wrong": (0.5, 1.0, False),
             "stand-in: crash": (0.5, 0.0, True),
             "addmm": (1.0, 0.0, False),                  # the reference
             "x[:, None, :]": (4.0, 0.0, False)}          # the starter


REFERENCE_SOURCE = ""


def setUpModule():
    global REFERENCE_SOURCE
    scratch = Path(tempfile.mkdtemp(prefix="gpu-template-ref-"))
    try:
        shutil.copytree(TEMPLATE / "repo", scratch / "repo")
        subprocess.run(["git", "apply", str(TEMPLATE / "judge/reference.patch")],
                       cwd=scratch / "repo", check=True, capture_output=True,
                       env={**os.environ, "GIT_CEILING_DIRECTORIES": str(scratch)})
        REFERENCE_SOURCE = (scratch / "repo/kernel/distances.py").read_text()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


class GpuTemplateJudge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = load_verifier()
        cls.scratch = Path(tempfile.mkdtemp(prefix="gpu-template-test-"))
        cls.bundle = cls.scratch / "task"
        shutil.copytree(task_manifest.judge_bundle(TEMPLATE), cls.bundle)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.scratch, ignore_errors=True)

    def verify(self, patch: str, mode: str, behaviour=BEHAVIOUR) -> tuple[dict, list]:
        run, calls = stand_in(behaviour)
        original = self.verifier.load_evaluator

        def load_evaluator(task_dir):
            evaluate, check_tree = original(task_dir)
            judge = evaluate.__wrapped__.__globals__
            judge.update(SUITES)
            judge["isolate"] = mock.Mock(run=run)
            return evaluate, check_tree

        work = Path(tempfile.mkdtemp(dir=self.scratch))
        with mock.patch.object(self.verifier, "load_evaluator", load_evaluator):
            self.verifier.verify(bundle=self.bundle, entrypoint="payload/solution.patch",
                                 payload=payload(patch, work),
                                 dataset=TEMPLATE / "datasets" / f"{mode}.json", mode=mode,
                                 result_path=work / "result.json",
                                 evidence_path=work / "evidence.tar.zst")
        return json.loads((work / "result.json").read_text()), calls

    def test_layout_is_valid(self):
        self.assertEqual(validate_task_layout.validate_task(TEMPLATE), "patch-workspace")

    def test_the_worker_compiles(self):
        py_compile.compile(str(TEMPLATE / "judge/worker.py"), doraise=True,
                           cfile=str(self.scratch / "worker.pyc"))

    def test_reference_scores_fifty_and_each_run_is_one_gpu_instance(self):
        train, calls = self.verify(REFERENCE, "train")
        self.assertEqual((train["status"], train["raw_score"]), ("passed", 50.0), train)
        self.assertEqual(set(train["metrics"]), {"score_unbounded", "beat_spec_progress",
                                                 "latency_ms"})
        self.assertEqual(train["metrics"]["latency_ms"], [1.0, 1.0, 1.0])
        self.assertEqual(len(calls), 3 * 3 * 2)  # replicates x workloads x implementations
        final, calls = self.verify(REFERENCE, "eval")
        self.assertEqual(final["private_metrics"]["beat_spec_progress"]["latency_ms"]["total"], 5)
        self.assertEqual(final["private_metrics"]["beats_reference"], 0)
        self.assertEqual(len(calls), 3 * 5 * 2)

    def test_twice_the_reference_speed_scores_seventy_five(self):
        faster = REFERENCE_SOURCE + "# stand-in: twice as fast\n"
        final, _ = self.verify(patch_of(faster), "eval")
        self.assertEqual(final["raw_score"], 75.0)
        self.assertEqual(final["private_metrics"]["beats_reference"], 1)

    def test_the_starter_at_a_quarter_of_the_speed_scores_zero(self):
        result, _ = self.verify(patch_of(STARTER + "\n"), "train")
        self.assertEqual(result["raw_score"], 0.0)  # 50 + 25 * log2(1/4)

    def test_an_output_outside_the_tolerance_is_a_scored_zero(self):
        # The judge recomputes the checked rows itself; a fast wrong kernel scores nothing.
        for mode in ("train", "eval"):
            result, _ = self.verify(patch_of(REFERENCE_SOURCE + "# stand-in: wrong\n"), mode)
            self.assertEqual((result["status"], result["raw_score"]), ("passed", 0.0), mode)
            self.assertIn("outside the tolerance", result["summary"])
        self.assertEqual(result["private_metrics"]["beats_reference"], 0)

    def test_a_failing_run_is_a_scored_zero(self):
        result, _ = self.verify(patch_of(REFERENCE_SOURCE + "# stand-in: crash\n"), "train")
        self.assertEqual((result["status"], result["raw_score"]), ("passed", 0.0))
        self.assertIn("run failed", result["summary"])

    def test_a_failing_reference_is_not_a_score(self):
        result, _ = self.verify(patch_of(REFERENCE_SOURCE + "# stand-in: twice as fast\n"),
                                "train", behaviour={**BEHAVIOUR, "addmm": (1.0, 0.0, True)})
        self.assertEqual(result["status"], "failed", result)
        self.assertNotIn("raw_score", result)


class GpuTemplateDefinition(unittest.TestCase):
    def test_the_whole_verification_runs_on_one_runner(self):
        task = task_manifest.read_task(TEMPLATE)
        self.assertEqual(task_manifest.verifier_parameters(task)["gpu"], {
            "placement": "runner", "max_gpu_instances": 1,
            "lease_seconds": {"train": 1800, "eval": 1800}})

    def test_the_agent_readme_states_the_gpu_budget(self):
        readme = task_manifest.agent_readme(TEMPLATE)
        self.assertIn("## Using a GPU", readme)
        self.assertIn("**32 SSH sessions for Agent GPU debugging**", readme)
        self.assertIn("1 at a time, type H100, at most 7200 s each", readme)

    def test_both_verifier_images_name_the_same_gpu_and_the_runner_takes_isolate(self):
        env = {}
        for name in ("verifier", "verifier-gpu"):
            text = (TEMPLATE / f"docker/{name}.Dockerfile").read_text()
            env[name] = re.findall(r"FRONTIER_VERIFIER_GPU_(TYPE|COUNT)=(\S+)", text)
        self.assertEqual(env["verifier"], [("TYPE", "H100"), ("COUNT", "1")])
        self.assertEqual(env["verifier"], env["verifier-gpu"])
        runner = (TEMPLATE / "docker/verifier-gpu.Dockerfile").read_text()
        self.assertIn("COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/isolate.py "
                      "/opt/frontier/isolate.py", runner)
        self.assertIn("install_frontier_pth.py /opt/frontier", runner)
        self.assertEqual(task_manifest.read_task(TEMPLATE)["gpu"]["agent"]["allowed_types"],
                         ["H100"])


class GpuTemplateControl(AdapterCase):
    """The control image only drives the runner: the shared verifier ships the verification."""

    def test_the_control_image_ships_the_verification_to_one_runner(self):
        self.parameters.write_bytes(task_manifest.generated_files(TEMPLATE)[
            ".frontier-generated/verifier/verifier.json"])
        bundle = self.scratch / "task"
        shutil.copytree(task_manifest.judge_bundle(TEMPLATE), bundle)
        out = self.scratch / "out"
        out.mkdir()
        os.environ.update({"FRONTIER_RESULT_PATH": str(out / "result.json"),
                           "FRONTIER_EVIDENCE_PATH": str(out / "evidence.tar.zst"),
                           "FRONTIER_DATASET_PATH": str(TEMPLATE / "datasets/train.json"),
                           "FRONTIER_PAYLOAD_PATH": str(payload(REFERENCE, self.scratch))})
        with mock.patch.object(self.gpu, "run") as run, \
                mock.patch.object(self.verifier, "load_evaluator") as judge:
            self.assertEqual(self.verifier.main(bundle, "payload/solution.patch", sandboxes=False), 0)
        self.assertEqual(run.call_args.kwargs["mode"], "train")
        judge.assert_not_called()  # no judge code runs on the control image
        self.assertEqual(self.gpu.lease_seconds("train"), 1800)


if __name__ == "__main__":
    unittest.main()
