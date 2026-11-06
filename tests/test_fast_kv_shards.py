"""fast_kv_compaction's shards on the trusted command test sandbox [test_sandboxes.shard] (#757):
the declaration, the runs evaluate() asks for and their combination, and shard_run's budget on
the runner. Command and document classification: task_tests/fast_kv_compaction/test_instances.py."""
import deployment_test_guard  # noqa: F401  # isort: skip

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/Machine Learning/fast_kv_compaction"
sys.path.insert(0, str(ROOT / "scripts"))
import task_manifest  # noqa: E402

BUNDLE = task_manifest.judge_bundle(TASK)
sys.path.insert(0, str(BUNDLE))
spec = importlib.util.spec_from_file_location("fast_kv_shards_evaluator", BUNDLE / "evaluator.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def shard_document(index, instances, role, rows_for):
    workloads = ev.shard(ev._workloads(role == "final"), instances)[index]
    return {"shard": index, "instances": instances, "role": role, "status": "ok",
            "result": {"device": "NVIDIA H100", "engine_anchor_seconds": 1.0,
                       "rows": [rows_for(w) for w in workloads]}}


class ShardTests(unittest.TestCase):
    def test_task_declares_the_trusted_shard_sandbox(self):
        task = task_manifest.read_task(TASK)
        self.assertEqual(task["gpu"]["verifier"]["placement"], "control")
        self.assertEqual(task["gpu"]["verifier"]["max_gpu_instances"], max(ev.INSTANCES.values()))
        sandbox = task_manifest.verifier_parameters(task)["gpu"]["sandboxes"][ev.SANDBOX]
        self.assertEqual(sandbox, {"image": "verifier_gpu", "workdir": "/workspace/frontier-verifier",
                                   "resources": {"gpus": 1, "gpu_type": "H100"},
                                   "max_concurrent": max(ev.INSTANCES.values()),
                                   "lease_seconds": {"train": 3600, "eval": 4500}})

    def test_run_harness_asks_one_rest_of_grant_run_per_shard_and_combines(self):
        calls = []

        def run_sandboxes(runs, *, into):
            calls.append(runs)
            results = []
            for index, run in enumerate(runs):
                output = Path(into) / str(index)
                (output / "output").mkdir(parents=True)
                (output / "output/shard.json").write_text(json.dumps(shard_document(
                    index, len(runs), "final", lambda w: {"workload": w, "correct": 1, "questions": 2,
                                                          "seconds": 1.0, "bytes": 10})))
                step = types.SimpleNamespace(exit_code=0, timed_out=False, lost_session=False)
                results.append(types.SimpleNamespace(steps=[step], outputs=output))
            return results

        adapter = types.SimpleNamespace(
            SandboxRun=lambda *a: a, SandboxStep=lambda *a: a, VerifierGpuError=LookupError,
            run_sandboxes=run_sandboxes, sandbox_workdir=lambda name: "/workspace/frontier-verifier")
        with tempfile.TemporaryDirectory() as tree, \
                mock.patch.dict(sys.modules, {"fcs2_verifier_gpu": adapter}):
            result = ev.run_harness(Path(tree), ev._workloads(True), final=True,
                                    timeout=ev.FINAL_TIMEOUT_SECONDS)
        runs = calls[0]
        self.assertEqual(len(runs), 8)
        sandbox, _files, steps, outputs, cap = runs[0]
        self.assertEqual((sandbox, outputs, cap), ("shard", ["output/shard.json"], ev.MAX_SHARD_BYTES))
        argv, seconds, env = steps[0]
        self.assertIsNone(seconds)
        self.assertEqual(argv[-1], str(ev.FINAL_TIMEOUT_SECONDS - ev.RESULT_MARGIN_SECONDS))
        self.assertEqual([row["workload"] for row in result["rows"]], list(ev._workloads(True)))

    def test_shard_run_measures_the_applied_tree_and_writes_one_document(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "out/shard.json"
            seen = {}

            def run_local(tree, workloads, **kwargs):
                seen.update(tree=tree, workloads=workloads, budget=kwargs["budget"])
                return {"device": "NVIDIA H100", "engine_anchor_seconds": 1.0,
                        "rows": [{"workload": w} for w in workloads]}

            with mock.patch.object(ev, "run_local", side_effect=run_local), \
                    mock.patch.dict(os.environ, {"FRONTIER_VERIFIER_LOCAL_DEADLINE": str(ev.time.monotonic() + 3100)}):
                self.assertEqual(ev.shard_run(0, 4, "train", Path("/tree"), output, 3000), 0)
            document = json.loads(output.read_text())
        self.assertEqual(document["status"], "ok")
        self.assertEqual(seen["tree"], Path("/tree"))
        self.assertEqual(seen["workloads"], ev.shard(ev._workloads(False), 4)[0])
        # The step leaves 3100 - 240 s, less than the 3000-s budget: the budget is capped.
        self.assertLessEqual(seen["budget"], 3100 - ev.RESULT_MARGIN_SECONDS + 1)

    def _shard_after_startup(self, role, startup):
        """shard_run on a runner whose lease (task.toml) already lost `startup` seconds to
        provisioning and upload; the budget is the one evaluate() passes. (status, budget)."""
        final = role == "final"
        lease = ev.FINAL_TIMEOUT_SECONDS if final else ev.TRAIN_TIMEOUT_SECONDS
        instances = ev.INSTANCES[role]
        seen = {}

        def run_local(tree, workloads, **kwargs):
            seen["budget"] = kwargs["budget"]
            return {"device": "NVIDIA H100", "engine_anchor_seconds": 1.0,
                    "rows": [{"workload": w} for w in workloads]}

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "shard.json"
            deadline = ev.time.monotonic() + lease - startup
            with mock.patch.object(ev, "run_local", side_effect=run_local), \
                    mock.patch.dict(os.environ, {"FRONTIER_VERIFIER_LOCAL_DEADLINE": str(deadline)}):
                ev.shard_run(0, instances, role, Path("/tree"), output,
                             lease - ev.RESULT_MARGIN_SECONDS)
            return json.loads(output.read_text())["status"], seen.get("budget")

    def test_a_1350_s_runner_startup_caps_the_budget_and_still_runs(self):
        # Registrar runs on gpu-tools v4: the shard started ~1,350 s (train) / ~1,380 s
        # (final) into the lease. That is a smaller budget, never an infrastructure failure.
        for role, startup in (("train", 1350), ("final", 1380)):
            with self.subTest(role=role):
                status, budget = self._shard_after_startup(role, startup)
                lease = ev.FINAL_TIMEOUT_SECONDS if role == "final" else ev.TRAIN_TIMEOUT_SECONDS
                self.assertEqual(status, "ok")
                self.assertLessEqual(budget, lease - startup - ev.RESULT_MARGIN_SECONDS + 1)
                self.assertGreaterEqual(budget, lease - startup - ev.RESULT_MARGIN_SECONDS - 5)
                # Still above the longest historical shard of the role.
                self.assertGreater(budget, 1781 if role == "final" else 640)

    def test_a_lease_below_the_minimal_need_is_an_infrastructure_failure(self):
        for role in ("train", "final"):
            with self.subTest(role=role):
                final = role == "final"
                lease = ev.FINAL_TIMEOUT_SECONDS if final else ev.TRAIN_TIMEOUT_SECONDS
                articles = len(ev.shard(ev._workloads(final), ev.INSTANCES[role])[0])
                needed = ev.minimal_shard_seconds(articles, final)
                left = lease - ev.RESULT_MARGIN_SECONDS - needed
                self.assertEqual(self._shard_after_startup(role, left + 30)[0], "infrastructure")
                self.assertEqual(self._shard_after_startup(role, left - 30)[0], "ok")

    def test_the_minimal_need_is_setup_plus_the_trusted_qa_of_every_cache(self):
        import harness
        self.assertEqual(ev.minimal_shard_seconds(3, False),
                         ev.MIN_SHARD_SETUP_SECONDS + harness.QA_RESERVE_SECONDS * 3 * ev.TRAIN_REPEATS)
        self.assertEqual(ev.minimal_shard_seconds(3, True),
                         ev.MIN_SHARD_SETUP_SECONDS + harness.QA_RESERVE_SECONDS * 3 * ev.FINAL_REPEATS)

    def test_evaluate_runs_on_the_tree_without_the_patch(self):
        # The frozen targets: a failed run still reports the reference score (#940).
        with mock.patch.object(ev, "run_harness", side_effect=RuntimeError("compaction failed: x")):
            score, unbounded, message, metrics = ev.evaluate(str(BUNDLE / "pristine"))
        self.assertEqual((score, unbounded), (0.0, 0.0))
        self.assertEqual(metrics["beats_reference"], 0)


if __name__ == "__main__":
    unittest.main()
