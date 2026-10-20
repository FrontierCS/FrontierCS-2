"""Population train and final GPU requests follow the published clock contract."""

import deployment_test_guard  # noqa: F401

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/Machine Learning/representation_frechet_loss"


class PopulationVerifierClockTests(unittest.TestCase):
    def adapter(self):
        spec = importlib.util.spec_from_file_location(
            "population_mode_clocks", TASK / "verifier/fcs2_verifier_gpu.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def environment(self):
        return {
            "FRONTIER_DRIVER_INTERNAL_URL": "http://driver.test:8081",
            "FRONTIER_VERIFICATION_ID": "verification_01M4K4RJWS29T6VDKVHPRJ14JD",
            "FRONTIER_VERIFIER_TOKEN": "test-scoped-verifier-token",
            "FRONTIER_VERIFIER_GPU_TYPE": "A100-80GB",
            "FRONTIER_VERIFIER_GPU_SECONDS": "7200",
            "FRONTIER_VERIFIER_GPU_EVAL_SECONDS": "10800",
            "FRONTIER_VERIFIER_GPU_COUNT": "1",
            "FRONTIER_MAX_EVIDENCE_BYTES": "1048576",
        }

    def test_actual_create_requests_and_remote_remaining_grant(self):
        adapter = self.adapter()
        for mode, seconds in (("train", 7200), ("eval", 10800)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                runner = {"runner_id": "gpurunner_test", "granted_seconds": seconds - 37}
                with (
                    mock.patch.dict(os.environ, self.environment(), clear=True),
                    mock.patch.object(adapter.time, "monotonic", return_value=100),
                    mock.patch.object(adapter, "_input_archive"),
                    mock.patch.object(adapter, "_create_runner", return_value=runner) as create,
                    mock.patch.object(adapter, "_ssh_argv", return_value=["ssh"]),
                    mock.patch.object(adapter, "_ssh_environment", return_value={}),
                    mock.patch.object(adapter, "_upload_inputs"),
                    mock.patch.object(adapter, "_run_remote_verifier", return_value=(0, False)) as execute,
                    mock.patch.object(adapter, "_copy_outputs"),
                    mock.patch.object(adapter, "_terminate_runner", return_value=True),
                ):
                    adapter.run(payload=root / "payload", dataset=root / "dataset", mode=mode,
                                result_path=root / "result", evidence_path=root / "evidence")
                    self.assertEqual(create.call_args.args[4], seconds)
                    command = execute.call_args.args[1]
                    self.assertIn(f"FRONTIER_VERIFICATION_MODE={mode}", command)
                    self.assertIn(f"FRONTIER_VERIFIER_REMAINING_SECONDS={seconds - 38:.6f}", command)
                    self.assertEqual(execute.call_args.args[3], seconds - 38)

    def test_eval_clock_is_required_and_never_falls_back_to_train(self):
        adapter = self.adapter()
        for invalid in (None, "", "0", "-1", "true", "604801"):
            env = self.environment()
            if invalid is None:
                del env["FRONTIER_VERIFIER_GPU_EVAL_SECONDS"]
            else:
                env["FRONTIER_VERIFIER_GPU_EVAL_SECONDS"] = invalid
            with self.subTest(value=invalid), mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(adapter._config("train")[4], 7200)
                with self.assertRaises(adapter.VerifierGpuError):
                    adapter._config("eval")

    def test_runtime_env_docs_and_policy_share_the_same_clocks(self):
        policy = json.loads((TASK / "policy.json").read_text())
        self.assertEqual(policy["quota"]["wall_time_seconds"], 18000)
        self.assertEqual(policy["quota"]["max_train_calls"], 3)
        self.assertEqual(policy["quota"]["max_train_wall_seconds"], 21600)
        self.assertEqual(policy["quota"]["eval_timeout_seconds"], 10800)
        self.assertEqual(policy["gpu"]["verifier"]["max_gpu_seconds"], 32400)
        for role in ("verifier", "verifier-gpu"):
            recipe = (TASK / role / "Dockerfile").read_text()
            self.assertIn("FRONTIER_VERIFIER_GPU_SECONDS=7200", recipe)
            self.assertIn("FRONTIER_VERIFIER_GPU_EVAL_SECONDS=10800", recipe)
        for name in ("README.md", "repo/README.md"):
            document = (TASK / "agent-base/frontier/task" / name).read_text()
            self.assertIn("18,000", document)
            self.assertIn("7,200-second limit", document)
            self.assertIn("21,600", document)
            self.assertIn("10,800-second limit", document)
            self.assertNotIn("28,800", document)
        self.assertIn("32,400", (TASK / "agent-base/frontier/task/repo/README.md").read_text())
