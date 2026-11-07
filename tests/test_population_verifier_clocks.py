"""Population train and final GPU leases follow the published clock contract (gpu-tools v4).

The per-mode runner lease is task.toml [gpu.verifier] lease_seconds; the shared adapter requests
it per verification mode (its own tests cover the request). The pool covers three train
verifications and one final.
"""

import deployment_test_guard  # noqa: F401

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/Machine Learning/representation_frechet_loss"
sys.path.insert(0, str(ROOT / "scripts"))
import task_manifest  # noqa: E402


class PopulationVerifierClockTests(unittest.TestCase):
    def test_lease_per_mode_and_pool(self):
        task = task_manifest.read_task(TASK)
        lease = task["gpu"]["verifier"]["lease_seconds"]
        self.assertEqual(lease, {"train": 7200, "eval": 10800})
        self.assertEqual(task["quota"]["eval_timeout_seconds"], lease["eval"])
        self.assertEqual(task["gpu"]["verifier"]["max_gpu_seconds"],
                         task["quota"]["max_train_calls"] * lease["train"] + lease["eval"])

    def test_judge_suite_timeouts_match_the_leases(self):
        import json
        runtime = json.loads((task_manifest.judge_bundle(TASK) / "config.yaml").read_text())["runtime"]
        self.assertEqual((runtime["timeout_seconds"], runtime["final_timeout_seconds"]), (7200, 10800))


if __name__ == "__main__":
    unittest.main()
