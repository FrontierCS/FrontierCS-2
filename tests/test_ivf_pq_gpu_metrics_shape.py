"""ivf_pq_gpu_kernel_optimization's metrics pass the shared verifier's result path unchanged.

The shared verifier bounds evaluator metrics (fcs2_verifier._bounded: nesting depth 6). The
judge used to nest every timed sample's distance-error distribution under
comparison_details[].{reference,candidate}.timed_samples[], eight levels deep, so every real
GPU verification failed invalid_evaluator_result "metrics are nested too deeply" (registrar,
an earlier task version). Unit tests of evaluate() alone did not see it.

This runs fcs2_verifier.verify() end to end (payload, edit scope, check_tree, evaluate, bounds,
result and evidence) on worker rows recorded by a real H100 train verification of the
pre-migration judge; only the GPU worker processes and the SIFT file check are replaced by
those rows. The Agent-visible train metrics must equal what that verification returned.
"""
import deployment_test_guard  # noqa: F401 # isort: skip

import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/MLSys/ivf_pq_gpu_kernel_optimization"
FIXTURE = ROOT / "tests/fixtures/ivf-pq-recorded-train/worker-rows.json"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tasks/shared/verifier"))

import fcs2_verifier  # noqa: E402
import task_manifest  # noqa: E402

try:
    import zstandard
except ImportError:  # pragma: no cover - the repo tool Python provides it
    zstandard = None


def depth(value, level=0):
    if isinstance(value, dict):
        return max([depth(item, level + 1) for item in value.values()] or [level])
    if isinstance(value, list):
        return max([depth(item, level + 1) for item in value] or [level])
    return level


@unittest.skipIf(zstandard is None, "the shared verifier writes zstd evidence")
class IvfPqMetricsShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recorded = json.loads(FIXTURE.read_text())
        cls.bundle = task_manifest.judge_bundle(TASK)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)
        payload = self.work / "payload.tar"
        with tarfile.open(payload, "w") as archive:
            info = tarfile.TarInfo("payload/solution.patch")  # empty patch: allowed, the starter
            archive.addfile(info, io.BytesIO(b""))
        self.payload = payload

    def rows(self, workload_ids):
        """The recorded rows, relabelled in order when the suite has more workloads (eval)."""
        recorded = (self.recorded["reference_rows"], self.recorded["candidate_rows"])
        return [[{**rows[i % len(rows)], "id": wid} for i, wid in enumerate(workload_ids)]
                for rows in recorded]

    def verify(self, mode):
        original = fcs2_verifier.load_evaluator

        def load_evaluator(task_dir):
            evaluate, check_tree = original(task_dir)
            judge = evaluate.__wrapped__.__globals__

            def measure(source_root, config):
                reference, candidate = self.rows([w["id"] for w in config["workloads"]])
                return reference if source_root.name == "reference" else candidate

            self.enterContext(mock.patch.dict(judge, {"_measure": measure}))
            self.enterContext(mock.patch.object(judge["ivfpq_worker"], "verified_data_path",
                                                return_value=None))
            return evaluate, check_tree

        result_path, evidence_path = self.work / f"{mode}.json", self.work / f"{mode}.tar.zst"
        with mock.patch.object(fcs2_verifier, "load_evaluator", load_evaluator), \
                mock.patch.dict(os.environ, {"FRONTIER_VERIFIER_SCRATCH": str(self.work)}):
            fcs2_verifier.verify(bundle=self.bundle, entrypoint="payload/solution.patch",
                                 payload=self.payload, dataset=TASK / f"datasets/{mode}.json",
                                 mode=mode, result_path=result_path, evidence_path=evidence_path)
        result = json.loads(result_path.read_text())
        raw = zstandard.ZstdDecompressor().decompress(evidence_path.read_bytes(),
                                                      max_output_size=1 << 26)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            evidence = json.load(archive.extractfile("metrics.json"))
        return result, evidence

    def test_train_result_equals_the_recorded_verification(self):
        result, evidence = self.verify("train")
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["raw_score"], self.recorded["raw_score"])
        self.assertEqual(result["metrics"], self.recorded["result_metrics"])
        metrics = evidence["metrics"]
        self.assertEqual(metrics["beat_spec_workload_passes"],
                         self.recorded["beat_spec_workload_passes"])
        self.assertEqual(metrics["beats_reference"], self.recorded["beats_reference"])
        self.assertEqual(metrics["reference_score"], self.recorded["reference_score"])
        # Every timed sample of both sides is kept, flat.
        self.assertEqual(len(metrics["timed_samples"]), 2 * 4 * 7)
        self.assertLessEqual(depth(metrics), 6)

    def test_eval_result_has_the_private_verdict(self):
        result, evidence = self.verify("eval")
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["metrics"], {"score_unbounded": result["raw_score"]})
        private = result["private_metrics"]
        self.assertEqual(private["beats_reference"], evidence["metrics"]["beats_reference"])
        self.assertEqual(sum(row["total"] for row in private["beat_spec_progress"].values()), 3 * 8)
        self.assertLessEqual(depth(evidence["metrics"]), 6)


if __name__ == "__main__":
    unittest.main()
