"""Check final transport against the platform's closed metrics contract."""

# Install isolation before runtime imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_task_layout as layout

TASKS = (
    "batch_local_counting",
    "bounded_evm_path_prediction",
    "evm_state_checkpoints",
    "graph_mining_method_selection",
)
CONTRACT = json.loads(
    (ROOT / "tests/fixtures/platform-contracts/run-metrics-v1.schema.json").read_text()
)
PRIVATE_SCHEMA = {"$ref": "#/$defs/PrivateMetrics", "$defs": CONTRACT["$defs"]}


class DatabasePrivateMetrics(unittest.TestCase):
    @patch.object(sys, "dont_write_bytecode", True)
    def test_closed_projection_preserves_decisions_and_rejects_malformed_results(self):
        for name in TASKS:
            export_root = os.environ.get("FRONTIER_PRIVATE_METRICS_EXPORT_ROOT")
            task = (
                Path(export_root) / name
                if export_root
                else ROOT / "tasks/frontier-cs-2.0-demo/problems/Database" / name
            )
            self.assertEqual(layout.validate_task(task), "patch-workspace")
            spec_path = task / "judge/beat_spec.json"
            beat_spec = json.loads(spec_path.read_text())
            spec = importlib.util.spec_from_file_location(
                "feedback_" + name, task / "verifier/beat_spec_feedback.py"
            )
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            for wins in (False, True):
                with self.subTest(task=name, wins=wins):
                    metrics = {
                        "beats_reference": int(wins),
                        "reference_score": 15.0,
                        "margin": -2.0,
                        "private_debug_field": "private-workload",
                        "beat_spec_workload_passes": [
                            {
                                "workload": "private-workload",
                                "metrics": {
                                    item["name"]: wins for item in beat_spec["metrics"]
                                },
                            },
                        ],
                    }
                    before = copy.deepcopy(metrics)
                    projected = helper.private_metrics(
                        metrics, beat_spec_path=spec_path
                    )
                    jsonschema.validate(projected, PRIVATE_SCHEMA)
                    self.assertEqual(metrics, before)
                    self.assertEqual(projected["beats_reference"], int(wins))
                    self.assertEqual(projected["aggregation"], {"kind": "conjunction"})
                    self.assertEqual(
                        projected["beat_spec_progress"],
                        helper.public_metrics(metrics, beat_spec_path=spec_path)[
                            "beat_spec_progress"
                        ],
                    )
                    self.assertNotIn("private-workload", json.dumps(projected))
                    with self.assertRaises(jsonschema.ValidationError):
                        jsonschema.validate(metrics, PRIVATE_SCHEMA)
                    for verdict in (wins, float(wins), 2, None):
                        invalid = {**metrics, "beats_reference": verdict}
                        with self.assertRaises(ValueError):
                            helper.private_metrics(invalid, beat_spec_path=spec_path)
                    invalid = {**metrics, "beats_reference": int(not wins)}
                    with self.assertRaises(ValueError):
                        helper.private_metrics(invalid, beat_spec_path=spec_path)
                    for rows in (
                        [],
                        metrics["beat_spec_workload_passes"] * 2,
                        [{"workload": "private-workload", "metrics": {}}],
                    ):
                        with self.assertRaises(ValueError):
                            helper.private_metrics(
                                {**metrics, "beat_spec_workload_passes": rows},
                                beat_spec_path=spec_path,
                            )
                    for key in ("reference_score", "margin"):
                        for value in (True, float("nan"), float("inf")):
                            with self.assertRaises(ValueError):
                                helper.private_metrics(
                                    {**metrics, key: value}, beat_spec_path=spec_path
                                )
            self.assertIsNone(helper.private_metrics({}, beat_spec_path=spec_path))


if __name__ == "__main__":
    unittest.main()
