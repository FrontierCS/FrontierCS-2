"""Check final transport against the platform's closed metrics contract."""

# Install isolation before runtime imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import contextlib
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
                else ROOT / "tasks/problems/Database" / name
            )
            self.assertEqual(layout.validate_task(task), "patch-workspace")
            spec_path = task / "judge/beat_spec.json"
            if not spec_path.is_file():  # a task.toml task: provenance copy
                spec_path = ROOT / "docs/task-authoring" / name / "provenance/beat_spec.json"
            beat_spec = json.loads(spec_path.read_text())
            # A task.toml task has no verifier/ of its own: the shared helper projects it.
            own = task / "verifier/beat_spec_feedback.py"
            shared = not own.is_file()
            spec = importlib.util.spec_from_file_location(
                "feedback_" + name,
                ROOT / "tasks/shared/verifier/beat_spec_feedback.py" if shared else own,
            )
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)

            # The shared helper (task.toml tasks) reads no spec file: progress comes from
            # the rows and the evaluator's beats_reference is the authority.
            def private(metrics):
                return (helper.private_metrics(metrics) if shared
                        else helper.private_metrics(metrics, beat_spec_path=spec_path))

            def public(metrics):
                return (helper.public_metrics(metrics, visible=[]) if shared
                        else helper.public_metrics(metrics, beat_spec_path=spec_path))
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
                    projected = private(metrics)
                    jsonschema.validate(projected, PRIVATE_SCHEMA)
                    self.assertEqual(metrics, before)
                    self.assertEqual(projected["beats_reference"], int(wins))
                    if not shared:  # the shared envelope has no aggregation field
                        self.assertEqual(projected["aggregation"], {"kind": "conjunction"})
                    self.assertEqual(
                        projected["beat_spec_progress"],
                        public(metrics)[
                            "beat_spec_progress"
                        ],
                    )
                    self.assertNotIn("private-workload", json.dumps(projected))
                    with self.assertRaises(jsonschema.ValidationError):
                        jsonschema.validate(metrics, PRIVATE_SCHEMA)
                    # The shared helper reads a boolean verdict as 0/1 (as the shared
                    # verifier always has); every other non-integer verdict is refused.
                    for verdict in ((float(wins), 2, None) if shared
                                    else (wins, float(wins), 2, None)):
                        invalid = {**metrics, "beats_reference": verdict}
                        with self.assertRaises(ValueError):
                            private(invalid)
                    invalid = {**metrics, "beats_reference": int(not wins)}
                    with (contextlib.nullcontext() if shared
                          else self.assertRaises(ValueError)):
                        private(invalid)
                    for rows in (
                        [],
                        metrics["beat_spec_workload_passes"] * 2,
                        [{"workload": "private-workload", "metrics": {}}],
                    ):
                        with self.assertRaises(ValueError):
                            private({**metrics, "beat_spec_workload_passes": rows})
                    for key in ("reference_score", "margin"):
                        for value in (True, float("nan"), float("inf")):
                            with self.assertRaises(ValueError):
                                private({**metrics, key: value})
            self.assertIsNone(private({}))


if __name__ == "__main__":
    unittest.main()
