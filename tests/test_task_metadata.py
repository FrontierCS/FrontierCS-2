"""Per-task metadata and TASKS.md must match the enforced task definitions."""

# Enforce offline tests before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import task_metadata


class TaskMetadataTests(unittest.TestCase):
    def test_metadata_and_table_are_current(self):
        self.assertEqual(task_metadata.main(["check"]), 0)

    def test_schema_is_flat_with_consistent_types(self):
        records = task_metadata.existing_records()
        self.assertEqual(
            sorted(records),
            sorted(
                t["name"]
                for t in json.loads((ROOT / "config/task-catalog.json").read_text())[
                    "tasks"
                ]
            ),
        )
        for name, record in records.items():
            with self.subTest(task=name):
                self.assertEqual(tuple(record), task_metadata.FIELDS)
                for key, value in record.items():
                    if key in ("task_model_keys", "network_domains"):
                        self.assertIsInstance(value, list)
                        self.assertTrue(all(isinstance(item, str) for item in value))
                    elif key.endswith(("_count", "_seconds", "train_calls")):
                        self.assertIsInstance(value, int)
                    elif key.endswith(("_cores", "_gib", "_usd")):
                        self.assertTrue(
                            value is None or isinstance(value, (int, float))
                        )
                    else:
                        self.assertTrue(value is None or isinstance(value, str), key)
                if record["agent_gpu_count"] == 0:
                    self.assertIsNone(record["agent_gpu_type"])
                if record["verifier_gpu_count"] == 0:
                    self.assertIsNone(record["verifier_gpu_type"])


if __name__ == "__main__":
    unittest.main()
