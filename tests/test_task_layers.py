"""Task layers (scripts/task_layers.py): a declared layer adds rows; it never changes the shared
layer's, and the files split back exactly."""

import deployment_test_guard  # noqa: F401  # isort: skip

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import task_layers  # noqa: E402

SCHEMA = task_layers.CATALOG_SCHEMA


def repository(directory: Path, layer_rows: list[str]) -> Path:
    root = directory / "repo"
    (root / "config").mkdir(parents=True)
    shared = {"schema_version": SCHEMA, "candidate": {"name": "codex"},
              "tasks": [{"name": "a_task"}]}
    (root / "config/task-catalog.json").write_text(task_layers.json_text(shared))
    (root / "config/task-definitions.json").write_text(task_layers.json_text({"a_task": 1}))
    (root / "extra/config").mkdir(parents=True)
    (root / "extra/layer.json").write_text(json.dumps(
        {"problems": "tasks", "catalog": "config/task-catalog.json",
         "definitions": "config/task-definitions.json"}))
    rows = {"schema_version": SCHEMA, "tasks": [{"name": name} for name in layer_rows]}
    (root / "extra/config/task-catalog.json").write_text(task_layers.json_text(rows))
    (root / "extra/config/task-definitions.json").write_text(
        task_layers.json_text({name: 2 for name in layer_rows}))
    return root


class TaskLayerTests(unittest.TestCase):
    def test_layers_merge_and_split_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = repository(Path(directory), ["b_task"])
            catalog = task_layers.load_catalog(root=root)
            self.assertEqual([t["name"] for t in catalog["tasks"]], ["a_task", "b_task"])
            self.assertEqual(catalog["candidate"], {"name": "codex"})
            definitions = task_layers.load_definitions(root)
            self.assertEqual(definitions, {"a_task": 1, "b_task": 2})
            files = task_layers.layer_files(catalog, definitions, root=root)
            for relative, body in files.items():
                self.assertEqual(body, (root / relative).read_text(), relative)
            catalog["tasks"].append({"name": "c_task"})
            definitions["c_task"] = 3
            files = task_layers.layer_files(catalog, definitions, root / "extra", root)
            self.assertEqual(
                json.loads(files[Path("extra/config/task-definitions.json")]),
                {"b_task": 2, "c_task": 3})
            self.assertEqual(
                json.loads(files[Path("config/task-definitions.json")]), {"a_task": 1})

    def test_a_task_in_two_layers_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = repository(Path(directory), ["a_task"])
            with self.assertRaisesRegex(ValueError, "in another layer"):
                task_layers.load_catalog(root=root)
            with self.assertRaisesRegex(ValueError, "in another layer"):
                task_layers.load_definitions(root)

    def test_a_layer_declares_exactly_its_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = repository(Path(directory), ["b_task"])
            (root / "extra/layer.json").write_text('{"catalog": "x"}')
            with self.assertRaisesRegex(ValueError, "must declare exactly"):
                task_layers.layers(root)


if __name__ == "__main__":
    unittest.main()
