from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import frontier_deploy
import render_task_definition as renderer
from preview import publication
from task_selection import requirements, select_tasks
from test_task_seeding import mirrored_images, resolved_catalog

KEYS = [
    {"name": "explore", "model_aliases": ["progent-agent-gpt-4o-mini"], "budget_usd": 15, "envs": ["agent"]},
    {"name": "train", "model_aliases": ["progent-agent-gpt-4o-mini"], "budget_usd": 20, "envs": ["verifier-train"]},
    {"name": "final", "model_aliases": ["progent-agent-gpt-4o-mini"], "budget_usd": 30.5, "envs": ["verifier-eval"]},
]


class ValidationTests(unittest.TestCase):
    def test_accepts_platform_shape(self):
        self.assertEqual(renderer.validate_additional_keys(copy.deepcopy(KEYS)), KEYS)

    def test_rejects_malformed_declarations(self):
        cases = {
            "empty list": [],
            "duplicate name": [KEYS[0], dict(KEYS[1], name="explore")],
            "bad name": [dict(KEYS[0], name="Explore")],
            "no alias": [dict(KEYS[0], model_aliases=[])],
            "duplicate alias": [dict(KEYS[0], model_aliases=["a-1", "a-1"])],
            "zero budget": [dict(KEYS[0], budget_usd=0)],
            "seven decimals": [dict(KEYS[0], budget_usd=1.0000001)],
            "boolean budget": [dict(KEYS[0], budget_usd=True)],
            "unknown env": [dict(KEYS[0], envs=["verifier-final"])],
            "duplicate env": [dict(KEYS[0], envs=["agent", "agent"])],
            "extra field": [dict(KEYS[0], budget_duration="30d")],
        }
        for label, value in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                renderer.validate_additional_keys(copy.deepcopy(value))


class PublicationTests(unittest.TestCase):
    name = "sparse_video_attention_policy"

    def setUp(self):
        # One task keeps the fixture independent of other tasks' publication pins.
        self.definitions = {self.name: copy.deepcopy(publication.load_task_definitions()[self.name])}
        self.catalog = frontier_deploy.validate_task_catalog(
            resolved_catalog([self.name]), require_resolved=True, definitions=self.definitions
        )
        self.task = next(t for t in self.catalog["tasks"] if t["name"] == self.name)
        self.mirrors = mirrored_images(self.catalog)

    def document(self):
        return frontier_deploy.task_version_document(
            self.task, self.mirrors[self.name], definitions=self.definitions
        )

    def test_keys_reach_the_task_config(self):
        before = self.document()
        self.assertNotIn("additional_keys", before["driver_config"])
        self.definitions[self.name]["additional_keys"] = copy.deepcopy(KEYS)
        after = self.document()
        self.assertEqual(after["driver_config"]["additional_keys"], KEYS)
        self.assertNotEqual(before["definition_digest"], after["definition_digest"])

    def test_task_config_with_keys_matches_contract_schema(self):
        import jsonschema
        schema = json.loads((ROOT / "docs/contract/schemas/task-config-v1.schema.json").read_text())
        self.definitions[self.name]["additional_keys"] = copy.deepcopy(KEYS)
        jsonschema.validate(self.document()["driver_config"], schema)
        bad = self.document()["driver_config"]
        bad["additional_keys"][0]["envs"] = ["verifier-final"]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)

    def test_malformed_keys_fail_publication(self):
        self.definitions[self.name]["additional_keys"] = [dict(KEYS[0], envs=[])]
        with self.assertRaises(ValueError):
            self.document()


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = frontier_deploy.load_json(ROOT / "config/site.example.json")
        self.config["features"]["modal_gpu"] = True
        self.catalog = frontier_deploy.load_json(frontier_deploy.TASK_CATALOG_PATH)
        self.definitions = copy.deepcopy(frontier_deploy.TASK_DEFINITIONS)
        self.name = self.catalog["tasks"][0]["name"]
        self.definitions[self.name]["additional_keys"] = copy.deepcopy(KEYS)

    def test_requirement_lists_aliases(self):
        self.assertEqual(requirements(self.definitions[self.name])["task_model_aliases"],
                         ["progent-agent-gpt-4o-mini"])

    def test_site_without_alias_excludes_task(self):
        self.config["features"].pop("task_model_aliases", None)
        selection = select_tasks(self.config, self.catalog, self.definitions)
        self.assertEqual(selection.excluded[self.name],
                         "requires features.task_model_aliases: progent-agent-gpt-4o-mini")
        self.assertNotIn(self.name, selection.selected)

    def test_config_validator_accepts_and_checks_aliases(self):
        from preview import config as site_config
        raw = frontier_deploy.load_json(ROOT / "config/site.example.json")
        site_config.validate_config(copy.deepcopy(raw))
        for bad in (["Bad-Alias"], ["a-1", "a-1"], "progent-agent-gpt-4o-mini"):
            changed = copy.deepcopy(raw)
            changed["features"]["task_model_aliases"] = bad
            with self.subTest(bad=bad), self.assertRaises(Exception):
                site_config.validate_config(changed)

    def test_site_serving_alias_selects_task(self):
        self.config["features"]["task_model_aliases"] = ["progent-agent-gpt-4o-mini"]
        selection = select_tasks(self.config, self.catalog, self.definitions)
        self.assertIn(self.name, selection.selected)


if __name__ == "__main__":
    unittest.main()
