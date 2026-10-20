from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import frontier_deploy as deploy
import reference_candidates as references
import reference_parity as parity
from preview import images as image_ops
from preview import publication
from task_selection import select_tasks
from test_task_seeding import mirrored_images, resolved_catalog


class NativeDeployTests(unittest.TestCase):
    def setUp(self):
        self.catalog = deploy.validate_task_catalog(
            resolved_catalog(), require_resolved=True
        )
        self.mirrors = mirrored_images(self.catalog)

    def test_explicit_verifier_workspace_is_separate_and_content_addressed(self):
        config = deploy.load_json(ROOT / "config/site.example.json")
        config["features"]["modal_gpu"] = True
        policies = copy.deepcopy(deploy.TASK_DEFINITIONS)
        original = copy.deepcopy(policies)
        selected = select_tasks(config, self.catalog, policies)
        empty = copy.deepcopy(config)
        empty["execution"]["task_verifier_workspace_bytes"] = {}
        unchanged = select_tasks(empty, self.catalog, policies)
        self.assertEqual(selected.selection_digest, unchanged.selection_digest)
        for task in self.catalog["tasks"]:
            name = task["name"]
            self.assertEqual(
                deploy.task_version_document(task, self.mirrors[name]),
                deploy.task_version_document(
                    task, self.mirrors[name], definitions=unchanged.definitions
                ),
            )
        task = next(
            t
            for t in self.catalog["tasks"]
            if t["name"] == "residual_state_advancement"
        )
        name = task["name"]
        config["execution"]["task_verifier_workspace_bytes"] = {name: 16 << 30}
        overlaid = select_tasks(config, self.catalog, policies)
        old = deploy.task_version_document(task, self.mirrors[name])
        new = deploy.task_version_document(
            task, self.mirrors[name], definitions=overlaid.definitions
        )
        self.assertNotEqual(old["definition_digest"], new["definition_digest"])
        self.assertNotEqual(
            old["driver_config"]["task_version_id"],
            new["driver_config"]["task_version_id"],
        )
        expected = copy.deepcopy(old["driver_config"])
        expected["task_version_id"] = new["driver_config"]["task_version_id"]
        expected["verifier"]["workspace_bytes"] = 16 << 30
        self.assertEqual(expected, new["driver_config"])
        self.assertEqual(old["metadata"], new["metadata"])
        self.assertEqual(policies, original)

    def test_verifier_workspace_rejects_invalid_or_unselected_budgets(self):
        config = deploy.load_json(ROOT / "config/site.example.json")
        name = "residual_state_advancement"
        for bad in (True, 0, -1, None, 1.5, (1 << 40) + 1):
            with self.subTest(budget=bad), self.assertRaises(deploy.DeployError):
                config["execution"]["task_verifier_workspace_bytes"] = {name: bad}
                select_tasks(config, self.catalog, deploy.TASK_DEFINITIONS)
        config["execution"]["task_verifier_workspace_bytes"] = {
            "not_selected": 16 << 30
        }
        with self.assertRaises(deploy.DeployError):
            select_tasks(config, self.catalog, deploy.TASK_DEFINITIONS)

    def test_local_candidate_requires_the_explicit_deployment_mirror(self):
        candidate = self.catalog["candidate"]
        mirror = self.mirrors["candidate"]["overlay_image"]
        registry = mirror["reference"].split("/", 1)[0]
        expected = deploy.candidate_version_document(candidate, mirror)
        self.assertEqual(
            expected,
            deploy.candidate_version_document(
                candidate, mirror, deployment_registry=registry
            ),
        )
        for wrong in (
            None,
            {**mirror, "reference": "docker.io/public/overlay@" + mirror["digest"]},
            {**mirror, "reference": "127.0.0.1:5000/overlay@" + mirror["digest"]},
        ):
            with self.subTest(mirror=wrong), self.assertRaises(deploy.DeployError):
                deploy.candidate_version_document(
                    candidate, wrong, deployment_registry=registry
                )

    def test_default_candidate_document_matches_pre_change_bytes(self):
        expected = json.loads(
            (ROOT / "tests/fixtures/codex-candidate-before-helper.json").read_text()
        )
        overlay = expected["driver_config"]["overlay_image"]
        candidate = {
            "name": "codex",
            "overlay_image": {
                "repository": overlay["reference"].split("@", 1)[0],
                "tag": expected["version"].split("-cfg-", 1)[0],
                "digest": overlay["digest"],
                "reference": overlay["reference"],
            },
        }
        # Historical execution inputs reproduce the frozen identity algorithm.
        actual = deploy.canonical_json(
            deploy.candidate_version_document(
                candidate, execution=expected["driver_config"]["execution"]
            )
        )
        self.assertEqual(actual, deploy.canonical_json(expected))

    def test_default_candidate_matches_historical_document_with_historical_budget(self):
        expected = json.loads(
            (ROOT / "tests/fixtures/codex-candidate-e64d130.json").read_text()
        )
        overlay = expected["driver_config"]["overlay_image"]
        candidate = {
            "name": "codex",
            "overlay_image": {
                "repository": overlay["reference"].split("@", 1)[0],
                "tag": expected["version"].split("-cfg-", 1)[0],
                "digest": overlay["digest"],
                "reference": overlay["reference"],
            },
        }
        actual = deploy.candidate_version_document(
            candidate, execution=expected["driver_config"]["execution"]
        )
        self.assertEqual(deploy.canonical_json(actual), deploy.canonical_json(expected))

    def test_every_reference_execution_and_identity_input_is_hashed(self):
        image = references.load_manifest()["batch_local_counting"]
        mirror = {
            **self.mirrors["candidate"]["overlay_image"],
            "source_sha256": image["digest"],
            "source_reference": image["repository"] + "@" + image["digest"],
        }
        original = parity.candidate_document("batch_local_counting", image, mirror)
        self.assertEqual(original["driver_config"]["execution"], parity.EXECUTION)
        identity = {
            key: original[key] for key in ("candidate_id", "display_name", "metadata")
        }
        identity["version"] = image["tag"]
        candidate = {"name": "batch_local_counting", "overlay_image": image}
        first = deploy.candidate_version_document(
            candidate, mirror, execution=parity.EXECUTION, identity=identity
        )
        for key in identity:
            changed = copy.deepcopy(identity)
            changed[key] = (
                {**changed[key], "extra": "changed"}
                if key == "metadata"
                else changed[key] + "-changed"
            )
            new = deploy.candidate_version_document(
                candidate, mirror, execution=parity.EXECUTION, identity=changed
            )
            self.assertNotEqual(
                first["driver_config"]["candidate_version_id"],
                new["driver_config"]["candidate_version_id"],
            )
            self.assertNotEqual(first["definition_digest"], new["definition_digest"])
        for key in parity.EXECUTION:
            execution = copy.deepcopy(parity.EXECUTION)
            if key == "env":
                execution[key]["FRONTIER_TRAIN_POLL_TIMEOUT_SECONDS"] = "86399"
            elif isinstance(execution[key], list):
                execution[key].append("changed")
            else:
                execution[key] += "changed"
            new = deploy.candidate_version_document(
                candidate, mirror, execution=execution, identity=identity
            )
            self.assertNotEqual(first["definition_digest"], new["definition_digest"])
        self.assertEqual(
            first["definition_digest"],
            deploy.canonical_object_sha256(first["driver_config"]),
        )

    def test_missing_schema_fails_and_every_task_merges_site_domains_without_mutating_policy(
        self,
    ):
        before = copy.deepcopy(deploy.TASK_DEFINITIONS)
        for task in self.catalog["tasks"]:
            name = task["name"]
            doc = deploy.task_version_document(task, self.mirrors[name])
            self.assertEqual(
                doc["driver_config"]["network"]["allowed_domains"], ["openrouter.ai"]
            )
            self.assertEqual(
                doc["driver_config"]["network"]["audit"],
                before[name]["network"]["audit"],
            )
            other = deploy.task_version_document(
                task,
                self.mirrors[name],
                model_domains=["model.example", "second.example"],
            )
            self.assertEqual(
                other["driver_config"]["network"]["allowed_domains"],
                ["model.example", "second.example"],
            )
            self.assertNotEqual(
                doc["driver_config"]["task_version_id"],
                other["driver_config"]["task_version_id"],
            )
            self.assertNotEqual(doc["definition_digest"], other["definition_digest"])
        self.assertEqual(deploy.TASK_DEFINITIONS, before)
        name = self.catalog["tasks"][0]["name"]
        before[name].pop("policy_schema_version")
        with (
            mock.patch.object(
                publication, "load_task_definitions", return_value=before
            ),
            self.assertRaisesRegex(deploy.DeployError, "policy_schema_version"),
        ):
            deploy.task_version_document(self.catalog["tasks"][0], self.mirrors[name])

    def test_authored_domains_are_retained_and_deduplicated(self):
        definitions = copy.deepcopy(deploy.TASK_DEFINITIONS)
        task = self.catalog["tasks"][0]
        definitions[task["name"]]["network"]["allowed_domains"] = [
            "data.example",
            "openrouter.ai",
        ]
        with mock.patch.object(
            publication, "load_task_definitions", return_value=definitions
        ):
            doc = deploy.task_version_document(task, self.mirrors[task["name"]])
            self.assertEqual(
                doc["driver_config"]["network"]["allowed_domains"],
                ["data.example", "openrouter.ai"],
            )

    def test_empty_model_domains_preserve_authored_network_and_identity(self):
        definitions = copy.deepcopy(deploy.TASK_DEFINITIONS)
        task = self.catalog["tasks"][0]
        for authored in ([], ["data.example"]):
            with self.subTest(authored=authored):
                definitions[task["name"]]["network"]["allowed_domains"] = authored
                with mock.patch.object(
                    publication, "load_task_definitions", return_value=definitions
                ):
                    default = deploy.task_version_document(
                        task, self.mirrors[task["name"]]
                    )
                    doc = deploy.task_version_document(
                        task, self.mirrors[task["name"]], model_domains=[]
                    )
                self.assertEqual(
                    doc["driver_config"]["network"]["allowed_domains"], authored
                )
                self.assertNotEqual(
                    default["definition_digest"], doc["definition_digest"]
                )
                self.assertNotEqual(
                    default["driver_config"]["task_version_id"],
                    doc["driver_config"]["task_version_id"],
                )
                self.assertEqual(
                    definitions[task["name"]]["network"]["allowed_domains"], authored
                )

    def test_reference_manifest_binds_payload_launcher_tag_and_exact_task_set(self):
        original = deploy.load_json(ROOT / "config/task-catalog.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            placeholder = copy.deepcopy(original)
            placeholder["tasks"][0]["reference"]["digest"] = references.PLACEHOLDER
            path.write_text(json.dumps(placeholder))
            with self.assertRaisesRegex(ValueError, "placeholder"):
                references.load_manifest(path, require_resolved=True)
            for field in (
                "repository",
                "tag",
                "digest",
                "launcher_sha256",
                "payload_sha256",
            ):
                changed = copy.deepcopy(original)
                changed["tasks"][0]["reference"][field] = "wrong"
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    references.load_manifest(path)
            original["tasks"][0]["reference"] = None
            path.write_text(json.dumps(original))
            self.assertNotIn(
                original["tasks"][0]["name"], references.load_manifest(path)
            )

    def test_placeholder_references_fail_before_mirroring(self):
        self.catalog["tasks"][0]["reference"]["digest"] = references.PLACEHOLDER
        with (
            mock.patch.object(image_ops, "run_command") as run,
            mock.patch.object(image_ops, "executable") as executable,
        ):
            with self.assertRaisesRegex(ValueError, "placeholder"):
                deploy.mirror_task_images(
                    {}, Path("unused"), self.catalog, selection=mock.sentinel.selection
                )
            run.assert_not_called()
            executable.assert_not_called()

    def test_native_task_and_reference_configs_match_contract_schemas(self):
        import jsonschema

        task_schema = json.loads(
            (ROOT / "docs/contract/schemas/task-config-v1.schema.json").read_text()
        )
        candidate_schema = json.loads(
            (ROOT / "docs/contract/schemas/candidate-config-v1.schema.json").read_text()
        )
        for task in self.catalog["tasks"]:
            jsonschema.validate(
                deploy.task_version_document(task, self.mirrors[task["name"]])[
                    "driver_config"
                ],
                task_schema,
            )
        image = references.load_manifest()["batch_local_counting"]
        mirror = {
            **self.mirrors["candidate"]["overlay_image"],
            "source_sha256": image["digest"],
            "source_reference": image["repository"] + "@" + image["digest"],
        }
        jsonschema.validate(
            parity.candidate_document("batch_local_counting", image, mirror)[
                "driver_config"
            ],
            candidate_schema,
        )


if __name__ == "__main__":
    unittest.main()
