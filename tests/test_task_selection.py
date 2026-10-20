from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import frontier_deploy as deploy
import task_selection
from task_selection import attested_agent_gpu_interfaces, requirements, select_tasks

SPARSE = "sparse_video_attention_policy"


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = deploy.load_json(ROOT / "config/site.example.json")
        self.catalog = deploy.load_json(deploy.TASK_CATALOG_PATH)
        self.definitions = copy.deepcopy(deploy.TASK_DEFINITIONS)

    def test_cpu_selection_preserves_raw_catalog_order_and_objects(self):
        self.config["features"]["modal_gpu"] = False
        before = copy.deepcopy(self.catalog)
        selection = select_tasks(self.config, self.catalog, self.definitions)
        self.assertEqual(
            selection.selected,
            [
                t["name"]
                for t in self.catalog["tasks"]
                if not requirements(self.definitions[t["name"]])["modal_gpu"]
            ],
        )
        self.assertEqual(
            selection.excluded,
            {
                t["name"]: "requires features.modal_gpu"
                for t in self.catalog["tasks"]
                if requirements(self.definitions[t["name"]])["modal_gpu"]
            },
        )
        self.assertEqual(list(selection.effective_catalog), list(self.catalog))
        self.assertIs(
            selection.effective_catalog["candidate"], self.catalog["candidate"]
        )
        self.assertEqual(
            selection.reference_names,
            [
                t["name"]
                for t in selection.effective_catalog["tasks"]
                if t["reference"] is not None
            ],
        )
        for task in selection.effective_catalog["tasks"]:
            self.assertIs(
                task,
                next(t for t in self.catalog["tasks"] if t["name"] == task["name"]),
            )
            self.assertNotIn("reference", task["agent_image"])
            self.assertIs(
                selection.definitions[task["name"]], self.definitions[task["name"]]
            )
        self.assertEqual(self.catalog, before)

    def test_gpu_selection_includes_every_task(self):
        self.config["features"]["modal_gpu"] = True
        selection = select_tasks(self.config, self.catalog, self.definitions)
        self.assertEqual(selection.selected, [t["name"] for t in self.catalog["tasks"]])
        self.assertEqual(selection.excluded, {})
        self.assertEqual(selection.effective_catalog_sha256, selection.catalog_sha256)
        self.assertNotIn(SPARSE, selection.reference_names)

    def test_selection_and_references_follow_catalog_order(self):
        self.config["features"]["modal_gpu"] = False
        self.catalog["tasks"].reverse()
        selection = select_tasks(self.config, self.catalog, self.definitions)
        expected = [
            task["name"]
            for task in self.catalog["tasks"]
            if not requirements(self.definitions[task["name"]])["modal_gpu"]
        ]
        self.assertEqual(selection.selected, expected)
        self.assertEqual(
            selection.reference_names,
            [
                t["name"]
                for t in self.catalog["tasks"]
                if t["name"] in expected and t["reference"] is not None
            ],
        )

    def test_requirements_agent_verifier_and_cpu(self):
        self.assertEqual(requirements({}), {"modal_gpu": False})
        gpu = self.definitions[SPARSE]["gpu"]
        for role in gpu:
            self.assertTrue(requirements({"gpu": {role: gpu[role]}})["modal_gpu"])
        zero = copy.deepcopy(gpu)
        zero["agent"]["max_instances"] = zero["verifier"]["max_gpu_instances"] = 0
        self.assertFalse(requirements({"gpu": zero})["modal_gpu"])

    def test_malformed_gpu_is_never_silently_cpu(self):
        cases = [
            None,
            False,
            [],
            {},
            {"unknown": {}},
            {"agent": []},
            {"agent": {}},
            {"verifier": {"max_gpu_instances": True, "max_gpu_seconds": 0}},
            {"verifier": {"max_gpu_instances": "0", "max_gpu_seconds": 0}},
            {"verifier": {"max_gpu_instances": 0, "max_gpu_seconds": -1}},
        ]
        for types in (None, "H100", [1], ["H100", "H100"], [""]):
            cases.append(
                {
                    "agent": dict(
                        self.definitions[SPARSE]["gpu"]["agent"], allowed_types=types
                    )
                }
            )
        cases.append(
            {"agent": dict(self.definitions[SPARSE]["gpu"]["agent"], unknown=0)}
        )
        for gpu in cases:
            with self.subTest(gpu=gpu), self.assertRaises(deploy.DeployError):
                requirements({"gpu": gpu})

    def test_empty_selection_is_rejected(self):
        self.config["features"]["modal_gpu"] = False
        self.catalog["tasks"] = [
            t for t in self.catalog["tasks"] if t["name"] == SPARSE
        ]
        with self.assertRaisesRegex(deploy.DeployError, "selection is empty"):
            select_tasks(self.config, self.catalog, self.definitions)

    def test_only_selected_interfaces_must_be_attested(self):
        accelerator = self.definitions[SPARSE]["metadata"]["accelerator"]
        accelerator["agent_gpu_interface"] = "frontier.gpu-helper/v4"
        self.config["features"]["modal_gpu"] = False
        select_tasks(self.config, self.catalog, self.definitions)
        self.config["features"]["modal_gpu"] = True
        with self.assertRaisesRegex(deploy.DeployError, "not attested"):
            select_tasks(self.config, self.catalog, self.definitions)
        for interface in ("frontier.gpu-helper/v2", "frontier.gpu-helper/v3"):
            accelerator["agent_gpu_interface"] = interface
            self.assertIn(
                SPARSE,
                select_tasks(self.config, self.catalog, self.definitions).selected,
            )

    def test_kimi_v3_requires_exact_image_and_source_receipt_bindings(self):
        receipt = deploy.load_json(ROOT / "config/kimi-candidate-image.json")
        for field in ("digest", "tag", "source_commit"):
            changed = copy.deepcopy(receipt)
            changed[field] = "invalid"
            with (
                self.subTest(field=field),
                mock.patch.object(task_selection, "load_json", return_value=changed),
            ):
                with self.assertRaises(deploy.DeployError):
                    attested_agent_gpu_interfaces()
        for field in ("contract", "overlay_digest", "files"):
            changed = copy.deepcopy(receipt)
            changed["gpu_helper"][field] = (
                {"candidates/kimi-code/Dockerfile": "sha256:" + "0" * 64}
                if field == "files"
                else "invalid"
            )
            with (
                self.subTest(field=field),
                mock.patch.object(task_selection, "load_json", return_value=changed),
            ):
                with self.assertRaises(deploy.DeployError):
                    attested_agent_gpu_interfaces()

    def test_complete_kimi_receipt_attests_v3(self):
        self.assertIn("frontier.gpu-helper/v3", attested_agent_gpu_interfaces())

    def test_kimi_manifest_selection_must_bind_the_executable_leaf(self):
        receipt = deploy.load_json(ROOT / "config/kimi-candidate-image.json")
        mutations = {
            "leaf_digest": "sha256:" + "0" * 64,
            "index_digest": receipt["digest"],
            "os": "windows",
            "architecture": "arm64",
            "verified": False,
            "media_type": "application/vnd.oci.image.index.v1+json",
            "public_manifest_sha256": "sha256:" + "0" * 64,
            "internal_manifest_sha256": "sha256:" + "0" * 64,
        }
        for key, value in mutations.items():
            changed = copy.deepcopy(receipt)
            changed["manifest_selection"][key] = value
            with (
                self.subTest(key=key),
                mock.patch.object(task_selection, "load_json", return_value=changed),
            ):
                with self.assertRaisesRegex(
                    deploy.DeployError, "verified linux/amd64 manifest"
                ):
                    attested_agent_gpu_interfaces()
        changed = copy.deepcopy(receipt)
        changed["manifest_selection"]["media_type"] = (
            "application/vnd.docker.distribution.manifest.list.v2+json"
        )
        with mock.patch.object(task_selection, "load_json", return_value=changed):
            with self.assertRaisesRegex(
                deploy.DeployError, "verified linux/amd64 manifest"
            ):
                attested_agent_gpu_interfaces()

    def test_pure_leaf_publication_needs_no_parent_index(self):
        receipt = deploy.load_json(ROOT / "config/kimi-candidate-image.json")
        del receipt["manifest_selection"]["index_digest"]
        with mock.patch.object(task_selection, "load_json", return_value=receipt):
            self.assertIn("frontier.gpu-helper/v3", attested_agent_gpu_interfaces())

    def test_kimi_core_source_and_registry_readback_cannot_drift(self):
        receipt = deploy.load_json(ROOT / "config/kimi-candidate-image.json")
        mutations = [
            ("source", "commit", "0" * 40),
            ("source", "repository", "https://example.invalid/other"),
            ("source", "dockerfile_sha256", "sha256:" + "0" * 64),
            ("registry_readback", "image_digest", "sha256:" + "0" * 64),
            ("registry_readback", "files", {}),
        ]
        for section, field, value in mutations:
            changed = copy.deepcopy(receipt)
            changed[section][field] = value
            with (
                self.subTest(section=section, field=field),
                mock.patch.object(task_selection, "load_json", return_value=changed),
            ):
                with self.assertRaises(deploy.DeployError):
                    attested_agent_gpu_interfaces()
        changed = copy.deepcopy(receipt)
        changed["registry_readback"]["files"]["launcher.py"]["mode"] = 0o777
        with mock.patch.object(task_selection, "load_json", return_value=changed):
            with self.assertRaisesRegex(deploy.DeployError, "source/image file proof"):
                attested_agent_gpu_interfaces()

    def test_kimi_receipt_requires_every_source_hash(self):
        receipt = deploy.load_json(ROOT / "config/kimi-candidate-image.json")
        self.config["features"]["modal_gpu"] = True
        self.definitions[SPARSE]["metadata"]["accelerator"]["agent_gpu_interface"] = (
            "frontier.gpu-helper/v3"
        )
        for name in receipt["gpu_helper"]["files"]:
            changed = copy.deepcopy(receipt)
            del changed["gpu_helper"]["files"][name]
            with (
                self.subTest(name=name),
                mock.patch.object(task_selection, "load_json", return_value=changed),
            ):
                for action in (
                    attested_agent_gpu_interfaces,
                    lambda: select_tasks(self.config, self.catalog, self.definitions),
                ):
                    with self.assertRaisesRegex(
                        deploy.DeployError, "not attested:.*missing hashes"
                    ) as error:
                        action()
                    self.assertIn(name, str(error.exception))

    def test_kimi_receipt_rejects_invalid_source_hashes(self):
        receipt = deploy.load_json(ROOT / "config/kimi-candidate-image.json")
        for name in receipt["gpu_helper"]["files"]:
            for digest in (
                None,
                1,
                "",
                "a" * 64,
                "sha256:" + "g" * 64,
                "sha256:" + "a" * 63,
                "sha256:" + "a" * 64 + "\n",
            ):
                changed = copy.deepcopy(receipt)
                changed["gpu_helper"]["files"][name] = digest
                with (
                    self.subTest(name=name, digest=digest),
                    mock.patch.object(
                        task_selection, "load_json", return_value=changed
                    ),
                ):
                    with self.assertRaisesRegex(
                        deploy.DeployError, "not attested:.*invalid sha256"
                    ):
                        attested_agent_gpu_interfaces()

    def test_digest_binds_domains_and_catalog_even_when_excluded(self):
        self.config["features"]["modal_gpu"] = False
        first = select_tasks(self.config, self.catalog, self.definitions)
        binding = {
            "selected": first.selected,
            "excluded": first.excluded,
            "effective_catalog_sha256": first.effective_catalog_sha256,
            "model_domains": ["openrouter.ai"],
            "catalog_sha256": first.catalog_sha256,
        }
        self.assertEqual(
            first.selection_digest, deploy.sha256_bytes(deploy.canonical_json(binding))
        )
        self.config["execution"]["model_domains"] = [
            "z.example",
            "a.example",
            "z.example",
        ]
        second = select_tasks(self.config, self.catalog, self.definitions)
        self.assertEqual(second.model_domains, ["a.example", "z.example"])
        self.assertNotEqual(first.selection_digest, second.selection_digest)
        sparse = next(t for t in self.catalog["tasks"] if t["name"] == SPARSE)
        sparse["agent_image"]["digest"] = "sha256:" + "a" * 64
        third = select_tasks(self.config, self.catalog, self.definitions)
        self.assertEqual(
            second.effective_catalog_sha256, third.effective_catalog_sha256
        )
        self.assertNotEqual(second.selection_digest, third.selection_digest)


class ExecutionSchemaTests(unittest.TestCase):
    def test_schema_and_validator_agree_on_execution_keys_and_domains(self):
        raw = deploy.load_json(ROOT / "config/site.example.json")
        schema = deploy.load_json(ROOT / "config/site.schema.json")
        execution = schema["properties"]["execution"]
        required = {
            "object_store_gib",
            "registry_storage_gib",
            "max_active_drivers",
            "agent_external_egress",
            "smoke",
        }
        self.assertEqual(set(execution["required"]), required)
        self.assertEqual(
            set(execution["properties"]),
            required
            | {
                "model_domains",
                "verifier_node_allowlists",
                "verifier_placement_required_task_ids",
                "task_verifier_workspace_bytes",
            },
        )
        self.assertFalse(execution["additionalProperties"])
        cases = [(raw, True)]
        for key in required:
            config = copy.deepcopy(raw)
            del config["execution"][key]
            cases.append((config, False))
        config = copy.deepcopy(raw)
        config["execution"]["unknown"] = True
        cases.append((config, False))
        for domains, valid in [
            (["openrouter.ai"], True),
            (["a.example"] * 32, True),
            (["a.example"] * 33, False),
            (["a.example", "a.example"], True),
            (["x" * 63 + ".example"], True),
            ([], True),
            (None, False),
            ("openrouter.ai", False),
            ([1], False),
            (["UPPER.ai"], False),
            (["https://openrouter.ai"], False),
            (["a..b"], False),
            (["-a.example"], False),
            (["a-.example"], False),
            (["a.example."], False),
            (["a.example\n"], False),
            (["x" * 64 + ".example"], False),
        ]:
            config = copy.deepcopy(raw)
            config["execution"]["model_domains"] = domains
            cases.append((config, valid))
        for config, valid in cases:
            with self.subTest(execution=config["execution"]):
                self.assertEqual(
                    jsonschema.Draft202012Validator(schema).is_valid(config), valid
                )
                if valid:
                    deploy.validate_config(config)
                else:
                    with self.assertRaises(deploy.DeployError):
                        deploy.validate_config(config)


if __name__ == "__main__":
    unittest.main()
