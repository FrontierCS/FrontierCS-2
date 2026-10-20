from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import frontier_deploy as deploy
from preview import images as image_ops
from preview import publication
from task_selection import deployment_catalog

FIXTURES = ROOT / "tests/fixtures/live-state-2026-09-11"
FILE_SHA256 = {
    "task-catalog.json": "sha256:584e26a921ff8b88c9ee9c04a156b9e2d5b4cbb1a37e75ef9ee2db0b0ed14fd8",
    "task-definitions.json": "sha256:c26bf5888d9eceaf85be500f521aca1dd06a9f785878f3ff0152c7129447f002",
    "task-seed-receipt.json": "sha256:fa68403ee6c3f083a62b0a2fb207f686172badf794f092351da16f5829c5135d",
    "task-images.json": "sha256:9596d707921a8a19426887377ee99fe41b3f0a9729fb2e87363eedb5faad6706",
}


class LiveStateMigrationTests(unittest.TestCase):
    def setUp(self):
        self.current_definitions = deploy.TASK_DEFINITIONS
        self.receipt = deploy.load_json(FIXTURES / "task-seed-receipt.json")
        self.mirrors = deploy.load_json(FIXTURES / "task-images.json")
        # These exact release inputs belong to the captured receipt, not current main.
        self.full_catalog = deploy.load_json(FIXTURES / "task-catalog.json")
        # Present this fixture in v2 form without changing the archived v1 bytes.
        # It predates reference/baseline publication descriptors.
        self.full_catalog["schema_version"] = "frontier.deployment-task-catalog/v2"
        for task in self.full_catalog["tasks"]:
            task.update(
                reference=None,
                baseline={"provenance": "unverified", "comparable": False},
            )
        definitions = deploy.load_json(FIXTURES / "task-definitions.json")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        catalog_path = self.catalog_path = Path(temporary.name) / "task-catalog.json"
        catalog_path.write_bytes(deploy.canonical_json(self.full_catalog))
        self.enterContext(
            mock.patch.object(publication, "TASK_CATALOG_PATH", catalog_path)
        )
        self.enterContext(
            mock.patch.object(
                publication, "load_task_definitions", return_value=definitions
            )
        )
        self.pruned_catalog = copy.deepcopy(self.full_catalog)
        # Reproduce the site-local pruning documented for the September 8 applies.
        # Hash the raw catalog before validation adds normalized image references.
        self.pruned_catalog["tasks"] = [
            task
            for task in self.pruned_catalog["tasks"]
            if task["name"] != "sparse_video_attention_policy"
        ]
        self.config = deploy.load_json(ROOT / "config/site.example.json")
        self.config["features"]["modal_gpu"] = False
        self.catalog, self.selection = deploy.load_task_selection(
            self.config, require_resolved=True
        )
        self.definitions = self.selection.definitions

    def test_fixtures_are_the_exact_audited_live_state_bytes(self):
        for name, expected in FILE_SHA256.items():
            with self.subTest(fixture=name):
                self.assertEqual(
                    deploy.sha256_bytes((FIXTURES / name).read_bytes()), expected
                )
        self.assertEqual(
            self.receipt["schema_version"], "frontier.deployment-task-seed-receipt/v1"
        )

    def test_receipt_binds_raw_cpu_pruned_catalog(self):
        config = deploy.load_json(ROOT / "config/site.example.json")
        config["features"]["modal_gpu"] = False
        selection = deploy.select_tasks(
            config, self.full_catalog, deploy.TASK_DEFINITIONS
        )
        self.assertEqual(
            selection.effective_catalog_sha256, self.receipt["catalog_sha256"]
        )
        self.assertEqual(selection.effective_catalog, self.pruned_catalog)
        self.assertEqual(
            deploy.sha256_bytes(
                deploy.canonical_json(deployment_catalog(self.pruned_catalog))
            ),
            self.receipt["catalog_sha256"],
        )
        self.assertNotEqual(
            deploy.sha256_bytes(deploy.canonical_json(self.full_catalog)),
            self.receipt["catalog_sha256"],
        )
        self.assertTrue(
            all("gpu" not in definition for definition in self.definitions.values())
        )
        self.assertEqual(
            [task["name"] for task in self.catalog["tasks"]], self.selection.selected
        )
        deploy.validate_seed_receipt(self.receipt, self.selection, "preview")
        # Structural validation still requires the complete tracked catalog.
        with self.assertRaisesRegex(deploy.DeployError, "every fixed task"):
            deploy.validate_task_catalog(self.pruned_catalog, require_resolved=True)

    def test_every_seeded_task_id_is_reproduced_with_model_domain_merge(self):
        before = copy.deepcopy(deploy.TASK_DEFINITIONS)
        ids = []
        for task in self.catalog["tasks"]:
            name = task["name"]
            with self.subTest(task=name):
                mirrors = self.mirrors[name]
                for kind in ("agent_image", "verifier_image", "dataset_image"):
                    self.assertEqual(
                        mirrors[kind]["source_reference"], task[kind]["reference"]
                    )
                    self.assertEqual(
                        mirrors[kind]["source_sha256"], task[kind]["digest"]
                    )
                document = deploy.task_version_document(
                    task, mirrors, model_domains=self.selection.model_domains
                )
                config = document["driver_config"]
                ids.append(config["task_version_id"])
                expected_domains = sorted(
                    set(self.definitions[name]["network"]["allowed_domains"])
                    | {"openrouter.ai"}
                )
                self.assertEqual(config["network"]["allowed_domains"], expected_domains)
                without_model_domain = deploy.task_version_document(
                    task, mirrors, model_domains=[]
                )
                self.assertNotEqual(
                    config["task_version_id"],
                    without_model_domain["driver_config"]["task_version_id"],
                )
        self.assertEqual(ids, self.receipt["task_version_ids"])
        self.assertEqual(deploy.TASK_DEFINITIONS, before)

    def test_tracked_reexports_deliberately_differ_from_the_archived_receipt(self):
        with mock.patch.object(
            publication, "load_task_definitions", return_value=self.current_definitions
        ):
            current, selection = deploy.load_task_selection(
                self.config, ROOT / "config/task-catalog.json"
            )
            self.assertNotEqual(
                selection.effective_catalog_sha256, self.receipt["catalog_sha256"]
            )
            with self.assertRaisesRegex(deploy.DeployError, "catalog_sha256"):
                deploy.validate_seed_receipt(self.receipt, selection, "preview")
            for task in current["tasks"]:
                if task["name"] not in self.mirrors:
                    continue  # Additional imported tasks have no historical binding.
                document = deploy.task_version_document(
                    task,
                    self.mirrors[task["name"]],
                    model_domains=selection.model_domains,
                )
                self.assertNotIn(
                    document["driver_config"]["task_version_id"],
                    self.receipt["task_version_ids"],
                )

    def test_codex_candidate_id_is_reproduced_with_mirrored_overlay(self):
        overlay = self.mirrors["candidate"]["overlay_image"]
        self.assertEqual(
            overlay["source_reference"],
            self.catalog["candidate"]["overlay_image"]["reference"],
        )
        self.assertEqual(
            overlay["source_sha256"],
            self.catalog["candidate"]["overlay_image"]["digest"],
        )
        historical_execution = {
            "entrypoint": ["/opt/frontier/candidate/launch"], "args": [],
            "working_dir": "/workspace", "user": "1000:1000",
            "env": {"FRONTIER_TRAIN_POLL_TIMEOUT_SECONDS": str(max(
                definition["quota"]["wall_time_seconds"]
                for definition in deploy.TASK_DEFINITIONS.values()
            ))},
        }
        document = deploy.candidate_version_document(
            self.catalog["candidate"], overlay, execution=historical_execution
        )
        self.assertEqual(
            document["driver_config"]["candidate_version_id"],
            self.receipt["candidate_version_id"],
        )
        self.assertEqual(
            document["driver_config"]["overlay_image"]["reference"],
            overlay["reference"],
        )
        self.assertEqual(
            document["driver_config"]["execution"]["env"][
                "FRONTIER_TRAIN_POLL_TIMEOUT_SECONDS"
            ],
            str(
                max(
                    definition["quota"]["wall_time_seconds"]
                    for definition in deploy.TASK_DEFINITIONS.values()
                )
            ),
        )

    def test_async_candidate_has_no_poll_timeout_even_when_gpu_task_is_excluded(
        self,
    ):
        definitions = copy.deepcopy(deploy.TASK_DEFINITIONS)
        definitions["sparse_video_attention_policy"]["quota"]["wall_time_seconds"] = (
            4000000
        )
        with mock.patch.object(
            publication, "load_task_definitions", return_value=definitions
        ):
            catalog, selection = deploy.load_task_selection(
                self.config, require_resolved=True
            )
            self.assertNotIn("sparse_video_attention_policy", selection.definitions)
            document = deploy.candidate_version_document(
                catalog["candidate"], self.mirrors["candidate"]["overlay_image"]
            )
        self.assertEqual(document["driver_config"]["execution"]["env"], {})

    def test_check_refuses_fixture_mirrors_for_another_site_registry_before_http(self):
        config = copy.deepcopy(self.config)
        config["network"]["registry_cluster_ip"] = "172.20.0.51"
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            shutil.copytree(FIXTURES, state)
            with (
                mock.patch.object(publication, "state_directory", return_value=state),
                mock.patch.object(publication, "port_forward") as forward,
                mock.patch.object(publication, "api_json") as api,
            ):
                with self.assertRaisesRegex(
                    deploy.DeployError,
                    "saved mirror destination differs from this site's registry:",
                ):
                    deploy.seed_tasks(
                        config,
                        state / "kubeconfig",
                        catalog_path=self.catalog_path,
                        check=True,
                    )
            forward.assert_not_called()
            api.assert_not_called()

    def test_check_validates_every_fixture_mirror_destination_before_http(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            for name, records in self.mirrors.items():
                for kind, record in records.items():
                    for field, value in (
                        (
                            "reference",
                            record["reference"].replace("172.20.0.50", "172.20.0.51"),
                        ),
                        (
                            "reference",
                            record["reference"].replace("@", "-wrong-repository@"),
                        ),
                        ("digest", "sha256:" + "0" * 64),
                        ("digest", None),
                    ):
                        with self.subTest(
                            name=name, kind=kind, field=field, value=value
                        ):
                            mirrors = copy.deepcopy(self.mirrors)
                            mirrors[name][kind][field] = value
                            (state / "task-images.json").write_bytes(
                                deploy.canonical_json(mirrors)
                            )
                            with (
                                mock.patch.object(
                                    publication, "state_directory", return_value=state
                                ),
                                mock.patch.object(
                                    publication, "port_forward"
                                ) as forward,
                                mock.patch.object(publication, "api_json") as api,
                            ):
                                with self.assertRaises(deploy.DeployError) as error:
                                    deploy.check_seed_tasks(
                                        self.config,
                                        state / "kubeconfig",
                                        self.catalog,
                                        self.selection,
                                    )
                            self.assertEqual(
                                str(error.exception),
                                f"saved mirror destination differs from this site's registry: {name}/{kind}",
                            )
                            forward.assert_not_called()
                            api.assert_not_called()

    def test_seed_changes_only_candidate_for_async_template_preserving_task_ids_and_mirrors(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            shutil.copytree(FIXTURES, state)
            mirror_path = state / "task-images.json"
            mirror_bytes = mirror_path.read_bytes()
            saved = deploy.load_json(mirror_path)
            # The archived v1 catalog never bound reference publication descriptors.
            # Test task/Codex seeding from that catalog; do not invent old references.
            selected_mirrors = {**saved, "reference_candidates": {}}
            selected_mirrors.pop("reference_candidates", None)
            cache = {
                (
                    record["reference"].split("/", 1)[1].split("@", 1)[0],
                    record["source_sha256"],
                ): record["digest"]
                for records in selected_mirrors.values()
                for record in records.values()
            }
            candidate = deploy.candidate_version_document(
                self.catalog["candidate"], saved["candidate"]["overlay_image"]
            )
            new_candidate_id = candidate["driver_config"]["candidate_version_id"]
            self.assertNotEqual(new_candidate_id, self.receipt["candidate_version_id"])
            collections = {
                "/api/v1/candidate-versions": [
                    {**candidate, "id": self.receipt["candidate_version_id"]}
                ],
                "/api/v1/task-versions": [
                    {
                        **deploy.task_version_document(
                            task,
                            saved[task["name"]],
                            model_domains=self.selection.model_domains,
                        ),
                        "id": version_id,
                    }
                    for task, version_id in zip(
                        self.catalog["tasks"],
                        self.receipt["task_version_ids"],
                        strict=False,
                    )
                ],
            }
            # The public list view omits publication receipts and full driver configs.
            for records in collections.values():
                for record in records:
                    record.pop("driver_config")
                records.reverse()
            sizes = {
                value.removeprefix("sha256:"): 123
                for task in self.catalog["tasks"]
                for value in task["datasets"].values()
            }

            def get(base, method, path, body=None):
                if method == "POST":
                    self.assertEqual(path, "/api/v1/candidate-versions")
                    self.assertEqual(body, candidate)
                    return {**body, "id": new_candidate_id}
                self.assertEqual(
                    method, "GET", "existing TaskVersions must never be republished"
                )
                self.assertIsNone(body)
                return collections[path]

            def command(args, **kwargs):
                self.assertEqual(
                    args[0], "kubectl", "fixture cache reuse must not execute Docker"
                )
                self.assertIn(args[3], {"create", "wait", "logs"})
                return json.dumps({"dataset_sizes": sizes}) if args[3] == "logs" else ""

            with (
                mock.patch.object(publication, "state_directory", return_value=state),
                mock.patch.object(
                    publication, "executable", side_effect=lambda name: name
                ),
                mock.patch.object(publication, "port_forward") as forward,
                mock.patch.object(
                    image_ops,
                    "cached_registry_image",
                    side_effect=lambda port, repository, digest: cache[
                        (repository, digest)
                    ],
                ) as cached,
                mock.patch.object(
                    image_ops,
                    "registry_digest",
                    side_effect=AssertionError("unexpected cache miss"),
                ),
                mock.patch.object(
                    publication,
                    "load_runtime_images",
                    return_value={
                        "storage_controller": {
                            "reference": "172.20.0.50:5000/frontier/storage-controller@sha256:"
                            + "a" * 64
                        }
                    },
                ),
                mock.patch.object(publication, "kubectl_apply") as apply,
                mock.patch.object(
                    publication, "run_command", side_effect=command
                ) as commands,
                mock.patch.object(publication, "api_json", side_effect=get) as api,
                mock.patch("sys.stdout", new_callable=io.StringIO),
            ):
                self.enterContext(mock.patch.object(image_ops, "port_forward", forward))
                self.enterContext(
                    mock.patch.object(image_ops, "state_directory", return_value=state)
                )
                self.enterContext(
                    mock.patch.object(
                        image_ops, "executable", side_effect=lambda name: name
                    )
                )
                self.enterContext(
                    mock.patch.object(
                        image_ops,
                        "run_command",
                        side_effect=AssertionError("unexpected image command"),
                    )
                )
                forward.return_value.__enter__.return_value = 58080
                published = deploy.seed_tasks(
                    self.config, state / "kubeconfig", catalog_path=self.catalog_path
                )

            migrated = deploy.load_json(state / "task-seed-receipt.json")
            self.assertEqual(
                migrated["schema_version"], "frontier.deployment-task-seed-receipt/v2"
            )
            deploy.validate_seed_receipt(migrated, self.selection, "preview")
            self.assertEqual(
                migrated["task_version_ids"], self.receipt["task_version_ids"]
            )
            self.assertEqual(
                [record["id"] for record in published], self.receipt["task_version_ids"]
            )
            self.assertEqual(
                migrated["candidate_version_id"], new_candidate_id
            )
            actual_mirrors = deploy.load_json(mirror_path)
            self.assertEqual(actual_mirrors.pop("reference_candidates"), {})
            self.assertEqual(actual_mirrors, selected_mirrors)
            self.assertEqual((FIXTURES / "task-images.json").read_bytes(), mirror_bytes)
            self.assertEqual(cached.call_count, len(cache))
            apply.assert_not_called()
            self.assertEqual(commands.call_count, 3 * len(self.selection.selected))
            self.assertEqual(
                api.call_args_list,
                [
                    mock.call(
                        "http://127.0.0.1:58080", "GET", "/api/v1/candidate-versions"
                    ),
                    mock.call(
                        "http://127.0.0.1:58080", "POST", "/api/v1/candidate-versions", candidate
                    ),
                    *[
                        mock.call(
                            "http://127.0.0.1:58080", "GET", "/api/v1/task-versions"
                        )
                        for _ in self.selection.selected
                    ],
                ],
            )

    def test_fixture_backup_v2_migration_and_v1_rollback_preserve_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, backup = root / "state", root / "backup"
            shutil.copytree(FIXTURES, state)
            shutil.copytree(state, backup)
            tasks = {}
            for task in self.catalog["tasks"]:
                name = task["name"]
                document = deploy.task_version_document(
                    task, self.mirrors[name], model_domains=self.selection.model_domains
                )
                tasks[name] = {
                    "task_version_id": document["driver_config"]["task_version_id"],
                    "definition_digest": document["definition_digest"],
                }
            migrated = {
                **self.receipt,
                "schema_version": "frontier.deployment-task-seed-receipt/v2",
                **{
                    key: getattr(self.selection, key)
                    for key in (
                        "catalog_sha256",
                        "effective_catalog_sha256",
                        "selection_digest",
                        "selected",
                        "excluded",
                        "model_domains",
                    )
                },
                "tasks": tasks,
            }
            path = state / "task-seed-receipt.json"
            deploy.atomic_write(path, deploy.canonical_json(migrated))
            deploy.validate_seed_receipt(
                deploy.load_json(path), self.selection, "preview"
            )
            self.assertEqual(
                migrated["task_version_ids"],
                [tasks[name]["task_version_id"] for name in self.selection.selected],
            )
            self.assertNotEqual(path.read_bytes(), (backup / path.name).read_bytes())
            shutil.copytree(backup, state, dirs_exist_ok=True)
            for name, digest in FILE_SHA256.items():
                self.assertEqual(
                    deploy.sha256_bytes((state / name).read_bytes()), digest
                )
            deploy.validate_seed_receipt(
                deploy.load_json(path), self.selection, "preview"
            )


if __name__ == "__main__":
    unittest.main()
