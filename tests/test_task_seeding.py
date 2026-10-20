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
import frontier_deploy
from preview import cli, infra, publication
from preview import images as image_ops


def resolved_catalog(task_names=None) -> dict:
    catalog = json.loads((ROOT / "config/task-catalog.json").read_text())
    if task_names is not None:
        tasks = {task["name"]: task for task in catalog["tasks"]}
        catalog["tasks"] = [tasks[name] for name in task_names]
    catalog["candidate"]["overlay_image"]["digest"] = "sha256:" + "a" * 64
    counter = 1
    for task in catalog["tasks"]:
        for image_name in (key for key in task if key.endswith("_image")):
            image = task[image_name]
            digest = "sha256:" + format(counter, "064x")
            source_tag = "source-" + frontier_deploy.TASK_DEFINITIONS[task["name"]][
                "source"
            ]["source_set_sha256"].removeprefix("sha256:")
            if image["tag"] != source_tag and infra.source_image_tag_matches(
                image["tag"], source_tag, image["digest"]
            ):
                # A fabricated digest needs its own content-addressed fixture tag.
                image["tag"] = source_tag + "-image-" + digest[7:39]
            image["digest"] = digest
            counter += 1
        for image in task.get("sidecar_images", {}).values():
            digest = "sha256:" + format(counter, "064x")
            source_tag = frontier_deploy.TASK_DEFINITIONS[task["name"]]["image_tag"]
            if image["tag"] != source_tag and infra.source_image_tag_matches(
                image["tag"], source_tag, image["digest"]
            ):
                image["tag"] = source_tag + "-image-" + digest[7:39]
            image["digest"] = digest
            counter += 1
        if (
            task.get("reference")
            and task["reference"]["digest"] == frontier_deploy.TASK_PLACEHOLDER_DIGEST
        ):
            task["reference"]["digest"] = "sha256:" + "b" * 64
    return catalog


def mirrored_images(catalog: dict) -> dict:
    candidate_source = catalog["candidate"]["overlay_image"]
    result = {
        "candidate": {
            "overlay_image": {
                "tag": "172.20.0.50:5000/frontier/candidates/codex:source-test",
                "digest": "sha256:" + format(100, "064x"),
                "reference": (
                    "172.20.0.50:5000/frontier/candidates/codex@sha256:"
                    + format(100, "064x")
                ),
                "source_reference": candidate_source["repository"]
                + "@"
                + candidate_source["digest"],
                "source_sha256": candidate_source["digest"],
            },
        },
    }
    counter = 101
    for task in catalog["tasks"]:
        problem = task["name"]
        result[problem] = {}
        for image_name in ("agent_image", "verifier_image", "dataset_image"):
            digest = "sha256:" + format(counter, "064x")
            repository = f"172.20.0.50:5000/frontier/tasks/{problem}/{image_name}"
            result[problem][image_name] = {
                "tag": f"{repository}:source-test",
                "digest": digest,
                "reference": f"{repository}@{digest}",
                "source_reference": task[image_name]["repository"]
                + "@"
                + task[image_name]["digest"],
                "source_sha256": task[image_name]["digest"],
            }
            counter += 1
    return result


class TaskCatalogTests(unittest.TestCase):
    def test_resolved_fixture_preserves_artifact_interlock_and_rejects_tampering(
        self,
    ) -> None:
        original = json.loads((ROOT / "config/task-catalog.json").read_text())
        task = original["tasks"][0]
        source_tag = frontier_deploy.TASK_DEFINITIONS[task["name"]]["image_tag"]
        image = task["agent_image"]
        image["digest"] = "sha256:" + "e" * 64
        image["tag"] = source_tag + "-image-" + "e" * 32
        read_text = Path.read_text

        def fixture_catalog(path, *args, **kwargs):
            if path == ROOT / "config/task-catalog.json":
                return json.dumps(original)
            return read_text(path, *args, **kwargs)

        with mock.patch.object(
            Path, "read_text", autospec=True, side_effect=fixture_catalog
        ):
            fixture = resolved_catalog()
        actual = fixture["tasks"][0]["agent_image"]
        self.assertEqual(actual["tag"], source_tag + "-image-" + actual["digest"][7:39])
        frontier_deploy.validate_task_catalog(fixture, require_resolved=True)
        for field, value in (
            ("digest", "sha256:" + "d" * 64),
            ("tag", source_tag + "-image-" + "d" * 32),
            ("tag", source_tag + "-image-" + actual["digest"][7:39] + "-extra"),
        ):
            broken = copy.deepcopy(fixture)
            broken["tasks"][0]["agent_image"][field] = value
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(frontier_deploy.DeployError),
            ):
                frontier_deploy.validate_task_catalog(broken, require_resolved=True)

    def test_dataset_revision_accepts_exact_source_tag_and_preserves_legacy_v1(
        self,
    ) -> None:
        catalog = resolved_catalog()
        task = catalog["tasks"][0]
        name = task["name"]
        expected = frontier_deploy.TASK_DEFINITIONS[name]["image_tag"]
        for tag in ("v1", expected):
            task["dataset_image"]["tag"] = tag
            result = frontier_deploy.validate_task_catalog(
                catalog, require_resolved=True
            )
            actual = next(row for row in result["tasks"] if row["name"] == name)
            self.assertEqual(actual["dataset_image"]["tag"], tag)
        for tag in ("latest", "v2", "source-" + "0" * 64):
            task["dataset_image"]["tag"] = tag
            with self.subTest(tag=tag), self.assertRaises(frontier_deploy.DeployError):
                frontier_deploy.validate_task_catalog(catalog)

    def test_source_tag_binds_the_reviewed_source_hash(self) -> None:
        definitions = copy.deepcopy(frontier_deploy.TASK_DEFINITIONS)
        definitions["sparse_video_attention_policy"]["source"]["source_set_sha256"] = (
            "sha256:" + "a" * 64
        )
        with mock.patch.object(
            publication, "load_task_definitions", return_value=definitions
        ):
            with self.assertRaisesRegex(
                frontier_deploy.DeployError, "bind source_set_sha256"
            ):
                frontier_deploy.validate_task_catalog(resolved_catalog())

    def test_fixed_set_can_change_size_but_rejects_missing_and_unknown_tasks(
        self,
    ) -> None:
        catalog = resolved_catalog()
        selected = catalog["tasks"][:2]
        definitions = {
            task["name"]: frontier_deploy.TASK_DEFINITIONS[task["name"]]
            for task in selected
        }
        with mock.patch.object(
            publication, "load_task_definitions", return_value=definitions
        ):
            catalog["tasks"] = selected
            self.assertEqual(
                len(frontier_deploy.validate_task_catalog(catalog)["tasks"]), 2
            )
            catalog["tasks"] = selected[:1]
            with self.assertRaisesRegex(frontier_deploy.DeployError, "missing="):
                frontier_deploy.validate_task_catalog(catalog)
            catalog["tasks"] = [dict(selected[0], name="unreviewed_task")]
            with self.assertRaisesRegex(
                frontier_deploy.DeployError, "fixed task definitions"
            ):
                frontier_deploy.validate_task_catalog(catalog)

    def test_catalog_contains_exact_reviewed_tasks_and_refuses_placeholder_seeding(
        self,
    ) -> None:
        catalog = frontier_deploy.validate_task_catalog(
            frontier_deploy.load_json(frontier_deploy.TASK_CATALOG_PATH)
        )
        self.assertEqual(
            {task["name"] for task in catalog["tasks"]},
            set(frontier_deploy.TASK_DEFINITIONS),
        )
        self.assertEqual(
            catalog["candidate"]["overlay_image"]["repository"],
            "docker.io/joyemang33/fcs2-codex-overlay",
        )
        placeholder = json.loads((ROOT / "config/task-catalog.json").read_text())
        placeholder["tasks"][0]["agent_image"]["digest"] = (
            frontier_deploy.TASK_PLACEHOLDER_DIGEST
        )
        with self.assertRaisesRegex(
            frontier_deploy.DeployError, "placeholder image digests"
        ):
            frontier_deploy.validate_task_catalog(placeholder, require_resolved=True)

    def test_catalog_sha_and_repository_contracts_fail_closed(self) -> None:
        valid = resolved_catalog()
        validated = frontier_deploy.validate_task_catalog(valid, require_resolved=True)
        self.assertEqual(validated["unresolved_digests"], [])

        bad_sha = copy.deepcopy(valid)
        bad_sha["tasks"][0]["datasets"]["train_sha256"] = "sha256:" + "f" * 64
        with self.assertRaisesRegex(
            frontier_deploy.DeployError, "reviewed task definition"
        ):
            frontier_deploy.validate_task_catalog(bad_sha)

        bad_repository = copy.deepcopy(valid)
        bad_repository["tasks"][0]["verifier_image"]["repository"] += "-wrong"
        with self.assertRaisesRegex(
            frontier_deploy.DeployError, "repository must equal"
        ):
            frontier_deploy.validate_task_catalog(bad_repository)

        bad_tag = copy.deepcopy(valid)
        bad_tag["tasks"][0]["agent_image"]["tag"] = "latest"
        with self.assertRaises(frontier_deploy.DeployError):
            frontier_deploy.validate_task_catalog(bad_tag)

        bad_candidate = copy.deepcopy(valid)
        bad_candidate["candidate"]["overlay_image"]["repository"] += "-wrong"
        with self.assertRaisesRegex(
            frontier_deploy.DeployError, "repository must equal"
        ):
            frontier_deploy.validate_task_catalog(bad_candidate)

    def test_apply_refuses_placeholders_before_cloud_mutation(self) -> None:
        raw = json.loads((ROOT / "config/site.example.json").read_text())
        config = frontier_deploy.validate_config(raw)
        placeholder = json.loads((ROOT / "config/task-catalog.json").read_text())
        placeholder["tasks"][0]["agent_image"]["digest"] = (
            frontier_deploy.TASK_PLACEHOLDER_DIGEST
        )
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "task-catalog.json"
            catalog_path.write_text(json.dumps(placeholder))
            with (
                mock.patch.object(publication, "TASK_CATALOG_PATH", catalog_path),
                mock.patch.object(
                    cli, "render_bundle", return_value=(config, Path("/tmp/render"))
                ),
                mock.patch.object(cli, "modal_credentials", return_value={}),
                mock.patch.object(cli, "aws_identity") as aws_identity,
            ):
                with self.assertRaisesRegex(
                    frontier_deploy.DeployError, "placeholder image digests"
                ):
                    frontier_deploy.apply_site(
                        ROOT / "config/site.example.json",
                        config["site"]["expected_account_id"],
                        config["site"]["name"],
                    )
        aws_identity.assert_not_called()


class TaskDocumentTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = json.loads((ROOT / "config/site.example.json").read_text())
        self.config = frontier_deploy.validate_config(raw)
        self.catalog = frontier_deploy.validate_task_catalog(
            resolved_catalog(), require_resolved=True
        )
        self.mirrors = mirrored_images(self.catalog)

    def test_sparse_video_preserves_gpu_policy_and_json_dataset(self) -> None:
        task = next(
            item
            for item in self.catalog["tasks"]
            if item["name"] == "sparse_video_attention_policy"
        )
        document = frontier_deploy.task_version_document(
            task, self.mirrors[task["name"]]
        )
        config = document["driver_config"]
        self.assertEqual(
            document["task_id"], "frontier-cs-2.0-sparse-video-attention-policy"
        )
        self.assertEqual(config["quota"]["agent_gpu"]["allowed_types"], ["H100"])
        self.assertEqual(config["quota"]["agent_gpu"]["max_gpu_seconds"], 54000)
        self.assertEqual(config["quota"]["verifier"]["max_gpu_instances"], 4)
        self.assertEqual(config["quota"]["verifier"]["eval_timeout_seconds"], 12600)
        self.assertEqual(
            config["images"]["verifier"]["gpu_digest"],
            task["verifier_gpu_image"]["digest"],
        )
        self.assertTrue(
            config["images"]["verifier"]["gpu_tag"].startswith("docker.io/")
        )
        self.assertTrue(config["images"]["agent_gpu"]["tag"].startswith("docker.io/"))
        self.assertTrue(config["datasets"]["eval"]["uri"].endswith(".json"))
        self.assertEqual(config["network"]["audit"]["max_bundle_bytes"], 32 * 1024**3)
        self.assertEqual(config["network"]["audit"]["max_model_sessions"], 1024)
        self.assertEqual(
            config["network"]["audit"]["ssh_session_budgets"],
            {"agent": 1024, "verifier_train": 50, "verifier_eval": 2},
        )
        changed = copy.deepcopy(task)
        changed["verifier_gpu_image"]["digest"] = "sha256:" + "e" * 64
        newer = frontier_deploy.task_version_document(
            changed, self.mirrors[task["name"]]
        )
        self.assertNotEqual(
            config["task_version_id"], newer["driver_config"]["task_version_id"]
        )

    def test_graph_task_version_matches_publication_shape(self) -> None:
        task = next(
            task
            for task in self.catalog["tasks"]
            if task["name"] == "graph_mining_method_selection"
        )
        document = frontier_deploy.task_version_document(
            task, self.mirrors[task["name"]]
        )
        config = document["driver_config"]

        self.assertRegex(
            config["task_version_id"], r"^taskver_[0-7][0-9A-HJKMNP-TV-Z]{25}$"
        )
        self.assertEqual(config["schema_version"], "frontier.task-config/v1")
        mirrors = self.mirrors[task["name"]]
        self.assertEqual(
            config["images"]["agent_cpu_base"]["reference"],
            mirrors["agent_image"]["reference"],
        )
        self.assertEqual(
            config["images"]["verifier"]["reference"],
            mirrors["verifier_image"]["reference"],
        )
        self.assertEqual(config["submission"]["entrypoint"], "payload/solution.patch")
        self.assertEqual(config["verifier"]["command"], ["/opt/frontier/verifier"])
        self.assertEqual(
            config["verifier"]["evidence_path"], "/frontier/output/evidence.tar.zst"
        )
        self.assertEqual(config["network"]["allowed_domains"], ["openrouter.ai"])
        for mode in ("train", "eval"):
            dataset = config["datasets"][mode]
            self.assertIn(
                f"/graph_mining_method_selection/{mode}/sha256/",
                dataset["uri"],
            )
            self.assertTrue(dataset["uri"].endswith(dataset["version"][7:] + ".json"))
        self.assertEqual(
            document["definition_digest"],
            frontier_deploy.canonical_object_sha256(config),
        )
        self.assertEqual(
            document["metadata"]["frontier_cs_2_0"]["policy_schema_version"],
            "frontier.frontiersmith-task-policy/v1",
        )
        self.assertEqual(
            document,
            frontier_deploy.task_version_document(task, self.mirrors[task["name"]]),
        )

    def test_codex_candidate_version_pins_public_overlay(self) -> None:
        mirrored_overlay = self.mirrors["candidate"]["overlay_image"]
        document = frontier_deploy.candidate_version_document(
            self.catalog["candidate"], mirrored_overlay
        )
        config = document["driver_config"]

        self.assertRegex(
            config["candidate_version_id"], r"^candver_[0-7][0-9A-HJKMNP-TV-Z]{25}$"
        )
        self.assertEqual(
            config["overlay_image"]["reference"], mirrored_overlay["reference"]
        )
        self.assertEqual(config["overlay_image"]["digest"], mirrored_overlay["digest"])
        self.assertEqual(
            config["execution"]["entrypoint"], ["/opt/frontier/candidate/launch"]
        )
        self.assertNotIn("FRONTIER_CODEX_TIMEOUT_SECONDS", config["execution"]["env"])
        self.assertNotIn("FRONTIER_TRAIN_POLL_TIMEOUT_SECONDS", config["execution"]["env"])
        self.assertEqual(
            config["model_api"], {"allowed_domains": [], "secret_refs": []}
        )
        self.assertEqual(
            document["definition_digest"],
            frontier_deploy.canonical_object_sha256(config),
        )

    def test_dataset_seed_job_copies_content_addressed_tar_to_seed_pvc(self) -> None:
        task = self.catalog["tasks"][0]
        runner = "172.20.0.50:5000/frontier/storage-controller@sha256:" + "a" * 64
        encoded = frontier_deploy.render_task_dataset_seed_manifest(
            self.config,
            task,
            self.mirrors[task["name"]],
            runner,
        )
        job = json.loads(encoded)["items"][0]
        from test_frontier_deploy import Fixture
        bootstrap = json.loads(frontier_deploy.render_bootstrap_manifest(
            self.config, Fixture.outputs(self.config)))
        policy = next(item for item in bootstrap["items"]
                      if item["metadata"]["name"] == "frontier-registry-boundary")
        seed_peer = next(peer for rule in policy["spec"]["ingress"]
                         for peer in rule["from"]
                         if peer.get("podSelector", {}).get("matchLabels", {}).get(
                             "app.kubernetes.io/name") == "frontier-task-seed")
        self.assertEqual(job["metadata"]["namespace"], policy["metadata"]["namespace"])
        self.assertNotIn("namespaceSelector", seed_peer)
        self.assertLessEqual(seed_peer["podSelector"]["matchLabels"].items(),
                             job["spec"]["template"]["metadata"]["labels"].items())
        seed_egress = next(item for item in bootstrap["items"]
                           if item["metadata"]["name"] == "frontier-task-seed-registry-egress")
        self.assertEqual(seed_egress["spec"]["podSelector"], seed_peer["podSelector"])
        pod = job["spec"]["template"]["spec"]
        container = pod["containers"][0]
        self.assertEqual(container["image"], runner)
        self.assertEqual(container["command"][:2], ["/usr/local/bin/python3", "-c"])
        self.assertEqual(
            pod["volumes"],
            [
                {
                    "name": "data",
                    "persistentVolumeClaim": {"claimName": "frontier-object-seed"},
                }
            ],
        )
        script = container["command"][2]
        self.assertIn(self.mirrors[task["name"]]["dataset_image"]["digest"], script)
        for mode in ("train", "eval"):
            digest = task["datasets"][f"{mode}_sha256"][7:]
            self.assertIn(f"datasets/{mode}/sha256/{digest}.json", script)
            self.assertIn(
                f"/data/datasets/frontier-cs-2.0-demo/{task['name']}/{mode}/sha256/{digest}.json",
                script,
            )
        self.assertIn("hashlib.sha256", script)

    def test_task_images_are_mirrored_from_source_digests(self) -> None:
        self.config["features"]["modal_gpu"] = True
        registry_forward = mock.MagicMock()
        registry_forward.return_value.__enter__.return_value = 5500
        local_digest = "sha256:" + "f" * 64
        self.catalog["candidate"]["overlay_image"]["digest"] = local_digest
        self.catalog["candidate"]["overlay_image"]["reference"] = (
            self.catalog["candidate"]["overlay_image"]["repository"] + "@" + local_digest
        )
        references = publication.load_manifest()
        for task in self.catalog["tasks"]:
            task["reference"] = references.get(task["name"])
        for image in references.values():
            image["digest"] = "sha256:" + "d" * 64
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(image_ops, "executable", return_value="docker"),
                mock.patch.object(image_ops, "port_forward", registry_forward),
                mock.patch.object(image_ops, "run_command") as run_command,
                mock.patch.object(
                    image_ops, "cached_registry_image", return_value=None
                ),
                mock.patch.object(
                    image_ops, "registry_digest", return_value=local_digest
                ),
                mock.patch.object(
                    image_ops, "state_directory", return_value=Path(directory)
                ),
            ):
                records = frontier_deploy.mirror_task_images(
                    self.config,
                    Path("/tmp/kubeconfig"),
                    self.catalog,
                    selection=frontier_deploy.select_tasks(
                        self.config,
                        resolved_catalog(),
                        frontier_deploy.TASK_DEFINITIONS,
                    ),
                )

        pulls = [
            call.args[0]
            for call in run_command.call_args_list
            if call.args[0][1:3] == ["pull", "--platform"]
        ]
        self.assertEqual(
            len(pulls), 1 + len(references) + 3 * len(self.catalog["tasks"])
        )
        self.assertTrue(all(command[3] == "linux/amd64" for command in pulls))
        self.assertTrue(all("@sha256:" in command[4] for command in pulls))
        self.assertEqual(
            set(records),
            {"candidate", "reference_candidates", *frontier_deploy.TASK_DEFINITIONS},
        )
        for problem_records in records.values():
            for kind, record in problem_records.items():
                if kind == "sidecar_images":
                    # Declared sidecars stay public Modal-side images, one
                    # record per sidecar, exactly like the GPU roles.
                    self.assertTrue(record)
                    for sidecar_record in record.values():
                        self.assertTrue(
                            sidecar_record["reference"].startswith("docker.io/")
                        )
                        self.assertNotEqual(sidecar_record["digest"], local_digest)
                    continue
                if kind in {"agent_gpu_image", "verifier_gpu_image"}:
                    self.assertTrue(record["reference"].startswith("docker.io/"))
                    self.assertNotEqual(record["digest"], local_digest)
                    continue
                self.assertEqual(record["digest"], local_digest)
                self.assertTrue(record["reference"].endswith("@" + local_digest))

    def test_existing_task_version_must_match_the_entire_request(self) -> None:
        task = self.catalog["tasks"][0]
        document = frontier_deploy.task_version_document(
            task, self.mirrors[task["name"]]
        )
        version_id = document["driver_config"]["task_version_id"]
        existing = {**document, "id": version_id, "display_name": "changed"}
        with mock.patch.object(publication, "api_json", return_value=[existing]) as api:
            with self.assertRaisesRegex(
                frontier_deploy.DeployError, "immutable version differs"
            ):
                frontier_deploy.ensure_version(
                    "http://127.0.0.1:8080",
                    "/api/v1/task-versions",
                    version_id,
                    document,
                )
        api.assert_called_once_with(
            "http://127.0.0.1:8080",
            "GET",
            "/api/v1/task-versions",
        )


class TaskSeedingIdempotencyTests(unittest.TestCase):
    def test_gpu_replay_uses_native_gpu_and_train_final_without_model_calls(
        self,
    ) -> None:
        catalog = frontier_deploy.validate_task_catalog(
            resolved_catalog(), require_resolved=True
        )
        overlay = mirrored_images(catalog)["candidate"]["overlay_image"]
        patch = b"diff --git a/a b/a\n"
        first = frontier_deploy.gpu_smoke_candidate(
            catalog["candidate"], overlay, patch
        )
        second = frontier_deploy.gpu_smoke_candidate(
            catalog["candidate"], overlay, patch
        )
        self.assertEqual(first, second)
        config = first["driver_config"]
        script = config["execution"]["args"][0]
        compile(script, "gpu-replay", "exec")
        self.assertIn("'/frontier/task/frontier-gpu', 'build'", script)
        self.assertIn(
            "'frontier-submit'", script.replace("/opt/frontier/candidate/bin/", "")
        )
        self.assertIn("'train'", script)
        self.assertIn("'final'", script)
        self.assertEqual(config["execution"]["env"]["FRONTIER_GPU_TYPE"], "H100")
        self.assertEqual(
            config["model_api"], {"allowed_domains": [], "secret_refs": []}
        )
        self.assertEqual(
            first["metadata"]["model_execution"], "archived-submission-replay"
        )
        changed = frontier_deploy.gpu_smoke_candidate(
            catalog["candidate"], overlay, patch + b"\n"
        )
        self.assertNotEqual(
            config["candidate_version_id"],
            changed["driver_config"]["candidate_version_id"],
        )
        with self.assertRaises(frontier_deploy.DeployError):
            frontier_deploy.gpu_smoke_candidate(catalog["candidate"], overlay, b"")

    def test_local_receipt_is_bound_to_config_and_observed_sizes(self) -> None:
        catalog = frontier_deploy.validate_task_catalog(
            resolved_catalog(), require_resolved=True
        )
        task = next(
            task
            for task in catalog["tasks"]
            if task["name"] == "sparse_video_attention_policy"
        )
        document = frontier_deploy.task_version_document(
            task, mirrored_images(catalog)[task["name"]]
        )
        sizes = {
            value.removeprefix("sha256:"): 445 for value in task["datasets"].values()
        }
        receipt = frontier_deploy.local_author_receipt(
            document, sizes, {"source_set_sha256": "sha256:" + "a" * 64}
        )
        self.assertEqual(receipt["definition"]["digest"], document["definition_digest"])
        self.assertEqual(
            receipt["images"]["verifier"]["gpu_digest"],
            task["verifier_gpu_image"]["digest"],
        )
        self.assertNotIn("platform", receipt["images"]["agent_cpu_base"])
        self.assertEqual(
            document["driver_config"]["network"]["audit"][
                "max_ssh_bytes_per_direction"
            ],
            512 * 1024**2,
        )
        for mode, dataset in receipt["datasets"].items():
            self.assertEqual(
                dataset["s3_version_id"],
                "local-" + document["driver_config"]["datasets"][mode]["version"],
            )
            self.assertEqual(dataset["size_bytes"], 445)
        with self.assertRaises(frontier_deploy.DeployError):
            frontier_deploy.local_author_receipt(document, {}, {})

    def test_rerun_reuses_task_versions_and_creates_fresh_jobs(self) -> None:
        raw_config = json.loads((ROOT / "config/site.example.json").read_text())
        raw_config["execution"]["model_domains"] = ["api.moonshot.ai", "openrouter.ai"]
        raw_config["features"]["modal_gpu"] = True
        config = frontier_deploy.validate_config(raw_config)
        raw_catalog = resolved_catalog()
        validated_catalog = frontier_deploy.validate_task_catalog(
            raw_catalog, require_resolved=True
        )
        mirrors = mirrored_images(validated_catalog)
        stored: dict[str, dict[str, dict]] = {
            "/api/v1/candidate-versions": {},
            "/api/v1/task-versions": {},
        }

        def api_json(_base, method, path, body=None, headers=None):
            self.assertIn(path, stored)
            self.assertIsNone(headers)
            if method == "GET":
                return list(stored[path].values())
            self.assertEqual(method, "POST")
            version_id = body["driver_config"].get(
                "task_version_id",
                body["driver_config"].get("candidate_version_id"),
            )
            created = {
                key: value for key, value in body.items() if key != "author_receipt"
            }
            created["id"] = version_id
            stored[path][version_id] = created
            return created

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(raw_catalog))
            management_forward = mock.MagicMock()
            management_forward.return_value.__enter__.return_value = 58080
            with (
                mock.patch.object(
                    publication, "mirror_task_images", return_value=mirrors
                ) as mirror,
                mock.patch.object(publication, "kubectl_apply") as kubectl_apply,
                mock.patch.object(
                    publication,
                    "run_command",
                    return_value=json.dumps(
                        {
                            "dataset_sizes": {
                                value.removeprefix("sha256:"): 123
                                for task in raw_catalog["tasks"]
                                for value in task["datasets"].values()
                            }
                        }
                    ),
                ) as run_command,
                mock.patch.object(publication, "api_json", side_effect=api_json) as api,
                mock.patch.object(publication, "port_forward", management_forward),
                mock.patch.object(publication, "executable", return_value="kubectl"),
                mock.patch.object(
                    publication, "state_directory", return_value=root / "state"
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
            ):
                first = frontier_deploy.seed_tasks(
                    config, Path("/tmp/kubeconfig"), catalog_path
                )
                second = frontier_deploy.seed_tasks(
                    config, Path("/tmp/kubeconfig"), catalog_path
                )

        self.assertEqual(
            [item["id"] for item in first], [item["id"] for item in second]
        )
        self.assertEqual(len(stored["/api/v1/candidate-versions"]), 1)
        count = len(raw_catalog["tasks"])
        self.assertEqual(len(stored["/api/v1/task-versions"]), count)
        for task in stored["/api/v1/task-versions"].values():
            self.assertEqual(
                task["driver_config"]["network"]["allowed_domains"],
                ["api.moonshot.ai", "openrouter.ai"],
            )
        self.assertEqual(
            sum(call.args[1] == "POST" for call in api.call_args_list), count + 1
        )
        self.assertEqual(
            sum(call.args[1] == "GET" for call in api.call_args_list), 2 * (count + 1)
        )
        kubectl_apply.assert_not_called()
        self.assertEqual(run_command.call_count, 6 * count)
        self.assertEqual(mirror.call_count, 2)
        commands = [call.args[0] for call in run_command.call_args_list]
        self.assertFalse(any("delete" in command for command in commands))
        self.assertEqual(sum("create" in command for command in commands), 2 * count)
        self.assertEqual(
            sum("--for=condition=complete" in command for command in commands),
            2 * count,
        )


if __name__ == "__main__":
    unittest.main()


class DeclaredSidecarDocumentTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = json.loads((ROOT / "config/site.example.json").read_text())
        self.config = frontier_deploy.validate_config(raw)
        self.definitions = copy.deepcopy(publication.load_task_definitions())
        self.raw = resolved_catalog()
        task = next(
            item
            for item in self.raw["tasks"]
            if item["name"] == "sparse_video_attention_policy"
        )
        self.name = task["name"]
        definition = self.definitions[self.name]
        definition["sidecars"] = {
            "scheduler": {
                "args": ["/opt/frontier/scheduler"],
                "workdir": "/opt/frontier",
                "phases": ["train", "eval"],
                "attestation_key": True,
            },
            "swe-000": {
                "args": ["bash", "-lc", "exec sleep infinity"],
                "workdir": "/testbed",
                "phases": ["eval"],
            },
            "swe-000-check": {
                "args": ["bash", "-lc", "exec sleep infinity"],
                "workdir": "/testbed",
                "phases": ["eval"],
                "image": "swe-000",
            },
        }
        definition["gpu_parent_resources"] = {
            "verifier": {"cpu_millis": 16000, "memory_bytes": 64 * 1024**3}
        }
        namespace = task["verifier_image"]["repository"].rsplit("/", 1)[0]
        task["sidecar_images"] = {
            sidecar: {
                "repository": f"{namespace}/fcs2-task-{self.name}-sidecar-{sidecar}",
                "tag": task["verifier_image"]["tag"],
                "digest": "sha256:" + format(200 + index, "064x"),
            }
            for index, sidecar in enumerate(("scheduler", "swe-000"))
        }

    def task(self, catalog):
        return next(item for item in catalog["tasks"] if item["name"] == self.name)

    def test_sidecars_render_into_the_task_config_and_author_receipt(self) -> None:
        catalog = frontier_deploy.validate_task_catalog(
            self.raw, require_resolved=True, definitions=self.definitions
        )
        task = self.task(catalog)
        self.assertEqual(set(task["sidecar_images"]), {"scheduler", "swe-000"})
        mirrors = mirrored_images(catalog)
        document = frontier_deploy.task_version_document(
            task, mirrors[self.name], definitions=self.definitions
        )
        config = document["driver_config"]
        sidecars = config["images"]["gpu_sidecars"]
        self.assertEqual(list(sidecars), ["scheduler", "swe-000", "swe-000-check"])
        self.assertEqual(sidecars["swe-000-check"]["image"], sidecars["swe-000"]["image"])
        self.assertEqual(
            sidecars["swe-000"]["image"]["tag"],
            f"{task['sidecar_images']['swe-000']['repository']}:{task['sidecar_images']['swe-000']['tag']}",
        )
        self.assertEqual(
            sidecars["swe-000"]["image"]["digest"], "sha256:" + format(201, "064x")
        )
        self.assertEqual(sidecars["swe-000"]["phases"], ["eval"])
        self.assertNotIn("attestation_key", sidecars["swe-000"])
        self.assertTrue(sidecars["scheduler"]["attestation_key"])
        self.assertEqual(sidecars["scheduler"]["args"], ["/opt/frontier/scheduler"])
        self.assertEqual(
            config["images"]["gpu_parent_resources"],
            {"verifier": {"cpu_millis": 16000, "memory_bytes": 64 * 1024**3}},
        )
        # A different sidecar digest is a different immutable TaskVersion.
        changed = copy.deepcopy(task)
        changed["sidecar_images"]["swe-000"]["digest"] = "sha256:" + "e" * 64
        newer = frontier_deploy.task_version_document(
            changed, mirrors[self.name], definitions=self.definitions
        )
        self.assertNotEqual(
            config["task_version_id"], newer["driver_config"]["task_version_id"]
        )
        sizes = {
            dataset["version"].removeprefix("sha256:"): 1
            for dataset in config["datasets"].values()
        }
        receipt = publication.local_author_receipt(
            document, sizes, {"source_set_sha256": "sha256:" + "c" * 64}
        )
        self.assertEqual(
            receipt["images"]["gpu_sidecars"],
            {
                name: profile["image"]
                for name, profile in sidecars.items()
            },
        )
        self.assertNotIn("gpu_parent_resources", receipt["images"])
        # The sidecar images are public Modal-side images: recorded, not mirrored.
        self.assertEqual(
            set(mirrors[self.name]) & {"sidecar_images"}, set()
        )

    def test_catalog_rejects_wrong_or_missing_sidecar_images(self) -> None:
        broken = copy.deepcopy(self.raw)
        self.task(broken)["sidecar_images"]["swe-000"]["repository"] += "-other"
        with self.assertRaises(infra.DeployError):
            frontier_deploy.validate_task_catalog(broken, definitions=self.definitions)
        missing = copy.deepcopy(self.raw)
        del self.task(missing)["sidecar_images"]
        with self.assertRaises(infra.DeployError):
            frontier_deploy.validate_task_catalog(missing, definitions=self.definitions)
        extra = copy.deepcopy(self.raw)
        self.task(extra)["sidecar_images"]["swe-999"] = dict(
            self.task(extra)["sidecar_images"]["swe-000"]
        )
        with self.assertRaises(infra.DeployError):
            frontier_deploy.validate_task_catalog(extra, definitions=self.definitions)
