from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import http.client
import io
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "scripts")]
import frontier_deploy as deploy
from preview import acceptance, cli, infra, publication
from preview import config as site_config
from preview import images as image_ops
from test_task_seeding import mirrored_images, resolved_catalog


class SeedSelectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.state = self.root / "state"
        self.state.mkdir()
        self.config = deploy.load_json(ROOT / "config/site.example.json")
        self.config["features"]["modal_gpu"] = False
        self.config_path = self.root / "site.json"
        self.config_path.write_text(json.dumps(self.config))
        self.raw = resolved_catalog()
        self.catalog_path = self.root / "catalog.json"
        self.catalog_path.write_text(json.dumps(self.raw))
        self.catalog, self.selection = deploy.load_task_selection(
            self.config, self.catalog_path, require_resolved=True
        )
        self.mirrors = mirrored_images(self.catalog)
        # The shared document/golden fixture predates mirror_task_images' repository layout.
        for task in self.catalog["tasks"]:
            name = task["name"]
            for kind, record in self.mirrors[name].items():
                repository = (
                    f"{self.config['network']['registry_cluster_ip']}:5000/frontier/tasks/"
                    f"{name.replace('_', '-')}/{kind.removesuffix('_image').replace('_', '-')}"
                )
                record["reference"] = f"{repository}@{record['digest']}"
                record["tag"] = f"{repository}:source-test"
        (self.state / "task-images.json").write_bytes(
            deploy.canonical_json(self.mirrors)
        )
        (self.state / "kubeconfig").write_text("fixture")
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(
            publication, "state_directory", return_value=self.state
        ).start()
        forward = mock.patch.object(publication, "port_forward").start()
        forward.return_value.__enter__.return_value = 58080
        mock.patch.object(image_ops, "port_forward", forward).start()
        mock.patch.object(acceptance, "port_forward", forward).start()
        mock.patch.object(image_ops, "state_directory", return_value=self.state).start()
        mock.patch.object(
            acceptance, "state_directory", return_value=self.state
        ).start()
        mock.patch.object(cli, "state_directory", return_value=self.state).start()
        self.forward = forward
        self.stdout = mock.patch("sys.stdout", new_callable=io.StringIO).start()
        # Capacity/admission HTTP contracts are covered in test_v3_runtime;
        # these tests isolate immutable publication and its GET-only behavior.
        self.real_preflight = deploy.runtime_admission_preflight
        self.preflight = mock.patch.object(
            publication, "runtime_admission_preflight"
        ).start()

    def documents(self):
        candidate = deploy.candidate_version_document(
            self.catalog["candidate"], self.mirrors["candidate"]["overlay_image"]
        )
        documents = [("candidate-versions", "candidate_version_id", candidate)]
        documents.extend(
            (
                "task-versions",
                "task_version_id",
                deploy.task_version_document(
                    task,
                    self.mirrors[task["name"]],
                    model_domains=self.selection.model_domains,
                ),
            )
            for task in self.catalog["tasks"]
        )
        collections = {}
        for kind, key, doc in documents:
            collections.setdefault(f"/api/v1/{kind}", []).append(
                {**doc, "id": doc["driver_config"][key]}
            )
        return collections

    def state_bytes(self):
        return {
            p.relative_to(self.state).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in self.state.rglob("*")
            if p.is_file()
        }

    def test_check_cli_only_gets_existing_ids_without_writes_or_credentials(self):
        documents = self.documents()
        before = self.state_bytes()

        def get(base, method, path):
            self.assertEqual(method, "GET")
            return [{"id": "unrelated-version"}, *reversed(documents[path])]

        with (
            mock.patch.object(publication, "api_json", side_effect=get) as api,
            mock.patch.object(
                publication, "atomic_write", side_effect=AssertionError("write")
            ),
            mock.patch.object(
                publication, "mirror_task_images", side_effect=AssertionError("mirror")
            ),
            mock.patch.object(
                publication, "run_command", side_effect=AssertionError("command")
            ),
            mock.patch.object(
                cli, "modal_credentials", side_effect=AssertionError("credentials")
            ),
            mock.patch.object(
                cli, "update_kubeconfig", side_effect=AssertionError("kubeconfig write")
            ),
            mock.patch.object(
                cli, "platform_outputs", side_effect=AssertionError("outputs write")
            ),
            mock.patch.object(
                cli, "render_bundle", side_effect=AssertionError("render write")
            ),
        ):
            self.assertEqual(
                deploy.main(
                    [
                        "seed-tasks",
                        "--check",
                        "--config",
                        str(self.config_path),
                        "--catalog",
                        str(self.catalog_path),
                    ]
                ),
                0,
            )
        self.assertEqual(api.call_count, len(self.selection.selected) + 1)
        self.preflight.assert_called_once_with(
            deploy.validate_config(copy.deepcopy(self.config)),
            "http://127.0.0.1:58080",
            informational=True,
        )
        self.assertEqual(self.state_bytes(), before)
        self.assertEqual(
            self.stdout.getvalue().count(": MATCH"), len(self.selection.selected) + 1
        )
        self.assertEqual(
            api.call_args_list[0],
            mock.call("http://127.0.0.1:58080", "GET", "/api/v1/candidate-versions"),
        )

    def test_check_reports_missing_and_different_versions_and_fails_after_all_gets(
        self,
    ):
        documents = self.documents()
        documents["/api/v1/candidate-versions"] = []
        tasks = documents["/api/v1/task-versions"]
        tasks[0]["definition_digest"] = "sha256:" + "0" * 64
        tasks.pop()
        before = self.state_bytes()

        def get(base, method, path):
            self.assertEqual(method, "GET")
            return documents[path]

        with mock.patch.object(publication, "api_json", side_effect=get) as api:
            with self.assertRaisesRegex(deploy.DeployError, "missing or differ"):
                deploy.seed_tasks(
                    self.config,
                    self.state / "kubeconfig",
                    self.catalog_path,
                    check=True,
                )
        self.assertEqual(api.call_count, len(self.selection.selected) + 1)
        self.assertEqual(self.stdout.getvalue().count(": MISSING"), 2)
        self.assertIn(": DIFFERS", self.stdout.getvalue())
        self.assertEqual(self.state_bytes(), before)

    def test_check_collection_404_is_an_error_not_a_missing_version(self):
        documents = self.documents()
        before = self.state_bytes()
        for endpoint in documents:
            with self.subTest(endpoint=endpoint):
                self.stdout.seek(0)
                self.stdout.truncate()

                def get(base, method, path, *, endpoint=endpoint):
                    self.assertEqual(method, "GET")
                    if path == endpoint:
                        error = HTTPError(
                            base + path, 404, "missing collection", {}, None
                        )
                        error.close()
                        raise deploy.DeployError(
                            "collection unavailable: 404"
                        ) from error
                    return documents[path]

                with mock.patch.object(publication, "api_json", side_effect=get):
                    with self.assertRaisesRegex(
                        deploy.DeployError, "collection unavailable: 404"
                    ):
                        deploy.seed_tasks(
                            self.config,
                            self.state / "kubeconfig",
                            self.catalog_path,
                            check=True,
                        )
                self.assertNotIn(": MISSING", self.stdout.getvalue())
                self.assertEqual(self.state_bytes(), before)

    def test_capacity_diagnostics_do_not_mask_seed_mismatches_or_write_state(self):
        documents = self.documents()
        documents["/api/v1/candidate-versions"] = []
        snapshot = deploy.load_json(
            ROOT / "tests/fixtures/evidence/v3-runtime-incident-2026-09-12.json"
        )["observed"]["scheduler_capacity_snapshot"]
        capacity_path = "/api/v1/infrastructure/scheduler-capacity"
        before = self.state_bytes()
        for capacity in (
            snapshot,
            {},
            *(
                deploy.ManagementHTTPError(code, capacity_path, "unavailable")
                for code in (404, 503, 500)
            ),
        ):
            with self.subTest(capacity=capacity):

                def get(base, method, path, *, capacity=capacity):
                    self.assertEqual(method, "GET")
                    if path == capacity_path:
                        if isinstance(capacity, Exception):
                            raise capacity
                        return capacity
                    return documents[path]

                with (
                    mock.patch.object(
                        publication,
                        "runtime_admission_preflight",
                        wraps=self.real_preflight,
                    ),
                    mock.patch.object(
                        infra,
                        "api_json_with_headers",
                        return_value=(
                            {},
                            {
                                "x-management-driver-mode": "kubernetes",
                                "x-management-accepting-evaluations": "true",
                            },
                        ),
                    ),
                    mock.patch.object(publication, "api_json", side_effect=get),
                    mock.patch.object(infra, "api_json", side_effect=get),
                    mock.patch.object(
                        publication, "atomic_write", side_effect=AssertionError("write")
                    ),
                ):
                    with self.assertRaisesRegex(
                        deploy.DeployError,
                        "seed check failed: existing versions are missing or differ",
                    ):
                        deploy.seed_tasks(
                            self.config,
                            self.state / "kubeconfig",
                            self.catalog_path,
                            check=True,
                        )
                self.assertEqual(self.state_bytes(), before)
        self.assertIn("retained BLOCKED reservations=3", self.stdout.getvalue())
        self.assertIn(
            "capacity endpoint unavailable (pre-V3 runtime or non-kubernetes mode)",
            self.stdout.getvalue(),
        )

    def test_truncated_http_response_diagnostics_preserve_seed_verdict(self):
        overview_path = "/api/v1/overview"
        capacity_path = "/api/v1/infrastructure/scheduler-capacity"
        snapshot = deploy.load_json(
            ROOT / "tests/fixtures/evidence/v3-runtime-incident-2026-09-12.json"
        )["observed"]["scheduler_capacity_snapshot"]
        before = self.state_bytes()
        for truncated_path in (overview_path, capacity_path):
            for missing in (False, True):
                with self.subTest(truncated_path=truncated_path, missing=missing):
                    self.stdout.seek(0)
                    self.stdout.truncate()
                    documents = self.documents()
                    if missing:
                        documents["/api/v1/candidate-versions"] = []
                    requested = []

                    def urlopen(
                        request,
                        *,
                        documents=documents,
                        requested=requested,
                        truncated_path=truncated_path,
                        **kwargs,
                    ):
                        self.assertEqual(request.get_method(), "GET")
                        path = request.selector
                        requested.append(path)
                        headers = (
                            b"HTTP/1.1 200 OK\r\n"
                            b"X-Management-Driver-Mode: kubernetes\r\n"
                            b"X-Management-Accepting-Evaluations: true\r\n"
                        )
                        if path == truncated_path:
                            # Real HTTPResponse chunk parsing raises IncompleteRead
                            # even when the client bounds response.read(size).
                            wire = (
                                headers
                                + b'Transfer-Encoding: chunked\r\n\r\n20\r\n{"cut":'
                            )
                        else:
                            payload = json.dumps(
                                snapshot
                                if path == capacity_path
                                else {}
                                if path == overview_path
                                else documents[path]
                            ).encode()
                            wire = (
                                headers
                                + f"Content-Length: {len(payload)}\r\n\r\n".encode()
                                + payload
                            )
                        connection = mock.Mock()
                        connection.makefile.return_value = io.BytesIO(wire)
                        response = http.client.HTTPResponse(connection)
                        response.begin()
                        return response

                    with (
                        mock.patch.object(
                            publication,
                            "runtime_admission_preflight",
                            wraps=self.real_preflight,
                        ),
                        mock.patch.object(
                            infra.urllib_request, "urlopen", side_effect=urlopen
                        ),
                        mock.patch.object(
                            publication,
                            "atomic_write",
                            side_effect=AssertionError("write"),
                        ),
                    ):
                        if missing:
                            with self.assertRaisesRegex(
                                deploy.DeployError,
                                "seed check failed: existing versions are missing or differ",
                            ):
                                deploy.seed_tasks(
                                    self.config,
                                    self.state / "kubeconfig",
                                    self.catalog_path,
                                    check=True,
                                )
                        else:
                            deploy.seed_tasks(
                                self.config,
                                self.state / "kubeconfig",
                                self.catalog_path,
                                check=True,
                            )
                    self.assertIn(
                        "runtime preflight (informational): Management request failed",
                        self.stdout.getvalue(),
                    )
                    self.assertIn("IncompleteRead", self.stdout.getvalue())
                    self.assertIn(overview_path, requested)
                    self.assertIn(capacity_path, requested)
                    self.assertEqual(
                        self.stdout.getvalue().count(": MISSING"), int(missing)
                    )
                    self.assertEqual(
                        self.stdout.getvalue().count(": MATCH"),
                        len(self.selection.selected) + (not missing),
                    )
                    self.assertEqual(self.state_bytes(), before)

    def test_check_rejects_malformed_or_duplicate_collection_matches(self):
        candidate = self.documents()["/api/v1/candidate-versions"][0]
        for response, message in (
            ({}, "version list is malformed"),
            ([candidate, candidate], "duplicate version"),
        ):
            with self.subTest(message=message):
                with mock.patch.object(
                    publication, "api_json", return_value=response
                ) as api:
                    with self.assertRaisesRegex(deploy.DeployError, message):
                        deploy.seed_tasks(
                            self.config,
                            self.state / "kubeconfig",
                            self.catalog_path,
                            check=True,
                        )
                api.assert_called_once_with(
                    "http://127.0.0.1:58080", "GET", "/api/v1/candidate-versions"
                )

    def test_check_refuses_stale_mirrors_before_http(self):
        self.mirrors[self.selection.selected[0]]["agent_image"]["source_sha256"] = (
            "sha256:" + "0" * 64
        )
        (self.state / "task-images.json").write_bytes(
            deploy.canonical_json(self.mirrors)
        )
        with mock.patch.object(publication, "api_json") as api:
            with self.assertRaisesRegex(deploy.DeployError, "saved mirror differs"):
                deploy.seed_tasks(
                    self.config,
                    self.state / "kubeconfig",
                    self.catalog_path,
                    check=True,
                )
        api.assert_not_called()
        self.forward.assert_not_called()

    def test_seed_writes_v2_and_only_selected_tasks_with_site_domains(self):
        self.assert_seed_with_model_domains(["model.example", "model.example"])

    def test_seed_preserves_empty_authored_network_with_no_model_domains(self):
        self.assert_seed_with_model_domains([])

    def assert_seed_with_model_domains(self, domains):
        self.config["execution"]["model_domains"] = domains
        deploy.validate_config(self.config)
        catalog, selection = deploy.load_task_selection(
            self.config, self.catalog_path, require_resolved=True
        )
        sizes = {
            value.removeprefix("sha256:"): 123
            for task in catalog["tasks"]
            for value in task["datasets"].values()
        }
        posted = []

        def api(base, method, path, body=None):
            if method == "GET":
                return []
            self.assertEqual(method, "POST")
            posted.append(body)
            driver = body["driver_config"]
            return {
                **body,
                "id": driver.get("task_version_id", driver.get("candidate_version_id")),
            }

        with (
            mock.patch.object(
                publication, "mirror_task_images", return_value=self.mirrors
            ) as mirror,
            mock.patch.object(publication, "api_json", side_effect=api),
            mock.patch.object(
                publication,
                "load_runtime_images",
                return_value={"storage_controller": {"reference": "fixture"}},
            ),
            mock.patch.object(publication, "executable", return_value="kubectl"),
            mock.patch.object(publication, "kubectl_apply") as apply,
            mock.patch.object(
                publication,
                "run_command",
                return_value=json.dumps({"dataset_sizes": sizes}),
            ),
        ):
            published = deploy.seed_tasks(
                self.config, self.state / "kubeconfig", self.catalog_path
            )
        receipt = deploy.load_json(self.state / "task-seed-receipt.json")
        self.assertEqual(
            receipt["schema_version"], "frontier.deployment-task-seed-receipt/v2"
        )
        self.assertEqual(
            receipt["task_version_ids"], [item["id"] for item in published]
        )
        self.assertEqual(
            set(receipt),
            {
                "schema_version",
                "site",
                "catalog_sha256",
                "effective_catalog_sha256",
                "selection_digest",
                "selected",
                "excluded",
                "model_domains",
                "candidate_version_id",
                "task_version_ids",
                "tasks",
            },
        )
        deploy.validate_seed_receipt(receipt, selection, "preview")
        self.assertEqual(mirror.call_args.kwargs["selection"], selection)
        apply.assert_not_called()
        self.assertEqual(
            receipt["candidate_version_id"],
            self.documents()["/api/v1/candidate-versions"][0]["id"],
        )
        for name, document in zip(selection.selected, posted[1:], strict=False):
            deploy.validate_seed_task(receipt, name, document)
            self.assertEqual(
                sorted(
                    set(deploy.TASK_DEFINITIONS[name]["network"]["allowed_domains"])
                    | set(domains)
                ),
                document["driver_config"]["network"]["allowed_domains"],
            )
        self.assertNotIn("sparse_video_attention_policy", receipt["tasks"])
        self.assertEqual(
            {
                json.loads(p.read_text())["items"][0]["metadata"]["labels"]["frontier.dev/task"]
                for p in (self.state / "task-seeds").iterdir()
            },
            {name.replace("_", "-")[:63] for name in selection.selected},
        )

    def test_render_selection_is_separate_from_bundle_checksums(self):
        with mock.patch.object(
            site_config, "generated_directory", return_value=self.root / "generated"
        ):
            config, output = deploy.render_bundle(
                self.config_path, catalog_path=self.catalog_path
            )
        self.assertEqual(
            deploy.load_json(output / "task-selection.json"), asdict(self.selection)
        )
        self.assertNotIn("task-selection", (output / "bundle.sha256").read_text())

    def test_mirroring_omits_excluded_task_images_and_reference_candidates(self):
        definitions = copy.deepcopy(deploy.TASK_DEFINITIONS)
        excluded = self.raw["tasks"][0]["name"]
        definitions[excluded]["gpu"] = {
            "verifier": {"max_gpu_instances": 1, "max_gpu_seconds": 60}
        }
        raw = copy.deepcopy(self.raw)
        task = raw["tasks"][0]
        task["verifier_gpu_image"] = {
            **task["verifier_image"],
            "repository": task["verifier_image"]["repository"] + "-gpu",
        }
        with mock.patch.object(
            publication, "load_task_definitions", return_value=definitions
        ):
            catalog = deploy.validate_task_catalog(raw, require_resolved=True)
            selection = deploy.select_tasks(self.config, raw, definitions)
        with (
            mock.patch.object(image_ops, "executable", return_value="docker"),
            mock.patch.object(image_ops, "cached_registry_image", return_value=None),
            mock.patch.object(
                image_ops, "registry_digest",
                side_effect=lambda port, repository, tag: (
                    catalog["candidate"]["overlay_image"]["digest"]
                    if repository == "frontier/candidates/codex" else "sha256:" + "f" * 64
                ),
            ),
            mock.patch.object(image_ops, "run_command") as commands,
        ):
            records = deploy.mirror_task_images(
                self.config, self.state / "kubeconfig", catalog, selection=selection
            )
        self.assertEqual(
            set(records), {"candidate", "reference_candidates", *selection.selected}
        )
        self.assertEqual(
            set(records["reference_candidates"]), set(selection.reference_names)
        )
        pulled = [
            call.args[0][-1]
            for call in commands.call_args_list
            if call.args[0][1] == "pull"
        ]
        self.assertEqual(
            len(pulled),
            1 + 3 * len(selection.selected) + len(selection.reference_names),
        )
        self.assertFalse(
            any(
                excluded in source or "sparse_video_attention_policy" in source
                for source in pulled
            )
        )

    def test_gpu_smoke_accepts_v1_and_v2_receipts_and_binds_domains(self):
        config = copy.deepcopy(self.config)
        config["features"]["modal_gpu"] = True
        config["execution"]["model_domains"] = ["gpu-model.example"]
        catalog, selection = deploy.load_task_selection(
            config, self.catalog_path, require_resolved=True
        )
        task = next(
            t for t in catalog["tasks"] if t["name"] == "sparse_video_attention_policy"
        )
        endpoint = config["network"]["registry_cluster_ip"] + ":5000"
        mirrors = {
            kind: {
                "digest": task[kind]["digest"],
                "reference": f"{endpoint}/frontier/tasks/sparse-video-attention-policy/{kind.removesuffix('_image')}@{task[kind]['digest']}",
            }
            for kind in ("agent_image", "verifier_image")
        }
        document = deploy.task_version_document(
            task, mirrors, model_domains=selection.model_domains
        )
        task_id = document["driver_config"]["task_version_id"]
        patch = self.root / "submission.patch"
        patch.write_bytes(b"diff --git a/a b/a\n")
        seed = {
            "schema_version": "frontier.deployment-task-seed-receipt/v1",
            "site": "preview",
            "catalog_sha256": selection.effective_catalog_sha256,
            "task_version_ids": [task_id],
            "candidate_version_id": "unused-by-gpu-replay",
        }

        def api(base, method, path, body=None, headers=None):
            if path == f"/api/v1/task-versions/{task_id}":
                return document
            if path == "/api/v1/evaluations" and method == "POST":
                self.assertEqual(body["task_version_ids"], [task_id])
                return {"id": "evaluation_test", "items": [{"run_id": "run_test"}]}
            if path == "/api/v1/runs/run_test":
                return {"state": "COMPLETED", "score": 1.0}
            if path == "/api/v1/evaluations/evaluation_test":
                return {"id": "evaluation_test", "status": "COMPLETED"}
            self.fail((method, path))

        for schema in ("v1", "v2"):
            with self.subTest(schema=schema):
                if schema == "v2":
                    tasks = {}
                    all_mirrors = mirrored_images(catalog)
                    all_mirrors[task["name"]] = mirrors
                    for selected in catalog["tasks"]:
                        doc = deploy.task_version_document(
                            selected,
                            all_mirrors[selected["name"]],
                            model_domains=selection.model_domains,
                        )
                        tasks[selected["name"]] = {
                            "task_version_id": doc["driver_config"]["task_version_id"],
                            "definition_digest": doc["definition_digest"],
                        }
                    seed.update(
                        schema_version="frontier.deployment-task-seed-receipt/v2",
                        tasks=tasks,
                        task_version_ids=[
                            tasks[name]["task_version_id"]
                            for name in selection.selected
                        ],
                        **{
                            key: getattr(selection, key)
                            for key in (
                                "catalog_sha256",
                                "effective_catalog_sha256",
                                "selection_digest",
                                "selected",
                                "excluded",
                                "model_domains",
                            )
                        },
                    )
                (self.state / "task-seed-receipt.json").write_bytes(
                    deploy.canonical_json(seed)
                )
                with (
                    mock.patch.object(
                        publication, "TASK_CATALOG_PATH", self.catalog_path
                    ),
                    mock.patch.object(acceptance, "api_json", side_effect=api),
                    mock.patch.object(
                        acceptance,
                        "ensure_version",
                        side_effect=lambda base, path, version_id, body: {
                            "id": version_id
                        },
                    ),
                ):
                    result = deploy.run_gpu_smoke(
                        config,
                        self.state / "kubeconfig",
                        patch,
                        deploy.sha256_bytes(patch.read_bytes()),
                        "test-" + schema,
                    )
                self.assertEqual(result["task_version_id"], task_id)
                self.assertEqual(result["status"], "COMPLETED")
        seed["selection_digest"] = "sha256:" + "0" * 64
        (self.state / "task-seed-receipt.json").write_bytes(deploy.canonical_json(seed))
        with (
            mock.patch.object(publication, "TASK_CATALOG_PATH", self.catalog_path),
            mock.patch.object(acceptance, "api_json") as api,
        ):
            with self.assertRaisesRegex(deploy.DeployError, "selection_digest"):
                deploy.run_gpu_smoke(
                    config,
                    self.state / "kubeconfig",
                    patch,
                    deploy.sha256_bytes(patch.read_bytes()),
                    "stale",
                )
        api.assert_not_called()

    def test_cpu_selection_never_reads_modal_environment_even_with_feature_enabled(
        self,
    ):
        # A release whose selected tasks do not need Modal must not request credentials.
        config = copy.deepcopy(self.config)
        config["features"]["modal_gpu"] = True
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(deploy.modal_credentials(config, self.selection), {})


if __name__ == "__main__":
    unittest.main()
