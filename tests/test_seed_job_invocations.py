from __future__ import annotations

import deployment_test_guard  # noqa: F401

import copy
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from preview import publication
from test_task_seeding import mirrored_images, resolved_catalog


class SeedJobInvocationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = json.loads((ROOT / "config/site.example.json").read_text())
        self.catalog = resolved_catalog()
        self.catalog["tasks"] = self.catalog["tasks"][:1]
        self.task = self.catalog["tasks"][0]
        self.mirrors = mirrored_images(self.catalog)
        self.definitions = publication.load_task_definitions()
        self.selection = publication.select_tasks(self.config, self.catalog, self.definitions)
        for name, value in (
            ("load_task_selection", (self.catalog, self.selection)),
            ("mirror_task_images", self.mirrors),
            ("load_runtime_images", {"storage_controller": {"reference": "runner"}}),
            ("state_directory", self.root),
            ("executable", "kubectl"),
        ):
            self.enterContext(mock.patch.object(publication, name, return_value=value))
        forward = self.enterContext(mock.patch.object(publication, "port_forward"))
        forward.return_value.__enter__.return_value = 58080
        self.ensure = self.enterContext(mock.patch.object(
            publication, "ensure_version", side_effect=lambda base, path, identity, document: {
                **document, "id": identity
            }
        ))
        self.apply = self.enterContext(mock.patch.object(publication, "kubectl_apply"))
        self.sizes = {value[7:]: 123 for value in self.task["datasets"].values()}

    def seed(self):
        return publication.seed_tasks(self.config, self.root / "kubeconfig")

    def test_concurrent_calls_keep_manifest_wait_and_logs_bound_to_their_job(self):
        barrier = threading.Barrier(2)
        local = threading.local()
        jobs = []

        def command(args, **kwargs):
            operation = args[3]
            if operation == "create":
                path = Path(args[5])
                barrier.wait(timeout=10)
                job = json.loads(path.read_bytes())["items"][0]
                local.job = job["metadata"]["name"]
                self.assertEqual(path.stem, local.job)
                self.assertLessEqual(len(local.job), 63)
                self.assertRegex(local.job, r"^[a-z0-9-]+$")
                content_name = publication.task_dataset_seed_job_name(
                    self.task, self.mirrors[self.task["name"]]["dataset_image"]["digest"]
                )
                self.assertIn(content_name.rsplit("-", 1)[1], local.job)
                script = job["spec"]["template"]["spec"]["containers"][0]["command"][2]
                self.assertIn("/data/.seed-scratch-" + local.job + "-", script)
                jobs.append((local.job, path))
                return ""
            self.assertIn(operation, {"wait", "logs"})
            self.assertIn("job/" + local.job, args)
            return json.dumps({"dataset_sizes": self.sizes}) if operation == "logs" else ""

        with mock.patch.object(publication, "run_command", side_effect=command), ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(self.seed) for _ in range(2)]
            results = [future.result(timeout=20) for future in futures]
        self.assertEqual(results[0], results[1])
        self.assertEqual(len({name for name, path in jobs}), 2)
        self.assertEqual(len({path for name, path in jobs}), 2)
        self.apply.assert_not_called()

    def test_name_collision_fails_create_without_touching_existing_job(self):
        existing = {}
        commands = []

        def command(args, **kwargs):
            commands.append(args)
            if args[3] == "create":
                body = json.loads(Path(args[5]).read_bytes())
                name = body["items"][0]["metadata"]["name"]
                if name in existing:
                    raise publication.DeployError("AlreadyExists")
                existing[name] = body
                return ""
            self.assertIn(args[3], {"wait", "logs"})
            return json.dumps({"dataset_sizes": self.sizes}) if args[3] == "logs" else ""

        with mock.patch.object(publication.secrets, "token_hex", return_value="a" * 16), mock.patch.object(publication, "run_command", side_effect=command):
            self.seed()
            snapshot = copy.deepcopy(existing)
            self.ensure.reset_mock()
            with self.assertRaisesRegex(publication.DeployError, "AlreadyExists"):
                self.seed()
        self.assertEqual(existing, snapshot)
        self.assertEqual([args[3] for args in commands], ["create", "wait", "logs", "create"])
        self.assertTrue(all(call.args[1] == "/api/v1/candidate-versions" for call in self.ensure.call_args_list))
        self.apply.assert_not_called()

    def test_fresh_jobs_recheck_pvc_hashes_and_sizes_with_same_pod_pid(self):
        task = copy.deepcopy(self.task)
        bodies = {"train": b"train data", "eval": b"eval data"}
        task["datasets"] = {
            mode + "_sha256": "sha256:" + hashlib.sha256(body).hexdigest()
            for mode, body in bodies.items()
        }
        archive_bytes = io.BytesIO()
        extension = self.definitions[task["name"]]["dataset_extension"]
        with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
            for mode, body in bodies.items():
                digest = task["datasets"][mode + "_sha256"][7:]
                info = tarfile.TarInfo(f"datasets/{mode}/sha256/{digest}.{extension}")
                info.size = len(body)
                archive.addfile(info, io.BytesIO(body))
        scripts = []
        for name in ("seed-one", "seed-two", "seed-three"):
            manifest = publication.render_task_dataset_seed_manifest(
                self.config, task, self.mirrors[task["name"]], "runner", job_name=name
            )
            script = json.loads(manifest)["items"][0]["spec"]["template"]["spec"]["containers"][0]["command"][2]
            scripts.append(script.replace("/data/", str(self.root) + "/"))
        barrier = threading.Barrier(2)

        def urlopen(request, **kwargs):
            if not isinstance(request, str):
                return io.BytesIO(json.dumps({"layers": [{"digest": "sha256:fixture"}]}).encode())
            if barrier is not None:
                barrier.wait(timeout=10)
            return io.BytesIO(archive_bytes.getvalue())

        def execute(script):
            output = []
            namespace = {"print": output.append}
            try:
                exec(script, namespace)
            finally:
                # A real seed process closes its archive when it exits on error.
                if "archive" in namespace:
                    namespace["archive"].close()
            return json.loads(output[0])

        with mock.patch("urllib.request.urlopen", side_effect=urlopen), mock.patch("os.getpid", return_value=1):
            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(execute, script) for script in scripts[:2]]
                results = [future.result(timeout=20) for future in futures]
            expected = {task["datasets"][mode + "_sha256"][7:]: len(body) for mode, body in bodies.items()}
            self.assertEqual(results, [{"dataset_sizes": expected}] * 2)
            self.assertFalse(list(self.root.glob(".seed-scratch-*")))
            barrier = None
            self.assertEqual(execute(scripts[2]), {"dataset_sizes": expected})
            target = self.root / publication.task_dataset_relative_path(task["name"], "train", task["datasets"]["train_sha256"])
            target.chmod(0o644)
            target.write_bytes(b"X" * len(bodies["train"]))
            with self.assertRaisesRegex(AssertionError, "existing file differs"):
                execute(scripts[2])


if __name__ == "__main__":
    unittest.main()
