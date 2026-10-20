from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.dont_write_bytecode = True
import render_task_definition as renderer


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definitions = json.loads(renderer.DEFINITIONS.read_text())
        cls.fixture = json.loads(
            (ROOT / "tests/fixtures/native-identity.json").read_text()
        )

    def test_catalog_order_and_malformed_or_missing_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            with self.assertRaisesRegex(ValueError, "cannot read task catalog"):
                renderer.catalog_task_names(path)
            path.write_text('{"tasks": [{"name": "z_task"}, {"name": "a_task"}]}')
            self.assertEqual(renderer.catalog_task_names(path), ["z_task", "a_task"])
            for body in (
                "{",
                "[]",
                "{}",
                '{"tasks": {}}',
                '{"tasks": [null]}',
                '{"tasks": [{"name": "../escape"}]}',
                '{"tasks": [], "tasks": []}',
                '{"tasks": [{"name": "a", "name": "b"}]}',
                '{"tasks": [{"name": "a"}, {"name": "a"}]}',
            ):
                path.write_text(body)
                with self.subTest(body=body), self.assertRaises(ValueError):
                    renderer.catalog_task_names(path)

    def test_empty_catalog_exits_two_before_rendering(self):
        read_catalog = renderer.catalog_task_names
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text('{"tasks": []}')
            with self.assertRaisesRegex(ValueError, "task catalog lists no tasks"):
                read_catalog(path)
            with (
                mock.patch.object(
                    renderer,
                    "catalog_task_names",
                    side_effect=lambda: read_catalog(path),
                ),
                mock.patch.object(renderer, "render") as render,
                mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                with self.assertRaises(SystemExit) as error:
                    renderer.main(["--all"])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("task catalog lists no tasks", stderr.getvalue())
            render.assert_not_called()

    def test_single_task_render_failure_retains_traceback_and_exit_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "policy.json").write_text("{}")
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "scripts/render_task_definition.py"),
                    str(root),
                    "--name",
                    "invalid_task",
                ],
                env={
                    key: value
                    for key, value in os.environ.items()
                    if key != "TAR_OPTIONS"
                },
                capture_output=True,
                text=True,
            )
        # Reproduced with HEAD~1 before the Phase 0 exception wrapper.
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("Traceback (most recent call last):", result.stderr)
        self.assertTrue(
            result.stderr.endswith(
                "ValueError: unsupported native policy schema_version\n"
            )
        )

    def test_all_render_and_write_failures_propagate(self):
        with mock.patch.object(
            renderer,
            "catalog_problem_directories",
            return_value={"a_task": Path("a_task")},
        ):
            for error in (ValueError("invalid policy"), OSError("cannot read policy")):
                with (
                    self.subTest(error=error),
                    mock.patch.object(renderer, "render", side_effect=error),
                ):
                    with self.assertRaises(type(error)) as raised:
                        renderer.main(["--all"])
                    self.assertIs(raised.exception, error)
            with (
                mock.patch.object(renderer, "render", return_value={}),
                mock.patch.object(
                    Path, "write_text", side_effect=OSError("cannot write definitions")
                ),
            ):
                with self.assertRaisesRegex(OSError, "cannot write definitions"):
                    renderer.main(["--all"])

    def test_all_ignores_extra_directory_with_one_warning(self):
        paths = list((renderer.TASK_ROOT / "problems").iterdir())
        with tempfile.TemporaryDirectory() as directory:
            extra = Path(directory) / "extra_task"
            extra.mkdir()
            output = Path(directory) / "definitions.json"
            with (
                mock.patch.object(Path, "iterdir", return_value=iter([*paths, extra])),
                mock.patch.object(renderer, "DEFINITIONS", output),
                mock.patch.object(
                    renderer,
                    "render",
                    side_effect=lambda root, name: self.definitions[name],
                ) as render,
                mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                self.assertEqual(renderer.main(["--all"]), 0)
            names = [
                task["name"]
                for task in json.loads(renderer.CATALOG.read_text())["tasks"]
            ]
            self.assertEqual(
                [call.args[1] for call in render.call_args_list], sorted(names)
            )
            self.assertEqual(json.loads(output.read_bytes()), self.definitions)
            self.assertEqual(
                stderr.getvalue().splitlines(),
                [f"{extra}: ignored: not in config/task-catalog.json"],
            )

    def test_all_missing_catalog_directory_exits_two_before_rendering(self):
        with (
            mock.patch.object(
                renderer, "catalog_task_names", return_value=["missing_task"]
            ),
            mock.patch.object(renderer, "render") as render,
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            with self.assertRaises(SystemExit) as error:
                renderer.main(["--all"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn(
            "catalog task missing_task has no problem directory", stderr.getvalue()
        )
        render.assert_not_called()

    def test_catalog_cpu_fixture_identities_and_authored_fields(self):
        self.assertEqual(
            set(self.fixture),
            {
                "batch_local_counting",
                "bounded_evm_path_prediction",
                "evm_state_checkpoints",
                "graph_mining_method_selection",
                "join_tree_rewriting",
            },
        )
        for name, expected in self.fixture.items():
            with self.subTest(task=name):
                root = renderer.problem_directory(renderer.TASK_ROOT / "problems", name)
                row = renderer.render(root, name)
                self.assertEqual(row, self.definitions[name])
                self.assertEqual(
                    row["source"]["source_set_sha256"], expected["source_set_sha256"]
                )
                self.assertEqual(
                    row["source"]["identity_algorithm"], expected["identity_algorithm"]
                )
                self.assertEqual(row["image_tag"], expected["image_tag"])
                policy = json.loads((root / "policy.json").read_text())
                for key in (
                    "family",
                    "display_name",
                    "version",
                    "quota",
                    "agent",
                    "metadata",
                    "network",
                ):
                    self.assertEqual(row[key], policy[key])
                for mode in ("train", "eval"):
                    self.assertEqual(
                        row["datasets"][f"{mode}_sha256"],
                        "sha256:" + expected["datasets"][mode],
                    )

    def test_sparse_has_native_identity_and_preserves_scoring_inputs(self):
        row = renderer.render(
            renderer.problem_directory(
                renderer.TASK_ROOT / "problems", renderer.SPARSE
            ),
            renderer.SPARSE,
        )
        historical = json.loads(
            (ROOT / "tests/fixtures/sparse-definition-e64d130.json").read_text()
        )
        self.assertEqual(row, self.definitions[renderer.SPARSE])
        self.assertNotEqual(row["source"], historical["source"])
        self.assertEqual(row["source"]["identity_algorithm"], renderer.IDENTITY)
        self.assertEqual(row["quota"]["wall_time_seconds"], 18000)
        authored = json.loads(
            (ROOT / "tests/fixtures/svg-authored-policy-pr25.json").read_text()
        )
        self.assertEqual(authored["commit"], "89dce3bc39109eea534d2beccb57a5fb8ec2b1bf")
        expected = copy.deepcopy(authored["policy"])
        expected["verifier"]["evidence_path"] = "/frontier/output/evidence.tar.zst"
        # Preserve authored science and the operator-approved transport capacity.
        expected["network"]["audit"]["max_model_sessions"] = 1024
        expected["network"]["audit"]["max_bundle_bytes"] = 32 * 1024**3
        # HunyuanVideo extension: second editable
        # surface, 12,600 s final timeout with the GPU budget scaled by the same
        # (25 + 1) x 4 x timeout formula, and the revised beat-spec text.
        expected["quota"]["eval_timeout_seconds"] = 12600
        expected["gpu"]["verifier"]["max_gpu_seconds"] = 26 * 4 * 12600
        expected["metadata"]["runtime"]["allowed_paths"].insert(
            1, "svg/models/hyvideo/attention.py"
        )
        expected["metadata"]["evaluation"]["beat_spec_sha256"] = (
            "sha256:19bb12002bd89b264f142f9e38a570133d98dee964051c032c2e5050135dff5d"
        )
        # Internal repository URL replaced in the public release tree.
        expected["metadata"]["conversion"]["source_repo"] = "private authoring repository"
        for field in (
            "quota",
            "agent",
            "gpu",
            "verifier",
            "submission",
            "network",
            "metadata",
        ):
            self.assertEqual(row[field], expected[field], field)
        self.assertEqual(row["datasets"], historical["datasets"])
        self.assertEqual(
            row["metadata"]["accelerator"]["agent_gpu_interface"],
            "frontier.gpu-helper/v3",
        )

    def test_all_twice_is_byte_identical_to_committed_definitions(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                renderer, "DEFINITIONS", Path(directory) / "definitions.json"
            ),
        ):
            renderer.main(["--all"])
            first = renderer.DEFINITIONS.read_bytes()
            renderer.main(["--all"])
            self.assertEqual(first, renderer.DEFINITIONS.read_bytes())
            self.assertEqual(json.loads(first), self.definitions)

    def test_identity_repeatability_includes_hidden_and_lock_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "file").write_text("hello")
            initial = renderer.source_sha256(root)
            self.assertEqual(initial, renderer.source_sha256(root))
            os.utime(root / "file", (12345, 12345))
            self.assertEqual(initial, renderer.source_sha256(root))
            for name in (".hidden", "upstream.lock.json"):
                (root / name).write_text("included")
                changed = renderer.source_sha256(root)
                self.assertNotEqual(initial, changed)
                initial = changed

    def test_rejects_tar_options_even_empty_symlinks_special_files_and_tar_failure(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in ("", "--exclude=*"):
                with (
                    mock.patch.dict(os.environ, {"TAR_OPTIONS": value}),
                    self.assertRaisesRegex(ValueError, "TAR_OPTIONS"),
                ):
                    renderer.source_sha256(root)
            (root / "link").symlink_to("missing")
            with self.assertRaisesRegex(ValueError, "regular files"):
                renderer.source_sha256(root)
            (root / "link").unlink()
            os.mkfifo(root / "fifo")
            with self.assertRaisesRegex(ValueError, "regular files"):
                renderer.source_sha256(root)
            (root / "fifo").unlink()
            with mock.patch.object(renderer.subprocess, "Popen") as popen:
                proc = popen.return_value.__enter__.return_value
                proc.stdout = __import__("io").BytesIO(b"partial archive")
                proc.wait.return_value = 2
                with self.assertRaisesRegex(ValueError, "tar failed"):
                    renderer.source_sha256(root)

    def test_policy_and_dataset_errors(self):
        source = renderer.problem_directory(
            renderer.TASK_ROOT / "problems", "batch_local_counting"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(source / "datasets", root / "datasets")
            original = json.loads((source / "policy.json").read_text())
            variants = []
            for key in ("quota", "agent", "verifier", "submission"):
                changed = copy.deepcopy(original)
                del changed[key]
                variants.append(changed)
            changed = copy.deepcopy(original)
            del changed["metadata"]["datasets"]
            variants.append(changed)
            variants.append({**original, "schema_version": "wrong"})
            changed = copy.deepcopy(original)
            changed["metadata"]["datasets"]["train"]["version"] = "sha256:" + "a" * 64
            variants.append(changed)
            for policy in variants:
                (root / "policy.json").write_text(json.dumps(policy))
                with self.assertRaises(ValueError):
                    renderer.render(root, "batch_local_counting")


if __name__ == "__main__":
    unittest.main()
