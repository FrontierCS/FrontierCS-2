from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip
import copy
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import frontier_deploy as deploy
import import_task as importer
import render_task_definition as renderer
from preview.publication import baseline_from_result_json, load_baselines

REAL_RESOLVE_DIGEST = importer.resolver.resolve_digest


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


class ImportCliTests(unittest.TestCase):
    def test_missing_jsonschema_is_one_line_with_exit_two(self):
        code = """
import runpy, sys
sys.modules['jsonschema'] = None
sys.argv = ['scripts/import_task.py', '/unused/export', '--plan']
runpy.run_path('scripts/import_task.py', run_name='__main__')
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "install jsonschema (pip install jsonschema) to import tasks\n",
        )

    def test_make_requires_export_before_invoking_python(self):
        for value in ("", "   "):
            with self.subTest(export=value):
                result = subprocess.run(
                    [
                        "make",
                        "--no-print-directory",
                        "import-task",
                        "EXPORT=" + value,
                        "PYTHON=/must-not-run-python",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertEqual(
                    result.stderr.splitlines()[0], "EXPORT=<dir> is required"
                )
                self.assertNotIn("/must-not-run-python", result.stderr)


class ImportTests(unittest.TestCase):
    def setUp(self):
        # macOS exposes its system temporary directory through /var -> /private/var.
        # Evidence fixtures need real parents while explicit symlink tests stay intact.
        self.temp = tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.source = Path(self.temp.name) / "minimal_task"
        self.root.mkdir()
        shutil.copytree(ROOT / "tests/fixtures/export-minimal", self.source)
        (self.root / "config").mkdir()
        shutil.copytree(ROOT / "docs/contract", self.root / "docs/contract")
        launcher = self.root / importer.LAUNCHER
        launcher.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / importer.LAUNCHER, launcher)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(importer, "REPOSITORY_ROOT", self.root).start()
        self.build = mock.patch.object(
            importer.builder, "build_problem", return_value={"images": {}}
        ).start()
        self.docker = mock.patch.object(importer, "docker", return_value="[]").start()
        self.checks = mock.patch.object(importer, "run_checks").start()
        self.registry = mock.patch.object(
            importer.resolver,
            "resolve_digest",
            side_effect=AssertionError("unexpected network"),
        ).start()
        self.stdout = mock.patch("sys.stdout", new_callable=io.StringIO).start()
        existing = self.root / importer.PROBLEMS / "existing_task"
        shutil.copytree(self.source, existing)
        for path, key in [
            ("policy.json", "problem_id"),
            ("upstream.lock.json", "problem_id"),
            ("judge/result.json", "task_id"),
        ]:
            body = deploy.load_json(existing / path)
            body[key] = "existing_task"
            (existing / path).write_text(json.dumps(body))
        definition = importer.validate_export(existing)
        entry = importer.catalog_entry(
            existing, definition, "joyemang33", existing / "judge/reference.patch"
        )
        for image, _ in importer.resolver.task_images(entry):
            image["digest"] = "sha256:" + "b" * 64
        catalog = {
            "schema_version": "frontier.deployment-task-catalog/v2",
            "candidate": deploy.load_json(ROOT / "config/task-catalog.json")[
                "candidate"
            ],
            "tasks": [entry],
        }
        (self.root / "config/task-catalog.json").write_text(renderer.json_text(catalog))
        # Noncanonical row whitespace must survive adding and replacing another row.
        (self.root / "config/task-definitions.json").write_text(
            json.dumps({"existing_task": definition}, indent=4) + "\n"
        )

    def invoke(self, *args):
        return importer.main([str(self.source), *args])

    def entry(self):
        return deploy.load_json(self.root / "config/task-catalog.json")["tasks"][-1]

    def manifest(self):
        definition = importer.validate_export(self.source)
        entry = importer.catalog_entry(
            self.source, definition, "joyemang33", self.source / "judge/reference.patch"
        )
        images = {
            role: {k: entry[key][k] for k in ("repository", "tag", "digest")}
            for role, key in importer.ROLE_KEYS.items()
            if entry.get(key) is not None
        }
        for image in images.values():
            image["digest"] = "sha256:" + "c" * 64
        body = {
            "schema_version": "frontier.task-release-images/v1",
            "problem_id": "minimal_task",
            "source_sha256": definition["source"]["source_set_sha256"],
            "images": images,
        }
        path = Path(self.temp.name) / "published.json"
        path.write_text(json.dumps(body))
        return path, body

    def test_plan_writes_nothing_and_lists_every_action(self):
        before = snapshot(Path(self.temp.name))
        self.assertEqual(self.invoke("--plan"), 0)
        self.assertEqual(snapshot(Path(self.temp.name)), before)
        for text in (
            "copy verbatim",
            "definition row",
            "catalog v2 entry",
            "baseline source",
            "reference:",
            "make test",
        ):
            self.assertIn(text, self.stdout.getvalue())
        self.build.assert_not_called()
        self.docker.assert_not_called()
        self.registry.assert_not_called()

    def test_import_repeat_replace_and_unrelated_definition_bytes(self):
        definitions = self.root / "config/task-definitions.json"
        old = definitions.read_text()
        spans = importer.row_spans(old)
        original_row = old[slice(*spans["existing_task"])]
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(
            snapshot(self.source),
            snapshot(self.root / importer.PROBLEMS / "minimal_task"),
        )
        self.assertEqual(self.entry()["agent_image"]["digest"], importer.PLACEHOLDER)
        self.assertEqual(self.entry()["baseline"]["historical_mean"], 2.0)
        before = snapshot(self.root)
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(snapshot(self.root), before)
        self.assertIn("already imported, no changes", self.stdout.getvalue())
        self.assertEqual(self.build.call_count, 1)
        (self.source / "judge/evaluator.py").write_text("changed fixture\n")
        with self.assertRaisesRegex(ValueError, "changed: judge/evaluator.py"):
            self.invoke()
        self.assertEqual(snapshot(self.root), before)
        self.invoke("--replace")
        self.assertEqual(self.build.call_count, 2)
        changed = definitions.read_text()
        self.assertEqual(
            changed[slice(*importer.row_spans(changed)["existing_task"])], original_row
        )
        self.assertNotEqual(snapshot(self.root), before)

    def source_first_update(self, name="minimal_task"):
        target = self.root / importer.PROBLEMS / name
        (target / "judge/evaluator.py").write_text("source-first replacement\n")
        if name == "minimal_task":
            (self.source / "judge/evaluator.py").write_bytes(
                (target / "judge/evaluator.py").read_bytes()
            )
        path = self.root / "config/task-definitions.json"
        definitions = deploy.load_json(path)
        definitions[name] = importer.validate_export(target)
        path.write_text(renderer.json_text(definitions))

    def test_replace_reconciles_only_source_first_target_catalog_row(self):
        self.invoke()
        old = self.entry()["agent_image"]["tag"]
        self.source_first_update()
        before = snapshot(self.root)
        with self.assertRaisesRegex(deploy.DeployError, "reviewed source tag"):
            self.invoke("--plan")
        self.assertEqual(self.invoke("--replace", "--plan"), 0)
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(self.invoke("--replace"), 0)
        self.assertNotEqual(self.entry()["agent_image"]["tag"], old)
        deploy.validate_task_catalog(
            deploy.load_json(self.root / "config/task-catalog.json"),
            definitions=deploy.load_json(self.root / "config/task-definitions.json"),
            root=self.root,
        )

    def test_replace_still_rejects_stale_unrelated_catalog_row(self):
        self.invoke()
        self.source_first_update("existing_task")
        before = snapshot(self.root)
        with self.assertRaisesRegex(deploy.DeployError, "reviewed source tag"):
            self.invoke("--replace", "--plan")
        self.assertEqual(snapshot(self.root), before)

    def test_replace_rejects_malformed_previous_target_image(self):
        self.invoke()
        self.source_first_update()
        path = self.root / "config/task-catalog.json"
        catalog = deploy.load_json(path)
        catalog["tasks"][-1]["agent_image"]["digest"] = "invalid"
        path.write_text(renderer.json_text(catalog))
        before = snapshot(self.root)
        with self.assertRaisesRegex(deploy.DeployError, "digest"):
            self.invoke("--replace", "--plan")
        self.assertEqual(snapshot(self.root), before)

    def test_replace_rejects_duplicate_previous_target_rows(self):
        self.invoke()
        path = self.root / "config/task-catalog.json"
        catalog = deploy.load_json(path)
        catalog["tasks"].append(copy.deepcopy(catalog["tasks"][-1]))
        path.write_text(renderer.json_text(catalog))
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "duplicate problem"):
            self.invoke("--replace", "--plan")
        self.assertEqual(snapshot(self.root), before)

    def test_failed_staged_validation_preserves_destination_and_config_bytes(self):
        self.invoke()
        (self.source / "judge/evaluator.py").write_text("replacement\n")
        before = snapshot(self.root)
        real_validate = importer.validate_staged

        def fail_after_validation(staged, name, identity):
            self.assertTrue(
                staged.is_relative_to(self.root / ".generated/import/minimal_task")
            )
            self.assertEqual(
                snapshot(self.root / importer.PROBLEMS / name),
                {
                    p.removeprefix(str(importer.PROBLEMS / name) + "/"): body
                    for p, body in before.items()
                    if p.startswith(str(importer.PROBLEMS / name) + "/")
                },
            )
            self.assertTrue((staged / "config/task-catalog.json").is_file())
            self.assertTrue((staged / "config/task-definitions.json").is_file())
            real_validate(staged, name, identity)
            raise ValueError("staged validation failure")

        self.build.reset_mock()
        with mock.patch.object(
            importer, "validate_staged", side_effect=fail_after_validation
        ):
            with self.assertRaisesRegex(ValueError, "staged validation failure"):
                self.invoke("--replace")
        self.assertEqual(snapshot(self.root), before)
        self.build.assert_not_called()

    def test_failures_after_swap_restore_tree_config_and_receipt(self):
        self.invoke()
        (self.source / "judge/evaluator.py").write_text("first replacement\n")
        self.invoke("--replace")  # Retain an older previous/ tree too.
        (self.source / "judge/evaluator.py").write_text("second replacement\n")
        before = snapshot(self.root)
        atomic_write = deploy.atomic_write
        for failure in (
            "task-definitions.json",
            "task-catalog.json",
            "minimal_task.json",
            "checks",
        ):
            failed = False

            def write(path, body, *, failure=failure, **kwargs):
                nonlocal failed
                atomic_write(path, body, **kwargs)
                if (
                    path.name == failure
                    and path.parent
                    == (
                        self.root / "config"
                        if failure != "minimal_task.json"
                        else self.root / ".generated/import"
                    )
                    and not failed
                ):
                    failed = True
                    raise OSError("write failure after rename")

            self.checks.side_effect = (
                ValueError("post-import failure") if failure == "checks" else None
            )
            with (
                self.subTest(failure=failure),
                mock.patch.object(importer, "atomic_write", side_effect=write),
            ):
                with self.assertRaises((OSError, ValueError)):
                    self.invoke("--replace")
            self.assertEqual(snapshot(self.root), before)

    def test_failed_new_import_check_removes_destination_and_receipt(self):
        before = snapshot(self.root)
        self.checks.side_effect = ValueError("post-import failure")
        with self.assertRaisesRegex(ValueError, "post-import failure"):
            self.invoke()
        self.assertFalse((self.root / importer.PROBLEMS / "minimal_task").exists())
        self.assertEqual(snapshot(self.root), before)

    def test_persistent_catalog_write_failure_restores_tree_and_both_configs(self):
        self.invoke()
        (self.source / "judge/evaluator.py").write_text("first replacement\n")
        self.invoke("--replace")
        (self.source / "judge/evaluator.py").write_text("second replacement\n")
        before = snapshot(self.root)
        atomic_write = deploy.atomic_write
        catalog = self.root / "config/task-catalog.json"
        for after_rename in (False, True):

            def write(path, body, *, after_rename=after_rename, **kwargs):
                if path == catalog and not after_rename:
                    raise OSError("persistent catalog failure")
                atomic_write(path, body, **kwargs)
                if path == catalog:
                    raise OSError("persistent catalog failure")

            with (
                self.subTest(after_rename=after_rename),
                mock.patch.object(
                    importer, "atomic_write", side_effect=write
                ) as writes,
            ):
                with self.assertRaisesRegex(
                    deploy.DeployError, "import failed:.*rollback errors:"
                ) as raised:
                    self.invoke("--replace")
            self.assertIsInstance(raised.exception.__cause__, OSError)
            self.assertIn("restore config/task-catalog.json", str(raised.exception))
            self.assertEqual(
                [call.args[0].name for call in writes.call_args_list],
                [
                    "task-definitions.json",
                    "task-catalog.json",
                    "task-definitions.json",
                    "task-catalog.json",
                    "minimal_task.json",
                ],
            )
            self.assertEqual(snapshot(self.root), before)

    def test_rollback_collects_tree_and_each_file_error_without_stopping(self):
        self.invoke()
        (self.source / "judge/evaluator.py").write_text("replacement\n")
        before = snapshot(self.root)
        failure = ValueError("post-import failure")
        self.checks.side_effect = failure
        atomic_write, rmtree = deploy.atomic_write, shutil.rmtree
        rolling_back = False

        def remove(path, *args, **kwargs):
            nonlocal rolling_back
            rmtree(path, *args, **kwargs)
            if path == self.root / importer.PROBLEMS / "minimal_task":
                rolling_back = True
                raise OSError("tree removal reported failure")

        def write(path, body, **kwargs):
            atomic_write(path, body, **kwargs)
            if rolling_back:
                raise OSError("restore reported failure: " + path.name)

        with (
            mock.patch.object(importer.shutil, "rmtree", side_effect=remove),
            mock.patch.object(importer, "atomic_write", side_effect=write),
        ):
            with self.assertRaises(deploy.DeployError) as raised:
                self.invoke("--replace")
        self.assertIs(raised.exception.__cause__, failure)
        message = str(raised.exception)
        steps = [
            "post-import failure",
            "remove installed tree",
            "restore config/task-definitions.json",
            "restore config/task-catalog.json",
            "restore .generated/import/minimal_task.json",
        ]
        positions = [message.index(step) for step in steps]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(snapshot(self.root), before)

    def test_invalid_export_fails_before_copy_build_registry(self):
        mutations = [
            lambda: (self.source / "datasets/eval.json").unlink(),
            lambda: (self.source / "link").symlink_to("/etc/passwd"),
            lambda: (self.source / "pointer").write_text(
                "version https://git-lfs.github.com/spec/v1\noid sha256:abc\n"
            ),
            lambda: self.change_policy(problem_id="wrong"),
            lambda: self.change_policy(password="fake-fixture-value"),
            lambda: self.change_policy(
                submission={"entrypoint": "payload/../../escape"}
            ),
            lambda: (self.source / "datasets/train.json").write_text("{}"),
            lambda: (self.source / "agent-base/Dockerfile").write_text(
                "FROM private.example/image:tag\n"
            ),
            lambda: (self.source / "agent-base/Dockerfile").write_text(
                "FROM scratch\n# comment \\\nFROM private.example/image:latest\n"
            ),
        ]
        original = snapshot(self.source)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                shutil.rmtree(self.source)
                self.source.mkdir()
                for name, body in original.items():
                    path = self.source / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(body)
                mutation()
                before = snapshot(self.root)
                with self.assertRaises(
                    (ValueError, deploy.DeployError, importer.builder.BuildFailure)
                ):
                    self.invoke("--plan")
                self.assertEqual(snapshot(self.root), before)
        self.build.assert_not_called()
        self.registry.assert_not_called()

    def change_policy(self, **changes):
        policy = deploy.load_json(self.source / "policy.json")
        policy.update(changes)
        (self.source / "policy.json").write_text(json.dumps(policy))

    def test_empty_directories_fail_before_publication(self):
        for relative in ("tests", "maintenance/nested"):
            with self.subTest(relative=relative):
                path = self.source / relative
                path.mkdir(parents=True)
                with self.assertRaisesRegex(
                    ValueError, "empty directory cannot survive Git checkout"
                ):
                    self.invoke("--plan")
                path.rmdir()
                if "/" in relative:
                    path.parent.rmdir()
        self.build.assert_not_called()
        self.registry.assert_not_called()

    def test_import_and_builder_share_from_parsing_and_base_policy(self):
        digest = "@sha256:" + "a" * 64
        valid = [
            "python:3.11-slim-bookworm" + digest,
            "ubuntu" + digest,
            "docker.io/library/ubuntu" + digest,
            "pytorch/pytorch" + digest,
            "python:3.11-slim-bookworm",
            "scratch",
            "${FRONTIER_GPU_SSH_BASE_IMAGE}",
        ]
        invalid = [
            "ubuntu:latest",
            "python:3.13",
            "ghcr.io/example/image" + digest,
            "private.example/image" + digest,
            "localhost:5000/image" + digest,
            "docker.io/example/image:latest",
            "${UNKNOWN_BASE}",
        ]
        dockerfile = self.source / "agent-base/Dockerfile"
        for source in valid + invalid:
            for prefix in ("FROM ", "  from ", "\tFrOm --platform=linux/amd64 "):
                dockerfile.write_text(
                    prefix + source + " aS base\nFROM base AS final\n"
                )
                with (
                    self.subTest(source=source, prefix=prefix),
                    mock.patch.object(
                        importer.builder,
                        "pull_digest_reference",
                        return_value="ubuntu" + digest,
                    ) as pull,
                ):
                    if source in valid:
                        importer.validate_export(self.source)
                        importer.builder.freeze_bases(
                            dockerfile, {}, "docker.io/example/ssh" + digest
                        )
                    else:
                        with self.assertRaises(importer.builder.BuildFailure):
                            importer.validate_export(self.source)
                        with self.assertRaises(importer.builder.BuildFailure):
                            importer.builder.freeze_bases(dockerfile, {}, None)
                        pull.assert_not_called()

    def test_size_limit_and_lfs_tracking(self):
        with (
            mock.patch.object(importer, "MAX_FILE_BYTES", 1),
            mock.patch.object(importer, "lfs_tracked", return_value=False),
        ):
            with self.assertRaisesRegex(ValueError, "512 MiB"):
                self.invoke("--plan")
        with (
            mock.patch.object(importer, "MAX_FILE_BYTES", 1),
            mock.patch.object(importer, "lfs_tracked", return_value=True),
        ):
            self.invoke("--plan")

    def test_published_dataset_revision_is_bound_to_this_source(self):
        path, manifest = self.manifest()
        definition = importer.validate_export(self.source)
        entry = importer.catalog_entry(
            self.source, definition, "joyemang33", self.source / "judge/reference.patch"
        )
        expected = entry["agent_image"]["tag"]
        for tag in ("v1", expected):
            manifest["images"]["datasets"]["tag"] = tag
            path.write_text(json.dumps(manifest))
            adopted = importer.published_manifest(
                path, self.source.name, manifest["source_sha256"], entry
            )
            self.assertEqual(adopted["dataset_image"]["tag"], tag)
        for tag in ("latest", "v2", "source-" + "0" * 64):
            manifest["images"]["datasets"]["tag"] = tag
            path.write_text(json.dumps(manifest))
            with self.subTest(tag=tag), self.assertRaises(deploy.DeployError):
                importer.published_manifest(
                    path, self.source.name, manifest["source_sha256"], entry
                )
        self.registry.assert_not_called()

    def test_source_artifact_revision_keeps_source_digest_and_role_interlocks(self):
        path, manifest = self.manifest()
        definition = importer.validate_export(self.source)
        entry = importer.catalog_entry(
            self.source, definition, "joyemang33", self.source / "judge/reference.patch"
        )
        source_tag = definition["image_tag"]
        for _role, image in manifest["images"].items():
            image["tag"] = source_tag + "-image-" + image["digest"][7:39]
        path.write_text(json.dumps(manifest))
        adopted = importer.published_manifest(
            path, self.source.name, manifest["source_sha256"], entry
        )
        self.assertEqual(
            adopted["agent_image"]["tag"], manifest["images"]["agent"]["tag"]
        )
        catalog = {
            "schema_version": "frontier.deployment-task-catalog/v2",
            "candidate": deploy.load_json(ROOT / "config/task-catalog.json")[
                "candidate"
            ],
            "tasks": [adopted],
        }
        shutil.copytree(self.source, self.root / importer.PROBLEMS / self.source.name)
        deploy.validate_task_catalog(
            catalog, definitions={self.source.name: definition}, root=self.root
        )
        for tag in (
            source_tag + "-runtime-r1",
            source_tag + "-image-" + "d" * 32,
            "source-" + "d" * 64 + "-image-" + "c" * 32,
            source_tag + "-image-" + "c" * 31,
            source_tag + "-image-" + "C" * 32,
            source_tag + "-image-" + "c" * 33,
        ):
            with self.subTest(tag=tag):
                bad = copy.deepcopy(manifest)
                bad["images"]["agent"]["tag"] = tag
                path.write_text(json.dumps(bad))
                with self.assertRaises(deploy.DeployError):
                    importer.published_manifest(
                        path, self.source.name, manifest["source_sha256"], entry
                    )
        bad = copy.deepcopy(manifest)
        bad["images"]["agent"]["repository"] = bad["images"]["verifier"]["repository"]
        path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "repository does not match role"):
            importer.published_manifest(
                path, self.source.name, manifest["source_sha256"], entry
            )
        self.registry.assert_not_called()

    def test_published_import_identity_mismatch_and_check_readonly(self):
        path, manifest = self.manifest()
        manifest["source_sha256"] = "sha256:" + "0" * 64
        path.write_text(json.dumps(manifest))
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            self.invoke("--published-images", str(path))
        self.assertEqual(snapshot(self.root), before)
        path, _ = self.manifest()
        self.registry.side_effect = None
        self.registry.return_value = "sha256:" + "c" * 64
        self.invoke("--published-images", str(path))
        self.assertEqual(self.entry()["agent_image"]["digest"], "sha256:" + "c" * 64)
        self.build.assert_not_called()
        before = snapshot(self.root)
        self.invoke("--check")
        self.assertEqual(snapshot(self.root), before)
        self.assertTrue(
            any(c.kwargs.get("single_layer") for c in self.registry.call_args_list)
        )
        self.registry.return_value = "sha256:" + "a" * 64
        with self.assertRaisesRegex(ValueError, "same linux/amd64"):
            self.invoke("--check")

    def test_check_refuses_placeholders(self):
        self.invoke()
        with self.assertRaisesRegex(ValueError, "placeholder"):
            self.invoke("--check")
        self.registry.assert_not_called()

    def test_check_compares_all_derived_baseline_fields_but_allows_annotations(self):
        path, _ = self.manifest()
        self.registry.side_effect = None
        self.registry.return_value = "sha256:" + "c" * 64
        self.invoke("--published-images", str(path))
        catalog_path = self.root / "config/task-catalog.json"
        original = deploy.load_json(catalog_path)
        baseline = original["tasks"][-1]["baseline"]
        annotations = {"provenance", "comparable", "comparability_note"}
        for field in baseline.keys() - annotations:
            changed = copy.deepcopy(original)
            value = changed["tasks"][-1]["baseline"][field]
            if isinstance(value, list):
                value = [score + 1 for score in value]
            elif isinstance(value, dict):
                value = {**value, next(iter(value)): "changed"}
            elif isinstance(value, (int, float)):
                value += 1
            else:
                value = (
                    "sha256:" + "a" * 64 if value.startswith("sha256:") else "changed"
                )
            changed["tasks"][-1]["baseline"][field] = value
            catalog_path.write_text(renderer.json_text(changed))
            before = snapshot(self.root)
            with (
                self.subTest(field=field),
                self.assertRaises((ValueError, deploy.DeployError)),
            ):
                self.invoke("--check")
            self.assertEqual(snapshot(self.root), before)
        changed = copy.deepcopy(original)
        changed["tasks"][-1]["baseline"]["historical_scores"][0] = True
        catalog_path.write_text(renderer.json_text(changed))
        with self.assertRaisesRegex(ValueError, "freshly derived"):
            self.invoke("--check")
        baseline.update(
            provenance="measured",
            comparable=True,
            comparability_note="operator verified revision",
        )
        catalog_path.write_text(renderer.json_text(original))
        self.invoke("--check")

    def test_publish_conflict_never_pushes_or_logs_in(self):
        with (
            mock.patch.object(
                importer, "remote_digest", return_value="sha256:" + "d" * 64
            ),
            mock.patch.object(
                importer, "local_digest", return_value="sha256:" + "e" * 64
            ),
        ):
            before = snapshot(self.root)
            with self.assertRaisesRegex(ValueError, "refusing to push existing tag"):
                self.invoke("--publish")
            self.assertEqual(snapshot(self.root), before)
        self.assertFalse(
            any(
                call.args[0][0] in ("push", "login")
                for call in self.docker.call_args_list
            )
        )
        self.build.assert_called_once()
        self.assertFalse(self.build.call_args.kwargs["push"])

    def test_publish_success_resolves_only_new_task_after_verified_build(self):
        self.registry.side_effect = None
        self.registry.return_value = "sha256:" + "c" * 64
        before = deploy.load_json(self.root / "config/task-catalog.json")
        with mock.patch.object(importer, "remote_digest", return_value=None):
            self.invoke("--publish")
        pushed = [
            call.args[0]
            for call in self.docker.call_args_list
            if call.args[0][0] == "push"
        ]
        self.assertEqual(len(pushed), 4)
        self.assertFalse(
            any(call.args[0][0] == "login" for call in self.docker.call_args_list)
        )
        self.assertEqual(self.entry()["reference"]["digest"], "sha256:" + "c" * 64)
        after = deploy.load_json(self.root / "config/task-catalog.json")
        self.assertEqual(before["tasks"][0], after["tasks"][0])
        self.assertEqual(before["candidate"], after["candidate"])

    def test_reference_build_uses_external_context_and_launcher_layout(self):
        def docker(command, **kwargs):
            if command[0] == "build":
                context = Path(command[-1])
                self.assertFalse(context.is_relative_to(self.source))
                self.assertEqual(
                    (context / "payload").read_bytes(),
                    (self.source / "judge/reference.patch").read_bytes(),
                )
                text = (context / "Dockerfile").read_text()
                self.assertIn("FROM scratch\n", text)
                self.assertIn("/opt/frontier/reference-payload/solution.patch", text)
                self.assertEqual(text.count("--chmod=0555"), 2)
                self.assertIn("--provenance=false", command)
                self.assertIn("linux/amd64", command)

        self.docker.side_effect = docker
        self.invoke()

    def test_no_reference_missing_result_and_explicit_payload(self):
        (self.source / "judge/result.json").unlink()
        self.invoke("--no-reference")
        self.assertIsNone(self.entry()["reference"])
        self.assertEqual(
            self.entry()["baseline"],
            {
                "provenance": "unverified",
                "comparable": False,
                "comparability_note": "no judge/result.json in the export",
            },
        )
        (self.source / "judge/reference.patch").unlink()
        (self.source / "judge/reference.py").write_text('print("payload")\n')
        self.invoke("--replace")
        self.assertIsNone(self.entry()["reference"])
        payload = Path(self.temp.name) / "solution.py"
        payload.write_text("print(42)\n")
        self.invoke("--replace", "--reference-payload", str(payload))
        self.assertEqual(
            self.entry()["reference"]["payload_sha256"], importer.file_sha(payload)
        )

    def test_baseline_comparable_requires_operator_note_and_preserves_derivation(self):
        self.invoke()
        baseline = self.entry()["baseline"]
        self.assertFalse(baseline["comparable"])
        note = "Author confirmed result.json measured this exact exported revision"
        with self.assertRaisesRegex(ValueError, "use --replace"):
            self.invoke("--baseline-comparable", note)
        self.invoke("--replace", "--baseline-comparable", note)
        self.assertEqual(
            self.entry()["baseline"],
            {
                **baseline,
                "provenance": "historical-authoring",
                "comparable": True,
                "comparability_note": note,
            },
        )
        before = snapshot(self.root)
        self.invoke("--baseline-comparable", note)
        self.assertEqual(snapshot(self.root), before)
        receipt = deploy.load_json(self.root / ".generated/import/minimal_task.json")
        self.assertEqual(receipt["intent"]["baseline_comparable"], note)

    def test_baseline_comparable_refuses_missing_results_bindings_and_empty_notes(self):
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "nonempty operator note"):
            self.invoke("--baseline-comparable", "  ")
        for member in (
            "judge/result.json",
            "judge/evaluator.py",
            "judge/reference.patch",
        ):
            path = self.source / member
            body = path.read_bytes()
            path.unlink()
            with (
                self.subTest(member=member),
                self.assertRaisesRegex(ValueError, "complete baseline binding"),
            ):
                self.invoke("--baseline-comparable", "operator assertion")
            path.write_bytes(body)
        self.assertEqual(snapshot(self.root), before)
        self.build.assert_not_called()
        self.docker.assert_not_called()
        self.registry.assert_not_called()

    def test_other_relative_entrypoint_allowed_only_without_reference(self):
        policy = deploy.load_json(self.source / "policy.json")
        self.change_policy(
            submission={**policy["submission"], "entrypoint": "answer/solution.py"}
        )
        with self.assertRaisesRegex(ValueError, "under payload/"):
            self.invoke()
        self.build.assert_not_called()
        self.invoke("--no-reference")
        self.assertIsNone(self.entry()["reference"])

    def test_schema_and_catalog_validation_precede_build_push_and_destination_write(
        self,
    ):
        policy = deploy.load_json(self.source / "policy.json")
        self.change_policy(agent={**policy["agent"], "cpu_millis": -1})
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "task-config schema"):
            self.invoke("--publish")
        self.assertEqual(snapshot(self.root), before)
        self.build.assert_not_called()
        self.docker.assert_not_called()
        self.registry.assert_not_called()
        self.change_policy(agent=policy["agent"])
        real_entry = importer.catalog_entry

        def invalid_entry(*args):
            entry = real_entry(*args)
            entry["agent_image"]["tag"] = "mutable"
            return entry

        with mock.patch.object(importer, "catalog_entry", side_effect=invalid_entry):
            with self.assertRaisesRegex(deploy.DeployError, "reviewed source tag"):
                self.invoke("--publish")
        self.assertEqual(snapshot(self.root), before)
        self.build.assert_not_called()
        self.docker.assert_not_called()
        self.registry.assert_not_called()

    def test_verifier_only_gpu_creates_only_enabled_variant(self):
        self.change_policy(
            gpu={"verifier": {"max_gpu_instances": 1, "max_gpu_seconds": 60}}
        )
        (self.source / "verifier-gpu").mkdir()
        (self.source / "verifier-gpu/Dockerfile").write_text(
            "FROM python:3.11-slim-bookworm\n"
        )
        self.invoke()
        self.assertIn("verifier_gpu_image", self.entry())
        self.assertNotIn("agent_gpu_image", self.entry())

    def test_declared_sidecars_publish_task_owned_images(self):
        self.change_policy(
            gpu={"verifier": {"max_gpu_instances": 1, "max_gpu_seconds": 60}},
            sidecars={
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
            },
            gpu_parent_resources={
                "verifier": {"cpu_millis": 16000, "memory_bytes": 64 * 1024**3}
            },
        )
        for role in ("verifier-gpu", "sidecar-scheduler", "sidecar-swe-000"):
            (self.source / role).mkdir()
            (self.source / role / "Dockerfile").write_text(
                "FROM python:3.11-slim-bookworm\n"
            )
        self.invoke()
        entry = self.entry()
        # Image roles, not sidecar names: swe-000-check shares the swe-000 image.
        self.assertEqual(set(entry["sidecar_images"]), {"scheduler", "swe-000"})
        image = entry["sidecar_images"]["swe-000"]
        self.assertEqual(
            image["repository"],
            "docker.io/joyemang33/fcs2-task-minimal_task-sidecar-swe-000",
        )
        self.assertEqual(image["tag"], entry["verifier_image"]["tag"])
        self.assertEqual(image["digest"], importer.PLACEHOLDER)
        definition = deploy.load_json(self.root / "config/task-definitions.json")[
            "minimal_task"
        ]
        self.assertEqual(definition["sidecars"]["scheduler"]["phases"], ["train", "eval"])
        self.assertTrue(definition["sidecars"]["scheduler"]["attestation_key"])
        self.assertEqual(
            definition["gpu_parent_resources"]["verifier"]["cpu_millis"], 16000
        )
        names = importer.builder.image_names(
            "docker.io/joyemang33", "minimal_task", definition["image_tag"], ["swe-000"]
        )
        self.assertEqual(
            names["sidecar-swe-000"],
            f"docker.io/joyemang33/fcs2-task-minimal_task-sidecar-swe-000:{definition['image_tag']}",
        )
        # Sidecar images are part of the publication set and resolve like roles.
        resolved = {
            image["repository"]
            for image, _ in importer.resolver.task_images(entry)
        }
        self.assertIn(image["repository"], resolved)

    def test_result_without_reference_retains_scores_as_unverified(self):
        (self.source / "judge/reference.patch").unlink()
        self.invoke()
        self.assertIsNone(self.entry()["reference"])
        self.assertEqual(self.entry()["baseline"]["historical_scores"], [1.0, 3.0])
        self.assertEqual(self.entry()["baseline"]["provenance"], "unverified")
        self.assertFalse(self.entry()["baseline"]["comparable"])
        self.assertEqual(
            self.entry()["baseline"]["comparability_note"],
            "missing baseline binding files: judge/reference.patch",
        )

    def test_result_with_reference_is_historical_authoring_and_not_comparable(self):
        self.assertTrue((self.source / "judge/result.json").is_file())
        self.assertTrue((self.source / "judge/reference.patch").is_file())
        self.invoke()
        baseline = self.entry()["baseline"]
        self.assertEqual(baseline["historical_scores"], [1.0, 3.0])
        self.assertEqual(baseline["provenance"], "historical-authoring")
        self.assertFalse(baseline["comparable"])
        self.assertIn("does not attest", baseline["comparability_note"])

    def test_published_plan_is_offline_and_repeat_is_unchanged(self):
        path, _ = self.manifest()
        before = snapshot(self.root)
        self.invoke("--published-images", str(path), "--plan")
        self.assertEqual(snapshot(self.root), before)
        self.registry.assert_not_called()
        self.registry.side_effect = None
        self.registry.return_value = "sha256:" + "c" * 64
        self.invoke("--published-images", str(path))
        before = snapshot(self.root)
        self.registry.reset_mock()
        self.invoke("--published-images", str(path))
        self.assertEqual(snapshot(self.root), before)
        self.registry.assert_not_called()

    def test_publish_checks_late_conflicts_before_any_push(self):
        with (
            mock.patch.object(
                importer,
                "remote_digest",
                side_effect=[None, None, "sha256:" + "d" * 64],
            ),
            mock.patch.object(importer, "local_digest", return_value=None),
        ):
            with self.assertRaisesRegex(ValueError, "unknown local digest"):
                self.invoke("--publish")
        self.assertFalse(
            any(call.args[0][0] == "push" for call in self.docker.call_args_list)
        )

    def test_existing_matching_digests_are_not_pushed(self):
        with (
            mock.patch.object(
                importer, "remote_digest", return_value="sha256:" + "d" * 64
            ),
            mock.patch.object(
                importer, "local_digest", return_value="sha256:" + "d" * 64
            ),
        ):
            self.registry.side_effect = None
            self.registry.return_value = "sha256:" + "d" * 64
            self.invoke("--publish")
        self.assertFalse(
            any(call.args[0][0] == "push" for call in self.docker.call_args_list)
        )

    def test_remote_only_manifest_404_is_absent(self):
        image = {"repository": "docker.io/test/image", "tag": "v1"}
        for url, absent in [
            ("https://registry-1.docker.io/v2/test/image/manifests/v1", True),
            (
                "https://registry-1.docker.io/v2/test/image/manifests/sha256:"
                + "a" * 64,
                False,
            ),
            ("https://registry-1.docker.io/v2/other/image/manifests/v1", False),
            ("https://auth.docker.io/token", False),
            ("https://registry-1.docker.io/v2/test/image/blobs/sha256:abc", False),
        ]:
            error = HTTPError(url, 404, "missing", {}, None)
            self.addCleanup(error.close)
            self.registry.side_effect = error
            if absent:
                self.assertIsNone(importer.remote_digest(image))
            else:
                with self.assertRaises(HTTPError):
                    importer.remote_digest(image)

    def test_index_child_404_refuses_push(self):
        child = "sha256:" + "a" * 64
        index = json.dumps(
            {
                "manifests": [
                    {
                        "digest": child,
                        "platform": {"os": "linux", "architecture": "amd64"},
                    }
                ]
            }
        ).encode()

        def http(url, headers):
            if url.startswith("https://auth.docker.io/"):
                return b'{"token":"anonymous-fixture"}', {}
            if url.endswith("/manifests/" + child):
                error = HTTPError(url, 404, "missing child", {}, None)
                self.addCleanup(error.close)
                raise error
            return index, {"Docker-Content-Digest": deploy.sha256_bytes(index)}

        self.registry.side_effect = REAL_RESOLVE_DIGEST
        with mock.patch.object(importer.resolver, "http_bytes", side_effect=http):
            with self.assertRaisesRegex(HTTPError, "missing child"):
                self.invoke("--publish")
        self.assertFalse(
            any(call.args[0][0] == "push" for call in self.docker.call_args_list)
        )

    def test_check_detects_imported_tree_drift_without_writes(self):
        self.invoke()
        (self.root / importer.PROBLEMS / "minimal_task/new-file").write_text("drift")
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.invoke("--check")
        self.assertEqual(snapshot(self.root), before)

    def test_checks_run_as_required_subprocesses(self):
        with (
            mock.patch.dict(os.environ),
            mock.patch.object(importer.subprocess, "run") as run,
        ):
            os.environ.pop("PYTHON", None)
            # Invoke the unmocked function saved on the class before setUp.
            self.real_checks()
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [["make", "validate", "CONFIG=config/site.example.json"], ["make", "test"]],
        )
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["cwd"], self.root)
            self.assertTrue(call.kwargs["check"])
            self.assertEqual(
                call.kwargs["env"]["PYTHON"],
                shlex.quote(sys.executable).replace("$", "$$"),
            )

    real_checks = staticmethod(importer.run_checks)

    def test_published_registry_index_config_and_exact_payload_bodies(self):
        path, manifest = self.manifest()
        blobs, manifests = {}, {}
        for role, image in manifest["images"].items():
            members = {"unused": b"image layer"}
            if role == "datasets":
                definition = importer.validate_export(self.source)
                members = {
                    f"datasets/{mode}/sha256/{definition['datasets'][f'{mode}_sha256'][7:]}.json": (
                        self.source / f"datasets/{mode}.json"
                    ).read_bytes()
                    for mode in ("train", "eval")
                }
            elif role == "reference":
                members = {
                    "opt/frontier/reference-candidate": (
                        self.root / importer.LAUNCHER
                    ).read_bytes(),
                    "opt/frontier/reference-payload/solution.patch": (
                        self.source / "judge/reference.patch"
                    ).read_bytes(),
                }
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                if role == "reference":
                    for name in (
                        "opt",
                        "opt/frontier",
                        "opt/frontier/reference-payload",
                    ):
                        info = tarfile.TarInfo(name)
                        info.type, info.mode = tarfile.DIRTYPE, 0o755
                        archive.addfile(info)
                for name, body in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(body)
                    info.mode = 0o555
                    archive.addfile(info, io.BytesIO(body))
            layer = stream.getvalue()
            config = json.dumps({"os": "linux", "architecture": "amd64"}).encode()
            config_descriptor = {
                "digest": deploy.sha256_bytes(config),
                "size": len(config),
                "mediaType": "application/vnd.oci.image.config.v1+json",
            }
            layer_descriptor = {
                "digest": deploy.sha256_bytes(layer),
                "size": len(layer),
                "mediaType": "application/vnd.oci.image.layer.v1.tar",
            }
            body = json.dumps(
                {
                    "schemaVersion": 2,
                    "config": config_descriptor,
                    "layers": [layer_descriptor],
                }
            ).encode()
            image["digest"] = deploy.sha256_bytes(body)
            index = json.dumps(
                {
                    "schemaVersion": 2,
                    "manifests": [
                        {
                            "digest": image["digest"],
                            "size": len(body),
                            "mediaType": "application/vnd.oci.image.manifest.v1+json",
                            "platform": {"os": "linux", "architecture": "amd64"},
                        }
                    ],
                }
            ).encode()
            repo = image["repository"].removeprefix("docker.io/")
            manifests[f"{repo}/manifests/{image['tag']}"] = index
            manifests[f"{repo}/manifests/{image['digest']}"] = body
            blobs[config_descriptor["digest"]] = config
            blobs[layer_descriptor["digest"]] = layer
        path.write_text(json.dumps(manifest))

        def http(url, headers):
            if url.startswith("https://auth.docker.io/"):
                return b'{"token":"anonymous-fixture"}', {}
            key = url.split("/v2/", 1)[1]
            body = (
                blobs[key.split("/blobs/")[1]] if "/blobs/" in key else manifests[key]
            )
            return body, {"Docker-Content-Digest": deploy.sha256_bytes(body)}

        self.registry.side_effect = REAL_RESOLVE_DIGEST
        with (
            mock.patch.object(importer.resolver, "http_bytes", side_effect=http),
            mock.patch.object(
                importer.resolver,
                "http_blob",
                side_effect=lambda url, headers, limit: blobs[url.split("/blobs/")[1]],
            ),
        ):
            self.invoke("--published-images", str(path))
            self.invoke("--check")
        self.build.assert_not_called()
        self.docker.assert_not_called()


class BaselineTests(unittest.TestCase):
    def test_duplicate_result_keys_are_rejected_at_every_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "minimal_task"
            shutil.copytree(ROOT / "tests/fixtures/export-minimal", root)
            definition = importer.validate_export(root)
            path = root / "judge/result.json"
            original = path.read_text()
            for key in ("schema", "reference", "scores", "mean"):
                body = original.replace(
                    '"' + key + '":', '"' + key + '": null, "' + key + '":', 1
                )
                self.assertNotEqual(body, original)
                path.write_text(body)
                with (
                    self.subTest(key=key),
                    self.assertRaisesRegex(deploy.DeployError, "duplicate"),
                ):
                    baseline_from_result_json(root, definition)

    def test_import_baselines_bind_current_sources_and_retain_original_authoring_scores(
        self,
    ):
        baselines = load_baselines()
        # These are external author/operator annotations, absent from result.json.
        annotations = {
            "result_json_source",
            "provenance",
            "comparable",
            "comparability_note",
            "calibration",
            "historical_scores",
            "historical_mean",
            "historical_spread",
        }
        count = 0
        for name, baseline in baselines.items():
            if "historical_scores" not in baseline:
                continue
            if "measurement" in baseline:
                count += 1
                definition = deploy.TASK_DEFINITIONS[name]
                # External evidence deliberately has no result.json derivation.
                # This catalog regression checks its stored summary and original
                # source bindings; importer qualification/check are exercised
                # with complete positive/negative packets in test_measured_baseline.
                from preview import measured_baseline

                measurement = baseline["measurement"]
                self.assertEqual(
                    measurement["schema_version"], measured_baseline.SCHEMA
                )
                self.assertEqual(baseline["provenance"], "measured")
                self.assertIs(baseline["comparable"], True)
                self.assertNotIn("result_json_source", baseline)
                for field, prefix in (
                    ("evaluation_id", "eval"),
                    ("run_id", "run"),
                    ("task_version_id", "taskver"),
                    ("candidate_version_id", "candver"),
                ):
                    measured_baseline.typed_id(measurement[field], prefix)
                measured_baseline.digest(measurement["packet_sha256"])
                self.assertIn(measurement["verdict"], ("PASS", "FAIL"))
                self.assertTrue(measured_baseline.finite(measurement["train_score"]))
                self.assertTrue(
                    measured_baseline.same_score(
                        measurement["management_score"], baseline["historical_mean"]
                    )
                )
                self.assertEqual(
                    baseline["historical_scores"], [baseline["historical_mean"]]
                )
                self.assertEqual(baseline["historical_spread"], 0.0)
                task = next(
                    row
                    for row in deploy.load_json(ROOT / "config/task-catalog.json")[
                        "tasks"
                    ]
                    if row["name"] == name
                )
                importer.publication.validate_baseline(
                    name, baseline, task["reference"], definition, root=ROOT
                )
                continue
            count += 1
            definition = deploy.TASK_DEFINITIONS[name]
            derived = baseline_from_result_json(
                renderer.problem_directory(ROOT / importer.PROBLEMS, name), definition
            )
            self.assertEqual(
                renderer.json_text(
                    {k: v for k, v in derived.items() if k not in annotations}
                ),
                renderer.json_text(
                    {k: v for k, v in baseline.items() if k not in annotations}
                ),
            )
            prior = deploy.load_json(
                ROOT / "tests/fixtures/evidence/beat-spec-20260911-prior-reference-baselines.json"
            ).get(name, baseline)
            for key in ("historical_scores", "historical_mean", "historical_spread"):
                self.assertEqual(derived[key], prior[key])
            self.assertFalse(derived["comparable"])
            root = renderer.problem_directory(ROOT / importer.PROBLEMS, name)
            missing = [
                member
                for member in ("judge/evaluator.py", "judge/reference.patch")
                if not (root / member).is_file()
            ]
            if missing:
                self.assertEqual(derived["provenance"], "unverified")
                self.assertEqual(
                    derived["comparability_note"],
                    "missing baseline binding files: " + ", ".join(missing),
                )
            else:
                self.assertEqual(derived["provenance"], "historical-authoring")
                self.assertIn("does not attest", derived["comparability_note"])
        self.assertEqual(
            count,
            sum("historical_scores" in baseline for baseline in baselines.values()),
        )

    def test_bad_result_statistics_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "minimal_task"
            shutil.copytree(ROOT / "tests/fixtures/export-minimal", root)
            definition = importer.validate_export(root)
            path = root / "judge/result.json"
            original = json.loads(path.read_text())
            for field, value in [
                ("ok", False),
                ("scores", [float("nan")]),
                ("samples", 3),
                ("spread", {"n": 2, "mean": 42, "spread": 2}),
            ]:
                result = copy.deepcopy(original)
                result["final"]["reference"][field] = value
                path.write_text(json.dumps(result))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    baseline_from_result_json(root, definition)
