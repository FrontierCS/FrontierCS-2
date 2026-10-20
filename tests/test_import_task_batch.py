"""Source-first releases validate and install together or leave both untouched."""

import deployment_test_guard  # noqa: F401  # isort: skip

import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

import test_import_task as legacy
from test_import_task import deploy, importer, renderer, snapshot


class BatchTests(unittest.TestCase):
    def setUp(self):
        legacy.ImportTests.setUp(self)
        legacy.ImportTests.invoke(self)
        self.other = Path(self.temp.name) / "existing_task"
        shutil.copytree(self.root / importer.PROBLEMS / "existing_task", self.other)
        self.exports = [self.source, self.other]
        self.batch = Path(self.temp.name) / "batch.json"
        self.members = []
        for source in self.exports:
            definition = importer.validate_export(source)
            entry = importer.catalog_entry(
                source, definition, "joyemang33", source / "judge/reference.patch"
            )
            images = {
                role: {k: entry[key][k] for k in ("repository", "tag", "digest")}
                for role, key in importer.ROLE_KEYS.items()
                if entry.get(key) is not None
            }
            for image in images.values():
                image["digest"] = "sha256:" + "c" * 64
            manifest = Path(self.temp.name) / (source.name + "-images.json")
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "frontier.task-release-images/v1",
                        "problem_id": source.name,
                        "source_sha256": definition["source"]["source_set_sha256"],
                        "images": images,
                    }
                )
            )
            self.members.append(
                {
                    "export_dir": str(source),
                    "published_images": str(manifest),
                    "no_reference": False,
                }
            )
        peer = self.root / importer.PROBLEMS / "untouched_peer"
        shutil.copytree(self.other, peer)
        for name, key in [
            ("policy.json", "problem_id"),
            ("upstream.lock.json", "problem_id"),
            ("judge/result.json", "task_id"),
        ]:
            body = deploy.load_json(peer / name)
            body[key] = "untouched_peer"
            (peer / name).write_text(json.dumps(body))
        definition = importer.validate_export(peer)
        entry = importer.catalog_entry(
            peer, definition, "joyemang33", peer / "judge/reference.patch"
        )
        for image, _ in importer.resolver.task_images(entry):
            image["digest"] = "sha256:" + "b" * 64
        catalog = deploy.load_json(self.root / "config/task-catalog.json")
        catalog["tasks"].append(entry)
        (self.root / "config/task-catalog.json").write_text(renderer.json_text(catalog))
        definitions = deploy.load_json(self.root / "config/task-definitions.json")
        definitions["untouched_peer"] = definition
        (self.root / "config/task-definitions.json").write_text(
            json.dumps(definitions, indent=4) + "\n"
        )
        self.registry.side_effect = None
        self.registry.return_value = "sha256:" + "c" * 64
        self.write_request()
        self.checks.reset_mock()

    def write_request(self):
        self.batch.write_text(
            json.dumps(
                {
                    "schema_version": "frontier.task-import-batch/v1",
                    "exports": self.members,
                }
            )
        )

    def invoke(self, *args):
        return importer.main(["--batch", str(self.batch), "--replace", *args])

    def test_complete_batch_and_repeat_are_idempotent(self):
        before_trees = [snapshot(source) for source in self.exports]
        peer_before = snapshot(self.root / importer.PROBLEMS / "untouched_peer")
        row_before = deploy.load_json(self.root / "config/task-catalog.json")["tasks"][
            -1
        ]
        self.assertEqual(self.invoke(), 0)
        first = snapshot(self.root)
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(snapshot(self.root), first)
        for source, expected in zip(self.exports, before_trees, strict=True):
            self.assertEqual(
                snapshot(self.root / importer.PROBLEMS / source.name), expected
            )
        self.assertEqual(self.checks.call_count, 2)
        self.assertEqual(
            snapshot(self.root / importer.PROBLEMS / "untouched_peer"), peer_before
        )
        self.assertEqual(
            deploy.load_json(self.root / "config/task-catalog.json")["tasks"][-1],
            row_before,
        )

    def test_two_source_first_changes_import_together(self):
        for source, member in zip(self.exports, self.members, strict=True):
            path = source / "judge/evaluator.py"
            path.write_text(path.read_text() + "\n# source-first release change\n")
            shutil.copyfile(
                path, self.root / importer.PROBLEMS / source.name / "judge/evaluator.py"
            )
            manifest = Path(member["published_images"])
            body = json.loads(manifest.read_text())
            definition = importer.validate_export(source)
            body["source_sha256"] = definition["source"]["source_set_sha256"]
            for role, image in body["images"].items():
                if role != "datasets":
                    image["tag"] = definition["image_tag"]
            manifest.write_text(json.dumps(body))
        before = snapshot(self.root)
        with self.assertRaises((ValueError, deploy.DeployError)):
            importer.main(
                [
                    str(self.source),
                    "--replace",
                    "--published-images",
                    self.members[0]["published_images"],
                ]
            )
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(self.invoke(), 0)

    def test_invalid_either_member_leaves_all_unchanged(self):
        for index in (0, 1):
            path = Path(self.members[index]["published_images"])
            original = path.read_bytes()
            body = json.loads(original)
            body["source_sha256"] = "sha256:" + "0" * 64
            path.write_text(json.dumps(body))
            before = snapshot(self.root)
            with self.assertRaises((ValueError, deploy.DeployError)):
                self.invoke()
            self.assertEqual(snapshot(self.root), before)
            path.write_bytes(original)
        self.checks.assert_not_called()

    def test_closed_schema_duplicate_and_source_conflict(self):
        original = json.loads(self.batch.read_text())
        cases = [
            dict(original, unexpected=True),
            dict(original, exports=[self.members[0], self.members[0]]),
            dict(
                original,
                exports=[dict(self.members[0], no_reference="false"), self.members[1]],
            ),
            dict(
                original,
                exports=[
                    dict(
                        self.members[0],
                        export_dir=str(self.root / importer.PROBLEMS / "minimal_task"),
                    ),
                    self.members[1],
                ],
            ),
        ]
        for body in cases:
            self.batch.write_text(json.dumps(body))
            before = snapshot(self.root)
            with self.assertRaises((ValueError, deploy.DeployError)):
                self.invoke()
            self.assertEqual(snapshot(self.root), before)

    def test_write_or_check_failure_rolls_back_both_members(self):
        actual_write = importer.atomic_write
        for failure in ("task-definitions.json", "existing_task.json", "checks"):
            before = snapshot(self.root)
            failed = False

            def write(path, body, failure=failure):
                nonlocal failed
                actual_write(path, body)
                if path.name == failure and not failed:
                    failed = True
                    raise OSError("injected second-step write")

            with mock.patch.object(importer, "atomic_write", side_effect=write):
                self.checks.side_effect = (
                    OSError("injected checks") if failure == "checks" else None
                )
                with self.assertRaises(OSError):
                    self.invoke()
            self.assertEqual(snapshot(self.root), before)
        self.checks.side_effect = None

    def test_second_tree_swap_failure_restores_both(self):
        before = snapshot(self.root)
        actual = Path.rename
        failed = False

        def rename(path, target):
            nonlocal failed
            if (
                path.name == "existing_task"
                and "staged-" in str(path)
                and "previous" not in path.parts
                and not failed
            ):
                failed = True
                raise OSError("second tree install failed")
            return actual(path, target)

        with mock.patch.object(Path, "rename", rename), self.assertRaises(OSError):
            self.invoke()
        self.assertEqual(snapshot(self.root), before)

    def test_plan_never_installs_or_runs_checks(self):
        before = snapshot(self.root)
        self.assertEqual(self.invoke("--plan"), 0)
        self.assertEqual(snapshot(self.root), before)
        self.checks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
