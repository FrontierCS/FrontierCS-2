#
import deployment_test_guard  # noqa: F401  # isort: skip
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import render_task_definition as renderer


class DomainPaths(unittest.TestCase):
    def test_flat_grouped_ambiguous_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            flat = root / "sample_task"
            flat.mkdir()
            (flat / "policy.json").write_text("{}")
            before = renderer.source_sha256(flat)
            self.assertEqual(renderer.problem_directory(root, "sample_task"), flat)
            domain = root / "databases"
            domain.mkdir()
            moved = domain / flat.name
            flat.rename(moved)
            self.assertEqual(renderer.problem_directory(root, "sample_task"), moved)
            self.assertEqual(renderer.source_sha256(moved), before)
            self.assertEqual(
                renderer.catalog_problem_directories(["sample_task"], root),
                {"sample_task": moved},
            )
            flat.mkdir()
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                renderer.problem_directory(root, "sample_task")

    def test_invalid_name(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            renderer.problem_directory(pathlib.Path("."), "../outside")


class PreviousDomainPaths(unittest.TestCase):
    def test_historical_flat_and_grouped_layouts(self):
        from unittest import mock

        from preview import images

        prefix = "tasks/frontier-cs-2.0-demo/problems/"
        for relative in ("sample_task", "databases/sample_task"):
            with mock.patch.object(
                images, "command", return_value=prefix + relative + "/policy.json"
            ):
                self.assertEqual(
                    images.previous_problem_path(
                        pathlib.Path("."), "commit", "sample_task"
                    ),
                    prefix + relative,
                )
        duplicate = (
            prefix
            + "sample_task/policy.json\n"
            + prefix
            + "databases/sample_task/policy.json"
        )
        with mock.patch.object(images, "command", return_value=duplicate):
            with self.assertRaisesRegex(
                images.IncrementalFailure, "one source directory"
            ):
                images.previous_problem_path(pathlib.Path("."), "commit", "sample_task")
