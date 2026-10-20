from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_test_prerequisites as prerequisites


class PrerequisiteTests(unittest.TestCase):
    def test_root_lfs_rules_detect_pointers_including_hidden_nested_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".gitattributes").write_text(
                "# Only these explicit rules own LFS inputs\n"
                "tasks/demo/data/** filter=lfs diff=lfs merge=lfs -text\n"
                "tasks/demo/bin/runner filter=lfs diff=lfs merge=lfs -text\n"
                "*.weights filter=lfs diff=lfs merge=lfs -text\n"
            )
            matched = [
                "tasks/demo/data/a.parquet",
                "tasks/demo/data/.nested/b.parquet",
                "tasks/demo/bin/runner",
                "tasks/another/model.weights",
            ]
            ignored = ["tasks/demo/untracked.txt", "outside/model.weights"]
            for name in matched + ignored:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(prerequisites.LFS_POINTER_PREFIX + b"\nextension\n")
            with mock.patch.object(prerequisites.importlib, "import_module"):
                errors = prerequisites.check_prerequisites(root)
                self.assertEqual(len(errors), 1)
                for name in matched:
                    self.assertIn(name, errors[0])
                for name in ignored:
                    self.assertNotIn(name, errors[0])
                self.assertIn("git lfs pull", errors[0])
                self.assertIn("must not be skipped", errors[0])
                for name in matched:
                    (root / name).write_bytes(
                        b"real object bytes\n" + prerequisites.LFS_POINTER_PREFIX
                    )
                self.assertEqual(prerequisites.check_prerequisites(root), [])

    def test_missing_modules_explain_dependencies_and_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".gitattributes").write_text("")
            with mock.patch.object(
                prerequisites.importlib,
                "import_module",
                side_effect=ModuleNotFoundError,
            ) as imported:
                errors = prerequisites.check_prerequisites(root)
            self.assertEqual(
                imported.call_args_list,
                [mock.call(name) for name in prerequisites.REQUIRED_MODULES],
            )
            self.assertEqual(len(errors), len(prerequisites.REQUIRED_MODULES))
            for error, (name, reason) in zip(
                errors, prerequisites.REQUIRED_MODULES.items(), strict=False
            ):
                for text in (name, reason, sys.executable, "pip install", "make test"):
                    self.assertIn(text, error)

    def test_missing_attributes_fails_and_cli_returns_failure(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(prerequisites.importlib, "import_module"),
        ):
            errors = prerequisites.check_prerequisites(Path(directory))
        self.assertIn("Cannot check root .gitattributes", errors[0])
        with (
            mock.patch.object(
                prerequisites, "check_prerequisites", return_value=errors
            ),
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            self.assertEqual(prerequisites.main(), 1)
        self.assertIn("Test prerequisites: FAILED", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
