"""Exercise real post-import gates without recursively running the outer suite."""

from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_task_definition as renderer


def copy_repository(source: Path, target: Path) -> None:
    def ignore(directory, names):
        relative = Path(directory).relative_to(source)
        # Every task member participates in its source identity, even names
        # normally used for repository infrastructure or Python caches.
        if relative.parts and relative.parts[0] == "tasks":
            return []
        excluded = []
        for name in names:
            path = relative / name
            if (
                path.as_posix() in {".git", ".state", ".generated", "config/site.json"}
                or name == "__pycache__"
                or path.parts[0] == "terraform"
                and (name == ".terraform" or ".tfvars" in name or ".tfstate" in name)
            ):
                excluded.append(name)
        return excluded

    # Infrastructure tests also read these sources; site credentials and state
    # stay outside the copy. Independent task inodes preserve source audits and
    # prevent nested tests from changing the original repository.
    for member in (
        "config",
        "tasks",
        "scripts",
        "tests",
        "docs",
        "runtime",
        "smoke",
        "terraform",
        "candidates",
    ):
        shutil.copytree(
            source / member,
            target / member,
            ignore=ignore,
            copy_function=shutil.copy2,
            symlinks=True,
        )
    for member in ("Makefile", ".gitattributes"):
        shutil.copyfile(source / member, target / member)


class RepositoryCopyTests(unittest.TestCase):
    def test_copy_exclusions_preserve_task_members_and_export_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.mkdir()
            target.mkdir()
            for member in (
                "config",
                "tasks",
                "scripts",
                "tests",
                        "docs",
                "runtime",
                "smoke",
                "terraform",
                "candidates",
            ):
                (source / member).mkdir()
            for member in ("Makefile", ".gitattributes"):
                (source / member).write_text("fixture\n")
            export = source / "tasks/minimal_task"
            shutil.copytree(ROOT / "tests/fixtures/export-minimal", export)
            excluded = [
                ".git/config",
                ".state/state.json",
                ".generated/output.json",
                "config/site.json",
                "terraform/live/main.tfvars",
                "terraform/live/main.tfvars.json",
                "terraform/live/main.tfstate.backup",
                "terraform/.terraform/cache",
                "scripts/__pycache__/script.pyc",
            ]
            task_members = excluded + [
                "site.json",
                "__pycache__/task.pyc",
                "nested/config/site.json",
            ]
            for base, members in ((source, excluded), (export, task_members)):
                for member in members:
                    path = base / member
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("included in task identity: " + member)
            (export / "verifier/verify.py").chmod(0o755)
            # Similar names elsewhere are ordinary files, not site configuration.
            (source / "docs/site.json").write_text("documentation fixture")
            copy_repository(source, target)
            for member in excluded:
                self.assertFalse((target / member).exists(), member)
            copied_export = target / "tasks/minimal_task"
            for member in task_members:
                self.assertEqual(
                    (copied_export / member).read_bytes(),
                    (export / member).read_bytes(),
                )
            self.assertEqual(
                renderer.source_sha256(copied_export), renderer.source_sha256(export)
            )
            for path in export.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    copied = copied_export / path.relative_to(export)
                    self.assertFalse(copied.samefile(path), str(path))
                    self.assertEqual(copied.stat().st_nlink, 1, str(path))
                    self.assertEqual(copied.stat().st_mode, path.stat().st_mode)
            self.assertEqual(
                (target / "docs/site.json").read_bytes(),
                (source / "docs/site.json").read_bytes(),
            )

    def test_task_copy_mutations_are_isolated_in_both_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            shutil.copytree(ROOT / "tests/fixtures/export-minimal", source / "tasks/task")
            for member in ("config", "scripts", "tests", "docs",
                           "runtime", "smoke", "terraform", "candidates"):
                (source / member).mkdir()
            for member in ("Makefile", ".gitattributes"):
                (source / member).write_text("fixture\n")
            original = source / "tasks/task/verifier/verify.py"
            original.chmod(0o755)
            original_bytes = original.read_bytes()
            copy_repository(source, target)
            copied = target / "tasks/task/verifier/verify.py"
            copied.write_bytes(b"copied mutation\n")
            copied.chmod(0o600)
            self.assertEqual(original.read_bytes(), original_bytes)
            self.assertEqual(original.stat().st_mode & 0o777, 0o755)
            original.write_bytes(b"original mutation\n")
            original.chmod(0o644)
            self.assertEqual(copied.read_bytes(), b"copied mutation\n")
            self.assertEqual(copied.stat().st_mode & 0o777, 0o600)


# Docker execution and registry access are replaced. The importer, dataset
# verification, renderer, catalog/schema validation and run_checks remain real;
# only the disposable repository's Makefile test recipe uses a focused suite.
IMPORT_DRIVER = r"""
import io
import json
from pathlib import Path
import sys
import tarfile
from unittest import mock
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import import_task as importer

dataset_context = None
def docker(command, **kwargs):
    global dataset_context
    assert command[0] == 'docker', command
    if command[1:3] == ['image', 'inspect']:
        return command[-1].split(':')[0] + '@sha256:' + 'a' * 64
    if command[1] == 'build' and (Path(command[-1]) / 'datasets').is_dir():
        dataset_context = Path(command[-1])
    if command[1:3] == ['image', 'save']:
        layer = io.BytesIO()
        with tarfile.open(fileobj=layer, mode='w') as archive:
            for path in sorted((dataset_context / 'datasets').rglob('*.json')):
                archive.add(path, arcname=path.relative_to(dataset_context).as_posix())
        with tarfile.open(command[command.index('--output') + 1], 'w') as archive:
            for name, body in [('layer.tar', layer.getvalue()),
                               ('manifest.json', b'[{"Layers":["layer.tar"]}]')]:
                member = tarfile.TarInfo(name)
                member.size = len(body)
                archive.addfile(member, io.BytesIO(body))
    assert command[1] not in ('push', 'login'), command
    return ''

with mock.patch.object(importer.builder, 'run', side_effect=docker), \
        mock.patch.object(importer.resolver, 'resolve_digest', side_effect=AssertionError('unexpected registry access')):
    assert importer.main([sys.argv[1]]) == 0
"""


class ImportIntegrationTests(unittest.TestCase):
    def test_import_runs_real_validate_and_tests_then_repeats_without_diff(self):
        with tempfile.TemporaryDirectory(prefix="frontier-import-gate-") as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            copy_repository(ROOT, root)
            self.assertFalse((root / ".git").exists())
            self.assertFalse((root / ".state").exists())
            self.assertFalse((root / "config/site.json").exists())
            names = {
                task["name"]
                for task in json.loads((root / "config/task-catalog.json").read_text())[
                    "tasks"
                ]
            }
            problems = Path("tasks/frontier-cs-2.0-demo/problems")
            for existing_name in sorted(names):
                with self.subTest(copied_task=existing_name):
                    self.assertEqual(
                        renderer.source_sha256(
                            renderer.problem_directory(root / problems, existing_name)
                        ),
                        renderer.source_sha256(
                            renderer.problem_directory(ROOT / problems, existing_name)
                        ),
                    )
            name = "minimal_task"
            while name in names:
                name += "_new"
            source = Path(directory) / name
            shutil.copytree(ROOT / "tests/fixtures/export-minimal", source)
            for member, key in (
                ("policy.json", "problem_id"),
                ("upstream.lock.json", "problem_id"),
                ("judge/result.json", "task_id"),
            ):
                path = source / member
                body = json.loads(path.read_text())
                body[key] = name
                path.write_text(json.dumps(body))
            # The newer layout gate checks the complete inventory, including
            # formatting changes made while naming this disposable export.
            policy_path = source / "policy.json"
            policy = json.loads(policy_path.read_text())
            policy["quota"]["wall_time_seconds"] = 18000
            instructions = source / "agent-base/frontier/task/README.md"
            instructions.write_text(
                instructions.read_text()
                + "\nFormal Agent answering: 18000 seconds (five hours). Final judging\n"
                f"has its own {policy['quota']['eval_timeout_seconds']}-second budget and\n"
                "does not consume the answering window.\n"
            )
            for mode in ("train", "eval"):
                path = source / f"datasets/{mode}.json"
                body = json.loads(path.read_text())
                body["problem_id"] = name
                path.write_text(json.dumps(body) + "\n")
                policy["metadata"]["datasets"][mode]["version"] = (
                    "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                )
            policy_path.write_text(json.dumps(policy))
            records = [
                {
                    "path": path.relative_to(source).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(source.rglob("*"))
                if path.is_file() and path.name != "upstream.lock.json"
            ]
            lock_path = source / "upstream.lock.json"
            lock = json.loads(lock_path.read_text())
            lock.update(files=records, file_count=len(records))
            lock_path.write_text(json.dumps(lock))
            # The outer make test already discovers the complete repository
            # suite. Both imports still invoke real make validate and make test,
            # with an actual focused test configured only in this copied repo.
            fixture_tests = root / "import-gate-tests"
            fixture_tests.mkdir()
            (fixture_tests / "test_imported_state.py").write_text(
                "import json\nfrom pathlib import Path\nimport unittest\n"
                "class ImportedStateTests(unittest.TestCase):\n"
                "    def test_imported_catalog_and_definition_exist(self):\n"
                "        root = Path.cwd()\n"
                "        catalog = json.loads((root / 'config/task-catalog.json').read_text())\n"
                "        definitions = json.loads((root / 'config/task-definitions.json').read_text())\n"
                f"        name = {name!r}\n"
                "        self.assertEqual(sum(t['name'] == name for t in catalog['tasks']), 1)\n"
                "        self.assertIn(name, definitions)\n"
                "        self.assertTrue((root / 'tasks/frontier-cs-2.0-demo/problems' / name / 'policy.json').is_file())\n"
            )
            makefile = root / "Makefile"
            original = makefile.read_text()
            full_suite = "$(PYTHON) -m unittest discover -s tests -p 'test_*.py'"
            self.assertEqual(original.count(full_suite), 1)
            makefile.write_text(original.replace(
                full_suite,
                "$(PYTHON) -m unittest discover -v -s import-gate-tests -p 'test_*.py'",
            ))
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

            def run_import():
                result = subprocess.run(
                    [sys.executable, "-B", "-c", IMPORT_DRIVER, str(source)],
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(
                    "validate --config config/site.example.json", result.stdout
                )
                self.assertIn("unittest discover -v -s import-gate-tests", result.stdout)
                self.assertIn("test_imported_catalog_and_definition_exist", result.stderr)
                self.assertRegex(result.stderr, r"(?m)^Ran 1 test in ")
                self.assertRegex(result.stderr, r"(?m)^OK$")
                return result

            run_import()
            copied_export = root / problems / name
            self.assertEqual(
                renderer.source_sha256(copied_export), renderer.source_sha256(source)
            )
            catalog = json.loads((root / "config/task-catalog.json").read_text())
            self.assertEqual(
                {task["name"] for task in catalog["tasks"]}, names | {name}
            )

            def inventory():
                result = {}
                for path in root.rglob("*"):
                    if path.is_file():
                        with path.open("rb") as stream:
                            result[path.relative_to(root).as_posix()] = (
                                hashlib.file_digest(stream, "sha256").hexdigest(),
                                path.stat().st_mode & 0o777,
                            )
                return result

            before = inventory()
            self.assertIn("already imported, no changes", run_import().stdout)
            self.assertEqual(inventory(), before)
