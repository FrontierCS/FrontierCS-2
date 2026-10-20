"""Package import, compatibility and executable CLI regression checks."""

from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import ast
import contextlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import frontier_deploy as deploy
from preview import cli, config, publication

MODULES = ("config", "infra", "images", "publication", "acceptance", "cli")
IMPORT_PROBE = r"""
import importlib
import io
import importlib
import os
from pathlib import Path
import sys
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'scripts'))

def repository_path(path):
    return isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).absolute().is_relative_to(root)

def audit(event, args):
    # Python must read module code to import it. No repository data may be read.
    if event == 'open' and repository_path(args[0]):
        path = Path(os.fsdecode(args[0]))
        if path.suffix not in {'.py', '.pyc'}:
            raise AssertionError('repository data read during import: ' + str(path))

sys.addaudithook(audit)
original_open = Path.open

def guarded_open(path, *args, **kwargs):
    # Also reject explicit pathlib reads of Python source, not just JSON/data.
    if repository_path(path):
        raise AssertionError('explicit repository read during import: ' + str(path))
    return original_open(path, *args, **kwargs)

Path.open = guarded_open
for name in sys.argv[2:]:
    importlib.import_module(name)
"""


class PackageImportsTests(unittest.TestCase):
    def test_each_module_and_shim_import_from_empty_directory_without_data_reads(self):
        names = [*("preview." + name for name in MODULES), "frontier_deploy"]
        with tempfile.TemporaryDirectory(prefix="preview-import-") as directory:
            for first in names:
                with self.subTest(first=first):
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            "-c",
                            IMPORT_PROBE,
                            str(ROOT),
                            first,
                            *(name for name in reversed(names) if name != first),
                        ],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_other_script_imports_work_from_empty_directory(self):
        scripts = (
            "render_task_definition",
            "import_task",
            "reference_parity",
            "reference_candidates",
            "task_selection",
            "build_task_images",
            "resolve_catalog_digests",
            "task_baselines",
            "check_test_prerequisites",
        )
        with tempfile.TemporaryDirectory(prefix="preview-script-import-") as directory:
            for name in scripts:
                with self.subTest(script=name):
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            "-c",
                            "import importlib, sys; sys.path.insert(0, sys.argv[1]); "
                            "importlib.import_module(sys.argv[2]); import frontier_deploy",
                            str(ROOT / "scripts"),
                            name,
                        ],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_entire_package_import_graph_is_acyclic(self):
        edges = {}
        for name in MODULES:
            tree = ast.parse((ROOT / "scripts/preview" / (name + ".py")).read_text())
            edges[name] = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.level == 1
                and node.module in MODULES
            }

        def visit(name, active):
            self.assertNotIn(
                name, active, f"circular package dependency: {active} -> {name}"
            )
            for child in edges[name]:
                visit(child, (*active, name))

        for name in MODULES:
            visit(name, ())

    def test_shim_reexports_every_original_function_and_constant(self):
        exports = json.loads(
            (ROOT / "tests/fixtures/evidence/phase3/shim-exports.json").read_text()
        )
        for module, names in exports.items():
            owner = __import__("preview." + module, fromlist=["unused"])
            for name in names:
                with self.subTest(module=module, name=name):
                    self.assertIs(getattr(deploy, name), getattr(owner, name))
        self.assertEqual(deploy.TASK_DEFINITIONS, publication.load_task_definitions())


def shim_patch_targets(source, namespace):
    """Resolve names/attribute aliases to the actual module, including shim copies."""
    bindings = dict(namespace)
    missing = object()

    def resolve(node):
        if isinstance(node, ast.Name):
            return bindings.get(node.id, missing)
        if isinstance(node, ast.Attribute):
            return getattr(resolve(node.value), node.attr, missing)
        return missing

    def dotted(target):
        parts = target.split(".")
        value = bindings.get(parts[0], missing)
        if value is not missing:
            for part in parts[1:]:
                value = getattr(value, part, missing)
            return value
        for size in range(len(parts), 0, -1):
            try:
                value = importlib.import_module(".".join(parts[:size]))
            except (ImportError, ValueError):
                continue
            for part in parts[size:]:
                value = getattr(value, part, missing)
            return value
        return missing

    tree = ast.parse(source)
    # Resolve assignment aliases without executing any test body.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = resolve(node.value)
            if value is not missing:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bindings[target.id] = value
    failures = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        owner = missing
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "object"
            and len(node.args) >= 2
        ):
            owner = resolve(node.args[0])
        elif (isinstance(node.func, ast.Attribute) and node.func.attr == "patch") or (
            isinstance(node.func, ast.Name) and node.func.id == "patch"
        ):
            target = node.args[0]
            if (
                isinstance(target, ast.Constant)
                and isinstance(target.value, str)
                and "." in target.value
            ):
                owner = dotted(target.value.rsplit(".", 1)[0])
        if isinstance(owner, types.ModuleType) and Path(
            getattr(owner, "__file__", "") or ""
        ).resolve() in {
            ROOT / "scripts" / name
            for name in (
                "frontier_deploy.py",
                "reference_parity.py",
                "reference_candidates.py",
                "task_baselines.py",
            )
        }:
            failures.append(node.lineno)
    return failures


class ShimPatchGuardTests(unittest.TestCase):
    def test_tests_never_patch_the_compatibility_shim(self):
        failures = []
        for path in (ROOT / "tests").rglob("test_*.py"):
            module = importlib.import_module(
                ".".join(path.relative_to(ROOT).with_suffix("").parts)
            )
            failures.extend(
                f"{path.relative_to(ROOT)}:{line}"
                for line in shim_patch_targets(path.read_text(), vars(module))
            )
        self.assertEqual(
            failures,
            [],
            "Patch the module that looks up the dependency: " + ", ".join(failures),
        )

    def test_scanner_resolves_attribute_assignment_and_importlib_aliases(self):
        parity = types.SimpleNamespace(deploy=deploy)
        spec = importlib.util.spec_from_file_location(
            "copied_shim", ROOT / "scripts/frontier_deploy.py"
        )
        copied = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(copied)
        namespace = {
            "parity": parity,
            "deploy": deploy,
            "d": copied,
            "publication": publication,
        }
        for expression in (
            "mock.patch.object(parity.deploy, 'api_json')",
            "mock.patch('reference_parity.save_receipt')",
            "mock.patch('reference_candidates.load_manifest')",
            "mock.patch('task_baselines.baseline_from_result_json')",
            "mock.patch('parity.deploy.api_json')",
            "alias = parity.deploy\nmock.patch.object(alias, 'api_json')",
            "mock.patch.object(d, 'run')",
            "mock.patch('frontier_deploy.api_json')",
        ):
            with self.subTest(expression=expression):
                self.assertTrue(shim_patch_targets(expression, namespace))
        self.assertEqual(
            shim_patch_targets("mock.patch.object(publication, 'api_json')", namespace),
            [],
        )


class CliSubprocessTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", "scripts/frontier_deploy.py", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_validate_wrapper_matches_dispatch_output_and_order(self):
        expected = io.StringIO()
        with contextlib.redirect_stdout(expected):
            cli.validate_command(ROOT / "config/site.example.json")
        result = self.run_cli("validate", "--config", "config/site.example.json")
        self.assertEqual(
            (result.returncode, result.stdout, result.stderr),
            (0, expected.getvalue(), ""),
        )
        self.assertTrue(result.stdout.startswith("selected tasks: "))
        self.assertLess(
            result.stdout.index("effective catalog:"),
            result.stdout.index("config: PASS"),
        )

    def test_render_wrapper_writes_the_expected_bundle(self):
        result = self.run_cli("render", "--config", "config/site.example.json")
        site, _ = config.read_and_validate(ROOT / "config/site.example.json")
        output = config.generated_directory(site)
        expected = f"rendered: {output}\nprofile: {site['management']['profile']} (fixed task catalog; scripted CPU smoke; optional Modal GPU)\n"
        self.assertEqual(
            (result.returncode, result.stdout, result.stderr), (0, expected, "")
        )
        import hashlib

        for line in (output / "bundle.sha256").read_text().splitlines():
            digest, name = line.split("  ")
            self.assertEqual(
                hashlib.sha256((output / name).read_bytes()).hexdigest(), digest
            )
        self.assertEqual(
            json.loads((output / "release.json").read_text())["config_sha256"],
            config.read_and_validate(ROOT / "config/site.example.json")[1],
        )

    def test_all_help_wrappers_match_existing_golden_bytes(self):
        for path in sorted((ROOT / "tests/fixtures/golden/cli").glob("*.txt")):
            with self.subTest(command=path.stem):
                result = self.run_cli(
                    *([] if path.stem == "root" else [path.stem]), "--help"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.encode(), path.read_bytes())
                self.assertEqual(result.stderr, "")

    def test_parity_wrappers_match_captured_cli_bytes(self):
        before = json.loads((ROOT / "tests/fixtures/evidence/phase5/cli-before.json").read_text())
        for name in ("reference_parity", "reference_candidates", "task_baselines"):
            with self.subTest(script=name):
                expected = before[name]
                result = subprocess.run(
                    expected["command"], cwd=ROOT, capture_output=True, timeout=30
                )
                self.assertEqual(
                    (result.returncode, result.stdout, result.stderr),
                    (
                        expected["returncode"],
                        expected["stdout"].encode(),
                        expected["stderr"].encode(),
                    ),
                )

    def test_invalid_config_keeps_exit_two_and_stderr(self):
        with tempfile.TemporaryDirectory(prefix="preview-invalid-config-") as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("{}")
            result = self.run_cli("validate", "--config", str(path))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr.startswith("error: config "), result.stderr)


if __name__ == "__main__":
    unittest.main()
