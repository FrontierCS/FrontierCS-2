"""Official gates preserve locked Python source trees, including child imports."""

from __future__ import annotations

import deployment_test_guard  # noqa: F401 # isort: skip

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = """
import importlib.util, os, pathlib, subprocess, sys
assert sys.dont_write_bytecode
assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
path = pathlib.Path("locked_fixture.py").resolve()
spec = importlib.util.spec_from_file_location("locked_fixture", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.VALUE == 7
if "--child" not in sys.argv:
    subprocess.run([sys.executable, "probe.py", "--child"], check=True)
assert not list(path.parent.rglob("*.pyc"))
assert not list(path.parent.rglob("__pycache__"))
"""


class BytecodeGateTests(unittest.TestCase):
    def prepare(self, root):
        (root / "Makefile").write_bytes((ROOT / "Makefile").read_bytes())
        (root / "locked_fixture.py").write_text("VALUE = 7\n")
        (root / "probe.py").write_text(PROBE)
        (root / "scripts").mkdir()
        for name in (
            "check_test_prerequisites.py",
            "validate_task_layout.py",
            "frontier_deploy.py",
        ):
            (root / "scripts" / name).write_text(PROBE)
        (root / "tests").mkdir()
        (root / "tests/test_probe.py").write_text(
            PROBE
            + """
import unittest
class ProbeTest(unittest.TestCase):
    def test_dynamic_import_preserved_tree(self):
        self.assertEqual(module.VALUE, 7)
"""
        )

    def clean_environment(self):
        env = dict(os.environ)
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        env.pop("PYTHONPYCACHEPREFIX", None)
        return env

    def run_probe(self, root, command):
        result = subprocess.run(
            command,
            cwd=root,
            env=self.clean_environment(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(list(root.rglob("*.pyc")))
        self.assertFalse(list(root.rglob("__pycache__")))

    def test_default_make_gates_protect_interpreter_and_children(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.prepare(root)
            self.run_probe(
                root,
                [
                    "make",
                    "test",
                    "validate",
                    "PYTHON=" + sys.executable,
                    "CONFIG=config/site.example.json",
                ],
            )

    def test_importer_enables_current_and_child_bytecode_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.prepare(root)
            # Importer is executed as its CLI script is, without caller -B/env.
            code = f"""
import pathlib, runpy, sys
sys.path.insert(0, {str(ROOT / "scripts")!r})
module = runpy.run_path({str(ROOT / "scripts/import_task.py")!r})
check = module["run_checks"]
check.__globals__["REPOSITORY_ROOT"] = pathlib.Path.cwd()
runpy.run_path("probe.py")
check()
"""
            # run_checks uses Make's normal interpreter selection.
            makefile = root / "Makefile"
            makefile.write_text(
                makefile.read_text().replace(
                    "PYTHON ?= ./python", "PYTHON ?= " + sys.executable
                )
            )
            self.run_probe(root, [sys.executable, "-c", code])

    def test_direct_test_guard_protects_dynamic_fixture_and_children(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.prepare(root)
            code = f"""
import runpy, sys
sys.path.insert(0, {str(ROOT / "tests")!r})
import deployment_test_guard
runpy.run_path("probe.py")
"""
            self.run_probe(root, [sys.executable, "-c", code])
