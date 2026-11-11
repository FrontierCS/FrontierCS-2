"""lib/install_frontier_pth.py: after it runs with an interpreter, every process of that
interpreter imports the shared modules by name, with no sys.path edits. The GPU worker
shape: a judge re-runs a file with sys.executable, so the script's directory, not
/opt/frontier, is sys.path[0]; `-I` workers drop cwd and PYTHONPATH as well."""

import deployment_test_guard  # noqa: F401  # isort: skip

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

INSTALLER = (
    Path(__file__).resolve().parents[1] / "tasks/shared/lib/install_frontier_pth.py"
)


class InstallFrontierPth(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="frontier-pth-"))
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.root)], check=True)
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(self.root / "venv")],
            check=True,
        )
        self.python = str(self.root / "venv/bin/python")
        self.shared = self.root / "opt-frontier"
        self.shared.mkdir()
        (self.shared / "isolate.py").write_text("LAUNCHER = 'shared'\n")
        worker = self.root / "task" / "evaluator.py"
        worker.parent.mkdir()
        worker.write_text("import isolate\nprint(isolate.LAUNCHER)\n")
        self.worker = str(worker)

    def run_worker(self, *flags):
        return subprocess.run(
            [self.python, *flags, self.worker],
            cwd="/",
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin"},
        )

    def test_a_worker_imports_the_shared_module_only_after_the_install(self):
        before = self.run_worker()
        self.assertNotEqual(before.returncode, 0)
        self.assertIn("ModuleNotFoundError: No module named 'isolate'", before.stderr)
        subprocess.run([self.python, str(INSTALLER), str(self.shared)], check=True)
        for flags in ((), ("-I",)):
            with self.subTest(flags=flags):
                after = self.run_worker(*flags)
                self.assertEqual(
                    (after.returncode, after.stdout, after.stderr), (0, "shared\n", "")
                )

    def test_a_second_install_fails(self):
        subprocess.run([self.python, str(INSTALLER), str(self.shared)], check=True)
        again = subprocess.run(
            [self.python, str(INSTALLER), str(self.shared)],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("FileExistsError", again.stderr)

    def test_a_relative_directory_is_refused(self):
        refused = subprocess.run(
            [self.python, str(INSTALLER), "opt/frontier"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(refused.returncode, 1)
        self.assertIn(
            "usage: install_frontier_pth.py <absolute directory>", refused.stderr
        )


if __name__ == "__main__":
    unittest.main()
