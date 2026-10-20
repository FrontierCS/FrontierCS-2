"""Large native archives must remain LFS-managed after category moves."""

import deployment_test_guard  # noqa: F401  # isort: skip

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_task_definition as renderer


class TaskMoveLFSTests(unittest.TestCase):
    def test_clusterwise_archives_retain_lfs_filter(self):
        task = renderer.problem_directory(
            ROOT / "tasks/frontier-cs-2.0-demo/problems", "clusterwise_spgemm"
        )
        paths = [
            (task / "datasets" / (mode + ".tar")).relative_to(ROOT).as_posix()
            for mode in ("train", "eval")
        ]
        # Import integration copies source without its .git directory.
        with tempfile.TemporaryDirectory(prefix="frontier-lfs-attributes-") as tmp:
            root = Path(tmp)
            (root / ".gitattributes").write_bytes(
                (ROOT / ".gitattributes").read_bytes()
            )
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            output = subprocess.check_output(
                ["git", "check-attr", "filter", "--", *paths], cwd=root, text=True
            )
        self.assertEqual(
            output.splitlines(), [path + ": filter: lfs" for path in paths]
        )


if __name__ == "__main__":
    unittest.main()
