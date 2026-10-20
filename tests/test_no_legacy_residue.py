"""Reject legacy authority paths and adapter identifiers in current sources."""

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERMS = json.loads((ROOT / "tests/fixtures/legacy-residue-terms.json").read_text())


def legacy_residue(root: Path) -> list[str]:
    violations = []
    scopes = (
        "Makefile",
        "README.md",
        "AGENTS.md",
        "scripts",
        "config",
        "tests",
        "docs",
        ".github",
        "skills",
        ".agents/skills",
        ".codex/skills",
    )
    for directory in scopes:
        scope = root / directory
        paths = [scope] if scope.is_file() else sorted(scope.rglob("*"))
        for path in paths:
            relative = path.relative_to(root)
            if (
                path.is_symlink()
                or not path.is_file()
                or any(
                    relative.is_relative_to(excluded)
                    for excluded in ("tests/fixtures", "docs/history", "docs/evidence")
                )
                or "__pycache__" in relative.parts
                or path.suffix in {".pyc", ".pyo"}
            ):
                continue
            for number, line in enumerate(path.read_bytes().splitlines(), 1):
                for term in TERMS:
                    if term.encode() in line:
                        violations.append(f"{relative}:{number}: {term}")
    return violations


class NoLegacyResidueTests(unittest.TestCase):
    def test_file_symlinks_are_not_scanned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            history = root / "docs/history"
            history.mkdir(parents=True)
            content = TERMS[0] + "\n"
            (scripts / "current.py").write_text(content)
            for target in (history / "retired.md", Path(directory) / "outside.md"):
                with self.subTest(target=target.name):
                    target.write_text(content)
                    (scripts / target.name).symlink_to(target)
                    self.assertEqual(
                        legacy_residue(root), [f"scripts/current.py:1: {TERMS[0]}"]
                    )

    def test_root_files_and_workflow_skill_callers_are_scanned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = (
                "Makefile",
                "README.md",
                "AGENTS.md",
                ".github/workflows/test.yml",
                "skills/build/SKILL.md",
                ".agents/skills/build/SKILL.md",
                ".codex/skills/build/SKILL.md",
            )
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(TERMS[-1] + "\n")
            self.assertEqual(
                set(legacy_residue(root)), {f"{name}:1: {TERMS[-1]}" for name in paths}
            )

    def test_no_legacy_residue(self):
        violations = legacy_residue(ROOT)
        self.assertFalse(violations, "Legacy residue:\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
