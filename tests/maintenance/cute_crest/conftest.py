"""Task runtime imports are isolated in runtime_imports.py."""
import pytest


@pytest.fixture(autouse=True)
def scratch_outside_the_task_tree(monkeypatch, tmp_path):
    """The evaluator's scratch directory defaults to a directory beside it."""
    monkeypatch.setenv("CREST_TASK_SCRATCH", str(tmp_path / "judge-scratch"))
