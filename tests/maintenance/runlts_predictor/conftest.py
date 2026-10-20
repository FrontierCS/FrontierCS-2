"""Task runtime imports are isolated in runtime_imports.py."""
import pytest


@pytest.fixture(autouse=True)
def one_trace_worker(monkeypatch):
    """Tests that assert the order of simulator calls run one simulation at a time.

    test_parallel_traces.py sets the worker count itself."""
    monkeypatch.setenv("RUNLTS_TRACE_WORKERS", "1")
