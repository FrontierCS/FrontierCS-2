import importlib.util
import json
import math
import tempfile
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lrr_evaluator", ROOT / "evaluator.py")
ev = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(ev)


def test_constructor_uses_stable_anchor_when_judge_is_under_workspace(monkeypatch, tmp_path):
    relocated = tmp_path / "workspace" / "frontier-verify" / "task"
    relocated.mkdir(parents=True)
    (relocated / "constructor_worker.py").write_text("# retained worker\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    commands = []
    def launch(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(pid=12345, returncode=0,
                               communicate=lambda **kw: ('{"rows": [[]]}', ''), wait=lambda: 0)
    monkeypatch.setattr(ev, "TASK_DIR", relocated)
    monkeypatch.setattr(ev.subprocess, "Popen", launch)
    monkeypatch.setattr(ev.os, "killpg", lambda *args: None)
    assert ev._render_chats(repo, "repair", ["input"]) == [[]]
    assert commands[0][2] == str(relocated / "policy_sandbox.py")
    assert commands[0][3] == "/opt/frontier/task"


@pytest.mark.parametrize("startup,accepted", [
    ("load_tensors: offloaded 37/37 layers to GPU\n", True),
    ("load_tensors: offloaded 0/37 layers to GPU\n", False),
    ("load_tensors: offloaded 36/37 layers to GPU\n", False),
    ("model loaded\n", False),
])
def test_endpoint_requires_verbose_full_gpu_offload(monkeypatch, startup, accepted):
    commands = []
    with tempfile.TemporaryFile(mode="w+t") as log:
        log.write(startup)
        log.flush()
        monkeypatch.setattr(ev, "_runtime_state", {"server": None})
        monkeypatch.setattr(ev.tempfile, "TemporaryFile", lambda **kwargs: log)
        def launch(command, **kwargs):
            commands.append(command)
            return SimpleNamespace(poll=lambda: None, terminate=lambda: None)
        monkeypatch.setattr(ev.subprocess, "Popen", launch)
        monkeypatch.setattr(ev.urllib.request, "urlopen", lambda *args, **kwargs: SimpleNamespace(read=lambda: b"{}"))
        monkeypatch.setattr(ev.atexit, "register", lambda callback: None)
        if accepted:
            assert ev._endpoint() == "http://127.0.0.1:18081"
            assert ev._runtime_state["gpu_backend"]["offloaded_layers"] == 37
        else:
            with pytest.raises(RuntimeError, match="complete GPU layer offload"):
                ev._endpoint()
        assert commands[0][commands[0].index("-lv") + 1] == "5"
        assert commands[0][commands[0].index("--device") + 1] == "CUDA0"


@pytest.fixture
def fast_native(monkeypatch):
    exact = {name: 1.0 for name in ev.METRIC_NAMES}
    exact["rli"] = 0.9
    exact["fidelity"] = 1.0
    calls = {"mode": "exact"}

    def run(_repo, workloads):
        values = dict(exact)
        if calls["mode"] == "partial":
            values.update(repair_rate=0.5, well_formed_rate=0.75, nls=0.7, jaccard=0.6, precision=0.6, recall=0.6, rli=1.0, fidelity=0.3)
        return {w["id"]: dict(values) for w in workloads}

    monkeypatch.setattr(ev, "_run_policy", run)
    monkeypatch.setattr(ev, "_reference", lambda workloads: {w["id"]: dict(exact) for w in workloads})
    return calls


def test_reference_scores_and_metrics_complete_repeat(fast_native):
    observed = [ev.evaluate(str(ROOT / "reference.patch")) for _ in range(3)]
    assert len({row[0] for row in observed}) == 1
    for score, unbounded, message, metrics in observed:
        assert math.isfinite(score) and 0 <= score <= 100 and score == unbounded
        assert {"beats_reference", "beat_spec_workload_passes", "reference_score", "margin"} <= metrics.keys()
        assert len(metrics["beat_spec_workload_passes"]) == len(json.loads(ev.DATA.read_text())["development"])
        for row in metrics["beat_spec_workload_passes"]:
            assert row["workload"]
            assert set(row["metrics"]) == set(ev.METRIC_NAMES)
            assert all(type(v) is bool for v in row["metrics"].values())
        assert metrics["beats_reference"] == int(all(v for row in metrics["beat_spec_workload_passes"] for v in row["metrics"].values()))


def test_valid_solver_patch_is_accepted(fast_native):
    score, _, message, metrics = ev.evaluate(str(ROOT / "reference.patch"))
    assert score > 0 and metrics
    assert "evaluated" in message


def test_invalid_submission_is_zero(tmp_path):
    bad = tmp_path / "bad.patch"
    bad.write_text("not a diff\n")
    assert ev.evaluate(str(bad))[:2] == (0.0, 0.0)


def test_metric_orders_controlled_partial_below_exact(fast_native):
    exact = ev.evaluate(str(ROOT / "reference.patch"))[0]
    fast_native["mode"] = "partial"
    partial = ev.evaluate(str(ROOT / "reference.patch"))[0]
    assert 0 <= partial < exact


def _row(n, jaccard):
    return {"compilability": [True] * n, "length_ratio": [1.0] * n,
            "normalized-edit-distance": [0.0] * n, "jaccard": [jaccard] * n,
            "precision": [jaccard] * n, "recall": [jaccard] * n}


SAFE = {"error": None, "result": []}


def test_score_uses_worst_direction_not_mean():
    # A superset rewrite: every original sample kept, 6% of its own samples admitted.
    m = ev._aggregate_scores(_row(2, 0.53), [SAFE, SAFE], [True, True], [0.06, 0.06])
    assert m["repair_rate"] == 1.0
    assert m["fidelity"] == pytest.approx(0.06)


def test_uncredited_cases_earn_no_fidelity():
    vulnerable = {"error": None, "result": [{"attack_success": True}]}
    m = ev._aggregate_scores(_row(2, 1.0), [vulnerable, {"error": "timeout", "result": None}],
                             [True, True], [1.0, 1.0])
    assert m["fidelity"] == 0.0 and m["repair_rate"] == 0.0


def test_directional_evidence_is_required():
    with pytest.raises(ValueError):
        ev._aggregate_scores(_row(1, 1.0), [SAFE], [True])
