import importlib.util
import json
import math
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
# The shared launcher: /opt/frontier/isolate.py in the image, tasks/shared/lib in the repo.
SHARED_LIB = ROOT.parents[4] / "shared" / "lib" if len(ROOT.parents) > 4 else None
if SHARED_LIB is not None and SHARED_LIB.is_dir():
    sys.path.insert(0, str(SHARED_LIB))
spec = importlib.util.spec_from_file_location("lrr_evaluator", ROOT / "evaluator.py")
ev = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(ev)


def test_constructor_runs_under_isolate(monkeypatch, tmp_path):
    relocated = tmp_path / "workspace" / "frontier-verify" / "task"
    relocated.mkdir(parents=True)
    (relocated / "constructor_worker.py").write_text("# retained worker\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(ok=True, limit=None, stdout=b'{"rows": [[]]}', stderr=b"")
    monkeypatch.setattr(ev, "TASK_DIR", relocated)
    monkeypatch.setattr(ev.isolate, "run", run)
    assert ev._render_chats(repo, "repair", ["input"]) == [[]]
    argv, kwargs = calls[0]
    assert argv[1:] == ["-I", "{in}/worker.py", "{in}/request.json"]
    assert kwargs["tree"] == str(repo) and kwargs["timeout_seconds"] == 600
    assert kwargs["inputs"]["worker.py"] == str(relocated / "constructor_worker.py")
    assert json.loads(kwargs["inputs"]["request.json"]) == {"task": "repair", "lines": ["input"]}
    assert not kwargs.get("devices") and not kwargs.get("read_paths")


@pytest.mark.parametrize("result,reason", [
    (SimpleNamespace(ok=False, limit="time", stdout=b"", stderr=b""), "exceeded its time limit"),
    (SimpleNamespace(ok=False, limit=None, stdout=b"", stderr=b"Traceback\nValueError: boom\n"), "ValueError: boom"),
    (SimpleNamespace(ok=False, limit="memory", stdout=b"", stderr=b""), "limit memory"),
    (SimpleNamespace(ok=True, limit=None, stdout=b"not json", stderr=b""), "malformed output"),
    (SimpleNamespace(ok=True, limit=None, stdout=b'{"rows": []}', stderr=b""), "wrong number"),
])
def test_constructor_failures_are_policy_errors(monkeypatch, tmp_path, result, reason):
    monkeypatch.setattr(ev.isolate, "run", lambda argv, **kwargs: result)
    with pytest.raises(ev.PolicyError, match=reason):
        ev._render_chats(tmp_path, "repair", ["input"])


def test_isolate_faults_are_evaluator_errors(monkeypatch, tmp_path):
    def run(argv, **kwargs):
        raise ev.isolate.IsolateError("no backend")
    monkeypatch.setattr(ev.isolate, "run", run)
    with pytest.raises(ev.isolate.IsolateError):
        ev._render_chats(tmp_path, "repair", ["input"])


def test_policy_input_keeps_only_pattern_and_first_localization():
    raw = {"pattern": "^(a+)+$", "error": None, "result": {
        "regex": "LEAK", "final_regex": "LEAK-REPAIRED",
        "repairs": [{"subpatterns": ["(a+)+"], "repaired": "LEAK", "nested_quantifier_result": "LEAK"},
                    {"subpatterns": ["LEAK-LATER"], "repaired": "LEAK"}]}}
    text = json.dumps(ev._policy_input(raw))
    assert "LEAK" not in text
    assert json.loads(text)["result"]["repairs"][0]["subpatterns"] == ["(a+)+"]


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
    observed = [ev.evaluate(str(ev.PRISTINE)) for _ in range(3)]
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
    score, _, message, metrics = ev.evaluate(str(ev.PRISTINE))
    assert score > 0 and metrics
    assert "evaluated" in message


def test_metric_orders_controlled_partial_below_exact(fast_native):
    exact = ev.evaluate(str(ev.PRISTINE))[0]
    fast_native["mode"] = "partial"
    partial = ev.evaluate(str(ev.PRISTINE))[0]
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
