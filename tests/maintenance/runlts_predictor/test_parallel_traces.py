"""Trace simulations run side by side; the outcome is the sequential one.

run_ordered is exercised with plain jobs and with real child processes;
evaluations use an invented simulator. No predictor is built or run.
"""
import deployment_test_guard  # noqa: F401  # isort: skip

import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import pytest
import yaml
from tests.maintenance.runlts_predictor.runtime_imports import MODULES, TASK
from tests.maintenance.runlts_predictor.test_pinned_reference import N, output, stage  # noqa: F401

e = MODULES["evaluator"]
adapter = MODULES["fcs2_verifier"]


def test_parallel_exec_applies_exact_limits_without_postfork_python(tmp_path, monkeypatch):
    popen = e.subprocess.Popen
    def checked_popen(*args, **kwargs):
        assert 'preexec_fn' not in kwargs
        assert args[0][:5] == ['/usr/bin/prlimit', '--as=4294967296:4294967296',
                              '--fsize=16777216:16777216', '--core=0:0', '--']
        return popen(*args, **kwargs)
    monkeypatch.setattr(e.subprocess, 'Popen', checked_popen)
    script = ('import json,resource;print(json.dumps([resource.getrlimit(k) for k in '
              '(resource.RLIMIT_AS,resource.RLIMIT_FSIZE,resource.RLIMIT_CORE)]))')
    jobs = [lambda: json.loads(e.run_command([sys.executable, '-c', script], tmp_path,
                                             10, time.monotonic() + 20)) for _ in range(8)]
    assert e.run_ordered(jobs, 8) == [[[4294967296, 4294967296],
                                     [16777216, 16777216], [0, 0]]] * 8


def sleeper(seconds, value=None, error=None):
    def job():
        time.sleep(seconds)
        if error is not None:
            raise error
        return value
    return job


def test_results_and_telemetry_follow_job_order_not_completion_order():
    finished = []
    def job(index):
        def run():
            time.sleep(0.02 * (8 - index))
            e._RUNTIME_TELEMETRY.get()["traces"].append(index)
            finished.append(index)
            return index * index
        return run
    ledger = {"traces": []}
    token = e._RUNTIME_TELEMETRY.set(ledger)
    try:
        assert e.run_ordered([job(i) for i in range(8)], 8) == [i * i for i in range(8)]
    finally:
        e._RUNTIME_TELEMETRY.reset(token)
    assert finished != sorted(finished)  # they really overlapped
    assert ledger["traces"] == list(range(8))
    assert e.run_ordered([], 8) == []


def test_at_most_the_requested_number_of_jobs_run_at_once():
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}
    def job():
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.03)
        with lock:
            state["now"] -= 1
    e.run_ordered([job] * 12, 3)
    assert state["peak"] == 3
    state["peak"] = 0
    e.run_ordered([job] * 4, 1)
    assert state["peak"] == 1


def test_first_failure_in_job_order_wins_whatever_finishes_first():
    early, late = e.CandidateError("job 2"), e.EvaluationError("job 5")
    jobs = [sleeper(0, i) for i in range(8)]
    jobs[2] = sleeper(0.3, error=early)   # fails last
    jobs[5] = sleeper(0, error=late)      # fails first
    for workers in (1, 2, 8):
        with pytest.raises(e.CandidateError) as caught:
            e.run_ordered(jobs, workers)
        assert caught.value is early


def test_telemetry_stops_at_the_first_failure_like_the_sequential_loop():
    def job(index, fail=False):
        def run():
            e._RUNTIME_TELEMETRY.get()["traces"].append(index)
            time.sleep(0.05 if index < 3 else 0)
            if fail:
                raise e.CandidateError("failed")
        return run
    for workers in (1, 8):
        ledger = {"traces": []}
        token = e._RUNTIME_TELEMETRY.set(ledger)
        try:
            with pytest.raises(e.CandidateError):
                e.run_ordered([job(i, fail=i == 3) for i in range(8)], workers)
        finally:
            e._RUNTIME_TELEMETRY.reset(token)
        assert ledger["traces"] == [0, 1, 2, 3]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_failure_kills_later_simulations_and_leaves_nothing_behind(tmp_path):
    tree, pids = tmp_path / "tree", tmp_path / "pids"
    tree.mkdir()
    pids.mkdir()
    deadline = time.monotonic() + 60
    script = ("import os,sys,time;open(os.path.join(sys.argv[1],str(os.getpid())),'w').close();"
              "print('partial output',flush=True);time.sleep(float(sys.argv[2]));sys.exit(int(sys.argv[3]))")
    def process(seconds, code=0):
        return lambda: e.run_command([sys.executable, "-c", script, str(pids), str(seconds), str(code)], tree, 30, deadline)
    started = []
    def never():
        started.append(True)
    # Job 0 outlives the failure and must still finish; job 1 fails; 2 and 3 are
    # running and must be killed; 4 and 5 must never start.
    jobs = [process(1.0), process(0.4, 7), process(30), process(30), never, never]
    began = time.monotonic()
    with pytest.raises(e.CommandFailure) as caught:
        e.run_ordered(jobs, 4)
    assert time.monotonic() - began < 10
    assert caught.value.diagnostic["returncode"] == 7 and caught.value.diagnostic["kind"] == "process_exit"
    assert started == []
    children = [int(p.name) for p in pids.iterdir()]
    assert len(children) == 4 and not any(alive(pid) for pid in children)
    assert list(tree.iterdir()) == []
    assert not [t for t in threading.enumerate() if t.name.startswith("runlts-trace")]


def test_a_cancelled_command_is_killed_and_is_not_a_candidate_failure(tmp_path):
    flag = threading.Event()
    e._JOB.cancel = flag
    try:
        threading.Timer(0.2, flag.set).start()
        began = time.monotonic()
        with pytest.raises(e.Cancelled):
            e.run_command([sys.executable, "-c", "import time;time.sleep(30)"], tmp_path, 30, time.monotonic() + 60)
        assert time.monotonic() - began < 5
    finally:
        e._JOB.cancel = None
    assert not issubclass(e.Cancelled, Exception)
    assert list(tmp_path.iterdir()) == []


def test_watchdog_and_total_deadline_hold_inside_the_pool(tmp_path):
    command = [sys.executable, "-c", "import time;time.sleep(30)"]
    with pytest.raises(e.CommandFailure) as caught:
        e.run_ordered([lambda: e.run_command(command, tmp_path, 0.1, time.monotonic() + 60)] * 3, 3)
    assert caught.value.diagnostic["kind"] == "timeout" and caught.value.diagnostic["returncode"] == -signal.SIGKILL
    with pytest.raises(e.CommandFailure) as caught:
        e.run_ordered([lambda: e.run_command(command, tmp_path, 30, time.monotonic() + 0.1)] * 3, 3)
    assert caught.value.diagnostic["kind"] == "timeout"
    with pytest.raises(e.CandidateError, match="evaluation timeout"):
        e.run_ordered([lambda: e.run_command(command, tmp_path, 30, time.monotonic() - 1)] * 3, 3)


def test_worker_count_is_the_verifier_allocation_lowered_never_raised(monkeypatch):
    policy = json.loads((TASK / "policy.json").read_text())
    config = yaml.safe_load((TASK / "judge/config.yaml").read_text())
    assert e.TRACE_WORKERS == 8 == config["environment"]["cpus"]
    assert policy["verifier"]["cpu_millis"] == 1000 * e.TRACE_WORKERS
    assert policy["verifier"]["memory_bytes"] == config["environment"]["memory_mb"] * 1024 * 1024
    # The agent and the envelope of one simulation are what they were.
    assert policy["agent"] == {"cpu_millis": 1000, "memory_bytes": 4294967296, "workspace_bytes": 42949672960}
    assert config["environment"]["agent"] == {"cpus": 1, "memory_mb": 4096}
    assert e.MEMORY_BYTES == 4096 * 1024 * 1024
    # Even if every simulation reached its address-space limit, they fit.
    assert e.TRACE_WORKERS * e.MEMORY_BYTES <= policy["verifier"]["memory_bytes"]

    monkeypatch.setattr(e, "available_cpus", lambda: 64)
    monkeypatch.delenv(e.TRACE_WORKERS_ENV)
    assert e.trace_workers() == 8
    for value, expected in (("1", 1), ("3", 3), ("8", 8), ("9", 8), ("64", 8)):
        monkeypatch.setenv(e.TRACE_WORKERS_ENV, value)
        assert e.trace_workers() == expected
    monkeypatch.setattr(e, "available_cpus", lambda: 2)
    assert e.trace_workers() == 2
    monkeypatch.delenv(e.TRACE_WORKERS_ENV)
    assert e.trace_workers() == 2
    for value in ("0", "-1", "two", "1.5", " 4"):
        monkeypatch.setenv(e.TRACE_WORKERS_ENV, value)
        with pytest.raises(e.EvaluationError, match="unknown trace worker count"):
            e.trace_workers()


def test_available_cpus_respects_affinity_and_cgroup_quota(monkeypatch, tmp_path):
    real = Path
    quota = tmp_path / "cpu.max"
    monkeypatch.setattr(e, "Path", lambda value: quota if value == "/sys/fs/cgroup/cpu.max" else real(value))
    monkeypatch.setattr(e.os, "sched_getaffinity", lambda pid: set(range(16)))
    assert e.available_cpus() == 16  # no quota file
    for text, expected in (("max 100000", 16), ("800000 100000", 8), ("250000 100000", 2), ("50000 100000", 1),
                           ("3200000 100000", 16), ("garbage", 16)):
        quota.write_text(text + "\n")
        assert e.available_cpus() == expected


def comparable(result):
    score, unbounded, message, metrics = result
    metrics = json.loads(json.dumps(metrics))
    telemetry = metrics.pop("private_runtime_telemetry", None)
    simulation = metrics.pop("trace_simulation", None)
    rows = None if telemetry is None else [(r["role"], r["workload"], r["completed"]) for r in telemetry["traces"]]
    return score, unbounded, message, metrics, rows, simulation and simulation["simulations"]


def fake_simulator(stage, monkeypatch, behaviour=lambda side, index: None):
    """Per-trace counters that differ by workload; later workloads finish first."""
    stage.suite = [{"id": f"unit/w{i}", "path": f"unit/w{i}.gz", "sha256": f"{i:064x}", "full_instr": N, "measured_instr": N}
                   for i in range(24)]
    running = {"now": 0, "peak": 0}
    lock = threading.Lock()
    def command(argv, cwd, timeout, deadline):
        side = "reference" if Path(cwd) == stage.trees["reference"] else "candidate"
        if argv[0] == sys.executable:
            return json.dumps({"valid": True, "logical_bits": MODULES["source_policy"].REFERENCE_BITS, "method": "fixture"})
        if argv[0] == "make":
            return ""
        index = int(Path(argv[1]).stem[1:])
        with lock:
            running["now"] += 1
            running["peak"] = max(running["peak"], running["now"])
        try:
            time.sleep(0.002 * (24 - index))
            special = behaviour(side, index)
            if isinstance(special, BaseException):
                raise special
            if special is not None:
                return special
            return output() if side == "reference" else output(2990 - 7 * index, 139990 - 11 * index)
        finally:
            with lock:
                running["now"] -= 1
    monkeypatch.setattr(e, "run_command", command)
    monkeypatch.setattr(e, "available_cpus", lambda: 64)
    return running


def evaluate_with(monkeypatch, workers):
    monkeypatch.setenv(e.TRACE_WORKERS_ENV, str(workers))
    try:
        return comparable(e.evaluate("candidate.patch"))
    except e.EvaluationError as error:
        evidence = dict(error.private_execution_evidence)
        telemetry = evidence.pop("private_runtime_telemetry")
        evidence.pop("exception_chain", None)
        return "EvaluationError", str(error), evidence, [(r["role"], r["workload"], r["completed"]) for r in telemetry["traces"]]


@pytest.mark.parametrize("audit", [False, True])
def test_one_worker_and_eight_workers_give_the_same_result(stage, monkeypatch, audit):
    running = fake_simulator(stage, monkeypatch)
    if audit:
        monkeypatch.setenv(e.REFERENCE_AUDIT_ENV, "live")
    sequential = evaluate_with(monkeypatch, 1)
    assert running["peak"] == 1
    parallel = evaluate_with(monkeypatch, 8)
    assert running["peak"] == 8
    assert sequential == parallel and sequential[2] == "evaluation complete"
    assert sequential[5] == (48 if audit else 24)
    order = [row["id"] for row in stage.suite]
    assert [row["workload"] for row in sequential[3]["paired_measurements"]] == order
    assert [row["misp_br"] for row in sequential[3]["paired_measurements"]] == [2990 - 7 * i for i in range(24)]
    roles = ["reference", "candidate"] if audit else ["candidate"]
    assert sequential[4] == [(role, wid, True) for wid in order for role in roles]
    monkeypatch.setenv(e.TRACE_WORKERS_ENV, "8")
    metrics = e.evaluate("candidate.patch")[3]
    assert metrics["trace_simulation"] == {"workers": 8, "configured_workers": 8, "simulations": 48 if audit else 24}
    adapter.validated_runtime_telemetry(metrics["private_runtime_telemetry"])


def command_failure():
    return e.CommandFailure("process_exit", elapsed_seconds=1, timeout_seconds=2, returncode=1, output_tail="")


@pytest.mark.parametrize("failures,category", [
    ({5: "garbled", 11: "exit"}, "output_format"),
    ({5: "exit", 11: "garbled"}, "simulator_run_failed"),
    ({9: "short", 3: "exit", 20: "garbled"}, "simulator_run_failed"),
    ({0: "short"}, "instruction_count_mismatch"),
    ({23: "timeout"}, "simulator_run_failed"),
])
def test_candidate_failure_category_is_the_first_failing_workload(stage, monkeypatch, failures, category):
    def behaviour(side, index):
        kind = failures.get(index)
        if kind == "exit":
            return command_failure()
        if kind == "timeout":
            return e.CommandFailure("timeout", elapsed_seconds=1, timeout_seconds=1, returncode=-9, output_tail="")
        if kind == "garbled":
            return "no accounting here"
        if kind == "short":
            return output(3000, 140000).replace(f"{N} 500000", f"{2 * N} 500000").replace("3.0000 140000", "1.5000 140000").replace("140.0000", "70.0000")
        if kind is None and index > min(failures):
            time.sleep(0.05)  # later workloads are still running when the failure lands
        return None
    fake_simulator(stage, monkeypatch, behaviour)
    sequential = evaluate_with(monkeypatch, 1)
    parallel = evaluate_with(monkeypatch, 8)
    assert sequential == parallel
    assert sequential[3] == {"candidate_diagnostic": MODULES["candidate_failure"].diagnostic(category)}


@pytest.mark.parametrize("failing_side,expected", [
    ("reference", "private reference execution failed"),
    ("mismatch", "pinned reference counts differ from live reference execution"),
])
def test_live_audit_reference_failures_stay_infrastructure_errors(stage, monkeypatch, failing_side, expected):
    def behaviour(side, index):
        if side == "reference" and index == 6:
            return command_failure() if failing_side == "reference" else output(3001)
        if side == "candidate" and index in (6, 10):
            return command_failure()  # later candidate failures must not win
        return None
    fake_simulator(stage, monkeypatch, behaviour)
    monkeypatch.setenv(e.REFERENCE_AUDIT_ENV, "live")
    sequential = evaluate_with(monkeypatch, 1)
    parallel = evaluate_with(monkeypatch, 8)
    assert sequential == parallel
    assert sequential[:2] == ("EvaluationError", expected)
    assert sequential[3][-1] == ("reference", "unit/w6", failing_side == "mismatch")
    assert len(sequential[3]) == 13


def test_memory_kill_of_a_simulator_is_not_a_candidate_failure(stage, monkeypatch):
    killed = e.CommandFailure("process_exit", elapsed_seconds=1, timeout_seconds=2, returncode=-signal.SIGKILL, output_tail="")
    fake_simulator(stage, monkeypatch, lambda side, index: killed if index == 4 else None)
    counts = iter([0, 1])
    monkeypatch.setattr(e, "oom_kills", lambda: next(counts))
    assert evaluate_with(monkeypatch, 8)[:2] == ("EvaluationError", "verifier memory exhausted during trace simulation")
    # The same exit without a recorded memory kill is the candidate's.
    monkeypatch.setattr(e, "oom_kills", lambda: 0)
    assert evaluate_with(monkeypatch, 8)[3] == {
        "candidate_diagnostic": MODULES["candidate_failure"].diagnostic("simulator_run_failed")}
    monkeypatch.setattr(e, "oom_kills", lambda: None)
    assert evaluate_with(monkeypatch, 8)[3] == {
        "candidate_diagnostic": MODULES["candidate_failure"].diagnostic("simulator_run_failed")}
