"""The search runs on the trusted search machine: what goes up, what the harness measures, who is
at fault.

The shared adapter's run_sandboxes() runs for real on the harness of tests/test_shared_gpu_adapter.py
(a fake Driver; a fake SSH gateway that runs each remote command on this host with /workspace
mapped to a directory per runner), so the upload, step and collect commands of a command sandbox
run unchanged, on one CPU machine of the declared shape (runtime_imports.SEARCH_SANDBOX).

The machine's interpreter is a virtual environment with the shared lib on its import path
(frontier.pth, as the Verifier image installs it). The built tree is a stand-in: the real
crest_runner.py and frontier_build.py with a fake search harness that starts the side's
strategy_host as the real one does (fork, its pipes on fds 0 and 1, an empty environment, wait4)
and reports what the "strategy" printed, how it ended and its CPU time. So every strategy here
runs through judge/strategy_bridge.py in a real isolate instance (isolate.exec_on_stdio).
"""

import deployment_test_guard  # noqa: F401  # isort: skip

import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest
from test_shared_gpu_adapter import AdapterCase

from tests.maintenance.cute_crest.runtime_imports import (
    MODULES,
    REPO_TREE,
    ROOT,
    SEARCH_SANDBOX,
)

e = MODULES["evaluator"]
DEV = e.PROGRAMS["development"][0]
# The system interpreter: an instance reads the system directories, not a virtual environment.
SYSTEM_PYTHON = os.path.realpath(sys.executable)
pytestmark = pytest.mark.skipif(
    not e.isolate.available(), reason="needs an isolate backend"
)

# Stands in for bin/search_harness: argv = launch budget seed strategy_host result_file. The
# strategy prints one JSON line (its report); the harness adds what it measured.
FAKE_HARNESS = textwrap.dedent(
    """\
    #!%s
    import json, os, signal, sys
    launch, budget, seed, strategy, result = sys.argv[1:6]
    down_r, down_w = os.pipe()
    up_r, up_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.dup2(down_r, 0)
        os.dup2(up_w, 1)
        os.closerange(3, 1024)
        os.execve(strategy, ["strategy_host"], {})
    os.close(down_r)
    os.close(up_w)
    os.close(down_w)
    line = os.fdopen(up_r, "rb").readline()
    _, status, usage = os.wait4(pid, 0)
    code = os.waitstatus_to_exitcode(status)
    spec = json.loads(line) if line else {}
    status = ("harness_error:sandbox_unavailable" if code == 71 else
              "strategy_fault:sandbox_kill" if code == -signal.SIGSYS else
              "strategy_fault:exited" if code else "budget_exhausted")
    covered = spec.get("covered", 0)
    run = {"status": status, "covered": covered, "iterations": int(budget),
           "solver_attempts": 16000, "strategy_cpu_seconds": usage.ru_utime + usage.ru_stime,
           "solver_cpu_seconds": 70.0, "wall_seconds": 100.0, "curve": [[4000, covered]],
           "probe": spec.get("probe")}
    run.update(spec.get("override", {}))
    for name in spec.get("drop", []):
        run.pop(name)
    open(result, "w").write(json.dumps(run))
    """
    % SYSTEM_PYTHON
)

# Inside the strategy's instance: whether it can read the other side's strategy, and its
# environment.
PROBE = """
def readable(path):
    try:
        open(path, "rb").read(1)
        return True
    except OSError:
        return False
sides = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
probe = {"other_side": readable(os.path.join(sides, "reference", "strategy")),
         "environ": sorted(os.environ)}
"""


def strategy(report, before="", probe=False):
    """A stand-in strategy: Python code `before`, then one JSON line `report`."""
    line = "dict(%r, probe=probe)" % (report,) if probe else repr(report)
    return "#!%s -I\nimport json, os, sys\n%s\nprint(json.dumps(%s), flush=True)\n" % (
        SYSTEM_PYTHON,
        before + (PROBE if probe else ""),
        line,
    )


class SearchSandbox(AdapterCase):
    def setUp(self):
        super().setUp()
        self.parameters.write_text(
            json.dumps(
                {
                    "entrypoint": "payload/solution.patch",
                    "sandboxes": {"search": SEARCH_SANDBOX},
                }
            )
        )
        venv = self.scratch / "venv"
        subprocess.run(
            [SYSTEM_PYTHON, "-m", "venv", "--without-pip", str(venv)], check=True
        )
        subprocess.run(
            [
                str(venv / "bin/python"),
                str(ROOT / "tasks/shared/lib/install_frontier_pth.py"),
                str(ROOT / "tasks/shared/lib"),
            ],
            check=True,
        )
        built = self.built = self.scratch / "crest-task"
        (built / "tree/bin").mkdir(parents=True)
        (built / "tree/lib").mkdir()
        for name in ("crest_runner.py", "frontier_build.py", "solution_policy.py"):
            shutil.copy2(REPO_TREE / name, built / "tree" / name)
        (built / "tree/bin/search_harness").write_text(FAKE_HARNESS)
        (built / "tree/bin/search_harness").chmod(0o755)
        (built / "tree/lib/jail.so").write_bytes(b"")
        (built / "build/toolchain").mkdir(parents=True)
        (built / "build/toolchain/ready").touch()
        for program in e.PROGRAMS["development"] + e.PROGRAMS["final"]:
            directory = built / "bench" / program["directory"]
            directory.mkdir(parents=True)
            for name in ("branches", "cfg_branches", program["argv"][0]):
                (directory / name).write_text(name)
        self.hosts = {}
        patches = {
            "BUILT": built,
            "PYTHON": str(venv / "bin/python"),
            "SandboxRun": self.gpu.SandboxRun,
            "SandboxStep": self.gpu.SandboxStep,
            "run_sandboxes": self.gpu.run_sandboxes,
            "sandbox_workdir": self.gpu.sandbox_workdir,
            "build_candidate": lambda policy, solution, out: self.host(out),
        }
        saved = {name: getattr(e, name) for name in patches}
        for name, value in patches.items():
            setattr(e, name, value)
        self.addCleanup(
            lambda: [setattr(e, name, value) for name, value in saved.items()]
        )
        os.environ["CREST_TASK_SCRATCH"] = str(self.scratch / "judge")

    def host(self, out):
        out.mkdir(parents=True, exist_ok=True)
        path = out / "strategy_host"
        path.write_text(self.hosts[out.name])
        path.chmod(0o755)
        return path

    def evaluate(self, candidate, reference=None, role="development"):
        shutil.rmtree(self.scratch / "runners", ignore_errors=True)
        self.hosts = {
            "candidate": candidate,
            "reference": reference or strategy({"covered": 1586}),
        }
        os.environ["FRONTIER_SUBMISSION_ROLE"] = role
        os.environ["FRONTIER_VERIFICATION_MODE"] = (
            "eval" if role == "final" else "train"
        )
        return e.evaluate(str(e.PRISTINE))

    def upload(self):
        (upload,) = (self.scratch / "runners").glob("*/workspace/search")
        return upload

    def runs(self):
        """(job, result) of every run on the machine."""
        upload = self.upload()
        return list(
            zip(
                json.loads((upload / "jobs.json").read_text())["runs"],
                json.loads((upload / "out/results.json").read_text()),
                strict=False,
            )
        )

    def test_both_sides_run_interleaved_on_one_trusted_machine_and_score_as_before(
        self,
    ):
        score, unbounded, message, metrics = self.evaluate(strategy({"covered": 1700}))
        assert message == "evaluation complete"
        assert score == pytest.approx(e.program_score(DEV, 1700))
        assert metrics["reference_score"] == pytest.approx(e.program_score(DEV, 1586))
        assert [
            (c["image"], c["cpu_millis"], c["memory_bytes"])
            for c in self.driver.creates
        ] == [("verifier_cpu", 8000, 34359738368)]
        assert len(self.driver.deleted) == 1
        upload = self.upload()
        assert {p.name for p in upload.iterdir()} == {
            "run_search.py",
            "strategy_bridge.py",
            "sides",
            "jobs.json",
            "out",
            "scratch",
            "launcher-failures",
        }
        assert sorted(p.name for p in (upload / "sides").iterdir()) == [
            "candidate",
            "reference",
        ]
        runs = self.runs()
        assert [(job["role"], run["covered"]) for job, run in runs[:4]] == [
            ("candidate", 1700),
            ("reference", 1586),
            ("reference", 1586),
            ("candidate", 1700),
        ]

    def test_the_strategy_runs_in_an_instance_and_its_cpu_reaches_the_harness(self):
        burn = "import time\nend = time.process_time() + 0.5\nwhile time.process_time() < end: pass\n"
        self.evaluate(strategy({"covered": 1700}, burn, probe=True))
        candidate = [run for job, run in self.runs() if job["role"] == "candidate"]
        assert len(candidate) == e.gate.K_SEEDS
        for run in candidate:
            assert (
                run["probe"]["other_side"] is False
            )  # the reference's strategy is unseen
            assert not any(
                name.startswith(("FRONTIER_", "CREST_"))
                for name in run["probe"]["environ"]
            )
            assert (
                run["strategy_cpu_seconds"] >= 0.5
            )  # wait4 counts the instance's command

    def test_a_strategy_killed_by_its_seal_is_a_scored_zero(self):
        # SIGSYS: what a forbidden system call ends a sealed strategy with.
        kill = strategy(
            {"covered": 1700}, "import signal\nos.kill(os.getpid(), signal.SIGSYS)"
        )
        score, _, message, _ = self.evaluate(kill)
        assert (score, message) == (
            0.0,
            "invalid candidate: strategy_fault_sandbox_kill",
        )

    def test_a_strategy_that_exits_is_a_scored_zero(self):
        score, _, message, _ = self.evaluate(strategy({"covered": 1700}, "os._exit(3)"))
        assert (score, message) == (0.0, "invalid candidate: strategy_fault_exited")

    def test_a_strategy_that_exits_71_is_a_scored_zero(self):
        # 71 is strategy_host.cc's "sandbox unavailable", which the harness reports as
        # harness_error:sandbox_unavailable; the strategy's own exit 71 is still its fault.
        score, _, message, _ = self.evaluate(
            strategy({"covered": 1700}, "os._exit(71)")
        )
        assert (score, message) == (
            0.0,
            "invalid candidate: harness_error_sandbox_unavailable",
        )
        assert not any((self.upload() / "launcher-failures").iterdir())

    def test_a_launcher_failure_is_never_a_score(self):
        # The isolate launcher cannot build the strategy's instance: it writes its record out
        # of band and the search step fails as the judge's own fault.
        bridge = e.TASK / "strategy_bridge.py"
        original = bridge.read_text()
        bridge.chmod(0o644)
        self.addCleanup(bridge.write_text, original)
        bridge.write_text(
            original.replace("read_paths=[side]", "read_paths=['/nonexistent/side']")
        )
        with pytest.raises(e.EvaluatorFailure, match="isolate launcher failed"):
            self.evaluate(strategy({"covered": 1700}))

    def test_a_result_missing_a_field_fails_loudly(self):
        for name in (
            "strategy_cpu_seconds",
            "solver_cpu_seconds",
            "wall_seconds",
            "solver_attempts",
            "curve",
        ):
            with pytest.raises(e.EvaluatorFailure, match=name):
                self.evaluate(strategy({"covered": 1700, "drop": [name]}))

    def test_a_malformed_result_fails_loudly(self):
        for override in (
            {"covered": 10**6},
            {"covered": 17.5},
            {"status": 5},
            {"strategy_cpu_seconds": "x"},
            {"curve": [[1, 2, 3]]},
        ):
            with pytest.raises(e.EvaluatorFailure):
                self.evaluate(strategy({"covered": 1700, "override": override}))

    def test_a_failing_search_step_fails_loudly(self):
        (self.built / "tree/frontier_build.py").write_text("raise SystemExit(5)\n")
        with pytest.raises(e.EvaluatorFailure, match="search machine failed"):
            self.evaluate(strategy({"covered": 1700}))


def test_the_search_machine_is_one_trusted_verifier_image_machine():
    assert (SEARCH_SANDBOX["image"], SEARCH_SANDBOX["max_concurrent"]) == (
        "verifier_cpu",
        1,
    )
    assert SEARCH_SANDBOX["resources"] == {
        "cpu_millis": 8000,
        "memory_bytes": 34359738368,
    }
