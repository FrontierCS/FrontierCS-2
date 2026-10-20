#!/usr/bin/env python3
"""Runs the search harness on an instrumented program, one process per seed.

Used by the judge and available to the solver for local experiments.  Every
launch of the program under test is confined: 2 GB of address space, a
10-second timeout, no core dumps, and a write jail around the run directory.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import signal
import subprocess
import tempfile

# ---- Limits of one run.  Defined once; the judge and the statement use these.
BUDGET_ITERATIONS = 4000
# One launch of the program under test.
EXEC_TIMEOUT_SECONDS = 10
# 2 GB, not more: eight runs share the judge's 32 GiB, and a runaway program
# (one generated sed script was measured growing past 3.7 GB resident) must
# not be able to exhaust it even if all eight coincide.
EXEC_ADDRESS_SPACE_KB = 2000000
# Safety stops of one run.  A run that reaches one ends early and keeps the
# coverage it has; none of them is a validity failure.
#
# The binding one is the run's wall-clock budget, set per program at about
# 1.2 to 1.4 times the slowest reference run measured on it, eight runs at a
# time (judge/programs.json; RUN_WALL_SECONDS is the development program's).
# The compute budget (strategy + solving CPU) equals the wall budget, so it
# never binds later; the solver-attempt and request caps are about ten times
# the reference's means and stop a request loop in seconds.
RUN_WALL_SECONDS = 320           # development program (grep 2.2)
MAX_SOLVER_ATTEMPTS = 160000     # feasible + infeasible solver queries
MAX_NEGATE_REQUESTS = 225000     # Negate calls, including those rejected without a query
STRATEGY_ADDRESS_SPACE_KB = 1048576
HARNESS_ADDRESS_SPACE_KB = 4194304
# The harness ends a run within about 30 s of its wall budget (one launch or
# one bounded solver query in flight); this hard stop is the backstop.
HARD_TIMEOUT_SLACK_SECONDS = 60

EARLY_END_STATUSES = frozenset(['run_time_budget_exhausted', 'compute_budget_exhausted',
                                'solver_attempt_budget_exhausted', 'negate_request_budget_exhausted',
                                'strategy_returned'])
COMPLETE_STATUSES = frozenset(['budget_exhausted']) | EARLY_END_STATUSES


class ProgramSpec:
    """An instrumented program: its directory holds the binary, `branches`
    and `cfg_branches`."""

    def __init__(self, directory, argv, needs_pty=False, term='dumb'):
        self.directory = Path(directory)
        self.argv = list(argv)          # argv[0] is the binary name inside directory
        self.needs_pty = needs_pty
        self.term = term


def _limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cap = HARNESS_ADDRESS_SPACE_KB * 1024
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    os.setsid()


def run_one(repo, spec, strategy_host, seed, scratch, *, budget=BUDGET_ITERATIONS,
            relabel_key=None, run_wall_seconds=RUN_WALL_SECONDS, compute_cpu_seconds=None,
            max_solver_attempts=MAX_SOLVER_ATTEMPTS, max_negate_requests=MAX_NEGATE_REQUESTS,
            hard_timeout=None, toolchain_env=None):
    """One run.  Returns the harness result dict; `status` tells how it ended.

    A run whose harness did not produce a result has status `harness_failure`.
    """
    repo = Path(repo).resolve()
    if compute_cpu_seconds is None:
        compute_cpu_seconds = run_wall_seconds
    if hard_timeout is None:
        hard_timeout = run_wall_seconds + HARD_TIMEOUT_SLACK_SECONDS
    run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=str(scratch))).resolve()
    try:
        for name in ('branches', 'cfg_branches'):
            shutil.copyfile(spec.directory / name, run_dir / name)
        (run_dir / 'in.txt').write_text('hello world 12\nfoo bar baz\nline three here\n')
        command = [str(spec.directory / spec.argv[0])] + spec.argv[1:]
        jail = repo / 'lib/jail.so'
        (run_dir / 'cmd.sh').write_text(
            '#!/bin/bash\nulimit -v %d\nulimit -c 0\n'
            'timeout --foreground -k 1 %d env LD_PRELOAD=%s %s\n'
            'rc=$?\n[ $rc -ge 124 ] && echo $rc >> killed\nexit 0\n'
            % (EXEC_ADDRESS_SPACE_KB, EXEC_TIMEOUT_SECONDS, shlex.quote(str(jail)),
               ' '.join(shlex.quote(a) for a in command)))
        (run_dir / 'cmd.sh').chmod(0o755)
        launch = 'script -qec ./cmd.sh /dev/null' if spec.needs_pty else './cmd.sh'
        env = dict(toolchain_env or {})
        env.setdefault('PATH', '/usr/bin:/bin')
        env.update({'HOME': str(run_dir), 'CREST_JAIL': str(run_dir), 'TERM': spec.term,
                    'LC_ALL': 'C', 'TZ': 'UTC',
                    'CREST_EXEC_WALL_SECONDS': str(EXEC_TIMEOUT_SECONDS + 10),
                    'CREST_EXEC_AS_KB': str(EXEC_ADDRESS_SPACE_KB),
                    'CREST_COMPUTE_CPU_SECONDS': str(compute_cpu_seconds),
                    'CREST_MAX_SOLVER_ATTEMPTS': str(max_solver_attempts),
                    'CREST_MAX_NEGATE_REQUESTS': str(max_negate_requests),
                    'CREST_RUN_WALL_SECONDS': str(run_wall_seconds),
                    'CREST_STRATEGY_AS_KB': str(STRATEGY_ADDRESS_SPACE_KB)})
        if relabel_key is not None:
            env['CREST_RELABEL_KEY'] = str(int(relabel_key))
        result_file = run_dir / 'result.json'
        process = subprocess.Popen(
            [str(repo / 'bin/search_harness'), launch, str(int(budget)), str(int(seed)),
             str(strategy_host), str(result_file)],
            cwd=str(run_dir), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, preexec_fn=_limits)
        try:
            process.wait(timeout=hard_timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait()
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            result = json.loads(result_file.read_text())
        except (OSError, ValueError):
            result = {'status': 'harness_failure', 'covered': 0, 'iterations': 0, 'curve': []}
        if timed_out or result.get('status') == 'running':
            result['status'] = 'harness_failure'
        try:
            result['killed_executions'] = len((run_dir / 'killed').read_text().split())
        except OSError:
            result['killed_executions'] = 0
        result['seed'] = int(seed)
        result['children_peak_kb'] = usage.ru_maxrss
        return result
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def run_many(jobs, workers=8):
    """jobs: list of (key, kwargs for run_one).  Returns {key: result}."""
    workers = max(1, min(int(workers), 8))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [(key, pool.submit(run_one, **kwargs)) for key, kwargs in jobs]
        return {key: future.result() for key, future in futures}


def search_overhead_seconds(result):
    """The efficiency measurement of a run: CPU seconds of strategy code plus
    constraint solving.  Executing the program under test is not included."""
    return float(result.get('strategy_cpu_seconds', 0.0)) + float(result.get('solver_cpu_seconds', 0.0))
