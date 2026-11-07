#!/usr/bin/env python3
"""Judge of cute_crest (CREST search strategy) for the shared verifier: evaluate(tree).

Score: per graded program, 100 * max(0, C - B) / (T - B) with C the candidate's mean coverage
over the 30 seeds, B the program's pinned bounded-DFS (depth 12) mean coverage and T its branch
count; raw_score = score_unbounded = the mean over the graded programs (development: grep 2.2;
final: two held-out programs). A source/object rule break is a rejection; a build failure or a
strategy fault in any run (crash, sandbox kill, API misuse, protocol, exit) scores 0. Early
ends (caps, Run returned) are valid. A reference or harness failure raises.

Beat-spec metrics (beat_spec_workload_passes, one row per graded program; gate.py, exact
integer arithmetic, K = 30 seeds, s^2 = (var_c + var_r) / 2 with sample variances):
- branches_covered, higher is better: passes when mean(cov_c) - mean(cov_r) > 3 s sqrt(2/K).
- search_overhead_seconds (strategy + solver CPU, whole ms), lower is better: passes when
  mean(ovh_r) - mean(ovh_c) > 3 s sqrt(2/K).
beats_reference = 1 iff both metrics pass on every graded program.

Candidate and reference run interleaved on the same seeds and secret relabelling; check_tree
runs the starter's solution_policy source rules. The strategy is compiled, and its objects are
listed for the object rules, in a shared isolate instance (no judge files, no network, a
clean environment); the strategy process seals itself (seccomp) before any solution code runs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

TASK = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(TASK))
import gate  # noqa: E402
import isolate  # noqa: E402  the shared launcher, beside edit_scope in the Verifier base

# Built once per verifier image by build_benchmarks.py; read-only afterwards.
# Kept outside the task bundle so a verification does not copy it.
BUILT = Path(os.environ.get('CREST_TASK_BUILT_ROOT', '/opt/crest-task'))
# The starter tree (repo/), copied here by the Verifier image; the reference patch applies to it.
PRISTINE = TASK / 'pristine'
SOLUTION = 'src/frontier_solution'
METRIC_NAMES = ['branches_covered', 'search_overhead_seconds']
PROGRAMS = json.loads((TASK / 'programs.json').read_text())
WORKERS = 8


# Solution-rule codes (solution_policy.PolicyViolation) -> the shared rejection categories.
# A layout fault (a file kind, a subdirectory, a missing strategy.cc or MakeStrategy()) is a
# change outside the editable surface; everything else is content the task forbids.
REJECTION = {'solution_layout': 'out_of_scope'}


def rejection_category(code):
    return REJECTION.get(code, 'content_policy')


class Rejected(Exception):
    """A solution rule the compiled objects break: a scored zero with a shared category."""

    def __init__(self, category):
        Exception.__init__(self, category)
        self.category = category


class InvalidCandidate(Exception):
    """The submission cannot be scored.  `code` is safe to show the solver."""

    def __init__(self, code):
        Exception.__init__(self, code)
        self.code = code


class EvaluatorFailure(RuntimeError):
    """The judge itself could not measure.  Never a candidate verdict."""


def suite(final):
    return PROGRAMS['final'] if final else PROGRAMS['development']


def workload_name(index):
    return 'w%d' % index


def empty_metrics(final):
    return {'beats_reference': 0, 'reference_score': 0.0, 'margin': 0.0,
            'beat_spec_workload_passes': [
                {'workload': workload_name(i), 'metrics': {name: False for name in METRIC_NAMES}}
                for i in range(len(suite(final)))]}


def relabel_key(program, seed):
    """Per-run key of the secret id permutation: the same for candidate and
    reference, unknown to the solver, stable across evaluations."""
    text = '%s|%s|%d' % (PROGRAMS['relabel_salt'], program['id'], seed)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') >> 1


def seeds_of(program):
    return [int(program['seed_base']) + i for i in range(gate.K_SEEDS)]


def program_score(program, mean_coverage):
    """Share, in percent, of the branches the bounded-DFS baseline leaves
    uncovered that the strategy covers at budget.  Uses the program's branch
    count and the pinned baseline mean only, so the score of a candidate does
    not depend on any reference measurement."""
    baseline = float(program['baseline_mean_coverage'])
    total = float(program['branch_count'])
    return 100.0 * max(0.0, mean_coverage - baseline) / (total - baseline)


def score_of(programs, coverage_lists):
    """Continuous score: mean over the graded programs of program_score()."""
    parts = [program_score(p, sum(c) / float(len(c))) for p, c in zip(programs, coverage_lists)]
    return sum(parts) / len(parts)


def overhead_ms(result):
    return int(round(1000.0 * (float(result['strategy_cpu_seconds']) + float(result['solver_cpu_seconds']))))


def load_tools():
    tree = BUILT / 'tree'
    if not (tree / 'bin/search_harness').is_file():
        raise EvaluatorFailure('built harness is missing')
    sys.path.insert(0, str(tree))
    try:
        import crest_runner
        import frontier_build
    finally:
        sys.path.remove(str(tree))
    return tree, crest_runner, frontier_build


# The strategy build of the starter's frontier_build.build_strategy (same compiler flags and
# order), run in one isolate instance: the solution's .cc files are compiled, each object
# is listed (nm -P, readelf -S -W) for the object rules, then strategy_host.cc is compiled and
# everything is linked statically.  Exit 10: a solution file does not compile; 11: an object
# cannot be listed; 20: the host does not compile or the link fails.
BUILD_SCRIPT = r"""set -u
src=$1; shift
cxx="g++ -std=c++17 -O2 -Wall -fno-plt -fno-pic -no-pie"
for source in "$@"; do
  timeout -k 5 300 $cxx -I"$src" -I. -c "$source" -o "$OUT/${source%.cc}.solution.o" >>"$OUT/build.log" 2>&1 || exit 10
done
for source in "$@"; do
  object="$OUT/${source%.cc}.solution.o"
  nm -P "$object" >"$object.nm" 2>>"$OUT/build.log" || exit 11
  readelf -S -W "$object" >"$object.sections" 2>>"$OUT/build.log" || exit 11
done
timeout -k 5 300 $cxx -I"$src" -c "$src/strategy/strategy_host.cc" -o "$TMPDIR/strategy_host.o" >>"$OUT/build.log" 2>&1 || exit 20
objects=""
for source in "$@"; do objects="$objects $OUT/${source%.cc}.solution.o"; done
timeout -k 5 300 $cxx -static "$TMPDIR/strategy_host.o" $objects -o "$OUT/strategy_host" >>"$OUT/build.log" 2>&1 || exit 20
"""


def build_candidate(policy, solution_dir, out_dir):
    """Checks and compiles a solution directory into out_dir/strategy_host."""
    try:
        sources = [p.name for p in policy.check_sources(solution_dir)]
    except policy.PolicyViolation as error:
        raise Rejected(rejection_category(error.code))
    names = []
    for source in sources:
        stem = source[:-len('.cc')]
        names += [stem + '.solution.o.nm', stem + '.solution.o.sections']
    result = isolate.run(
        ['/bin/sh', '{in}/build.sh', str(BUILT / 'tree/src'), *sources],
        tree=solution_dir, inputs={'build.sh': BUILD_SCRIPT.encode()},
        outputs=['strategy_host', *names], read_paths=[str(BUILT / 'tree/src')],
        env={'OUT': '{out}', 'LC_ALL': 'C', 'TZ': 'UTC'}, output_bytes=256 << 20,
        timeout_seconds=300 * (len(sources) + 2) + 60)
    if result.limit or result.returncode == 10:
        raise InvalidCandidate('build_failed')
    if result.returncode == 11:
        raise Rejected(rejection_category('object_audit'))
    entry = False
    try:
        for index in range(0, len(names), 2):
            nm, sections = (result.outputs[names[index]], result.outputs[names[index + 1]])
            if nm is None or sections is None:
                raise policy.PolicyViolation('object_audit', 'cannot inspect compiled object')
            sections = re.findall(r'\]\s+(\.\S+)', sections.decode('utf-8', 'replace'))
            entry = policy.check_object_listing(nm.decode('utf-8', 'replace'), sections) or entry
        if not entry:
            raise policy.PolicyViolation('solution_layout', 'frontier::MakeStrategy() is not defined')
    except policy.PolicyViolation as error:
        raise Rejected(rejection_category(error.code))
    binary = result.outputs.get('strategy_host')
    if result.returncode or not binary:
        raise InvalidCandidate('build_failed')
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'strategy_host'
    path.write_bytes(binary)
    path.chmod(0o755)
    return path


def load_policy():
    """The judge's own copy of the solution rules (the starter's solution_policy.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('crest_judge_solution_policy',
                                                  PRISTINE / 'solution_policy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_tree(tree, pristine, changes):
    """Shared-verifier hook: the source rules of src/frontier_solution/ (which files, and what
    their text may contain). Returns (shared category, paths) to reject, or None."""
    policy = load_policy()
    try:
        policy.check_sources(Path(tree) / SOLUTION)
    except policy.PolicyViolation as error:
        return rejection_category(error.code), sorted({c.path for c in changes})
    return None


def reference_tree(scratch):
    """The pristine starter with judge/reference.patch applied (trusted, judge-owned)."""
    tree = Path(scratch) / 'reference'
    shutil.copytree(PRISTINE, tree, symlinks=True)
    for path in (tree, *tree.rglob('*')):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(Path(scratch).resolve()))
    subprocess.run(['git', 'apply', '--whitespace=nowarn', str(TASK / 'reference.patch')],
                   cwd=tree, check=True, capture_output=True, timeout=60, env=env)
    return tree


def measure(tools, programs, hosts, scratch, env):
    """Runs every (program, role, seed).  Returns {(program index, role): [result per seed]}."""
    tree, crest_runner, _ = tools
    jobs = []
    for index, program in enumerate(programs):
        spec = crest_runner.ProgramSpec(BUILT / 'bench' / program['directory'], program['argv'],
                                        needs_pty=bool(program['needs_pty']), term=program['term'])
        for n, seed in enumerate(seeds_of(program)):
            # Alternate which side starts, so neither is systematically first.
            for role in (('candidate', 'reference') if n % 2 == 0 else ('reference', 'candidate')):
                jobs.append(((index, role, seed), dict(
                    repo=tree, spec=spec, strategy_host=hosts[role], seed=seed, scratch=scratch,
                    relabel_key=relabel_key(program, seed), run_wall_seconds=int(program['run_wall_seconds']),
                    toolchain_env=env)))
    results = crest_runner.run_many(jobs, WORKERS)
    return {(index, role): [results[(index, role, seed)] for seed in seeds_of(program)]
            for index, program in enumerate(programs) for role in ('candidate', 'reference')}


def grade(tools, programs, measured, final):
    _, crest_runner, _ = tools
    rows = []
    for index, program in enumerate(programs):
        candidate, reference = measured[(index, 'candidate')], measured[(index, 'reference')]
        if any(r['status'] != 'budget_exhausted' for r in reference):
            raise EvaluatorFailure('a reference run did not complete its budget')
        # A candidate run may end early (budget caps, or Run() returned): it
        # keeps the coverage it reached.  Anything else is a fault of the
        # strategy process and voids the submission.
        faults = sorted({r['status'] for r in candidate if r['status'] not in crest_runner.COMPLETE_STATUSES})
        if any(s == 'harness_failure' or s.startswith('harness_error') for s in faults):
            raise EvaluatorFailure('a candidate run was lost: ' + ','.join(faults))
        if faults:
            raise InvalidCandidate(faults[0].replace(':', '_'))
        rows.append({'workload': workload_name(index), 'candidate_valid': True,
                     'candidate_coverage': [int(r['covered']) for r in candidate],
                     'reference_coverage': [int(r['covered']) for r in reference],
                     'candidate_overhead_ms': [overhead_ms(r) for r in candidate],
                     'reference_overhead_ms': [overhead_ms(r) for r in reference]})
    beats, decisions = gate.decide(rows)
    score = score_of(programs, [row['candidate_coverage'] for row in rows])
    reference_score = score_of(programs, [row['reference_coverage'] for row in rows])
    all_candidate = [r for index in range(len(programs)) for r in measured[(index, 'candidate')]]
    private = []
    for index, row in enumerate(rows):
        k = float(gate.K_SEEDS)
        private.append({
            'workload': row['workload'],
            'candidate_mean_coverage': sum(row['candidate_coverage']) / k,
            'reference_mean_coverage': sum(row['reference_coverage']) / k,
            'coverage_margin_required': gate.margin(row['candidate_coverage'], row['reference_coverage']),
            'overhead_margin_required_seconds': gate.margin(row['candidate_overhead_ms'], row['reference_overhead_ms']) / 1000.0,
            'candidate_mean_overhead_seconds': sum(row['candidate_overhead_ms']) / k / 1000.0,
            'reference_mean_overhead_seconds': sum(row['reference_overhead_ms']) / k / 1000.0,
            'candidate_coverage': row['candidate_coverage'], 'reference_coverage': row['reference_coverage'],
            'candidate_overhead_ms': row['candidate_overhead_ms'],
            'reference_overhead_ms': row['reference_overhead_ms'],
            'candidate_statuses': [r['status'] for r in measured[(index, 'candidate')]],
            'candidate_solver_attempts': [int(r['solver_attempts']) for r in measured[(index, 'candidate')]],
            'reference_solver_attempts': [int(r['solver_attempts']) for r in measured[(index, 'reference')]],
            'candidate_wall_seconds': [r['wall_seconds'] for r in measured[(index, 'candidate')]],
            'reference_wall_seconds': [r['wall_seconds'] for r in measured[(index, 'reference')]],
        })
    metrics = {'beats_reference': beats, 'beat_spec_workload_passes': decisions,
               'reference_score': reference_score, 'margin': score - reference_score,
               'per_workload_comparisons': private}
    if not final:
        # Development feedback: the candidate's own measurements only.
        n = float(len(all_candidate))
        checkpoints = sorted({point[0] for r in all_candidate for point in r.get('curve', [])})
        curve = []
        for checkpoint in checkpoints:
            values = [point[1] for r in all_candidate for point in r.get('curve', []) if point[0] == checkpoint]
            # Runs that ended before a checkpoint keep their final coverage.
            values += [int(r['covered']) for r in all_candidate
                       if all(point[0] != checkpoint for point in r.get('curve', []))]
            curve.append([checkpoint, round(sum(values) / n, 2)])
        metrics.update({
            'dev_branches_covered_mean': round(sum(int(r['covered']) for r in all_candidate) / n, 3),
            'dev_coverage_curve': curve,
            'dev_search_overhead_seconds_mean': round(sum(overhead_ms(r) for r in all_candidate) / n / 1000.0, 3),
            'dev_solver_attempts_mean': round(sum(int(r['solver_attempts']) for r in all_candidate) / n, 1),
            'dev_runs_ended_early': sum(1 for r in all_candidate if r['status'] != 'budget_exhausted'),
            'dev_killed_executions_mean': round(sum(int(r.get('killed_executions', 0)) for r in all_candidate) / n, 2),
        })
    return score, metrics


def evaluate(tree: str) -> tuple[float, float, str, dict]:
    final = os.environ.get('FRONTIER_SUBMISSION_ROLE') == 'final'
    cleanup = []
    previous_tempdir = tempfile.tempdir
    try:
        candidate = Path(tree).resolve()
        scratch = Path(os.environ.get('CREST_TASK_SCRATCH') or TASK / '.judge-cache')
        scratch.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(scratch)
        work = Path(tempfile.mkdtemp(prefix='measure-', dir=scratch))
        cleanup.append(work)
        try:
            tools = load_tools()
            env, _ = tools[2].environment(tools[0], BUILT / 'build', tmp=work / 'tmp')
            policy = load_policy()
            hosts = {'candidate': build_candidate(policy, candidate / SOLUTION, work / 'candidate')}
        except Rejected as error:
            return 0.0, 0.0, 'rejected: ' + error.category, {'rejected': error.category}
        except InvalidCandidate as error:
            return 0.0, 0.0, 'invalid candidate: ' + error.code, empty_metrics(final)
        try:
            reference = reference_tree(work)
        except subprocess.CalledProcessError:
            raise EvaluatorFailure('the private reference patch does not apply')
        try:
            hosts['reference'] = build_candidate(policy, reference / SOLUTION, work / 'reference')
        except (InvalidCandidate, Rejected) as error:
            raise EvaluatorFailure('the private reference does not build: ' + str(error))
        programs = suite(final)
        runs = work / 'runs'
        runs.mkdir()
        measured = measure(tools, programs, hosts, runs, env)
        try:
            score, metrics = grade(tools, programs, measured, final)
        except InvalidCandidate as error:
            return 0.0, 0.0, 'invalid candidate: ' + error.code, empty_metrics(final)
        return score, score, 'evaluation complete', metrics
    finally:
        tempfile.tempdir = previous_tempdir
        for path in reversed(cleanup):
            if path:
                shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    result = evaluate(sys.argv[1] if len(sys.argv) > 1 else '')
    print(json.dumps({'score': result[0], 'score_unbounded': result[1], 'message': result[2],
                      'metrics': result[3]}, allow_nan=False))
